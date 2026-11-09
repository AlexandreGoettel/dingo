#
#  Adapted from bilby_pipe.
#

import copy
import os
import sys
from types import SimpleNamespace
from urllib.parse import quote

import pycondor

from bilby_pipe.job_creation.dag import Dag
from bilby_pipe.utils import BilbyPipeError, logger

from dingo.pipe.utils import _strip_unwanted_submission_keys
from dingo.pipe.nodes.generation_node import GenerationNode
from .nodes.importance_sampling_node import ImportanceSamplingNode
from .nodes.merge_node import MergeNode
from .nodes.pe_summary_node import PESummaryNode
from .nodes.plot_node import PlotNode, PlotPPNode
from .nodes.sampling_node import SamplingNode
from .sampling_retry import PRE_SKIP


logger.name = "dingo_pipe"


class PipelineDag(Dag):
    def write_bash_script(self):
        if any(isinstance(node, pycondor.Dagman) for node in self.pycondor_dag.nodes):
            logger.info("Adaptive retry sub-DAGs require HTCondor; no bash runner generated.")
        else:
            super().write_bash_script()


def get_trigger_time_list(inputs):
    """Returns a list of GPS trigger times for each data segment"""
    if (inputs.gaussian_noise or inputs.zero_noise) and inputs.trigger_time is None:
        trigger_times = [inputs.model_reference_time] * inputs.n_simulation
    elif (inputs.gaussian_noise or inputs.zero_noise) and isinstance(
        inputs.trigger_time, float
    ):
        trigger_times = [inputs.trigger_time] * inputs.n_simulation
    elif inputs.trigger_time is not None:
        trigger_times = [inputs.trigger_time]
    elif getattr(inputs, "gpstimes", None) is not None:
        start_times = inputs.gpstimes
        trigger_times = start_times + inputs.duration - inputs.post_trigger_duration
    else:
        raise BilbyPipeError("Unable to determine input trigger times from ini file")
    logger.info(f"Setting segment trigger-times {trigger_times}")
    return trigger_times


def get_parallel_list(inputs):
    if inputs.n_parallel == 1:
        return [""]
    else:
        return [f"part{idx}" for idx in range(inputs.n_parallel)]


def generate_sampling_rounds(
    inputs, sampling_node, generation_node, proposal_generation_node, dag
):
    """Build one event's importance-sampling rounds and return its final result node."""
    adaptive = inputs.min_rss > 0
    max_retries = inputs.max_retries if adaptive else 0
    event = sampling_node.job_name
    decision_file = os.path.join(inputs.result_directory, event + "_retry.json")

    final_result = previous_result = parent = None
    for retry_index in range(max_retries + 1):
        round_inputs = copy.copy(inputs)
        round_dag = dag
        config_file = os.path.join(inputs.result_directory, f"{event}_retry{retry_index}.ini")
        if retry_index:
            subdag = pycondor.Dagman(
                name=f"{event}_retry{retry_index}_dag", submit=inputs.submit_directory,
                dag=dag.pycondor_dag,
            )
            subdag.add_parent(parent)
            # DAGMan executes PRE directly; encode the path as one argument.
            decision_arg = quote(os.path.abspath(decision_file), safe="/")
            command = f"{sys.executable} -m dingo.pipe.sampling_retry {decision_arg}"
            dag.pycondor_dag.extra_lines.extend([
                f"SCRIPT PRE {subdag.name} {command}",
                f"PRE_SKIP {subdag.name} {PRE_SKIP}",
            ])
            round_dag = SimpleNamespace(pycondor_dag=subdag)
            if inputs.sampling_seed is not None:
                round_inputs.sampling_seed += retry_index * inputs.n_parallel
            sampling_node = SamplingNode(
                round_inputs, proposal_generation_node, round_dag,
                config_file=config_file, retry_index=retry_index,
            )

        parts = [ImportanceSamplingNode(
            round_inputs, sampling_node, generation_node, part, round_dag
        ) for part in get_parallel_list(inputs)]
        final_result = final_result or parts[0].final_result_file
        options = {}
        if adaptive:
            options = dict(
                previous_result=previous_result, decision_file=decision_file,
                final_result=final_result,
                next_config=os.path.join(inputs.result_directory,
                                         f"{event}_retry{retry_index + 1}.ini"),
            )
        merged = (MergeNode(round_inputs, parts, round_dag, **options)
                  if adaptive or len(parts) > 1 else parts[0])
        previous_result = merged.result_file
        parent = subdag if retry_index else merged.job
    if not adaptive:
        return merged
    # Consumers wait for the whole chain and use the result published on stopping.
    return SimpleNamespace(job=parent, result_file=final_result,
                           label=event.replace("sampling", "importance_sampling"),
                           job_name=event.replace("sampling", "importance_sampling"))


def generate_dag(inputs):
    inputs = copy.deepcopy(inputs)
    if (inputs.importance_sample and inputs.min_rss > 0 and inputs.max_retries > 0
            and (inputs.run_local or inputs.scheduler.lower() != "condor")):
        raise NotImplementedError("Adaptive retry jobs require HTCondor (not local or Slurm).")
    dag = PipelineDag(inputs)
    trigger_times = get_trigger_time_list(inputs)

    if inputs.simple_submission:
        _strip_unwanted_submission_keys(dag.pycondor_dag)
    #
    # 1. Generate data for inference.
    #

    # Iterate over all generation nodes and store them in a list
    generation_node_list = []
    for idx, trigger_time in enumerate(trigger_times):
        kwargs = dict(trigger_time=trigger_time, idx=idx, dag=dag)
        if idx > 0:
            # Make all generation nodes depend on the 0th generation node
            # Ensures any cached files (e.g. the distance-marginalization
            # lookup table) are only built once.
            kwargs["parent"] = generation_node_list[0]
        generation_node = GenerationNode(inputs, **kwargs)
        generation_node_list.append(generation_node)

    #
    # 2. Sample the posterior using Dingo.
    #
    # Reconstruct the posterior density if necessary. This requires training a new
    # network and sampling from it.

    sampling_node_list = []
    for generation_node in generation_node_list:
        sampling_node = SamplingNode(
            inputs,
            generation_node=generation_node,
            dag=dag,
        )
        sampling_node_list.append(sampling_node)

    if inputs.importance_sample:
        #
        # 3. Generate new data for importance sampling **if different settings requested**.
        #
        # If injecting into simulated noise, be sure to use consistent noise
        # realization. Currently simulated noise + importance_sampling_updates is
        # prohibited in MainInput.

        if len(inputs.importance_sampling_updates) > 0:
            # Iterate over all generation nodes and store them in a list
            importance_sampling_generation_node_list = []
            for idx, trigger_time in enumerate(trigger_times):
                kwargs = dict(trigger_time=trigger_time, idx=idx, dag=dag)
                if idx > 0:
                    # Make all generation nodes depend on the 0th generation node
                    # Ensures any cached files (e.g. the distance-marginalization
                    # lookup table) are only built once.
                    kwargs["parent"] = generation_node_list[0]
                generation_node = GenerationNode(
                    inputs, importance_sampling=True, **kwargs
                )
                importance_sampling_generation_node_list.append(generation_node)
        else:
            importance_sampling_generation_node_list = generation_node_list

        #
        # 4. Importance sample
        #
        # If the phase is not present and phase marginalization is not being used, also
        # sample the phase synthetically. This adds between 1x and 50x to the cost of
        # importance sampling, depending on the waveform model. Indeed, IMRPhenomXPHM
        # waveform modes are much more expensive to generate than polarizations.

        merged_importance_sampling_node_list = [
            generate_sampling_rounds(inputs, sampling, generation, proposal, dag)
            for sampling, generation, proposal in zip(
                sampling_node_list, importance_sampling_generation_node_list,
                generation_node_list,
            )
        ]

    else:
        merged_importance_sampling_node_list = sampling_node_list

    #
    # 5. Plotting
    #

    for merged_node in merged_importance_sampling_node_list:
        if inputs.plot_node_needed:
            PlotNode(inputs, merged_node, dag=dag)

    if inputs.plot_pp:
        PlotPPNode(inputs, merged_importance_sampling_node_list, dag=dag)

    #
    # 6. PESummary
    #

    if inputs.create_summary:
        PESummaryNode(
            inputs, merged_importance_sampling_node_list, generation_node_list, dag=dag
        )

    dag.build()
