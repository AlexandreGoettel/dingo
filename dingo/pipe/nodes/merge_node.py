import os

from bilby_pipe.job_creation.nodes import MergeNode as BilbyMergeNode

from dingo.pipe.utils import (
    configure_file_transfer,
    _strip_unwanted_submission_keys,
)


class MergeNode(BilbyMergeNode):

    run_node_on_osg = False
    
    def __init__(self, inputs, parallel_node_list, dag, *, previous_result=None,
                 decision_file=None, final_result=None, next_config=None):
        # Adapted from Bilby Pipe 1.10.1 to guard the container append before
        # transfer directives are built. Keep Node's automatic image transfer.
        super(BilbyMergeNode, self).__init__(inputs)
        self.adaptive = decision_file is not None
        self.dag = dag
        self.job_name = f"{parallel_node_list[0].base_job_name}_merge"
        self.label = self.job_name
        self.request_cpus = 1

        if inputs.transfer_files or inputs.osg:
            input_files_to_transfer = [
                self._relative_topdir(pn.result_file, inputs.initialdir)
                for pn in parallel_node_list
            ] + inputs.additional_transfer_paths
            if self.adaptive:
                input_files_to_transfer.append(str(inputs.complete_ini_file))
                if previous_result:
                    input_files_to_transfer.append(previous_result)
            configure_file_transfer(self, input_files_to_transfer)

        self.setup_arguments(
            add_ini=False, add_unknown_args=False, add_command_line_args=False
        )
        self.arguments.append("--result")
        for pn in parallel_node_list:
            self.arguments.append(os.path.relpath(pn.result_file))
        self.arguments.add("outdir", os.path.relpath(inputs.result_directory))
        self.arguments.add("label", self.label)
        self.arguments.add("extension", inputs.result_format)
        self.arguments.add_flag("merge")

        if self.adaptive:
            options = dict(
                previous_result=previous_result, decision_file=decision_file,
                final_result=final_result, next_config=next_config,
                config_file=inputs.complete_ini_file, min_rss=inputs.min_rss,
                max_retries=inputs.max_retries, max_n_samples=inputs.max_n_samples,
                n_parallel=inputs.n_parallel,
                shortfall_safety_margin=inputs.shortfall_safety_margin,
            )
            for key, value in options.items():
                if value is not None:
                    self.arguments.add(key.replace("_", "-"), value)
        self.process_node()
        for pn in parallel_node_list:
            self.job.add_parent(pn.job)
        if self.inputs.simple_submission:
            _strip_unwanted_submission_keys(self.job)

    @property
    def executable(self):
        return self._get_executable_path("dingo_result")

    @property
    def result_file(self):
        label = self.label if self.adaptive else self.label.replace("_merge", "")
        return f"{self.inputs.result_directory}/{label}.hdf5"
