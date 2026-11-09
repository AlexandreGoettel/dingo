"""Merge-round retry decisions and the lightweight sub-DAG PRE check."""

import copy
import json
import math
from pathlib import Path

STATE_KEY = "sampling_retry"
PRE_SKIP = 77


def resolve_result_path(filename):
    from bilby_pipe.utils import resolve_filename_with_transfer_fallback
    return resolve_filename_with_transfer_fallback(str(filename)) or str(filename)


def pre_check(filename):
    """Missing/corrupt decisions are errors, never successful skips."""
    state = json.loads(Path(filename).read_text())
    return 0 if state["next_samples"] else PRE_SKIP


def write_atomic(filename, text):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    temporary = filename.with_suffix(filename.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(filename)


def retry_decision(
    num_samples, rss, retries, min_rss, max_retries, max_n_samples=None, n_parallel=1,
    shortfall_safety_margin=2.0,
):
    """Estimate additional proposals from the cumulative realized RSS efficiency.

    Requests have a minimum of 10000 proposals and at least one per worker.
    max_n_samples caps the request after applying these minimums.
    """
    if rss >= min_rss:
        return 0, "target reached"
    if retries >= max_retries:
        return 0, f"max-retries={max_retries} reached"
    if rss <= 0 or num_samples <= 0:
        return 0, "zero RSS efficiency; cannot estimate a finite sampling request"
    target = shortfall_safety_margin * (min_rss - rss)
    requested = max(10_000, n_parallel, math.ceil(target * num_samples / rss))
    if max_n_samples is not None and requested > max_n_samples:
        return 0, (
            f"estimated request of {requested} proposals exceeds "
            f"max-n-samples={max_n_samples}"
        )
    return requested, "additional sampling required"


def merge_sampling_rounds(previous, current):
    """Combine proposals with their own log densities, retaining round provenance.

    Proposal provenance can change (seed, GNPE recovery, chirp-mass scan), while
    the model, event and target metadata must still pass Result.merge's checks.
    """
    from dingo.gw.result import Result

    previous_state = previous.importance_sampling_metadata.pop(STATE_KEY)
    retries = int(previous_state["retries"]) + 1
    rounds = previous.metadata.pop("sampling_rounds")
    rounds[str(retries)] = copy.deepcopy(current.metadata.get("sampler", {}))
    if "sampler" in previous.metadata:
        current.metadata["sampler"] = copy.deepcopy(previous.metadata["sampler"])
    else:
        current.metadata.pop("sampler", None)
    result = Result.merge([previous, current])
    result.metadata["sampling_rounds"] = rounds
    return result, retries


def check_rss(
    result, output_file, *, previous_file=None, min_rss=5000,
    max_retries=1, max_n_samples=None, n_parallel=1,
    shortfall_safety_margin=2.0,
):
    """Accumulate a round, rejection sample, and persist the next-round decision."""
    import numpy as np
    from bilby_pipe.utils import logger
    from dingo.gw.result import RANDOM_STATE, Result

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_file.with_suffix(".rss.tmp")
    if previous_file is None:
        retries = 0
        result.metadata["sampling_rounds"] = {
            "0": copy.deepcopy(result.metadata.get("sampler", {}))
        }
    else:
        previous = Result(file_name=resolve_result_path(previous_file))
        result, retries = merge_sampling_rounds(previous, result)

    # Match get_pesummary_samples without doing its expensive parameter conversion.
    weights = result.samples["weights"].to_numpy()
    valid_weights = (
        len(weights) > 0
        and np.all(np.isfinite(weights))
        and np.all(weights >= 0)
        and np.any(weights > 0)
    )
    rss = (
        len(result.rejection_sample(clip_weights=True, random_state=RANDOM_STATE))
        if valid_weights else 0
    )
    num_samples = len(result.samples)
    next_samples, reason = retry_decision(
        num_samples, rss, retries, min_rss, max_retries, max_n_samples, n_parallel,
        shortfall_safety_margin=shortfall_safety_margin,
    )
    efficiency = rss / num_samples if num_samples else 0.0
    result.importance_sampling_metadata[STATE_KEY] = {
        "rss": rss,
        "num_samples": num_samples,
        "efficiency": efficiency,
        "retries": retries,
        "next_samples": next_samples,
        "min_rss": min_rss,
        "shortfall_safety_margin": shortfall_safety_margin,
        "reason": reason,
    }
    logger.info(
        f"RSS check: {rss}/{min_rss}, {num_samples} proposals, "
        f"efficiency={efficiency:.6g}, restarts={retries}/{max_retries}."
    )
    if next_samples:
        target = shortfall_safety_margin * (min_rss - rss)
        logger.info(
            f"Targeting at least {target:g} additional rejection samples "
            f"with {next_samples} new proposals."
        )
    elif rss < min_rss:
        logger.warning(
            f"LOW RSS: {rss} rejection samples below min-rss={min_rss}; "
            f"{reason}. Finishing normally with the available samples."
        )
    # Publish only a complete result: Asimov detects completion by its filename.
    result.to_file(temporary)
    temporary.replace(output_file)
    return result.importance_sampling_metadata[STATE_KEY]


def finish_round(result, output_file, *, decision_file, final_result, config_file,
                 next_config=None, **kwargs):
    """Publish a round, its next ordinary sampling INI, and the PRE decision."""
    import shutil
    import re

    state = check_rss(result, output_file, **kwargs)
    if state["next_samples"]:
        # Preserve Bilby's multiline dictionaries and every other setting verbatim.
        with open(resolve_result_path(config_file)) as stream:
            config = re.sub(r"(?m)^num-samples[ \t]*=[^\n]*", "", stream.read())
        config += f"\nnum-samples = {state['next_samples']}\n"
        write_atomic(next_config, config)
    else:
        final_result = Path(final_result)
        temporary = final_result.with_suffix(".tmp")
        shutil.copyfile(output_file, temporary)
        temporary.replace(final_result)
    write_atomic(decision_file, json.dumps(state))


if __name__ == "__main__":
    import sys
    from urllib.parse import unquote

    sys.exit(pre_check(unquote(sys.argv[1])))
