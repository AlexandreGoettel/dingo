import argparse
from pathlib import Path

from dingo.gw.result import Result
from dingo.pipe.parser import add_sampling_retry_arguments
from dingo.pipe.sampling_retry import finish_round, resolve_result_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str)
    parser.add_argument("--label", type=str)
    parser.add_argument("--extension", type=str)
    parser.add_argument("--result", type=str, nargs="+")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--previous-result")
    parser.add_argument("--decision-file")
    parser.add_argument("--final-result")
    parser.add_argument("--config-file")
    parser.add_argument("--next-config")
    add_sampling_retry_arguments(parser)
    parser.add_argument("--n-parallel", type=int, default=1)
    args = parser.parse_args()

    if args.merge:
        results = [Result(file_name=resolve_result_path(f)) for f in args.result]
        result = Result.merge(results) if len(results) > 1 else results[0]
        result.print_summary()
        label = args.label if args.decision_file else args.label.replace("_merge", "")
        output_file = Path(args.outdir) / (label + ".hdf5")
        if args.decision_file:
            finish_round(
                result, output_file, previous_file=args.previous_result,
                decision_file=args.decision_file, final_result=args.final_result,
                config_file=args.config_file, next_config=args.next_config,
                min_rss=args.min_rss, max_retries=args.max_retries,
                max_n_samples=args.max_n_samples, n_parallel=args.n_parallel,
                shortfall_safety_margin=args.shortfall_safety_margin,
            )
        else:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            result.to_file(output_file)
