r"""Validate joint recovery with a trained network and a prepared Dingo event.

Run from the repository root in the model's environment:

    python examples/validate_phase_psi_recovery.py --model model.pt \
        --event event_data.hdf5 --output recovered.hdf5

The network must infer neither phase nor psi and provide sample log probabilities
(a standard posterior model or an unconditional density-recovery model). The
event is an EventDataset, as produced by dingo_pipe. Use --add-glitch when the
network infers Antiglitch parameters; use --approximation-22-mode to try the
analytic phase proposal. The higher-mode path requires spin_conversion_phase=0.
A corner plot comparing recovered and importance-weighted samples is saved beside
the result as <output_stem>_corner.pdf, including phase and psi.
"""

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from bilby.core.prior import Constraint
from bilby.core.utils import random

from dingo.core.posterior_models.build_model import build_model_from_kwargs
from dingo.gw.data.event_dataset import EventDataset
from dingo.gw.inference.gw_samplers import GWSampler
from dingo.gw.inference.synthetic_parameters import PhasePsiProposal
from dingo.gw.result import Result


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--event", required=True)
    parser.add_argument("--output", default="phase_psi_recovered.hdf5")
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-grid-phase", type=int, default=512)
    parser.add_argument("--n-grid-psi", type=int, default=128)
    parser.add_argument("--approximation-22-mode", action="store_true")
    parser.add_argument("--add-glitch", action="store_true")
    parser.add_argument("--use-base-domain", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.num_samples < 1:
        parser.error("--num-samples must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = build_model_from_kwargs(
        filename=args.model, device=args.device, load_training_info=False
    )
    sampler = GWSampler(model=model)
    if {"phase", "psi"}.intersection(sampler.inference_parameters):
        parser.error("The network must infer neither phase nor psi.")
    event = EventDataset(file_name=args.event)
    sampler.event_metadata = deepcopy(event.settings)
    sampler.context = deepcopy(event.data)
    sampler.run_sampler(args.num_samples, batch_size=min(128, args.num_samples))
    result = sampler.to_result()
    if "log_prob" not in result.samples:
        raise ValueError(
            "The model must provide log_prob; use density recovery for GNPE samples."
        )
    if not args.add_glitch and any("_glitch_" in k for k in result.samples):
        parser.error("This model infers glitch parameters; include --add-glitch.")
    result.use_base_domain = args.use_base_domain
    original_log_prob = result.samples["log_prob"].to_numpy(copy=True)
    parameters = [k for k, p in result.prior.items() if not isinstance(p, Constraint)]
    theta = result.samples[parameters]
    valid = np.isfinite(
        result.prior.ln_prob(theta[result.search_parameter_keys], axis=0)
    )
    valid &= (
        np.asarray(
            result.prior.evaluate_constraints({k: theta[k].to_numpy() for k in theta})
        )
        != 0
    )
    if not valid.any():
        raise ValueError(
            "No samples within the prior; increase --num-samples or check the model/event."
        )
    settings = dict(
        n_grid_phase=args.n_grid_phase,
        n_grid_psi=args.n_grid_psi,
        approximation_22_mode=args.approximation_22_mode,
        num_processes=args.num_processes,
        add_glitch=args.add_glitch,
    )
    result.sample_synthetic_parameters(settings)
    for name, upper in (("phase", 2 * np.pi), ("psi", np.pi)):
        assert result.samples[name].between(0, upper, inclusive="left").all()
        assert name in result.prior and getattr(result, f"{name}_prior") is None

    # Re-evaluate q at the actual sampled points for a few valid rows.
    indices = np.flatnonzero(valid)[:8]
    result._build_likelihood()
    likelihood = result.likelihood
    proposal = PhasePsiProposal(likelihood, theta.iloc[indices], **settings)
    drawn = result.samples.iloc[indices]
    phase, psi = drawn["phase"].to_numpy(), drawn["psi"].to_numpy()
    q = proposal.log_prob(phase, psi)
    np.testing.assert_allclose(
        q,
        drawn["log_prob"].to_numpy() - original_log_prob[indices],
        rtol=1e-7,
        atol=1e-7,
    )
    if args.approximation_22_mode:
        terms = likelihood.terms_22_at_psi(proposal.terms, psi[:, None])
        reconstructed = likelihood.log_likelihood_22_from_terms(
            terms, phase[:, None, None]
        )[:, 0, 0]
    else:
        reconstructed = likelihood.log_likelihood_from_phase_grid_terms(
            proposal.terms, phase[:, None], psi[:, None]
        )[:, 0, 0]
    direct = np.array(
        [
            likelihood.log_likelihood(row.to_dict())
            for _, row in drawn[parameters + ["phase", "psi"]].iterrows()
        ]
    )
    print(f"Proposal log densities agree for {len(indices)} recovered samples.")
    print(
        "Maximum |grid/approximation log L - direct log L|: "
        f"{np.max(np.abs(reconstructed - direct)):.6g}"
    )
    print("Importance sampling evaluates the target likelihood directly.")

    result.importance_sample(num_processes=args.num_processes)
    result.print_summary()
    result.to_file(args.output)
    reloaded = Result(file_name=args.output)
    assert reloaded.add_glitch == args.add_glitch
    assert reloaded.synthetic_parameters_kwargs == result.synthetic_parameters_kwargs
    np.testing.assert_allclose(
        reloaded.samples[["phase", "psi", "log_prob"]],
        result.samples[["phase", "psi", "log_prob"]],
    )
    output_path = Path(args.output)
    corner_path = output_path.with_name(f"{output_path.stem}_corner.pdf")
    reloaded.plot_corner(
        parameters=sorted(reloaded.search_parameter_keys),
        filename=str(corner_path),
    )
    print(f"Recovery, importance sampling, and save/reload complete: {args.output}")
    print(f"Corner plot saved to: {corner_path}")


if __name__ == "__main__":
    main()
