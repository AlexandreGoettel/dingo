import logging
from copy import deepcopy
from typing import Any, Dict

import numpy as np
from bilby.core.prior import Cosine, Sine, Uniform
from bilby.gw.conversion import (convert_to_lal_binary_black_hole_parameters,
                                 fill_from_fixed_priors)
from bilby.gw.prior import BBHPriorDict

from dingo.core.posterior_models import NormalizingFlowPosteriorModel
from dingo.core.prior import DingoPrior

# Silence INFO and WARNING messages from bilby
logging.getLogger("bilby").setLevel("ERROR")


class NFPrior:

    def __init__(
        self,
        model_filename: str,
        weight: float,
        bounds: Dict[str, Dict] = None,
        device: str = "cuda",
    ):
        self.flow = NormalizingFlowPosteriorModel(
            model_filename=model_filename,
            device=device,
            load_training_info=False,
        )
        self.parameters = self.flow.metadata["train_settings"]["data"]["parameters"]
        self.standardization = self.flow.metadata["train_settings"]["data"][
            "standardization"
        ]
        self.weight = weight
        self.model_filename = model_filename
        self.bounds = bounds

    def min(self, x):
        """If configured in train.yml, return the minimum allowed value for parameter x."""
        if self.bounds is None or self.bounds.get(x, {}).get("min") is None:
            return None
        return float(self.bounds[x]["min"])

    def max(self, x):
        """If configured in train.yml, return the maximum allowed value for parameter x."""
        if self.bounds is None or self.bounds.get(x, {}).get("max") is None:
            return None
        return float(self.bounds[x]["max"])


class DingoGWPrior(DingoPrior):
    """Dingo gravitational wave prior class that wraps BBHPriorDict."""

    # When sampling from an NF prior, invalid draws are rejected and redrawn
    # (see _sample_flow). If the fraction of valid draws is below this, the
    # flow is considered broken and sampling fails.
    MIN_SAMPLING_EFFICIENCY = 0.5
    MIN_DRAWS_FOR_EFFICIENCY_CHECK = 1000

    def __init__(self, prior_dict: Dict, device: str = "cuda"):
        """
        Initialize DingoGWPrior with a BBHPriorDict.

        Parameters
        ----------
        prior_dict : BBHPriorDict
            The bilby BBHPriorDict to wrap. Its contents are copied into self
            so that innate dict functions work naturally.
        """
        self.device = device
        self.flows = []
        self._prior_dict = BBHPriorDict(
            self.parse_prior_dict(prior_dict),
            conversion_function=self.conversion_function,
        )
        super().__init__(self._prior_dict)

    @property
    def conversion_function(self):
        """Override for use in the BBHPriorDict."""
        return None

    def parse_prior_dict(self, prior_dict: Dict) -> dict:
        """Convert a simple config dict to prior dict for BBHPriorDict, can handle NF priors."""
        out_prior_dict = {}
        for k, v in prior_dict.items():
            if not isinstance(v, dict):
                out_prior_dict[k] = v
                continue

            # Now expecting NF prior with parameters, model_filename, weight,
            # and optional bounds
            self.flows.append(
                NFPrior(
                    v["model_filename"],
                    v["weight"],
                    bounds=v.get("bounds"),
                    device=self.device,
                )
            )
            for param in v["parameters"]:
                out_prior_dict[param] = "flow"

        if self.flows:
            self._validate_flow_parameter_sets()

        return out_prior_dict

    def _validate_flow_parameter_sets(self):
        """
        Check that the NF priors have either identical or pairwise
        non-overlapping parameter sets.

        Flows with identical parameter sets are combined into a per-event
        mixture (see sample); flows with non-overlapping parameter sets are
        sampled independently. Any other configuration is ambiguous and
        rejected.
        """
        for i, flow_i in enumerate(self.flows):
            for j in range(i + 1, len(self.flows)):
                params_i = set(flow_i.parameters)
                params_j = set(self.flows[j].parameters)
                if params_i != params_j and params_i & params_j:
                    raise ValueError(
                        f"NF priors must have either identical or "
                        f"non-overlapping parameter sets. Got overlapping, "
                        f"but not identical, sets {sorted(params_i)} and "
                        f"{sorted(params_j)} (shared parameters: "
                        f"{sorted(params_i & params_j)})."
                    )

    @staticmethod
    def _reverse_standardize(samples: np.ndarray, mu: float, sigma: float):
        """
        Apply reverse standardization to samples.

        Parameters
        ----------
        samples : scalar or np.ndarray
            Standardized samples to transform
        standardization : Dict[str, float]
            Dictionary with 'mean' and 'std' keys for reverse standardization

        Returns
        -------
        scalar or np.ndarray
            De-standardized samples
        """
        return samples * sigma + mu

    @staticmethod
    def _valid_flow_draws(flow, draws):
        """
        Marking valid draws: all values finite, and (optionally) within the
        per-parameter bounds configured for the flow (see NFPrior.min/max;
        both bounds are exclusive).
        """
        mask = np.isfinite(draws).all(axis=1)
        for col, param in enumerate(flow.parameters):
            minimum = flow.min(param)
            maximum = flow.max(param)
            if minimum is None and maximum is None:
                continue
            destandardized = DingoGWPrior._reverse_standardize(
                draws[:, col],
                flow.standardization["mean"][param],
                flow.standardization["std"][param],
            )
            if minimum is not None:
                mask &= destandardized > minimum
            if maximum is not None:
                mask &= destandardized < maximum
        return mask

    @staticmethod
    def _draw_flow(flow, num_samples):
        """
        Draw num_samples samples from a flow, in standardized units, and
        mark valid draws (see valid_flow_draws).
        """
        draws = flow.flow.sample(num_samples=num_samples)
        draws = draws.detach().cpu().numpy()
        return draws, DingoGWPrior._valid_flow_draws(flow, draws)

    def _sample_flow(self, flow, num_samples):
        """
        Draw num_samples samples from a flow, in standardized units.
        Invalid draws (see valid_flow_draws) are rejected and replaced by
        fresh draws from the same flow, until num_samples valid draws are
        collected.
        """
        samples = np.empty((num_samples, len(flow.parameters)))
        num_valid = num_drawn = 0
        while num_valid < num_samples:
            draws, mask = self._draw_flow(flow, num_samples - num_valid)
            n = int(mask.sum())
            samples[num_valid : num_valid + n] = draws[mask]
            num_valid += n
            num_drawn += len(draws)
            if (
                num_drawn >= self.MIN_DRAWS_FOR_EFFICIENCY_CHECK
                and num_valid < self.MIN_SAMPLING_EFFICIENCY * num_drawn
            ):
                raise ValueError(
                    f"NF prior sampling efficiency ({num_valid / num_drawn * 100:.2f} %) is "
                    f"below {self.MIN_SAMPLING_EFFICIENCY}. Please verify its validity."
                )
        return samples

    def nf_sampling_efficiency(self, num_samples: int = 1000) -> Dict[str, float]:
        """
        Measure the sampling efficiency of each NF prior: the fraction of
        draws that are valid (see valid_flow_draws), based on a single batch
        of num_samples draws per flow, without rejection.
        """
        efficiencies = {}
        for flow in self.flows:
            draws, mask = self._draw_flow(flow, num_samples)
            efficiencies[", ".join(flow.parameters)] = mask.sum() / len(draws)
        return efficiencies

    def sample(self, num_samples: int, **kwargs) -> Dict[str, Any]:
        """
        Sample parameters from the prior.

        Parameters
        ----------
        num_samples : int
            Number of samples to draw from the prior.
        **kwargs : dict
            Additional keyword arguments passed to the underlying sample method.

        Returns
        -------
        Dict[str, Any]
            Dictionary of sampled parameters, where keys are parameter names
            and values are numpy arrays of shape (num_samples,).
        """
        base_samples = self._prior_dict.sample(num_samples, **kwargs)
        if not self.flows:
            return base_samples

        if num_samples is None:
            num_samples = 1

        flow_draws = [self._sample_flow(flow, num_samples) for flow in self.flows]
        weights = np.array([flow.weight for flow in self.flows], dtype=float)

        # Group the flows by their parameter sets. Within a group, one flow
        # is selected per sample, so that all parameters of a sample belonging
        # to that group come from a single joint draw.
        groups = {}
        for flow_idx, flow in enumerate(self.flows):
            groups.setdefault(frozenset(flow.parameters), []).append(flow_idx)

        for flow_indices in groups.values():
            if len(flow_indices) == 1:
                flow = self.flows[flow_indices[0]]
                for col, param in enumerate(flow.parameters):
                    base_samples[param] = self._reverse_standardize(
                        flow_draws[flow_indices[0]][:, col],
                        flow.standardization["mean"][param],
                        flow.standardization["std"][param],
                    )
                continue

            # Per-sample mixture within the group: one choice of flow
            # shared by all parameters of the group.
            group_weights = weights[flow_indices]
            group_weights = group_weights / group_weights.sum()
            chosen = np.random.choice(
                len(flow_indices), size=num_samples, p=group_weights
            )

            final_samples = {
                param: np.empty(num_samples)
                for param in self.flows[flow_indices[0]].parameters
            }
            for local_idx, flow_idx in enumerate(flow_indices):
                mask = chosen == local_idx
                if not mask.any():
                    continue
                flow = self.flows[flow_idx]
                for col, param in enumerate(flow.parameters):
                    final_samples[param][mask] = self._reverse_standardize(
                        flow_draws[flow_idx][mask, col],
                        flow.standardization["mean"][param],
                        flow.standardization["std"][param],
                    )
            base_samples.update(final_samples)

        return base_samples

    # Delegate BBHPriorDict-specific methods that aren't in dict
    def sample_subset(self, keys, size):
        """Delegate to BBHPriorDict.sample_subset"""
        # TODO: need to update _prior_dict with setattr?
        for flow in self.flows:
            if any(k in flow.parameters for k in keys):
                # Some subset parameters are from a flow.
                # "Dirty" solution is to sample from the whole prior,
                # and then restrict to the subset
                samples = self.sample(size)
                return {k: v for k, v in samples.items() if k in keys}

        return self._prior_dict.sample_subset(keys, size)

    def ln_prob(self, *args, **kwargs):
        """Delegate to BBHPriorDict.ln_prob"""
        return self._prior_dict.ln_prob(*args, **kwargs)


class BBHExtrinsicPriorDict(DingoGWPrior):

    def conversion_function(self, sample):
        out_sample = fill_from_fixed_priors(sample, self)
        out_sample, _ = convert_to_lal_binary_black_hole_parameters(out_sample)

        # The previous call sometimes adds phi_jl, phi_12 parameters. These are
        # not needed so they can be deleted.
        if "phi_jl" in out_sample.keys():
            del out_sample["phi_jl"]
        if "phi_12" in out_sample.keys():
            del out_sample["phi_12"]

        return out_sample

    def mean_std(self, keys=([]), sample_size=50000, force_numerical=False):
        """
        Calculate the mean and standard deviation over the prior.

        Parameters
        ----------
        keys: list(str)
            A list of desired parameter names
        sample_size: int
            For nonanalytic priors, number of samples to use to estimate the
            result.
        force_numerical: bool (False)
            Whether to force a numerical estimation of result, even when
            analytic results are available (useful for testing)

        Returns dictionaries for the means and standard deviations.

        TODO: Fix for constrained priors. Shouldn't be an issue for extrinsic parameters.
        """
        mean = {}
        std = {}

        if not force_numerical:
            # First try to calculate analytically (works for standard priors)
            estimation_keys = []
            for key in keys:
                p = self[key]
                # A few analytic cases
                if isinstance(p, Uniform):
                    mean[key] = (p.maximum + p.minimum) / 2.0
                    std[key] = np.sqrt((p.maximum - p.minimum) ** 2.0 / 12.0).item()
                elif isinstance(p, Sine) and p.minimum == 0.0 and p.maximum == np.pi:
                    mean[key] = np.pi / 2.0
                    std[key] = np.sqrt(0.25 * (np.pi**2) - 2).item()
                elif (
                    isinstance(p, Cosine)
                    and p.minimum == -np.pi / 2
                    and p.maximum == np.pi / 2
                ):
                    mean[key] = 0.0
                    std[key] = np.sqrt(0.25 * (np.pi**2) - 2).item()
                else:
                    estimation_keys.append(key)
        else:
            estimation_keys = keys

        # For remaining parameters, estimate numerically
        if len(estimation_keys) > 0:
            samples = self.sample_subset(keys, size=sample_size)
            samples = self.conversion_function(samples)
            for key in estimation_keys:
                if key in samples.keys():
                    mean[key] = np.mean(samples[key]).item()
                    std[key] = np.std(samples[key]).item()

        return mean, std


default_extrinsic_dict = {
    "dec": "bilby.core.prior.Cosine(minimum=-np.pi/2, maximum=np.pi/2, name='dec')",
    "ra": 'bilby.core.prior.Uniform(minimum=0., maximum=2*np.pi, boundary="periodic", name="ra")',
    "geocent_time": "bilby.core.prior.Uniform(minimum=-0.1, maximum=0.1, name='geocent_time')",
    "psi": 'bilby.core.prior.Uniform(minimum=0.0, maximum=np.pi, boundary="periodic", name="psi")',
    "luminosity_distance": "bilby.core.prior.Uniform(minimum=100.0, maximum=6000.0, name='luminosity_distance')",
}

default_intrinsic_dict = {
    "mass_1": "bilby.core.prior.Constraint(minimum=10.0, maximum=80.0, name='mass_1')",
    "mass_2": "bilby.core.prior.Constraint(minimum=10.0, maximum=80.0, name='mass_2')",
    "mass_ratio": "bilby.gw.prior.UniformInComponentsMassRatio(minimum=0.125, maximum=1.0, name='mass_ratio')",
    "chirp_mass": "bilby.gw.prior.UniformInComponentsChirpMass(minimum=25.0, maximum=100.0, name='chirp_mass')",
    "luminosity_distance": 1000.0,
    "theta_jn": "bilby.core.prior.Sine(minimum=0.0, maximum=np.pi, name='theta_jn')",
    "phase": 'bilby.core.prior.Uniform(minimum=0.0, maximum=2*np.pi, boundary="periodic", name="phase")',
    "a_1": "bilby.core.prior.Uniform(minimum=0.0, maximum=0.99, name='a_1')",
    "a_2": "bilby.core.prior.Uniform(minimum=0.0, maximum=0.99, name='a_2')",
    "tilt_1": "bilby.core.prior.Sine(minimum=0.0, maximum=np.pi, name='tilt_1')",
    "tilt_2": "bilby.core.prior.Sine(minimum=0.0, maximum=np.pi, name='tilt_2')",
    "phi_12": 'bilby.core.prior.Uniform(minimum=0.0, maximum=2*np.pi, boundary="periodic", name="phi_12")',
    "phi_jl": 'bilby.core.prior.Uniform(minimum=0.0, maximum=2*np.pi, boundary="periodic", name="phi_jl")',
    "geocent_time": 0.0,
}

default_inference_parameters = [
    "chirp_mass",
    "mass_ratio",
    "phase",
    "a_1",
    "a_2",
    "tilt_1",
    "tilt_2",
    "phi_12",
    "phi_jl",
    "theta_jn",
    "luminosity_distance",
    "geocent_time",
    "ra",
    "dec",
    "psi",
]


def build_prior_with_defaults(prior_settings: Dict[str, str]):
    """
    Generate DingoGWPrior based on dictionary of prior settings,
    allowing for default values.

    Parameters
    ----------
    prior_settings: Dict
        A dictionary containing prior definitions for intrinsic parameters
        Allowed values for each parameter are:
            * 'default' to use a default prior
            * a string for a custom prior, e.g.,
               "Uniform(minimum=10.0, maximum=80.0, name=None, latex_label=None, unit=None, boundary=None)"

    Depending on the particular prior choices the dimensionality of a
    parameter sample obtained from the returned DingoGWPrior will vary.

    Returns
    -------
    DingoGWPrior
        A DingoGWPrior object wrapping a BBHPriorDict.
    """

    full_prior_settings = deepcopy(prior_settings)
    for k, v in prior_settings.items():
        if v == "default":
            full_prior_settings[k] = default_intrinsic_dict[k]

    bbh_prior_dict = BBHPriorDict(full_prior_settings)
    return DingoGWPrior(bbh_prior_dict)


def split_off_extrinsic_parameters(theta):
    """
    Split theta into intrinsic and extrinsic parameters.

    Parameters
    ----------
    theta: dict
        BBH parameters. Includes intrinsic parameters to be passed to waveform
        generator, and extrinsic parameters for detector projection.

    Returns
    -------
    theta_intrinsic: dict
        BBH intrinsic parameters.
    theta_extrinsic: dict
        BBH extrinsic parameters (includes calibration parameters).
    """
    extrinsic_parameters = ["geocent_time", "luminosity_distance", "ra", "dec", "psi"]
    theta_intrinsic = {}
    theta_extrinsic = {}
    for k, v in theta.items():
        if k in extrinsic_parameters or "recalib" in k:
            theta_extrinsic[k] = v
        else:
            theta_intrinsic[k] = v
    # set fiducial values for time and distance
    theta_intrinsic["geocent_time"] = 0
    theta_intrinsic["luminosity_distance"] = 100
    return theta_intrinsic, theta_extrinsic
