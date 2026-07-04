from copy import deepcopy
import logging
from typing import Dict, Any

import numpy as np

from bilby.gw.prior import BBHPriorDict
from bilby.gw.conversion import (
    fill_from_fixed_priors,
    convert_to_lal_binary_black_hole_parameters,
)
from bilby.core.prior import Uniform, Sine, Cosine


from dingo.core.prior import DingoPrior
from dingo.core.posterior_models import NormalizingFlowPosteriorModel

# Silence INFO and WARNING messages from bilby
logging.getLogger("bilby").setLevel("ERROR")


class NFPrior:

    def __init__(self, model_filename: str, weight: float, device: str = "cuda"):
        self.flow = NormalizingFlowPosteriorModel(
            model_filename=model_filename,
            device=device,
            load_training_info=False,
        )
        self.parameters = self.flow.metadata["train_settings"]["data"]["parameters"]
        self.standardization = self.flow.metadata["train_settings"]["data"]["standardization"]
        self.weight = weight


class DingoGWPrior(DingoPrior):
    """Dingo gravitational wave prior class that wraps BBHPriorDict."""

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

            # Now expecting NF prior with parameters, model_path, weight
            self.flows.append(NFPrior(v["model_filename"], v["weight"], device=self.device))
            for param in v["parameters"]:
                out_prior_dict[param] = "flow"

        return out_prior_dict

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

        param_flow_map = {}
        if num_samples is None:
            num_samples = 1
        for flow in self.flows:
            samples = flow.flow.sample(num_samples=num_samples)
            samples = samples.detach().cpu().numpy()
            if num_samples == 1:
                samples = samples[0]
            for param, param_samples in zip(flow.parameters, samples.T):
                if param not in param_flow_map:
                    param_flow_map[param] = []
                    param_flow_map[param].append((
                        param_samples,
                        flow.weight,
                        flow.standardization["mean"][param],
                        flow.standardization["std"][param],
                    ))

        # For each parameter with flows, select and de-standardize
        for param, flow_data in param_flow_map.items():
            if len(flow_data) == 1:
                # Single flow for this parameter
                samples, _, mu, sigma = flow_data[0]
                base_samples[param] = self._reverse_standardize(samples, mu, sigma)
            else:
                # Multiple flows - choose according to weights
                weights = np.array([fd[1] for fd in flow_data])
                weights = weights / weights.sum()

                all_samples = [fd[0] for fd in flow_data]
                all_standardizations = [(fd[2], fd[3]) for fd in flow_data]

                # Choose which flow to use for each sample
                chosen_indices = np.random.choice(
                    len(flow_data),
                    size=num_samples,
                    p=weights,
                )

                # Build final samples with reverse standardization
                final_samples = np.zeros(num_samples)
                for i, flow_idx in enumerate(chosen_indices):
                    sample_val = all_samples[flow_idx][i]
                    final_samples[i] = self._reverse_standardize(
                        sample_val, *all_standardizations[flow_idx]
                    )
                base_samples[param] = final_samples

        return base_samples

    # Delegate BBHPriorDict-specific methods that aren't in dict
    def sample_subset(self, keys, size):
        """Delegate to BBHPriorDict.sample_subset"""
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
