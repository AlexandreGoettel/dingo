from copy import deepcopy

from bilby.gw.prior import BBHPriorDict
from bilby.gw.conversion import (
    fill_from_fixed_priors,
    convert_to_lal_binary_black_hole_parameters,
)
from bilby.core.prior import Uniform, Sine, Cosine, Prior

import numpy as np
from typing import Dict, Set, Any
import warnings

from scipy.special import logsumexp

from dingo.gw.nf_prior import NFPrior

# Silence INFO and WARNING messages from bilby
import logging

logging.getLogger("bilby").setLevel("ERROR")


class DingoGWPrior(BBHPriorDict):
    """Bilby priors plus joint normalising-flow priors.

    Dictionary entries describing flows use ``model_filename``, ``parameters``,
    ``weight`` and optional ``bounds``. Flows with the same parameter set form a
    mixture; disjoint sets are independent. Bilby's constraint sampling and
    normalization operate on the joint draws through ``sample_subset``.
    """

    def __init__(self, dictionary=None, device="cpu", **kwargs):
        if dictionary is None or isinstance(dictionary, str):
            dictionary = BBHPriorDict(dictionary, filename=kwargs.pop("filename", None))
        if isinstance(dictionary, DingoGWPrior):
            dictionary = dictionary.to_config()
        dictionary = dict(dictionary or {})
        self.flows = []
        self._flow_settings = {}
        self._flow_priors = {}
        for name, settings in list(dictionary.items()):
            if not isinstance(settings, dict):
                continue
            flow = NFPrior(device=device, **settings)
            self.flows.append(flow)
            self._flow_settings[name] = deepcopy(settings)
            del dictionary[name]
        for flow in self.flows:
            for other in self.flows:
                a, b = set(flow.parameters), set(other.parameters)
                if a != b and a & b:
                    raise ValueError("NF parameter sets must be identical or disjoint.")
            for parameter in flow.parameters:
                # A Prior marker lets Bilby classify flow parameters as non-fixed.
                self._flow_priors.setdefault(parameter, Prior(name=parameter))
        dictionary.update(self._flow_priors)
        super().__init__(dictionary, **kwargs)
        self._normalization_priors = dict(self)
        for flows in self._flow_groups().values():
            total_weight = sum(f.weight for f in flows)
            if not np.isfinite(total_weight) or total_weight <= 0:
                raise ValueError(
                    "NF mixture weights must sum to a finite positive value."
                )

    def _flow_groups(self):
        """Return active flow groups, rejecting partial replacement of a joint prior."""
        groups = {}
        for flow in self.flows:
            active = [p for p in flow.parameters if self.get(p) is self._flow_priors[p]]
            if active and len(active) != len(flow.parameters):
                raise ValueError(
                    f"A joint NF prior must be retained or replaced as a whole: {flow.parameters}."
                )
            if active:
                groups.setdefault(frozenset(active), []).append(flow)
        return groups

    def sample_subset(self, keys=(), size=None):
        """Draw whole joint samples, then select the requested parameters."""
        keys = list(keys)
        groups = self._flow_groups()
        flow_keys = set().union(*groups)
        samples = super().sample_subset([k for k in keys if k not in flow_keys], size)
        shape = () if size is None else (size,) if np.isscalar(size) else tuple(size)
        count = int(np.prod(shape))
        for parameters, flows in groups.items():
            selected = parameters.intersection(keys)
            if not selected:
                continue
            weights = np.array([f.weight for f in flows])
            choices = np.random.choice(len(flows), count, p=weights / weights.sum())
            values = {p: np.empty(count) for p in selected}
            for index, flow in enumerate(flows):
                mask = choices == index
                if mask.any():
                    draws = flow.sample(int(mask.sum()))
                    for p in selected:
                        values[p][mask] = draws[:, flow.parameters.index(p)]
            samples.update(
                {p: v.reshape(shape) if shape else v.item() for p, v in values.items()}
            )
        return samples

    def ln_prob(self, sample, axis=None, normalized=True):
        """Evaluate joint densities in physical units, including mixture weights."""
        if self.keys() != self._normalization_priors.keys() or any(
            self[k] is not p for k, p in self._normalization_priors.items()
        ):
            self._cached_normalizations.clear()
            self._normalization_priors = dict(self)
        groups = self._flow_groups()
        if not groups:
            return super().ln_prob(sample, axis=axis, normalized=normalized)
        if axis not in (None, 0):
            raise ValueError("Joint NF prior evaluation supports axis=0 or axis=None.")
        sample = {k: np.asarray(v) for k, v in sample.items()}
        flow_keys = set().union(*groups)
        factors = [self[k].ln_prob(v) for k, v in sample.items() if k not in flow_keys]
        for parameters, flows in groups.items():
            if not parameters.intersection(sample):
                continue
            if not parameters.issubset(sample):
                raise ValueError(
                    f"Joint NF density requires all parameters {sorted(parameters)}."
                )
            weight_sum = sum(f.weight for f in flows)
            factors.append(
                logsumexp(
                    [
                        f.ln_prob(sample) + np.log(f.weight / weight_sum)
                        for f in flows
                        if f.weight > 0
                    ],
                    axis=0,
                )
            )
        log_prob = np.sum(np.broadcast_arrays(*factors), axis=0)
        log_prob = self.check_ln_prob(sample, log_prob, normalized=normalized)
        return np.sum(log_prob) if axis is None else log_prob

    def prob(self, sample, **kwargs):
        """Probability corresponding to the joint log density."""
        return np.exp(self.ln_prob(sample, **kwargs))

    def copy(self):
        """Preserve the joint models and their markers when copying a prior."""
        return deepcopy(self)

    def to_config(self):
        """Serialize joint prior definitions rather than Bilby's parameter markers."""
        active = set().union(*self._flow_groups())
        return {
            **{k: v for k, v in self.items() if k not in active},
            **{
                name: deepcopy(settings)
                for name, settings in self._flow_settings.items()
                if set(settings["parameters"]).issubset(active)
            },
        }

    def sample_as_df(self, num_samples, **kwargs):
        """Return joint prior draws as a parameter table."""
        import pandas as pd

        return pd.DataFrame(self.sample(num_samples, **kwargs))


class BBHExtrinsicPriorDict(DingoGWPrior):
    """
    This class is the same as BBHPriorDict except that it does not require mass parameters.

    It also contains a method for estimating the standardization parameters.

    TODO:
        * Add support for zenith/azimuth
        * Defaults?
    """

    def default_conversion_function(self, sample):
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
                    std[key] = np.sqrt(0.25 * (np.pi ** 2) - 2).item()
                elif (
                    isinstance(p, Cosine)
                    and p.minimum == -np.pi / 2
                    and p.maximum == np.pi / 2
                ):
                    mean[key] = 0.0
                    std[key] = np.sqrt(0.25 * (np.pi ** 2) - 2).item()
                else:
                    estimation_keys.append(key)
        else:
            estimation_keys = keys

        # For remaining parameters, estimate numerically
        if len(estimation_keys) > 0:
            samples = self.sample_subset(keys, size=sample_size)
            samples = self.default_conversion_function(samples)
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
    "lambda_1": "bilby.core.prior.Uniform(minimum=0.0, maximum=5000.0, name='lambda_1')",
    "lambda_2": "bilby.core.prior.Uniform(minimum=0.0, maximum=5000.0, name='lambda_2')",
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
    Generate BBHPriorDict based on dictionary of prior settings,
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
    parameter sample obtained from the returned GWPriorDict will vary.
    """

    full_prior_settings = deepcopy(prior_settings)
    for k, v in prior_settings.items():
        if v == "default":
            full_prior_settings[k] = default_intrinsic_dict[k]

    return DingoGWPrior(full_prior_settings)


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
        if k in extrinsic_parameters or "recalib" in k or "_glitch_" in k:
            theta_extrinsic[k] = v
        else:
            theta_intrinsic[k] = v
    # set fiducial values for time and distance
    theta_intrinsic["geocent_time"] = 0
    theta_intrinsic["luminosity_distance"] = 100
    return theta_intrinsic, theta_extrinsic
