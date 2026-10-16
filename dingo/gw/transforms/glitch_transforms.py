"""Analytic AntiGlitch signals and optional network amplitude preprocessing."""

import numpy as np
import torch

from dingo.gw.domains import UniformFrequencyDomain, MultibandedFrequencyDomain


def antiglitch_model(
    fk: np.ndarray,
    amp: np.ndarray,
    gamma: np.ndarray,
    f0: np.ndarray,
    phi: np.ndarray,
    t0: np.ndarray,
    normalization_frequencies: np.ndarray = None,
) -> np.ndarray:
    """
    Return FD log-normal glitch.

    Parameters:
    -----------
    fk : np.ndarray (M,)
        Frequency array.
    amp : np.ndarray (N,)
        Amplitude.
    gamma : np.ndarray (N,)
        Shape parameter.
    f0 : np.ndarray (N,)
        Central frequency.
    phi : np.ndarray (N,)
        Phase offset.
    t0 : np.ndarray (N,)
        Time offset.
    normalization_frequencies : np.ndarray, optional
        Training-grid frequencies defining amp. Defaults to fk.

    Returns:
    --------
    np.ndarray
        The FD log-normal glitch.
    """
    fk = np.asarray(fk, dtype=float).reshape(-1)
    amp, gamma, f0, phi, t0 = [
        np.asarray(v).reshape(-1) for v in np.broadcast_arrays(amp, gamma, f0, phi, t0)
    ]
    if (
        not all(np.isfinite(v).all() for v in (amp, gamma, f0, phi, t0))
        or np.any(f0 <= 0)
        or np.any(gamma <= 0)
    ):
        raise ValueError(
            "AntiGlitch parameters must be finite, with positive f0 and gamma."
        )
    mask = fk > 0
    h0 = np.zeros((len(f0), len(fk)), dtype=np.complex128)
    fk = fk.reshape(1, -1)

    h0[..., mask] = np.exp(
        -0.5
        * gamma.reshape(-1, 1)
        * (np.log(fk[..., mask]) - np.log(f0.reshape(-1, 1))) ** 2
    )
    phase_term = 1j * phi.reshape(-1, 1) - 2j * np.pi * fk * t0.reshape(-1, 1)
    norm = np.sum(np.abs(h0), axis=1, keepdims=True)
    if normalization_frequencies is not None:
        frequencies = np.asarray(normalization_frequencies, dtype=float)
        frequencies = frequencies[frequencies > 0]
        norm = np.sum(
            np.exp(
                -0.5 * gamma[:, None] * (np.log(frequencies) - np.log(f0[:, None])) ** 2
            ),
            axis=1,
            keepdims=True,
        )
    return amp.reshape(-1, 1) * np.exp(phase_term) * h0 / norm


class ClipWaveform(object):
    """Clips the absolute values of the waveforms to a maximum value.

    This is applied at the same stage of the data pipeline during training and
    inference (to the whitened strain), so that the network never sees
    out-of-distribution amplitudes.
    """

    def __init__(self, max_abs: float = 1.0):
        if not np.isfinite(max_abs) or max_abs <= 0:
            raise ValueError("max_abs must be finite and positive.")
        self.max_abs = max_abs

    def __call__(self, input_sample):
        sample = input_sample.copy()
        sample["waveform"] = sample["waveform"].copy()

        for ifo, w in sample["waveform"].items():
            sample["waveform"][ifo] = self.clip_array(w)

        return sample

    def clip_array(self, w):
        if isinstance(w, torch.Tensor):
            abs_w = torch.abs(w)
            return w / torch.clamp(abs_w / self.max_abs, min=1)
        abs_w = np.abs(w)
        mask = abs_w > self.max_abs
        w = np.array(w, copy=True)
        w[mask] = w[mask] / abs_w[mask] * self.max_abs
        return w


class ScaleWaveformLog(object):
    """Compresses waveform amplitudes above max_abs with a logarithm.

    For magnitudes m = |w|, the (continuous injective) transform is

        m' = m                                          for m <= max_abs
        m' = max_abs * (1 + slope * log(m / max_abs))   for m > max_abs

    applied per bin while preserving phase, i.e., w' = (m' / m) * w. slope
    controls the strength of the compression.
    """

    def __init__(self, max_abs: float = 1.0, slope: float = 1.0):
        if not np.isfinite(slope) or slope <= 0:
            raise ValueError(f"slope must be positive, got {slope}.")
        if not np.isfinite(max_abs) or max_abs <= 0:
            raise ValueError("max_abs must be finite and positive.")
        self.max_abs = max_abs
        self.slope = slope

    def __call__(self, input_sample):
        sample = input_sample.copy()
        sample["waveform"] = sample["waveform"].copy()

        for ifo, w in sample["waveform"].items():
            sample["waveform"][ifo] = self.scale_array(w)

        return sample

    def scale_array(self, w):
        if isinstance(w, torch.Tensor):
            abs_w = torch.abs(w)
            mask = abs_w > self.max_abs
            w = w.clone()
            m_scaled = self.max_abs * (
                1 + self.slope * torch.log(abs_w[mask] / self.max_abs)
            )
            w[mask] = w[mask] / abs_w[mask] * m_scaled
            return w
        abs_w = np.abs(w)
        mask = abs_w > self.max_abs
        w = np.array(w, copy=True)
        m_scaled = self.max_abs * (1 + self.slope * np.log(abs_w[mask] / self.max_abs))
        w[mask] = w[mask] / abs_w[mask] * m_scaled
        return w


class AddAntiglitch(object):
    """Add analytic glitches with times relative to CBC geocent_time.

    Based on doi.org/10.1103/PhysRevD.108.122004.
    """

    param_map = {
        "glitch_amp": "amp",
        "glitch_phi": "phi",
        "glitch_f0": "f0",
        "glitch_gamma": "gamma",
        "glitch_time": "t0",
    }

    def __init__(
        self,
        domain: UniformFrequencyDomain | MultibandedFrequencyDomain,
        colour: bool = False,
    ):
        self.domain = domain
        self.colour = colour

    def __call__(self, input_sample):
        sample = input_sample.copy()
        sample["waveform"] = sample["waveform"].copy()

        parameters = sample.get("parameters", {}) | sample.get(
            "extrinsic_parameters", {}
        )
        for ifo in sample["waveform"]:
            if not self.ifo_has_glitch_parameters(ifo, parameters):
                continue
            sample["waveform"][ifo] = sample["waveform"][ifo].copy()
            self.add_glitch_to_waveform(
                waveform=sample["waveform"],
                domain=self.domain,
                params=parameters,
                ifo=ifo,
                colour=self.colour,
                asds=sample.get("asds"),
            )

        return sample

    @classmethod
    def ifo_has_glitch_parameters(cls, ifo, parameters):
        """Check for ifo-related glitch variables in the prior."""
        expected = {f"{ifo}_{name}" for name in cls.param_map}
        present = expected.intersection(parameters)
        if present and present != expected:
            raise ValueError(
                f"Incomplete AntiGlitch parameters for {ifo}: missing {sorted(expected - present)}."
            )
        return bool(present)

    @classmethod
    def add_glitch_to_waveform(
        self,
        waveform: dict,
        domain: UniformFrequencyDomain | MultibandedFrequencyDomain,
        params: dict,
        ifo: str,
        colour: bool = False,
        asds: dict = None,
        normalization_domain: UniformFrequencyDomain = None,
    ) -> None:
        """
        Add analytic glitch to a waveform for a single interferometer.

        Parameters
        ----------
        waveform : dict
            Dictionary of waveforms, modified in place.
        domain : UniformFrequencyDomain
        params : dict
        ifo : str
            Interferometer name.
        colour : bool
            If True, "un-whiten" the glitch using ASDs
        asds : dict
            Dictionary of ASDs, one per interferometer. Required if colour=True.
        normalization_domain : UniformFrequencyDomain, optional
            Training base grid defining the amplitude and whitening scale.
            Defaults to the base grid of domain.
        """
        # Extract glitch parameters for this IFO
        glitch_params = {}
        for k, v in self.param_map.items():
            glitch_params[v] = params[f"{ifo}_{k}"]

        # Make glitch time relative to geocent_time
        glitch_params["t0"] = glitch_params["t0"] + params["geocent_time"]

        # The amplitude convention is defined on the uniform base grid. Decimate
        # whitened glitches just as event data are decimated before inference.
        base = getattr(domain, "base_domain", domain)
        reference = normalization_domain if normalization_domain is not None else base
        glitch = antiglitch_model(
            base.sample_frequencies,
            normalization_frequencies=(
                reference.sample_frequencies if reference != base else None
            ),
            **{k: np.atleast_1d(v) for k, v in glitch_params.items()},
        )
        # Preserve physical strain / ASD when delta_f changes the whitening scale.
        glitch *= reference.noise_std / base.noise_std
        glitch[..., : base.min_idx] = 0
        if isinstance(domain, MultibandedFrequencyDomain):
            glitch = domain.decimate(glitch) * base.noise_std / domain.noise_std

        if not np.isfinite(glitch).all():
            raise ValueError(
                f"Glitch model produced non-finite values for {ifo} with parameters: "
                + ", ".join(
                    f"{k} in [{np.min(v):.3g}, {np.max(v):.3g}]"
                    for k, v in glitch_params.items()
                )
                + "."
            )

        # Apply colouring (un-whitening) if needed
        if colour:
            if asds is None:
                raise ValueError("asds must be provided when colour=True")
            glitch *= (
                np.where(np.isfinite(asds[ifo]), asds[ifo], 0.0) * domain.noise_std
            )

        # Add to waveform, but only above fmin
        if len(waveform[ifo].shape) == 1:
            glitch[0, : domain.min_idx] = 0
            waveform[ifo] += glitch[0]
        else:  # batched
            glitch[:, : domain.min_idx] = 0
            waveform[ifo] += glitch
