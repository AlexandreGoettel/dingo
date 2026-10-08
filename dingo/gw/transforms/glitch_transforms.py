import numpy as np
import torch

from dingo.gw.domains import UniformFrequencyDomain


def antiglitch_model(
    fk: np.ndarray,
    amp: np.ndarray,
    gamma: np.ndarray,
    f0: np.ndarray,
    phi: np.ndarray,
    t0: np.ndarray,
) -> np.ndarray:
    """
    Return FD log-normal glitch.

    Parameters:
    -----------
    fk : np.ndarray (1, M)
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

    Returns:
    --------
    np.ndarray
        The FD log-normal glitch.
    """
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
    return amp.reshape(-1, 1) * np.exp(phase_term) * h0 / norm


class ClipWaveform(object):
    """Clips the absolute values of the waveforms to a maximum value.

    This is applied at the same stage of the data pipeline during training and
    inference (to the whitened strain), so that the network never sees
    out-of-distribution amplitudes.
    """

    def __init__(self, max_abs: float = 1.0):
        self.max_abs = max_abs

    def __call__(self, input_sample):
        sample = input_sample.copy()

        for ifo, w in sample["waveform"].items():
            sample["waveform"][ifo] = self.clip_array(w)

        return sample

    def clip_array(self, w):
        if isinstance(w, torch.Tensor):
            abs_w = torch.abs(w)
            mask = abs_w > self.max_abs
            return torch.where(mask, w / abs_w * self.max_abs, w)
        abs_w = np.abs(w)
        mask = abs_w > self.max_abs
        w = np.array(w, copy=True)
        w[mask] = w[mask] / abs_w[mask] * self.max_abs
        return w


class AddAntiglitch(object):
    """Adds analytic glitches based on doi.org/10.1103/PhysRevD.108.122004."""

    param_map = {
        "glitch_amp": "amp",
        "glitch_phi": "phi",
        "glitch_f0": "f0",
        "glitch_gamma": "gamma",
        "glitch_time": "t0",
    }

    def __init__(
        self,
        domain: UniformFrequencyDomain,
        colour: bool = False,
    ):
        self.domain = domain
        self.colour = colour

    def __call__(self, input_sample):
        sample = input_sample.copy()

        parameters = sample.get("parameters", {}) | sample.get(
            "extrinsic_parameters", {}
        )
        for ifo in sample["waveform"]:
            if not self.ifo_has_glitch_parameters(ifo, parameters):
                continue

            self.add_glitch_to_waveform(
                waveform=sample["waveform"],
                domain=self.domain,
                params=parameters,
                ifo=ifo,
                colour=self.colour,
                asds=sample["asds"],
            )

        return sample

    @classmethod
    def ifo_has_glitch_parameters(self, ifo, parameters):
        """Check for ifo-related glitch variables in the prior."""
        for name in self.param_map:
            if f"{ifo}_{name}" not in parameters:
                return False
        return True

    @classmethod
    def add_glitch_to_waveform(
        self,
        waveform: dict,
        domain: UniformFrequencyDomain,
        params: dict,
        ifo: str,
        colour: bool = False,
        asds: dict = None,
    ) -> None:
        """
        Add analytic glitch to a waveform for a single interferometer.

        Parameters
        ----------
        waveform : dict
            Dictionary of waveforms, modified in place.
        domain : UniformFrequencyDomain
        glitch_params : dict
            Dictionary containing glitch parameters
        ifo : str
            Interferometer name.
        colour : bool
            If True, "un-whiten" the glitch using ASDs
        asds : dict
            Dictionary of ASDs, one per interferometer. Required if colour=True.
        """
        # Extract glitch parameters for this IFO
        glitch_params = {}
        for k, v in self.param_map.items():
            glitch_params[v] = params[f"{ifo}_{k}"]

        # Make glitch time relative to geocent_time
        glitch_params["t0"] += params["geocent_time"]

        # Get analytical glitch
        glitch = antiglitch_model(
            domain.sample_frequencies,
            **{k: np.atleast_1d(v) for k, v in glitch_params.items()},
        )

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
            glitch *= asds[ifo] * domain.noise_std

        # Add to waveform, but only above fmin
        if len(waveform[ifo].shape) == 1:
            glitch[0, : domain.min_idx] = 0
            waveform[ifo] += glitch[0]
        else:  # batched
            glitch[:, : domain.min_idx] = 0
            waveform[ifo] += glitch
