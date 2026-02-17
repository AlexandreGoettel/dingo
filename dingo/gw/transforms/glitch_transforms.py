import numpy as np

from dingo.gw.domains import UniformFrequencyDomain


def antiglitch_model(
    fk: np.ndarray,
    amp: np.ndarray,
    gamma: np.ndarray,
    f0: np.ndarray,
    phi: np.ndarray,
    t0: np.ndarray
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

    h0[..., mask] = np.exp(-0.5 * gamma.reshape(-1, 1) * (np.log(fk[..., mask]) - np.log(f0.reshape(-1, 1))) ** 2)
    phase_term = 1j * phi.reshape(-1, 1) - 2j * np.pi * fk * t0.reshape(-1, 1)
    norm = np.sum(np.abs(h0), axis=1, keepdims=True)
    return amp.reshape(-1, 1) * np.exp(phase_term) * h0 / norm


class AddAntiglitch(object):
    """Adds analytic glitches based on doi.org/10.1103/PhysRevD.108.122004."""

    def __init__(self,
                 domain: UniformFrequencyDomain,
                 colour: bool = False,
    ):
        self.domain = domain
        self.colour = colour
        self.param_map = {
            "glitch_time": "t0",
            "glitch_amp": "amp",
            "glitch_phi": "phi",
            "glitch_f0": "f0",
            "glitch_gamma": "gamma",
        }

    def __call__(self, input_sample):
        sample = input_sample.copy()

        # Check if ifo-related glitch variables are in the prior
        ifos = set(sample["waveform"].keys())
        for ifo in sample["waveform"].keys():
            for name in self.param_map:
                if f"{ifo}_{name}" not in sample["extrinsic_parameters"]\
                        and f"{ifo}_{name}" not in sample.get("parameters"):
                    ifos.remove(ifo)
                    break

        for ifo in ifos:
            # Place the glitch time prior to be relative to geocent_time
            params = {}
            for source in sample.get("parameters", {}), sample.get("extrinsic_parameters", {}):
                params.update(source)

            params[f"{ifo}_glitch_time"] += params["geocent_time"]

            glitch = antiglitch_model(
                self.domain.sample_frequencies,
                **{
                    v: np.atleast_1d(params[f"{ifo}_{k}"])
                    for k, v in self.param_map.items()
                },
            )

            if self.colour:  # "un-whiten"
                glitch *= sample["asds"][ifo] * self.domain.noise_std

            if len(sample["waveform"][ifo].shape) == 1:
                sample["waveform"][ifo] += glitch[0]
            else:  # batched, ergo len(shape) == 2
                sample["waveform"][ifo] += glitch

        return sample
