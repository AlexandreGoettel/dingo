"""Conditional phase/psi proposals built from reusable GW likelihood overlaps."""

import bilby
import numpy as np
from bilby.gw.utils import ln_i0
from scipy.special import logsumexp

from dingo.core.multiprocessing import apply_func_with_multiprocessing


def normalize_synthetic_settings(settings, joint=True):
    """Copy settings, accept the legacy grid name, and validate before drawing."""
    settings = dict(settings)
    if "n_grid" in settings:
        n_grid = settings.pop("n_grid")
        if "n_grid_phase" in settings and settings["n_grid_phase"] != n_grid:
            raise ValueError("n_grid and n_grid_phase disagree.")
        settings["n_grid_phase"] = n_grid
    defaults = {
        "n_grid_phase": 512 if joint else 5001,
        "n_grid_psi": 128,
        "approximation_22_mode": False,
        "uniform_weight": 0.01,
        "num_processes": 1,
    }
    unknown = settings.keys() - (defaults.keys() | {"add_glitch"})
    if unknown:
        raise ValueError(f"Unknown synthetic parameter settings: {sorted(unknown)}")
    settings = {**defaults, **settings}
    for key in ("n_grid_phase", "n_grid_psi", "num_processes"):
        value = settings[key]
        minimum = 1 if key == "num_processes" else 2
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or value < minimum
        ):
            raise ValueError(f"{key} must be an integer >= {minimum}.")
    weight = settings["uniform_weight"]
    if not np.isfinite(weight) or weight < 0:
        raise ValueError("uniform_weight must be finite and nonnegative.")
    for key in ("approximation_22_mode", "add_glitch"):
        if key in settings and not isinstance(settings[key], (bool, np.bool_)):
            raise ValueError(f"{key} must be a bool.")
    return settings


def _floored_density(log_density, uniform_weight):
    density = np.exp(log_density - np.max(log_density, axis=-1, keepdims=True))
    if not np.isfinite(density).all():
        raise ValueError("Non-finite phase/psi likelihood profile.")
    return density + uniform_weight * density.mean(axis=-1, keepdims=True)


def _histogram(grid, values):
    """Normalized cell heights with the trapezoidal mass of the grid values.

    Sampling and density evaluation use the same piecewise-constant density.
    This avoids the sampling/density mismatch of Bilby's Interped.
    """
    heights = (values[..., :-1] + values[..., 1:]) / 2
    mass = heights * np.diff(grid)
    total = mass.sum(axis=-1, keepdims=True)
    return heights / total, mass / total


def _histogram_sample(grid, values, uniforms):
    heights, mass = _histogram(grid, values)
    cdf = np.cumsum(mass, axis=-1)
    cdf[:, -1] = 1.0
    cells = np.sum(uniforms[:, None] >= cdf, axis=-1)
    rows = np.arange(len(values))
    previous = np.where(cells > 0, cdf[rows, np.maximum(cells - 1, 0)], 0.0)
    draws = grid[cells] + (uniforms - previous) / heights[rows, cells]
    return draws, np.log(heights[rows, cells])


def _histogram_log_prob(grid, values, points):
    heights, _ = _histogram(grid, values)
    # Both angles are periodic, so the upper endpoint is the lower endpoint.
    points = (np.asarray(points) - grid[0]) % (grid[-1] - grid[0]) + grid[0]
    cells = np.searchsorted(grid, points, side="right") - 1
    return np.log(heights[np.arange(len(values)), cells])


def _log_prob_phase_22(z, phase, uniform_weight):
    log_conditional = (z * np.exp(2j * phase)).real - ln_i0(np.abs(z))
    log_floor = np.log(uniform_weight) if uniform_weight else -np.inf
    return (
        np.logaddexp(log_conditional, log_floor)
        - np.log(2 * np.pi)
        - np.log1p(uniform_weight)
    )


class PhasePsiProposal:
    """One conditional proposal per row of theta, independent of inference chains.

    Higher modes use q(phase) q(psi|phase). The (2,2) approximation uses the
    analytic phase integral to construct q(psi), followed by q(phase|psi).
    log_prob evaluates exactly the density used by sample_and_log_prob.
    """

    max_grid_elements = 8_000_000

    def __init__(self, likelihood, theta, **settings):
        self.likelihood = likelihood
        self.settings = normalize_synthetic_settings(settings)
        self.n = len(theta)
        self.approximation_22_mode = self.settings["approximation_22_mode"]
        self.uniform_weight = self.settings["uniform_weight"]
        self.phases = np.linspace(0, 2 * np.pi, self.settings["n_grid_phase"])
        self.psis = np.linspace(0, np.pi, self.settings["n_grid_psi"])
        self.terms = {}
        if not self.n:
            return
        producer = (
            likelihood.phase_psi_terms_22
            if self.approximation_22_mode
            else likelihood.phase_grid_terms
        )
        terms = apply_func_with_multiprocessing(
            producer, theta, min(self.settings["num_processes"], max(1, self.n // 10))
        )
        for key in terms[0]:
            if key in ("m_vals", "deltas"):
                if any(not np.array_equal(t[key], terms[0][key]) for t in terms):
                    raise ValueError("Waveform mode orders must agree across samples.")
                self.terms[key] = terms[0][key]
            else:
                self.terms[key] = np.stack([t[key] for t in terms])

    def _terms_slice(self, sl):
        return {
            k: v if k in ("m_vals", "deltas") else v[sl] for k, v in self.terms.items()
        }

    def _chunks(self):
        elements = len(self.psis) * (
            1 if self.approximation_22_mode else len(self.phases)
        )
        chunk = max(1, self.max_grid_elements // elements)
        for start in range(0, self.n, chunk):
            yield slice(start, min(start + chunk, self.n))

    def _phase_profile(self, terms):
        # Splitting the phase axis also bounds a single unusually large grid.
        n = len(terms["log_Zn"])
        phase_chunk = max(1, self.max_grid_elements // (n * len(self.psis)))
        marginal = np.empty((n, len(self.phases)))
        for start in range(0, len(self.phases), phase_chunk):
            sl = slice(start, start + phase_chunk)
            grid = self.likelihood.log_likelihood_from_phase_grid_terms(
                terms, self.phases[sl], self.psis
            )
            # psi=pi duplicates psi=0; omit it when integrating the periodic grid.
            marginal[:, sl] = logsumexp(grid[..., :-1], axis=-1)
        return _floored_density(marginal, self.uniform_weight)

    def _psi_profile(self, terms, phase):
        grid = self.likelihood.log_likelihood_from_phase_grid_terms(
            terms, phase[:, None], self.psis
        )[:, 0, :]
        return _floored_density(grid, self.uniform_weight)

    def _psi_profile_22(self, terms):
        at_psi = self.likelihood.terms_22_at_psi(terms, self.psis)
        return _floored_density(
            ln_i0(np.abs(at_psi["d_inner_h"])) - at_psi["h_inner_h"] / 2,
            self.uniform_weight,
        )

    def sample_and_log_prob(self):
        """Return phase, psi, and joint log density arrays in the input row order."""
        phase, psi, log_prob = (np.empty(self.n) for _ in range(3))
        if not self.n:
            return phase, psi, log_prob
        rng = bilby.core.utils.random.rng
        uniforms = rng.uniform(size=(self.n, 2))
        for sl in self._chunks():
            terms = self._terms_slice(sl)
            if self.approximation_22_mode:
                psi[sl], log_prob[sl] = _histogram_sample(
                    self.psis, self._psi_profile_22(terms), uniforms[sl, 0]
                )
            else:
                phase[sl], log_prob[sl] = _histogram_sample(
                    self.phases, self._phase_profile(terms), uniforms[sl, 0]
                )
                psi[sl], conditional = _histogram_sample(
                    self.psis, self._psi_profile(terms, phase[sl]), uniforms[sl, 1]
                )
                log_prob[sl] += conditional
        if self.approximation_22_mode:
            z = self.likelihood.terms_22_at_psi(self.terms, psi[:, None])["d_inner_h"][
                :, 0
            ]
            two_phase = rng.vonmises(-np.angle(z), np.abs(z))
            phase[:] = (two_phase / 2 + np.pi * rng.integers(2, size=self.n)) % (
                2 * np.pi
            )
            floor = rng.uniform(size=self.n) < self.uniform_weight / (
                1 + self.uniform_weight
            )
            phase[:] = np.where(floor, rng.uniform(0, 2 * np.pi, size=self.n), phase)
            log_prob += _log_prob_phase_22(z, phase, self.uniform_weight)
        return phase, psi, log_prob

    def log_prob(self, phase, psi):
        """Evaluate one angle pair per input row, including the proposal floor."""
        phase, psi = np.asarray(phase), np.asarray(psi)
        if phase.shape != (self.n,) or psi.shape != (self.n,):
            raise ValueError("Provide one phase and psi per proposal row.")
        out = np.empty(self.n)
        for sl in self._chunks():
            terms = self._terms_slice(sl)
            if self.approximation_22_mode:
                out[sl] = _histogram_log_prob(
                    self.psis, self._psi_profile_22(terms), psi[sl]
                )
                z = self.likelihood.terms_22_at_psi(terms, psi[sl, None])["d_inner_h"][
                    :, 0
                ]
                out[sl] += _log_prob_phase_22(z, phase[sl], self.uniform_weight)
            else:
                out[sl] = _histogram_log_prob(
                    self.phases, self._phase_profile(terms), phase[sl]
                )
                out[sl] += _histogram_log_prob(
                    self.psis, self._psi_profile(terms, phase[sl]), psi[sl]
                )
        return out
