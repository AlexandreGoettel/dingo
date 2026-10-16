"""Direct likelihood checks for AntiGlitch and upstream synthetic phase/psi."""

import numpy as np
import pandas as pd
import pytest
import torch

from dingo.gw.domains import UniformFrequencyDomain, MultibandedFrequencyDomain
from dingo.gw.injection import Injection
from dingo.gw.inference.context import GWSamplerContext
from dingo.gw.inference.steps import SyntheticPhaseFactor, SyntheticPhasePsiFactor
from dingo.gw.likelihood import AntiglitchGWLikelihood

THETA = dict(
    mass_1=45.0,
    mass_2=33.0,
    a_1=0.2,
    a_2=0.1,
    tilt_1=0.0,
    tilt_2=0.0,
    phi_12=0.0,
    phi_jl=0.0,
    luminosity_distance=800.0,
    theta_jn=1.0,
    geocent_time=0.02,
    phase=0.3,
    ra=1.3,
    dec=-0.4,
    psi=2.0,
    H1_glitch_amp=300.0,
    H1_glitch_phi=0.7,
    H1_glitch_f0=40.0,
    H1_glitch_gamma=3.0,
    H1_glitch_time=0.02,
)


@pytest.fixture(params=[False, True], ids=["uniform", "multibanded"])
def setup(request):
    base = UniformFrequencyDomain(20.0, 128.0, 0.5)
    domain = (
        MultibandedFrequencyDomain(
            nodes=[20.0, 40.0, 80.0, 128.0],
            delta_f_initial=0.5,
            base_domain=base.domain_dict,
        )
        if request.param
        else base
    )
    rng = np.random.default_rng(0)
    event = {
        "waveform": {
            ifo: base.update_data(
                (rng.normal(size=len(base)) + 1j * rng.normal(size=len(base))) * 1e-23
            )
            for ifo in ["H1", "L1"]
        },
        "asds": {
            ifo: np.where(base.frequency_mask, 1e-23, 1.0) for ifo in ["H1", "L1"]
        },
    }
    metadata = {
        "dataset_settings": {
            "domain": domain.domain_dict,
            "waveform_generator": {
                "approximant": "IMRPhenomD",
                "f_ref": 20.0,
                "spin_conversion_phase": 0.0,
                "use_dft_phase_decomposition": True,
                "mode_list": [(2, 2)],
            },
            "intrinsic_prior": {},
        },
        "train_settings": {
            "data": {
                "ref_time": 1126259462.4,
                "detectors": ["H1", "L1"],
                "inference_parameters": list(THETA),
                "extrinsic_prior": {
                    k: float(v) for k, v in THETA.items() if "_glitch_" in k
                },
            }
        },
    }
    context = GWSamplerContext.from_model_metadata(metadata, event)
    return context


def test_grid_and_22_terms_match_direct(setup):
    likelihood = setup.likelihood()
    assert isinstance(likelihood, AntiglitchGWLikelihood)
    phases, psis = np.array([0.1, 2.1]), np.array([0.2, 1.7])
    terms = likelihood.phase_grid_terms(THETA)
    terms22 = likelihood.terms_22_at_psi(likelihood.phase_psi_terms_22(THETA), psis)
    actual = likelihood.log_likelihood_from_phase_grid_terms(terms, phases, psis)
    expected = [
        [
            likelihood.log_likelihood({**THETA, "phase": phase, "psi": psi})
            for psi in psis
        ]
        for phase in phases
    ]
    np.testing.assert_allclose(actual, expected, atol=1e-8)
    np.testing.assert_allclose(
        likelihood.log_likelihood_22_from_terms(terms22, phases).T, expected, atol=1e-8
    )
    single = likelihood.phase_grid_terms_22(THETA)
    np.testing.assert_allclose(
        likelihood.log_likelihood_22_from_terms(single, phases),
        [likelihood.log_likelihood({**THETA, "phase": p}) for p in phases],
        atol=1e-8,
    )


@pytest.mark.parametrize("joint", [False, True])
@pytest.mark.parametrize("approx22", [False, True])
def test_synthetic_phase_cache_matches_direct_for_each_glitch(setup, joint, approx22):
    excluded = ["phase", "psi"] if joint else ["phase"]
    conditioning = [k for k in THETA if k not in excluded]
    theta = pd.DataFrame([{**THETA, "H1_glitch_amp": a} for a in [100.0, 300.0, 500.0]])
    given = {k: torch.tensor(theta[k].to_numpy()) for k in conditioning}
    cls = SyntheticPhasePsiFactor if joint else SyntheticPhaseFactor
    kwargs = {"n_grid_phase": 16, "n_grid_psi": 8} if joint else {"n_grid_phase": 16}
    factor = cls(
        conditioning=conditioning,
        approximation_22_mode=approx22,
        cache_log_likelihood=True,
        **kwargs
    )
    factor.max_grid_elements = 128
    # The factor API returns newly produced columns and their conditional log probability.
    out, log_prob = factor.sample_and_log_prob(1, setup, given=given)
    assert torch.isfinite(log_prob).all()
    for i in range(len(theta)):
        row = theta.iloc[i].to_dict()
        row.update(
            {k: float(v[i]) for k, v in out.items() if k != "log_likelihood_cache"}
        )
        assert float(out["log_likelihood_cache"][i]) == pytest.approx(
            setup.likelihood().log_likelihood(row), abs=1e-7
        )


def test_injection_matches_likelihood_signal(setup):
    likelihood = setup.likelihood(use_base_domain=True)
    injection = Injection.from_posterior_model_metadata(setup.model_metadata)
    injection.use_base_domain = True
    injection.asd = setup.event_data["asds"]
    physical = injection.signal(THETA)["waveform"]
    whitened = likelihood.signal(THETA)["waveform"]
    for ifo in physical:
        mask = likelihood.frequency_masks[ifo]
        np.testing.assert_allclose(
            (physical[ifo] / injection.asd[ifo] / injection.data_domain.noise_std)[
                mask
            ],
            whitened[ifo][mask],
            atol=1e-9,
        )


def test_phase_psi_marginalization_with_glitch(setup):
    """Both angle integrals agree with direct GW-plus-glitch likelihood calls."""
    from bilby.gw.utils import ln_i0
    from scipy.special import logsumexp

    # Inject on the uniform base grid before the likelihood decimates the data.
    injection = Injection.from_posterior_model_metadata(setup.model_metadata)
    injection.use_base_domain = True
    injection.asd = setup.event_data["asds"]
    signal = injection.signal(THETA)["waveform"]
    event = {
        "waveform": {
            ifo: noise + signal[ifo]
            for ifo, noise in setup.event_data["waveform"].items()
        },
        "asds": setup.event_data["asds"],
    }
    context = GWSamplerContext.from_model_metadata(setup.model_metadata, event)
    likelihood = context.likelihood()
    assert isinstance(likelihood, AntiglitchGWLikelihood)
    assert likelihood.waveform_generator.phase_is_global_factor
    assert np.linalg.norm(likelihood._glitch_signal(THETA)["H1"]) > 0

    phases = np.linspace(0.0, 2 * np.pi, 512, endpoint=False)
    psis = np.linspace(0.0, np.pi, 128, endpoint=False)
    # This reference bypasses all phase-grid helpers and keeps the glitch fixed.
    direct = np.array(
        [
            [
                likelihood.log_likelihood({**THETA, "phase": phase, "psi": psi})
                for psi in psis
            ]
            for phase in phases
        ]
    )
    expected_phase_marginal = logsumexp(direct, axis=0) - np.log(len(phases))
    expected_joint_marginal = logsumexp(direct) - np.log(direct.size)

    terms22 = likelihood.terms_22_at_psi(likelihood.phase_psi_terms_22(THETA), psis)
    # Preserve the absolute offset: it depends on the sampled glitch parameters.
    phase_marginal = (
        terms22["log_Zn"]
        + ln_i0(np.abs(terms22["d_inner_h"]))
        - terms22["h_inner_h"] / 2
    )
    np.testing.assert_allclose(
        phase_marginal, expected_phase_marginal, rtol=0, atol=1e-7
    )
    np.testing.assert_allclose(
        logsumexp(phase_marginal) - np.log(len(psis)),
        expected_joint_marginal,
        rtol=0,
        atol=1e-7,
    )

    mode_grid = likelihood.log_likelihood_from_phase_grid_terms(
        likelihood.phase_grid_terms(THETA), phases, psis
    )
    np.testing.assert_allclose(
        logsumexp(mode_grid) - np.log(mode_grid.size),
        expected_joint_marginal,
        rtol=0,
        atol=1e-7,
    )


def test_phase_marginalization_rotates_only_gw(setup):
    from scipy.special import logsumexp

    setup.model_metadata["train_settings"]["data"]["inference_parameters"].remove(
        "phase"
    )
    setup.model_metadata["dataset_settings"]["intrinsic_prior"]["phase"] = "default"
    direct = setup.likelihood()
    phases = np.linspace(0.0, 2 * np.pi, 4096, endpoint=False)
    expected = logsumexp(
        direct.log_likelihood_22_from_terms(direct.phase_grid_terms_22(THETA), phases)
    ) - np.log(len(phases))
    marginalized = setup.likelihood(
        phase_marginalization_kwargs={"approximation_22_mode": True}
    )
    assert marginalized.log_likelihood(THETA.copy()) == pytest.approx(
        expected, abs=1e-8
    )


@pytest.mark.parametrize("notch_asd", [1.0, np.inf])
def test_glitch_respects_base_grid_notches_before_decimation(setup, notch_asd):
    from dingo.gw.transforms import (
        AddAntiglitch,
        DecimateWaveformsAndASDS,
        WhitenAndScaleStrain,
    )

    base = getattr(setup.domain, "base_domain", setup.domain)
    notches = (base.sample_frequencies > 39) & (base.sample_frequencies < 45)
    # Inject into physical strain before save_hdf5 applies the PSD notch.
    raw = AddAntiglitch(base, colour=True)(
        {
            "parameters": THETA,
            "waveform": {ifo: np.zeros(len(base), complex) for ifo in ["H1", "L1"]},
            "asds": setup.event_data["asds"],
        }
    )
    setup.event_data["asds"]["H1"][notches] = notch_asd
    if isinstance(setup.domain, MultibandedFrequencyDomain):
        raw = DecimateWaveformsAndASDS(setup.domain, decimation_mode="whitened")(raw)
    expected = WhitenAndScaleStrain(setup.domain.noise_std)(raw)["waveform"]
    actual = setup.likelihood()._glitch_signal(THETA)
    for ifo in actual:
        np.testing.assert_allclose(actual[ifo], expected[ifo], atol=1e-12)


@pytest.mark.parametrize(
    "f_min, f_max, delta_f",
    [(30.0, 64.0, 0.5), (10.0, 256.0, 0.5), (20.0, 128.0, 0.25), (10.0, 256.0, 1.0)],
)
def test_glitch_amplitude_uses_training_grid(setup, f_min, f_max, delta_f):
    """Changing the event grid preserves the physical signal at shared frequencies."""
    original = setup.likelihood(use_base_domain=True)
    grid = UniformFrequencyDomain(f_min, f_max, delta_f)
    event = {
        "waveform": {ifo: np.zeros(len(grid), complex) for ifo in ["H1", "L1"]},
        "asds": {
            ifo: np.where(grid.frequency_mask, 1e-23, 1.0) for ifo in ["H1", "L1"]
        },
    }
    context = GWSamplerContext.from_model_metadata(
        setup.model_metadata, event, event_metadata={"domain": grid.domain_dict}
    )
    updated = context.likelihood(use_base_domain=True)
    _, old_idx, new_idx = np.intersect1d(
        original.data_domain.sample_frequencies,
        grid.sample_frequencies,
        return_indices=True,
    )
    valid = (
        original.frequency_masks["H1"][old_idx] & updated.frequency_masks["H1"][new_idx]
    )
    old_idx, new_idx = old_idx[valid], new_idx[valid]
    old_signal = original._glitch_signal(THETA)["H1"]
    new_signal = updated._glitch_signal(THETA)["H1"]
    np.testing.assert_allclose(
        new_signal[new_idx] * grid.noise_std,
        old_signal[old_idx] * original.data_domain.noise_std,
        rtol=1e-12,
        atol=1e-12,
    )
