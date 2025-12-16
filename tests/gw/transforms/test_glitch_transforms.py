"""Collect util functions to make sure the glitches make sense."""
import numpy as np
from matplotlib import pyplot as plt
import torchvision

from bilby.gw.detector import InterferometerList
from bilby.core.prior import Gaussian, DeltaFunction

from dingo.gw.transforms import (
    SampleExtrinsicParameters,
    GetDetectorTimes,
    ProjectOntoDetectors,
    SampleNoiseASD,
    WhitenAndScaleStrain,
    AddWhiteNoiseComplex,
    SelectStandardizeRepackageParameters,
    RepackageStrainsAndASDS,
    UnpackDict,
    AddAntiglitch,
)
from dingo.gw.prior import default_extrinsic_dict
from dingo.gw.domains import UniformFrequencyDomain
from dingo.gw.noise.asd_dataset import ASDDataset

from test_batch_transforms import input_sample_batched


STD_DICT = {
    "mean": {
        "chirp_mass": 10.0,
        "mass_ratio": 0.5,
    },
    "std": {
        "chirp_mass": 1.0,
        "mass_ratio": 0.1,
    },
}
PEAK_TIME = 2


def asd_dataset(domain):
    asd = np.random.rand(5, len(domain))
    asds = {"H1": asd, "L1": asd}
    settings = {"domain_dict": domain.domain_dict}
    dictionary = {"asds": asds, "settings": settings}

    return ASDDataset(dictionary=dictionary)


def default_glitch_extrinsic_dict():
    """Default parameters for O3 H1 Tomte glitches, for simplicity."""
    glitch_dict = {
        "H1_glitch_amp": Gaussian(name="glitch_amp", mu=322, sigma=50),
        "H1_glitch_phi": Gaussian(name="glitch_phi", mu=-2.77, sigma=0.49),
        "H1_glitch_f0": Gaussian(name="glitch_f0", mu=33.8, sigma=1.3),
        "H1_glitch_gamma": Gaussian(name="glitch_gamma", mu=3.17, sigma=0.64),
        "H1_glitch_time": DeltaFunction(name="glitch_time", peak=PEAK_TIME)
    }
    return default_extrinsic_dict | glitch_dict


def get_base_transforms(domain):
    ifo_list = InterferometerList(["H1", "L1"])
    ref_time = 1126259462.391
    transforms_pre = [
        SampleExtrinsicParameters(default_glitch_extrinsic_dict()),
        GetDetectorTimes(ifo_list, ref_time),
        ProjectOntoDetectors(ifo_list, domain, ref_time),
        SampleNoiseASD(asd_dataset(domain)),
        WhitenAndScaleStrain(domain.noise_std),
        AddWhiteNoiseComplex(),
    ]
    transforms_post = [
        SelectStandardizeRepackageParameters(
            {"inference_parameters": ["chirp_mass", "mass_ratio"]}, STD_DICT
        ),
        RepackageStrainsAndASDS(
            [ifo.name for ifo in ifo_list], first_index=domain.min_idx
        ),
        UnpackDict(["inference_parameters", "waveform"]),
    ]
    return transforms_pre, transforms_post


def transform_list(domain):
    transforms_pre, transforms_post = get_base_transforms(domain)
    return transforms_pre + transforms_post


def transform_list_glitch(domain):
    transforms_pre, transforms_post = get_base_transforms(domain)
    return transforms_pre + [AddAntiglitch(domain)] + transforms_post


def visualise_glitch_single():
    """Inject glitches in random waveforms and plot them in both domains."""
    frequency_domain = UniformFrequencyDomain(30, 300, 0.25)
    input_sample = input_sample_batched(1, frequency_domain)
    transforms = torchvision.transforms.Compose(
        transform_list(frequency_domain)
    )
    output = transforms(input_sample)
    transforms_glitch = torchvision.transforms.Compose(
        transform_list_glitch(frequency_domain)
    )
    output_glitch = transforms_glitch(input_sample)

    # Debug plots
    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1])

    ax0 = fig.add_subplot(gs[0])
    ax0.loglog(frequency_domain.sample_frequencies[frequency_domain.frequency_mask],
            np.sqrt(output[1][0, 0, 0, :]**2 + output[1][0, 0, 1, :]**2),
            label="Default", alpha=.7)
    ax0.loglog(frequency_domain.sample_frequencies[frequency_domain.frequency_mask],
            np.sqrt(output_glitch[1][0, 0, 0, :]**2 + output_glitch[1][0, 0, 1, :]**2),
            label="Glitch", alpha=.7)
    ax0.legend()

    ax1 = fig.add_subplot(gs[1])
    to_fft = np.zeros(len(frequency_domain.frequency_mask) + 1, dtype=np.complex128)

    for data in output, output_glitch:
        to_fft[1:][frequency_domain.frequency_mask]\
            = (data[1][0, 0, 0, :] + 1j * data[1][0, 0, 1, :]) / (1e23 * data[1][0, 0, 2, :])
        waveform = np.fft.irfft(to_fft)
        t = np.linspace(0, frequency_domain.duration, len(waveform))
        ax1.plot(t, waveform, alpha=.7)

    ax1.set_xlabel("Time (s)")
    ax1.set_xlim(PEAK_TIME - 0.15, PEAK_TIME + 0.15)
    plt.savefig("glitch.png")


visualise_glitch_single()
