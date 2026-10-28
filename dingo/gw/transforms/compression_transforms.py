"""Pair Dingo's noisy detector inputs with matching noiseless strain targets."""

from copy import deepcopy

import numpy as np

from .general_transforms import UnpackDict
from .noise_transforms import AddWhiteNoiseComplex, RepackageStrainsAndASDS
from .waveform_transforms import CropMaskStrainRandom


class DenoisingTransforms:
    """Wrap the standard, non-tokenized training transforms.

    Random transforms run once. Save whitened strain just before noise addition,
    then package it using the same ASDs and frequency range as the network input.
    Temporarily append the two clean strain channels to the three input channels
    so that any frequency crop acts on both with exactly the same random mask.

    Returns [input, target], with shapes (..., detectors, 3, bins) and
    (..., detectors, 2, bins), respectively. The input includes inverse ASD;
    the target contains only real and imaginary noiseless strain.

    With keep_parameters=True, return [parameters, input, target].
    """

    def __init__(self, transforms, keep_parameters=False):
        self.transforms = list(transforms)
        self.keep_parameters = keep_parameters
        # After repackaging, only cropping and unpacking support the temporary
        # five-channel representation. Reject other layouts rather than silently
        # changing the network inputs or selecting the wrong target channels.
        repack_index = next(
            i
            for i, t in enumerate(self.transforms)
            if isinstance(t, RepackageStrainsAndASDS)
        )
        if not isinstance(self.transforms[-1], UnpackDict) or any(
            not isinstance(t, CropMaskStrainRandom)
            for t in self.transforms[repack_index + 1 : -1]
        ):
            raise ValueError(
                "Compression training requires non-tokenized detector strain; "
                "only frequency cropping is supported after repackaging."
            )

    def __call__(self, sample):
        clean_strain = None
        for transform in self.transforms:
            if isinstance(transform, AddWhiteNoiseComplex):
                clean_strain = deepcopy(sample["waveform"])
            elif isinstance(transform, RepackageStrainsAndASDS):
                # With zero_noise enabled, the current waveform is already clean.
                clean_sample = sample.copy()
                if clean_strain is not None:
                    clean_sample["waveform"] = clean_strain
                target = transform(clean_sample)["waveform"][..., :2, :]
                sample = transform(sample)
                sample["waveform"] = np.concatenate(
                    (sample["waveform"], target), axis=-2
                )
                continue
            elif isinstance(transform, UnpackDict):
                waveform = sample["waveform"]
                inputs, target = waveform[..., :3, :], waveform[..., 3:, :]
                if self.keep_parameters:
                    sample = sample.copy()
                    sample["waveform"] = inputs
                    parameters, inputs = transform(sample)
                    return [parameters, inputs, target]
                return [inputs, target]
            sample = transform(sample)
