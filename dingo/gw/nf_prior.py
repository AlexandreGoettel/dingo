"""Joint priors from unconditional Dingo normalising-flow checkpoints."""

import numpy as np
import torch

from dingo.core.posterior_models import NormalizingFlowPosteriorModel


class NFPrior:
    """A flow in physical units, optionally truncated to rectangular bounds.

    Bounds use rejection sampling. Their retained probability is estimated once
    with a fixed, isolated RNG, so density evaluations remain reproducible. The
    relative Monte Carlo error is at most about 0.0032 at 50% acceptance.
    """

    NORMALIZATION_SAMPLES = 100_000
    BATCH_SIZE = 4096
    MIN_SAMPLING_EFFICIENCY = 0.5

    def __init__(
        self, model_filename, parameters, weight=1.0, bounds=None, device="cpu"
    ):
        if "geocent_time" in parameters:
            raise ValueError(
                "NF priors for geocent_time are not supported; use an independent time prior."
            )
        self.weight = float(weight)
        if not np.isfinite(self.weight) or self.weight < 0:
            raise ValueError("NF weights must be finite and nonnegative.")
        self.model_filename = model_filename
        self.flow = NormalizingFlowPosteriorModel(
            model_filename=model_filename, device=device, load_training_info=False
        )
        self.flow.network.eval()
        data = self.flow.metadata["train_settings"]["data"]
        self.parameters = list(data["parameters"])
        if len(set(parameters)) != len(parameters) or set(parameters) != set(
            self.parameters
        ):
            raise ValueError("NF parameters must match the checkpoint's parameter set.")
        self.mean = np.array(
            [data["standardization"]["mean"][p] for p in self.parameters]
        )
        self.std = np.array(
            [data["standardization"]["std"][p] for p in self.parameters]
        )
        if not np.all(np.isfinite(self.mean)) or not np.all(
            np.isfinite(self.std) & (self.std > 0)
        ):
            raise ValueError(
                "NF standardization requires finite means and positive finite scales."
            )
        bounds = bounds or {}
        if set(bounds) - set(self.parameters):
            raise ValueError("NF bounds contain parameters absent from the checkpoint.")
        self.lower = np.array(
            [
                float(bounds.get(p, {}).get("min", -np.inf))
                if bounds.get(p, {}).get("min") is not None
                else -np.inf
                for p in self.parameters
            ]
        )
        self.upper = np.array(
            [
                float(bounds.get(p, {}).get("max", np.inf))
                if bounds.get(p, {}).get("max") is not None
                else np.inf
                for p in self.parameters
            ]
        )
        if not np.all(self.lower < self.upper):
            raise ValueError("NF bounds must have minimum < maximum.")
        self._log_bounds_mass = None

    def _valid(self, values):
        """Accept finite physical draws strictly inside the configured bounds."""
        return np.all(
            np.isfinite(values) & (values > self.lower) & (values < self.upper), axis=-1
        )

    def _draw(self, count):
        """Draw a batch in physical units without recording gradients."""
        with torch.no_grad():
            return (
                self.flow.sample(num_samples=count).detach().cpu().numpy() * self.std
                + self.mean
            )

    def sample(self, count):
        """Reject invalid draws while keeping all coordinates from each joint draw."""
        samples = np.empty((count, len(self.parameters)))
        accepted = drawn = 0
        while accepted < count:
            values = self._draw(min(count - accepted, self.BATCH_SIZE))
            valid = values[self._valid(values)]
            samples[accepted : accepted + len(valid)] = valid
            accepted += len(valid)
            drawn += len(values)
            if drawn >= 1000 and accepted / drawn < self.MIN_SAMPLING_EFFICIENCY:
                raise ValueError(
                    f"NF prior {self.model_filename} acceptance is below 50%; check its bounds/model."
                )
        return samples

    def _bounds_log_mass(self):
        """Estimate truncation normalization without advancing the caller's RNG."""
        if self._log_bounds_mass is None:
            if np.all(np.isneginf(self.lower) & np.isposinf(self.upper)):
                self._log_bounds_mass = 0.0
            else:
                device = torch.device(self.flow.device)
                devices = (
                    [
                        device.index
                        if device.index is not None
                        else torch.cuda.current_device()
                    ]
                    if device.type == "cuda"
                    else []
                )
                accepted = 0
                with torch.random.fork_rng(devices=devices):
                    torch.random.default_generator.manual_seed(150914)
                    if devices:
                        with torch.cuda.device(devices[0]):
                            torch.cuda.manual_seed(150914)
                    for start in range(0, self.NORMALIZATION_SAMPLES, self.BATCH_SIZE):
                        accepted += self._valid(
                            self._draw(
                                min(self.BATCH_SIZE, self.NORMALIZATION_SAMPLES - start)
                            )
                        ).sum()
                probability = accepted / self.NORMALIZATION_SAMPLES
                if probability < self.MIN_SAMPLING_EFFICIENCY:
                    raise ValueError(
                        f"NF prior {self.model_filename} bounds acceptance is below 50%."
                    )
                self._log_bounds_mass = float(np.log(probability))
        return self._log_bounds_mass

    def ln_prob(self, sample):
        """Joint log density with inverse-standardization Jacobian and truncation."""
        values = np.stack(
            np.broadcast_arrays(
                *[np.asarray(sample[p], dtype=float) for p in self.parameters]
            ),
            axis=-1,
        )
        rows = values.reshape(-1, len(self.parameters))
        indices = np.flatnonzero(self._valid(rows))
        log_prob = np.full(len(rows), -np.inf)
        if len(indices):
            normalization = self._bounds_log_mass()
            with torch.no_grad():
                for start in range(0, len(indices), self.BATCH_SIZE):
                    selected = indices[start : start + self.BATCH_SIZE]
                    inputs = torch.as_tensor(
                        (rows[selected] - self.mean) / self.std,
                        device=self.flow.device,
                        dtype=next(self.flow.network.parameters()).dtype,
                    )
                    log_prob[selected] = (
                        self.flow.log_prob(inputs).detach().cpu().numpy().reshape(-1)
                        - np.log(self.std).sum()
                        - normalization
                    )
        return log_prob.reshape(values.shape[:-1])
