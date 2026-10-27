"""Train a flow conditioned directly on a jointly trained autoencoder's latent.

Runs once from scratch, recording total and component losses for each epoch.
Set model.embedding_type to compression, model.embedding_kwargs to
DenseResidualAutoencoder settings, and model.posterior_kwargs to the usual NSF
settings. Dimensions are inferred.
training.lambda weights the clean-strain MSE (default 1.0):

    loss = -mean(log q(theta | encoder(strain))) + lambda * MSE

No SVD projection or initialization is used.

Set model.embedding_kwargs.condition_on_parameters: true to enable two passes.
The data-only encoder initializes parameters using one fixed flow base draw,
saved with the network (embedding_kwargs.initialization_seed defaults to 0).
The second encoder conditions on those estimates. training.beta (default 1.0)
weights latent MSE, which updates only the data-only encoder; its target is the
detached conditional latent. True parameters are only the SBI targets.
"""

import argparse
import csv
import os
import subprocess

import torch
import yaml
from threadpoolctl import threadpool_limits

from dingo.core.posterior_models.build_model import (
    autocomplete_model_kwargs,
    build_model_from_kwargs,
)
from dingo.core.utils import build_train_and_test_loaders
from dingo.core.utils.trainutils import LossInfo
from dingo.core.utils.torchutils import (
    perform_scheduler_step,
    print_number_of_model_parameters,
    set_float32_matmul_precision,
)
from dingo.gw.training.train_builders import set_train_transforms
from dingo.gw.training.train_pipeline_compression import prepare_training_data
from dingo.gw.transforms import DenoisingTransforms


def prepare_training(train_settings, local_settings):
    """Build a new posterior model and matching noisy/clean waveform pairs."""
    model_settings = train_settings["model"]
    model_type = model_settings.setdefault("posterior_model_type", "normalizing_flow")
    if model_type.lower() != "normalizing_flow":
        raise ValueError(
            "dingo_train_sbicomp requires posterior_model_type: normalizing_flow."
        )
    embedding_type = model_settings.setdefault("embedding_type", "compression")
    if embedding_type.lower() != "compression":
        raise ValueError("dingo_train_sbicomp requires embedding_type: compression.")
    if "tokenization" in train_settings["data"]:
        raise ValueError("SBI compression requires non-tokenized detector strain.")
    wfd = prepare_training_data(train_settings, local_settings)
    set_train_transforms(
        wfd, train_settings["data"], train_settings["training"]["asd_dataset_path"]
    )
    # Infer dimensions before adding reconstruction targets to the data samples.
    autocomplete_model_kwargs(train_settings["model"], wfd[0])
    pm = build_model_from_kwargs(
        settings={"dataset_settings": wfd.settings, "train_settings": train_settings},
        device=local_settings.get("device", "cpu"),
    )
    wfd.transform = DenoisingTransforms(wfd.transform.transforms, keep_parameters=True)
    return pm, wfd


def run_epoch(pm, loader, *, training):
    """Return sample-averaged (total, SBI, reconstruction, latent) losses."""
    pm.network.train(training)
    loss_info = LossInfo(
        epoch=pm.epoch,
        len_dataset=len(loader.dataset),
        batch_size_per_grad_update=loader.batch_size,
        mode="Train" if training else "Test",
        print_freq=1,
        device=pm.device,
    )
    component_sums = torch.zeros(3, device=pm.device, dtype=torch.float64)
    num_samples = 0
    with torch.set_grad_enabled(training):
        for batch_idx, data in enumerate(loader):
            loss_info.update_timer("Dataloader")
            data = [d.to(pm.device, non_blocking=True) for d in data]
            if training:
                pm.optimizer.zero_grad(set_to_none=True)
            loss, *components = pm.compression_loss_components(*data)
            if training:
                loss.backward()
                pm.optimizer.step()
            batch_size = len(data[0])
            component_sums += torch.stack(
                [component.detach() for component in components]
            ).to(dtype=torch.float64) * batch_size
            num_samples += batch_size
            loss_info.cache_loss(loss, batch_size)
            loss_info.update()
            loss_info.print_info(batch_idx)
    component_means = (component_sums / num_samples).tolist()
    return (loss_info.get_avg(), *component_means)


def run_training(train_settings, local_settings, train_dir):
    """Record losses and save the model whenever combined validation loss improves."""
    training = train_settings["training"]
    epochs = training["epochs"]
    if not isinstance(epochs, int) or epochs < 1:
        raise ValueError("training.epochs must be a positive integer.")
    set_float32_matmul_precision(local_settings)
    with threadpool_limits(limits=1, user_api="blas"):
        pm, wfd = prepare_training(train_settings, local_settings)
        train_loader, test_loader, _ = build_train_and_test_loaders(
            dataset=wfd,
            train_fraction=train_settings["data"]["train_fraction"],
            batch_size=training["batch_size"],
            num_workers=local_settings.get("num_workers", 0),
        )
        pm.optimizer_kwargs = training["optimizer"]
        pm.scheduler_kwargs = training.get("scheduler")
        pm.initialize_optimizer_and_scheduler()
        print_number_of_model_parameters(pm.network)

        os.makedirs(train_dir, exist_ok=True)
        best_validation_loss = float("inf")
        with open(os.path.join(train_dir, "history.txt"), "w", newline="") as f:
            # The comment header names columns and is compatible with numpy.loadtxt.
            header = "# train,valid,train_sbi,valid_sbi,train_compression,valid_compression"
            if pm.condition_on_parameters:
                header += ",train_latent,valid_latent"
            f.write(header + "\n")
            writer = csv.writer(f)
            num_losses = 4 if pm.condition_on_parameters else 3
            for pm.epoch in range(1, epochs + 1):
                train_losses = run_epoch(pm, train_loader, training=True)
                validation_losses = run_epoch(pm, test_loader, training=False)
                train_loss, validation_loss = train_losses[0], validation_losses[0]
                if pm.scheduler is not None:
                    perform_scheduler_step(pm.scheduler, validation_loss)
                # Keep each training loss next to its validation counterpart.
                row = []
                for pair in zip(
                    train_losses[:num_losses], validation_losses[:num_losses]
                ):
                    row.extend(pair)
                writer.writerow(row)
                f.flush()
                if validation_loss < best_validation_loss:
                    pm.save_model(
                        os.path.join(train_dir, "model_latest.pt"),
                        save_training_info=False,
                    )
                    best_validation_loss = validation_loss
                print(
                    f"Epoch {pm.epoch}/{epochs}: training loss = {train_loss:.6g}, "
                    f"validation loss = {validation_loss:.6g}"
                )


def train_local():
    parser = argparse.ArgumentParser(description="Train SBI compression from scratch.")
    parser.add_argument("--settings_file", required=True, help="Training settings YAML.")
    parser.add_argument("--train_dir", required=True, help="Training output directory.")
    parser.add_argument(
        "--exit_command",
        type=str,
        default="",
        help="Optional command to execute after completion of training.",
    )
    args = parser.parse_args()
    with open(args.settings_file, "r") as f:
        train_settings = yaml.safe_load(f)
    local_settings = train_settings.pop("local")
    run_training(train_settings, local_settings, args.train_dir)
    if args.exit_command:
        print(f"Training complete. Executing exit command: {args.exit_command}.")
        subprocess.run(args.exit_command, shell=True, check=False)


if __name__ == "__main__":
    train_local()
