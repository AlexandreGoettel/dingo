"""Train a denoising ResNet autoencoder on gravitational-wave detector strain."""

import argparse
import csv
import os
import time
from copy import deepcopy

import torch
import yaml
from threadpoolctl import threadpool_limits
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dingo.core.nn.autoencoder import DenseResidualAutoencoder
from dingo.core.utils import build_train_and_test_loaders
from dingo.core.utils.torchutils import (
    get_lr,
    get_optimizer_from_kwargs,
    get_scheduler_from_kwargs,
    perform_scheduler_step,
    print_number_of_model_parameters,
    set_float32_matmul_precision,
)
from dingo.core.utils.trainutils import LossInfo
from dingo.gw.dataset import WaveformDataset
from dingo.gw.training.train_builders import build_dataset, set_train_transforms
from dingo.gw.training.train_pipeline import copy_files_to_local
from dingo.gw.transforms import DenoisingTransforms


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a denoising autoencoder on gravitational-wave data."
    )
    parser.add_argument(
        "--settings_file",
        required=True,
        help="YAML file containing data, model, training, and local settings.",
    )
    parser.add_argument(
        "--train_dir", required=True, help="Directory for Dingo training output."
    )
    return parser.parse_args()


def prepare_training_data(
    train_settings: dict, local_settings: dict
) -> WaveformDataset:
    """Load waveforms using the same settings and optional cache as dingo_train."""
    data_settings = deepcopy(train_settings["data"])
    leave_waveforms_on_disk = local_settings.get("leave_waveforms_on_disk", True)
    data_settings["waveform_dataset_path"] = copy_files_to_local(
        file_path=data_settings["waveform_dataset_path"],
        local_dir=local_settings.get("local_cache_path"),
        leave_keys_on_disk=leave_waveforms_on_disk,
    )
    return build_dataset(
        data_settings=data_settings,
        leave_waveforms_on_disk=leave_waveforms_on_disk,
    )


def run_epoch(model, loader, device, optimizer=None, *, epoch: int = 1) -> float:
    """Return sample-averaged strain MSE, optionally updating model weights."""
    training = optimizer is not None
    model.train(training)
    loss_info = LossInfo(
        epoch=epoch,
        len_dataset=len(loader.dataset),
        batch_size_per_grad_update=loader.batch_size,
        mode="Train" if training else "Test",
        print_freq=1,
        device=device,
    )
    with torch.set_grad_enabled(training):
        for batch_idx, (inputs, targets) in enumerate(loader):
            loss_info.update_timer("Dataloader")
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            reconstruction = model(inputs)
            # The third output channel corresponds to ASD and has no loss.
            loss = F.mse_loss(reconstruction[..., :2, :], targets)
            if training:
                loss.backward()
                optimizer.step()
            loss_info.cache_loss(loss, len(inputs))
            loss_info.update()
            loss_info.print_info(batch_idx)
    return loss_info.get_avg()


def train(
    model: DenseResidualAutoencoder,
    train_loader: DataLoader,
    test_loader: DataLoader,
    train_settings: dict,
    train_dir: str,
    local_settings: dict,
    *,
    dataset_settings: dict,
) -> None:
    """Record per-epoch MSE and save the model whenever validation MSE improves."""
    training_settings = train_settings["training"]
    epochs = training_settings["epochs"]
    if not isinstance(epochs, int) or epochs < 1:
        raise ValueError("training.epochs must be a positive integer.")
    if len(train_loader) == 0 or len(test_loader) == 0:
        raise ValueError("Training and validation loaders must both contain batches.")
    device = torch.device(local_settings.get("device", "cpu"))
    print(f"Putting autoencoder to device {device}.")
    model.to(device)
    print_number_of_model_parameters(model)
    print("Initializing new optimizer and scheduler.")
    optimizer = get_optimizer_from_kwargs(
        model.parameters(), **training_settings["optimizer"]
    )
    if isinstance(optimizer, torch.optim.LBFGS):
        raise ValueError("LBFGS is not supported by compression training.")
    scheduler_settings = training_settings.get("scheduler")
    scheduler = (
        get_scheduler_from_kwargs(optimizer, **scheduler_settings)
        if scheduler_settings
        else None
    )

    os.makedirs(train_dir, exist_ok=True)
    best_validation_loss = float("inf")
    with open(os.path.join(train_dir, "history.txt"), "w", newline="") as f:
        writer = csv.writer(f)
        for epoch in range(1, epochs + 1):
            print(f"\nStart training epoch {epoch} with lr {get_lr(optimizer)}")
            time_start = time.time()
            train_loss = run_epoch(
                model, train_loader, device, optimizer, epoch=epoch
            )
            print(
                "Done. This took {:2.0f}:{:2.0f} min.".format(
                    *divmod(time.time() - time_start, 60)
                )
            )
            print(f"Start testing epoch {epoch}")
            time_start = time.time()
            validation_loss = run_epoch(model, test_loader, device, epoch=epoch)
            print(
                "Done. This took {:2.0f}:{:2.0f} min.".format(
                    *divmod(time.time() - time_start, 60)
                )
            )
            if scheduler is not None:
                perform_scheduler_step(scheduler, validation_loss)
            writer.writerow([train_loss, validation_loss])
            f.flush()
            if validation_loss < best_validation_loss:
                torch.save(
                    {
                        "model_kwargs": train_settings["model"],
                        "model_state_dict": model.state_dict(),
                        "train_settings": train_settings,
                        "dataset_settings": dataset_settings,
                        "epoch": epoch,
                        "validation_loss": validation_loss,
                    },
                    os.path.join(train_dir, "model_latest.pt"),
                )
                best_validation_loss = validation_loss
            print(
                f"Epoch {epoch}/{epochs}: training MSE = {train_loss:.6g}, "
                f"validation MSE = {validation_loss:.6g}"
            )
            print(f"Finished training epoch {epoch}.\n")


def run_training(
    train_settings: dict, local_settings: dict, train_dir: str
) -> None:
    """Build the autoencoder and train on matched noisy/clean GW examples.

    Data settings follow dingo_train's non-tokenized ResNet path. The model
    section contains DenseResidualAutoencoder kwargs (input_dims is inferred).
    Training directly contains asd_dataset_path, batch_size, epochs, optimizer,
    and an optional scheduler, using dingo_train's optimizer/scheduler format.
    """
    if "tokenization" in train_settings["data"]:
        raise ValueError("Compression ResNet training does not support tokenization.")
    set_float32_matmul_precision(local_settings)
    wfd = prepare_training_data(train_settings, local_settings)
    training_settings = train_settings["training"]
    with threadpool_limits(limits=1, user_api="blas"):
        print("\nPreparing compression training.")
        set_train_transforms(
            wfd,
            train_settings["data"],
            training_settings["asd_dataset_path"],
        )
        wfd.transform = DenoisingTransforms(wfd.transform.transforms)
        train_fraction = train_settings["data"]["train_fraction"]
        train_size = int(train_fraction * len(wfd))
        if not 0 < train_size < len(wfd):
            raise ValueError("train_fraction must produce nonempty train/test splits.")

        model_kwargs = deepcopy(train_settings["model"])
        model_kwargs["input_dims"] = list(wfd[0][0].shape)
        print("\nInitializing new autoencoder. Model settings:")
        print(yaml.dump(model_kwargs, default_flow_style=False, sort_keys=False))
        model = DenseResidualAutoencoder(**model_kwargs)
        train_settings["model"] = model_kwargs
        batch_size = training_settings["batch_size"]
        drop_last = bool(local_settings.get("torch_compile", False))
        train_loader, test_loader, _ = build_train_and_test_loaders(
            dataset=wfd,
            train_fraction=train_fraction,
            batch_size=batch_size,
            num_workers=local_settings.get("num_workers", 0),
            drop_last=drop_last,
        )
        print(
            f"Prepared {len(train_loader.dataset)} training and "
            f"{len(test_loader.dataset)} validation samples."
        )
        train(
            model,
            train_loader,
            test_loader,
            train_settings,
            train_dir,
            local_settings,
            dataset_settings=wfd.settings,
        )


def train_local():
    args = parse_args()
    print("Beginning new compression training run.")
    with open(args.settings_file, "r") as f:
        train_settings = yaml.safe_load(f)
    local_settings = train_settings.pop("local")

    os.makedirs(args.train_dir, exist_ok=True)
    with open(os.path.join(args.train_dir, "local_settings.yaml"), "w") as f:
        yaml.dump(local_settings, f, default_flow_style=False, sort_keys=False)

    run_training(train_settings, local_settings, args.train_dir)
    print("Compression training complete. Best model: model_latest.pt; losses: history.txt.")


if __name__ == "__main__":
    train_local()
