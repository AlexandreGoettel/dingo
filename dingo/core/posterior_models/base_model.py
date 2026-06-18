"""
This module contains the abstract base class for representing posterior models,
as well as functions for training and testing across an epoch.
"""

from abc import abstractmethod, ABC
import os
from os.path import join
import json
from collections import OrderedDict
from typing import Optional
import time
from threadpoolctl import threadpool_limits
import numpy as np
import h5py

import torch

import dingo.core.utils as utils
import dingo.core.utils.trainutils
from dingo.core.utils.trainutils import EarlyStopping
from dingo.core.utils.backward_compatibility import update_model_config
from dingo.core.utils.misc import get_version


class BasePosteriorModel(ABC):
    """
    Abstract base class for PosteriorModels. This is intended to construct and hold a
    neural network for estimating the posterior density, as well as saving / loading,
    and training.

    Subclasses must implement methods for constructing the specific network, sampling,
    density evaluation, and computing the loss during training.
    """

    def __init__(
        self,
        model_filename: str = None,
        metadata: dict = None,
        initial_weights: dict = None,
        device: str = "cuda",
        load_training_info: bool = True,
    ):
        """
        Initialize a model for the posterior distribution.

        Parameters
        ----------
        model_filename: str
            If given, loads data from the given file.
        metadata: dict
            If given, initializes the model from these settings
        initial_weights: dict
            Initial weights for the model
        device: str
        load_training_info: bool
        """

        self.version = f"dingo={get_version()}"  # dingo version

        self.device = None
        self.optimizer_kwargs = None
        self.network_kwargs = None
        self.scheduler_kwargs = None
        self.initial_weights = initial_weights

        self.metadata = metadata
        if self.metadata is not None:
            self.model_kwargs = self.metadata["train_settings"]["model"]
            # Expect self.optimizer_settings and self.scheduler_settings to be set
            # separately, and before calling initialize_optimizer_and_scheduler().

        self.epoch = 0
        self.network = None
        self.optimizer = None
        self.scheduler = None
        self.context = None
        self.event_metadata = None

        # build model
        if model_filename is not None:
            self.load_model(
                model_filename, load_training_info=load_training_info, device=device
            )
        else:
            self.initialize_network()
            self.network_to_device(device)

    @abstractmethod
    def initialize_network(self):
        """
        Initialize the network backbone for the posterior model.
        """
        pass

    @abstractmethod
    def sample(self, *context: torch.Tensor, num_samples: int = 1):
        """
        Sample parameters theta from the posterior model,

        theta ~ p(theta | context)

        Parameters
        ----------
        context: torch.Tensor
            Context information (typically observed data). Should have a batch
            dimension (even if size B = 1).
        num_samples: int = 1
            Number of samples to generate.

        Returns
        -------
        samples: torch.Tensor
            Shape (B, num_samples, dim(theta))
        """
        pass

    @abstractmethod
    def sample_and_log_prob(self, *context: torch.Tensor, num_samples: int = 1):
        """
        Sample parameters theta from the posterior model,

        theta ~ p(theta | context)

        and also return the log_prob. For models such as normalizing flows, it is more
        economical to calculate the log_prob at the same time as sampling, rather than
        as a separate step.

        Parameters
        ----------
        context: torch.Tensor
            Context information (typically observed data). Should have a batch
            dimension (even if size B = 1).
        num_samples: int = 1
            Number of samples to generate.

        Returns
        -------
        samples, log_prob: torch.Tensor, torch.Tensor
            Shapes (B, num_samples, dim(theta)), (B, num_samples)
        """
        pass

    @abstractmethod
    def log_prob(self, theta: torch.Tensor, *context: torch.Tensor):
        """
        Evaluate the log posterior density,

        log p(theta | context)

        Parameters
        ----------
        theta: torch.Tensor
            Parameter values at which to evaluate the density. Should have a batch
            dimension (even if size B = 1).
        context: torch.Tensor
            Context information (typically observed data). Must have context.shape[0] = B.

        Returns
        -------
        log_prob: torch.Tensor
            Shape (B,)
        """
        pass

    @abstractmethod
    def loss(self, theta: torch.Tensor, *context: torch.Tensor):
        """
        Compute the loss for a batch of data.

        Parameters
        ----------
        theta: torch.Tensor
            Parameter values at which to evaluate the density. Should have a batch
            dimension (even if size B = 1).
        context: torch.Tensor
            Context information (typically observed data). Must have the same leading
            (batch) dimension as theta.

        Returns
        -------
        loss: torch.Tensor
            Mean loss across the batch (a scalar).
        """
        pass

    def network_to_device(self, device):
        """
        Put model to device, and set self.device accordingly.
        """
        device = str(device)
        if device != "cpu" and not device.startswith("cuda"):
            raise ValueError(f"Device should be either cpu or cuda, got {device}.")
        self.device = torch.device(device)
        # Commented below so that code runs on first cuda device in the case of multiple.
        # if device == 'cuda' and torch.cuda.device_count() > 1:
        #     print("Using", torch.cuda.device_count(), "GPUs.")
        #     raise NotImplementedError('This needs testing!')
        #     # dim = 0 [512, ...] -> [256, ...], [256, ...] on 2 GPUs
        #     self.network = torch.nn.DataParallel(self.network)
        print(f"Putting posterior model to device {self.device}.")
        self.network.to(self.device)

    def initialize_optimizer_and_scheduler(self):
        """
        Initializes the optimizer and scheduler with self.optimizer_kwargs
        and self.scheduler_kwargs, respectively.
        """
        if self.optimizer_kwargs is not None:
            self.optimizer = utils.get_optimizer_from_kwargs(
                self.network.parameters(), **self.optimizer_kwargs
            )
        if self.scheduler_kwargs is not None:
            self.scheduler = utils.get_scheduler_from_kwargs(
                self.optimizer, **self.scheduler_kwargs
            )

    def save_model(
        self,
        model_filename: str,
        save_training_info: bool = True,
    ):
        """
        Save the posterior model to the disk.

        Parameters
        ----------
        model_filename: str
            filename for saving the model
        save_training_info: bool
            specifies whether information required to proceed with training is
            saved, e.g. optimizer state dict

        """
        model_dict = {
            "model_kwargs": self.model_kwargs,
            "model_state_dict": self.network.state_dict(),
            "epoch": self.epoch,
            "version": self.version,
        }

        if self.metadata is not None:
            model_dict["metadata"] = self.metadata

        if self.context is not None:
            model_dict["context"] = self.context

        if self.event_metadata is not None:
            model_dict["event_metadata"] = self.event_metadata

        if save_training_info:
            model_dict["optimizer_kwargs"] = self.optimizer_kwargs
            model_dict["scheduler_kwargs"] = self.scheduler_kwargs
            if self.optimizer is not None:
                model_dict["optimizer_state_dict"] = self.optimizer.state_dict()
            if self.scheduler is not None:
                model_dict["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(model_dict, model_filename)

    def _load_model_from_hdf5(self, model_filename: str):
        """
        Helper function to load a trained model that has been
        saved in HDF5 format using `dingo_pt_to_hdf5`.
        Parameters
        ----------
        model_filename: str
            path to saved model; must have extension '.hdf5'
        Returns
        -------
        d: dict
            A stripped down version of the dict saved by torch.save()
            Specifically, it does not include 'optimizer_state_dict'
            to save space at inference time.
        """
        d = {}
        with h5py.File(model_filename, "r") as fp:
            model_basename = os.path.basename(model_filename)
            if fp.attrs["CANONICAL_FILE_BASENAME"] != model_basename:
                raise ValueError(
                    "HDF5 attribute CANONICAL_FILE_BASENAME differs from model name",
                    model_basename,
                )

            # Load small nested dicts from json
            for k, v in fp["serialized_dicts"].items():
                d[k] = json.loads(v[()])

            # Load model weights
            model_state_dict = OrderedDict()
            for k, v in fp["model_weights"].items():
                model_state_dict[k] = torch.from_numpy(np.array(v, dtype=np.float32))
            d["model_state_dict"] = model_state_dict

        return d

    def load_model(
        self,
        model_filename: str,
        load_training_info: bool = True,
        device: str = "cuda",
    ):
        """
        Load a posterior model from the disk.

        Parameters
        ----------
        model_filename: str
            path to saved model
        load_training_info: bool #TODO: load information for training
            specifies whether information required to proceed with training is
            loaded, e.g. optimizer state dict
        device: str
        """

        # Make sure that when the model is loaded, the torch tensors are put on the
        # device indicated in the saved metadata. External routines run on a cpu
        # machine may have moved the model from 'cuda' to 'cpu'.
        ext = os.path.splitext(model_filename)[-1]
        if ext == ".pt":
            d = torch.load(model_filename, map_location=device, weights_only=False)
        elif ext == ".hdf5":
            d = self._load_model_from_hdf5(model_filename)
        else:
            raise ValueError("Models should be ether in .pt or .hdf5 format.")

        self.version = d.get("version")

        self.model_kwargs = d["model_kwargs"]
        update_model_config(self.model_kwargs)  # For backward compatibility
        self.initialize_network()
        if any(("grow" in k for k in d["model_state_dict"])):
            # Since the grownet embedding is identical to the new embedding
            # use the "normal" embedding to init grow embedding
            # Note: network was initialised without grow net
            self.grow_model = self.network.embedding_net
            self.initialize_network()

        self.epoch = d["epoch"]

        self.metadata = d["metadata"]

        if "context" in d:
            self.context = d["context"]

        if "event_metadata" in d:
            self.event_metadata = d["event_metadata"]

        if device != "meta":
            self.network.load_state_dict(d["model_state_dict"])

            self.network_to_device(device)

            if load_training_info:
                if "optimizer_kwargs" in d:
                    self.optimizer_kwargs = d["optimizer_kwargs"]
                if "scheduler_kwargs" in d:
                    self.scheduler_kwargs = d["scheduler_kwargs"]
                # initialize optimizer and scheduler
                self.initialize_optimizer_and_scheduler()
                # load optimizer and scheduler state dict
                if "optimizer_state_dict" in d:
                    self.optimizer.load_state_dict(d["optimizer_state_dict"])
                if "scheduler_state_dict" in d:
                    self.scheduler.load_state_dict(d["scheduler_state_dict"])
            else:
                # put model in evaluation mode
                self.network.eval()

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        train_dir: str,
        runtime_limits: object = None,
        checkpoint_epochs: int = None,
        use_wandb=False,
        test_only=False,
        n_wfs_to_plot=0,
        early_stopping: Optional[EarlyStopping] = None,
    ):
        """

        Parameters
        ----------
        train_loader
        test_loader
        train_dir
        runtime_limits
        checkpoint_epochs
        use_wandb
        test_only: bool = False
            if True, training is skipped
        n_wfs_to_plot: int = 0
            if > 0, plot a few random training waveforms to disk
        early_stopping: EarlyStopping
            Optional EarlyStopping instance.

        Returns
        -------

        """

        if n_wfs_to_plot > 0:
            print(f"Plotting {n_wfs_to_plot} waveforms to {train_dir}")
            self.plot_waveforms(n_wfs_to_plot, train_dir, train_loader)
        if test_only:
            test_loss = test_epoch(self, test_loader)
            print(f"test loss: {test_loss:.3f}")

        else:
            while not runtime_limits.limits_exceeded(self.epoch):
                self.epoch += 1

                # Training
                lr = utils.get_lr(self.optimizer)
                with threadpool_limits(limits=1, user_api="blas"):
                    print(f"\nStart training epoch {self.epoch} with lr {lr}")
                    time_start = time.time()
                    train_loss = train_epoch(self, train_loader)
                    train_time = time.time() - time_start

                    print(
                        "Done. This took {:2.0f}:{:2.0f} min.".format(
                            *divmod(train_time, 60)
                        )
                    )

                    # Testing
                    print(f"Start testing epoch {self.epoch}")
                    time_start = time.time()
                    test_loss = test_epoch(self, test_loader)
                    test_time = time.time() - time_start

                    print(
                        "Done. This took {:2.0f}:{:2.0f} min.".format(
                            *divmod(time.time() - time_start, 60)
                        )
                    )

                # scheduler step for learning rate
                utils.perform_scheduler_step(self.scheduler, test_loss)

                # write history and save model
                utils.write_history(train_dir, self.epoch, train_loss, test_loss, lr)
                utils.save_model(self, train_dir, checkpoint_epochs=checkpoint_epochs)
                if use_wandb:
                    try:
                        import wandb

                        wandb.define_metric("epoch")
                        wandb.define_metric("*", step_metric="epoch")
                        wandb.log(
                            {
                                "epoch": self.epoch,
                                "learning_rate": lr[0],
                                "train_loss": train_loss,
                                "test_loss": test_loss,
                                "train_time": train_time,
                                "test_time": test_time,
                            }
                        )
                    except ImportError:
                        print("wandb not installed. Skipping logging to wandb.")

                if early_stopping is not None:
                    # Whether to use train or test loss
                    early_stopping_loss = (
                        test_loss
                        if early_stopping.metric == "validation"
                        else train_loss
                    )
                    is_best_model = early_stopping(early_stopping_loss)
                    if is_best_model:
                        self.save_model(
                            join(train_dir, "best_model.pt"), save_training_info=False
                        )
                    if early_stopping.early_stop:
                        print("Early stopping")
                        break
                print(f"Finished training epoch {self.epoch}.\n")

    def plot_waveforms(self, n, outdir, dataloader):
        """Plot n random training waveforms to outdir.

        Parameters
        ----------
        n : int
            Number of waveforms to plot
        outdir : str
            Directory to save the plots
        dataloader : torch.utils.data.DataLoader
            Dataloader containing the training data
        """
        import matplotlib.pyplot as plt

        # Build domain from metadata to get frequency information
        from dingo.gw.domains.build_domain import build_domain_from_model_metadata
        domain = build_domain_from_model_metadata(self.metadata, base=True)

        for batch_data in dataloader:
            waveforms = batch_data[1].cpu().numpy()

            # Get the number of waveforms in this batch
            batch_size = waveforms.shape[0]
            num_to_plot = min(n, batch_size)

            # Create figure
            fig, axes = plt.subplots(num_to_plot, 1, figsize=(10, 4 * num_to_plot))
            if num_to_plot == 1:
                axes = [axes]

            # For each waveform to plot
            idx = np.random.randint(0, batch_size, size=num_to_plot)
            for i in range(num_to_plot):
                ax = axes[i]

                # Get the first detector's data (typically H1 or the first IFO)
                # Shape: (num_ifos, 3, num_freq_bins)
                waveform_data = waveforms[idx[i], 0, :, :]  # (3, num_freq_bins)

                # Extract real, imaginary, and scaling factor
                real_part = waveform_data[0, :]
                imag_part = waveform_data[1, :]
                #scaling = waveform_data[2, :]
                complex_waveform = (real_part + 1j * imag_part)# * scaling

                # Create full frequency array for irfft
                # The data only includes frequencies >= f_min (masked)
                # irfft expects the full array with Nyquist frequency
                num_freqs_full = len(domain.frequency_mask) + 1
                to_fft = np.zeros(num_freqs_full, dtype=np.complex128)
                to_fft[1:][domain.frequency_mask] = complex_waveform

                time_domain_waveform = np.fft.irfft(to_fft)
                duration = domain.duration
                num_times = len(time_domain_waveform)
                time_array = np.linspace(0, duration, num_times, endpoint=False)

                # Check if H1_glitch_time is in the parameters
                tc = None
                param_names = self.metadata["train_settings"]["data"]["inference_parameters"]
                standardization = self.metadata["train_settings"]["data"].get("standardization")

                # Find glitch_time index
                for j, name in enumerate(param_names):
                    if "glitch_time" in name:
                        tc_standardized = batch_data[0][idx[i], j].cpu().numpy()
                        mean = standardization["mean"][name]
                        std = standardization["std"][name]
                        tc = tc_standardized * std + mean

                        if tc < 0:
                            tc += duration
                        break

                ax.plot(time_array, time_domain_waveform.real, alpha=0.7)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Strain')

                # Add vertical line at H1_glitch_time if available
                if tc is not None:
                    ax.axvline(x=tc, color='red', linestyle='--', alpha=0.7, label=f'Glitch time = {tc:.1f}s')
                    ax.legend()

            plt.tight_layout()

            # Save figure
            output_path = f"{outdir}/waveforms.png"
            plt.savefig(output_path)
            plt.close()

            print(f"Saved waveform plots to {output_path}")
            break  # Only process first batch

def train_epoch(pm, dataloader):
    pm.network.train()
    loss_info = dingo.core.utils.trainutils.LossInfo(
        pm.epoch,
        len(dataloader.dataset),
        dataloader.batch_size,
        mode="Train",
        print_freq=1,
    )

    for batch_idx, data in enumerate(dataloader):
        loss_info.update_timer()
        pm.optimizer.zero_grad()
        # data to device
        data = [d.to(pm.device, non_blocking=True) for d in data]
        # compute loss
        loss = pm.loss(data[0], *data[1:])
        # backward pass and optimizer step
        loss.backward()
        pm.optimizer.step()
        # update loss for history and logging
        loss_info.update(loss.detach().item(), len(data[0]))
        loss_info.print_info(batch_idx)

    return loss_info.get_avg()


def test_epoch(pm, dataloader):
    with torch.no_grad():
        pm.network.eval()
        loss_info = dingo.core.utils.trainutils.LossInfo(
            pm.epoch,
            len(dataloader.dataset),
            dataloader.batch_size,
            mode="Test",
            print_freq=1,
        )

        for batch_idx, data in enumerate(dataloader):
            loss_info.update_timer()
            # data to device
            data = [d.to(pm.device, non_blocking=True) for d in data]
            # compute loss
            loss = pm.loss(data[0], *data[1:])
            # update loss for history and logging
            loss_info.update(loss.item(), len(data[0]))
            loss_info.print_info(batch_idx)

        return loss_info.get_avg()
