from torch.nn import functional as F

from .base_model import BasePosteriorModel

from dingo.core.nn.nsf import (
    create_nsf_with_compression_embedding_net,
    create_nsf_with_rb_projection_embedding_net,
    create_nsf_with_transformer_embedding_net,
    create_nsf_wrapped,
)


class NormalizingFlowPosteriorModel(BasePosteriorModel):
    """
    Posterior model based on a (discrete) normalizing flow.

    A normalizing flow describes a distribution as a sequence of discrete
    transformations on a parameter space, ultimately taking samples from the base space
    (multivariate standard normal) to the desired distribution. The discrete transforms
    are parametrized functions (e.g., splines), which are designed to be invertible with
    simple Jacobian determinant. The probability density is given by the change of
    variables rule,

    q(theta | d) = pi(f_d^{-1}(theta)) | det J_{f_d^{-1}} |

    where
        pi = N(0,1)^D is the base space distribution
        f_d is the normalizing flow on the D-dimensional space

    The flow f_d is allowed to depend on context information d, which would be
    observational data in the case of posterior estimation. By construction, the flow
    has fast sampling and density evaluation, require just forward passes of the network.

    This class uses normalizing flows from the dingo.core.nn.nsf module (which in turn
    uses glasflow, which is based on nflows). It is intended to construct and hold a
    neural network for estimating the posterior density, as well as saving / loading,
    and training. It also calls the sampling and density evaluation routines from the
    flows.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def initialize_network(self):
        """Instantiate the NSF network, dispatching on embedding_type."""
        _skip = {"posterior_model_type", "embedding_type"}
        model_kwargs = {k: v for k, v in self.model_kwargs.items() if k not in _skip}
        if self.initial_weights is not None:
            model_kwargs["initial_weights"] = self.initial_weights

        embedding_type = (self.model_kwargs.get("embedding_type") or "").lower()
        if embedding_type == "transformer":
            self.network = create_nsf_with_transformer_embedding_net(**model_kwargs)
        elif embedding_type == "compression":
            training = self.metadata["train_settings"].setdefault("training", {})
            self.mse_weight = float(training.setdefault("lambda", 1.0))
            self.latent_weight = 0.0
            if self.condition_on_parameters:
                self.latent_weight = float(training.setdefault("beta", 1.0))
                self.model_kwargs["embedding_kwargs"].setdefault("initialization_seed", 0)
            self.network = create_nsf_with_compression_embedding_net(**model_kwargs)
        elif self.model_kwargs.get("embedding_kwargs", False):
            self.network = create_nsf_with_rb_projection_embedding_net(**model_kwargs)
        else:
            self.network = create_nsf_wrapped(**model_kwargs["posterior_kwargs"])

    def log_prob(self, theta, *context):
        return self.network(theta, *context)

    def sample(self, *context, num_samples: int = 1):
        return self.network.sample(*context, num_samples=num_samples)

    def sample_and_log_prob(self, *context, num_samples: int = 1):
        return self.network.sample_and_log_prob(*context, num_samples=num_samples)

    @property
    def condition_on_parameters(self):
        """Whether compression uses a distilled, two-pass initialization."""
        return (
            (self.model_kwargs.get("embedding_type") or "").lower() == "compression"
            and self.model_kwargs["embedding_kwargs"].get(
                "condition_on_parameters", False
            )
        )

    def prepare_compression_proposal(self, strain, *context, initial_parameters=None):
        """Prepare two-pass inference context for sample/sample_and_log_prob/log_prob.

        Only for parameter-conditioned compression. The optional replay estimate
        is in standardized inference-parameter order. Retain the returned proposal
        for all draws and density evaluations of the same observation.
        """
        return self.network.prepare_proposal(
            strain, *context, initial_parameters=initial_parameters
        )

    def loss(self, theta, *context):
        if (self.model_kwargs.get("embedding_type") or "").lower() == "compression":
            return self.compression_loss_components(theta, *context)[0]
        return -self.network(theta, *context).mean()

    def compression_loss_components(self, theta, strain, target, *context):
        """Return (total, SBI, reconstruction MSE, latent MSE).

        The data-only encoder receives only latent-matching gradients; the
        conditional latent is detached only for that matching target.
        """
        # Share one encoder evaluation between the flow and decoder.
        log_prob, reconstruction, *latents = self.network(
            theta, strain, *context, return_reconstruction=True
        )
        # The ASD channel has no reconstruction loss.
        strain_reconstruction = reconstruction[..., :2, :]
        sbi_loss = -log_prob.mean()
        compression_loss = F.mse_loss(strain_reconstruction, target)
        if self.condition_on_parameters:
            data_latent, conditional_latent = latents
            latent_loss = F.mse_loss(data_latent, conditional_latent.detach())
        else:
            latent_loss = compression_loss.new_zeros(())
        loss = (
            sbi_loss
            + self.mse_weight * compression_loss
            + self.latent_weight * latent_loss
        )
        return loss, sbi_loss, compression_loss, latent_loss
