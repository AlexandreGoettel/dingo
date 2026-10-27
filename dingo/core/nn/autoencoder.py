"""Dense residual autoencoder for compression of multi-dimensional inputs."""

from math import prod
from typing import Optional, Sequence

from torch import Tensor, cat, nn

from dingo.core.nn.resnet import DenseResidualNet
from dingo.core.utils.torchutils import get_activation_function_from_string


def create_dense_encoder(
    input_dims: Sequence[int],
    output_dim: int,
    hidden_dims: Sequence[int],
    activation: str = "elu",
    dropout: float = 0.0,
    norm: Optional[str] = "BatchNorm",
    parameter_dim: int = 0,
) -> nn.Sequential:
    """Build an encoder, optionally including flattened parameter inputs."""
    return nn.Sequential(
        nn.Flatten(start_dim=1),
        DenseResidualNet(
            input_dim=prod(input_dims) + parameter_dim,
            output_dim=output_dim,
            hidden_dims=tuple(hidden_dims),
            activation=get_activation_function_from_string(activation),
            dropout=dropout,
            norm=norm,
        ),
    )


class DenseResidualAutoencoder(nn.Module):
    """Encode flattened inputs and reconstruct them with a mirrored ResNet.

    Both networks use the same residual blocks as Dingo's ResNet embedding.
    The decoder reverses the encoder's hidden widths and has independent weights.
    There is no SVD projection or initialization.

    Parameters
    ----------
    input_dims : sequence of int
        Input shape excluding the batch dimension, e.g. (detectors, channels, bins).
    output_dim : int
        Dimension of the compressed representation.
    hidden_dims : sequence of int
        Encoder residual-block widths, reversed for the decoder.
    activation : str
        Activation name, as in Dingo's ResNet embedding settings.
    dropout : float
        Dropout probability in both networks' residual blocks.
    norm : str or None
        Residual-block normalization: "BatchNorm", "LayerNorm", or None.
    parameter_dim : int
        Number of standardized parameters concatenated to the encoder input.
        The decoder continues to reconstruct only the original data shape.
    """

    def __init__(
        self,
        input_dims: Sequence[int],
        output_dim: int,
        hidden_dims: Sequence[int],
        activation: str = "elu",
        dropout: float = 0.0,
        norm: Optional[str] = "BatchNorm",
        parameter_dim: int = 0,
    ):
        super().__init__()
        self.input_dims = tuple(input_dims)
        self.hidden_dims = tuple(hidden_dims)
        self.output_dim = output_dim
        self.parameter_dim = parameter_dim
        if not self.input_dims or any(dim <= 0 for dim in self.input_dims):
            raise ValueError("input_dims must contain positive dimensions.")
        if not self.hidden_dims or any(dim <= 0 for dim in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive dimensions.")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive.")

        self.input_dim = prod(self.input_dims)
        activation_fn = get_activation_function_from_string(activation)
        self.encoder = create_dense_encoder(
            self.input_dims,
            output_dim,
            self.hidden_dims,
            activation,
            dropout,
            norm,
            parameter_dim,
        )
        self.decoder = nn.Sequential(
            DenseResidualNet(
                input_dim=output_dim,
                output_dim=self.input_dim,
                hidden_dims=self.hidden_dims[::-1],
                activation=activation_fn,
                dropout=dropout,
                norm=norm,
            ),
            nn.Unflatten(dim=1, unflattened_size=self.input_dims),
        )

    def encode(self, x: Tensor, parameters: Optional[Tensor] = None) -> Tensor:
        """Map (batch, *input_dims) to (batch, output_dim)."""
        if self.parameter_dim:
            x = cat((x.flatten(start_dim=1), parameters), dim=-1)
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        """Map (batch, output_dim) back to (batch, *input_dims)."""
        return self.decoder(z)

    def forward(self, x: Tensor, parameters: Optional[Tensor] = None) -> Tensor:
        """Reconstruct the input through the compressed representation."""
        return self.decode(self.encode(x, parameters))
