from enum import Enum
from typing import Callable
import jax
import jax.numpy as jnp
from flax import linen as nn


class SkipConnectionType(Enum):
    linear = "linear"
    identity = "identity"


def get_skip_connection(
    in_channels: int,
    out_channels: int,
    skip_type: SkipConnectionType,
):
    """Create a skip connection module."""
    if skip_type == SkipConnectionType.linear:
        return _LinearSkipConnection(in_channels, out_channels)
    if skip_type == SkipConnectionType.identity:
        if in_channels == out_channels:
            return lambda x: x
        else:
            return _LinearSkipConnection(in_channels, out_channels)
    raise ValueError(f"Unsupported skip connection type: {skip_type}")


class _LinearSkipConnection(nn.Module):
    """A linear skip connection implemented as a flax.linen Module."""

    in_features: int
    out_features: int

    def setup(self):
        self.linear = nn.Dense(
            features=self.out_features,
            use_bias=False,
            kernel_init=nn.initializers.lecun_normal(),
        )

    def __call__(self, x):
        return self.linear(x)


class ActivationType(Enum):
    """
    Enumeration of supported activation functions for neural network layers.

    These values are used to configure the non-linearity applied after linear
    transformations in the model configuration.

    Attributes:
        identity: Applies no activation (f(x) = x). typically used for the final
            output layer to produce unbounded linear predictions.
        gelu: Gaussian Error Linear Unit. A smooth approximation of ReLU often
            used in Transformer architectures and modern MLPs.
        relu: Rectified Linear Unit (f(x) = max(0, x)). A standard non-linear
            activation that introduces sparsity.
        leaky_relu: Leaky Rectified Linear Unit. Similar to ReLU but allows a
            small, non-zero gradient when the unit is not active.
        sigmoid: Sigmoid function. Squashes values to the range [0, 1], often
            used for binary classification probabilities.
        tanh: Hyperbolic Tangent. Squashes values to the range [-1, 1].
    """

    identity = "identity"
    gelu = "gelu"
    relu = "relu"
    leaky_relu = "leaky_relu"
    sigmoid = "sigmoid"
    tanh = "tanh"


def get_activation_fn(
    activation_type: ActivationType,
) -> Callable[[jax.Array], jax.Array]:
    if activation_type == ActivationType.identity:
        return lambda x: x
    if activation_type == ActivationType.gelu:
        return jax.nn.gelu
    if activation_type == ActivationType.relu:
        return jax.nn.relu
    if activation_type == ActivationType.leaky_relu:
        return jax.nn.leaky_relu
    if activation_type == ActivationType.sigmoid:
        return jax.nn.sigmoid
    if activation_type == ActivationType.tanh:
        return jax.nn.tanh
    raise ValueError(f"Invalid activation type: {activation_type}")


class InitType(Enum):
    """
    Enumeration of initialization strategies for model parameters.

    These values define how weights or biases are initialized before training begins.
    Used primarily for `bias_init` in the model configuration.

    Attributes:
        zeros: Initializes parameters to exactly 0. This is the standard practice
            for bias terms in most neural network layers.
        ones: Initializes parameters to exactly 1.
        uniform: Initializes parameters with values drawn from a uniform distribution.
            The range is typically determined by the specific layer implementation.
        normal: Initializes parameters with values drawn from a normal (Gaussian)
            distribution.
    """

    zeros = "zeros"
    ones = "ones"
    uniform = "uniform"
    normal = "normal"


def get_init_fn(
    init_type: InitType,
):
    if init_type == InitType.zeros:
        return nn.initializers.zeros
    if init_type == InitType.ones:
        return nn.initializers.ones
    if init_type == InitType.uniform:
        scale = 0.01

        def init(key, shape, dtype=jnp.float64, out_sharding=None) -> jax.Array:
            return jax.random.uniform(
                key, shape, dtype=dtype, out_sharding=out_sharding
            ) * jnp.array(scale, dtype)

        return jax.tree_util.Partial(init)
    if init_type == InitType.normal:
        scale = 0.1

        def init(key, shape, dtype=jnp.float64, out_sharding=None) -> jax.Array:
            return jax.random.normal(
                key, shape, dtype=dtype, out_sharding=out_sharding
            ) * jnp.array(scale, dtype)

        return jax.tree_util.Partial(init)
    raise ValueError(f"Invalid init type: {init_type}")
