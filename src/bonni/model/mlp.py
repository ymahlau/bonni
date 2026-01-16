from dataclasses import dataclass
from flax import linen as nn
import jax
from flax.linen import initializers

from bonni.model.utils import ActivationType, InitType, get_activation_fn, get_init_fn

class MLPLayer(nn.Module):
    out_channels: int
    activation_type: ActivationType
    norm_groups: int | None
    dropout_prob: float = 0.0
    bias_init: InitType = InitType.zeros
    skip_if_possible: bool = False
    
    def setup(self):
        bias_init_fn = get_init_fn(self.bias_init)
        self.fc1 = nn.Dense(
            features=self.out_channels, 
            bias_init=bias_init_fn,
            kernel_init=initializers.he_normal(),
        )
        self.norm = None
        if self.norm_groups is not None:
            self.norm = nn.GroupNorm(
                num_groups=self.norm_groups,
                epsilon=1e-5,  # Default epsilon value
            )
        self.activation = get_activation_fn(self.activation_type)
        self.dropout = None
        if self.dropout_prob > 0:
            self.dropout = nn.Dropout(
                rate=self.dropout_prob,
            )
    
    def __call__(
        self, 
        x: jax.Array, 
        deterministic: bool = False,
    ) -> jax.Array:
        # Apply first linear transformation
        y = self.fc1(x)
        # group norm
        if self.norm is not None:
            y = self.norm(y)
        # Apply activation
        y = self.activation(y)
        if self.skip_if_possible and x.shape == y.shape:
            y = x + y
        # Dropout
        if self.dropout is not None:
            y = self.dropout(y, deterministic=deterministic)
        return y


@dataclass(frozen=True, kw_only=True)
class MLPModelConfig:
    """
    Configuration object for a Multi-Layer Perceptron (MLP) model.

    This dataclass defines the structural and hyperparameter settings for an MLP,
    including layer dimensions, normalization, dropout, and activation strategies.
    It is frozen (immutable) and requires keyword arguments for initialization.

    Attributes:
        num_layer (int): The total number of linear layers in the MLP.
        out_channels (int): The dimensionality of the output features.
        hidden_channels (int | None): The dimensionality of the hidden layers. 
            If None, this is typically inferred from the input or output channels 
            depending on the implementation. Defaults to None.
        norm_groups (int | None): The number of groups to use for Group Normalization 
            in the hidden layers. If None, normalization is skipped. Defaults to None.
        last_norm_groups (int | None): The number of groups for Group Normalization 
            applied to the final layer. If None, no normalization is applied to the 
            output. Defaults to None.
        dropout_prob (float): The dropout probability applied after hidden layers. 
            Must be between 0.0 and 1.0. Defaults to 0.0.
        last_dropout_prob (float): The dropout probability applied after the final layer. 
            Defaults to 0.0.
        activation_type (ActivationType): The activation function used after hidden layers. 
            Defaults to ActivationType.gelu.
        different_last_activation (ActivationType | None): The activation function 
            used after the final layer. If set to `ActivationType.identity`, the output 
            is linear. If None, the model typically uses the same activation as 
            `activation_type`. Defaults to ActivationType.identity.
        bias_init (InitType): The initialization strategy for the layer biases 
            (e.g., zeros, uniform). Defaults to InitType.zeros.
        skip_if_possible (bool): If True, adds residual connections (skip connections) 
            around layers where the input and output dimensions are identical. 
            Defaults to True.
    """
    num_layer: int
    out_channels: int
    hidden_channels: int | None = None
    norm_groups: int | None = None
    last_norm_groups: int | None = None
    dropout_prob: float = 0.0
    last_dropout_prob: float = 0.0
    activation_type: ActivationType = ActivationType.gelu
    different_last_activation: ActivationType | None = ActivationType.identity  # if none, use same activation
    bias_init: InitType = InitType.zeros
    skip_if_possible: bool = True


class MLP(nn.Module):
    cfg: MLPModelConfig
    
    def setup(self):
        if self.cfg.num_layer > 1:
            assert self.cfg.hidden_channels is not None, "need hidden dim with >1 layer"
            
        layers = []
        for idx in range(self.cfg.num_layer):
            # select activation
            cur_activ = (
                self.cfg.activation_type 
                if idx != self.cfg.num_layer-1 or self.cfg.different_last_activation is None 
                else self.cfg.different_last_activation
            )
            # select out channels
            cur_out_channels = self.cfg.out_channels
            if idx != self.cfg.num_layer-1:
                assert self.cfg.hidden_channels is not None
                cur_out_channels = self.cfg.hidden_channels
            # select dropout prob, norm_groups
            dropout_prob = self.cfg.dropout_prob if idx != self.cfg.num_layer-1 else self.cfg.last_dropout_prob
            norm_groups = self.cfg.norm_groups if idx != self.cfg.num_layer-1 else self.cfg.last_norm_groups
            cur_skip_if_possible = self.cfg.skip_if_possible if idx != self.cfg.num_layer-1 else False
            # build current layer
            cur_layer = MLPLayer(
                out_channels=cur_out_channels,
                activation_type=cur_activ,
                norm_groups=norm_groups,
                dropout_prob=dropout_prob,
                bias_init=self.cfg.bias_init,
                skip_if_possible=cur_skip_if_possible,
            )
            layers.append(cur_layer)
        self.layers = layers
    
    def __call__(
        self,
        x: jax.Array,
        deterministic: bool = False,
    ) -> jax.Array:
        for layer in self.layers:
            x = layer(x, deterministic=deterministic)
        return x
    
    




