from dataclasses import dataclass

import jax
import jax.numpy as jnp
    

class SinCosActionEmbedding:
    def __init__(
        self,
        num_channels: int,
    ):
        self.num_channels = num_channels
        
    @property
    def channels_per_action(self) -> int:
        return 2 * self.num_channels + 1
    
    def get_full_num_channel(self, num_actions: int) -> int:
        return self.channels_per_action * num_actions
    
    def embed_single_action(
        self,
        action: jax.Array,
        action_bounds: jax.Array,
    ):
        assert action_bounds.ndim == 1
        assert action_bounds.shape[0] == 2
        assert action.size == 1
        action = action.flatten()
        # map action between 0 and pi
        min_a, max_a = action_bounds[0], action_bounds[1]
        mapped = (action - min_a) / (max_a - min_a)
        relative_pos = mapped * jnp.pi
        
        freqs = jnp.arange(1, self.num_channels + 1)
        
        # data and concat
        orig_val = mapped * 2 - 1
        sin_vals = jnp.sin(relative_pos * freqs)
        cos_vals = jnp.cos(relative_pos * freqs)
        
        result = jnp.concat((orig_val, sin_vals, cos_vals), axis=0).flatten()
        return result    
    
    def __call__(
        self,
        actions: jax.Array,
        action_bounds: jax.Array,
    ) -> jax.Array:
        assert actions.ndim == 1
        assert action_bounds.ndim == 2
        assert action_bounds.shape[0] == actions.shape[0]
        assert action_bounds.shape[1] == 2
        
        num_actions = actions.shape[0]
        full_num_channels = self.get_full_num_channel(num_actions)
        
        vmapped_embed_fn = jax.vmap(self.embed_single_action, in_axes=[0, 0])
        embedded_actions = vmapped_embed_fn(actions, action_bounds)
        
        return embedded_actions.reshape(full_num_channels,)
