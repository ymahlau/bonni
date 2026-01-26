import numpy as np

import jax
import jax.numpy as jnp

from bonni.acquisition.ei import EIConfig, ExpectedImprovement
from bonni.ipopt import optimize_ipopt
from bonni.model.embedding import SinCosActionEmbedding
from bonni.model.ensemble import MLPEnsemble, MLPEnsembleConfig


class AcqFnWrapper:
    def __init__(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        bounds: np.ndarray,
        num_embedding_channels: int,
        ei_cfg: EIConfig,
        ensemble_cfg: MLPEnsembleConfig,
        params,
        sample_mask: np.ndarray,
    ):
        self.xs = jnp.asarray(xs)
        self.ys = jnp.asarray(ys)
        self.bounds = jnp.asarray(bounds)
        self.num_embedding_channels = num_embedding_channels
        self.ei_cfg = ei_cfg
        self.ensemble_cfg = ensemble_cfg
        self.params = params
        self.af = ExpectedImprovement(ei_cfg)
        self.embedding = SinCosActionEmbedding(num_channels=num_embedding_channels)
        self.model = MLPEnsemble(ensemble_cfg)
        self.sample_mask = sample_mask
    
    def _forward(self, x_local: jax.Array) -> jax.Array:
        jitted_af = jax.jit(self.af, static_argnames=["model", "embedding"])
        acq_value = jitted_af(
            x_test=x_local,
            xs=jnp.asarray(self.xs),
            ys=jnp.asarray(self.ys),
            bounds=jnp.asarray(self.bounds),
            params=self.params,
            model=self.model,
            embedding=self.embedding,
            sample_mask=jnp.asarray(self.sample_mask),
        )
        return acq_value
    
    def __call__(
        self,
        x: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        x_jax = jnp.asarray(x)
        value, grad = jax.value_and_grad(self._forward)(x_jax)
        value_numpy = np.asarray(value, dtype=float)
        grad_numpy = np.asarray(grad, dtype=float)
        return value_numpy, grad_numpy
    
    def jax_call(
        self,
        x: jax.Array,
    ) -> jax.Array:
        assert x.ndim == 1 and x.shape[0] == self.bounds.shape[0]
        return self._forward(x)


def optimize_acquisition_ipopt(
    params,
    key: jax.Array,
    xs: np.ndarray,
    ys: np.ndarray,
    bounds: np.ndarray,
    num_acq_optim_samples: int,
    num_embedding_channels: int,
    ei_cfg: EIConfig,
    ensemble_cfg: MLPEnsembleConfig,
    sample_mask: np.ndarray,
    num_runs: int,
    num_initial_random_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    max_af_value, best_new_sample = None, None
    for _ in range(num_runs):
        acq_wrapper = AcqFnWrapper(
            xs=xs,
            ys=ys,
            bounds=bounds,
            num_embedding_channels=num_embedding_channels,
            ei_cfg=ei_cfg,
            ensemble_cfg=ensemble_cfg,
            params=params,
            sample_mask=sample_mask,
        )
        
        num_actions = bounds.shape[0]
        key, subkey = jax.random.split(key)
        random_actions = jax.random.uniform(subkey, shape=(num_initial_random_samples, num_actions,))
        random_actions_np = np.asarray(random_actions, dtype=float)
        action_ranges = bounds[:, 1] - bounds[:, 0]
        all_x0 = random_actions_np * action_ranges + bounds[:, 0]
        all_y0 = jax.vmap(acq_wrapper._forward)(jnp.asarray(all_x0))
        
        best_x0_idx = np.argmax(all_y0)
        x0 = all_x0[best_x0_idx]
        
        ax, ay, _ = optimize_ipopt(
            fn=acq_wrapper,
            x0=x0,
            bounds=bounds,
            max_fn_eval=num_acq_optim_samples,
            direction="maximize",
        )
        
        max_idx = np.argmax(ay)
        
        cur_best_sample = ax[max_idx]
        af_value = ay[max_idx]
        if best_new_sample is None or af_value > max_af_value:
            best_new_sample = cur_best_sample
        if max_af_value is None or af_value > max_af_value:
            max_af_value = af_value
    
    assert best_new_sample is not None
    assert max_af_value is not None
    return best_new_sample, max_af_value
    
    
    
    

