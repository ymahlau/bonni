from dataclasses import dataclass

import jax

from bonni.model.mlp import MLP, MLPModelConfig


@dataclass(frozen=True, kw_only=True)
class MLPEnsembleConfig:
    base_cfg: MLPModelConfig
    num_models: int


class MLPEnsemble:
    def __init__(self, cfg: MLPEnsembleConfig) -> None:
        self.cfg = cfg
        self.base_model = MLP(self.cfg.base_cfg)

    def init(
        self,
        key: jax.Array,
        x: jax.Array,
    ):
        key_list = jax.random.split(key, self.cfg.num_models)
        vmapped_params = jax.vmap(self.base_model.init, in_axes=[0, None])(key_list, x)
        return vmapped_params

    def apply(
        self,
        params,
        x: jax.Array,
        deterministic: bool = False,
        rngs: jax.Array | None = None,
        single_forward: bool = False,
    ) -> jax.Array:
        assert x.ndim == 1

        def _single_sample_single_model(p, k):
            return self.base_model.apply(p, x, deterministic=deterministic, rngs=k)

        if rngs is not None:
            assert rngs.size == 2
            if single_forward:
                forward_fn = _single_sample_single_model
                k = rngs
            else:
                forward_fn = jax.vmap(_single_sample_single_model, in_axes=(0, 0))
                k = jax.random.split(rngs, self.cfg.num_models)
            results = forward_fn(params, k)
        else:
            forward_fn = (
                _single_sample_single_model
                if single_forward
                else jax.vmap(_single_sample_single_model, in_axes=(0, None))
            )
            results = forward_fn(params, None)

        assert isinstance(results, jax.Array)
        results = results.flatten()
        if single_forward:
            assert results.size == 1
        else:
            assert results.size == self.cfg.num_models
        return results
