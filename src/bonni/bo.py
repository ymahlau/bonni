from pathlib import Path

from bonni.acquisition.ei import EIConfig
import numpy as np
from tqdm import tqdm

import jax
import jax.numpy as jnp

from bonni.acquisition.optim import optimize_acquisition_ipopt
from bonni.function import FunctionWrapper
from bonni.model.ensemble import MLPEnsembleConfig
from bonni.model.optim import OptimConfig
from bonni.model.training import full_regression_training_bnn
from bonni.visualization.surrogate import visualize_bo_1d
from bonni.visualization.training import plot_info


def bo_loop(
    fn: FunctionWrapper,
    bounds: np.ndarray,
    key: jax.Array,
    xs: np.ndarray,
    ys: np.ndarray,
    gs: np.ndarray,
    samples_after_init: int,
    ei_cfg: EIConfig,
    ensemble_cfg: MLPEnsembleConfig,
    optim_cfg: OptimConfig,
    training_plot_directory: Path | None,
    surrogate_plot_directory: Path | None,
    num_acq_optim_samples: int,
    num_embedding_channels: int,
):
    num_actions = bounds.shape[0]
    for idx in tqdm(range(samples_after_init)):
        # train ensemble surrogate
        key, subkey = jax.random.split(key)
        train_fn = jax.jit(full_regression_training_bnn, static_argnames=["model_cfg", "optim_cfg", "num_embedding_channels"])
        train_state, info = train_fn(
            key=subkey,
            x=jnp.asarray(xs),
            y=jnp.asarray(ys),
            g=jnp.asarray(gs),
            bounds=jnp.asarray(bounds),
            model_cfg=ensemble_cfg,
            optim_cfg=optim_cfg,
            num_embedding_channels=num_embedding_channels,
        )
        
        if training_plot_directory is not None:
            plot_info(info, idx, training_plot_directory)

        # optimize acquisition function
        key, subkey = jax.random.split(key)
        new_x, _ = optimize_acquisition_ipopt(
            params=train_state.params,
            key=subkey,
            xs=xs,
            ys=ys,
            bounds=bounds,
            num_acq_optim_samples=num_acq_optim_samples,
            num_embedding_channels=num_embedding_channels,
            ei_cfg=ei_cfg,
            ensemble_cfg=ensemble_cfg,
        )
        
        # visualize results (optionally)
        if surrogate_plot_directory is not None and num_actions == 1:
            key, subkey = jax.random.split(key)
            visualize_bo_1d(
                train_state.params,
                img_path=surrogate_plot_directory / f"surrogate_{idx}.png",
                key=subkey,
                xs=xs,
                ys=ys,
                bounds=bounds,
                num_embedding_channels=num_embedding_channels,
                ei_cfg=ei_cfg,
                ensemble_cfg=ensemble_cfg,
                next_point=new_x,
            )
        
        # sample new point
        new_y, new_grads = fn(new_x)
        xs = np.concatenate([xs, new_x[None, :]], axis=0)
        ys = np.concatenate([ys, new_y[None]], axis=0)
        gs = np.concatenate([gs, new_grads[None, :]], axis=0)
        
        
    return xs, ys, gs