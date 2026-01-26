from pathlib import Path

from bonni.acquisition.ei import EIConfig
import numpy as np
from tqdm import tqdm

import jax
import jax.numpy as jnp

from bonni.acquisition.optim import optimize_acquisition_ipopt
from bonni.function import FunctionWrapper
from bonni.ipopt import optimize_ipopt
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
    max_iteration_count: int | None,
    ei_cfg: EIConfig,
    ensemble_cfg: MLPEnsembleConfig,
    optim_cfg: OptimConfig,
    training_plot_directory: Path | None,
    surrogate_plot_directory: Path | None,
    num_acq_optim_samples: int,
    num_embedding_channels: int,
    num_iter_until_recompile: int,
    non_diff_params: np.ndarray,
    max_num_local_samples: int,
):
    num_actions = bounds.shape[0]
    cur_num_samples = xs.shape[0]

    # Create the mask: True for valid data, False for the buffer
    cur_mask = np.zeros(cur_num_samples, dtype=bool)
    cur_mask[:cur_num_samples] = True

    # Track the actual index we are filling. This is the maximum index where arrays are written in each iteration.
    # For a single sample this is the actual index, for multiple local search iteration the last possible index
    next_idx = (cur_num_samples - 1) + max_num_local_samples
    iteration_counter: int = 0
    
    for idx in tqdm(range(samples_after_init)):
        # If we have filled the buffer, extend it by another chunk.
        # This changes the array shape and triggers ONE recompilation for the next batch.
        if next_idx >= len(cur_mask):
            extra_pad = num_iter_until_recompile
            
            # Pad arrays
            xs = np.pad(xs, ((0, extra_pad), (0, 0)), constant_values=0)
            ys = np.pad(ys, ((0, extra_pad)), constant_values=0)
            gs = np.pad(gs, ((0, extra_pad), (0, 0)), constant_values=0)
            
            # Extend mask
            new_mask_chunk = np.zeros(extra_pad, dtype=bool)
            cur_mask = np.concatenate([cur_mask, new_mask_chunk])
        
        # train ensemble surrogate
        key, subkey = jax.random.split(key)
        train_fn = jax.jit(full_regression_training_bnn, static_argnames=["model_cfg", "optim_cfg", "num_embedding_channels"])
        train_state, info = train_fn(
            key=subkey,
            x=jnp.asarray(xs),
            y=jnp.asarray(ys),
            g=jnp.asarray(gs),
            bounds=jnp.asarray(bounds),
            sample_mask=jnp.asarray(cur_mask),
            non_diff_params=jnp.asarray(non_diff_params, dtype=jnp.bool),
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
            sample_mask=cur_mask,
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
                sample_mask=cur_mask,
            )
        
        # sample new point
        if max_num_local_samples > 1:
            # run local search
            new_xs, new_ys, new_gs = optimize_ipopt(
                fn=fn,
                x0=new_x,
                bounds=bounds,
                max_fn_eval=max_num_local_samples,
                direction="maximize",  # internally we always maximize
            )
            
            # update arrays
            start_idx = next_idx - max_num_local_samples
            end_idx = start_idx + new_xs.shape[0]
            xs[start_idx:end_idx] = new_xs
            ys[start_idx:end_idx] = new_ys
            gs[start_idx:end_idx] = new_gs
            cur_mask[start_idx:end_idx] = True
            
            # set max next idx value
            next_idx = end_idx + max_num_local_samples
        else:
            # simple function evaluation
            new_y, new_grads = fn(new_x)
        
            # Update the padded arrays at the current index
            xs[next_idx] = new_x
            ys[next_idx] = new_y
            gs[next_idx] = new_grads
            cur_mask[next_idx] = True
            
            # Increment index for next iteration
            next_idx += 1
        
        # break condition: maximum iteration count
        iteration_counter += 1
        if max_iteration_count is not None and iteration_counter > max_iteration_count:
            break
        
    return xs, ys, gs