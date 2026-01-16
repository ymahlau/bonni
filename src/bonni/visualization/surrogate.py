from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
import seaborn as sns

from bonni.acquisition.ei import EIConfig
from bonni.acquisition.optim import AcqFnWrapper
from bonni.model.embedding import SinCosActionEmbedding
from bonni.model.ensemble import MLPEnsemble, MLPEnsembleConfig


def visualize_bo_1d(
    params,
    img_path: Path | str,
    key: jax.Array,
    xs: np.ndarray,
    ys: np.ndarray,
    bounds: np.ndarray,
    num_embedding_channels: int,
    ei_cfg: EIConfig,
    ensemble_cfg: MLPEnsembleConfig,
    next_point: np.ndarray | None = None,
    x_resolution: int = 200,
    figsize: tuple = (6, 4),
):
    # compute acquition data
    key, subkey = jax.random.split(key)
    acq_wrapper = AcqFnWrapper(
        xs=xs,
        ys=ys,
        bounds=bounds,
        num_embedding_channels=num_embedding_channels,
        ei_cfg=ei_cfg,
        ensemble_cfg=ensemble_cfg,
        params=params,
        key=subkey,
    )
    min_a, max_a = bounds[0, 0].item(), bounds[0, 1].item()
    full_x = jnp.linspace(min_a, max_a, x_resolution)
    full_acq_vals = jax.vmap(acq_wrapper.jax_call)(full_x[:, None]).flatten()
    
    # compute surrogate values
    embedding = SinCosActionEmbedding(num_embedding_channels)
    model = MLPEnsemble(ensemble_cfg)
    jax_bounds = jnp.asarray(bounds)
    full_obs = jax.vmap(embedding, in_axes=[0, None])(full_x[:, None], jax_bounds)
    key, subkey = jax.random.split(key)
    full_pred = jax.vmap(model.apply, in_axes=[None, 0, None])(params, full_obs, subkey)
    assert isinstance(full_pred, jax.Array) 
    assert full_pred.ndim == 2
    assert full_pred.shape[0] == x_resolution
    assert full_pred.shape[1] == model.cfg.num_models
    
    max_af_idx = jnp.argmax(full_acq_vals)
    if next_point is None:
        next_point = full_x[max_af_idx] # type: ignore
    next_mean = None
    if next_point is not None:
        next_obs = embedding(jnp.asarray(next_point), jax_bounds)
        key, subkey = jax.random.split(key)
        next_pred = jax.jit(model.apply)(params, next_obs, rngs=subkey)
        next_mean = jnp.mean(next_pred)
    
    # calculate mean and std
    mean, std = jnp.mean(full_pred, axis=1), jnp.std(full_pred, axis=1)
    if ys.size > 1:
        mean = mean * ys.std()
        next_mean = next_mean * ys.std()
        std = std * ys.std()
    mean = mean + ys.mean()
    next_mean = next_mean + ys.mean()
    
    # Set style
    plt.close()
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    
    # Compute values
    key = jax.random.PRNGKey(42)
    
    # Create figure with two subplots
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                     height_ratios=[3, 1],
                                     sharex=True)
    
    # --- Upper plot: Surrogate model ---
    # Plot surrogate mean
    ax1.plot(full_x, mean, 'b-', linewidth=2.5, label='Surrogate Mean')
    
    # Plot confidence intervals
    ax1.fill_between(full_x, 
                     mean - 2 * std, 
                     mean + 2 * std,
                     alpha=0.2, 
                     color='blue',
                     label='95% Confidence')
    ax1.fill_between(full_x, 
                     mean - std, 
                     mean + std,
                     alpha=0.3, 
                     color='blue',
                     label='68% Confidence')
    
    # Plot observed points
    ax1.scatter(xs, ys, 
                c='red', s=100, zorder=5,
                marker='o', edgecolors='darkred', linewidth=2,
                label='Observations')
    
    
    
    # Plot next query point
    if next_point is not None and next_mean is not None:
        key, subkey = jax.random.split(key)
        
        ax1.axvline(next_point, color='green', linestyle='--', 
                   linewidth=2, alpha=0.7, label='Next Query')
        ax1.scatter([next_point], [next_mean],
                   c='green', s=150, zorder=6,
                   marker='*', edgecolors='darkgreen', linewidth=2)
    
    ax1.set_ylabel('Function Value', fontsize=12)
    ax1.legend(loc='best', frameon=True, shadow=True, fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(min_a, max_a)
    
    n_obs = len(xs)
    ax1.set_title(f'Bonni - 1D ({n_obs} observations)', 
                    fontsize=14, fontweight='bold', pad=15)
    
    # --- Lower plot: Acquisition function ---
    max_af = jnp.max(full_acq_vals)
    min_af = max(max_af-100, jnp.min(full_acq_vals))
    ax2.set_ylim(min_af, max_af)
    ax2.set_xlim(min_a, max_a)
    ax2.plot(full_x, full_acq_vals, 'purple', linewidth=2.5, label='Acquisition Function')
    
    # Mark maximum of acquisition function
    # ax2.scatter([x.flatten()[max_af_idx]], [af_values[max_af_idx]], 
    #            c='orange', s=150, zorder=5,
    #            marker='^', edgecolors='darkorange', linewidth=2,
    #            label='AF Maximum')
    
    # Mark next query point on AF plot
    
    if next_point is not None:
        ax2.axvline(next_point, color='green', linestyle='--', 
                   linewidth=2, alpha=0.7)
    
    ax2.set_xlabel('Parameter Value', fontsize=12,)
    ax2.set_ylabel('Acquisition Value', fontsize=12,)
    ax2.legend(loc='best', frameon=True, shadow=True, fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(img_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    
