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
    sample_mask: np.ndarray,
    next_point: np.ndarray | None = None,
    x_resolution: int = 200,
    figsize: tuple = (6, 4),
):
    # Ensure sample_mask is boolean numpy array for indexing
    mask_np = np.array(sample_mask, dtype=bool)
    
    # --- 1. Robust Statistics for Un-normalization ---
    # We must calculate the mean/std used during training to un-normalize the predictions correctly.
    # Convert to JAX array for consistent calculation with training code
    ys_jax = jnp.asarray(ys)
    ys_safe = jnp.nan_to_num(ys_jax, nan=0.0)
    
    # Calculate stats using the mask
    y_mean = jnp.mean(ys_safe, where=sample_mask)
    y_std = jnp.std(ys_safe, where=sample_mask)
    y_std = jnp.where(y_std < 1e-8, 1.0, y_std)
    
    # --- 2. Compute Acquisition Data ---
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
        sample_mask=sample_mask, # Pass mask to wrapper
    )
    min_a, max_a = bounds[0, 0].item(), bounds[0, 1].item()
    full_x = jnp.linspace(min_a, max_a, x_resolution)
    full_acq_vals = jax.vmap(acq_wrapper.jax_call)(full_x[:, None]).flatten()
    
    # --- 3. Compute Surrogate Values ---
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
    
    # --- 4. Un-normalize Predictions ---
    # Calculate mean and std of predictions
    mean, std = jnp.mean(full_pred, axis=1), jnp.std(full_pred, axis=1)
    
    # Apply the robust stats calculated in step 1
    # Check if we have valid samples (std > 0ish) to avoid garbage scaling
    if jnp.sum(sample_mask) > 1:
        mean = mean * y_std
        if next_mean is not None:
            next_mean = next_mean * y_std
        std = std * y_std
    
    mean = mean + y_mean
    if next_mean is not None:
        next_mean = next_mean + y_mean
    
    # --- 5. Plotting ---
    plt.close()
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    
    # Create figure with two subplots
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                     height_ratios=[3, 1],
                                     sharex=True)
    
    # --- Upper plot: Surrogate model ---
    ax1.plot(full_x, mean, 'b-', linewidth=2.5, label='Surrogate Mean')
    
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
    
    # --- FILTER POINTS FOR SCATTER ---
    # Only plot the valid points, ignore padding
    valid_xs = np.array(xs)[mask_np]
    valid_ys = np.array(ys)[mask_np]
    
    ax1.scatter(valid_xs, valid_ys, 
                c='red', s=100, zorder=5,
                marker='o', edgecolors='darkred', linewidth=2,
                label='Observations')
    
    # Plot next query point
    if next_point is not None and next_mean is not None:
        ax1.axvline(next_point, color='green', linestyle='--', 
                    linewidth=2, alpha=0.7, label='Next Query')
        ax1.scatter([next_point], [next_mean],
                    c='green', s=150, zorder=6,
                    marker='*', edgecolors='darkgreen', linewidth=2)
    
    ax1.set_ylabel('Function Value', fontsize=12)
    ax1.legend(loc='best', frameon=True, shadow=True, fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(min_a, max_a)
    
    # Update title count
    n_obs = len(valid_xs)
    ax1.set_title(f'Bonni - 1D ({n_obs} observations)', 
                    fontsize=14, fontweight='bold', pad=15)
    
    # --- Lower plot: Acquisition function ---
    max_af = jnp.max(full_acq_vals)
    min_af = max(max_af-100, jnp.min(full_acq_vals))
    
    # Handle edge case where ACQ is flat/inf
    if not np.isfinite(min_af) or not np.isfinite(max_af):
        min_af, max_af = 0, 1
        
    ax2.set_ylim(min_af, max_af + (max_af - min_af)*0.1) # Add slight headroom
    ax2.set_xlim(min_a, max_a)
    ax2.plot(full_x, full_acq_vals, 'purple', linewidth=2.5, label='Acquisition Function')
    
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