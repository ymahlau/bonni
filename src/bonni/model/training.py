import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
import optax

from bonni.model.ensemble import MLPEnsembleConfig, MLPEnsemble
from bonni.model.optim import OptimConfig
from bonni.model.embedding import SinCosActionEmbedding
from bonni.model.optim import get_optimizer_from_cfg


def full_regression_training_bnn(
    key: jax.Array,
    x: jax.Array,
    y: jax.Array,
    g: jax.Array,
    bounds: jax.Array,
    sample_mask: jax.Array,
    non_diff_params: jax.Array,
    model_cfg: MLPEnsembleConfig,
    optim_cfg: OptimConfig,
    num_embedding_channels: int,
) -> tuple[TrainState, dict[str, jax.Array]]:
    # --- Assertions ---
    assert x.ndim == 2
    num_samples, num_actions = x.shape[0], x.shape[1]
    assert bounds.ndim == 2
    assert bounds.shape[1] == 2
    assert bounds.shape[0] == num_actions
    assert y.ndim == 1
    assert y.shape[0] == num_samples
    if g is not None:
        assert g.ndim == 2
        assert g.shape == x.shape
    assert sample_mask.ndim == 1
    assert sample_mask.shape[0] == num_samples
    assert non_diff_params.ndim == 1
    assert non_diff_params.shape[0] == num_actions

    # --- Model Init ---
    embedding = SinCosActionEmbedding(num_channels=num_embedding_channels)
    model = MLPEnsemble(model_cfg)

    # Initialize with the first sample (safe assumption that idx 0 is valid,
    # or just use zeros for shape inference)
    obs = embedding(x[0], bounds)
    key, subkey = jax.random.split(key)
    params = model.init(subkey, obs)
    optim = get_optimizer_from_cfg(optim_cfg)

    ts = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optim,
    )

    # Calculate Mean using 'where'
    # JAX handles the division by the sum of the mask automatically
    y_mean = jnp.mean(y, where=sample_mask)

    # Calculate Std using 'where'
    y_std = jnp.std(y, where=sample_mask)

    # Safety clamp to prevent division by zero if std is extremely small
    y_std = jnp.where(y_std < 1e-8, 1.0, y_std)

    # Normalize Y
    # (The padded values become garbage, but that's fine as they are masked in loss)
    y = (y - y_mean) / y_std

    # Normalize G
    if g is not None:
        g = g / y_std
    g = jnp.where(jnp.isnan(g) & non_diff_params, 0.0, g)
    g = jnp.where(jnp.isinf(g) & non_diff_params, 0.0, g)

    def train_step(carry, _):
        ts_train, key = carry

        def _gradient_helper(cur_x, p, k):
            cur_o = embedding(cur_x, bounds)
            results = model.apply(
                p,
                cur_o,
                rngs=k,
                single_forward=True,
            )
            assert isinstance(results, jax.Array)
            assert results.size == 1
            return results.flatten()[0]

        def loss_fn(params, x_batch, y_batch, g_batch, k):
            # forward pass
            k, sk = jax.random.split(k)
            sk_list = jax.random.split(sk, num_samples)

            # --- Forward / Gradient Calculation ---
            fn = jax.vmap(jax.value_and_grad(_gradient_helper), in_axes=[None, 0, None])
            full_fn = jax.vmap(fn, in_axes=[0, None, 0])
            results, grads = full_fn(x_batch, params, sk_list)
            assert isinstance(results, jax.Array) and isinstance(grads, jax.Array)
            results = results.reshape(num_samples, model_cfg.num_models)
            grads = grads.reshape(num_samples, model_cfg.num_models, num_actions)

            info = {}
            # --- 2. MASKED LOSS CALCULATION ---
            sq_error = (y_batch[:, None] - results) ** 2

            # 'where' handles the counting and dividing automatically
            mask_expanded = sample_mask[:, None]
            loss = jnp.mean(sq_error, where=mask_expanded)
            info["loss"] = loss

            # B. Gradient Loss

            grad_sq_error = (g_batch[:, None, :] - grads) ** 2

            grad_mask_expanded = sample_mask[:, None, None]
            grad_mask_expanded = grad_mask_expanded & (~non_diff_params[None, None, :])
            grad_loss = jnp.mean(grad_sq_error, where=grad_mask_expanded)

            info["grad_loss"] = grad_loss
            loss = loss + grad_loss

            info["full_loss"] = loss
            return loss, info

        key, subkey = jax.random.split(key)
        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            ts_train.params, x, y, g, subkey
        )

        # Sanitize gradients just in case
        grads = jax.tree_util.tree_map(lambda x: jnp.nan_to_num(x), grads)

        ts_train = ts_train.apply_gradients(grads=grads)
        grad_norm = optax.global_norm(grads)
        info["grad_norm"] = grad_norm

        return (ts_train, key), info

    (final_ts, _), stacked_info = jax.lax.scan(
        train_step,
        (ts, key),
        length=optim_cfg.total_steps,
    )
    return final_ts, stacked_info
