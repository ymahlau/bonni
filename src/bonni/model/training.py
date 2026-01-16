from functools import partial
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
    model_cfg: MLPEnsembleConfig,
    optim_cfg: OptimConfig,
    num_embedding_channels: int,
) -> tuple[TrainState, dict[str, jax.Array]]:
    assert x.ndim == 2
    num_samples, num_actions = x.shape[0], x.shape[1]
    assert bounds.ndim == 2
    assert bounds.shape[1] == 2
    assert bounds.shape[0] == num_actions
    assert y.ndim == 1
    assert y.shape[0] == num_samples
    assert g.ndim == 2
    assert g.shape == x.shape
    
    num_actions = x.shape[1]
    num_samples = x.shape[0]
    
    embedding = SinCosActionEmbedding(num_channels=num_embedding_channels)
    
    model = MLPEnsemble(model_cfg)
    obs = embedding(x[0], bounds)
    key, subkey = jax.random.split(key)
    params = model.init(subkey, obs)
    optim = get_optimizer_from_cfg(optim_cfg)
    
    ts = TrainState.create(
        apply_fn=model.apply,
        params=(params),
        tx=optim,
    )
    
    # center ys
    y_mean = y.mean()
    y_std = y.std()
    y = y - y_mean
    if y.size > 1:
        y = y / y_std
        if g is not None:
            g = g / y_std
    
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
            
            if g_batch is not None:
                # calculate gradient and result in one go
                fn = jax.vmap(jax.value_and_grad(_gradient_helper), in_axes=[None, 0, None])
                full_fn = jax.vmap(fn, in_axes=[0, None, 0])
                results, grads = full_fn(x_batch, params, sk_list)
                assert isinstance(results, jax.Array) and isinstance(grads, jax.Array)
                results = results.reshape(num_samples, model_cfg.num_models)
                grads = grads.reshape(num_samples, model_cfg.num_models, num_actions)
            else:
                # only calculate forward results, not gradient
                fn = jax.vmap(_gradient_helper, in_axes=[None, 0, None])
                full_fn = jax.vmap(fn, in_axes=[0, None, 0])
                results = full_fn(x_batch, params, sk_list)
                results = results.reshape(num_samples, model_cfg.num_models)
                grads = None
            info = {}
            
            loss = jnp.mean((y_batch[:, None] - results) ** 2)
            info['loss'] = loss
            
            if grads is not None:
                grad_loss = jnp.mean((g_batch[:, None, :] - grads)**2)
                info['grad_loss'] = grad_loss
                loss = loss + grad_loss
            
            info['full_loss'] = loss
            return loss, info
        
        # forward pass and gradients
        key, subkey = jax.random.split(key)
        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(ts_train.params, x, y, g, subkey)
        grads = jax.tree_util.tree_map(lambda x: jnp.where(jnp.isnan(x), 0, x), grads)
        ts_train = ts_train.apply_gradients(grads=grads)
        grad_norm = optax.global_norm(grads)
        info["grad_norm"] = grad_norm
        info["loss"] = loss
        return (ts_train, key), info
    
    (final_ts, _), stacked_info = jax.lax.scan(
        train_step,
        (ts, key),
        length=optim_cfg.total_steps,
    )
    return final_ts, stacked_info
