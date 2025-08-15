import jax
import jax.numpy as jnp
from jax import lax

from .config import Config
from .structure import Params, init_state
from . import io, core

def pretrain(
    key: jax.Array,
    params: Params,
    config: Config,
    *,
    steps: int = 2000,
    batch_size: int = 64,
    K: int | None = None,
    lr: float = 1e-2,
    eps: float = 5e-2
) -> tuple[Params, jax.Array]:
    '''
    nca gets stuck in a local minima without pretraining
    trains to reproduce input-node values at output nodes
    '''
    K = config.k_default if K is None else int(K)
    m = int(config.num_output_nodes)
    k = int(config.num_input_nodes)
    dtype = config.dtype

    def _project_inputs_to_outputs(x: jnp.ndarray) -> jnp.ndarray:
        if m <= k:
            return x[:m]
        times = (m + k - 1) // k
        return jnp.tile(x, times)[:m]

    def _forward_once(p: Params, k_: jax.Array, x_: jnp.ndarray) -> jnp.ndarray:
        st0 = init_state(k_, config)
        st1 = io.inform(st0, config, value=x_, mode='set')
        st2, _ = core.rollout(st1, p, k_, K, config)
        y = io.extract(st2, config)
        return y

    def loss_fn(p: Params, keys: jax.Array, xs: jnp.ndarray) -> jnp.ndarray:
        def _one(k_, x_):
            y = _forward_once(p, k_, x_)
            y_true = _project_inputs_to_outputs(x_)
            return jnp.mean((y - y_true) ** 2)
        return jnp.mean(jax.vmap(_one)(keys, xs))

    @jax.jit
    def _step(carry, _):
        p, k0 = carry
        k0, kx, kk = jax.random.split(k0, 3)
        xs = jax.random.uniform(kx, (batch_size, k), minval=-1.0, maxval=1.0, dtype=dtype)
        ks = jax.random.split(kk, batch_size)
        loss, grads = jax.value_and_grad(loss_fn)(p, ks, xs)
        p = jax.tree_util.tree_map(lambda w, g: w - lr * g, p, grads)   # plain SGD
        return (p, k0), loss

    (p_out, key_out), _ = lax.scan(_step, (params, key), xs=None, length=int(steps))

    # --- final eval (fresh batch) ---
    key_out, kx, kk = jax.random.split(key_out, 3)
    xs_eval = jax.random.uniform(kx, (batch_size, k), minval=-1.0, maxval=1.0, dtype=dtype)
    ks_eval = jax.random.split(kk, batch_size)

    def _eval_one(k_, x_):
        y = _forward_once(p_out, k_, x_)
        y_true = _project_inputs_to_outputs(x_)
        err = jnp.abs(y - y_true)
        return jnp.mean(err < eps)

    final_acc = jnp.mean(jax.vmap(_eval_one)(ks_eval, xs_eval)).astype(dtype)
    return p_out, key_out, final_acc
