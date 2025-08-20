import jax
import jax.numpy as jnp
from jax import lax
import optax

from .config import Config
from .structure import Params, init_state
from . import io, core

def pretrain(
    key: jax.Array,
    params: Params,
    config: Config,
    *,
    steps: int = 500,
    batch_size: int = 64,
    K: int | None = None,
    lr: float = 1e-3,
    eps: float = 5e-2,
    clip_by_global_norm: float = 1.0,
    grid_clip_penalty: float = 1e-3,
) -> tuple[Params, jax.Array]:
    '''
    curriculum pretraining: both outputs learn the mean of inputs.
    also regularizes the grid to stay within [-1, 1] (excl. flag channels)
    '''
    K = config.k_default if K is None else int(K)
    m = int(config.num_output_nodes)
    k = int(config.num_input_nodes)
    dtype = config.dtype

    def _target_mean(x: jnp.ndarray) -> jnp.ndarray:
        mu = jnp.mean(x)
        return jnp.full((m,), mu, dtype=x.dtype)

    def _forward_once(p: Params, k_: jax.Array, x_: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        st0 = init_state(k_, config)
        st1 = io.inform(st0, config, value=x_, mode="set")
        st2, _ = core.rollout(st1, p, k_, K, config)
        y = io.extract(st2, config)
        return y, st2.grid

    def _clip_reg(grid: jnp.ndarray, bound: float = 1.0) -> jnp.ndarray:
        C = grid.shape[0]
        mask = jnp.ones((C, 1, 1), dtype=grid.dtype)
        mask = mask.at[config.idx_in_flag].set(0.0)
        mask = mask.at[config.idx_out_flag].set(0.0)
        g = grid * mask  # no boolean indexing

        excess = jnp.maximum(0.0, jnp.abs(g) - bound)
        return jnp.mean(excess * excess)

    def loss_fn(p: Params, keys: jax.Array, xs: jnp.ndarray) -> jnp.ndarray:
        def _one(k_, x_):
            y, G = _forward_once(p, k_, x_)
            y_true = _target_mean(x_)
            mse = jnp.mean((y - y_true) ** 2)
            reg = _clip_reg(G)
            return mse + grid_clip_penalty * reg
        return jnp.mean(jax.vmap(_one)(keys, xs))

    tx = optax.chain(
        optax.clip_by_global_norm(clip_by_global_norm),
        optax.adam(lr),
    )
    opt_state = tx.init(params)

    def _step(carry, _):
        p, opt_state, k0 = carry
        k0, kx, kk = jax.random.split(k0, 3)
        xs = jax.random.uniform(kx, (batch_size, k), minval=-1.0, maxval=1.0, dtype=dtype)
        ks = jax.random.split(kk, batch_size)
        loss, grads = jax.value_and_grad(loss_fn)(p, ks, xs)
        grads = jax.tree_util.tree_map(jnp.nan_to_num, grads)
        updates, opt_state = tx.update(grads, opt_state, p)
        p = optax.apply_updates(p, updates)
        return (p, opt_state, k0), None

    (p_out, opt_state, key_out), _ = lax.scan(
        _step, (params, opt_state, key), xs=None, length=int(steps)
    )

    kx, kk = jax.random.split(key_out)
    xs_eval = jax.random.uniform(kx, (batch_size, k), minval=-1.0, maxval=1.0, dtype=dtype)
    ks_eval = jax.random.split(kk, batch_size)

    def _eval_one(k_, x_):
        y, _ = _forward_once(p_out, k_, x_)
        y_true = _target_mean(x_)
        err = jnp.abs(y - y_true)
        return jnp.mean(err < eps)

    final_acc = jnp.mean(jax.vmap(_eval_one)(ks_eval, xs_eval))
    return p_out, final_acc