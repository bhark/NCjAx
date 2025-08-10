import jax
import jax.numpy as jnp
from jax import lax, nn

from .config import Config
from .structure import State, Params

# -- perception --

def _reflect_pad(grid: jnp.ndarray) -> jnp.ndarray:
    ''' pad spatial dims by 1 with reflect '''
    return jnp.pad(grid, ((0, 0), (1, 1), (1, 1)), mode='reflect')

def _neighbor_taps(grid: jnp.ndarray) -> tuple[jnp.ndarray, ...]:
    '''
    return 9 taps for each channel (center + neigbors)
    each tap is (C, grid_size, grid_size), uses reflect padding
    '''
    p = _reflect_pad(grid)
    center     = p[:, 1:-1, 1:-1]
    up         = p[:, 0:-2, 1:-1]
    down       = p[:, 2:  , 1:-1]
    left       = p[:, 1:-1, 0:-2]
    right      = p[:, 1:-1, 2:  ]
    up_left    = p[:, 0:-2, 0:-2]
    up_right   = p[:, 0:-2, 2:  ]
    down_left  = p[:, 2:  , 0:-2]
    down_right = p[:, 2:  , 2:  ]
    return (center, up, down, left, right, up_left, up_right, down_left, down_right)

def perception(state: State, config: Config) -> jnp.ndarray:
    '''
    compute perception features for each cell
    '''
    g = state.grid

    # identity + laplacian
    if config.perception == 'id_lap':
        center, up, down, left, right, *_ = _neighbor_taps(g)
        lap = (up + down + left + right) - 4.0 * center
        feats = jnp.concatenate([center, lap], axis=0)
        return feats.astype(config.dtype)

    # raw9
    taps = _neighbor_taps(g)
    feats = jnp.concatenate(list(taps), axis=0)
    return feats.astype(config.dtype)


# -- mlp --

def mlp(feats: jnp.ndarray, params: Params, config: Config) -> jnp.ndarray:
    '''
    the main update rule shared by all cells,
    and applied to each cell's perception vector

    outputs a delta
    '''
    h = jnp.tensordot(feats, params.w1, axes=((0,), (0,)))
    h = h + params.b1
    h = nn.relu(h)

    out = jnp.tensordot(h, params.w2, axes=((2,), (0,)))
    out = out + params.b2

    delta = jnp.moveaxis(out, -1, 0)
    delta = jnp.tanh(delta) # prevent explosion
    return delta.astype(config.dtype)

# -- fire-rate --

def _apply_fire_rate(
    key: jax.Array,
    updated: jnp.ndarray,
    old: jnp.ndarray,
    p: float,
) -> tuple[jnp.ndarray, jax.Array]:
    '''
    per-cell stochastic update
    '''
    if p >= 1.0:
        return updated, key # as-is
    if p <= 0.0:
        return old, key # nothing fires
    
    _, N, _ = updated.shape
    key, sub = jax.random.split(key)
    mask = jax.random.bernoulli(sub, p, shape=(1, N, N))
    mixed = jnp.where(mask, updated, old)
    return mixed, key

def _apply_read_only(updated: jnp.ndarray, old: jnp.ndarray, config: Config) -> jnp.ndarray:
    '''  flag channels are immutable, and info channel is immutable at input cells '''
    in_idx = config.idx_in_flag
    out_idx = config.idx_out_flag
    info_idx = config.idx_info

    updated = updated.at[in_idx, :, :].set(old[in_idx, :, :])
    updated = updated.at[out_idx, :, :].set(old[out_idx, :, :])

    in_mask = old[in_idx, :, :] > 0.5
    info_protected = jnp.where(in_mask, old[info_idx, :, :], updated[info_idx, :, :])
    updated = updated.at[info_idx, :, :].set(info_protected)
    return updated

# -- step and rollout --
def step(
    state: State, 
    params: Params, 
    key: jax.Array, 
    config: Config,
) -> tuple[State, jax.Array]:
    '''
    a single CA tick
    '''
    feats = perception(state, config)
    delta = mlp(feats, params, config)
    updated = state.grid + delta
    mixed, key = _apply_fire_rate(key, updated, state.grid, config.fire_rate)
    return State(grid=mixed), key

def rollout(
    state: State,
    params: Params,
    key: jax.Array,
    K: int,
    config: Config
) -> tuple[State, jax.Array]:
    '''
    unroll k steps
    '''
    def _body(carry, _):
        st, k = carry
        st2, k2 = step(st, params, k, config)
        return (st2, k2), None

    (final_state, final_key), _ = lax.scan(_body, (state, key), xs=None, length=K)
    return final_state, final_key