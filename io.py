import jax.numpy as jnp
from .structure import State
from .config import Config

def _circle_layout(N: int, k: int, radius_ratio: float = 0.35) -> jnp.ndarray:
    c = (N - 1) / 2.0
    r = max(1.0, radius_ratio * N)
    idx = jnp.arange(k, dtype=jnp.float32)
    theta = 2.0 * jnp.pi * (idx / float(k))
    xs = jnp.clip(jnp.round(c + r * jnp.cos(theta)), 0, N - 1).astype(jnp.int32)
    ys = jnp.clip(jnp.round(c + r * jnp.sin(theta)), 0, N - 1).astype(jnp.int32)
    return jnp.stack([xs, ys], axis=-1)

def _default_output_nodes(N: int, m: int) -> jnp.ndarray:
    c = (N - 1) // 2
    offsets = jnp.array([(0,-1),(0,1),(1,0),(-1,0),(1,1),(-1,-1),(1,-1),(-1,1)], dtype=jnp.int32)
    pos = []
    used = set()
    for i in range(m):
        dx, dy = tuple(offsets[i % offsets.shape[0]].tolist())
        x = int(jnp.clip(c + dx, 0, N - 1))
        y = int(jnp.clip(c + dy, 0, N - 1))
        if (x,y) in used:
            x, y = c, c
        used.add((x,y))
        pos.append((x,y))
    return jnp.asarray(pos, dtype=jnp.int32)

def input_positions(config: Config) -> jnp.ndarray:
    return _circle_layout(config.grid_size, config.num_input_nodes)

def output_positions(config: Config) -> jnp.ndarray:
    return _default_output_nodes(config.grid_size, config.num_output_nodes)

def inform_nodes(state: State, config: Config, values, *, mode: str = 'set') -> State:
    ''' write to info channel on input nodes '''
    g = state.grid
    info_idx = config.idx_info
    xy = input_positions(config)  # (K,2)
    K = xy.shape[0]

    if isinstance(values, (int, float)):
        val = jnp.full((K,), float(values), dtype=g.dtype)
    else:
        val = jnp.asarray(values, dtype=g.dtype).reshape(-1)
        if val.shape[0] != K:
            raise ValueError(f'values length {val.shape[0]} must equal num_input_nodes={K}')

    x = xy[:,0]
    y = xy[:,1]
    if mode == 'set':
        g = g.at[info_idx, y, x].set(val)
    elif mode == 'add':
        g = g.at[info_idx, y, x].add(val)
    else:
        raise ValueError('mode must be set or add')
    return State(grid=g)

def extract_nodes(state: State, config: Config) -> jnp.ndarray:
    ''' read info channel at output nodes '''
    g = state.grid
    info_idx = config.idx_info
    xy = output_positions(config)
    x = xy[:,0]
    y = xy[:,1]
    return g[info_idx, y, x]

# convenience wrappers
def inform(state: State, config: Config, *, value, mode: str = 'set') -> State:
    return inform_nodes(state, config, value, mode=mode)

def extract(state: State, config: Config):
    return extract_nodes(state, config)