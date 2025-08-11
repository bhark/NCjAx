import jax.numpy as jnp
from .structure import State
from .config import Config
from .utils import _circle_layout, _default_output_nodes

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