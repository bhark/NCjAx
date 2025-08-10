from dataclasses import dataclass
import jax
import jax.numpy as jnp
from jax.nn.initializers import glorot_uniform
from .config import Config

# -- helpers for node layout --

def _circle_layout(N: int, k: int, radius_ratio: float = 0.35) -> jnp.ndarray:
    ''' return integer (x,y) positions on a circle, shape (k,2) '''
    c = (N - 1) / 2.0
    r = max(1.0, radius_ratio * N)
    idx = jnp.arange(k, dtype=jnp.float32)
    theta = 2.0 * jnp.pi * (idx / float(k))
    xs = jnp.clip(jnp.round(c + r * jnp.cos(theta)), 0, N - 1).astype(jnp.int32)
    ys = jnp.clip(jnp.round(c + r * jnp.sin(theta)), 0, N - 1).astype(jnp.int32)
    return jnp.stack([xs, ys], axis=-1)  # (k,2)

def _default_output_nodes(N: int, m: int) -> jnp.ndarray:
    ''' place m outputs around center (on a small cross) '''
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

# -- state --

@jax.tree_util.register_pytree_node_class
@dataclass(slots=True, frozen=True)
class State:
    grid: jnp.ndarray # shape = (C, grid_size, grid_size)

    def tree_flatten(self):
        return ((self.grid,), None)
    
    @classmethod
    def tree_unflatten(cls, aux, children):
        (grid,) = children
        return cls(grid=grid)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.grid.shape

@jax.tree_util.register_pytree_node_class
@dataclass(slots=True, frozen=True)
class Params:
    '''
    MLP is two dense layers applied channel-wise
    '''

    w1: jnp.ndarray # (in_dim, hidden)
    b1: jnp.ndarray # (hidden,)
    w2: jnp.ndarray # (hidden, out_dim)
    b2: jnp.ndarray # (out_dim,)

    def tree_flatten(self):
        return ((self.w1, self.b1, self.w2, self.b2), None)

    @classmethod
    def tree_unflatten(cls, aux, children):
        w1, b1, w2, b2 = children
        return cls(w1=w1, b1=b1, w2=w2, b2=b2)

    @property
    def sizes(self) -> tuple[tuple[int, int], tuple[int], tuple[int, int], tuple[int]]:
        return self.w1.shape, self.b1.shape, self.w2.shape, self.b2.shape


# -- initializers --

def init_state(key: jax.Array, config: Config) -> State:
    ''' init grid state '''
    C, N = config.C, config.grid_size
    g = jnp.zeros((C, N, N), dtype=config.dtype)
    in_idx = config.idx_in_flag
    out_idx = config.idx_out_flag

    # input nodes: circle layout
    inp_xy = _circle_layout(N, config.num_input_nodes)
    g = g.at[in_idx, inp_xy[:,1], inp_xy[:,0]].set(1.0)

    # output nodes near center
    out_xy = _default_output_nodes(N, config.num_output_nodes)
    g = g.at[out_idx, out_xy[:,1], out_xy[:,0]].set(1.0)

    return State(grid=g)

def init_params(key: jax.Array, config: Config) -> Params:
    ''' init params '''
    in_dim = config.input_feats_per_cell
    hidden = config.hidden
    out_dim = config.C

    k1, k2 = jax.random.split(key, 2)
    W1_init = glorot_uniform()
    W2_init = glorot_uniform()

    w1 = W1_init(k1, (in_dim, hidden), dtype=config.dtype)
    b1 = jnp.zeros((hidden,), dtype=config.dtype)
    w2 = W2_init(k2, (hidden, out_dim), dtype=config.dtype)
    b2 = jnp.zeros((out_dim,), dtype=config.dtype)

    return Params(w1=w1, b1=b1, w2=w2, b2=b2)

# -- utils --

def num_params(p: Params) -> int:
    return sum(int(jnp.size(arr)) for arr in (p.w1, p.b1, p.w2, p.b2))