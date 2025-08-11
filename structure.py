from dataclasses import dataclass
import jax
import jax.numpy as jnp
from jax.nn.initializers import glorot_uniform
from .config import Config
from .utils import _circle_layout, _default_output_nodes

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

    # learned conv frontend
    conv_w: jnp.ndarray
    conv_b: jnp.ndarray

    w1: jnp.ndarray # (in_dim, hidden)
    b1: jnp.ndarray # (hidden,)
    w2: jnp.ndarray # (hidden, out_dim)
    b2: jnp.ndarray # (out_dim,)

    gain: jnp.ndarray # (config.C,)

    def tree_flatten(self):
        return ((self.conv_w, self.conv_b, self.w1, self.b1, self.w2, self.b2, self.gain), None)

    @classmethod
    def tree_unflatten(cls, aux, children):
        conv_w, conv_b, w1, b1, w2, b2, gain = children
        return cls(conv_w=conv_w, conv_b=conv_b, w1=w1, b1=b1, w2=w2, b2=b2, gain=gain)

    @property
    def sizes(self):
        return (self.conv_w.shape, self.conv_b.shape, self.w1.shape, self.b1.shape, self.w2.shape, self.b2.shape, self.gain.shape)


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

    k1, k2, k3, k4 = jax.random.split(key, 4)
    W1_init = glorot_uniform()
    W2_init = glorot_uniform()

    # learned 3x3 conv filters if requested
    if config.perception == 'learned3x3':
        kf = config.conv_features
        conv_w = jax.random.normal(k1, (kf, config.C, 3, 3), dtype=config.dtype) * jnp.sqrt(2.0 / (9*config.C + kf))
        conv_b = jnp.zeros((kf,), dtype=config.dtype)
    else:
        conv_w = jnp.zeros((0,), dtype=config.dtype)
        conv_b = jnp.zeros((0,), dtype=config.dtype)

    w1 = W1_init(k3, (in_dim, hidden), dtype=config.dtype)
    b1 = jnp.zeros((hidden,), dtype=config.dtype)
    w2 = W2_init(k4, (hidden, out_dim), dtype=config.dtype)
    b2 = jnp.zeros((out_dim,), dtype=config.dtype)

    gain = jnp.full((out_dim,), 0.05, dtype=config.dtype)
    gain = gain.at[config.idx_in_flag].set(0.0)
    gain = gain.at[config.idx_out_flag].set(0.0)

    return Params(conv_w=conv_w, conv_b=conv_b, w1=w1, b1=b1, w2=w2, b2=b2, gain=gain)

# -- utils --

def num_params(p: Params) -> int:
    return sum(int(jnp.size(arr)) for arr in (p.conv_w, p.conv_b, p.w1, p.b1, p.w2, p.b2, p.gain))