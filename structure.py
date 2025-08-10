from dataclasses import dataclass
import jax
import jax.numpy as jnp
from jax.nn.initializers import glorot_uniform, zeros
from .config import Config

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
    grid = jnp.zeros(config.grid_shape, dtype=config.dtype)
    return State(grid=grid)

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