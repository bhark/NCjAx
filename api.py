import jax
import jax.numpy as jnp

from .config import Config
from .structure import State, Params, init_state, init_params, num_params
from .core import rollout as core_rollout, step as core_step
from . import io as io_ops

class NCASubstrate:
    def __init__(
        self,
        config: Config,
        *,
        key: jax.Array | None = None,
        params: Params | None = None,
        state: State | None = None,
        jit_compile: bool = True
    ) -> None:
        self.config = config
        self.key = key if key is not None else jax.random.key(420)

        # init params/state if missing
        if params is None:
            k1, self.key = jax.random.split(self.key)
            params = init_params(k1, config)
        if state is None:
            k2, self.key = jax.random.split(self.key)
            state = init_state(k2, config)

        self._params = params
        self._state = state

        # prep jitted callables (close over conf)
        # just jit hot paths
        if jit_compile:
            self._jit_rollout = jax.jit(
                lambda st, pr, key, K: core_rollout(st, pr, key, K, self.config),
                static_argnames=('K',),
            )
            self._jit_step = jax.jit(
                lambda st, pr, key: core_step(st, pr, key, self.config)
            )
        else:
            self._jit_rollout = lambda st, pr, key, K: core_rollout(st, pr, key, K, self.config)
            self._jit_step = lambda st, pr, key: core_step(st, pr, key, self.config)

    def step(self, K: int | None = None, *, key: jax.Array | None = None) -> State:
        '''
        process K ticks, return updated State and update internal state
        '''
        K = self.config.k_default if K is None else int(K)
        key = self._use_or_split_key(key)

        new_state, new_key = self._jit_rollout(self._state, self._params, key, K)
        self._state = new_state
        self.key = new_key
        return self._state

    def rollout(self, K: int, *, key: jax.Array | None = None) -> State:
        '''
        like step(), but simply returns final state without mutating internal state
        '''
        key = self._use_or_split_key(key)
        out_state, _ = self._jit_rollout(self._state, self._params, key, int(K))
        return out_state

    # -- i/o helpers to go slightly less insane --

    def inform(
        self,
        value,
        *,
        channels = None,
        size: tuple[int, int] = (1, 1),
        margin: int = 3,
        mode: str = 'set'
    ) -> State:
        '''
        convenient input writing
        (defaults to upper left of the substrate)
        '''
        self._state = io_ops.inform(
            self._state, self.config, value=value, channels=channels, size=size, margin=margin, mode=mode
        )
        return self._state

    def extract(
        self,
        *,
        channels=None,
        size: tuple[int, int] = (1, 1),
        margin: int = 3,
        reduction: str = 'mean'
    ):
        '''
        convenient output reading
        (defaults to lower right of the substrate)
        '''
        return io_ops.extract(
            self._state, self.config, channels=channels, size=size, margin=margin, reduction=reduction
        )

    ### maybe add more low-level access here at some point


    # -- state and params management --
    
    def feed(self, input_data, *, key=None):
        """
        the universal step function:
        1. Write input to input patch
        2. process K steps 
        3. read and return output
        
        state is maintained between calls
        """
        key = self._use_or_split_key(key)
        
        if input_data is not None:
            self.inform(
                value=input_data,
                channels=range(self.config.io_channels),  # visible channels
                size=(1, 1),  # or could be configurable
                margin=3,
                mode='set'
            )
        
        self.step(self.config.k_default, key=key)
        
        # Read output
        output = self.extract(
            channels=range(self.config.io_channels),  # visible channels
            size=(1, 1),
            margin=3,
            reduction='mean'
        )
        
        return output

    @property
    def state(self) -> State:
        return self._state

    def reset(self, *, key: jax.Array | None = None, state: State | None = None) -> State:
        ''' reset grid to zeros or a provided state '''
        if state is not None:
            self._state = state
            return self._state
        key = self._use_or_split_key(key)
        self._state = init_state(key, self.config)
        return self._state

    def get_params(self) -> Params:
        return self._params

    def set_params(self, params: Params) -> None:
        self._params = params


    # -- utils --

    def explain(self) -> str:
        C, N = self.config.C, self.config.grid_size
        F = self.config.F
        pcount = num_params(self._params)
        return (
            f'NCASubstrate:\n'
            f'  Grid: C={C} (visible={self.config.io_channels}, hidden={self.config.hidden_channels}), N={N}\n'
            f'  Perception: {self.config.perception} (F={F}) → input_dim={C*F}\n'
            f'  MLP: hidden={self.config.hidden}, params={pcount}\n'
            f'  Dynamics: fire_rate={self.config.fire_rate}, K_default={self.config.k_default}\n'
            f'  Dtype: {self.config.dtype.name}'
        )

    
    def _use_or_split_key(self, key: jax.Array | None) -> jax.Array:
        if key is not None:
            return key
        self.key, sub = jax.random.split(self.key)
        return sub