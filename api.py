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
        self.key = key if key is not None else jax.random.PRNGKey(420)

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
            self._jit_inform_nodes = jax.jit(
                lambda st, vals, mode: io_ops.inform_nodes(st, self.config, vals, mode=mode),
                static_argnames=('mode',)
            )
            self._jit_extract_nodes = jax.jit(
                lambda st: io_ops.extract_nodes(st, self.config)
            )
        else:
            self._jit_rollout = lambda st, pr, key, K: core_rollout(st, pr, key, K, self.config)
            self._jit_inform_nodes = lambda st, vals, mode: io_ops.inform_nodes(st, self.config, vals, mode=mode)
            self._jit_extract_nodes = lambda st: io_ops.extract_nodes(st, self.config)



    def step(self, K: int | None = None, *, key: jax.Array | None = None) -> State:
        '''
        process K ticks, return updated State and update internal state
        '''
        K = self.config.k_default if K is None else int(K)
        key = self._use_or_split_key(key)

        new_state, new_key = self._jit_rollout(self._state, self._params, key, K=K)
        self._state = new_state
        self.key = new_key
        return self._state

    def rollout(self, K: int, *, key: jax.Array | None = None) -> State:
        '''
        like step(), but simply returns final state without mutating internal state
        '''
        key = self._use_or_split_key(key)
        out_state, _ = self._jit_rollout(self._state, self._params, key, K=int(K))
        return out_state

    def process(self, inputs, K: int | None = None, *, mode: str = 'set', key: jax.Array | None = None) -> jnp.ndarray:
        '''
        complete processing pipeline
        1. writes input
        2. processes K ticks (while mutating internal state)
        3. returns output
        '''
        self.inform(inputs, mode=mode)
        self.step(K=K, key=key)
        return self.extract()

    # -- i/o helpers to go slightly less insane --

    def write_inputs(self, values, *, mode: str = 'set') -> State:
        self._state = self._jit_inform_nodes(self._state, self.config, values, mode=mode)
        return self._state

    def read_outputs(self) -> jnp.ndarray:
        return self._jit_extract_nodes(self._state, self.config)

    def inform(self, value, *, mode: str = 'set') -> State:
        self._state = io_ops.inform(self._state, self.config, value=value, mode=mode)
        return self._state

    def extract(self):
        return io_ops.extract(self._state, self.config)


    # -- state and params management --

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

    def clone(
        self,
        *,
        copy_state: bool = True,
        copy_params: bool = True,
        key: jax.Array | None = None
    ) -> 'NCASubstrate':
        '''
        create a clone of this substrate
        '''
        if key is None:
            self.key, new_key = jax.random.split(self.key)
        else:
            new_key = key

        new_state = self._state if copy_state else None
        new_params = self._params if copy_params else None

        return NCASubstrate(
            config=self.config,
            key=new_key,
            params=new_params,
            state=new_state,
            jit_compile=hasattr(self, '_jit_rollout')
        )

    def explain(self) -> str:
        C, N = self.config.C, self.config.grid_size
        pcount = num_params(self._params)
        if self.config.perception == 'learned3x3':
            perc = f"learned3x3(K={self.config.conv_features}, padding={self.config.padding})"
            in_dim = self.config.input_feats_per_cell
        else:
            perc = f"{self.config.perception} (F={self.config.F}, padding={self.config.padding})"
            in_dim = self.config.input_feats_per_cell
        return (
            f'NCASubstrate:\n'
            f'  Grid: C={C} (info=1, hidden={self.config.hidden_channels}, id=2), N={N}\n'
            f'  Perception: {perc} → input_dim={in_dim}\n'
            f'  MLP: hidden={self.config.hidden}, params={pcount}\n'
            f'  Dynamics: fire_rate={self.config.fire_rate}, K_default={self.config.k_default}\n'
            f'  I/O nodes: inputs={self.config.num_input_nodes}, outputs={self.config.num_output_nodes}\n'
            f'  Dtype: {self.config.dtype.name}'
        )

    
    def _use_or_split_key(self, key: jax.Array | None) -> jax.Array:
        if key is not None:
            return key
        self.key, sub = jax.random.split(self.key)
        return sub