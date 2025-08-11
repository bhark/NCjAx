import jax
import jax.numpy as jnp
from . import core, io
from .structure import State, Params
from .config import Config

def process(
    state: State,
    params: Params,
    key: jax.Array,
    inputs,
    *,
    config: Config,
    K: int | None = None,
    mode: str = 'set'
):
    ''' write inputs, run k ticks, read outputs '''
    st1 = io.inform(state, config, value=inputs, mode=mode)
    k = config.k_default if K is None else int(K)
    st2, key2 = core.rollout(st1, params, key, k, config)
    out = io.extract(st2, config)
    return out, st2, key2

def make_fns(config: Config):
    step = jax.jit(lambda st, pr, key: core.step(st, pr, key, config))
    rollout = jax.jit(lambda st, pr, key, K: core.rollout(st, pr, key, K, config),
                      static_argnames=("K",))
    inform = jax.jit(lambda st, vals, mode: io.inform(st, config, value=vals, mode=mode),
                     static_argnames=("mode",))
    extract = jax.jit(lambda st: io.extract(st, config))
    process_jit = jax.jit(
        lambda st, pr, key, x, K, mode: process(st, pr, key, x, config=config, K=K, mode=mode),
        static_argnames=("K","mode"),
    )
    return step, rollout, inform, extract, process_jit