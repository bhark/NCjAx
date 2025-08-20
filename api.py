import jax
from . import core, io
from .config import Config
from .structure import init_params as _init_params, init_state as _init_state
from .pretrain import pretrain
from dataclasses import dataclass

@dataclass(frozen=True)
class NCA:
    config: Config

    def __post_init__(self):
        c = self.config

        def _step(st, pr, key):
            return core.step(st, pr, key, c)

        def _rollout(st, pr, key, *, K=None):
            k = c.k_default if K is None else int(K)
            return core.rollout(st, pr, key, k, c)

        def _inform(st, *, value, mode):
            return io.inform(st, c, value=value, mode=mode)

        def _extract(st):
            return io.extract(st, c)

        def _process(st, pr, key, x, *, K=None, mode="set"):
            k = c.k_default if K is None else int(K)
            st1 = _inform(st, value=x, mode=mode)
            st2, key2 = _rollout(st1, pr, key, K=k)
            out = _extract(st2)
            return out, st2

        def _pretrain(pr, key, **kwargs):
            return pretrain(key=key, params=pr, config=c, **kwargs)

        object.__setattr__(self, "step",    jax.jit(_step))
        object.__setattr__(self, "rollout", jax.jit(_rollout, static_argnames=("K",)))
        object.__setattr__(self, "inform",  jax.jit(_inform,  static_argnames=("mode",)))
        object.__setattr__(self, "extract", jax.jit(_extract))
        object.__setattr__(self, "process", jax.jit(_process, static_argnames=("K","mode")))
        object.__setattr__(self, "pretrain", jax.jit(_pretrain, static_argnames=("steps","batch_size","K")))

    def init_params(self, key):
            return _init_params(key, self.config)

    def init_state(self, key):
        return _init_state(key, self.config)