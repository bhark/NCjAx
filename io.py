import jax.numpy as jnp
from .structure import State
from .config import Config

def _visible_channels(config: Config) -> jnp.ndarray:
    return jnp.arange(config.io_channels, dtype=jnp.int32)

def _normalize_channels(C: int, channels):
    if channels is None:
        return jnp.arange(C, dtype=jnp.int32)
    idx = jnp.asarray(channels, dtype=jnp.int32)
    return idx

def _clip_region(
    N: int, x: int, y: int, w: int, h: int
) -> tuple[int, int, int, int]:
    x0 = max(0, min(x, N))
    y0 = max(0, min(y, N))
    x1 = max(x0, min(x + w, N))
    y1 = max(y0, min(y + h, N))
    return x0, y0, x1 - x0, y1 - y0

def write(
    state: State,
    config: Config,
    *,
    channels = None,
    region: tuple[int, int, int, int],
    value,
    mode: str = 'set'
) -> State:
    ''' write into a rectangular region over selected channels '''
    C, N = config.C, config.grid_size
    ch = _normalize_channels(C, channels)

    x, y, w, h = _clip_region(N, *region)
    if w == 0 or h == 0 or ch.size == 0:
        return state

    g = state.grid
    if isinstance(value, (int, float)):
        val = jnp.full((ch.size, h, w), value, dtype=g.dtype)
    else:
        val = jnp.asarray(value, dtype=g.dtype)
        if val.ndim == 1 and (w, h) == (1, 1):
            if val.shape != (ch.size,):
                raise ValueError(f'value shape {val.shape} must be (len(ch),) for 1x1 writing')
            val = val.reshape(ch.size, 1, 1)
        elif val.shape != (ch.size, h, w):
            raise ValueError(f'value shape {val.shape} must be (len(ch), h, w) = {(ch.size, h, w)}')

    idx = (ch, slice(y, y + h), slice(x, x + w))
    if mode == 'set':
        new_grid = g.at[idx].set(val)
    elif mode == 'add':
        new_grid = g.at[idx].set(val)
    else:
        raise ValueError('mode must be set or add')

    return State(grid=new_grid)

def read(
    state: State,
    config: Config,
    *,
    channels = None,
    region: tuple[int, int, int, int] | None = None,
    reduction: str = 'mean'
) -> int:
    ''' read from selected channels and region '''
    C, N = config.C, config.grid_size
    ch = _normalize_channels(C, channels)

    if region is None:
        x, y, w, h = 0, 0, N, N
    else:
        x, y, w, h = _clip_region(N, *region)

    if w == 0 or h == 0 or ch.size == 0:
        if reduction == 'raw':
            return jnp.zeros((ch.size, 0, 0), dtype=state.grid.dtype)
        return jnp.zeros((ch.size,), dtype=state.grid.dtype)

    sub = state.grid[ch[:, None, None], y:y + h, x:x + w]

    if reduction == 'raw':
        return sub
    if reduction == 'mean':
        return sub.mean(axis=(1,2))
    if reduction == 'max':
        return sub.max(axis=(1,2))
    raise ValueError('reduction must be raw, mean or max')

# -- convenience and quality of life --

def _default_input_region(
    config: Config, 
    *, 
    size: tuple[int, int] = (1, 1),
    margin: int = 3
) -> tuple[int, int, int, int]:
    ''' upper left corner with some margin '''
    w, h = size
    x = margin
    y = margin
    return (x, y, w, h)

def _default_output_region(
    config: Config,
    *,
    size: tuple[int, int] = (1, 1),
    margin: int = 3
) -> tuple[int, int, int, int]:
    w, h = size
    N = config.grid_size
    x = max(0, N - margin - w)
    y = max(0, N - margin - h)
    return (x, y, w, h)

def inform(
    state: State,
    config: Config,
    *,
    value,
    channels = None,
    size: tuple[int, int] = (1, 1),
    margin: int = 3,
    mode: str = 'set'
) -> State:
    '''
    convenience write for input
    writes to visible channels of default input region
    '''
    if channels is None:
        channels = _visible_channels(config)
    region = _default_input_region(config, size=size, margin=margin)
    return write(state, config, channels=channels, region=region, value=value, mode=mode)

def extract(
    state: State,
    config: Config,
    *,
    channels = None,
    size: tuple[int, int] = (1, 1),
    margin: int = 3,
    reduction: str = 'mean'
):
    '''
    convenience read for output
    reads from visible channel of default region
    '''
    if channels is None:
        channels = _visible_channels(config)
    region = _default_output_region(config, size=size, margin=margin)
    return read(state, config, channels=channels, region=region, reduction=reduction)