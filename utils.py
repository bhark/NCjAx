import jax.numpy as jnp

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
    c = (N - 1) // 2
    # (dx, dy) around center; order is irrelevant as long as it's fixed
    offsets = jnp.array(
        [(0, -2), (0, 2), (2, 0), (-2, 0), (2, 2), (-2, -2), (2, -2), (-2, 2)],
        dtype=jnp.int32
    )

    idx = jnp.arange(m, dtype=jnp.int32) % offsets.shape[0]  # (m,)
    sel = offsets[idx]                                       # (m, 2)

    center = jnp.array([c, c], dtype=jnp.int32)              # (2,)
    pos = center[None, :] + sel                              # (m, 2)
    pos = jnp.clip(pos, 0, N - 1)                            # keep in-bounds
    return pos  # (m,2) as (x, y)