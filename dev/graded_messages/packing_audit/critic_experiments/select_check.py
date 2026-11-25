import numpy as np, time
print('numpy', np.__version__)
x32 = np.ones((3, 1000, 64), dtype=np.float32); x32[..., 40:] = np.inf
mask = (np.arange(64) < 40)                                  # host bool, broadcast over the leading axes
for name, zero in (('python 0', 0), ('python 0.0', 0.0), ('np.float64(0)', np.float64(0)), ('0-d float64 array', np.zeros(())),
                   ('x.dtype.type(0)', x32.dtype.type(0))):
    print('  where(mask, float32, %-18s) -> %s' % (name, np.where(mask, x32, zero).dtype))
print('  multiply: nan in padding?', bool(np.isnan(x32 * mask).any()), '| select:', bool(np.isnan(np.where(mask, x32, x32.dtype.type(0))).any()))
x = np.random.default_rng(0).standard_normal((3, 20000, 64))
t = time.perf_counter(); [x * mask for _ in range(20)]; a = time.perf_counter() - t
t = time.perf_counter(); [np.where(mask, x, 0.0) for _ in range(20)]; b = time.perf_counter() - t
print('  eager numpy, 3.8M entries: multiply %.1f ms, where %.1f ms' % (a / 20 * 1e3, b / 20 * 1e3))
try:
    import jax, jax.numpy as jnp
    print('jax', jax.__version__)
    m = jnp.asarray(mask)
    f_mul = lambda y: jnp.sum(jnp.sin(y * mask) / jnp.where(m, 1.0, 0.0))        # a downstream 0/0 on the padding
    f_sel = lambda y: jnp.sum(jnp.where(mask, jnp.sin(jnp.where(mask, y, 0.0)), 0.0))
    y = jnp.ones(64).at[40:].set(jnp.inf)
    g = jax.grad(lambda y: jnp.sum(jnp.where(mask, y, 0.0) ** 2))(y)
    print('  grad through select, inf in padding: finite', bool(jnp.isfinite(g).all()), '| padding gradient', float(jnp.abs(g[40:]).max()))
    g2 = jax.grad(lambda y: jnp.sum((y * mask) ** 2))(y)
    print('  grad through multiply, inf in padding: finite', bool(jnp.isfinite(g2).all()))
    sel = jax.jit(lambda y: jnp.where(mask, y, 0.0)); mul = jax.jit(lambda y: y * mask)
    big = jnp.asarray(x, dtype=jnp.float32); sel(big).block_until_ready(); mul(big).block_until_ready()
    t = time.perf_counter(); [sel(big).block_until_ready() for _ in range(50)]; b = time.perf_counter() - t
    t = time.perf_counter(); [mul(big).block_until_ready() for _ in range(50)]; a = time.perf_counter() - t
    print('  jit, 3.8M entries: multiply %.2f ms, where %.2f ms' % (a / 50 * 1e3, b / 50 * 1e3))
except ImportError:
    print('no jax in this env')
