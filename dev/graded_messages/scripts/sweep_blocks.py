"""What does it cost to run a SWEEP in sample blocks smaller than it needs? (jax, jit, CPU)
A left sweep mu over d sites, either on all W samples at once or inside a sequential loop over W blocks.
Needs jax. Timings are machine-specific: indicative only. Usage: python3 sweep_blocks.py [<repo>]"""
import time, math
import numpy as np
import jax, jax.numpy as jnp

def run(d, r, n, W, blocks):
    rng = np.random.default_rng(0)
    f32 = lambda *s: jnp.asarray(rng.standard_normal(s), jnp.float32)
    G, xi = f32(d, r, n, r) / math.sqrt(r * n), f32(d, W, n)

    def sweep(G, xi):                                   # xi: (d, w, n) -> mu on bonds 1..d: (d, w, r)
        def step(mu, x):
            Gi, xii = x
            mu = jnp.einsum('wa,aib,wi->wb', mu, Gi, xii)
            return mu, mu
        return jax.lax.scan(step, jnp.ones((xi.shape[1], r), jnp.float32) / r, (G, xi))[1]

    def blocked(b):
        def f(G, xi):
            out = jax.lax.map(lambda x: sweep(G, x), jnp.moveaxis(xi.reshape(d, W // b, b, n), 1, 0))   # (nb, d, b, r)
            return jnp.moveaxis(out, 0, 1).reshape(d, W, r)
        return f

    out_mb = d * W * r * 4 / 2**20
    print('d=%d r=%d n=%d W=%d   (the sweep\'s output is %.0f MB)' % (d, r, n, W, out_mb))
    ref = None
    for b in blocks:
        fn = sweep if b == W else blocked(b)
        jf = jax.jit(fn)
        mem = jf.lower(G, xi).compile().memory_analysis().temp_size_in_bytes / 2**20
        o = jf(G, xi); o.block_until_ready()
        ts = []
        for _ in range(7):
            t0 = time.perf_counter(); jf(G, xi).block_until_ready(); ts.append(time.perf_counter() - t0)
        if ref is None: ref = o
        print('   block %6d   scratch %7.1f MB   time %7.1f ms   diff %.0e' % (b, mem, 1e3 * min(ts), float(jnp.abs(o - ref).max() / jnp.abs(ref).max())))

run(8, 32, 16, 16384, (16384, 4096, 1024, 256, 64, 16))
run(8, 128, 32, 4096, (4096, 1024, 256, 64, 16))
run(8, 8, 8, 16384, (16384, 4096, 1024, 256, 64, 16))
