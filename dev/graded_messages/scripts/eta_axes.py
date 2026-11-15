"""The same operation (eta jet combine) with each of two batch axes (order, sites) either sequential or
vectorized, at three problem sizes. XLA scratch and CPU time. Usage: python3 eta_axes.py <repo>"""
import sys, time, math
import numpy as np
sys.path.append(sys.argv[1])
import jax, jax.numpy as jnp
import t3toolbox.backend.sampling_derivatives as sd

def make(d, r, n, order, W):
    P = order + 1
    rng = np.random.default_rng(0)
    f32 = lambda *s: jnp.asarray(rng.standard_normal(s), jnp.float32)
    G, mu, nu = f32(d, r, n, r) / math.sqrt(r), f32(d, P, W, r), f32(d, P, W, r)
    trs = jnp.asarray([[[math.comb(t, a) if a + b == t else 0 for b in range(P)] for a in range(P)] for t in range(P)], jnp.float32)
    inv_fact = jnp.asarray([1.0 / math.factorial(k) for k in range(P)], jnp.float32)
    tcol = jnp.arange(P, dtype=jnp.float32)

    def shift(x, axis):                                   # (sh x)^(t) = t x^(t-1) along `axis`
        pad = [(0, 0)] * x.ndim; pad[axis] = (1, 0)
        y = jnp.pad(x, pad)[tuple(slice(0, P) if a == axis else slice(None) for a in range(x.ndim))]
        return y * tcol.reshape([P if a == axis else 1 for a in range(x.ndim)])

    def seq_order_seq_sites(G, mu, nu):                   # the library
        return sd.compute_eta_jets(G, mu, nu)

    def seq_order_vec_sites(G, mu, nu):                   # scan over order; all sites in each einsum
        def step(carry, x):
            eta, sh_nu = carry
            mu_r, w = x
            MG = jnp.einsum('dwa,daib->dwib', w * mu_r, G)
            return (eta + jnp.einsum('dwib,dtwb->dtwi', MG, sh_nu), shift(sh_nu, 1)), None
        (eta, _), _ = jax.lax.scan(step, (jnp.zeros((d, P, W, n), jnp.float32), nu), (jnp.moveaxis(mu, 1, 0), inv_fact))
        return eta

    def vec_order_seq_sites(G, mu, nu):                   # map over sites; all orders in each einsum
        def site(x):
            Gi, mui, nui = x
            return jnp.einsum('trs,rwib,swb->twi', trs, jnp.einsum('rwa,aib->rwib', mui, Gi), nui)
        return jax.lax.map(site, (G, mu, nu))

    def vec_order_vec_sites(G, mu, nu):                   # everything in two einsums
        return jnp.einsum('trs,drwib,dswb->dtwi', trs, jnp.einsum('drwa,daib->drwib', mu, G), nu)

    plans = (('order sequential, sites sequential (library)', seq_order_seq_sites),
             ('order sequential, sites vectorized', seq_order_vec_sites),
             ('order vectorized, sites sequential', vec_order_seq_sites),
             ('order vectorized, sites vectorized', vec_order_vec_sites))
    return plans, (G, mu, nu)

for d, r, n, order, W in ((8, 8, 8, 5, 256), (8, 32, 16, 5, 4096), (8, 96, 32, 5, 1024), (24, 16, 8, 2, 2048)):
    plans, args = make(d, r, n, order, W)
    print('d=%d r=%d n=%d order=%d W=%d' % (d, r, n, order, W))
    ref = None
    for name, fn in plans:
        jf = jax.jit(fn)
        mem = jf.lower(*args).compile().memory_analysis().temp_size_in_bytes / 2**20
        out = jf(*args); out.block_until_ready()
        ts = []
        for _ in range(7):
            t0 = time.perf_counter(); jf(*args).block_until_ready(); ts.append(time.perf_counter() - t0)
        if ref is None: ref = out
        print('   %-48s scratch %8.1f MB   time %8.2f ms   diff %.0e' % (name, mem, 1e3 * min(ts), float(jnp.abs(out - ref).max() / jnp.abs(ref).max())))
