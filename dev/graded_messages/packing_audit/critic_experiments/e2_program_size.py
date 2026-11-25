# E2: does the uniform engine's traced program grow with d today, and where? (the plan's §6 gate)
import numpy as np
import jax, jax.numpy as jnp
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.uniform_manifold as ut3m
import t3toolbox.backend.sampling_variables as sv
import t3toolbox.backend.ut3_operations as uo

def count(jaxpr):
    n = 0
    for e in jaxpr.eqns:
        n += 1
        for p in e.params.values():
            for sub in (p if isinstance(p, (tuple, list)) else (p,)):
                if hasattr(sub, 'jaxpr') and hasattr(sub.jaxpr, 'eqns'):
                    n += count(sub.jaxpr)
                elif hasattr(sub, 'eqns'):
                    n += count(sub)
    return n

def setup(d, kind, order):
    rng = np.random.default_rng(0)
    shape = tuple(4 + (i % 3) for i in range(d))
    x = t3.TuckerTensorTrain.randn(shape, (2,) * d, (1,) + (2,) * (d - 1) + (1,))
    ux = ut3.UniformTuckerTensorTrain.from_t3(x, N=7)
    fr = ut3m.UNIFORM_MANIFOLD.frame(ux).data
    F = tuple(n + 1 for n in order)
    ww = uo.pack_vectors([rng.standard_normal(F + (3, n)) for n in shape], 7, len_P=len(order))
    if kind == 'probe':
        y = uo.pack_vectors([rng.standard_normal(F + (3, n)) for n in shape], 7)
    else:
        y = rng.standard_normal(F + (3,))
    return fr, ww, y

for kind in ('probe', 'apply'):
    for want in (('z',), ('xi', 'mu', 'nu'), ('f',), ('f', 'g'), ('r_block_sumsq',)):
        out = []
        for d in (3, 5):
            fr, ww, y = setup(d, kind, (1,))
            shape, masks = fr[4], fr[5]
            def f(D, U, L, R, w, yy):
                return sv.sampling_variables(kind, (D, U, L, R, shape, masks), w, want, order=(1,), y=yy)
            jp = jax.make_jaxpr(f)(*[jnp.asarray(a) for a in fr[:4]], jnp.asarray(ww), jnp.asarray(y))
            hlo = jax.jit(f).lower(*[jnp.asarray(a) for a in fr[:4]], jnp.asarray(ww), jnp.asarray(y)).as_text()
            out.append((count(jp.jaxpr), len(hlo)))
        print(kind, want, 'eqns d=3,5:', out[0][0], out[1][0], '  hlo chars:', out[0][1], out[1][1])
