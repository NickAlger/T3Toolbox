# E2b: which primitives grow with d in the uniform engine's trace (want=('xi',) and ('mu',))
import collections
import numpy as np
import jax, jax.numpy as jnp
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.uniform_manifold as ut3m
import t3toolbox.backend.sampling_variables as sv
import t3toolbox.backend.ut3_operations as uo

def hist(jaxpr, depth=0, acc=None):
    acc = collections.Counter() if acc is None else acc
    for e in jaxpr.eqns:
        acc[(depth, e.primitive.name)] += 1
        for p in e.params.values():
            for sub in (p if isinstance(p, (tuple, list)) else (p,)):
                if hasattr(sub, 'jaxpr') and hasattr(sub.jaxpr, 'eqns'):
                    hist(sub.jaxpr, depth + 1, acc)
    return acc

res = {}
for d in (3, 5):
    rng = np.random.default_rng(0)
    shape = tuple(4 + (i % 3) for i in range(d))
    x = t3.TuckerTensorTrain.randn(shape, (2,) * d, (1,) + (2,) * (d - 1) + (1,))
    ux = ut3.UniformTuckerTensorTrain.from_t3(x, N=7)
    fr = ut3m.UNIFORM_MANIFOLD.frame(ux).data
    ww = uo.pack_vectors([rng.standard_normal((3, n)) for n in shape], 7)
    def f(D, U, L, R, w):
        return sv.sampling_variables('probe', (D, U, L, R, fr[4], fr[5]), w, ('xi',))
    jp = jax.make_jaxpr(f)(*[jnp.asarray(a) for a in fr[:4]], jnp.asarray(ww))
    res[d] = hist(jp.jaxpr)
keys = sorted(set(res[3]) | set(res[5]))
for k in keys:
    if res[3][k] != res[5][k]:
        print(k, res[3][k], res[5][k])
