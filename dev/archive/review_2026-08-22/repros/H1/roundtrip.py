"""H1: pytree round-trips for every registered node type, and the 'geometries are interchangeable' claim."""
import numpy as np
import jax, jax.numpy as jnp
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.frame_variations_format as fvf
import t3toolbox.uniform_frame_variations_format as ufvf
import t3toolbox.manifold as t3m
import t3toolbox.uniform_manifold as um
import t3toolbox.shared_geometry as sg
import t3toolbox.fitting as fitting
from t3toolbox.backend import optimizers as bopt, fitting as bfit, geometry as bgeo, sharing as bsh, uniform_fitting as ufit

np.random.seed(0)
SH, TK, TT = (6, 6, 5), (2, 3, 2), (1, 2, 2, 1)
x = t3.TuckerTensorTrain.randn(SH, TK, TT).share((0, 0, 1))
ux = ufit.uniform_minimal(ut3.UniformTuckerTensorTrain.from_t3(x), sharing=(0, 0, 1))
ww = [np.random.randn(7, n) for n in SH]; r = np.random.randn(7)

def rt(obj):
    leaves, td = jax.tree_util.tree_flatten(obj)
    return jax.tree_util.tree_unflatten(td, leaves), td

objs = {
 'TuckerTensorTrain': x, 'T3Weights': t3.T3Weights.from_t3svd(x),
 'UniformTuckerTensorTrain': ux,
 'T3Frame': t3m.MANIFOLD.frame(x), 'T3Tangent': t3m.MANIFOLD.randn(t3m.MANIFOLD.frame(x)),
 'UT3Frame': um.UNIFORM_MANIFOLD.frame(ux), 'UT3Tangent': um.UNIFORM_MANIFOLD.randn(um.UNIFORM_MANIFOLD.frame(ux)),
 'ManifoldGeometry': t3m.MANIFOLD, 'CorewiseGeometry': t3m.COREWISE,
 'SharedGeometry': sg.shared_manifold((0, 0, 1)),
 'SharedFrameData': bsh.fv_shared_frame_data(t3m.MANIFOLD.frame(x).data, bsh.validate_sharing((0, 0, 1), SH)),
 'GaussNewtonModel(ragged, shared)': fitting.apply_model(sg.shared_manifold((0, 0, 1)), x, ww, r),
 'GaussNewtonModel(uniform)': fitting.apply_model(um.UNIFORM_MANIFOLD, ux, ww, r),
 'Problem(uniform)': ufit.uniform_least_squares_problem('manifold', 'apply', ux, ww, r, sharing=(0, 0, 1)),
}
objs['LocalModel(uniform)'] = objs['Problem(uniform)'].local_model((ux.tucker_supercore, ux.tt_supercore))
for name, o in objs.items():
    if o is None: continue
    o2, td = rt(o)
    _, td2 = jax.tree_util.tree_flatten(o2)
    print('%-32s type kept=%-5s treedef equal=%-5s same-object=%s' % (name, type(o2) is type(o), td == td2, o2 is o))

print('--- the "interchangeable" claim (manifold.py:1530): a round-tripped geometry used in the fitting factories')
g2, _ = rt(t3m.MANIFOLD)
print('    roundtripped is MANIFOLD:', g2 is t3m.MANIFOLD, '; ==:', g2 == t3m.MANIFOLD)
for label, fn in (('fitting.apply_model', lambda: fitting.apply_model(g2, x, ww, r)),
                  ('shared(g2, ...)', lambda: sg.shared(g2, (0, 0, 1))),
                  ('optimizers._geometry_ops', lambda: __import__('t3toolbox.optimizers', fromlist=['x'])._geometry_ops(g2, SH))):
    try:
        fn(); print('    %-28s accepted' % label)
    except Exception as e:
        print('    %-28s raises %s: %s' % (label, type(e).__name__, str(e)[:90]))
print('    MANIFOLD.frame via roundtripped instance works:', isinstance(g2.frame(x), fvf.T3Frame))

print('--- SharedGeometry subclass through jit')
class SubShared(sg.SharedGeometry): pass
try:
    jax.jit(lambda g: 0)(SubShared(t3m.MANIFOLD, (0, 0, 1))); print('    subclass as jit arg: OK')
except Exception as e:
    print('    subclass as jit arg raises', type(e).__name__, ':', str(e)[:100])
g3, _ = rt(sg.shared_manifold((0, 0, 1)))
print('    plain SharedGeometry roundtrip ==:', g3 == sg.shared_manifold((0, 0, 1)))
