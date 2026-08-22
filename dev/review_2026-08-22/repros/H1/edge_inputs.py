"""H1: edge inputs that reach the aux / hash path: array-valued regularizer strength, numpy-int shapes,
list shapes, jax masks."""
import numpy as np
import jax, jax.numpy as jnp
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.manifold as t3m
import t3toolbox.uniform_manifold as um
import t3toolbox.optimizers as opt
import t3toolbox.fitting as fitting
from t3toolbox.backend import regularization as reg, geometry as bgeo

np.random.seed(0)
SH, TK, TT = (6, 6, 5), (2, 3, 2), (1, 2, 2, 1)
A = t3.TuckerTensorTrain.randn(SH, TK, TT)
ww = [np.random.randn(30, n) for n in SH]; b = A.apply(ww)
x0 = t3.TuckerTensorTrain.randn(SH, TK, TT)

print('--- IdentityRegularizer strength types under use_jit=True (newton_cg, manifold, apply)')
for label, lam in (('float', 0.01), ('np.float64', np.float64(0.01)), ('np.array 0-d', np.array(0.01)),
                   ('jnp scalar', jnp.float32(0.01)), ('np.float32', np.float32(0.01))):
    try:
        x, st = opt.newton_cg(t3m.MANIFOLD, 'apply', ww, b, x0, regularizer=reg.IdentityRegularizer(lam), max_newton=2, use_jit=True)
        print('    %-14s OK  final obj %.6f' % (label, st['history'][-1]['objective']))
    except Exception as e:
        print('    %-14s raises %s: %s' % (label, type(e).__name__, str(e)[:110]))
print('    eager (no jit) with jnp scalar:')
try:
    x, st = opt.newton_cg(t3m.MANIFOLD, 'apply', ww, b, x0, regularizer=reg.IdentityRegularizer(jnp.float32(0.01)), max_newton=2)
    print('    OK  final obj %.6f' % st['history'][-1]['objective'])
except Exception as e:
    print('    raises %s: %s' % (type(e).__name__, str(e)[:110]))
print('    mc_sgd use_jit with np.array 0-d strength:')
try:
    opt.mc_sgd(t3m.MANIFOLD, 'apply', ww, b, x0, np.random.default_rng(0), batch=10, max_iter=2, regularizer=reg.IdentityRegularizer(np.array(0.01)), use_jit=True)
    print('    OK')
except Exception as e:
    print('    raises %s: %s' % (type(e).__name__, str(e)[:110]))

print('--- UniformTuckerTensorTrain.shape entry types')
ux = ut3.UniformTuckerTensorTrain.from_t3(x0)
for label, shp in (('tuple(np.int64)', tuple(np.int64(s) for s in SH)), ('list', list(SH)), ('np.array', np.array(SH))):
    try:
        u = ut3.UniformTuckerTensorTrain(ux.tucker_supercore, ux.tt_supercore, shp, ux.masks)
        print('    %-16s constructs; ' % label, end='')
    except Exception as e:
        print('    %-16s construct raises %s: %s' % (label, type(e).__name__, str(e)[:80])); continue
    try:
        jax.jit(lambda u: u.norm())(u.to_jax()); print('jit(norm) OK; ', end='')
    except Exception as e:
        print('jit(norm) raises %s: %s; ' % (type(e).__name__, str(e)[:60]), end='')
    try:
        x, st = opt.newton_cg(um.UNIFORM_MANIFOLD, 'apply', ww, b, u, max_newton=2, use_jit=True)
        print('newton_cg(use_jit) OK obj %.6f' % st['history'][-1]['objective'])
    except Exception as e:
        print('newton_cg(use_jit) raises %s: %s' % (type(e).__name__, str(e)[:120]))

print('--- jax-array masks in a UT3 (UT3Masks accepts any boolean ndarray incl. jax)')
uj = ut3.UniformTuckerTensorTrain(ux.tucker_supercore, ux.tt_supercore, ux.shape,
                                  ut3.UT3Masks(*(jnp.asarray(m) for m in ux.masks.data)))
print('    to_jax() keeps masks numpy?', type(ux.to_jax().masks.tucker_edge_mask).__name__)
try:
    uj.validate(); print('    validate accepts jax masks')
except Exception as e:
    print('    validate raises', type(e).__name__, str(e)[:80])
try:
    g = bgeo.UniformManifoldGeometryOps.from_point(uj.data); hash(g); print('    from_point+hash OK')
except Exception as e:
    print('    from_point/hash raises %s: %s' % (type(e).__name__, str(e)[:80]))
try:
    x, st = opt.newton_cg(um.UNIFORM_MANIFOLD, 'apply', ww, b, uj, max_newton=2, use_jit=True); print('    newton_cg(use_jit) OK')
except Exception as e:
    print('    newton_cg(use_jit) raises %s: %s' % (type(e).__name__, str(e)[:120]))
try:
    x, st = opt.newton_cg(um.UNIFORM_MANIFOLD, 'apply', ww, b, uj, max_newton=2); print('    newton_cg(eager) OK')
except Exception as e:
    print('    newton_cg(eager) raises %s: %s' % (type(e).__name__, str(e)[:120]))
