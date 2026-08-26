"""R10: jit every weighted op (ragged + uniform) and compare with eager numpy."""
import numpy as np, jax, jax.numpy as jnp
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.frame_variations_format as bvf
import t3toolbox.manifold as t3m
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.uniform_frame_variations_format as ubvf
import t3toolbox.uniform_manifold as ut3m
from t3toolbox.backend import ut3_conversions, ufv_conversions

def rel(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float); return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))
def J(x): return jax.tree_util.tree_map(jnp.asarray, x)
np.random.seed(0); rng = np.random.default_rng(2)
x = t3.TuckerTensorTrain.randn((5, 6, 7), (2, 3, 3), (1, 2, 3, 1), stack_shape=(2,))
y = t3.TuckerTensorTrain.randn((5, 6, 7), (2, 3, 3), (1, 2, 3, 1), stack_shape=(2,))
W = t3.T3Weights.from_t3svd(x); W2 = t3.T3Weights.from_t3svd(y)
frame, _ = bvf.t3_orthogonal_representations(x); v = t3m.COREWISE.randn(frame, stack_shape=(3,)); u = t3m.COREWISE.randn(frame, stack_shape=(3,))
FW = bvf.T3FrameWeights.from_t3weights(W)
res = []
def check(name, f, *args, tol=1e-4):
    eager = f(*args)
    try:
        jitted = jax.jit(f)(*J(args))
        e = jax.tree_util.tree_map(lambda a, b: rel(a, b), jax.tree_util.tree_leaves(jitted), jax.tree_util.tree_leaves(eager))
        worst = max(e) if e else 0.0
        res.append((name, 'OK' if worst < tol else 'MISMATCH %.2e' % worst))
    except Exception as ex:
        res.append((name, 'RAISED %s: %s' % (type(ex).__name__, str(ex).splitlines()[0][:90])))

print('--- ragged ---')
check('t3_absorb_weights', lambda a, b: t3.t3_absorb_weights(a, b).data, x, W)
check('t3_weighted_norm', t3.t3_weighted_norm, x, W)
check('t3_weighted_inner', t3.t3_weighted_inner, x, W, y, W2)
check('T3Weights.reciprocal', lambda w: w.reciprocal().data, W)
check('T3Weights.sqrt', lambda w: w.sqrt().data, W)
check('T3Weights.reverse', lambda w: w.reverse().data, W)
check('T3Weights.concatenate', lambda a, b: a.concatenate(b).data, W, W2)
check('T3Weights.kronecker', lambda a, b: a.kronecker(b).data, W, W2)
check('T3Weights.stack(unstack)', lambda w: t3.T3Weights.stack(w.unstack()).data, W)
check('T3Weights.from_t3svd', lambda a: t3.T3Weights.from_t3svd(a).data, x, tol=1e-3)
check('T3Tangent.absorb_weights', lambda t, w: t.absorb_weights(w).variations.data, v, FW)
check('T3Tangent.weighted_norm', lambda t, w: t.weighted_norm(w), v, FW)
check('T3Tangent.weighted_inner', lambda t, s, w: t.weighted_inner(s, w), v, u, FW)
check('fv_absorb_weights', lambda t, w: bvf.fv_absorb_weights(t, w).data, v.variations, FW)
check('T3FrameWeights.reciprocal/sqrt/reverse', lambda w: (w.reciprocal().data, w.sqrt().data, w.reverse().data), FW)
check('T3FrameWeights.concatenate/kronecker', lambda a: (a.concatenate(a).data, a.kronecker(a).data), FW)
check('T3FrameWeights.stack(unstack)', lambda w: bvf.T3FrameWeights.stack(w.unstack()).data, FW)
check('T3FrameWeights.from_t3weights', lambda w: bvf.T3FrameWeights.from_t3weights(w).data, W)
check('check_fw_pair (structural, inside jit)', lambda f, w: (bvf.check_fw_pair(f, w), f.data)[1], frame, FW)

print('--- uniform ---')
ux = ut3.UniformTuckerTensorTrain.from_t3(x, n=4, r=4); uy = ut3.UniformTuckerTensorTrain.from_t3(y, n=4, r=4)
UW = ut3.UT3Weights.from_t3weights(W, n=4, r=4); UW2 = ut3.UT3Weights.from_t3weights(W2, n=4, r=4)
uxt = ut3.UniformTuckerTensorTrain.from_t3(x)   # tight padding: the documented GK route needs it
uframe, _ = ubvf.ut3_orthogonal_representations(uxt)
uv = ut3m.UNIFORM_COREWISE.randn(uframe, stack_shape=(3,)); uu = ut3m.UNIFORM_COREWISE.randn(uframe, stack_shape=(3,))
UFW = ubvf.UT3FrameWeights.from_ut3weights(ut3.UT3Weights.from_ut3svd(uxt))
check('ut3_absorb_weights', lambda a, b: ut3.ut3_absorb_weights(a, b).supercores, ux, UW)
check('ut3_weighted_norm', ut3.ut3_weighted_norm, ux, UW)
check('ut3_weighted_norm(no orth)', lambda a, b: ut3.ut3_weighted_norm(a, b, use_orthogonalization=False), ux, UW)
check('ut3_weighted_inner', ut3.ut3_weighted_inner, ux, UW, uy, UW2)
check('UT3Weights.reciprocal', lambda w: w.reciprocal().supercores, UW)
check('UT3Weights.sqrt', lambda w: w.sqrt().supercores, UW)
check('UT3Weights.concatenate', lambda a, b: a.concatenate(b).supercores, UW, UW2)
check('UT3Weights.kronecker', lambda a, b: a.kronecker(b).supercores, UW, UW2)
check('UT3Weights.from_ut3svd', lambda a: ut3.UT3Weights.from_ut3svd(a).supercores, ux, tol=1e-3)
check('UT3Weights.to_t3weights (stacked tree)', lambda w: w.to_t3weights(), UW)
check('UT3Weights.from_t3weights', lambda w: ut3.UT3Weights.from_t3weights(w, n=4, r=4).supercores, W)
check('UT3Weights.is_consistent_with inside jit', lambda a, b: (b.is_consistent_with(a), a.supercores)[1], ux, UW)
check('UT3Tangent.absorb_weights', lambda t, w: t.absorb_weights(w).variations.supercores if hasattr(t.variations, 'supercores') else t.absorb_weights(w).variations.data[:2], uv, UFW)
check('UT3Tangent.weighted_norm', lambda t, w: t.weighted_norm(w), uv, UFW)
check('UT3Tangent.weighted_inner', lambda t, s, w: t.weighted_inner(s, w), uv, uu, UFW)
check('ufv_absorb_weights', lambda t, w: ubvf.ufv_absorb_weights(t, w).data[:2], uv.variations, UFW)
check('UT3FrameWeights.reciprocal/sqrt', lambda w: (w.reciprocal().supercores, w.sqrt().supercores), UFW)
check('UT3FrameWeights.concatenate/kronecker', lambda a: (a.concatenate(a).supercores, a.kronecker(a).supercores), UFW)
check('UT3FrameWeights.from_ut3weights', lambda w: ubvf.UT3FrameWeights.from_ut3weights(w).supercores, UW)
check('UT3FrameWeights.to_t3frameweights (stacked tree)', lambda w: w.to_t3frameweights(), UFW)
check('UT3FrameWeights.from_t3frameweights', lambda w: ubvf.UT3FrameWeights.from_t3frameweights(w).supercores, FW)
check('check_ufw_pair inside jit', lambda f, w: (ubvf.check_ufw_pair(f, w), f.supercores if hasattr(f, 'supercores') else f.data[:4])[1], uframe, UFW)
# grad through reciprocal padding guard (the documented reason for the double-where)
g = jax.grad(lambda w: ut3.ut3_weighted_norm(ux, w).sum())(J(UW))
res.append(('grad of ut3_weighted_norm wrt UT3Weights finite', 'OK' if all(np.isfinite(np.asarray(l)).all() for l in jax.tree_util.tree_leaves(g)) else 'NAN'))
g2 = jax.grad(lambda w: ut3.ut3_weighted_norm(ux, w.reciprocal()).sum())(J(UW))
res.append(('grad through UT3Weights.reciprocal finite', 'OK' if all(np.isfinite(np.asarray(l)).all() for l in jax.tree_util.tree_leaves(g2)) else 'NAN'))
for n, r in res: print('%-50s %s' % (n, r))
