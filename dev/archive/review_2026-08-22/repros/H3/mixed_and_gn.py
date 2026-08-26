"""H3: (a) ops combining two objects with different-but-broadcastable stacks -- documented or rejected?
(b) GaussNewtonModel on a C-stacked frame vs per-element models.
(c) backend geometry inner on stacked tangents (corewise_dot collapses stacks)."""
import itertools, traceback
import numpy as np
import t3toolbox as t3t
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.frame_variations_format as bvf
import t3toolbox.manifold as t3m
import t3toolbox.fitting as fit
import t3toolbox.corewise as cw
import t3toolbox.backend.geometry as bgeo
from t3toolbox import safety

np.random.seed(0)
shape, nn, rr = (5, 6, 7), (2, 3, 4), (1, 2, 3, 1)
d = 3


def idxs(S): return list(itertools.product(*[range(c) for c in S]))
def slice_t3(x, c): return t3.TuckerTensorTrain(tuple(B[c] for B in x.tucker_cores), tuple(G[c] for G in x.tt_cores))
def slice_frame(f, c): return bvf.T3Frame(*[tuple(A[c] for A in fam) for fam in f.data])
def slice_vars(v, kc): return bvf.T3Variations(*[tuple(A[kc] for A in fam) for fam in v.data])
def slice_tan(t, k, c): return t3m.T3Tangent(slice_frame(t.frame, c), slice_vars(t.variations, k + c))


def attempt(name, fn):
    try:
        r = fn()
        print('%-60s -> %s' % (name, r))
        return r
    except Exception as e:
        print('%-60s -> RAISES %s: %s' % (name, type(e).__name__, (str(e).splitlines() or [''])[0][:150]))
        return None


print('=== (a) mixed stacks on TuckerTensorTrain ===')
x3 = t3.TuckerTensorTrain.randn(shape, nn, rr, stack_shape=(3,))
x0 = t3.TuckerTensorTrain.randn(shape, nn, rr)
x1 = t3.TuckerTensorTrain.randn(shape, nn, rr, stack_shape=(1,))
x23 = t3.TuckerTensorTrain.randn(shape, nn, rr, stack_shape=(2, 3))
attempt('x3 + x0', lambda: (x3 + x0).stack_shape)
attempt('x3 + x1', lambda: (x3 + x1).stack_shape)
attempt('x23 + x3', lambda: (x23 + x3).stack_shape)
r = attempt('x3 * x0 (hadamard)', lambda: (x3 * x0).stack_shape)
r = attempt('x3 * x1 (hadamard)', lambda: (x3 * x1).stack_shape)
attempt('x3.inner(x0)', lambda: np.shape(x3.inner(x0)))
attempt('x3.inner(x0, use_orthogonalization=False)', lambda: np.shape(x3.inner(x0, use_orthogonalization=False)))
attempt('x3.inner(x0.to_dense())', lambda: np.shape(x3.inner(x0.to_dense())))
attempt('x3 * x0.to_dense()', lambda: np.shape(x3 * x0.to_dense()))
attempt('x3.t3m(x0, form_then_round)', lambda: x3.t3m(x0, method='form_then_round').stack_shape)
attempt('x3.t3m(x0, swap)', lambda: x3.t3m(x0, method='swap').stack_shape)
attempt('x3.t3m(x0, inplace_fused)', lambda: x3.t3m(x0, method='inplace_fused').stack_shape)
attempt('concatenate([x3.segment(0,1), x0.segment(1,3)])', lambda: t3.TuckerTensorTrain.concatenate([x3.segment(0, 1), x0.segment(1, 3)]).stack_shape)
attempt('x3.allclose(x0)', lambda: np.shape(x3.allclose(x0)) if hasattr(x3, 'allclose') else 'n/a')
# correctness of those that pass
for name, fn, ref in [
    ('x3.inner(x0) values', lambda: x3.inner(x0), lambda: np.array([slice_t3(x3, (i,)).inner(x0) for i in range(3)])),
    ('x3.inner(x0, orth=False) values', lambda: x3.inner(x0, use_orthogonalization=False), lambda: np.array([slice_t3(x3, (i,)).inner(x0) for i in range(3)])),
    ('x3 * x0.to_dense() values', lambda: x3 * x0.to_dense(), lambda: np.stack([slice_t3(x3, (i,)).to_dense() * x0.to_dense() for i in range(3)])),
    ('x3.inner(x0.to_dense()) values', lambda: x3.inner(x0.to_dense()), lambda: np.array([slice_t3(x3, (i,)).inner(x0.to_dense()) for i in range(3)])),
]:
    try:
        got = np.asarray(fn()); want = ref()
        print('%-60s -> shape %s  match=%s' % (name, got.shape, got.shape == want.shape and np.allclose(got, want)))
    except Exception as e:
        print('%-60s -> RAISES %s' % (name, type(e).__name__))

print('\n=== (a2) mixed stacks on tangents ===')
frame3 = bvf.T3Frame.random_orthogonal(shape, nn, rr, stack_shape=(3,))
frame0 = bvf.T3Frame.random_orthogonal(shape, nn, rr)
v3 = t3m.MANIFOLD.randn(frame3)
v0 = t3m.MANIFOLD.randn(frame0)
vk3 = t3m.MANIFOLD.randn(frame3, stack_shape=(2,))
attempt('T3Tangent(frame3, vars K=(2,)+C=(3,)).stacks', lambda: (vk3.frame_stack_shape, vk3.tangent_stack_shape))
attempt('T3Tangent(frame0, vars C=(3,))  [K=(3,)]', lambda: (t3m.T3Tangent(frame0, v3.variations).frame_stack_shape, t3m.T3Tangent(frame0, v3.variations).tangent_stack_shape))
attempt('T3Tangent(frame3, vars C=())  [frame stacked, vars not]', lambda: t3m.T3Tangent(frame3, v0.variations).stack_shape)
attempt('T3Tangent(frame3, vars stack=(1,3))', lambda: t3m.T3Tangent(frame3, bvf.T3Variations.randn(frame3.variation_shapes, (1, 3))).tangent_stack_shape)
attempt('T3Tangent(frame3, vars stack=(3,1)) [C not suffix]', lambda: t3m.T3Tangent(frame3, bvf.T3Variations.randn(frame3.variation_shapes, (3, 1))).tangent_stack_shape)
attempt('v3 + vk3 (K=() + K=(2,), same frame)', lambda: (v3 + vk3).tangent_stack_shape)
attempt('vk3 + v3', lambda: (vk3 + v3).tangent_stack_shape)
attempt('v3.corewise_inner(vk3)', lambda: np.shape(v3.corewise_inner(vk3)))
attempt('MANIFOLD.inner(v3, vk3)', lambda: np.shape(t3m.MANIFOLD.inner(v3, vk3)))
attempt('vk3.allclose(v3)', lambda: np.shape(vk3.allclose(v3)))
attempt('stack_tangents([v3, vk3]) (mixed K)', lambda: t3m.T3Tangent.stack_tangents([v3, vk3]).tangent_stack_shape)
attempt('stack_tangents([v3, v3*2]) ', lambda: t3m.T3Tangent.stack_tangents([v3, v3 * 2]).tangent_stack_shape)
attempt('stack_tangents([v3, v0]) (diff frames)', lambda: t3m.T3Tangent.stack_tangents([v3, v0]).tangent_stack_shape)
attempt('stack_frame([v0, v0*2])', lambda: (t3m.T3Tangent.stack_frame([v0, v0 * 2]).frame_stack_shape))
attempt('stack_frame([vk3_elem0, v0]) (K mismatch)', lambda: t3m.T3Tangent.stack_frame([slice_tan(vk3, (), (0,)), v0]).frame_stack_shape)
# frame-weights K+C
fw3 = bvf.T3FrameWeights.from_t3weights(t3.T3Weights.from_t3svd(x3.t3svd()[0]))
attempt('check_fw_pair(frame3, fw(C=3))', lambda: bvf.check_fw_pair(bvf.t3_orthogonal_representations(x3.t3svd()[0])[0], fw3))

print('\n=== (a3) probe with W and a mismatched-stack ww tuple ===')
ww_mixed = (np.random.randn(4, 5), np.random.randn(6), np.random.randn(4, 7))
attempt('x0.probe(ww with mixed W per mode)', lambda: [z.shape for z in x0.probe(ww_mixed)])
attempt('x0.apply(ww with mixed W per mode)', lambda: np.shape(x0.apply(ww_mixed)))

print('\n=== (b) GaussNewtonModel on a C-stacked frame vs per-element ===')
for C in [(3,), (2, 3)]:
    x = t3.TuckerTensorTrain.randn(shape, nn, rr, stack_shape=C)
    W = 4
    ww = tuple(np.random.randn(W, N) for N in shape)
    b = np.random.randn(*((W,) + C))
    for kindname in ['apply', 'probe', 'entries']:
        try:
            if kindname == 'apply':
                m = fit.apply_model(t3m.MANIFOLD, x, ww, b)
                ms = [fit.apply_model(t3m.MANIFOLD, slice_t3(x, c), ww, b[(slice(None),) + c]) for c in idxs(C)]
            elif kindname == 'entries':
                index = np.stack([np.random.randint(0, N, size=(W,)) for N in shape])
                m = fit.entries_model(t3m.MANIFOLD, x, index, b)
                ms = [fit.entries_model(t3m.MANIFOLD, slice_t3(x, c), index, b[(slice(None),) + c]) for c in idxs(C)]
            else:
                zz = tuple(np.random.randn(*((W,) + C + (N,))) for N in shape)
                m = fit.probe_model(t3m.MANIFOLD, x, ww, zz)
                ms = [fit.probe_model(t3m.MANIFOLD, slice_t3(x, c), ww, tuple(z[(slice(None),) + c] for z in zz)) for c in idxs(C)]
            obj = np.asarray(m.objective_value)
            objs = np.array([float(mm.objective_value) for mm in ms]).reshape(C)
            print('%s C=%s objective shape %s match=%s' % (kindname, C, obj.shape, obj.shape == C and np.allclose(obj, objs)))
            g = m.gradient
            GD = g.to_dense()
            ok = all(np.allclose(GD[c], ms[i].gradient.to_dense()) for i, c in enumerate(idxs(C)))
            print('   gradient stacks frame=%s tangent=%s per-elem match=%s' % (g.frame_stack_shape, g.tangent_stack_shape, ok))
            p = t3m.MANIFOLD.randn(m.frame)
            q = np.asarray(m.gn_quadratic(p))
            qs = np.array([float(ms[i].gn_quadratic(slice_tan(p, (), c))) for i, c in enumerate(idxs(C))]).reshape(C)
            print('   gn_quadratic shape %s match=%s' % (q.shape, q.shape == C and np.allclose(q, qs)))
            ev = np.asarray(m.evaluate(p))
            evs = np.array([float(ms[i].evaluate(slice_tan(p, (), c))) for i, c in enumerate(idxs(C))]).reshape(C)
            print('   evaluate shape %s match=%s' % (ev.shape, ev.shape == C and np.allclose(ev, evs)))
            Hp = m.gn_hessian(p).to_dense()
            ok = all(np.allclose(Hp[c], ms[i].gn_hessian(slice_tan(p, (), c)).to_dense()) for i, c in enumerate(idxs(C)))
            print('   gn_hessian per-elem match=%s' % ok)
            J = m.jacobian(p)
            if kindname == 'probe':
                ok = all(np.allclose(J[mm][(slice(None),) + c], ms[i].jacobian(slice_tan(p, (), c))[mm]) for i, c in enumerate(idxs(C)) for mm in range(d))
            else:
                ok = all(np.allclose(J[(slice(None),) + c], ms[i].jacobian(slice_tan(p, (), c))) for i, c in enumerate(idxs(C)))
            print('   jacobian per-elem match=%s' % ok)
            # K-stacked trial tangent on a stacked model
            try:
                pk = t3m.MANIFOLD.randn(m.frame, stack_shape=(2,))
                qk = np.asarray(m.gn_quadratic(pk))
                qks = np.array([[float(ms[i].gn_quadratic(slice_tan(pk, (k,), c))) for i, c in enumerate(idxs(C))] for k in range(2)]).reshape((2,) + C)
                print('   gn_quadratic(K=(2,)) shape %s match=%s' % (qk.shape, qk.shape == (2,) + C and np.allclose(qk, qks)))
                Hpk = m.gn_hessian(pk)
                print('   gn_hessian(K=(2,)) stacks frame=%s tangent=%s' % (Hpk.frame_stack_shape, Hpk.tangent_stack_shape))
            except Exception as e:
                print('   K-stacked trial: RAISES %s: %s' % (type(e).__name__, (str(e).splitlines() or [''])[0][:200]))
        except Exception as e:
            print('%s C=%s RAISES %s: %s' % (kindname, C, type(e).__name__, (str(e).splitlines() or [''])[0][:200]))
            traceback.print_exc()

print('\n=== (c) backend geometry inner on stacked tangents ===')
geom = bgeo.ManifoldGeometryOps()
x = t3.TuckerTensorTrain.randn(shape, nn, rr, stack_shape=(3,))
fr = geom.frame(x.data)
v = t3m.MANIFOLD.randn(bvf.T3Frame(*fr))
print('ragged ManifoldGeometryOps.inner(v,v) on C=(3,): shape', np.shape(geom.inner(v.variations.data, v.variations.data)),
      ' per-element corewise_inner shape', np.shape(v.corewise_inner(v)))
print('ragged point_norm_sq on C=(3,): shape', np.shape(geom.point_norm_sq(x.t3svd()[0].data)))
print('ragged stack_shape(x):', geom.stack_shape(x.data))
import t3toolbox.uniform_tucker_tensor_train as ut3
ux = ut3.UniformTuckerTensorTrain.from_t3(x)
ugeom = bgeo.UniformManifoldGeometryOps.from_point(ux.data, None)
ufr = ugeom.frame((ux.tucker_supercore, ux.tt_supercore))
import t3toolbox.backend.ufv_operations as ufvo
vv = ugeom.project(ufr, (np.random.randn(*ufr[0].shape), np.random.randn(*ufr[2].shape)))
print('uniform UniformManifoldGeometryOps.inner on C=(3,): shape', np.shape(ugeom.inner(vv, vv)))
print('uniform point_norm_sq on C=(3,): shape', np.shape(ugeom.point_norm_sq((ux.tucker_supercore, ux.tt_supercore))))
