"""R10: the C-vs-K+C batching rule at EVERY weighted entry point, ragged and uniform."""
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.frame_variations_format as bvf
import t3toolbox.manifold as t3m
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.uniform_frame_variations_format as ubvf
import t3toolbox.uniform_manifold as ut3m
from t3toolbox.backend import fv_operations, ufv_operations, utv_operations

np.random.seed(0)
C, K = (2,), (3,)
x = t3.TuckerTensorTrain.randn((5, 6, 7), (2, 3, 3), (1, 2, 3, 1), stack_shape=C)
frame, _ = bvf.t3_orthogonal_representations(x)
v = t3m.COREWISE.randn(frame, stack_shape=K)
u = t3m.COREWISE.randn(frame, stack_shape=K)
W_C = bvf.T3FrameWeights.from_t3weights(t3.T3Weights.from_t3svd(x))             # stack C
tile = lambda w: np.broadcast_to(w, K + w.shape).copy()
W_KC = bvf.T3FrameWeights(*[tuple(tile(w) for w in fam) for fam in W_C.data])     # stack K+C (tiled)
W_0 = bvf.T3FrameWeights(*[tuple(w[0] for w in fam) for fam in W_C.data])          # stack ()
assert W_C.stack_shape == C and W_KC.stack_shape == K + C and W_0.stack_shape == ()

def attempt(label, fn):
    try:
        out = fn(); shp = getattr(out, 'shape', None) or getattr(out, 'stack_shape', None)
        print('%-58s ACCEPTED  (result stack/shape %s)' % (label, shp))
    except Exception as e:
        print('%-58s REJECTED  %s: %s' % (label, type(e).__name__, (str(e).splitlines() or ['<no message>'])[0]))

print('=== RAGGED tangent-level (must accept C, reject K+C and ()) ===')
for name, W in (('C', W_C), ('K+C', W_KC), ('()', W_0)):
    attempt('T3Tangent.absorb_weights  W=%s' % name, lambda: v.absorb_weights(W))
    attempt('T3Tangent.weighted_norm   W=%s' % name, lambda: v.weighted_norm(W))
    attempt('T3Tangent.weighted_inner  W=%s' % name, lambda: v.weighted_inner(u, W))
    attempt('check_fw_pair             W=%s' % name, lambda: bvf.check_fw_pair(frame, W))
print('=== RAGGED variation-level (blind to frame by design: trailing rule) ===')
for name, W in (('C', W_C), ('K+C', W_KC), ('()', W_0)):
    print('  is_consistent_with(tangent) W=%s -> %s' % (name, W.is_consistent_with(v)))
    attempt('bvf.fv_absorb_weights(variations) W=%s' % name, lambda: bvf.fv_absorb_weights(v.variations, W))
    attempt('backend fv_weighted_norm          W=%s' % name, lambda: fv_operations.fv_weighted_norm(v.variations.data, W.data, len(v.stack_shape)))
# K-tiled weight gives identical numbers where accepted
n_C = fv_operations.fv_weighted_norm(v.variations.data, W_C.data, 2)
n_KC = fv_operations.fv_weighted_norm(v.variations.data, W_KC.data, 2)
print('  backend norm C vs K-tiled K+C agree:', np.allclose(n_C, n_KC))

print('=== RAGGED T3Weights with a stacked train (no K concept; is_consistent_with vs absorb) ===')
TW = t3.T3Weights.from_t3svd(x)                                                 # stack C
TW_big = t3.T3Weights(tuple(tile(w) for w in TW.tucker_weights), tuple(tile(w) for w in TW.tt_weights))   # (3,2)
TW_0 = t3.T3Weights(tuple(w[0] for w in TW.tucker_weights), tuple(w[0] for w in TW.tt_weights))
for name, W in (('C', TW), ('(3,)+C', TW_big), ('()', TW_0)):
    print('  T3Weights.is_consistent_with(x) W=%s -> %s' % (name, W.is_consistent_with(x)))
    attempt('t3_absorb_weights W=%s' % name, lambda: t3.t3_absorb_weights(x, W))
    attempt('t3_weighted_norm  W=%s' % name, lambda: t3.t3_weighted_norm(x, W))
    attempt('t3_weighted_inner W=%s' % name, lambda: t3.t3_weighted_inner(x, W, x, TW))

print('=== UNIFORM tangent-level ===')
ux = ut3.UniformTuckerTensorTrain.from_t3(x)
uframe, _ = ubvf.ut3_orthogonal_representations(ux)
uv = ut3m.UNIFORM_COREWISE.randn(uframe, stack_shape=K)
uu = ut3m.UNIFORM_COREWISE.randn(uframe, stack_shape=K)
UW_C = ubvf.UT3FrameWeights.from_ut3weights(ut3.UT3Weights.from_ut3svd(ux))
def utile(a, n_lead=1):  # (d,)+C+(s,) -> (d,)+K+C+(s,)
    return np.broadcast_to(a[:, None], a.shape[:1] + K + a.shape[1:]).copy()
UW_KC = ubvf.UT3FrameWeights(*[utile(a) for a in UW_C.supercores],
                             ubvf.UT3VariationsMasks(*[utile(m) for m in UW_C.masks.data]))
UW_0 = ubvf.UT3FrameWeights(*[a[:, 0] for a in UW_C.supercores], ubvf.UT3VariationsMasks(*[m[:, 0] for m in UW_C.masks.data]))
assert UW_C.stack_shape == C and UW_KC.stack_shape == K + C and UW_0.stack_shape == ()
for name, W in (('C', UW_C), ('K+C', UW_KC), ('()', UW_0)):
    attempt('UT3Tangent.absorb_weights  W=%s' % name, lambda: uv.absorb_weights(W))
    attempt('UT3Tangent.weighted_norm   W=%s' % name, lambda: uv.weighted_norm(W))
    attempt('UT3Tangent.weighted_inner  W=%s' % name, lambda: uv.weighted_inner(uu, W))
    attempt('check_ufw_pair             W=%s' % name, lambda: ubvf.check_ufw_pair(uframe, W))
print('=== UNIFORM variation-level (trailing rule) ===')
for name, W in (('C', UW_C), ('K+C', UW_KC), ('()', UW_0)):
    print('  UT3FrameWeights.is_consistent_with(tangent) W=%s -> %s' % (name, W.is_consistent_with(uv)))
    attempt('ubvf.ufv_absorb_weights(variations) W=%s' % name, lambda: ubvf.ufv_absorb_weights(uv.variations, W))
    attempt('backend ufv_weighted_norm           W=%s' % name, lambda: utv_operations.ufv_weighted_norm(uv.variations.data, W.data, len(uv.stack_shape)))
print('  uniform C vs K-tiled agree:', np.allclose(utv_operations.ufv_weighted_norm(uv.variations.data, UW_C.data, 2),
                                                   utv_operations.ufv_weighted_norm(uv.variations.data, UW_KC.data, 2)))
print('=== UNIFORM UT3Weights with a stacked train ===')
UTW = ut3.UT3Weights.from_ut3svd(ux)
UTW_big = ut3.UT3Weights(utile(UTW.tucker_weight_supercore), utile(UTW.tt_weight_supercore), ut3.UT3Masks(*[utile(m) for m in UTW.masks.data]))
UTW_0 = ut3.UT3Weights(UTW.tucker_weight_supercore[:, 0], UTW.tt_weight_supercore[:, 0], ut3.UT3Masks(*[m[:, 0] for m in UTW.masks.data]))
for name, W in (('C', UTW), ('(3,)+C', UTW_big), ('()', UTW_0)):
    print('  UT3Weights.is_consistent_with(ux) W=%s -> %s' % (name, W.is_consistent_with(ux)))
    attempt('ut3_absorb_weights W=%s' % name, lambda: ut3.ut3_absorb_weights(ux, W))
    attempt('ut3_weighted_norm  W=%s' % name, lambda: ut3.ut3_weighted_norm(ux, W))
