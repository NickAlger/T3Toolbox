"""R10: uniform weighted layer == ragged (to_uniform -> op -> to_ragged), incl. + / * partners, from_*svd, reciprocal zeros."""
import numpy as np, warnings
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.frame_variations_format as bvf
import t3toolbox.manifold as t3m
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.uniform_frame_variations_format as ubvf
import t3toolbox.uniform_manifold as ut3m
from t3toolbox.backend import ut3_linalg, ut3_operations

def rel(a, b):
    a, b = np.asarray(a), np.asarray(b); return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))
rng = np.random.default_rng(1)
def rand_W(x):
    ss = x.stack_shape
    return t3.T3Weights(tuple(rng.standard_normal(ss + (n,)) for n in x.tucker_ranks), tuple(rng.standard_normal(ss + (r,)) for r in x.tt_ranks))

print('--- A. tensor weights, asymmetric, padded above the real ranks, stacks ---')
for struct in [((5, 6, 7), (2, 3, 4), (1, 2, 3, 1)), ((4, 7, 5, 6), (2, 3, 3, 2), (1, 2, 4, 2, 1)), ((6, 5), (3, 3), (1, 3, 1))]:
    for ss in [(), (2,), (2, 3)]:
        x = t3.TuckerTensorTrain.randn(*struct, stack_shape=ss); y = t3.TuckerTensorTrain.randn(*struct, stack_shape=ss)
        W, W2 = rand_W(x), rand_W(y)
        n, r = max(x.tucker_ranks) + 2, max(x.tt_ranks) + 1
        ux, uy = ut3.UniformTuckerTensorTrain.from_t3(x, n=n, r=r), ut3.UniformTuckerTensorTrain.from_t3(y, n=n, r=r)
        UW, UW2 = ut3.UT3Weights.from_t3weights(W, n=n, r=r), ut3.UT3Weights.from_t3weights(W2, n=n, r=r)
        assert UW.is_consistent_with(ux)
        e = {}
        e['absorb'] = rel(ut3.ut3_absorb_weights(ux, UW).to_dense(), t3.t3_absorb_weights(x, W).to_dense())
        e['norm'] = rel(ut3.ut3_weighted_norm(ux, UW), t3.t3_weighted_norm(x, W))
        e['norm(no orth)'] = rel(ut3.ut3_weighted_norm(ux, UW, use_orthogonalization=False), t3.t3_weighted_norm(x, W))
        e['inner'] = rel(ut3.ut3_weighted_inner(ux, UW, uy, UW2), t3.t3_weighted_inner(x, W, y, W2))
        # + partner: backend ut3_add (unsquashed) pairs with concatenate; the frontend + squashes
        s_back = ut3._from_data(ut3_linalg.ut3_add(ux.data, uy.data)) if hasattr(ut3, '_from_data') else None
        UWc = UW.concatenate(UW2)
        cons_back = UWc.is_consistent_with(s_back)
        s_front = ux + uy
        cons_front = UWc.is_consistent_with(s_front)
        e['+(backend ut3_add, unsquashed)'] = rel(ut3.ut3_absorb_weights(s_back, UWc).to_dense(), t3.t3_absorb_weights(x, W).to_dense() + t3.t3_absorb_weights(y, W2).to_dense())
        # roundtrip of combined weights through ragged
        rag = UWc.to_t3weights()
        leaf = rag
        for _ in ss: leaf = leaf[0]
        Wc = W.concatenate(W2)
        e['concat to_t3weights'] = max(rel(a, b[(0,) * len(ss)]) for a, b in zip(leaf.tucker_weights + leaf.tt_weights, Wc.tucker_weights + Wc.tt_weights))
        UWk = UW.kronecker(UW2); rag = UWk.to_t3weights(); leaf = rag
        for _ in ss: leaf = leaf[0]
        Wk = W.kronecker(W2)
        e['kron to_t3weights'] = max(rel(a, b[(0,) * len(ss)]) for a, b in zip(leaf.tucker_weights + leaf.tt_weights, Wk.tucker_weights + Wk.tt_weights))
        # reciprocal / sqrt vs ragged (abs for sqrt)
        Wa = t3.T3Weights(tuple(np.abs(w) for w in W.tucker_weights), tuple(np.abs(w) for w in W.tt_weights))
        UWa = ut3.UT3Weights.from_t3weights(Wa, n=n, r=r)
        for name, ua, ra in (('reciprocal', UWa.reciprocal(), Wa.reciprocal()), ('sqrt', UWa.sqrt(), Wa.sqrt())):
            leaf = ua.to_t3weights()
            for _ in ss: leaf = leaf[0]
            e[name] = max(rel(a, b[(0,) * len(ss)]) for a, b in zip(leaf.tucker_weights + leaf.tt_weights, ra.tucker_weights + ra.tt_weights))
            assert np.isfinite(ua.tucker_weight_supercore).all() and np.isfinite(ua.tt_weight_supercore).all()
        bad = {k: v for k, v in e.items() if v > 1e-9}
        print('struct=%s ss=%s  concat consistent with backend ut3_add: %s, with frontend ux+uy: %s  %s' % (
            struct[0], ss, cons_back, cons_front, ('ALL OK' if not bad else 'FAIL %s' % bad)))

print('--- B. from_ut3svd on a train padded ABOVE its real rank (docstring: "tight padding") ---')
x = t3.TuckerTensorTrain.randn((5, 6, 7), (2, 3, 3), (1, 2, 3, 1))
ux = ut3.UniformTuckerTensorTrain.from_t3(x, n=4, r=4)
UW = ut3.UT3Weights.from_ut3svd(ux)
print('  ux n,r =', ux.n, ux.r, ' W n,r =', UW.n, UW.r, ' consistent:', UW.is_consistent_with(ux))
try:
    ut3.ut3_weighted_norm(ux, UW); print('  ut3_weighted_norm(ux, from_ut3svd(ux)) accepted')
except Exception as ex:
    print('  ut3_weighted_norm(ux, from_ut3svd(ux)) raised', type(ex).__name__)
xs, _, _ = ux.t3svd()
print('  documented workaround: W consistent with ux.t3svd()[0]:', UW.is_consistent_with(xs))
# ragged from_t3svd: weights equal svals; is tt[0] = tt[d] = [1.]?
Wr = t3.T3Weights.from_t3svd(x); xr, tk_s, tt_s = x.t3svd()
print('  ragged from_t3svd: tt_weights[0], [-1] =', Wr.tt_weights[0], Wr.tt_weights[-1], '; tucker ranks', Wr.tucker_ranks, 'tt ranks', Wr.tt_ranks)
print('  ragged from_t3svd consistent with x (minimal):', Wr.is_consistent_with(x), '; weighted_norm(x, W) == norm(absorb):', np.allclose(t3.t3_weighted_norm(x, Wr), t3.t3_absorb_weights(x, Wr).norm()))
# non-minimal x: from_t3svd not consistent with x
xn = t3.TuckerTensorTrain.randn((5, 6, 7), (4, 4, 4), (1, 4, 4, 1))
Wn = t3.T3Weights.from_t3svd(xn)
print('  non-minimal x ranks', xn.ranks, '-> from_t3svd ranks', Wn.tucker_ranks, Wn.tt_ranks, 'consistent with x:', Wn.is_consistent_with(xn))
try:
    t3.t3_absorb_weights(xn, Wn); print('  absorb(non-minimal x, from_t3svd(x)) ACCEPTED')
except Exception as ex:
    print('  absorb(non-minimal x, from_t3svd(x)) raised', type(ex).__name__, str(ex).splitlines()[0][:100])

print('--- C. reciprocal / sqrt on real zeros and negatives ---')
W0 = t3.T3Weights((np.array([1.0, 0.0]), np.array([2.0, 3.0, 1.0]), np.array([1.0, 1.0])), (np.ones(1), np.ones(2), np.ones(3), np.ones(1)))
with warnings.catch_warnings(record=True) as wl:
    warnings.simplefilter('always'); R = W0.reciprocal()
print('  ragged reciprocal real zero ->', R.tucker_weights[0], '; warnings:', [str(w.message)[:40] for w in wl])
UW0 = ut3.UT3Weights.from_t3weights(W0, n=4, r=4)
with warnings.catch_warnings(record=True) as wl:
    warnings.simplefilter('always'); UR = UW0.reciprocal()
print('  uniform reciprocal: real zero ->', UR.tucker_weight_supercore[0], '(padding finite:', np.isfinite(UR.tucker_weight_supercore[:, 2:]).all(), ') warnings:', [str(w.message)[:40] for w in wl])
Wneg = t3.T3Weights((np.array([1.0, -1.0]), np.array([2.0, 3.0, 1.0]), np.array([1.0, 1.0])), (np.ones(1), np.ones(2), np.ones(3), np.ones(1)))
with warnings.catch_warnings(record=True) as wl:
    warnings.simplefilter('always'); S = Wneg.sqrt(); US = ut3.UT3Weights.from_t3weights(Wneg, n=3, r=3).sqrt()
print('  sqrt of negative real: ragged', S.tucker_weights[0], ' uniform', US.tucker_weight_supercore[0])

print('--- D. frame weights uniform == ragged at C=(2,), K=(3,), asymmetric ---')
x = t3.TuckerTensorTrain.randn((5, 6, 7, 4), (2, 3, 3, 2), (1, 2, 4, 2, 1), stack_shape=(2,))
frame, _ = bvf.t3_orthogonal_representations(x); v = t3m.COREWISE.randn(frame, stack_shape=(3,)); u = t3m.COREWISE.randn(frame, stack_shape=(3,))
FW = bvf.T3FrameWeights.from_t3weights(t3.T3Weights.from_t3svd(x)).reciprocal()
ux = ut3.UniformTuckerTensorTrain.from_t3(x, n=5, r=4)
uv = ut3m.UT3Tangent.from_t3tangent(v); uu = ut3m.UT3Tangent(uv.frame, ubvf.UT3Variations.from_t3variations(u.variations) if False else ut3m.UT3Tangent.from_t3tangent(u).variations)
uframe = uv.frame
UFW = ubvf.UT3FrameWeights.from_t3frameweights(FW, nU=uframe.nU, nD=uframe.nD, rL=uframe.rL, rR=uframe.rR)
print('  check_ufw_pair ok:', ubvf.check_ufw_pair(uframe, UFW) is None)
print('  weighted_norm  rel err:', rel(uv.weighted_norm(UFW), v.weighted_norm(FW)))
print('  weighted_inner rel err:', rel(uv.weighted_inner(uu, UFW), v.weighted_inner(u, FW)))
print('  absorb -> corewise_norm rel err:', rel(uv.absorb_weights(UFW).corewise_norm(), v.absorb_weights(FW).corewise_norm()))
# from_ut3weights pairs with the frame from ut3_orthogonal_representations
UFW2 = ubvf.UT3FrameWeights.from_ut3weights(ut3.UT3Weights.from_ut3svd(ux))
uframe2, _ = ubvf.ut3_orthogonal_representations(ux)
print('  from_ut3weights(from_ut3svd(ux)) pairs with ut3_orthogonal_representations(ux) frame:', end=' ')
try: ubvf.check_ufw_pair(uframe2, UFW2); print('yes')
except Exception as ex: print('NO:', str(ex).splitlines()[0])
# concat/kron/reciprocal/sqrt roundtrip vs ragged
for name, ua, ra in (('concat', UFW.concatenate(UFW), FW.concatenate(FW)), ('kron', UFW.kronecker(UFW), FW.kronecker(FW)),
                     ('recip', UFW.reciprocal(), FW.reciprocal()), ('sqrt', UFW.reciprocal().sqrt(), FW.reciprocal().sqrt())):
    leaf = ua.to_t3frameweights()[0]
    err = max(rel(a, b[0]) for fa, fb in zip(leaf.data, ra.data) for a, b in zip(fa, fb))
    print('  frame-weights %-7s uniform==ragged rel err %.1e' % (name, err))
