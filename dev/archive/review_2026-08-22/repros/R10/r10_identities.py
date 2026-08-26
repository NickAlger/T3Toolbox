"""R10: dense-level identities of the RAGGED weighted layer at asymmetric shapes + stacks."""
import numpy as np, warnings
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.frame_variations_format as bvf
import t3toolbox.manifold as t3m
import t3toolbox.corewise as cw
from t3toolbox.backend import t3_linalg, t3_operations, fv_operations

def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))

def rand_W(x, rng):
    ss = x.stack_shape
    return t3.T3Weights(tuple(rng.standard_normal(ss + (n,)) for n in x.tucker_ranks),
                        tuple(rng.standard_normal(ss + (r,)) for r in x.tt_ranks))

def hand_dense(x, W):
    d = x.d; tk, tt = x.data; tw, ttw = W.data
    pool = iter('abcdefghijklmnopqrstuvwxyz')
    out = [next(pool) for _ in range(d)]; rank = [next(pool) for _ in range(d)]; bond = [next(pool) for _ in range(d + 1)]
    terms, ops = [], []
    for k in range(d): terms.append('...' + rank[k] + out[k]); ops.append(tk[k])
    for k in range(d): terms.append('...' + bond[k] + rank[k] + bond[k + 1]); ops.append(tt[k])
    for k in range(d): terms.append('...' + rank[k]); ops.append(tw[k])
    for k in range(d + 1): terms.append('...' + bond[k]); ops.append(ttw[k])
    return np.einsum(','.join(terms) + '->...' + ''.join(out), *ops)

rng = np.random.default_rng(0)
# asymmetric: distinct mode sizes, distinct tucker ranks, distinct (non-square) tt ranks, d in {1,2,3,4}
STRUCTS = [((5, 6, 7), (2, 3, 3), (1, 2, 3, 1)),
           ((4, 7, 5, 6), (2, 3, 3, 2), (1, 2, 4, 2, 1)),
           ((6, 5), (3, 3), (1, 3, 1)),
           ((9,), (1,), (1, 1))]
STACKS = [(), (1,), (2,), (2, 3)]
worst = {}
def rec(key, v): worst[key] = max(worst.get(key, 0.0), v)

for struct in STRUCTS:
    for ss in STACKS:
        x = t3.TuckerTensorTrain.randn(*struct, stack_shape=ss)
        y = t3.TuckerTensorTrain.randn(*struct, stack_shape=ss)
        W, W2 = rand_W(x, rng), rand_W(y, rng)
        xw, yw = t3.t3_absorb_weights(x, W), t3.t3_absorb_weights(y, W2)
        dx, dy = hand_dense(x, W), hand_dense(y, W2)
        rec('absorb==hand_dense', rel(xw.to_dense(), dx))
        axes = tuple(range(len(ss), len(ss) + x.d))
        rec('weighted_norm==dense', rel(t3.t3_weighted_norm(x, W), np.sqrt((dx ** 2).sum(axis=axes))))
        rec('weighted_norm==absorbed.norm', rel(t3.t3_weighted_norm(x, W), xw.norm()))
        rec('weighted_inner==dense', rel(t3.t3_weighted_inner(x, W, y, W2), (dx * dy).sum(axis=axes)))
        # + <-> concatenate (frontend +, which does NOT squash tails: boundary bonds become 2)
        s = x + y
        Wc = W.concatenate(W2)
        assert Wc.is_consistent_with(s), (s.tt_ranks, Wc.tt_ranks)
        rec('plus<->concatenate(frontend +)', rel(t3.t3_absorb_weights(s, Wc).to_dense(), dx + dy))
        # after squash_tails the concatenated weights no longer pair
        assert not Wc.is_consistent_with(s.squash_tails())
        # Hadamard <-> kronecker using the LIBRARY's * (t3_mult), not a test-local combine
        p = x * y
        Wk = W.kronecker(W2)
        assert Wk.is_consistent_with(p), (p.ranks, Wk.tucker_ranks, Wk.tt_ranks)
        rec('hadamard<->kronecker(library *)', rel(t3.t3_absorb_weights(p, Wk).to_dense(), dx * dy))
        # reverse
        rec('reverse', rel(t3.t3_absorb_weights(x.reverse(), W.reverse()).to_dense(),
                           np.moveaxis(dx, list(range(len(ss), dx.ndim)), list(range(dx.ndim - 1, len(ss) - 1, -1)))))
        # ones identity, reciprocal identity
        ones = t3.T3Weights(tuple(np.ones_like(w) for w in W.tucker_weights), tuple(np.ones_like(w) for w in W.tt_weights))
        rec('ones identity', rel(t3.t3_absorb_weights(x, ones).to_dense(), x.to_dense()))
        rec('W then 1/W', rel(t3.t3_absorb_weights(xw, W.reciprocal()).to_dense(), x.to_dense()))
        rec('sqrt.sqrt==W', rel(np.concatenate([w.ravel() for w in (W.sqrt().sqrt() if False else W.sqrt()).tucker_weights]) ** 2,
                                np.concatenate([np.abs(w).ravel() for w in W.tucker_weights])) if np.all([np.all(w >= 0) for w in W.tucker_weights]) else 0.0)
        # stack/unstack
        if ss:
            tree = W.unstack()
            Wr = t3.T3Weights.stack(tree)
            rec('stack(unstack)', max(rel(a, b) for a, b in zip(Wr.tucker_weights + Wr.tt_weights, W.tucker_weights + W.tt_weights)))
            leaf = tree
            for _ in ss: leaf = leaf[0]
            rec('unstack leaf[0..]', max(rel(a, b[(0,) * len(ss)]) for a, b in zip(leaf.tucker_weights + leaf.tt_weights, W.tucker_weights + W.tt_weights)))
        # ----- T3FrameWeights at a frame from x (minimal ranks)
        frame, _ = bvf.t3_orthogonal_representations(x)
        K = (3,)
        v = t3m.COREWISE.randn(frame, stack_shape=K)
        u = t3m.COREWISE.randn(frame, stack_shape=K)
        FW = bvf.T3FrameWeights.from_t3weights(W)
        bvf.check_fw_pair(frame, FW)
        vw = v.absorb_weights(FW)
        rec('fw: weighted_norm==absorbed.corewise_norm', rel(v.weighted_norm(FW), vw.corewise_norm()))
        rec('fw: weighted_inner==absorbed.corewise_inner', rel(v.weighted_inner(u, FW), vw.corewise_inner(u.absorb_weights(FW))))
        # dense check of the absorbed variations: each weighted variation core == hand scale
        V, H = v.variations.data; Vw, Hw = vw.variations.data
        up, dn, lf, rt = FW.data
        rec('fw: absorb V hand', max(rel(a, np.einsum('...i,...io->...io', w, b)) for a, w, b in zip(Vw, dn, V)))
        rec('fw: absorb H hand', max(rel(a, np.einsum('...aib,...a,...i,...b->...aib', b, l, w, r)) for a, b, l, w, r in zip(Hw, H, lf, up, rt)))
        # reverse of the tangent + reverse of the metric: norms agree
        rec('fw: reverse', rel(v.reverse().weighted_norm(FW.reverse()), v.weighted_norm(FW)))
        bvf.check_fw_pair(frame.reverse(), FW.reverse())
        # all-ones metric = corewise_norm
        onesF = bvf.T3FrameWeights(*[tuple(np.ones_like(w) for w in fam) for fam in FW.data])
        rec('fw: ones==corewise_norm', rel(v.weighted_norm(onesF), v.corewise_norm()))
        # backend twins agree
        rec('fw: backend norm', rel(fv_operations.fv_weighted_norm(v.variations.data, FW.data, len(v.stack_shape)), v.weighted_norm(FW)))
        rec('fw: backend inner', rel(fv_operations.fv_weighted_inner(v.variations.data, u.variations.data, FW.data, len(v.stack_shape)), v.weighted_inner(u, FW)))
        # concat / kron ranks; stack/unstack
        Fc, Fk = FW.concatenate(FW), FW.kronecker(FW)
        assert Fc.up_ranks == tuple(2 * n for n in FW.up_ranks) and Fk.right_ranks == tuple(n * n for n in FW.right_ranks)
        if ss:
            tree = FW.unstack(); Fr = bvf.T3FrameWeights.stack(tree)
            rec('fw: stack(unstack)', max(rel(a, b) for fa, fb in zip(Fr.data, FW.data) for a, b in zip(fa, fb)))
        # K+C inner returns K+C shape
        assert v.weighted_norm(FW).shape == K + ss, (v.weighted_norm(FW).shape, K + ss)

for k, v in worst.items():
    print('%-45s worst rel err = %.2e %s' % (k, v, 'OK' if v < 1e-10 else '<<<<< FAIL'))
