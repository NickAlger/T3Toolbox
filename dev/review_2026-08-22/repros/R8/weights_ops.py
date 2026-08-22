"""R8: UT3Weights ops three-prong: absorb (garbage transparency), weighted norm/inner, reciprocal/sqrt
guards, concatenate/kronecker masks (gappy) + to_t3weights through gappy masks, from_t3weights pad."""
import numpy as np, itertools
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.backend.ut3_operations as uo
from t3toolbox.backend.common import prefix_mask
np.random.seed(0)
TOL = 1e-9
FAILS = []
def relerr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))
def check(name, cond, detail=''):
    if not cond:
        FAILS.append((name, detail)); print('FAIL', name, detail)
def corrupt(ux, scale=10.0):
    tkm, ttm = ux.masks.data
    d, N, n, r = ux.d, ux.N, ux.n, ux.r
    stack = ux.stack_shape
    shape_mask = prefix_mask(ux.shape, N).reshape((d,) + (1,) * len(stack) + (1, N))
    tk_real = tkm[..., :, None] & shape_mask
    tt_real = ttm[:-1][..., :, None, None] & tkm[..., None, :, None] & ttm[1:][..., None, None, :]
    g_tk = scale * np.random.randn(*ux.tucker_supercore.shape) * (~tk_real)
    g_tt = scale * np.random.randn(*ux.tt_supercore.shape) * (~tt_real)
    return ut3.UniformTuckerTensorTrain(ux.tucker_supercore + g_tk, ux.tt_supercore + g_tt, ux.shape, ux.masks)
def corrupt_w(W, scale=10.0):
    tkm, ttm = W.masks.data
    return ut3.UT3Weights(W.tucker_weight_supercore + scale * np.random.randn(*tkm.shape) * (~tkm),
                          W.tt_weight_supercore + scale * np.random.randn(*ttm.shape) * (~ttm), W.masks)

CONFIGS = [((4, 6), (2, 3), (1, 3, 1)), ((4, 5, 3), (2, 3, 2), (1, 2, 3, 1)), ((3, 6, 4, 5), (2, 3, 3, 2), (1, 2, 4, 2, 1))]
for (shape, tk, tt), stack, pad in itertools.product(CONFIGS, [(), (2,), (2, 3)], [False, True]):
    tag = 'shape=%s stack=%s pad=%s' % (shape, stack, pad)
    d = len(shape)
    x = t3.TuckerTensorTrain.randn(shape, tk, tt, stack_shape=stack)
    xs, _, _ = x.t3svd()
    kw = dict(N=max(shape) + 2, n=max(tk) + 1, r=max(tt) + 1) if pad else {}
    ux = ut3.UniformTuckerTensorTrain.from_t3(xs, **kw)
    uxg = corrupt(ux)
    try:
        # weights from ragged svd, padded to ux
        Wr = t3.T3Weights.from_t3svd(xs)
        W = ut3.UT3Weights.from_t3weights(Wr, n=ux.n, r=ux.r)
        check('from_t3weights consistent ' + tag, W.is_consistent_with(ux))
        check('weights masks host ' + tag, all(type(m) is np.ndarray for m in W.masks.data))
        Wg = corrupt_w(W)
        # from_ut3svd twin
        Wu = ut3.UT3Weights.from_ut3svd(ux)
        back = Wu.to_t3weights()
        leaves_r = [Wr] if not stack else None
        def flat(t):
            return [t] if isinstance(t, t3.T3Weights) else [l for s in t for l in flat(s)]
        if not stack:
            for j in range(d):
                check('from_ut3svd tucker sval %d %s' % (j, tag), relerr(back.data[0][j], Wr.data[0][j]) < 1e-7)
            for j in range(d + 1):
                check('from_ut3svd tt sval %d %s' % (j, tag), relerr(back.data[1][j], Wr.data[1][j]) < 1e-7)
        # absorb: ragged twin, garbage transparency (real parts equal), masks unchanged
        xa = t3.t3_absorb_weights(xs, Wr)
        ua = ut3.ut3_absorb_weights(ux, W)
        check('absorb dense ' + tag, relerr(ua.to_dense(), xa.to_dense()) < TOL)
        check('absorb masks unchanged ' + tag, ua.masks == ux.masks)
        check('absorb garbage x ' + tag, relerr(ut3.ut3_absorb_weights(uxg, W).to_dense(), xa.to_dense()) < TOL)
        check('absorb garbage w ' + tag, relerr(ut3.ut3_absorb_weights(ux, Wg).to_dense(), xa.to_dense()) < TOL)
        check('absorb garbage both ' + tag, relerr(ut3.ut3_absorb_weights(uxg, Wg).to_dense(), xa.to_dense()) < TOL)
        # weighted norm/inner
        for orth in (True, False):
            check('wnorm orth=%s %s' % (orth, tag), relerr(ut3.ut3_weighted_norm(ux, W, orth), t3.t3_weighted_norm(xs, Wr)) < TOL)
            check('wnorm garbage orth=%s %s' % (orth, tag), relerr(ut3.ut3_weighted_norm(uxg, Wg, orth), t3.t3_weighted_norm(xs, Wr)) < TOL)
            y = t3.TuckerTensorTrain.randn(shape, tuple(max(1, a - 1) for a in tk), (1,) + tuple(max(1, b - 1) for b in tt[1:-1]) + (1,), stack_shape=stack)
            ys, _, _ = y.t3svd()
            uy = ut3.UniformTuckerTensorTrain.from_t3(ys, **kw)
            Wy = ut3.UT3Weights.from_t3weights(t3.T3Weights.from_t3svd(ys), n=uy.n, r=uy.r)
            check('winner orth=%s %s' % (orth, tag), relerr(ut3.ut3_weighted_inner(ux, W, uy, Wy, orth), t3.t3_weighted_inner(xs, Wr, ys, t3.T3Weights.from_t3svd(ys))) < TOL)
            check('winner garbage orth=%s %s' % (orth, tag), relerr(ut3.ut3_weighted_inner(uxg, Wg, corrupt(uy), corrupt_w(Wy), orth), t3.t3_weighted_inner(xs, Wr, ys, t3.T3Weights.from_t3svd(ys))) < TOL)
        # reciprocal / sqrt: real parts vs ragged, padding finite & zero, garbage in -> same real parts
        for nm, uf, rf in [('reciprocal', lambda w: w.reciprocal(), lambda w: w.reciprocal()), ('sqrt', lambda w: w.sqrt(), lambda w: w.sqrt())]:
            R = uf(W); Rg = uf(Wg); Rr = rf(Wr)
            check('%s finite %s' % (nm, tag), bool(np.isfinite(R.tucker_weight_supercore).all() and np.isfinite(R.tt_weight_supercore).all()))
            check('%s garbage finite %s' % (nm, tag), bool(np.isfinite(Rg.tucker_weight_supercore).all() and np.isfinite(Rg.tt_weight_supercore).all()))
            check('%s pad zero %s' % (nm, tag), float(np.abs(R.tucker_weight_supercore[~R.masks.tucker_edge_mask]).max(initial=0)) == 0.0 and float(np.abs(R.tt_weight_supercore[~R.masks.tt_edge_mask]).max(initial=0)) == 0.0)
            check('%s masks unchanged %s' % (nm, tag), R.masks == W.masks)
            if not stack:
                rb = R.to_t3weights(); rbg = Rg.to_t3weights()
                for j in range(d):
                    check('%s tucker %d %s' % (nm, j, tag), relerr(rb.data[0][j], Rr.data[0][j]) < TOL and relerr(rbg.data[0][j], Rr.data[0][j]) < TOL)
                for j in range(d + 1):
                    check('%s tt %d %s' % (nm, j, tag), relerr(rb.data[1][j], Rr.data[1][j]) < TOL and relerr(rbg.data[1][j], Rr.data[1][j]) < TOL)
        # concatenate / kronecker: exact masks, ragged twin through gappy masks (unstacked)
        if not stack:
            y = t3.TuckerTensorTrain.randn(shape, tuple(max(1, a - 1) for a in tk), (1,) + tuple(max(1, b - 1) for b in tt[1:-1]) + (1,))
            ys, _, _ = y.t3svd()
            uy = ut3.UniformTuckerTensorTrain.from_t3(ys, **kw)
            Wyr = t3.T3Weights.from_t3svd(ys)
            Wy = ut3.UT3Weights.from_t3weights(Wyr, n=uy.n, r=uy.r)
            C = W.concatenate(Wy); Cr = Wr.concatenate(Wyr)
            K = W.kronecker(Wy);   Kr = Wr.kronecker(Wyr)
            exp_ctk = np.concatenate([W.masks.tucker_edge_mask, Wy.masks.tucker_edge_mask], axis=-1)
            exp_ctt = np.concatenate([W.masks.tt_edge_mask, Wy.masks.tt_edge_mask], axis=-1)
            check('concat exact masks ' + tag, np.array_equal(C.masks.tucker_edge_mask, exp_ctk) and np.array_equal(C.masks.tt_edge_mask, exp_ctt))
            ktk = (W.masks.tucker_edge_mask[..., :, None] & Wy.masks.tucker_edge_mask[..., None, :]).reshape(d, -1)
            ktt = (W.masks.tt_edge_mask[..., :, None] & Wy.masks.tt_edge_mask[..., None, :]).reshape(d + 1, -1)
            check('kron exact masks ' + tag, np.array_equal(K.masks.tucker_edge_mask, ktk) and np.array_equal(K.masks.tt_edge_mask, ktt))
            cb, kb = C.to_t3weights(), K.to_t3weights()
            for j in range(d):
                check('concat tucker %d %s' % (j, tag), relerr(cb.data[0][j], Cr.data[0][j]) < TOL)
                check('kron tucker %d %s' % (j, tag), relerr(kb.data[0][j], Kr.data[0][j]) < TOL)
            for j in range(d + 1):
                check('concat tt %d %s' % (j, tag), relerr(cb.data[1][j], Cr.data[1][j]) < TOL)
                check('kron tt %d %s' % (j, tag), relerr(kb.data[1][j], Kr.data[1][j]) < TOL)
            # gappy-mask weights consistent with the ut3_add result (masks concat the same way)? the object
            # + squashes its boundary, weights do not -> masks differ at the boundary bonds: is that documented?
            s = ux + uy
            check('concat weights pair with x+y (boundary squash!) ' + tag, C.is_consistent_with(s),
                  'masks equal: tucker %s, tt interior %s, tt boundary %s' % (
                      np.array_equal(C.masks.tucker_edge_mask, s.masks.tucker_edge_mask),
                      np.array_equal(C.masks.tt_edge_mask[1:-1], s.masks.tt_edge_mask[1:-1]),
                      np.array_equal(C.masks.tt_edge_mask[[0, -1]], s.masks.tt_edge_mask[[0, -1]])))
            # reciprocal through a gappy mask
            Cr_ = C.reciprocal().to_t3weights()
            Crr = Cr.reciprocal()
            check('concat reciprocal gappy ' + tag, all(relerr(Cr_.data[0][j], Crr.data[0][j]) < TOL for j in range(d)) and all(relerr(Cr_.data[1][j], Crr.data[1][j]) < TOL for j in range(d + 1)))
    except Exception as e:
        import traceback; traceback.print_exc()
        FAILS.append(('EXC ' + tag, repr(e))); print('EXC', tag, repr(e))
print('\n==== %d failures' % len(FAILS))
from collections import Counter
print(Counter(f[0].split(' shape=')[0] for f in FAILS))
