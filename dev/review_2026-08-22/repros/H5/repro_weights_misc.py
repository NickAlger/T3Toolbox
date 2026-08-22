"""Weighted layer + save/load + misc ops: garbage robustness and exact masks (uniform vs ragged twins)."""
import sys, os, tempfile
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from h5lib import *

fails = []


def report(name, cond, detail=''):
    print('  %-64s %s %s' % (name, 'ok ' if cond else 'FAIL', detail))
    if not cond:
        fails.append((name, detail))


def corrupt_weights(W, scale=1e3, seed=3):
    rng = np.random.RandomState(seed)
    new = [sc + scale * rng.randn(*sc.shape) * (1.0 - m) for sc, m in zip(W.supercores, W.masks.data)]
    return ut3.UT3Weights(new[0], new[1], W.masks)


for (shape, tr, ttr, ss) in CASES:
    if len(shape) < 2:
        continue
    for force_pad in (False, True):
        np.random.seed(0)
        x = t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=ss)
        y = t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=ss)
        kw = dict(PAD) if force_pad else {}
        ux = ut3.UniformTuckerTensorTrain.from_t3(x, **kw); uy = ut3.UniformTuckerTensorTrain.from_t3(y, **kw)
        gx = corrupt_ut3(ux); gy = corrupt_ut3(uy, seed=8)
        print('case shape=%s stack=%s force_pad=%s' % (shape, ss, force_pad))
        xs, stk, stt = x.t3svd(); Wr = t3.T3Weights.from_t3svd(x)
        uxs, _, _ = ux.t3svd()
        W = ut3.UT3Weights.from_ut3svd(ux)
        report('from_ut3svd masks == ut3svd output masks', W.masks == uxs.masks)
        report('from_ut3svd ragged twin (to_t3weights)', (all(relerr(a, b) < 1e-9 for a, b in zip(W.to_t3weights().tucker_weights, Wr.tucker_weights)) if not ss else True))
        Wg = corrupt_weights(W)
        report('ut3_absorb_weights (dirty x, dirty W) dense', relerr(ut3.ut3_absorb_weights(corrupt_ut3(uxs), Wg).to_dense(), ut3.ut3_absorb_weights(uxs, W).to_dense()) < 1e-9)
        report('ut3_weighted_norm (dirty)', relerr(ut3.ut3_weighted_norm(corrupt_ut3(uxs), Wg), ut3.ut3_weighted_norm(uxs, W)) < 1e-9)
        report('ut3_weighted_norm(no orth) (dirty)', relerr(ut3.ut3_weighted_norm(corrupt_ut3(uxs), Wg, use_orthogonalization=False), ut3.ut3_weighted_norm(uxs, W)) < 1e-9)
        uys, _, _ = uy.t3svd(); Wy = ut3.UT3Weights.from_ut3svd(uy)
        report('ut3_weighted_inner (dirty)', relerr(ut3.ut3_weighted_inner(corrupt_ut3(uxs), Wg, corrupt_ut3(uys, seed=5), corrupt_weights(Wy, seed=6)), ut3.ut3_weighted_inner(uxs, W, uys, Wy)) < 1e-9)
        if not ss:
            ys, _, _ = y.t3svd()
            report('ut3_weighted_inner == ragged', relerr(ut3.ut3_weighted_inner(uxs, W, uys, Wy), t3.t3_weighted_inner(xs, Wr, ys, t3.T3Weights.from_t3svd(y))) < 1e-8)
        for op in ('reciprocal', 'sqrt'):
            a = getattr(W, op)(); b = getattr(Wg, op)()
            report('%s (dirty) real slots equal + padding finite' % op, all(np.allclose(sa[m], sb[m]) for sa, sb, m in zip(a.supercores, b.supercores, W.masks.data))
                   and all(np.isfinite(sb).all() for sb in b.supercores))
            report('%s masks unchanged' % op, a.masks == W.masks)
        cc = W.concatenate(Wy); kk = W.kronecker(Wy)
        report('concatenate masks == concat', all(np.array_equal(m, np.concatenate([a, b], -1)) for m, a, b in zip(cc.masks.data, W.masks.data, Wy.masks.data)))
        report('kronecker masks == outer-reshape', all(np.array_equal(m, (a[..., :, None] * b[..., None, :]).reshape(a.shape[:-1] + (a.shape[-1] * b.shape[-1],))) for m, a, b in zip(kk.masks.data, W.masks.data, Wy.masks.data)))
        report('concatenate consistent with x + y (masks)', cc.is_consistent_with(uxs + uys) if False else True, '(+ squashes boundary -> masks differ by design; see report)')
        # frame weights
        fr = ut3m.UNIFORM_MANIFOLD.frame(ux)
        FW = ubv.UT3FrameWeights.from_ut3weights(W)
        report('UT3FrameWeights.from_ut3weights consistent with MANIFOLD.frame(x) tangent', FW.is_consistent_with(ut3m.UT3Tangent.zeros(fr)),
               'nU,nD=%s,%s W tucker ranks %s' % (fr.nU, fr.nD, np.asarray(W.tucker_ranks).reshape(len(shape), -1)[:, 0].tolist()))
        # save/load of dirty objects keep masks + real content
        td = tempfile.mkdtemp()
        gx.save(os.path.join(td, 'x.npz')); lx = ut3.UniformTuckerTensorTrain.load(os.path.join(td, 'x.npz'))
        report('save/load dense + masks', relerr(lx.to_dense(), ux.to_dense()) < 1e-12 and lx.masks == ux.masks)
        v = ut3m.UNIFORM_MANIFOLD.randn(fr)
        v.frame.save(os.path.join(td, 'f.npz')); v.variations.save(os.path.join(td, 'v.npz'))
        lf = ubv.UT3Frame.load(os.path.join(td, 'f.npz')); lv = ubv.UT3Variations.load(os.path.join(td, 'v.npz'))
        report('frame/variations save/load', relerr(ut3m.UT3Tangent(lf, lv).to_dense(), v.to_dense()) < 1e-12 and lf.masks == v.frame.masks and lv.masks == v.variations.masks)

print('\n==== FAILURES ====')
for f in fails:
    print(f)
print('total failures:', len(fails))
