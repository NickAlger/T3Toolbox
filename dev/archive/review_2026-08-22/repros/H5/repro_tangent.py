"""Tangent layer (UT3Frame / UT3Variations / UT3Tangent / geometries): garbage-padding robustness, exact
masks, force-padded frames (retract output dims), varying-C sum_tangents, stack_frame, transport, weights."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from h5lib import *
import t3toolbox.manifold as t3m
import t3toolbox.frame_variations_format as bvf
import t3toolbox.safety as safety

fails = []
M, CW = ut3m.UNIFORM_MANIFOLD, ut3m.UNIFORM_COREWISE


def report(name, cond, detail=''):
    print('  %-64s %s %s' % (name, 'ok ' if cond else 'FAIL', detail))
    if not cond:
        fails.append((name, detail))


def vdense(t):  # dense of the tangent (real content only)
    return t.to_dense()


def run(shape, tr, ttr, ss, K, force_pad):
    np.random.seed(0)
    x = t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=ss)
    kw = dict(PAD) if force_pad else {}
    ux = ut3.UniformTuckerTensorTrain.from_t3(x, **kw)
    d = len(shape)
    print('case shape=%s tr=%s ttr=%s C=%s K=%s force_pad=%s' % (shape, tr, ttr, ss, K, force_pad))
    fr = M.frame(ux)
    v = M.randn(fr, stack_shape=K)
    w = M.randn(fr, stack_shape=K)
    gfr = corrupt_frame(fr); gv = ut3m.UT3Tangent(gfr, corrupt_variations(v.variations)); gw = ut3m.UT3Tangent(gfr, corrupt_variations(w.variations, seed=9))
    # frame checks
    report('frame.is_orthogonal (dirty)', bool(gfr.is_orthogonal().all()))
    report('frame.to_dense (dirty)', relerr(gfr.to_dense(), fr.to_dense()) < 1e-9)
    report('frame.to_ut3.norm (dirty)', relerr(gfr.to_ut3().norm(), fr.to_ut3().norm()) < 1e-9)
    report('frame.allclose(frame) (dirty -- routes via __sub__/squash)', bool(gfr.allclose(fr).all()))
    report('frame.orthogonalize (dirty) dense', relerr(gfr.orthogonalize().to_dense(), fr.to_dense()) < 1e-9)
    report('frame.reverse (dirty) dense', relerr(gfr.reverse().to_dense(), np.moveaxis(fr.to_dense(), list(range(len(ss), len(ss) + d)), list(range(len(ss) + d - 1, len(ss) - 1, -1)))) < 1e-9)
    report('frame.to_t3frame (dirty) dense', relerr(np.asarray([f.to_dense() for f in np.asarray(gfr.to_t3frame(), dtype=object).reshape(-1)]) if ss else gfr.to_t3frame().to_dense(),
                                                      np.asarray([f.to_dense() for f in np.asarray(fr.to_t3frame(), dtype=object).reshape(-1)]) if ss else fr.to_t3frame().to_dense()) < 1e-9)
    report('frame.has_minimal_ranks', True, str(np.asarray(fr.has_minimal_ranks).all()))
    report('frame.is_consistent (dirty)', bool(gfr.is_consistent().all()))
    # tangent ops
    report('tangent.to_dense (dirty)', relerr(vdense(gv), vdense(v)) < 1e-9)
    report('tangent.to_dense(include_shift) (dirty)', relerr(gv.to_dense(True), v.to_dense(True)) < 1e-9)
    report('corewise_inner (dirty)', relerr(gv.corewise_inner(gw), v.corewise_inner(w)) < 1e-9)
    report('MANIFOLD.inner (dirty)', relerr(M.inner(gv, gw), M.inner(v, w)) < 1e-9)
    report('gauge_residual (dirty)', relerr(gv.gauge_residual, v.gauge_residual) < 1e-6 or np.max(np.abs(gv.gauge_residual)) < 1e-9)
    report('project (dirty) dense', relerr(vdense(M.project(gv)), vdense(M.project(v))) < 1e-9)
    report('project_oblique (dirty) dense', relerr(vdense(M.project_oblique(gv)), vdense(M.project_oblique(v))) < 1e-9)
    rv, rgv = M.retract(v), M.retract(gv)
    report('retract (dirty) dense', relerr(rgv.to_dense(), rv.to_dense()) < 1e-9)
    report('retract masks clean==dirty', np.array_equal(rv.masks.data[0], rgv.masks.data[0]) and np.array_equal(rv.masks.data[1], rgv.masks.data[1]))
    report('retract output padded dims == frame padded dims (docstring claim)', (rv.n, rv.r) == (fr.nU, fr.rL), 'got n,r=%s,%s frame nU,rL=%s,%s' % (rv.n, rv.r, fr.nU, fr.rL))
    # retract vs ragged retract (per element)
    if not ss and not K:
        tv = v.to_t3tangent(); rr = t3m.MANIFOLD.retract(tv)
        report('retract dense == ragged retract', relerr(rv.to_dense(), rr.to_dense()) < 1e-8, '%.2e' % relerr(rv.to_dense(), rr.to_dense()))
        report('retract ranks == ragged retract ranks', tuple(rv.masks.data[0].sum(-1).tolist()) == tuple(rr.tucker_ranks) and tuple(rv.masks.data[1].sum(-1).tolist()) == tuple(rr.tt_ranks),
               '%s %s vs %s %s' % (rv.masks.data[0].sum(-1).tolist(), rv.masks.data[1].sum(-1).tolist(), rr.tucker_ranks, rr.tt_ranks))
    # to_ut3 exact masks (paper rule) on the dirty tangent
    du = gv.to_ut3()
    nU, nD, rL, rR = fr.nU, fr.nD, fr.rL, fr.rR
    upr, dnr, lr, rr_ = np.asarray(fr.up_ranks), np.asarray(fr.down_ranks), np.asarray(fr.left_ranks), np.asarray(fr.right_ranks)
    bc = lambda m: np.broadcast_to(m.reshape(m.shape[:1] + (1,) * len(K) + m.shape[1:]), m.shape[:1] + tuple(K) + m.shape[1:])
    exp_tk = bc(np.concatenate([prefix(upr, nU), prefix(dnr, nD)], -1))
    q = rr_.copy(); q[0] = 0; p = lr.copy(); p[-1] = 0
    exp_tt = bc(np.concatenate([prefix(q, rR), prefix(p, rL)], -1))
    report('to_ut3 exact masks (paper rule)', np.array_equal(du.masks.data[0], exp_tk) and np.array_equal(du.masks.data[1], exp_tt))
    # transport / project_ambient garbage
    fr2 = M.frame(ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=ss), **kw))
    if not K:
        tr_ = M.transport(v, fr2); tg_ = M.transport(gv, corrupt_frame(fr2, seed=5))
        report('transport (dirty) dense', relerr(vdense(tg_), vdense(tr_)) < 1e-9)
        pa = M.project_ambient(fr, ux); pg = M.project_ambient(gfr, corrupt_ut3(ux))
        report('project_ambient (dirty) dense', relerr(vdense(pg), vdense(pa)) < 1e-9)
        report('project_ambient masks == gauge masks', pa.variations.masks == ubv.UT3Variations._variation_masks_of(fr))
    # sum_tangents over K
    if K:
        st = v.sum_tangents(); gst = gv.sum_tangents()
        report('sum_tangents (dirty) dense', relerr(vdense(gst), vdense(st)) < 1e-9)
        report('sum_tangents dense == sum of unstacked', relerr(vdense(st), vdense(v).reshape((-1,) + ss + shape).sum(0)) < 1e-9)
        report('sum_tangents masks == gauge masks', st.variations.masks == ubv.UT3Variations._variation_masks_of(fr))
        # stack/unstack
        report('stack_tangents(unstack_tangents) (dirty) dense', relerr(vdense(ut3m.UT3Tangent.stack_tangents(gv.unstack_tangents())), vdense(v)) < 1e-9)
    if ss:
        report('stack_frame(unstack_frame) (dirty) dense', relerr(vdense(ut3m.UT3Tangent.stack_frame(gv.unstack_frame())), vdense(v)) < 1e-9)
    # corewise geometry
    cf = CW.frame(ux); cv = CW.randn(cf, stack_shape=K)
    gcf = corrupt_frame(cf); gcv = ut3m.UT3Tangent(gcf, corrupt_variations(cv.variations, seed=4))
    rc, rgc = CW.retract(cv), CW.retract(gcv)
    report('COREWISE.retract (dirty) dense', relerr(rgc.to_dense(), rc.to_dense()) < 1e-9)
    report('COREWISE.inner (dirty)', relerr(CW.inner(gcv, gcv), CW.inner(cv, cv)) < 1e-9)
    report('COREWISE.retract output masks == point masks (bcast over K)', np.array_equal(rc.masks.data[0][(slice(None),) + (0,) * len(K)], ux.masks.data[0]) if K else rc.masks == ux.masks)
    # weights
    W = ubv.UT3FrameWeights.from_ut3weights(ut3.UT3Weights.from_ut3svd(ux))
    if W.is_consistent_with(v):
        report('weighted_norm (dirty)', relerr(gv.weighted_norm(W), v.weighted_norm(W)) < 1e-9)
        report('weighted_inner (dirty)', relerr(gv.weighted_inner(gw, W), v.weighted_inner(w, W)) < 1e-9)
        report('absorb_weights (dirty) corewise_norm', relerr(gv.absorb_weights(W).corewise_norm(), v.absorb_weights(W).corewise_norm()) < 1e-9)
    else:
        report('UT3FrameWeights.from_ut3weights(from_ut3svd(x)) consistent with tangent at frame(x)', False, 'nU,nD=%s,%s' % (fr.nU, fr.nD))
    # to_t3tangent / from_t3tangent round trip on dirty
    if not K and not ss:
        report('to_t3tangent (dirty) dense', relerr(gv.to_t3tangent().to_dense(), v.to_dense()) < 1e-9)
        back = ut3m.UT3Tangent.from_t3tangent(v.to_t3tangent())
        report('from_t3tangent(to_t3tangent) dense', relerr(back.to_dense(), v.to_dense()) < 1e-9)
    # variations: zeros/randn default masks all-True; sum_stack of variations; allclose
    report('variations.allclose (dirty vs clean)', bool(gv.variations.allclose(v.variations).all()))
    try:
        report('tangent.allclose (dirty vs clean) -- same-frame guard on padding?', bool(gv.allclose(v).all()))
    except ValueError as e:
        report('tangent.allclose (dirty vs clean) -- same-frame guard on padding?', False, 'raises: ' + str(e)[:90])
    report('normalized (dirty) dense', relerr(vdense(gv.normalized()), vdense(v.normalized())) < 1e-9)


CFG = [  # (case index, K, force_pad)
    (0, (), False), (0, (), True), (0, (2,), False), (0, (2,), True),
    (1, (), True), (2, (), False), (2, (2,), True), (3, (), True), (3, (2,), False),
]
for ci, K, fp in CFG:
    shape, tr, ttr, ss = CASES[ci]
    try:
        run(shape, tr, ttr, ss, K, fp)
    except Exception as e:
        import traceback; traceback.print_exc()
        fails.append(('EXC %s %s %s %s' % (shape, ss, K, fp), type(e).__name__ + ': ' + str(e)[:300]))

# ---- varying-rank C stack: sum over C of variations (OR masks) and stack_frame of different ranks
print('varying-rank frame stack')
ust, (xa, xb) = varying_stack()
fr = M.frame(ust)
v = M.randn(fr)
report('varying: frame masks per element == per-element ragged frame ranks',
       all(tuple(np.asarray(fr.up_ranks)[:, i].tolist()) == tuple(bvf.T3Frame.from_t3(xx).up_ranks) for i, xx in enumerate((xa, xb))),
       '%s' % [np.asarray(fr.up_ranks)[:, i].tolist() for i in range(2)])
vs = v.variations.sum_stack()
report('varying: UT3Variations.sum_stack masks == OR of element masks', all(np.array_equal(m, np.any(m0, axis=1)) for m, m0 in zip(vs.masks.data, v.variations.masks.data)))
rt = M.retract(v)
for i, xx in enumerate((xa, xb)):
    tv = v.unstack_frame()[i].to_t3tangent()
    rr = t3m.MANIFOLD.retract(tv)
    report('varying: retract elem %d dense == ragged' % i, relerr(rt.to_dense()[i], rr.to_dense()) < 1e-8, '%.2e' % relerr(rt.to_dense()[i], rr.to_dense()))
    report('varying: retract elem %d ranks == ragged' % i, tuple(rt.masks.data[0][:, i].sum(-1).tolist()) == tuple(rr.tucker_ranks) and tuple(rt.masks.data[1][:, i].sum(-1).tolist()) == tuple(rr.tt_ranks),
           '%s %s vs %s %s' % (rt.masks.data[0][:, i].sum(-1).tolist(), rt.masks.data[1][:, i].sum(-1).tolist(), rr.tucker_ranks, rr.tt_ranks))
gv = ut3m.UT3Tangent(corrupt_frame(fr), corrupt_variations(v.variations))
report('varying: retract (dirty) dense', relerr(M.retract(gv).to_dense(), rt.to_dense()) < 1e-9)
report('varying: project (dirty) dense', relerr(M.project(gv).to_dense(), M.project(v).to_dense()) < 1e-9)
report('varying: inner (dirty)', relerr(M.inner(gv, gv), M.inner(v, v)) < 1e-9)

print('\n==== FAILURES ====')
for f in fails:
    print(f)
print('total failures:', len(fails))
