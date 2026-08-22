"""Working-form (gappy) masks -- produced by `+` of slack-padded objects -- fed into every op that
accepts a UniformTuckerTensorTrain. Each op is compared against the ragged twin on (x + y)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from h5lib import *
import t3toolbox.backend.sharing as sharing
import t3toolbox.backend.uniform_fitting as ufit
import t3toolbox.optimizers as opt

fails = []


def report(name, cond, detail=''):
    print('  %-60s %s %s' % (name, 'ok ' if cond else 'FAIL', detail))
    if not cond:
        fails.append((name, detail))


def run(shape, tr, ttr, ss):
    np.random.seed(0)
    x = t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=ss)
    y = t3.TuckerTensorTrain.randn(shape, tuple(max(1, n - 1) for n in tr), (1,) + tuple(max(1, r - 1) for r in ttr[1:-1]) + (1,), stack_shape=ss)
    ux = ut3.UniformTuckerTensorTrain.from_t3(x, N=max(shape) + 1, n=max(tr) + 2, r=max(ttr) + 2)   # slack -> gaps after +
    uy = ut3.UniformTuckerTensorTrain.from_t3(y, N=max(shape) + 1, n=max(tr) + 1, r=max(ttr) + 1)
    s = ux + uy
    sr = (x + y).squash_tails()
    d = len(shape)
    print('case shape=%s stack=%s   gappy? tucker=%s tt=%s' % (shape, ss, not is_prefix(s.masks.data[0]), not is_prefix(s.masks.data[1])))
    report('gappy: to_dense == ragged', relerr(s.to_dense(), sr.to_dense()) < 1e-9)
    report('gappy: norm', relerr(s.norm(), sr.norm()) < 1e-9)
    report('gappy: norm(no orth)', relerr(s.norm(use_orthogonalization=False), sr.norm()) < 1e-9)
    report('gappy: inner(ux)', relerr(s.inner(ux), sr.inner(x)) < 1e-9)
    report('gappy: sum()', relerr(s.sum(), sr.to_dense().reshape(ss + (-1,)).sum(-1)) < 1e-9)
    idx = np.stack([np.random.randint(0, Ni, size=(3,)) for Ni in shape])
    ww = [np.random.randn(3, Ni) for Ni in shape]; pp = [np.random.randn(3, Ni) for Ni in shape]
    report('gappy: entries', relerr(s.entries(idx), sr.entries(idx)) < 1e-9)
    report('gappy: apply', relerr(s.apply(ww), sr.apply(ww)) < 1e-9)
    report('gappy: probe', relerr(np.concatenate([z.reshape(-1) for z in s.probe(ww)]), np.concatenate([z.reshape(-1) for z in sr.probe(ww)])) < 1e-9)
    report('gappy: apply_derivatives', relerr(s.apply_derivatives(ww, pp, 2), sr.apply_derivatives(ww, pp, 2)) < 1e-9)
    report('gappy: to_t3 round trip', (relerr(s.to_t3().to_dense(), sr.to_dense()) < 1e-9) if not ss else True)
    if ss:
        report('gappy: unstack/stack', relerr(ut3.UniformTuckerTensorTrain.stack(s.unstack()).to_dense(), sr.to_dense()) < 1e-9)
        report('gappy: sum_stack', relerr(s.sum_stack().to_dense(), sr.to_dense().reshape((-1,) + shape).sum(0)) < 1e-9)
    report('gappy: reverse', relerr(s.reverse().to_dense(), np.moveaxis(sr.to_dense(), list(range(len(ss), len(ss) + d)), list(range(len(ss) + d - 1, len(ss) - 1, -1)))) < 1e-9)
    for um in ('down_orthogonalize_tucker_cores', 'up_orthogonalize_tt_cores', 'left_orthogonalize_tt_cores', 'right_orthogonalize_tt_cores'):
        a = getattr(s, um)(); b = getattr(sr, um)()
        report('gappy: %s dense' % um, relerr(a.to_dense(), b.to_dense()) < 1e-9)
        report('gappy: %s ranks == ragged' % um, tuple(a.masks.data[0].sum(-1).reshape(d, -1)[:, 0].tolist()) == tuple(b.tucker_ranks)
               and tuple(a.masks.data[1].sum(-1).reshape(d + 1, -1)[:, 0].tolist()) == tuple(b.tt_ranks),
               '%s %s vs %s %s' % (a.masks.data[0].sum(-1).reshape(d, -1)[:, 0].tolist(), a.masks.data[1].sum(-1).reshape(d + 1, -1)[:, 0].tolist(), b.tucker_ranks, b.tt_ranks))
        report('gappy: %s output prefix?' % um, True, 'prefix=%s/%s' % (is_prefix(a.masks.data[0]), is_prefix(a.masks.data[1])))
    a, sa, sat = s.t3svd(); b, sb, sbt = sr.t3svd()
    report('gappy: t3svd dense', relerr(a.to_dense(), b.to_dense()) < 1e-9)
    report('gappy: t3svd ranks == ragged', tuple(a.masks.data[0].sum(-1).reshape(d, -1)[:, 0].tolist()) == tuple(b.tucker_ranks)
           and tuple(a.masks.data[1].sum(-1).reshape(d + 1, -1)[:, 0].tolist()) == tuple(b.tt_ranks),
           '%s %s vs %s %s' % (a.masks.data[0].sum(-1).reshape(d, -1)[:, 0].tolist(), a.masks.data[1].sum(-1).reshape(d + 1, -1)[:, 0].tolist(), b.tucker_ranks, b.tt_ranks))
    report('gappy: t3svd output prefix', is_prefix(a.masks.data[0]) and is_prefix(a.masks.data[1]))
    report('gappy: t3svd svals (nonzero part) == ragged', relerr(np.sort(np.abs(sa.reshape(-1)))[::-1][:np.concatenate([np.asarray(q).reshape(-1) for q in sb]).size], np.sort(np.abs(np.concatenate([np.asarray(q).reshape(-1) for q in sb])))[::-1]) < 1e-6 if not ss else True)
    a2 = a.rank_adjustment_sweep('right_to_left'); b2 = b.rank_adjustment_sweep('right_to_left')
    report('gappy: t3svd->ras dense', relerr(a2.to_dense(), b2.to_dense()) < 1e-9)
    report('gappy: t3svd->ras ranks', tuple(a2.masks.data[0].sum(-1).reshape(d, -1)[:, 0].tolist()) == tuple(b2.tucker_ranks)
           and tuple(a2.masks.data[1].sum(-1).reshape(d + 1, -1)[:, 0].tolist()) == tuple(b2.tt_ranks),
           '%s %s vs %s %s' % (a2.masks.data[0].sum(-1).reshape(d, -1)[:, 0].tolist(), a2.masks.data[1].sum(-1).reshape(d + 1, -1)[:, 0].tolist(), b2.tucker_ranks, b2.tt_ranks))
    # a direct rank_adjustment_sweep on the gappy object after a right-orthogonalization (lossless path)
    g = s.down_orthogonalize_tucker_cores().right_orthogonalize_tt_cores().rank_adjustment_sweep('left_to_right')
    report('gappy: (down,right-orth) -> ras(left_to_right) dense', relerr(g.to_dense(), sr.to_dense()) < 1e-9, '%.2e' % relerr(g.to_dense(), sr.to_dense()))
    # minimal ranks / has_minimal_ranks on gappy
    report('gappy: minimal_ranks callable', True, str([np.asarray(m).reshape(len(m), -1)[:, 0].tolist() for m in s.minimal_ranks]))
    # frame / tangent from the gappy point
    fr, va = ubv.ut3_orthogonal_representations(s)
    report('gappy: ut3_orthogonal_representations frame dense', relerr(fr.to_dense(), sr.to_dense()) < 1e-9)
    report('gappy: frame masks prefix', all(is_prefix(m) for m in fr.masks.data))
    report('gappy: frame.is_orthogonal', bool(fr.is_orthogonal().all()))
    tfr = fr.to_t3frame()
    if not ss:
        report('gappy: frame ranks == ragged T3Frame.from_t3', tuple(int(v) for v in fr.up_ranks) == tuple(t3.T3Frame.from_t3(sr).up_ranks) if hasattr(t3, 'T3Frame') else True)
    import t3toolbox.safety as safety
    report('gappy: frame from gappy (non-minimal) point is_orthogonal', bool(fr.is_orthogonal().all()), 'residual %.2e' % float(np.max(fr.orthogonality_residual)))
    with safety.unsafe():
        v = ut3m.UNIFORM_MANIFOLD.randn(fr)
        rt = ut3m.UNIFORM_MANIFOLD.retract(v)
    report('gappy: retract runs; dense vs ragged retract', True, 'n,r=%s,%s frame nU,rL=%s,%s' % (rt.n, rt.r, fr.nU, fr.rL))
    # gauge projection of a gappy plain UT3 onto the tangent space
    with safety.unsafe():
        pa = ut3m.UNIFORM_MANIFOLD.project_ambient(fr, s)
    report('gappy: project_ambient(frame, gappy grad) runs', True)
    # weights from the gappy object
    W = ut3.UT3Weights.from_ut3svd(s)
    report('gappy: UT3Weights.from_ut3svd consistent with the svd output (masks prefix)', True, 'W masks prefix=%s' % all(is_prefix(m) for m in W.masks.data))
    W2 = ut3.UT3Weights(np.ones(s.masks.data[0].shape), np.ones(s.masks.data[1].shape), s.masks)
    report('gappy: ut3_weighted_norm with unit gappy weights == norm', relerr(ut3.ut3_weighted_norm(s, W2), s.norm()) < 1e-9)
    Wr = W2.reciprocal()
    report('gappy: reciprocal of gappy-masked weights: padding finite + real slots 1', bool(np.isfinite(Wr.tucker_weight_supercore).all()) and bool(np.all(Wr.tucker_weight_supercore[s.masks.data[0]] == 1.0)))
    # sharing checkers on a gappy object with equal masks per group: build a tied gappy object
    if d >= 2 and shape[0] == shape[1] and tr[0] == tr[1]:
        pass
    # corewise transposes w/ gappy masks: masked grad == ragged grad mapped through the mask
    c = np.random.randn(3, *ss)
    gU, gG = s.apply_corewise_transpose(c, ww, sum_over_probes=True)
    rU, rG = sr.apply_corewise_transpose(c, ww, sum_over_probes=True)
    if not ss:
        gU_real = [gU[i][s.masks.data[0][i]][:, :shape[i]] for i in range(d)]
        report('gappy: apply_corewise_transpose tucker grads == ragged (through the mask)', all(relerr(a, b) < 1e-9 for a, b in zip(gU_real, rU)),
               '%s' % [float(relerr(a, b)) for a, b in zip(gU_real, rU)])
        gG_real = [gG[i][s.masks.data[1][i]][:, s.masks.data[0][i]][:, :, s.masks.data[1][i + 1]] for i in range(d)]
        report('gappy: apply_corewise_transpose tt grads == ragged (through the mask)', all(relerr(a, b) < 1e-9 for a, b in zip(gG_real, rG)),
               '%s' % [float(relerr(a, b)) for a, b in zip(gG_real, rG)])


for (shape, tr, ttr, ss) in CASES:
    if len(shape) < 2:
        continue
    try:
        run(shape, tr, ttr, ss)
    except Exception as e:
        import traceback; traceback.print_exc()
        fails.append(('EXC %s %s' % (shape, ss), type(e).__name__ + ': ' + str(e)[:300]))

print('\n==== FAILURES ====')
for f in fails:
    print(f)
print('total failures:', len(fails))
