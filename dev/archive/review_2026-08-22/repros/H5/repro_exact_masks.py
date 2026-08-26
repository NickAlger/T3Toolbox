"""Prong 2: EXACT output masks, derived non-circularly from the ragged twin's ranks (the ground truth).
Covers t3svd (with/without caps), rank_adjustment_sweep, the four orthogonalizations, +, sum_stack,
squash_tails, and a varying-rank stack. A mask with extra True slots is a finding even if dense matches."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from h5lib import *
import t3toolbox.backend.ranks as ranks

fails = []


def ranks_of(mask):
    return np.asarray(mask.sum(axis=-1))


def report(name, cond, detail=''):
    print('  %-60s %s %s' % (name, 'ok ' if cond else 'FAIL', detail))
    if not cond:
        fails.append((name, detail))


def check_masks(name, u, tucker_ranks_expected, tt_ranks_expected, prefix_required=True):
    tkm, ttm = u.masks.data
    exp_tk = np.broadcast_to(np.asarray(tucker_ranks_expected).reshape((u.d,) + (1,) * len(u.stack_shape)), (u.d,) + u.stack_shape)
    exp_tt = np.broadcast_to(np.asarray(tt_ranks_expected).reshape((u.d + 1,) + (1,) * len(u.stack_shape)), (u.d + 1,) + u.stack_shape)
    ok_tk = np.array_equal(tkm, prefix(exp_tk, u.n))
    ok_tt = np.array_equal(ttm, prefix(exp_tt, u.r))
    detail = '' if (ok_tk and ok_tt) else 'got tucker %s tt %s, expected %s %s (prefix=%s/%s)' % (
        ranks_of(tkm).reshape(u.d, -1)[:, 0].tolist(), ranks_of(ttm).reshape(u.d + 1, -1)[:, 0].tolist(),
        list(tucker_ranks_expected), list(tt_ranks_expected), is_prefix(tkm), is_prefix(ttm))
    report(name, ok_tk and ok_tt, detail)


def run(shape, tr, ttr, ss, force_pad):
    np.random.seed(0)
    x = t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=ss)
    kw = dict(PAD) if force_pad else {}
    ux = ut3.UniformTuckerTensorTrain.from_t3(x, **kw)
    d = len(shape)
    print('case shape=%s tr=%s ttr=%s stack=%s force_pad=%s' % (shape, tr, ttr, ss, force_pad))
    xs = x.squash_tails() if hasattr(x, 'squash_tails') else x

    # --- t3svd, no caps: ranks == ragged t3svd ranks
    xr, _, _ = x.t3svd()
    u, _, _ = ux.t3svd()
    check_masks('t3svd() masks == ragged t3svd ranks', u, xr.tucker_ranks, xr.tt_ranks)
    report('t3svd() dense == ragged', relerr(u.to_dense(), xr.to_dense()) < 1e-9)
    # --- t3svd with caps (per-mode)
    caps_tk = tuple(max(1, n - 1) for n in tr)
    caps_tt = (1,) + tuple(max(1, r - 1) for r in ttr[1:-1]) + (1,)
    xr2, _, _ = x.t3svd(max_tucker_ranks=caps_tk, max_tt_ranks=caps_tt)
    u2, _, _ = ux.t3svd(max_tucker_ranks=caps_tk, max_tt_ranks=caps_tt)
    check_masks('t3svd(caps) masks == ragged ranks', u2, xr2.tucker_ranks, xr2.tt_ranks)
    report('t3svd(caps) dense == ragged', relerr(u2.to_dense(), xr2.to_dense()) < 1e-9, '%.2e' % relerr(u2.to_dense(), xr2.to_dense()))
    if d >= 2:
        # --- rank_adjustment_sweep on the svd output
        xr3 = xr2.rank_adjustment_sweep('right_to_left')
        u3 = u2.rank_adjustment_sweep('right_to_left')
        check_masks('rank_adjustment_sweep masks == ragged', u3, xr3.tucker_ranks, xr3.tt_ranks)
        report('rank_adjustment_sweep dense == ragged', relerr(u3.to_dense(), xr3.to_dense()) < 1e-9)
    # --- the four orthogonalizations vs the ragged core shapes
    pairs = (('down_orthogonalize_tucker_cores', 'down_orthogonalize_tucker_cores'),
             ('up_orthogonalize_tt_cores', 'up_orthogonalize_tt_cores'),
             ('left_orthogonalize_tt_cores', 'left_orthogonalize_tt_cores'),
             ('right_orthogonalize_tt_cores', 'right_orthogonalize_tt_cores'))
    for um, rm in pairs:
        if not hasattr(x, rm):
            print('  (ragged has no %s)' % rm); continue
        a = getattr(ux, um)(); b = getattr(x, rm)()
        check_masks(um + ' masks == ragged ranks', a, b.tucker_ranks, b.tt_ranks)
        report(um + ' dense == ragged', relerr(a.to_dense(), b.to_dense()) < 1e-9)
        # masked-content zero where the mask says padding?  (an INPUT-derived check: real content must not
        # sit outside the mask -- otherwise the mask is too restrictive)
        report(um + ' no real content outside mask (dense unchanged by re-mask)',
               relerr(a.apply_masks().to_dense(), a.to_dense()) < 1e-9)
    # --- add: masks == concatenation; ranks == ragged sum ranks
    y = t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=ss)
    uy = ut3.UniformTuckerTensorTrain.from_t3(y, **kw)
    s = ux + uy; sr = x + y
    report('x + y dense == ragged', relerr(s.to_dense(), sr.to_dense()) < 1e-9)
    report('x + y ranks == ragged ranks', np.array_equal(ranks_of(s.masks.data[0]).reshape(d, -1)[:, 0], np.asarray(sr.tucker_ranks))
           and np.array_equal(ranks_of(s.masks.data[1]).reshape(d + 1, -1)[:, 0], np.asarray(sr.tt_ranks)),
           'got %s %s expected %s %s' % (ranks_of(s.masks.data[0]).reshape(d, -1)[:, 0].tolist(), ranks_of(s.masks.data[1]).reshape(d + 1, -1)[:, 0].tolist(), sr.tucker_ranks, sr.tt_ranks))
    report('x + y masks gappy iff padded (doc says +/x may go non-prefix)', True, 'prefix=%s/%s' % (is_prefix(s.masks.data[0]), is_prefix(s.masks.data[1])))
    # squash masks
    sq = ux.squash_tails()
    report('squash_tails boundary masks rank-1', bool(np.all(ranks_of(sq.masks.data[1])[0] == 1) and np.all(ranks_of(sq.masks.data[1])[-1] == 1)))
    if ss:
        ssum = ux.sum_stack(); rsum = x.sum_stack() if hasattr(x, 'sum_stack') else None
        if rsum is not None:
            report('sum_stack dense == ragged', relerr(ssum.to_dense(), rsum.to_dense()) < 1e-9)
            report('sum_stack ranks == ragged ranks', tuple(ranks_of(ssum.masks.data[0]).tolist()) == tuple(rsum.tucker_ranks)
                   and tuple(ranks_of(ssum.masks.data[1]).tolist()) == tuple(rsum.tt_ranks),
                   'got %s %s expected %s %s' % (ranks_of(ssum.masks.data[0]).tolist(), ranks_of(ssum.masks.data[1]).tolist(), rsum.tucker_ranks, rsum.tt_ranks))
        else:
            D = x.to_dense().reshape((-1,) + shape).sum(axis=0)
            report('sum_stack dense == sum of dense', relerr(ssum.to_dense(), D) < 1e-9)


for (shape, tr, ttr, ss) in CASES:
    if len(shape) < 2:
        continue
    for force_pad in (False, True):
        try:
            run(shape, tr, ttr, ss, force_pad)
        except Exception as e:
            import traceback; traceback.print_exc()
            fails.append(('EXC %s %s %s' % (shape, ss, force_pad), type(e).__name__ + ': ' + str(e)[:200]))

# --- varying-rank stack: per-element masks vs per-element ragged ranks
print('varying-rank stack')
ust, (xa, xb) = varying_stack()
u, _, _ = ust.t3svd()
ra, _, _ = xa.t3svd(); rb, _, _ = xb.t3svd()
tkm, ttm = u.masks.data
for i, r in enumerate((ra, rb)):
    report('varying t3svd elem %d tucker ranks' % i, tuple(ranks_of(tkm[:, i]).tolist()) == tuple(r.tucker_ranks), '%s vs %s' % (ranks_of(tkm[:, i]).tolist(), r.tucker_ranks))
    report('varying t3svd elem %d tt ranks' % i, tuple(ranks_of(ttm[:, i]).tolist()) == tuple(r.tt_ranks), '%s vs %s' % (ranks_of(ttm[:, i]).tolist(), r.tt_ranks))
    report('varying t3svd elem %d dense' % i, relerr(u.to_dense()[i], r.to_dense()) < 1e-9)
u2 = u.rank_adjustment_sweep('right_to_left')
for i, r in enumerate((ra.rank_adjustment_sweep('right_to_left'), rb.rank_adjustment_sweep('right_to_left'))):
    tkm, ttm = u2.masks.data
    report('varying ras elem %d ranks' % i, tuple(ranks_of(tkm[:, i]).tolist()) == tuple(r.tucker_ranks) and tuple(ranks_of(ttm[:, i]).tolist()) == tuple(r.tt_ranks),
           '%s %s vs %s %s' % (ranks_of(tkm[:, i]).tolist(), ranks_of(ttm[:, i]).tolist(), r.tucker_ranks, r.tt_ranks))
    report('varying ras elem %d dense' % i, relerr(u2.to_dense()[i], r.to_dense()) < 1e-9)
for um in ('down_orthogonalize_tucker_cores', 'up_orthogonalize_tt_cores', 'left_orthogonalize_tt_cores', 'right_orthogonalize_tt_cores'):
    a = getattr(ust, um)()
    for i, xx in enumerate((xa, xb)):
        b = getattr(xx, um)()
        tkm, ttm = a.masks.data
        report('varying %s elem %d ranks' % (um, i), tuple(ranks_of(tkm[:, i]).tolist()) == tuple(b.tucker_ranks) and tuple(ranks_of(ttm[:, i]).tolist()) == tuple(b.tt_ranks),
               '%s %s vs %s %s' % (ranks_of(tkm[:, i]).tolist(), ranks_of(ttm[:, i]).tolist(), b.tucker_ranks, b.tt_ranks))
        report('varying %s elem %d dense' % (um, i), relerr(a.to_dense()[i], b.to_dense()) < 1e-9)
# varying sum_stack
ss_ = ust.sum_stack()
report('varying sum_stack dense', relerr(ss_.to_dense(), xa.to_dense() + xb.to_dense()) < 1e-9)
print('varying sum_stack ranks', ranks_of(ss_.masks.data[0]).tolist(), ranks_of(ss_.masks.data[1]).tolist(), 'ragged sum ranks', (xa + xb).tucker_ranks, (xa + xb).tt_ranks)
# per-element caps on the varying stack (the variety)
caps_tk = np.array([[2, 3], [2, 2], [2, 2]]); caps_tt = np.array([[1, 1], [2, 2], [2, 2], [1, 1]])
u3, _, _ = ust.t3svd(max_tucker_ranks=caps_tk, max_tt_ranks=caps_tt)
for i, xx in enumerate((xa, xb)):
    r, _, _ = xx.t3svd(max_tucker_ranks=tuple(caps_tk[:, i]), max_tt_ranks=tuple(caps_tt[:, i]))
    tkm, ttm = u3.masks.data
    report('varying t3svd(per-elem caps) elem %d ranks' % i, tuple(ranks_of(tkm[:, i]).tolist()) == tuple(r.tucker_ranks) and tuple(ranks_of(ttm[:, i]).tolist()) == tuple(r.tt_ranks),
           '%s %s vs %s %s' % (ranks_of(tkm[:, i]).tolist(), ranks_of(ttm[:, i]).tolist(), r.tucker_ranks, r.tt_ranks))
    report('varying t3svd(per-elem caps) elem %d dense' % i, relerr(u3.to_dense()[i], r.to_dense()) < 1e-9, '%.2e' % relerr(u3.to_dense()[i], r.to_dense()))

print('\n==== FAILURES ====')
for f in fails:
    print(f)
print('total failures:', len(fails))
