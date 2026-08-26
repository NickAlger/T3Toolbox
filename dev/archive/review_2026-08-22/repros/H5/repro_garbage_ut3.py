"""Prong 3 for every UniformTuckerTensorTrain op: garbage in the padding must not change the real output.
Runs each op on a clean object and on the same object with 1e3*randn in its padding; reports ops whose
real (masked / dense) output differs."""
import sys, os, tempfile
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from h5lib import *
import t3toolbox.backend.ut3_operations as ut3_ops
import t3toolbox.backend.ut3_linalg as ut3_linalg

TOL = 1e-8
fails = []


def check(name, clean_val, dirty_val, tol=TOL):
    e = relerr(dirty_val, clean_val)
    ok = e <= tol
    if not ok:
        fails.append((name, e))
    print('  %-52s %s  relerr=%.3e' % (name, 'ok ' if ok else 'FAIL', e))


def run_case(shape, tr, ttr, ss, force_pad, squash):
    np.random.seed(0)
    x = t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=ss)
    kw = dict(PAD) if force_pad else {}
    ux = ut3.UniformTuckerTensorTrain.from_t3(x, squash_tails=squash, **kw)
    gx = corrupt_ut3(ux)
    y = t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=ss)
    uy = ut3.UniformTuckerTensorTrain.from_t3(y, squash_tails=squash, **kw)
    gy = corrupt_ut3(uy, seed=7)
    d = len(shape)
    print('case shape=%s tr=%s ttr=%s stack=%s force_pad=%s squash_on_convert=%s' % (shape, tr, ttr, ss, force_pad, squash))

    check('to_dense', ux.to_dense(), gx.to_dense())
    check('apply_masks.to_dense', ux.apply_masks().to_dense(), gx.apply_masks().to_dense())
    check('reverse.to_dense', ux.reverse().to_dense(), gx.reverse().to_dense())
    check('squash_tails.to_dense', ux.squash_tails().to_dense(), gx.squash_tails().to_dense())
    check('x + y  (to_dense)', (ux + uy).to_dense(), (gx + gy).to_dense())
    check('x - y  (to_dense)', (ux - uy).to_dense(), (gx - gy).to_dense())
    check('x + y  (x dirty only)', (ux + uy).to_dense(), (gx + uy).to_dense())
    check('2.0 * x', (2.0 * ux).to_dense(), (2.0 * gx).to_dense())
    if ss:
        check('sum_stack.to_dense', ux.sum_stack().to_dense(), gx.sum_stack().to_dense())
    check('norm()', ux.norm(), gx.norm())
    check('norm(use_orthogonalization=False)', ux.norm(use_orthogonalization=False), gx.norm(use_orthogonalization=False))
    check('inner(y)', ux.inner(uy), gx.inner(gy))
    check('inner(y, use_orth=False)', ux.inner(uy, use_orthogonalization=False), gx.inner(gy, use_orthogonalization=False))
    check('sum()', ux.sum(), gx.sum())
    idx = np.stack([np.random.randint(0, Ni, size=(3,)) for Ni in shape])
    check('entries', ux.entries(idx), gx.entries(idx))
    ww = [np.random.randn(3, Ni) for Ni in shape]
    pp = [np.random.randn(3, Ni) for Ni in shape]
    check('apply', ux.apply(ww), gx.apply(ww))
    check('probe', np.concatenate([z.reshape(-1) for z in ux.probe(ww)]), np.concatenate([z.reshape(-1) for z in gx.probe(ww)]))
    check('probe_derivatives', np.concatenate([z.reshape(-1) for z in ux.probe_derivatives(ww, pp, 2)]),
          np.concatenate([z.reshape(-1) for z in gx.probe_derivatives(ww, pp, 2)]))
    check('apply_derivatives', ux.apply_derivatives(ww, pp, 2), gx.apply_derivatives(ww, pp, 2))
    check('entries_derivatives', ux.entries_derivatives(idx, pp, 2), gx.entries_derivatives(idx, pp, 2))
    c = np.random.randn(3, *ss)
    def masked_grads(u, g):
        return np.concatenate([a.reshape(-1) for a in ut3.UniformTuckerTensorTrain(g[0], g[1], u.shape, u.masks).apply_masks().supercores])
    check('apply_corewise_transpose (masked grads)', masked_grads(ux, ux.apply_corewise_transpose(c, ww, sum_over_probes=True)),
          masked_grads(gx, gx.apply_corewise_transpose(c, ww, sum_over_probes=True)))
    check('entries_corewise_transpose (masked grads)', masked_grads(ux, ux.entries_corewise_transpose(c, idx, sum_over_probes=True)),
          masked_grads(gx, gx.entries_corewise_transpose(c, idx, sum_over_probes=True)))
    zt = [np.random.randn(3, *ss, Ni) for Ni in shape]
    check('probe_corewise_transpose (masked grads)', masked_grads(ux, ux.probe_corewise_transpose(zt, ww, sum_over_probes=True)),
          masked_grads(gx, gx.probe_corewise_transpose(zt, ww, sum_over_probes=True)))
    cj = np.random.randn(3, 3, *ss)
    check('apply_corewise_derivatives_transpose', masked_grads(ux, ux.apply_corewise_derivatives_transpose(cj, ww, pp, 2, sum_over_probes=True)),
          masked_grads(gx, gx.apply_corewise_derivatives_transpose(cj, ww, pp, 2, sum_over_probes=True)))
    for meth in ('down_orthogonalize_tucker_cores', 'up_orthogonalize_tt_cores', 'left_orthogonalize_tt_cores', 'right_orthogonalize_tt_cores'):
        a = getattr(ux, meth)(); b = getattr(gx, meth)()
        check(meth + '.to_dense', a.to_dense(), b.to_dense())
        if not (np.array_equal(a.masks.data[0], b.masks.data[0]) and np.array_equal(a.masks.data[1], b.masks.data[1])):
            fails.append((meth + ' masks differ', 1.0)); print('  %-52s FAIL masks differ' % meth)
    check('is_left_orthogonal (after left-orth)', ux.down_orthogonalize_tucker_cores().left_orthogonalize_tt_cores().is_left_orthogonal().astype(float),
          gx.down_orthogonalize_tucker_cores().left_orthogonalize_tt_cores().is_left_orthogonal().astype(float))
    a, sa, sat = ux.t3svd(); b, sb, sbt = gx.t3svd()
    check('t3svd.to_dense', a.to_dense(), b.to_dense())
    check('t3svd svals (tucker)', sa, sb); check('t3svd svals (tt)', sat, sbt)
    if not (np.array_equal(a.masks.data[0], b.masks.data[0]) and np.array_equal(a.masks.data[1], b.masks.data[1])):
        fails.append(('t3svd masks differ', 1.0)); print('  t3svd masks differ FAIL')
    if d >= 2:
        a2 = a.rank_adjustment_sweep('right_to_left'); b2 = b.rank_adjustment_sweep('right_to_left')
        check('t3svd -> rank_adjustment_sweep', a2.to_dense(), b2.to_dense())
        ca = corrupt_ut3(a, seed=11)
        check('rank_adjustment_sweep on garbage-padded svd output', a2.to_dense(), ca.rank_adjustment_sweep('right_to_left').to_dense())
    # ragged round trip
    ra, rb = ux.to_t3(), gx.to_t3()
    if ss:
        fa = [t for t in np.asarray(ra, dtype=object).reshape(-1)] if False else None
    else:
        check('to_t3.to_dense', ra.to_dense(), rb.to_dense())
    if ss:
        check('unstack/stack', ut3.UniformTuckerTensorTrain.stack(ux.unstack()).to_dense(), ut3.UniformTuckerTensorTrain.stack(gx.unstack()).to_dense())
    # frame / variations
    if d >= 2:
        fa, va = ubv.ut3_orthogonal_representations(ux); fb, vb = ubv.ut3_orthogonal_representations(gx)
        check('ut3_orthogonal_representations frame.to_dense', fa.to_dense(), fb.to_dense())
        check('ut3_orthogonal_representations variations (masked)', np.concatenate([s.reshape(-1) for s in va.apply_masks().supercores]),
              np.concatenate([s.reshape(-1) for s in vb.apply_masks().supercores]))
        check('UT3Frame.from_ut3.is_orthogonal', fa.is_orthogonal().astype(float), fb.is_orthogonal().astype(float))
    # weights
    W = ut3.UT3Weights.from_ut3svd(ux)
    check('absorb_weights.to_dense', ux.absorb_weights(W).to_dense(), gx.absorb_weights(W).to_dense())
    check('weighted_norm', ux.weighted_norm(W), gx.weighted_norm(W))
    check('weighted_inner', ux.weighted_inner(W, uy, ut3.UT3Weights.from_ut3svd(uy)), gx.weighted_inner(W, gy, ut3.UT3Weights.from_ut3svd(gy)))
    # save/load
    td = tempfile.mkdtemp(); fn = os.path.join(td, 'x.npz')
    gx.save(fn); check('save/load', gx.to_dense(), ut3.UniformTuckerTensorTrain.load(fn).to_dense())


for (shape, tr, ttr, ss) in CASES:
    for force_pad in (False, True):
        for squash in (True, False):
            if len(shape) == 1 and not squash:
                continue
            try:
                run_case(shape, tr, ttr, ss, force_pad, squash)
            except Exception as e:
                print('  EXCEPTION in case', shape, ss, force_pad, squash, '->', type(e).__name__, str(e)[:200])
                fails.append(('EXC %s %s %s %s: %s' % (shape, ss, force_pad, squash, type(e).__name__), 1.0))

print('\n==== FAILURES ====')
for f in fails:
    print(f)
print('total failures:', len(fails))
