"""Ragged vs uniform twins on the OPTIONAL-argument paths: to_uniform -> op(opts) -> to_ragged == op_ragged(opts)."""
import numpy as np, t3toolbox as t3t, traceback
import t3toolbox.manifold as t3m, t3toolbox.uniform_manifold as ut3m, t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.tucker_tensor_train as t3mod
np.random.seed(0)
shape = (5, 6, 7)
def rel(a, b): return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / (np.linalg.norm(np.asarray(b)) + 1e-300))
def check(name, fn):
    try:
        r = fn()
        print(f'{name:60s} {r}')
    except Exception as e:
        print(f'{name:60s} RAISED {type(e).__name__}: {str(e)[:90]}')
x = t3t.TuckerTensorTrain.randn(shape, (2, 3, 2), (1, 2, 2, 1)) + t3t.TuckerTensorTrain.randn(shape, (2, 2, 3), (1, 3, 2, 1))
y = t3t.TuckerTensorTrain.randn(shape, (3, 2, 2), (1, 2, 3, 1))
ux, uy = ut3.UniformTuckerTensorTrain.from_t3(x), ut3.UniformTuckerTensorTrain.from_t3(y)
xd = x.to_dense()
# t3svd option paths
for kw in (dict(), dict(max_tt_ranks=2), dict(max_tucker_ranks=[2, 2, 2]), dict(max_tt_ranks=[1, 2, 3, 1]), dict(max_tucker_ranks=2, max_tt_ranks=2)):
    def f(kw=kw):
        a, sa, sta = x.t3svd(**kw); b, sb, stb = ux.t3svd(**kw)
        return ('dense rel err %.1e' % rel(a.to_dense(), b.to_dense()), 'ranks', a.tucker_ranks, a.tt_ranks, '|', tuple(b.tucker_ranks) if hasattr(b,'tucker_ranks') else None, 'svals rel %.1e' % max(rel(np.asarray(s)[:len(t)], t) for s, t in zip(list(sb), list(sa))))
    check('t3svd ' + str(kw), f)
# assume_orthogonal on a right-orthogonal input
xr = x.rank_adjustment_sweep('right_to_left'); uxr = ut3.UniformTuckerTensorTrain.from_t3(xr)
check('t3svd(assume_orthogonal=True) on right-orth input', lambda: ('rel err vs dense %.1e | %.1e' % (rel(xr.t3svd(assume_orthogonal=True)[0].to_dense(), xd), rel(uxr.t3svd(assume_orthogonal=True)[0].to_dense(), xd))))
check('t3svd(assume_orthogonal=True) on NON-orth input (both wrong?)', lambda: ('rel err vs dense %.1e | %.1e' % (rel(x.t3svd(assume_orthogonal=True)[0].to_dense(), xd), rel(ux.t3svd(assume_orthogonal=True)[0].to_dense(), xd))))
# rank_adjustment_sweep directions
for dr in ('right_to_left', 'left_to_right'):
    check(f'rank_adjustment_sweep({dr}) dense+orth', lambda dr=dr: ('rel %.1e' % rel(x.rank_adjustment_sweep(dr).to_dense(), ux.rank_adjustment_sweep(dr).to_dense()), 'L-orth', bool(x.rank_adjustment_sweep(dr).is_left_orthogonal()), bool(ux.rank_adjustment_sweep(dr).is_left_orthogonal()), 'R-orth', bool(x.rank_adjustment_sweep(dr).is_right_orthogonal()), bool(ux.rank_adjustment_sweep(dr).is_right_orthogonal())))
# inner / norm with use_orthogonalization False
for uo in (True, False):
    check(f'inner(use_orthogonalization={uo})', lambda uo=uo: 'rel %.1e' % rel(x.inner(y, use_orthogonalization=uo), ux.inner(uy, use_orthogonalization=uo)))
    check(f'norm(use_orthogonalization={uo})', lambda uo=uo: 'rel %.1e' % rel(x.norm(use_orthogonalization=uo), ux.norm(use_orthogonalization=uo)))
# weighted norm/inner
W = t3t.T3Weights.from_t3svd(x); UW = ut3.UT3Weights.from_t3weights(W, n=ux.n, r=ux.r)
for uo in (True, False):
    check(f't3_weighted_norm(use_orthogonalization={uo})', lambda uo=uo: 'rel %.1e' % rel(t3mod.t3_weighted_norm(x, W, use_orthogonalization=uo), ut3.ut3_weighted_norm(ux, UW, use_orthogonalization=uo)))
    check(f't3_weighted_inner(use_orthogonalization={uo})', lambda uo=uo: 'rel %.1e' % rel(t3mod.t3_weighted_inner(x, x, W, use_orthogonalization=uo), ut3.ut3_weighted_inner(ux, ux, UW, use_orthogonalization=uo)))
# tangent transposes with sum_over_probes False / True
frame = t3m.MANIFOLD.frame(x); uframe = ut3m.UNIFORM_MANIFOLD.frame(ux)
ww = [np.random.randn(4, N) for N in shape]; zt = [np.random.randn(4, N) for N in shape]; c = np.random.randn(4)
pp = [np.random.randn(4, N) for N in shape]
for sop in (False, True):
    check(f'T3Tangent.probe_transpose(sum_over_probes={sop})', lambda sop=sop: 'rel %.1e' % rel(t3m.T3Tangent.probe_transpose(zt, ww, frame, sum_over_probes=sop).to_dense(), ut3m.UT3Tangent.probe_transpose(zt, ww, uframe, sum_over_probes=sop).to_dense()))
    check(f'T3Tangent.apply_transpose(sum_over_probes={sop})', lambda sop=sop: 'rel %.1e' % rel(t3m.T3Tangent.apply_transpose(c, ww, frame, sum_over_probes=sop).to_dense(), ut3m.UT3Tangent.apply_transpose(c, ww, uframe, sum_over_probes=sop).to_dense()))
    check(f'T3Tangent.apply_derivatives_transpose(order=2, sop={sop})', lambda sop=sop: 'rel %.1e' % rel(t3m.T3Tangent.apply_derivatives_transpose(np.random.RandomState(1).randn(3, 4), ww, pp, frame, 2, sum_over_probes=sop).to_dense(), ut3m.UT3Tangent.apply_derivatives_transpose(np.random.RandomState(1).randn(3, 4), ww, pp, uframe, 2, sum_over_probes=sop).to_dense()))
    zj = [np.random.RandomState(2 + i).randn(3, 4, N) for i, N in enumerate(shape)]
    for cs in (None, 2, 100):
        check(f'T3Tangent.probe_derivatives_transpose(order=2, sop={sop}, chunk_size={cs})', lambda sop=sop, cs=cs: 'rel %.1e' % rel(t3m.T3Tangent.probe_derivatives_transpose(zj, ww, pp, frame, 2, sum_over_probes=sop, chunk_size=cs).to_dense(), ut3m.UT3Tangent.probe_derivatives_transpose(zj, ww, pp, uframe, 2, sum_over_probes=sop, chunk_size=cs).to_dense()))
# corewise transposes
for sop in (False, True):
    def f(sop=sop):
        gU, gG = x.apply_corewise_transpose(c, ww, sum_over_probes=sop); uU, uG = ux.apply_corewise_transpose(c, ww, sum_over_probes=sop)
        uU = np.asarray(uU); return ('shapes', gU[0].shape, uU.shape, 'rel core0 %.1e' % rel(uU[0][..., :gU[0].shape[-2], :gU[0].shape[-1]], gU[0]))
    check(f'apply_corewise_transpose(sum_over_probes={sop})', f)
# to_t3 include_shift
v = t3m.MANIFOLD.randn(frame); uv = ut3m.UT3Tangent.from_t3tangent(v) if hasattr(ut3m.UT3Tangent, 'from_t3tangent') else None
for inc in (False, True):
    check(f'to_t3/to_ut3(include_shift={inc})', lambda inc=inc: 'rel %.1e' % rel(v.to_t3(include_shift=inc).to_dense(), uv.to_ut3(include_shift=inc).to_dense()))
# sum_stack
xs = t3t.TuckerTensorTrain.randn(shape, (2, 3, 2), (1, 2, 2, 1), stack_shape=(2, 3)) if 'stack_shape' in t3t.TuckerTensorTrain.randn.__code__.co_varnames else None
if xs is not None:
    uxs = ut3.UniformTuckerTensorTrain.from_t3(xs)
    check('sum_stack() all axes', lambda: 'rel %.1e' % rel(xs.sum_stack().to_dense(), uxs.sum_stack().to_dense()))
    check('sum_stack(axis=0) uniform', lambda: uxs.sum_stack(axis=0))
