"""R8: ut3_{probe,apply,entries}_derivatives, the corewise transposes and the corewise derivative
transposes vs the ragged twins, at W stacks x C stacks, ragged & packed vectors, garbage-padded input."""
import numpy as np, itertools
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.backend.ut3_sampling as us
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
def real_grad(g_u, g_t, ux, x):
    """Slice the uniform gradient supercores down to the ragged core shapes (unstacked, prefix masks)."""
    tkm, ttm = ux.masks.data
    out_u = [g_u[i][..., tkm[i], :][..., :Ni] for i, Ni in enumerate(ux.shape)]
    out_t = [g_t[i][..., ttm[i], :, :][..., :, tkm[i], :][..., :, :, ttm[i + 1]] for i in range(ux.d)]
    return out_u, out_t

CONFIGS = [((4, 6), (2, 3), (1, 3, 1)), ((4, 5, 3), (2, 3, 2), (1, 2, 3, 1)), ((3, 6, 4, 5), (2, 3, 3, 2), (1, 2, 4, 2, 1))]
for (shape, tk, tt), stack, W, pad in itertools.product(CONFIGS, [(), (2,)], [(), (3,), (2, 2)], [False, True]):
    tag = 'shape=%s stack=%s W=%s pad=%s' % (shape, stack, W, pad)
    d = len(shape)
    x = t3.TuckerTensorTrain.randn(shape, tk, tt, stack_shape=stack)
    kw = dict(N=max(shape) + 2, n=max(tk) + 1, r=max(tt) + 1) if pad else {}
    ux = ut3.UniformTuckerTensorTrain.from_t3(x, **kw)
    uxg = corrupt(ux)
    ww = [np.random.randn(*W, Ni) for Ni in shape]
    pp = [np.random.randn(*W, Ni) for Ni in shape]
    idx = np.stack([np.random.randint(0, Ni, size=W) for Ni in shape])
    order = 2
    packed = lambda vs: np.stack([np.pad(v, ((0, 0),) * len(W) + ((0, ux.N - v.shape[-1]),)) for v in vs])
    try:
        # --- derivatives
        rz = x.probe_derivatives(ww, pp, order)
        uz = ux.probe_derivatives(ww, pp, order)
        uzg = uxg.probe_derivatives(ww, pp, order)
        for i in range(d):
            check('probe_derivs mode %d %s' % (i, tag), uz[i].shape == rz[i].shape and relerr(uz[i], rz[i]) < TOL, str((uz[i].shape, rz[i].shape)))
            check('probe_derivs garbage mode %d %s' % (i, tag), relerr(uzg[i], rz[i]) < TOL)
        uzp = ux.probe_derivatives(packed(ww), packed(pp), order)
        check('probe_derivs packed shape ' + tag, uzp.shape == (d, order + 1) + W + stack + (ux.N,), str(uzp.shape))
        for i in range(d):
            check('probe_derivs packed val %d %s' % (i, tag), relerr(uzp[i][..., :shape[i]], rz[i]) < TOL)
        ra = x.apply_derivatives(ww, pp, order); ua = ux.apply_derivatives(ww, pp, order)
        check('apply_derivs ' + tag, ua.shape == ra.shape and relerr(ua, ra) < TOL, str((ua.shape, ra.shape)))
        check('apply_derivs garbage ' + tag, relerr(uxg.apply_derivatives(ww, pp, order), ra) < TOL)
        re_ = x.entries_derivatives(idx, pp, order); ue = ux.entries_derivatives(idx, pp, order)
        check('entries_derivs ' + tag, ue.shape == re_.shape and relerr(ue, re_) < TOL, str((ue.shape, re_.shape)))
        check('entries_derivs garbage ' + tag, relerr(uxg.entries_derivatives(idx, pp, order), re_) < TOL)

        # --- corewise transposes (sum_over_probes True and False) vs ragged
        c = np.asarray(np.random.randn(*(W + stack)))
        zt = [np.random.randn(*(W + stack + (Ni,))) for Ni in shape]
        cj = np.random.randn(*((order + 1,) + W + stack))
        ztj = [np.random.randn(*((order + 1,) + W + stack + (Ni,))) for Ni in shape]
        for sop in (True, False):
            pairs = [
                ('apply_cT',   x.apply_corewise_transpose(c, ww, sum_over_probes=sop),   ux.apply_corewise_transpose(c, ww, sum_over_probes=sop),   uxg.apply_corewise_transpose(c, ww, sum_over_probes=sop)),
                ('entries_cT', x.entries_corewise_transpose(c, idx, sum_over_probes=sop), ux.entries_corewise_transpose(c, idx, sum_over_probes=sop), uxg.entries_corewise_transpose(c, idx, sum_over_probes=sop)),
                ('probe_cT',   x.probe_corewise_transpose(zt, ww, sum_over_probes=sop),  ux.probe_corewise_transpose(zt, ww, sum_over_probes=sop),  uxg.probe_corewise_transpose(zt, ww, sum_over_probes=sop)),
                ('apply_cdT',  x.apply_corewise_derivatives_transpose(cj, ww, pp, order, sum_over_probes=sop),   ux.apply_corewise_derivatives_transpose(cj, ww, pp, order, sum_over_probes=sop),   uxg.apply_corewise_derivatives_transpose(cj, ww, pp, order, sum_over_probes=sop)),
                ('entries_cdT',x.entries_corewise_derivatives_transpose(cj, idx, pp, order, sum_over_probes=sop), ux.entries_corewise_derivatives_transpose(cj, idx, pp, order, sum_over_probes=sop), uxg.entries_corewise_derivatives_transpose(cj, idx, pp, order, sum_over_probes=sop)),
                ('probe_cdT',  x.probe_corewise_derivatives_transpose(ztj, ww, pp, order, sum_over_probes=sop), ux.probe_corewise_derivatives_transpose(ztj, ww, pp, order, sum_over_probes=sop), uxg.probe_corewise_derivatives_transpose(ztj, ww, pp, order, sum_over_probes=sop)),
            ]
            for nm, (rU, rG), (uU, uG), (gU, gG) in pairs:
                exp_lead = (() if sop else W)   # W sits AFTER the leading d axis: (d,)+W+stack+(...)
                check('%s sop=%s grad shapes %s' % (nm, sop, tag), uU.shape == (d,) + exp_lead + ux.tucker_supercore.shape[1:] and uG.shape == (d,) + exp_lead + ux.tt_supercore.shape[1:], str((uU.shape, uG.shape)))
                # compare real parts: move the (d,) axis after W, slice through masks
                tkm, ttm = ux.masks.data
                for i in range(d):
                    uUi, uGi, gUi, gGi = uU[i], uG[i], gU[i], gG[i]   # W+stack+(n,N) / W+stack+(r,n,r)
                    # masks have stack: (stack)+(n,) -> for comparison, use the per-stack prefix (ranks uniform here)
                    mt = tkm[i].reshape(-1, ux.n)[0]; ml = ttm[i].reshape(-1, ux.r)[0]; mr = ttm[i + 1].reshape(-1, ux.r)[0]
                    ru = uUi[..., mt, :][..., :shape[i]]
                    rg = uGi[..., ml, :, :][..., :, mt, :][..., :, :, mr]
                    check('%s sop=%s tucker grad mode %d %s' % (nm, sop, i, tag), ru.shape == rU[i].shape and relerr(ru, rU[i]) < TOL, str((ru.shape, rU[i].shape, relerr(ru, rU[i]) if ru.shape == rU[i].shape else None)))
                    check('%s sop=%s tt grad mode %d %s' % (nm, sop, i, tag), rg.shape == rG[i].shape and relerr(rg, rG[i]) < TOL, str((rg.shape, rG[i].shape)))
                    check('%s sop=%s garbage tucker %d %s' % (nm, sop, i, tag), relerr(gUi[..., mt, :][..., :shape[i]], rU[i]) < TOL)
                    check('%s sop=%s garbage tt %d %s' % (nm, sop, i, tag), relerr(gGi[..., ml, :, :][..., :, mt, :][..., :, :, mr], rG[i]) < TOL)
                    # clean-padded output claim: gradient padding must be zero
                    check('%s sop=%s clean-pad tucker %d %s' % (nm, sop, i, tag), float(np.abs(uUi[..., ~mt, :]).max(initial=0)) == 0.0 and float(np.abs(uUi[..., :, shape[i]:]).max(initial=0)) == 0.0)
                    check('%s sop=%s clean-pad tt %d %s' % (nm, sop, i, tag), float(np.abs(uGi[..., ~ml, :, :]).max(initial=0)) == 0.0 and float(np.abs(uGi[..., :, ~mt, :]).max(initial=0)) == 0.0 and float(np.abs(uGi[..., :, :, ~mr]).max(initial=0)) == 0.0)
    except Exception as e:
        import traceback; traceback.print_exc()
        FAILS.append(('EXC ' + tag, repr(e))); print('EXC', tag, repr(e))
print('\n==== %d failures' % len(FAILS))
from collections import Counter
print(Counter(f[0].split(' shape=')[0] for f in FAILS))
