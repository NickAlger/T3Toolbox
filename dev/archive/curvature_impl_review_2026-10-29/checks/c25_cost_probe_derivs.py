# PREDATES the S3b API (2026-10-27): written against the dense `trs` jet kernels (`trs=` arguments, `sd.binomial_combine_tensor`, the `*_trs` twins), which S3b removed (now `tests/_jet_trs_twins.py`); kept as the record of the plan-time evidence, not rewritten. The shipped kernels are pinned by tests/backend/test_curvature.py.
"""Read-only plan check (v3, m26 + E): wall-clock of the probe_derivatives two-hole (dense-trs PROTOTYPE, numpy,
CPU) vs the GN matvec (library forward + transpose from the stored sweep), and the share of the nu re-sweep
through G inside the two-hole vs the nu = T nu transform. Indicative only (prototype, not the scan-form kernel)."""
import sys, os, time; sys.path.insert(0, __import__("os").getcwd()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import t3toolbox.backend.fv_conversions as fvc, t3toolbox.backend.sampling_derivatives as sd
import t3toolbox.backend.t3_constructors as t3k, t3toolbox.backend.tv_operations as tv, t3toolbox.backend.tt_operations as tto
from c22_probe_derivs_v3_recipe import two_hole_v3
np.random.seed(0); rng = np.random.default_rng(0)
def tmean(f, n=3):
    f(); t0 = time.perf_counter()
    for _ in range(n):
        f()
    return (time.perf_counter() - t0) / n
for shape, n, r, order, W in [((12,) * 4, 6, 6, 2, 400), ((20,) * 5, 8, 8, 2, 300)]:
    d = len(shape); tr = (n,) * d; ttr = (1,) + (r,) * (d - 1) + (1,)
    x = t3k.t3_corewise_randn(shape, tr, ttr); frame, var = fvc.t3_orthogonal_representations(x); U, O, L, R = frame
    ww = [rng.standard_normal((W, N)) for N in shape]; pp = [rng.standard_normal((W, N)) for N in shape]
    zt = [rng.standard_normal((order + 1, W, N)) for N in shape]
    msw = sd.tv_precompute_probe_frame_sweep_jets(frame, ww, pp, order); trs = sd.binomial_combine_tensor(order)
    v = tv.tv_orthogonal_gauge_projection(frame, (tuple(rng.standard_normal(a.shape) for a in var[0]),
                                                  tuple(rng.standard_normal(a.shape) for a in var[1])))
    Ud = tuple(rng.standard_normal((1,) + u.shape) for u in U); Gd = tuple(rng.standard_normal((1,) + l.shape) for l in L)
    gn = lambda: sd.tv_probe_transpose_derivatives_from_sweep(sd.tv_probe_jacobian_derivatives_from_sweep(v, ww, pp, frame, msw, order),
                                                              ww, pp, frame, msw, order, sum_over_probes=True, chunk_size=None)
    t_gn = tmean(gn)
    t_two = tmean(lambda: two_hole_v3(frame, msw, Ud, Gd, ww, pp, zt, order, (1,), (), 'resweep'))
    t_nu = tmean(lambda: sd.compute_nu_jets(L, msw[0], trs))
    T = tto.tt_zipper_right_to_left(L, R)[1:]
    t_T = tmean(lambda: [np.einsum('...ab,t...b->t...a', T[i], msw[2][i]) for i in range(d)])
    nu = sd.compute_nu_jets(L, msw[0], trs)
    def tildes():
        dets = sd.compute_deta_tilde_jets(U, zt)
        sd.compute_tau_tilde_jets(L, msw[0], dets, msw[1], trs); sd.compute_sigma_tilde_jets(L, msw[0], dets, nu, trs)
    t_til = tmean(tildes)
    print(f'd={d} n=r={n} W={W} order={order}: GN matvec {t_gn:.3f}s; two-hole {t_two:.3f}s -> exact/GN = {(t_gn + t_two) / t_gn:.1f}x; '
          f'nu re-sweep {100 * t_nu / t_two:.1f}% of the two-hole (T-transform {100 * t_T / t_two:.2f}%); '
          f'undotted tildes (the store variant saving) {100 * t_til / t_two:.1f}%')
