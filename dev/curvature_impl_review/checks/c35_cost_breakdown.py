"""Read-only plan check (v5, P5 / F20 re-basing): where the prototype two-hole's time goes, by FORM. Only the dotted
adjoint sweeps differ in form from the house standard (prototype: dense trs on all six terms; standard: the GN
counterpart's form -- prop as a two-term reverse recurrence, src as an order scan, `_adj_sweep_scanned`). The dotted
dx~i (dense trs, as production compute_dxi_tilde_jets) and the TT assembly (dense trs per chunk, as production
assemble_tt_variation_jets) already match their GN counterparts. Measures: GN matvec; two-hole total; the two dotted
adjoint sweeps; and, as the form factor, production dense `_adj_sweep` vs `_adj_sweep_scanned` on the undotted sweep."""
import sys, os, time; sys.path.insert(0, __import__("os").getcwd()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import t3toolbox.backend.fv_conversions as fvc, t3toolbox.backend.sampling_derivatives as sd
import t3toolbox.backend.t3_constructors as t3k, t3toolbox.backend.tv_operations as tv, t3toolbox.backend.tt_operations as tto
import t3toolbox.backend.probing as pr
from c22_probe_derivs_v3_recipe import two_hole_v3
from c14_probe_derivs_twohole import dotted_adj_sweep
from t3toolbox.backend.contractions import contract
np.random.seed(0); rng = np.random.default_rng(0)
def tmean(f, n=3):
    f(); t0 = time.perf_counter()
    for _ in range(n):
        f()
    return (time.perf_counter() - t0) / n
for shape, n, r, order, W in [((12,) * 4, 6, 6, 2, 400), ((20,) * 5, 8, 8, 2, 300), ((12,) * 4, 6, 6, 4, 300)] + ([((24,) * 4, 16, 16, 3, 2000), ((32,) * 4, 24, 24, 4, 2000)] if len(sys.argv) > 1 else []):
    d = len(shape); tr = (n,) * d; ttr = (1,) + (r,) * (d - 1) + (1,)
    x = t3k.t3_corewise_randn(shape, tr, ttr); frame, var = fvc.t3_orthogonal_representations(x); U, O, L, R = frame; G = L
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
    xi_j, mu_j = msw[0], msw[1]; nu_j = sd.compute_nu_jets(G, xi_j, trs)
    dets = sd.compute_deta_tilde_jets(U, zt); tts = sd.compute_tau_tilde_jets(G, xi_j, dets, mu_j, trs)
    xid_j = sd.build_input_jets(pr.compute_xi(Ud, ww), pr.compute_xi(Ud, pp))
    mud_j = sd.compute_sigma_jets(Gd, G, G, xi_j, xid_j, mu_j, trs)
    detd = [contract('tWCo,KCio->tWKCi', z, u, len_W=1, len_C=0) for z, u in zip(zt, Ud)]
    t_dadj = 2 * tmean(lambda: dotted_adj_sweep(G, Gd, xi_j, xid_j, dets, detd, mu_j, mud_j, tts, trs, (1,), (), (W,)))
    t_dense = tmean(lambda: sd._adj_sweep(G, xi_j, dets, mu_j, trs))
    t_scan = tmean(lambda: sd._adj_sweep_scanned(G, xi_j, dets, mu_j, trs))
    print(f'd={d} n=r={n} W={W} order={order}: GN {t_gn:.3f}s, two-hole {t_two:.3f}s (exact/GN {(t_gn + t_two) / t_gn:.2f}x); '
          f'dotted adjoint sweeps (dense prototype) {t_dadj:.3f}s = {100 * t_dadj / t_two:.0f}% of the two-hole; '
          f'GN-form factor scanned/dense on the undotted sweep {t_scan / t_dense:.2f}x -> exact/GN with the dotted sweeps '
          f'in the GN form ~{(t_gn + t_two - t_dadj + t_dadj * t_scan / t_dense) / t_gn:.2f}x')
