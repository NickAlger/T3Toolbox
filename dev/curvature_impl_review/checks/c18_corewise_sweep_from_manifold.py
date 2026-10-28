# PREDATES the S3b API (2026-10-27): written against the dense `trs` jet kernels (`trs=` arguments, `sd.binomial_combine_tensor`, the `*_trs` twins), which S3b removed (now `tests/_jet_trs_twins.py`); kept as the record of the plan-time evidence, not rewritten. The shipped kernels are pinned by tests/backend/test_curvature.py.
"""Read-only plan check (v2): the corewise-frame (U,L,L,L) probe sweep and adjoint state are CHEAP
TRANSFORMS of the manifold-frame (U,O,L,R) ones, by the exact identities L_{>i} = T_i R_{>i} (left-canonical
tail) and H_i = O_i x_2 S_i^T:   nu_c = T nu,  eta_c = S eta,  sigma~_c = T sigma~,  dxi~_c = S dxi~
(order-wise for the jets).  Also: ||X||^2 = sum sigma(S_0)^2 = sum sigma(T_i)^2 (eps for free)."""
import sys; sys.path.insert(0, __import__("os").getcwd())
import numpy as np
import t3toolbox.backend.fv_conversions as fvc
import t3toolbox.backend.tt_operations as tto
import t3toolbox.backend.sampling_derivatives as sd
import t3toolbox.backend.t3_conversions as t3c
import t3toolbox.backend.t3_constructors as t3k
np.random.seed(41); rng = np.random.default_rng(41)
for C in [(), (2,)]:
    shape, tr, ttr, order, Wn = (4, 5, 6, 4), (2, 3, 3, 2), (1, 2, 3, 2, 1), 2, 5
    d = len(shape)
    x = t3k.t3_corewise_randn(shape, tr, ttr, stack_shape=C)
    frame, _ = fvc.t3_orthogonal_representations(x)
    U, O, L, R = frame
    cf = fvc.t3_corewise_frame((U, L))
    ZZ = tto.tt_zipper_right_to_left(L, R)
    T = [ZZ[i + 1] for i in range(d)]                                   # T_{d-1} = ones(1,1)
    S = [np.einsum('...axb,...aub->...ux', O[i], np.einsum('...aib,...bc->...aic', L[i], ZZ[i + 1])) for i in range(d)]  # S_i: (nU, nD)
    ww = [rng.standard_normal((Wn, N)) for N in shape]; pp = [rng.standard_normal((Wn, N)) for N in shape]
    zt = [rng.standard_normal((order + 1, Wn) + C + (N,)) for N in shape]
    trs = sd.binomial_combine_tensor(order)
    xm, mm, nm, em = sd.tv_precompute_probe_frame_sweep_jets(frame, ww, pp, order)
    xc, mc, nc, ec = sd.tv_precompute_probe_frame_sweep_jets(cf, ww, pp, order)
    rel = lambda a, b: np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300)
    e_mu = max(rel(a, b) for a, b in zip(mm, mc))
    e_nu = max(rel(np.einsum('...ab,t...b->t...a', T[i], nm[i]), nc[i]) for i in range(d))
    e_eta = max(rel(np.einsum('...ux,t...x->t...u', S[i], em[i]), ec[i]) for i in range(d))
    dm = sd.compute_deta_tilde_jets(U, zt)
    stm = sd.compute_sigma_tilde_jets(R, xm, dm, nm, trs); stc = sd.compute_sigma_tilde_jets(L, xc, dm, nc, trs)
    ttm = sd.compute_tau_tilde_jets(L, xm, dm, mm, trs); ttc = sd.compute_tau_tilde_jets(L, xc, dm, mc, trs)
    dxm = sd.compute_dxi_tilde_jets(O, mm, nm, stm, ttm, trs); dxc = sd.compute_dxi_tilde_jets(L, mc, nc, stc, ttc, trs)
    e_st = max(rel(np.einsum('...ab,t...b->t...a', T[i], stm[i]), stc[i]) for i in range(d))
    e_tt = max(rel(a, b) for a, b in zip(ttm, ttc))
    e_dx = max(rel(np.einsum('...ux,t...x->t...u', S[i], dxm[i]), dxc[i]) for i in range(d))
    print('C=%-5s mu %.1e  nu_c=T nu %.1e  eta_c=S eta %.1e  tau~ %.1e  sigma~_c=T sigma~ %.1e  dxi~_c=S dxi~ %.1e' % (
        C, e_mu, e_nu, e_eta, e_tt, e_st, e_dx))
    Xn2 = np.sum(t3c.t3_to_dense((U, L)) ** 2, axis=tuple(range(len(C), len(C) + d)))
    sS0 = np.linalg.svd(S[0], compute_uv=False); sT1 = np.linalg.svd(T[1], compute_uv=False)
    print('       ||X||^2 = sum s(S_0)^2: %.1e   = sum s(T_1)^2: %.1e' % (
        np.max(np.abs(np.sum(sS0 ** 2, -1) - Xn2) / Xn2), np.max(np.abs(np.sum(sT1 ** 2, -1) - Xn2) / Xn2)))
