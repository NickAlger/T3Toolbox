"""Critic check (read-only): how the exact-curvature term scales with a small bond singular value delta
(the state right after a zero-padded restart's first step), vs GN, and the FD-along-retraction validity.
Reuses only existing library kernels + the block-sum oracle (as in the writer's c4)."""
import sys; sys.path.insert(0, __import__("os").getcwd())
import numpy as np
import t3toolbox.backend.fv_conversions as fvc
import t3toolbox.backend.tt_operations as tto
import t3toolbox.backend.tv_operations as tv
import t3toolbox.backend.apply as ap
import t3toolbox.backend.t3_conversions as t3c
from t3toolbox.backend.common import get_backend


def tt_svd(G, ranks):
    cores, C, rprev = [], G.reshape(1, -1), 1
    for k in range(G.ndim - 1):
        C = C.reshape(rprev * G.shape[k], -1)
        Uu, s, Vt = np.linalg.svd(C, full_matrices=False)
        Uu, s, Vt = Uu[:, :ranks[k]], s[:ranks[k]], Vt[:ranks[k]]
        cores.append(Uu.reshape(rprev, G.shape[k], ranks[k])); C = s[:, None] * Vt; rprev = ranks[k]
    cores.append(C.reshape(rprev, G.shape[-1], 1))
    return cores


def flat(a):
    return np.concatenate([x.ravel() for x in a[0]] + [x.ravel() for x in a[1]])


def lift(frame, var, eps=0.0):
    U, O, L, R = frame
    V, Hv = var
    d = len(U)
    Z = tto.tt_zipper_right_to_left(L, R)
    Ud, Gd = [], []
    for i in range(d):
        Hc = np.einsum('aib,bc->aic', L[i], Z[i + 1])
        ST = np.einsum('axb,aub->xu', O[i], Hc)
        u, s, vt = np.linalg.svd(ST, full_matrices=False)
        Ud.append(vt.T @ np.diag(s / (s * s + eps * eps)) @ u.T @ V[i])
        if i < d - 1:
            u, s, vt = np.linalg.svd(Z[i + 1], full_matrices=False)
            Tp = vt.T @ np.diag(s / (s * s + eps * eps)) @ u.T
            Gd.append(np.einsum('aib,bc->aic', Hv[i], Tp))
        else:
            Gd.append(Hv[i])
    return tuple(Ud), tuple(Gd)


def blocksum(gY, d, tr, ttr):
    gYU, gYG = gY
    gU = tuple(gYU[i][:tr[i]] for i in range(d))
    gG = []
    for i in range(d):
        r, n, rn = ttr[i], tr[i], ttr[i + 1]
        g = gYG[i]
        if i == 0:   gG.append(g[:, :n, rn:] + g[:, n:, :rn])
        elif i == d - 1: gG.append(g[:r, :n, :] + g[r:, n:, :])
        else:        gG.append(g[:r, :n, :rn] + g[r:, :n, rn:] + g[r:, n:, :rn])
    return gU, tuple(gG)


def ungauged_inner_grad(th, z):
    up, down, left, right = fvc.t3_corewise_frame(th)
    ztk, ztt = z
    ztt = tto.tt_squash_tails(ztt)
    _, xmap, _ = get_backend(False, False)
    (o2,) = xmap(tv._project_t3_rebasis_step, (ztt, ztk, up))
    zl = tto.tt_zipper_left_to_right(o2[:-1], left[:-1])
    zr = tto.tt_zipper_right_to_left(o2[1:], right[1:])
    dG, dB = xmap(tv._project_t3_variations_step, (zl, zr, ztt, ztk, down, up))
    return (tuple(dB), tuple(dG))



# Full-rank format: Tucker ranks = mode sizes, TT ranks maximal -> the T3 manifold is OPEN in R^{N1 x N2 x N3},
# so N = A^T r - grad f = 0 and the exact Hessian must equal Gauss-Newton (curvature term ~ 0).
for shape, tr, ttr in (((3, 3, 3), (3, 3, 3), (1, 3, 3, 1)), ((2, 3, 4), (2, 3, 4), (1, 2, 4, 1))):
    d = len(shape)
    rng = np.random.default_rng(5)
    Us = [np.linalg.qr(rng.standard_normal((N, n)))[0].T for N, n in zip(shape, tr)]
    x = (tuple(Us), tuple(tt_svd(rng.standard_normal(tr), ttr[1:-1])))
    frame, _ = fvc.t3_orthogonal_representations(x)
    U, O, L, R = frame
    cframe = fvc.t3_corewise_frame((tuple(U), tuple(L)))
    X = t3c.t3_to_dense((tuple(U), tuple(L)))
    W = 200
    ww = [rng.standard_normal((W, N)) for N in shape]
    A_ = lambda T: np.einsum('abc,wa,wb,wc->w', T, *ww)
    y = rng.standard_normal(W)
    r = A_(X) - y
    g = tv.tv_orthogonal_gauge_projection(frame, ap.tv_apply_transpose(r, ww, frame, sum_over_probes=True))
    Atr = np.einsum('w,wa,wb,wc->abc', r, *ww)
    Zg = tv.tv_to_t3(frame, g)
    vshapes = ([o.shape[1:2] + u.shape[1:] for o, u in zip(O, U)], [h.shape for h in L])
    v = tv.tv_orthogonal_gauge_projection(frame, (tuple(rng.standard_normal(s) for s in vshapes[0]),
                                                  tuple(rng.standard_normal(s) for s in vshapes[1])))
    th = lift(frame, v)
    Y = tv.tv_to_t3(cframe, th)
    K_r = flat(blocksum(ap.t3_apply_corewise_transpose(r, ww, Y, sum_over_probes=True), d, tr, ttr))
    K_g = flat(blocksum(ungauged_inner_grad(Y, Zg), d, tr, ttr))
    print(f"full-rank {shape}: |A^T r - grad f|/|A^T r| = {np.linalg.norm(Atr - tv.tv_to_dense(frame, g)) / np.linalg.norm(Atr):.1e}; "
          f"|K_N L v| / |K_(A^T r) L v| = {np.linalg.norm(K_r - K_g) / np.linalg.norm(K_r):.1e}")
