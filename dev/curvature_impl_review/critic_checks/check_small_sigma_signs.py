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



shape, tr, ttr = (5, 6, 6, 5), (2, 3, 3, 2), (1, 2, 3, 2, 1)
d = len(shape)
for delta in (1e-2, 1e-3):
    rng = np.random.default_rng(3)
    Us = [np.linalg.qr(rng.standard_normal((N, n)))[0].T for N, n in zip(shape, tr)]
    A = np.linalg.qr(rng.standard_normal((6, 3)))[0]
    B = np.linalg.qr(rng.standard_normal((6, 3)))[0].T
    Gc = (A @ np.diag([1.0, 0.5, delta]) @ B).reshape(tr)
    x = (tuple(Us), tuple(tt_svd(Gc, ttr[1:-1])))
    frame, _ = fvc.t3_orthogonal_representations(x)
    U, O, L, R = frame
    cframe = fvc.t3_corewise_frame((tuple(U), tuple(L)))
    X = t3c.t3_to_dense((tuple(U), tuple(L)))
    W = 300
    ww = [rng.standard_normal((W, N)) for N in shape]
    A_ = lambda T: np.einsum('abcd,wa,wb,wc,wd->w', T, *ww)
    y = A_(X) + 0.3 * rng.standard_normal(W) * np.linalg.norm(X)
    r = A_(X) - y
    g = tv.tv_orthogonal_gauge_projection(frame, ap.tv_apply_transpose(r, ww, frame, sum_over_probes=True))
    Zg = tv.tv_to_t3(frame, g)
    vshapes = ([o.shape[1:2] + u.shape[1:] for o, u in zip(O, U)], [h.shape for h in L])
    ratios = []
    for k in range(12):
        v = tv.tv_orthogonal_gauge_projection(frame, (tuple(rng.standard_normal(s) for s in vshapes[0]),
                                                      tuple(rng.standard_normal(s) for s in vshapes[1])))
        nrm = np.linalg.norm(tv.tv_to_dense(frame, v)); v = (tuple(a / nrm for a in v[0]), tuple(a / nrm for a in v[1]))
        gn = np.sum(A_(tv.tv_to_dense(frame, v)) ** 2)
        th = lift(frame, v)
        Y = tv.tv_to_t3(cframe, th)
        K_r = blocksum(ap.t3_apply_corewise_transpose(r, ww, Y, sum_over_probes=True), d, tr, ttr)
        K_g = blocksum(ungauged_inner_grad(Y, Zg), d, tr, ttr)
        ratios.append((flat(th) @ (flat(K_r) - flat(K_g))) / gn)
    ratios = np.array(ratios)
    print(f"delta={delta:.0e}: curvature/GN over 12 random unit tangents: min {ratios.min():+.2f} max {ratios.max():+.2f}; "
          f"negative-definite-along-v count {(1 + ratios < 0).sum()}/12")
