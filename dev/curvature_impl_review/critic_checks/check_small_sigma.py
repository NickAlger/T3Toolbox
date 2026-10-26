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
for delta in (1e-1, 1e-2, 1e-3, 1e-4):
    rng = np.random.default_rng(3)
    Us = [np.linalg.qr(rng.standard_normal((N, n)))[0].T for N, n in zip(shape, tr)]
    # Tucker core with a prescribed bond-2 spectrum [1, 0.5, delta] (unfolding (2*3) x (3*2))
    A = np.linalg.qr(rng.standard_normal((6, 3)))[0]
    B = np.linalg.qr(rng.standard_normal((6, 3)))[0].T
    Gc = (A @ np.diag([1.0, 0.5, delta]) @ B).reshape(tr)
    cores = tt_svd(Gc, ttr[1:-1])
    x = (tuple(Us), tuple(cores))
    frame, _ = fvc.t3_orthogonal_representations(x)
    U, O, L, R = frame
    theta = (tuple(U), tuple(L))
    cframe = fvc.t3_corewise_frame(theta)
    X = t3c.t3_to_dense(theta)
    W = 300
    ww = [rng.standard_normal((W, N)) for N in shape]
    y = np.einsum('abcd,wa,wb,wc,wd->w', X, *ww) + 0.3 * rng.standard_normal(W) * np.linalg.norm(X)
    A_ = lambda T: np.einsum('abcd,wa,wb,wc,wd->w', T, *ww)
    f_dense = lambda T: 0.5 * np.sum((A_(T) - y) ** 2)
    r = A_(X) - y
    g = tv.tv_orthogonal_gauge_projection(frame, ap.tv_apply_transpose(r, ww, frame, sum_over_probes=True))
    Zg = tv.tv_to_t3(frame, g)
    vshapes = ([o.shape[1:2] + u.shape[1:] for o, u in zip(O, U)], [h.shape for h in L])
    v = tv.tv_orthogonal_gauge_projection(frame, (tuple(rng.standard_normal(s) for s in vshapes[0]),
                                                  tuple(rng.standard_normal(s) for s in vshapes[1])))
    nrm = np.linalg.norm(tv.tv_to_dense(frame, v))
    v = (tuple(a / nrm for a in v[0]), tuple(a / nrm for a in v[1]))
    xi = tv.tv_to_dense(frame, v)
    gn = np.sum(A_(xi) ** 2)
    Xn = np.linalg.norm(X)
    out = []
    for rel_eps in (0.0, 1e-6, 1e-2):
        th_hat = lift(frame, v, rel_eps * Xn)
        Y = tv.tv_to_t3(cframe, th_hat)
        K_r = blocksum(ap.t3_apply_corewise_transpose(r, ww, Y, sum_over_probes=True), d, tr, ttr)
        K_g = blocksum(ungauged_inner_grad(Y, Zg), d, tr, ttr)
        out.append(flat(th_hat) @ (flat(K_r) - flat(K_g)))
    fx = lambda t: f_dense(t3c.t3_to_dense(tv.tv_retract(frame, (tuple(t * a for a in v[0]), tuple(t * a for a in v[1])))))
    fds = []
    for h in (1e-1 * delta, 1e-2 * delta):
        fds.append((fx(h) - 2 * fx(0.0) + fx(-h)) / h ** 2)
    # model error of the exact quadratic at step t = delta (the curvature radius) vs at t = 0.1*delta
    f0 = fx(0.0); slope = float(np.dot(flat(g), flat(v)))
    def model_err(t, curv=None):
        m = f0 + t * slope + 0.5 * t * t * (gn + (out[0] if curv is None else curv))
        return abs(fx(t) - m) / max(abs(fx(t) - f0), 1e-300)
    print(f"delta={delta:.0e} |X|={Xn:.2f} GN={gn:.4e} curv(eps=0)={out[0]:+.4e} curv(1e-6|X|)={out[1]:+.4e} "
          f"curv(1e-2|X|)={out[2]:+.4e} | FD h=.1d:{fds[0]:.4e} h=.01d:{fds[1]:.4e} exact={gn + out[0]:.4e} "
          f"| exact-model err t=d:{model_err(delta):.1e} 10d:{model_err(10 * delta):.1e} 100d:{model_err(100 * delta):.1e}"
          f" | GN-model err t=d:{model_err(delta, 0.0):.1e} 10d:{model_err(10 * delta, 0.0):.1e} 100d:{model_err(100 * delta, 0.0):.1e}")
