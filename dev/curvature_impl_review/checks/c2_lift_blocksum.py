import sys; sys.path.insert(0, __import__("os").getcwd())
import numpy as np
import t3toolbox.backend.fv_conversions as fvc
import t3toolbox.backend.tt_operations as tto
import t3toolbox.backend.tv_operations as tv
import t3toolbox.backend.apply as ap
import t3toolbox.backend.t3_conversions as t3c
import t3toolbox.backend.t3_constructors as t3k
from t3toolbox.backend.common import get_backend

rng = np.random.default_rng(1)
np.random.seed(1)   # t3_corewise_randn draws from the global RNG (v1 was unseeded)

def well_conditioned_point(shape, tr, ttr):
    # random point, then orthogonal representations; TT-SVD-like conditioning via t3 randn is ok here
    x = t3k.t3_corewise_randn(shape, tr, ttr)
    return x

def lift(frame, var):
    U, O, L, R = frame
    V, H = var
    d = len(U)
    Z = tto.tt_zipper_right_to_left(L, R)
    Ud, Gd = [], []
    for i in range(d):
        Hc = np.einsum('aib,bc->aic', L[i], Z[i + 1])      # the FRAME centre H_i^X = L_i T_i (v1 bug: used the tangent's H)
        ST = np.einsum('axb,aub->xu', O[i], Hc)             # nD x nU  (= S_i^T)
        Ud.append(np.linalg.pinv(ST) @ V[i])               # nU x N
        Ti = Z[i + 1]                                        # rL x rR
        Gd.append(np.einsum('aib,bc->aic', H[i], np.linalg.pinv(Ti)) if i < d - 1 else H[i])
    return tuple(Ud), tuple(Gd)

def blocksum(gY, d, tr, ttr):
    gYU, gYG = gY
    gU = tuple(gYU[i][:tr[i]] for i in range(d))
    gG = []
    for i in range(d):
        r, n, rn = ttr[i], tr[i], ttr[i + 1]
        g = gYG[i]
        if i == 0:
            gG.append(g[:, :n, rn:] + g[:, n:, :rn])
        elif i == d - 1:
            gG.append(g[:r, :n, :] + g[r:, n:, :])
        else:
            gG.append(g[:r, :n, :rn] + g[r:, :n, rn:] + g[r:, n:, :rn])
    return gU, tuple(gG)

def add(a, b, s=1.0):
    return (tuple(x + s * y for x, y in zip(a[0], b[0])), tuple(x + s * y for x, y in zip(a[1], b[1])))

def flat(a):
    return np.concatenate([x.ravel() for x in a[0]] + [x.ravel() for x in a[1]])

for (shape, tr, ttr) in [((5, 6, 6, 5), (2, 3, 3, 2), (1, 2, 3, 2, 1)), ((4, 5, 6), (3, 3, 3), (1, 3, 3, 1))]:
    d = len(shape)
    x = well_conditioned_point(shape, tr, ttr)
    frame, _ = fvc.t3_orthogonal_representations(x)
    U, O, L, R = frame
    theta = (tuple(U), tuple(L))
    cframe = fvc.t3_corewise_frame(theta)
    # random variations of the right shapes
    vshapes = ([o.shape[1:2] + u.shape[1:] for o, u in zip(O, U)], [h.shape for h in L])
    v_raw = (tuple(rng.standard_normal(s) for s in vshapes[0]), tuple(rng.standard_normal(s) for s in vshapes[1]))
    v = tv.tv_orthogonal_gauge_projection(frame, v_raw)
    for name, vv in (('gauged', v), ('ungauged', v_raw)):
        th = lift(frame, vv)
        lhs = tv.tv_to_dense(cframe, th)
        rhs = tv.tv_to_dense(frame, vv)
        print(f'{shape} lift identity DPhi[Lv]=Ev ({name}): rel err {np.linalg.norm(lhs - rhs) / np.linalg.norm(rhs):.2e}')

    # block-sum oracle for apply curvature K_c theta_hat
    W = 7
    ww = [rng.standard_normal((W, N)) for N in shape]
    c = rng.standard_normal((W,))
    th_hat = (tuple(rng.standard_normal(u.shape) for u in U), tuple(rng.standard_normal(g.shape) for g in L))
    Y = tv.tv_to_t3(cframe, th_hat)
    gY = ap.t3_apply_corewise_transpose(c, ww, Y, sum_over_probes=True)
    K_bs = blocksum(gY, d, tr, ttr)
    h = 1e-5
    gp = ap.t3_apply_corewise_transpose(c, ww, add(theta, th_hat, h), sum_over_probes=True)
    gm = ap.t3_apply_corewise_transpose(c, ww, add(theta, th_hat, -h), sum_over_probes=True)
    K_fd = flat(add(gp, gm, -1.0)) / (2 * h)
    print(f'{shape} block-sum oracle vs FD of corewise transpose: rel err {np.linalg.norm(flat(K_bs) - K_fd) / np.linalg.norm(K_fd):.2e}')
    # symmetry <a, K b> = <b, K a>
    a_hat = (tuple(rng.standard_normal(u.shape) for u in U), tuple(rng.standard_normal(g.shape) for g in L))
    Ka = blocksum(ap.t3_apply_corewise_transpose(c, ww, tv.tv_to_t3(cframe, a_hat), sum_over_probes=True), d, tr, ttr)
    print(f'{shape} symmetry <a,Kb>={flat(a_hat) @ flat(K_bs):.10f} <b,Ka>={flat(th_hat) @ flat(Ka):.10f}')
    # Lagrangian vertical null space: K_N theta_vert = 0 for N normal; take vertical theta from gauge generator
    # T3-T3: ungauged project at corewise frame = grad_theta <Z, Phi(theta)>
    Zt = t3k.t3_corewise_randn(shape, tuple(t + 1 for t in tr), tuple(1 if k in (0, d) else t + 1 for k, t in enumerate(ttr)))
    def ungauged_inner_grad(th, z):
        fr = fvc.t3_corewise_frame(th)
        up, down, left, right = fr
        ztk, ztt = z
        ztt = tto.tt_squash_tails(ztt)
        _, xmap, _ = get_backend(False, False)
        (o2,) = xmap(tv._project_t3_rebasis_step, (ztt, ztk, up))
        zl = tto.tt_zipper_left_to_right(o2[:-1], left[:-1])
        zr = tto.tt_zipper_right_to_left(o2[1:], right[1:])
        dG, dB = xmap(tv._project_t3_variations_step, (zl, zr, ztt, ztk, down, up))
        return (tuple(dB), tuple(dG))
    g0 = ungauged_inner_grad(theta, Zt)
    Zd = t3c.t3_to_dense(Zt)
    f = lambda th: np.sum(Zd * t3c.t3_to_dense(th))
    e = th_hat
    fd = (f(add(theta, e, h)) - f(add(theta, e, -h))) / (2 * h)
    print(f'{shape} T3-T3 one-hole: <grad, e>={flat(g0) @ flat(e):.10f}  FD={fd:.10f}')
