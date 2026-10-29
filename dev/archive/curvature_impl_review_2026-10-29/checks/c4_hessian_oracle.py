"""Read-only plan check: exact Riemannian Hessian quadratic form for f = 0.5 sum_w (apply(x,w) - y_w)^2
assembled from EXISTING kernels (block-sum oracle + lift), vs FD along tv_retract. Also K_N vertical null space."""
import sys; sys.path.insert(0, __import__("os").getcwd())
import numpy as np
import t3toolbox.backend.fv_conversions as fvc
import t3toolbox.backend.tt_operations as tto
import t3toolbox.backend.tv_operations as tv; import t3toolbox.backend.t3_linalg as t3_linalg
import t3toolbox.backend.apply as ap
import t3toolbox.backend.t3_conversions as t3c
from t3toolbox.backend.common import get_backend

rng = np.random.default_rng(3)

def tt_svd(G, ranks):
    cores, C, rprev = [], G.reshape(1, -1), 1
    for k in range(G.ndim - 1):
        C = C.reshape(rprev * G.shape[k], -1)
        Uu, s, Vt = np.linalg.svd(C, full_matrices=False)
        Uu, s, Vt = Uu[:, :ranks[k]], s[:ranks[k]], Vt[:ranks[k]]
        cores.append(Uu.reshape(rprev, G.shape[k], ranks[k])); C = s[:, None] * Vt; rprev = ranks[k]
    cores.append(C.reshape(rprev, G.shape[-1], 1))
    return cores

def add(a, b, s=1.0):
    return (tuple(x + s * y for x, y in zip(a[0], b[0])), tuple(x + s * y for x, y in zip(a[1], b[1])))
def flat(a):
    return np.concatenate([x.ravel() for x in a[0]] + [x.ravel() for x in a[1]])

def lift(frame, var):
    U, O, L, R = frame
    V, Hv = var
    d = len(U)
    Z = tto.tt_zipper_right_to_left(L, R)
    Ud, Gd = [], []
    for i in range(d):
        Hc = np.einsum('aib,bc->aic', L[i], Z[i + 1])           # frame center H_i = L_i T_i
        ST = np.einsum('axb,aub->xu', O[i], Hc)                 # S_i^T, (nD, nU)
        Ud.append(np.linalg.solve(ST.T @ ST, ST.T @ V[i]) if ST.shape[0] >= ST.shape[1] else np.linalg.pinv(ST) @ V[i])
        Gd.append(np.einsum('aib,bc->aic', Hv[i], np.linalg.inv(Z[i + 1])) if i < d - 1 else Hv[i])
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
    (o2,) = xmap(t3_linalg._inner_product_rebasis_step, (ztt, ztk, up))
    zl = tto.tt_zipper_left_to_right(o2[:-1], left[:-1])
    zr = tto.tt_zipper_right_to_left(o2[1:], right[1:])
    dG, dB = xmap(t3_linalg._inner_product_variations_step, (zl, zr, ztt, ztk, down, up))
    return (tuple(dB), tuple(dG))

shape, tr, ttr = (5, 6, 6, 5), (2, 3, 3, 2), (1, 2, 3, 2, 1)
d = len(shape)
Us = [np.linalg.qr(rng.standard_normal((N, n)))[0].T for N, n in zip(shape, tr)]
cores = tt_svd(rng.standard_normal(tr), ttr[1:-1])
x = (tuple(Us), tuple(cores))
frame, _ = fvc.t3_orthogonal_representations(x)
U, O, L, R = frame
theta = (tuple(U), tuple(L))
cframe = fvc.t3_corewise_frame(theta)
X = t3c.t3_to_dense(theta)

W = 300
ww = [rng.standard_normal((W, N)) for N in shape]
y = rng.standard_normal(W)
def apply_dense(T):
    out = T
    for w in ww[::-1]:
        out = np.einsum('...n,wn->w...', out, w) if out.ndim == T.ndim else None
    return None
def A(T):  # dense apply: sum_x T[x] prod_i w_i[x_i], per w
    return np.einsum('abcd,wa,wb,wc,wd->w', T, *ww)
def f_dense(T):
    return 0.5 * np.sum((A(T) - y) ** 2)
r = A(X) - y

# gradient (Riemannian) and its doubled-rank T3
g = tv.tv_orthogonal_gauge_projection(frame, ap.tv_apply_transpose(r, ww, frame, sum_over_probes=True))
Zg = tv.tv_to_t3(frame, g)

vshapes = ([o.shape[1:2] + u.shape[1:] for o, u in zip(O, U)], [h.shape for h in L])
v = tv.tv_orthogonal_gauge_projection(frame, (tuple(rng.standard_normal(s) for s in vshapes[0]),
                                              tuple(rng.standard_normal(s) for s in vshapes[1])))
xi = tv.tv_to_dense(frame, v)
nrm = np.linalg.norm(xi); v = (tuple(a / nrm for a in v[0]), tuple(a / nrm for a in v[1])); xi = xi / nrm
th_hat = lift(frame, v)
print('lift identity |DPhi[Lv]-Ev|/|Ev| =', np.linalg.norm(tv.tv_to_dense(cframe, th_hat) - xi))

Y = tv.tv_to_t3(cframe, th_hat)
K_r = blocksum(ap.t3_apply_corewise_transpose(r, ww, Y, sum_over_probes=True), d, tr, ttr)
K_g = blocksum(ungauged_inner_grad(Y, Zg), d, tr, ttr)
gn = np.sum(A(xi) ** 2)
curv_r = flat(th_hat) @ flat(K_r)
curv_g = flat(th_hat) @ flat(K_g)
print(f'GN |A xi|^2                        = {gn:.8f}')
print(f'exact: GN + <th,K_r th> - <th,K_g th> = {gn + curv_r - curv_g:.8f}   (level-1 GN+K_r only: {gn + curv_r:.8f})')
for h in (1e-2, 5e-3, 2.5e-3):
    fx = lambda t: f_dense(t3c.t3_to_dense(tv.tv_retract(frame, (tuple(t * a for a in v[0]), tuple(t * a for a in v[1])))))
    print(f'FD d2/dt2 f(tv_retract(t v)), h={h:.1e} = {(fx(h) - 2 * fx(0.0) + fx(-h)) / h ** 2:.8f}')

# vertical null space: theta_vert from gauge generator  U_i -> U_i + X U_i ... in RAW core coords:
# raw vertical: dU_i = M U_i, dG_i = -G_i x_2 M^T  (Tucker gauge), dG_i += G_i Y ; dG_{i+1} -= Y G_{i+1} (TT gauge)
i = 1
M = rng.standard_normal((tr[i], tr[i]))
tv_U = [np.zeros_like(u) for u in U]; tv_G = [np.zeros_like(l) for l in L]
tv_U[i] = M @ U[i]; tv_G[i] = -np.einsum('aib,ji->ajb', L[i], M.T)
Yb = rng.standard_normal((ttr[2], ttr[2]))
tv_G[1] = tv_G[1] + np.einsum('aib,bc->aic', L[1], Yb); tv_G[2] = tv_G[2] - np.einsum('bc,cid->bid', Yb, L[2])
th_v = (tuple(tv_U), tuple(tv_G))
print('vertical: |DPhi[th_v]| =', np.linalg.norm(tv.tv_to_dense(cframe, th_v)))
Yv = tv.tv_to_t3(cframe, th_v)
K_rv = blocksum(ap.t3_apply_corewise_transpose(r, ww, Yv, sum_over_probes=True), d, tr, ttr)
K_gv = blocksum(ungauged_inner_grad(Yv, Zg), d, tr, ttr)
KN = flat(K_rv) - flat(K_gv)
print(f'vertical null space: |K_N th_v| = {np.linalg.norm(KN):.3e}  vs |K_r th_v| = {np.linalg.norm(flat(K_rv)):.3e}')
