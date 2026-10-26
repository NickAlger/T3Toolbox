"""Read-only plan check: (a) the lift identity DPhi[Lv] = Ev on a NON-minimal frame (pinv lift);
(b) the damped lift at a zero-padded resize: no NaN, padded directions lift to 0, curvature on them = 0;
(c) at the zero point the damped lift is identically 0 (exact model == GN)."""
import sys; sys.path.insert(0, __import__("os").getcwd())
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.backend.fv_conversions as fvc
import t3toolbox.backend.tt_operations as tto
import t3toolbox.backend.tv_operations as tv
rng = np.random.default_rng(11)

def damped_pinv(M, eps):            # M (m x n) -> n x m, f(s) = s/(s^2+eps^2), guarded
    Uu, s, Vt = np.linalg.svd(M, full_matrices=False)
    den = s * s + eps * eps
    f = np.where(den > 0, s / np.where(den > 0, den, 1.0), 0.0)
    return Vt.T @ (f[:, None] * Uu.T), s

def lift(frame, var, eps):
    U, O, L, R = frame
    V, Hv = var
    d = len(U)
    Z = tto.tt_zipper_right_to_left(L, R)
    Ud, Gd, smin = [], [], []
    for i in range(d):
        Hc = np.einsum('aib,bc->aic', L[i], Z[i + 1])
        Sp, sS = damped_pinv(np.einsum('axb,aub->xu', O[i], Hc), eps)
        Ud.append(Sp @ V[i]); smin.append(sS.min())
        if i < d - 1:
            Tp, sT = damped_pinv(Z[i + 1], eps)
            Gd.append(np.einsum("aib,bc->aic", Hv[i], Tp)); smin.append(sT.min())
        else:
            Gd.append(Hv[i])
    return (tuple(Ud), tuple(Gd)), smin

def rand_var(frame):
    U, O, L, R = frame
    return tv.tv_orthogonal_gauge_projection(frame, (tuple(rng.standard_normal(o.shape[-2:-1] + u.shape[-1:]) for o, u in zip(O, U)),
                                                    tuple(rng.standard_normal(l.shape) for l in L)))

# (a) non-minimal: n_0 = 4 > rL_0 * rR_0 = 1*2
x = t3.TuckerTensorTrain.randn((6, 6, 6), (4, 4, 4), (1, 2, 2, 1))
print('non-minimal x has_minimal_ranks:', x.has_minimal_ranks)
frame, _ = fvc.t3_orthogonal_representations(x.data)
v = rand_var(frame)
th, smin = lift(frame, v, 0.0)
cf = fvc.t3_corewise_frame((frame[0], frame[2]))
a, b = tv.tv_to_dense(cf, th), tv.tv_to_dense(frame, v)
print('(a) non-minimal lift identity rel err: %.1e   min sv over S,T: %.2e' % (np.linalg.norm(a - b) / np.linalg.norm(b), min(smin)))

# (b) zero-padded resize (rank continuation warm start)
y = t3.TuckerTensorTrain.randn((5, 6, 7), (2, 2, 2), (1, 2, 2, 1))
y, _, _ = y.t3svd()
yp = y.resize(y.shape, (3, 3, 2), (1, 3, 2, 1))
frame, _ = fvc.t3_orthogonal_representations(yp.data)
v = rand_var(frame)
for eps in (0.0, 1e-3 * np.linalg.norm(yp.to_dense())):
    th, smin = lift(frame, v, eps)
    finite = all(np.all(np.isfinite(c)) for c in th[0] + th[1])
    print('(b) padded restart eps=%.1e: finite=%s  min sv=%.1e  |theta_hat|=%.2e' % (
        eps, finite, min(smin), np.sqrt(sum(np.sum(c * c) for c in th[0] + th[1]))))

# (c) zero point
z = t3.TuckerTensorTrain.zeros((5, 6, 7), (2, 2, 2), (1, 2, 2, 1))
frame, _ = fvc.t3_orthogonal_representations(z.data)
v = rand_var(frame)
th, smin = lift(frame, v, 1e-3 * 0.0)
print('(c) zero point: |lifted Tucker| = %.1e, |lifted TT (excl. last)| = %.1e, last TT slot = H_{d-1} (undamped): %.2e' % (
    np.sqrt(sum(np.sum(c * c) for c in th[0])), np.sqrt(sum(np.sum(c * c) for c in th[1][:-1])), np.linalg.norm(th[1][-1])))
