"""Is the T3-SVD retraction a SECOND-ORDER retraction on the fixed-rank T3 manifold?

Dense numpy oracle (no t3toolbox import).  Companion to `dev/riemannian_newton_curvature_plan.md`
§2.3.  Two things are checked on a random minimal-format T3 point X:

1. The tangential part of the acceleration of t -> R_X(t xi) vanishes for the T3-SVD retraction
   (HOSVD truncation of the Tucker layer, then TT-SVD of the core), i.e. R is second order, while
   the "parameterisation retraction" Phi(theta + t theta_dot) (add the core variations and do nothing
   else) is first order only.  Finite differences in h, with Richardson extrapolation.

2. The quadratic forms along the two retractions equal, respectively, the exact Riemannian Hessian
   |A xi|^2 + <N, II(xi,xi)> (N = normal part of the Euclidean gradient) and the "core-space"
   Hessian |A xi|^2 + <grad_E f, D^2 Phi[xi_hat, xi_hat]> (full Euclidean gradient).  The former is
   what a two-hole curvature sweep must reproduce; the latter is the level-1 model of the plan.

The tangent projector is built from the Jacobian of the parameterisation (SVD of a finite-difference
Jacobian), so nothing here depends on hand-derived projection formulas.

Pitfall (observed): the point must be in MINIMAL format (r_1 <= rho_1, r_d <= rho_{d-1},
r_k <= rho_{k-1} rho_k), and the TT core must be well conditioned, otherwise the Jacobian rank drops
below the formula and the finite differences are swamped by roundoff.  The core is therefore generated
as the TT-SVD of a random dense core (singular values O(1)).

Run:  python -I dev/curvature_oracles/check_t3svd_retraction_second_order.py
"""
import numpy as np

rng = np.random.default_rng(1)
d = 4
n = (5, 6, 6, 5)       # mode sizes
r = (2, 3, 3, 2)       # Tucker ranks
rho = (2, 3, 2)        # TT ranks of the core (minimal format: r_1 <= rho_1, r_d <= rho_{d-1})


# ---------------------------------------------------------------- dense tensor utilities
def unfold(T, k):
    return np.moveaxis(T, k, 0).reshape(T.shape[k], -1)


def fold(M, k, shape):
    shp = list(shape)
    nk = shp.pop(k)
    return np.moveaxis(M.reshape([nk] + shp), 0, k)


def mode_prod(T, M, k):
    """T x_k M  with M of shape (p, T.shape[k])."""
    return fold(M @ unfold(T, k), k, T.shape[:k] + (M.shape[0],) + T.shape[k + 1:])


def tt_full(cores):
    T = cores[0]                      # (1, r_0, rho_0)
    for C in cores[1:]:
        T = np.tensordot(T, C, axes=([-1], [0]))
    return T.reshape(T.shape[1:-1])


def tucker(G, Us):
    X = G
    for k, U in enumerate(Us):
        X = mode_prod(X, U, k)
    return X


def phi(params):
    """The multilinear T3 parameterisation Phi(U_1..U_d, G_1..G_d)."""
    Us, cores = params
    return tucker(tt_full(cores), Us)


def hosvd(Y, ranks):
    Us = []
    for k in range(Y.ndim):
        Uk, _, _ = np.linalg.svd(unfold(Y, k), full_matrices=False)
        Us.append(Uk[:, :ranks[k]])
    G = Y
    for k, U in enumerate(Us):
        G = mode_prod(G, U.T, k)
    return G, Us


def tt_svd(G, ranks):
    cores = []
    C = G.reshape(1, -1)
    rprev = 1
    for k in range(G.ndim - 1):
        C = C.reshape(rprev * G.shape[k], -1)
        U, s, Vt = np.linalg.svd(C, full_matrices=False)
        U, s, Vt = U[:, :ranks[k]], s[:ranks[k]], Vt[:ranks[k]]
        cores.append(U.reshape(rprev, G.shape[k], ranks[k]))
        C = s[:, None] * Vt
        rprev = ranks[k]
    cores.append(C.reshape(rprev, G.shape[-1], 1))
    return cores


def t3_svd_retraction(Y):
    """HOSVD truncation of the Tucker layer, then TT-SVD of the core (the T3-SVD retraction)."""
    G, Us = hosvd(Y, r)
    return tucker(tt_full(tt_svd(G, rho)), Us)


# ---------------------------------------------------------------- random minimal-format point
Us = [np.linalg.qr(rng.standard_normal((n[k], r[k])))[0] for k in range(d)]
cores = tt_svd(rng.standard_normal(r), rho)   # well-conditioned TT core of exact ranks rho
params = (Us, cores)
X = phi(params)


# ---------------------------------------------------------------- tangent projector from the Jacobian
def flatten(p):
    return np.concatenate([a.ravel() for a in p[0]] + [c.ravel() for c in p[1]])


def unflatten(v, like):
    out_U, out_C, i = [], [], 0
    for a in like[0]:
        out_U.append(v[i:i + a.size].reshape(a.shape))
        i += a.size
    for c in like[1]:
        out_C.append(v[i:i + c.size].reshape(c.shape))
        i += c.size
    return (out_U, out_C)


def jacobian(p, eps=1e-6):
    v0 = flatten(p)
    cols = []
    for i in range(v0.size):
        e = np.zeros_like(v0)
        e[i] = eps
        cols.append((phi(unflatten(v0 + e, p)) - phi(unflatten(v0 - e, p))).ravel() / (2 * eps))
    return np.array(cols).T


J = jacobian(params)
Q, s, _ = np.linalg.svd(J, full_matrices=False)
rank = int(np.sum(s > 1e-8 * s[0]))
print(f"Jacobian singular values: max {s[0]:.3e}, min nonzero {s[rank - 1]:.3e}, next {s[rank]:.3e}")
Q = Q[:, :rank]
rr = (1,) + tuple(rho) + (1,)
expected_dim = sum(n[k] * r[k] - r[k] * r[k] for k in range(d)) + \
    sum(rr[k] * r[k] * rr[k + 1] for k in range(d)) - sum(x * x for x in rho)
print(f"tangent space dim from Jacobian = {rank}, expected = {expected_dim}")
P = lambda Z: (Q @ (Q.T @ Z.ravel())).reshape(n)        # orthogonal projector onto T_X M

xi = P(rng.standard_normal(n))
xi /= np.linalg.norm(xi)

# ---------------------------------------------------------------- 1. tangential acceleration
print("\nTangential part of the retraction acceleration (-> 0 like h^2 for a second-order retraction):")
for h in [4e-2, 2e-2, 1e-2, 5e-3]:
    acc = (t3_svd_retraction(X + h * xi) - 2 * X + t3_svd_retraction(X - h * xi)) / h ** 2
    acc2 = (t3_svd_retraction(X + 2 * h * xi) - 2 * X + t3_svd_retraction(X - 2 * h * xi)) / (2 * h) ** 2
    rich = (4 * acc - acc2) / 3
    print(f"  T3-SVD: h={h:.1e}  |P acc| = {np.linalg.norm(P(acc)):.3e}"
          f"  |P acc_richardson| = {np.linalg.norm(P(rich)):.3e}   |acc| = {np.linalg.norm(acc):.3e}")

# Contrast: the parameterisation retraction Phi(theta + t theta_dot), theta_dot = min-norm lift of xi.
theta_dot = np.linalg.lstsq(J, xi.ravel(), rcond=None)[0]
v0 = flatten(params)
Rparam = lambda t: phi(unflatten(v0 + t * theta_dot, params))
print("\nContrast — parameterisation retraction Phi(theta + t theta_dot) (first order only):")
for h in [1e-2, 5e-3]:
    acc = (Rparam(h) - 2 * X + Rparam(-h)) / h ** 2
    print(f"  h={h:.1e}  |P acc| = {np.linalg.norm(P(acc)):.3e}   |acc| = {np.linalg.norm(acc):.3e}")

# ---------------------------------------------------------------- 2. quadratic forms
m = 80
A = rng.standard_normal((m, X.size))
y = rng.standard_normal(m)
f = lambda Y: 0.5 * np.sum((A @ Y.ravel() - y) ** 2)
egrad = (A.T @ (A @ X.ravel() - y)).reshape(n)          # Euclidean gradient A^T r
Nf = egrad - P(egrad)                                    # its normal part
h = 5e-3
acc_param = (Rparam(h) - 2 * X + Rparam(-h)) / h ** 2    # = D^2 Phi[theta_dot, theta_dot] + O(h^2)
II = acc_param - P(acc_param)                            # second fundamental form II(xi, xi)
gn = np.sum((A @ xi.ravel()) ** 2)
hess_riem = gn + np.vdot(Nf, II)
hess_core = gn + np.vdot(egrad, acc_param)
d2 = lambda fn: (fn(h) - 2 * fn(0.0) + fn(-h)) / h ** 2
print("\nQuadratic forms <xi, H xi> for f = 0.5|A x - y|^2:")
print(f"  Gauss-Newton  |A xi|^2                                   = {gn:.6f}")
print(f"  Riemannian Hessian  |A xi|^2 + <N, II(xi,xi)>            = {hess_riem:.6f}")
print(f"  FD d2/dt2 f(T3-SVD retraction)                           = {d2(lambda t: f(t3_svd_retraction(X + t * xi))):.6f}")
print(f"  core-space (level 1)  |A xi|^2 + <A^T r, D2Phi[xi,xi]>   = {hess_core:.6f}")
print(f"  FD d2/dt2 f(parameterisation retraction)                 = {d2(lambda t: f(Rparam(t))):.6f}")
print(f"  |Riemannian grad| = {np.linalg.norm(P(egrad)):.3f},  |normal part N| = {np.linalg.norm(Nf):.3f}")
