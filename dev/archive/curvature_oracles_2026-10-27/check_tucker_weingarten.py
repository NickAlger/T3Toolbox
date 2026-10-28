"""Explicit Weingarten map (curvature term) on the fixed-multilinear-rank Tucker manifold.

Dense numpy oracle (no t3toolbox import).  Companion to `dev/riemannian_newton_curvature_plan.md`
§2.2.  The Tucker layer is the part of the T3 manifold whose Weingarten map we derived by hand; the
TT layer is Psenka & Boumal's (arXiv:2011.13395).  Checks, on a random point X = G x_k U_k:

1. The explicit block formula for W_X(xi, N) (below) equals the finite-difference derivative of the
   tangent projector, P_X( d/dt P_{X(t)} N ), along a curve X(t) in M with velocity xi.
2. <W(xi,N), eta> == <N, D^2 Phi[xi_hat, eta_hat]> (second derivative of the multilinear
   parameterisation on lifts of xi, eta) and is symmetric in (xi, eta).  This is the identity the
   plan's "two-hole sweep" rests on.
3. The Riemannian Hessian quadratic form |A xi|^2 + <N, D^2 Phi[xi,xi]> equals d^2/dt^2 f(R(t xi))
   for the exact metric projection R (HOOI) and for the HOSVD retraction (both second order), while
   the parameterisation retraction Phi(theta + t theta_dot) reproduces the level-1 form
   |A xi|^2 + <A^T r, D^2 Phi[xi,xi]> instead.
4. Tangential accelerations: ~0 for the exact projection and for HOSVD; nonzero for the
   parameterisation retraction; the normal part equals D^2 Phi's normal part in all three cases.

Block formula (tangent coordinates (G_dot; V_1..V_d), U_k^T V_k = 0, N normal):
    N_k      := [N x_{j!=k} U_j^T]_(k)
    N_k^perp := P_{U_k}^perp N_k (I - G_(k)^+ G_(k))          single-mode normal slot k
    N_{k;l}(V_l) := [N x_l V_l^T x_{j!=k,l} U_j^T]_(k)          doubly-orthogonal slot (k,l)
    G_dot_out = sum_k fold_k( V_k^T N_k^perp )
    V_k_out   = sum_{l!=k} P_{U_k}^perp N_{k;l}(V_l) G_(k)^+  +  N_k^perp G_dot_(k)^T (G_(k) G_(k)^T)^{-1}
For d = 2 (matrices) the single-mode slots are empty and this is U Sigma^{-1} U_p^T N + N V_p Sigma^{-1} V^T.

Run:  python -I dev/curvature_oracles/check_tucker_weingarten.py
"""
import numpy as np

rng = np.random.default_rng(0)
d = 3
n = (7, 6, 8)
r = (3, 2, 3)


# ---------------------------------------------------------------- tensor utilities
def unfold(T, k):
    return np.moveaxis(T, k, 0).reshape(T.shape[k], -1)


def fold(M, k, shape):
    shp = list(shape)
    nk = shp.pop(k)
    return np.moveaxis(M.reshape([nk] + shp), 0, k)


def mode_prod(T, M, k):
    """T x_k M  (M is (p x T.shape[k]))."""
    return fold(M @ unfold(T, k), k, T.shape[:k] + (M.shape[0],) + T.shape[k + 1:])


def tucker(G, Us):
    X = G
    for k, U in enumerate(Us):
        X = mode_prod(X, U, k)
    return X


def orth(A):
    Q, _ = np.linalg.qr(A)
    return Q


def hosvd(Y, ranks):
    Us = []
    for k in range(Y.ndim):
        Uk, _, _ = np.linalg.svd(unfold(Y, k), full_matrices=False)
        Us.append(Uk[:, :ranks[k]])
    G = Y
    for k, U in enumerate(Us):
        G = mode_prod(G, U.T, k)
    return G, Us


def hooi(Y, ranks, iters=200, tol=1e-15):
    """Exact metric projection onto fixed multilinear rank (local optimum near the HOSVD)."""
    G, Us = hosvd(Y, ranks)
    for _ in range(iters):
        Us_old = [U.copy() for U in Us]
        for k in range(Y.ndim):
            Z = Y
            for j, U in enumerate(Us):
                if j != k:
                    Z = mode_prod(Z, U.T, j)
            Uk, _, _ = np.linalg.svd(unfold(Z, k), full_matrices=False)
            Us[k] = Uk[:, :ranks[k]]
        if max(np.linalg.norm(U @ U.T - Uo @ Uo.T) for U, Uo in zip(Us, Us_old)) < tol:
            break
    G = Y
    for k, U in enumerate(Us):
        G = mode_prod(G, U.T, k)
    return G, Us


# ---------------------------------------------------------------- point, tangent, normal
G = rng.standard_normal(r)
Us = [orth(rng.standard_normal((n[k], r[k]))) for k in range(d)]
X = tucker(G, Us)
Pperp = [np.eye(n[k]) - Us[k] @ Us[k].T for k in range(d)]


def rand_tangent():
    Gd = rng.standard_normal(r)
    Vs = [Pperp[k] @ rng.standard_normal((n[k], r[k])) for k in range(d)]
    return Gd, Vs


def tangent_to_ambient(Gd, Vs, G=G, Us=Us):
    xi = tucker(Gd, Us)
    for k in range(d):
        xi = xi + tucker(G, [Vs[k] if j == k else Us[j] for j in range(d)])
    return xi


def project(Z, G=G, Us=Us):
    """Orthogonal projection onto the Tucker tangent space (Koch-Lubich); returns (Gd, Vs)."""
    Gd = Z
    for k, U in enumerate(Us):
        Gd = mode_prod(Gd, U.T, k)
    Vs = []
    for k in range(d):
        Zk = Z
        for j, U in enumerate(Us):
            if j != k:
                Zk = mode_prod(Zk, U.T, j)
        Zk = unfold(Zk, k)
        Gk = unfold(G, k)
        Vk = (np.eye(n[k]) - Us[k] @ Us[k].T) @ Zk @ np.linalg.pinv(Gk)
        Vs.append(Vk)
    return Gd, Vs


def P(Z, G=G, Us=Us):
    return tangent_to_ambient(*project(Z, G, Us), G=G, Us=Us)


# ---------------------------------------------------------------- explicit Weingarten formula
def weingarten(xi_coords, N):
    """W_X(xi, N) in tangent coordinates (Gd_out, Vs_out), N normal.  See module docstring."""
    Gd, Vs = xi_coords
    Gd_out = np.zeros(r)
    Vs_out = []
    for k in range(d):
        Gk = unfold(G, k)
        Gk_pinv = np.linalg.pinv(Gk)
        Nk = N
        for j in range(d):
            if j != k:
                Nk = mode_prod(Nk, Us[j].T, j)
        Nk = unfold(Nk, k)
        Nk_perp = Pperp[k] @ Nk @ (np.eye(Gk.shape[1]) - Gk_pinv @ Gk)
        Gd_out += fold(Vs[k].T @ Nk_perp, k, r)
        Vk_out = Nk_perp @ unfold(Gd, k).T @ np.linalg.inv(Gk @ Gk.T)
        for l in range(d):
            if l == k:
                continue
            Nkl = N
            for j in range(d):
                if j == k:
                    continue
                Nkl = mode_prod(Nkl, Vs[l].T if j == l else Us[j].T, j)
            Vk_out += Pperp[k] @ unfold(Nkl, k) @ Gk_pinv
        Vs_out.append(Vk_out)
    return Gd_out, Vs_out


def d2phi(a, b):
    """D^2 Phi[a, b] for Phi(G, U_1..U_d) = G x_k U_k, along lifts a = (Gd, Vs), b = (Hd, Ws):
    the sum over all ordered pairs of distinct cores with both replaced by their variations."""
    Gd, Vs = a
    Hd, Ws = b
    out = np.zeros(n)
    for k in range(d):
        out += tucker(Gd, [Ws[k] if j == k else Us[j] for j in range(d)])
        out += tucker(Hd, [Vs[k] if j == k else Us[j] for j in range(d)])
        for l in range(d):
            if l != k:
                out += tucker(G, [Vs[k] if j == k else (Ws[l] if j == l else Us[j]) for j in range(d)])
    return out


def inner_tangent(a, b):
    return np.vdot(tangent_to_ambient(*a), tangent_to_ambient(*b))


# ---------------------------------------------------------------- check 1: formula vs FD of the projector
Z = rng.standard_normal(n)
N = Z - P(Z)
assert np.linalg.norm(P(N)) < 1e-12
xi = rand_tangent()
eta = rand_tangent()

h = 1e-5


def P_on_curve(t):
    """P_{X(t)} N for X(t) = Phi(theta + t theta_dot) (re-orthonormalised: same point of M)."""
    Gt = G + t * xi[0]
    Ut = [Us[k] + t * xi[1][k] for k in range(d)]
    Qs, Rs = zip(*[np.linalg.qr(U) for U in Ut])
    Gt2 = Gt
    for k, R in enumerate(Rs):
        Gt2 = mode_prod(Gt2, R, k)
    return P(N, G=Gt2, Us=list(Qs))


DP_N_fd = (P_on_curve(h) - P_on_curve(-h)) / (2 * h)
W_fd = P(DP_N_fd)
W_formula = tangent_to_ambient(*weingarten(xi, N))
print("check 1  |W_formula - W_fd| / |W| =",
      np.linalg.norm(W_formula - W_fd) / np.linalg.norm(W_formula))

# ---------------------------------------------------------------- check 2: symmetry and <N, D2Phi>
lhs = inner_tangent(weingarten(xi, N), eta)
lhs_sym = inner_tangent(weingarten(eta, N), xi)
rhs = np.vdot(N, d2phi(xi, eta))
print("check 2  <W(xi,N),eta> =", lhs, " <W(eta,N),xi> =", lhs_sym, " <N, D2Phi[xi,eta]> =", rhs)

# ---------------------------------------------------------------- check 3: full Hessian quadratic form
m = 60
A = rng.standard_normal((m, int(np.prod(n))))
y = rng.standard_normal(m)
f = lambda Y: 0.5 * np.sum((A @ Y.ravel() - y) ** 2)
res = A @ X.ravel() - y
egrad = (A.T @ res).reshape(n)
N_f = egrad - P(egrad)
xi_amb = tangent_to_ambient(*xi)
gn = np.sum((A @ xi_amb.ravel()) ** 2)
hess_formula = gn + np.vdot(N_f, d2phi(xi, xi))
hess_param = gn + np.vdot(egrad, d2phi(xi, xi))      # level-1 (core-space) Hessian

h = 1e-3
f_proj = lambda t: f(tucker(*hooi(X + t * xi_amb, r)))        # exact projection (second order)
f_hosvd = lambda t: f(tucker(*hosvd(X + t * xi_amb, r)))      # HOSVD retraction
f_param = lambda t: f(tucker(G + t * xi[0], [Us[k] + t * xi[1][k] for k in range(d)]))
d2 = lambda fn: (fn(h) - 2 * fn(0.0) + fn(-h)) / h ** 2
print("check 3  Riemannian Hessian quadratic form:")
print("   formula  |A xi|^2 + <N, D2Phi[xi,xi]>           =", hess_formula)
print("   FD along exact projection retraction (HOOI)     =", d2(f_proj))
print("   FD along HOSVD retraction                        =", d2(f_hosvd))
print("   FD along parameterisation retraction             =", d2(f_param))
print("   level-1 formula |A xi|^2 + <A^T r, D2Phi[xi,xi]> =", hess_param)
print("   Gauss-Newton  |A xi|^2                            =", gn)

# ---------------------------------------------------------------- check 4: tangential accelerations
acc = lambda curve: (curve(h) - 2 * curve(0.0) + curve(-h)) / h ** 2
Y_proj = lambda t: tucker(*hooi(X + t * xi_amb, r))
Y_hosvd = lambda t: tucker(*hosvd(X + t * xi_amb, r))
Y_param = lambda t: tucker(G + t * xi[0], [Us[k] + t * xi[1][k] for k in range(d)])
D2 = d2phi(xi, xi)
for name, Y in [("exact projection", Y_proj), ("HOSVD", Y_hosvd), ("parameterisation", Y_param)]:
    a = acc(Y)
    print(f"check 4  {name:17s}: |P acc| = {np.linalg.norm(P(a)):.3e}   |Pperp acc| = {np.linalg.norm(a - P(a)):.3e}"
          f"   |Pperp acc - Pperp D2Phi| = {np.linalg.norm((a - P(a)) - (D2 - P(D2))):.3e}")
