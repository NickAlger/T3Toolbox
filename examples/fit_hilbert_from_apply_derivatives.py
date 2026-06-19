"""Fit a Tucker tensor train to APPLY-DERIVATIVE data of a Hilbert tensor.

A worked example of Riemannian fixed-rank fitting from **symmetric directional derivatives** of the
all-modes ``apply`` operation -- the derivative analogue of ``examples/fit_hilbert_tensor_newton_cg.py``
(which fits from ordinary applies). It is the end-to-end integration test of the derivative pipeline:
the forward ``T3Tangent.apply_derivatives`` (the Jacobian ``J``) and its transpose
``apply_derivatives_transpose`` (``J^T``) drive the same Riemannian inexact Newton-CG solver.

The problem
-----------
The order-``d`` **Hilbert tensor** ``A[i0,...] = 1 / (1 + i0 + ... + i_{d-1})`` is smooth, positive, and
well approximated by low rank. We never see ``A`` directly.

    *Sampling: a few base points, many directions each.*  We sample ``N_X`` random base points
    ``X = (x0,...,x_{d-1})`` and, **for each ``X``**, ``N_P`` random perturbation directions
    ``P = (p0,...,p_{d-1})``.  So the perturbation stack is ``(N_P, N_X)`` and the base stack starts as
    ``(N_X,)``; we **replicate** ``X`` over the ``N_P`` axis to give it the same ``(N_P, N_X)`` stack as
    ``P`` (the sampling functions pair an ``X`` with a ``P`` per stack element).  This mimics the common
    use case where base points are the scarce resource and the directional derivatives at each base point
    are what is cheap -- exactly the regime where higher-order derivatives buy the most (see
    ``docs/derivative_order_information_and_conditioning.md``).

For each ``(X, P)`` we observe the **apply-derivative jet**

    b = [ d^t/ds^t  A(x0 + s p0, ..., x_{d-1} + s p_{d-1}) |_{s=0} ]_{t=0..K},

a vector of ``K+1`` numbers (order 0 is the ordinary apply). The restriction to the line ``X + sP`` is a
degree-``d`` polynomial in ``s``, so orders ``0..d`` carry the whole line and orders ``> d`` vanish. We
measure on a training set of base points and a held-out validation set, with noise.

    *Two normalizations.*  (1) **Unit-norm probe vectors** -- each sample's rank-1 row has unit Frobenius
    norm, so no sample dominates.  (2) **Per-order normalization** -- the order-``t`` derivative carries a
    ``t!``/binomial-type weight, so raw orders span wildly different magnitudes and (left alone) wreck the
    Gauss-Newton conditioning. We divide each order's data (and the operator output) by that order's RMS
    over the training set. This is the essential step that makes the derivative fit well-behaved.

The method
----------
Identical Riemannian inexact Newton-CG with Armijo backtracking as the apply-only example; only the
forward / transpose / measurement-inner-product closures change. The forward is
``Z.apply_derivatives(ww, pp, K)`` (works on a ``TuckerTensorTrain`` point or a ``T3Tangent`` Jacobian),
the gradient is the per-order-weighted ``apply_derivatives_transpose(..., sum_over_probes=True)``
gauge-projected (this sums the whole ``(N_P, N_X)`` sample stack), and the Gauss-Newton Hessian is
``V -> Jᵀ J V`` gauged. Rank continuation starts from the zero tensor and grows rank by zero-padding;
validation picks the rank.

Run from the repo root:  ``python examples/fit_hilbert_from_apply_derivatives.py``
"""
import numpy as np

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.basis_variations_format as bvf
import t3toolbox.manifold as t3m
import t3toolbox.backend.probe_derivatives as pd


# --------------------------------------------------------------------------------------------------
# Problem configuration (tweak these)
# --------------------------------------------------------------------------------------------------
SHAPE        = (12, 12, 12, 12)    # Hilbert tensor shape (order d = len(SHAPE))
ORDER        = 4                   # highest derivative order K (apply has degree d; orders 0..d carry it)
N_X_TRAIN    = 10                  # training base points X
N_X_VAL      = 5                   # held-out validation base points X
N_P          = 30                  # perturbation directions P sampled per base point X
NOISE_LEVEL  = 0.01                # measurement noise, fraction of the per-order RMS
RANK_LEVELS  = (1, 2, 3, 4, 5)     # rank-continuation schedule
SEED         = 0

MAX_NEWTON   = 30
GTOL_REL     = 1e-8
CG_MAXITER   = 200


# --------------------------------------------------------------------------------------------------
# Target tensor + the (dense) ground-truth derivative-measurement operator
# --------------------------------------------------------------------------------------------------
def hilbert_tensor(shape):
    """Dense Hilbert tensor A[i0,...] = 1 / (1 + i0 + ... + i_{d-1})."""
    grids = np.meshgrid(*[np.arange(N) for N in shape], indexing="ij")
    return 1.0 / (1.0 + sum(grids))


def unit_pairs_many_p(n_x, n_p, shape, rng):
    """``n_x`` unit-norm base points ``X``, each with ``n_p`` unit-norm directions ``P``; ``X``
    replicated over the ``n_p`` axis so it shares ``P``'s ``(n_p, n_x)`` sample stack.

    Returns ``ww, pp``: lists of ``d`` arrays each of shape ``(n_p, n_x, N)`` (``ww`` is ``X`` broadcast
    across the ``n_p`` directions; sample ``(p, x)`` pairs base point ``x`` with its ``p``-th direction)."""
    def unit(*lead):
        vs = [rng.standard_normal(lead + (N,)) for N in shape]
        return [v / np.linalg.norm(v, axis=-1, keepdims=True) for v in vs]
    xx = unit(n_x)                                              # X: stack (n_x,)
    pp = unit(n_p, n_x)                                         # P: stack (n_p, n_x)
    ww = [np.broadcast_to(x[None], (n_p, n_x, N)).copy()        # X replicated -> stack (n_p, n_x)
          for x, N in zip(xx, shape)]
    return ww, pp


def dense_apply_derivatives(A, ww, pp, order):
    """Ground-truth apply-derivative jets of the dense tensor ``A``: shape ``(order+1,) + W`` (``W`` the
    sample stack). Each ``W``-element is ``apply_derivatives_dense`` (the exact subset expansion)."""
    W = ww[0].shape[:-1]
    b = np.zeros((order + 1,) + W)
    for idx in np.ndindex(*W):
        ww_m = [w[idx] for w in ww]
        pp_m = [p[idx] for p in pp]
        b[(slice(None),) + idx] = pd.apply_derivatives_dense(ww_m, pp_m, A, order)
    return b


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


# --------------------------------------------------------------------------------------------------
# The measurement operator (per-order-normalized apply-derivatives). The optimizer below uses only
# `forward`, `transpose`, `meas_dot`.
# --------------------------------------------------------------------------------------------------
def apply_derivative_operator(ww, pp, order, s_vec):
    """forward / transpose / meas_dot for the per-order-normalized apply-derivative map.

    Dividing the output (and the data) by ``s_vec`` (per order, broadcast over the sample stack ``W``)
    folds the per-order normalization into the operator, so ``transpose(r)`` applies ``J^T`` to
    ``r / s_vec``. ``sum_over_probes=True`` sums the whole ``W = (n_p, n_x)`` stack -> one tangent."""
    nW = ww[0].ndim - 1
    sv = s_vec.reshape((order + 1,) + (1,) * nW)               # broadcast over the W sample stack
    forward = lambda Z: np.asarray(Z.apply_derivatives(ww, pp, order)) / sv
    transpose = lambda r, base: t3m.T3Tangent.apply_derivatives_transpose(
        np.asarray(r) / sv, ww, pp, base, order, sum_over_probes=True)
    meas_dot = lambda a, b: float(np.sum(np.asarray(a) * np.asarray(b)))
    return forward, transpose, meas_dot


# --------------------------------------------------------------------------------------------------
# Riemannian inexact Newton-CG with Armijo line search (same as the apply-only example)
# --------------------------------------------------------------------------------------------------
def _tangent_cg(H, rhs, base, tol, maxiter):
    x = t3m.T3Tangent.zeros(base)
    res = rhs
    p = res
    rs = float(res.corewise_inner(res))
    if np.sqrt(rs) <= tol:
        return x, 0
    i = 0
    for i in range(1, maxiter + 1):
        Hp = H(p)
        pHp = float(p.corewise_inner(Hp))
        if pHp <= 1e-30:
            if i == 1:
                x = rhs
            break
        alpha = rs / pHp
        x = x + alpha * p
        res = res - alpha * Hp
        rs_new = float(res.corewise_inner(res))
        if np.sqrt(rs_new) <= tol:
            break
        p = res + (rs_new / rs) * p
        rs = rs_new
    return x, i


def riemannian_newton_cg(X0, forward, transpose, meas_dot, b,
                         max_newton=MAX_NEWTON, gtol_rel=GTOL_REL, cg_maxiter=CG_MAXITER,
                         c_armijo=1e-4):
    X = X0
    g0norm = None
    newton_iters = 0
    for it in range(max_newton):
        base, _ = bvf.t3_orthogonal_representations(X)
        r = forward(X) - b
        f = 0.5 * meas_dot(r, r)
        g = t3m.MANIFOLD.project(transpose(r, base))
        gnorm = float(g.corewise_norm())
        if g0norm is None:
            g0norm = gnorm if gnorm > 0.0 else 1.0
        if gnorm <= gtol_rel * g0norm:
            break
        newton_iters += 1

        def H(V):
            return t3m.MANIFOLD.project(transpose(forward(V), base))

        eta = min(0.5, np.sqrt(gnorm / g0norm))
        p, _ = _tangent_cg(H, -g, base, tol=eta * gnorm, maxiter=cg_maxiter)
        slope = float(g.corewise_inner(p))
        if (not np.isfinite(slope)) or slope >= 0.0:
            p, slope = -g, -gnorm * gnorm
        alpha = 1.0
        X_trial = X
        for _ in range(40):
            X_trial = t3m.MANIFOLD.retract(alpha * p)
            r_trial = forward(X_trial) - b
            if 0.5 * meas_dot(r_trial, r_trial) <= f + c_armijo * alpha * slope:
                break
            alpha *= 0.5
        X = X_trial
    return X, dict(newton=newton_iters)


# --------------------------------------------------------------------------------------------------
# Rank continuation helpers
# --------------------------------------------------------------------------------------------------
def level_ranks(level, shape):
    d = len(shape)
    tucker_ranks = tuple(min(level, N) for N in shape)
    tt_ranks = (1,) + (level,) * (d - 1) + (1,)
    return tucker_ranks, tt_ranks


def oracle_relerr(A, tucker_ranks, tt_ranks):
    Xr, _, _ = t3.TuckerTensorTrain.t3svd_dense(A, max_tucker_ranks=tucker_ranks, max_tt_ranks=tt_ranks)
    return float(np.linalg.norm(Xr.to_dense() - A)) / float(np.linalg.norm(A))


# --------------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------------
def main():
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    d = len(SHAPE)

    print(__doc__.split("\n\n")[0])
    print(f"\nHilbert tensor: shape {SHAPE}  ({np.prod(SHAPE):,} entries),  order d={d}")
    print(f"Measurements: apply-derivatives orders 0..{ORDER}  "
          f"({N_X_TRAIN} train base pts + {N_X_VAL} val base pts, {N_P} directions each, "
          f"{NOISE_LEVEL*100:.0f}% noise).")
    print("Many directions per base point; probe vectors unit-norm; per-order normalized.\n")

    A = hilbert_tensor(SHAPE)
    A_norm = float(np.linalg.norm(A))

    ww_tr, pp_tr = unit_pairs_many_p(N_X_TRAIN, N_P, SHAPE, rng)   # train: W=(N_P, N_X_TRAIN)
    ww_va, pp_va = unit_pairs_many_p(N_X_VAL, N_P, SHAPE, rng)     # val:   W=(N_P, N_X_VAL)
    b_tr_clean = dense_apply_derivatives(A, ww_tr, pp_tr, ORDER)   # (order+1, N_P, N_X_TRAIN)
    b_va_clean = dense_apply_derivatives(A, ww_va, pp_va, ORDER)

    # per-order scale from the (clean) training data; floor avoids dividing a vanishing order by ~0
    s_vec = np.array([max(rms(b_tr_clean[t]), 1e-12) for t in range(ORDER + 1)])
    sv_tr = s_vec.reshape((ORDER + 1,) + (1,) * (b_tr_clean.ndim - 1))
    sv_va = s_vec.reshape((ORDER + 1,) + (1,) * (b_va_clean.ndim - 1))

    b_tr = b_tr_clean + NOISE_LEVEL * sv_tr * rng.standard_normal(b_tr_clean.shape)
    b_va = b_va_clean + NOISE_LEVEL * sv_va * rng.standard_normal(b_va_clean.shape)

    fwd_tr, T_tr, mdot = apply_derivative_operator(ww_tr, pp_tr, ORDER, s_vec)
    fwd_va, _, _ = apply_derivative_operator(ww_va, pp_va, ORDER, s_vec)
    b_tr_n = b_tr / sv_tr                                          # normalized data
    b_va_n = b_va / sv_va
    b_tr_rms = rms(b_tr_n)
    b_va_rms = rms(b_va_n)

    header = (f"{'level':>5} {'tucker / tt ranks':>24} {'DOF':>5} {'iters':>5} "
              f"{'train':>9} {'val':>9} {'true':>9} {'oracle':>9}")
    print(header)
    print("-" * len(header))

    X = t3.TuckerTensorTrain.zeros(SHAPE, *level_ranks(RANK_LEVELS[0], SHAPE))
    records = []
    for r in RANK_LEVELS:
        tucker_ranks, tt_ranks = level_ranks(r, SHAPE)
        dof = t3m.manifold_dim((SHAPE, tucker_ranks, tt_ranks))
        X0 = X.resize(SHAPE, tucker_ranks, tt_ranks)
        X, stats = riemannian_newton_cg(X0, fwd_tr, T_tr, mdot, b_tr_n)

        train_e = rms(fwd_tr(X) - b_tr_n) / b_tr_rms
        val_e = rms(fwd_va(X) - b_va_n) / b_va_rms
        true_e = float(np.linalg.norm(X.to_dense() - A)) / A_norm
        oracle_e = oracle_relerr(A, tucker_ranks, tt_ranks)
        records.append(dict(level=r, val=val_e, true=true_e, dof=dof))
        rank_str = f"{tucker_ranks} {tt_ranks}"
        print(f"{r:>5} {rank_str:>24} {dof:>5} {stats['newton']:>5} "
              f"{train_e:>9.3e} {val_e:>9.3e} {true_e:>9.3e} {oracle_e:>9.3e}")

    best = min(records, key=lambda rec: rec["val"])
    print("-" * len(header))
    print(f"\nNoise floor (relative): {NOISE_LEVEL:.1e}")
    print(f"Best ranks by validation error: level {best['level']}  "
          f"(val {best['val']:.3e}, true error {best['true']:.3e}, DOF {best['dof']}).")


if __name__ == "__main__":
    main()
