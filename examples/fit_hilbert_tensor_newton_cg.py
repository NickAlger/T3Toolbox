"""Fit a Tucker tensor train to sampled "apply" probes of a Hilbert tensor.

A worked example of Riemannian fixed-rank optimization with the *verified ragged* modules of
T3Toolbox (``tucker_tensor_train``, ``basis_variations_format``, ``manifold`` + their backends).

The problem
-----------
The order-``d`` **Hilbert tensor** ``A[i0, ..., i_{d-1}] = 1 / (1 + i0 + ... + i_{d-1})`` is smooth,
positive, and famously well approximated by low (multilinear) rank even though it is full rank. We
never assume we can see ``A`` directly. Instead we observe a set of **applies** (one of the three
sampling operations -- see ``docs/entries_apply_probe.md``): each measurement is the scalar
multilinear form

    b_s = <A, w0_s (x) w1_s (x) ... (x) w_{d-1}_s>  =  A.apply([w0_s, ..., w_{d-1}_s]),

for random probe vectors ``w``. We measure ``M`` of them, add a little noise, and **split them into a
training set and a held-out validation set**.

    *Normalize the probing data.*  Each measurement's "row" is the rank-1 tensor
    ``w0_s (x) ... (x) w_{d-1}_s``, whose Frobenius norm is ``prod_m ||w_m_s||``.  With raw Gaussian
    probes this norm is wildly heavy-tailed (a few measurements dwarf the rest), which makes the
    least-squares badly conditioned.  We scale every probe vector to unit norm so each row has unit
    Frobenius norm -- equivalently, divide each ``b_s`` by ``prod_m ||w_m_s||`` after the fact.  This
    one step is what makes the fit recover ``A`` from a modest number of samples.

The method
----------
We fit a ``TuckerTensorTrain`` ``X`` of fixed rank by minimizing the training misfit

    f(X) = 1/2 ||S(X) - b_train||^2,   S(X)_s = X.apply([w0_s, ...]),

over the fixed-rank manifold, with **Riemannian inexact Newton-CG and an Armijo line search**:

  * the orthogonal frame at ``X`` comes from ``t3_orthogonal_representations`` (a ``T3Basis``);
  * the gradient is matrix-free -- ``T3Tangent.apply_transpose(r, ww, base, sum_over_probes=True)``
    gauge-projected -- and the Gauss-Newton Hessian-vector product is
    ``V -> apply_transpose(V.apply(ww), ...)`` gauge-projected (symmetric PSD);
  * the Newton system is solved approximately by CG in the tangent space (the "inexact" part), with a
    forcing term that tightens as we converge;
  * each step is retracted back to the manifold by ``T3Tangent.retract`` and accepted by backtracking.

The optimizer only touches the forward map, its transpose, and a measurement-space inner product, so
swapping ``apply`` for ``entries`` or ``probe`` is a one-line change of the operator closures.

Rank continuation + model selection
-----------------------------------
We do **basic rank continuation**: start from the **zero tensor**, fit at rank 1, then raise the ranks
by one and refit, each time **warm-starting by zero-padding** the converged cores up to the new ranks
(``resize``). No random restarts or nudges are used -- starting at zero and growing by zero-padding is,
perhaps surprisingly, the most robust initialization here. A cold *random* start often stalls at a bad
local minimum (Gauss-Newton has no negative-curvature escape); the zero / zero-padded start is
deterministic and lands in the right basin, because ``t3_orthogonal_representations`` *completes* the
rank-deficient frame with arbitrary orthonormal directions, and the first gradient step then bumps the
iterate off the rank-deficient submanifold onto the full-rank manifold (the ranks come out minimal at
every level).

*Why continuation helps (not just the zero start).* The Gauss-Newton Hessian here is ill-conditioned
and only gets *worse* with rank, so continuation does not improve its conditioning. What it does is
keep each refit **near its solution**: the lower ranks are already converged, so the warm-started
gradient is tiny and carries almost no weight in those already-resolved (and worst-conditioned)
directions -- CG only has to resolve the small new rank block. (Jumping straight to a high rank from
zero instead lands in a bad local minimum and fails.) Normalizing the probes (above) and continuation
attack two *different* ill-conditionings -- the sampling operator's, and the tensor's spectrum.

We track the **validation** misfit at every rank level and pick the level that minimizes it -- the
held-out data tells us when extra rank stops helping and starts fitting noise. The ``oracle`` column
(the best possible rank-``r`` approximation of ``A``, from a dense T3-SVD) shows the fit is
near-optimal until overfitting sets in.

Run from the repo root:  ``python examples/fit_hilbert_tensor_newton_cg.py``
(or with ``PYTHONPATH`` set to the repo root from elsewhere).
"""
import numpy as np

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.basis_variations_format as bvf
import t3toolbox.manifold as t3m


# --------------------------------------------------------------------------------------------------
# Problem configuration (tweak these)
# --------------------------------------------------------------------------------------------------
SHAPE        = (16, 16, 16, 16)   # the Hilbert tensor's shape (order d = len(SHAPE))
N_TRAIN      = 800                 # number of training applies
N_VAL        = 400                 # number of held-out validation applies
NOISE_LEVEL  = 0.01               # measurement noise, as a fraction of the measurement RMS
RANK_LEVELS  = (1, 2, 3, 4, 5, 6)  # rank-continuation schedule (level r below)
SEED         = 0

# Optimizer knobs
MAX_NEWTON    = 30
GTOL_REL      = 1e-8              # stop a fit when ||grad|| <= GTOL_REL * ||grad_0||
CG_MAXITER    = 200


# --------------------------------------------------------------------------------------------------
# The target tensor and the (dense) ground-truth measurement operator
# --------------------------------------------------------------------------------------------------
def hilbert_tensor(shape):
    """Dense Hilbert tensor A[i0,...] = 1 / (1 + i0 + ... + i_{d-1})."""
    grids = np.meshgrid(*[np.arange(N) for N in shape], indexing="ij")
    return 1.0 / (1.0 + sum(grids))


def dense_apply(A, ww):
    """Ground-truth applies of the dense tensor A against a W=(M,)-stack of probe-vector tuples.

    ww[m] has shape (M, N_m); returns shape (M,).  This is the same multilinear contraction that
    ``TuckerTensorTrain.apply`` computes structurally -- used here only to generate the data.
    """
    res = np.einsum("i...,si->s...", A, ww[0])          # contract mode 0 for every sample
    for m in range(1, len(ww)):
        res = np.einsum("sj...,sj->s...", res, ww[m])   # contract the next free mode
    return res                                          # shape (M,)


def unit_probes(M, shape, rng):
    """M probe-vector tuples (a W=(M,) stack), each vector scaled to unit norm (see the docstring:
    this normalizes every measurement's rank-1 row to unit Frobenius norm)."""
    ww = [rng.standard_normal((M, N)) for N in shape]
    return [w / np.linalg.norm(w, axis=1, keepdims=True) for w in ww]


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


# --------------------------------------------------------------------------------------------------
# The measurement operator (apply).  Everything below is operator-agnostic: the optimizer only uses
# `forward`, `transpose`, and `meas_dot`.  Swap these three for entries / probe to change the demo.
# --------------------------------------------------------------------------------------------------
def apply_operator(ww):
    forward   = lambda Z: Z.apply(ww)   # works for a TuckerTensorTrain (point) or a T3Tangent (Jacobian)
    transpose = lambda r, base: t3m.T3Tangent.apply_transpose(r, ww, base, sum_over_probes=True)
    meas_dot  = lambda a, b: float(np.dot(np.asarray(a), np.asarray(b)))
    return forward, transpose, meas_dot


# --------------------------------------------------------------------------------------------------
# Riemannian inexact Newton-CG with Armijo line search
# --------------------------------------------------------------------------------------------------
def _tangent_cg(H, rhs, base, tol, maxiter):
    """Solve H x = rhs in the tangent space at ``base`` (all iterates share the one T3Basis object).

    H is symmetric positive-semidefinite (Gauss-Newton), so plain CG is safe; we stop early on the
    forcing-term tolerance ``tol`` (the "inexact" Newton solve)."""
    x = t3m.T3Tangent.zeros(base)
    res = rhs                                  # residual = rhs - H(0) = rhs
    p = res
    rs = float(res.inner(res))
    if np.sqrt(rs) <= tol:
        return x, 0
    i = 0
    for i in range(1, maxiter + 1):
        Hp = H(p)
        pHp = float(p.inner(Hp))
        if pHp <= 1e-30:                       # nonpositive curvature guard (GN is PSD; rarely hit)
            if i == 1:
                x = rhs                         # fall back to the gradient direction
            break
        alpha = rs / pHp
        x = x + alpha * p
        res = res - alpha * Hp
        rs_new = float(res.inner(res))
        if np.sqrt(rs_new) <= tol:
            break
        p = res + (rs_new / rs) * p
        rs = rs_new
    return x, i


def riemannian_newton_cg(X0, forward, transpose, meas_dot, b,
                         max_newton=MAX_NEWTON, gtol_rel=GTOL_REL, cg_maxiter=CG_MAXITER,
                         c_armijo=1e-4, verbose=False):
    """Fit X to data b by minimizing 1/2 ||forward(X) - b||^2 on the fixed-rank manifold."""
    X = X0
    g0norm = None
    newton_iters = 0
    cg_total = 0
    for it in range(max_newton):
        base, _ = bvf.t3_orthogonal_representations(X)   # orthogonal frame at the current point

        r = forward(X) - b
        f = 0.5 * meas_dot(r, r)

        # Riemannian gradient: matrix-free transpose, then gauge-project onto the tangent space.
        g = transpose(r, base).orthogonal_gauge_projection()
        gnorm = float(g.norm())
        if g0norm is None:
            g0norm = gnorm if gnorm > 0.0 else 1.0
        if verbose:
            print(f"      newton {it:2d}: f={f:.6e}  |grad|={gnorm:.3e}")
        if gnorm <= gtol_rel * g0norm:
            break
        newton_iters += 1

        # Gauss-Newton Hessian-vector product (symmetric PSD on the gauged tangent space).
        def H(V):
            return transpose(forward(V), base).orthogonal_gauge_projection()

        # Inexact Newton: solve H p = -g by CG to a forcing-term tolerance (tighter as we converge).
        eta = min(0.5, np.sqrt(gnorm / g0norm))
        p, cg_iters = _tangent_cg(H, -g, base, tol=eta * gnorm, maxiter=cg_maxiter)
        cg_total += cg_iters

        slope = float(g.inner(p))
        if (not np.isfinite(slope)) or slope >= 0.0:     # ensure a descent direction
            p, slope = -g, -gnorm * gnorm

        # Armijo backtracking along the retraction curve alpha -> retract(alpha * p).
        alpha = 1.0
        X_trial = X
        for _ in range(40):
            X_trial = (alpha * p).retract()
            r_trial = forward(X_trial) - b
            f_trial = 0.5 * meas_dot(r_trial, r_trial)
            if f_trial <= f + c_armijo * alpha * slope:
                break
            alpha *= 0.5
        X = X_trial

    return X, dict(newton=newton_iters, cg=cg_total)


# --------------------------------------------------------------------------------------------------
# Rank continuation helpers
# --------------------------------------------------------------------------------------------------
def level_ranks(level, shape):
    """A simple feasible (minimal) rank schedule for continuation level r = ``level``.

    Tucker ranks all r (capped by the mode size); TT bond ranks (1, r, ..., r, 1)."""
    d = len(shape)
    tucker_ranks = tuple(min(level, N) for N in shape)
    tt_ranks = (1,) + (level,) * (d - 1) + (1,)
    return tucker_ranks, tt_ranks


def oracle_relerr(A, tucker_ranks, tt_ranks):
    """Relative error of the best rank-(tucker,tt) approximation of A (dense T3-SVD truncation)."""
    Xr, _, _ = t3.TuckerTensorTrain.t3svd_dense(A, max_tucker_ranks=tucker_ranks, max_tt_ranks=tt_ranks)
    return float(np.linalg.norm(Xr.to_dense() - A)) / float(np.linalg.norm(A))


# --------------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------------
def main():
    np.random.seed(SEED)               # defensive; the fit itself is deterministic (zero start)
    rng = np.random.default_rng(SEED)  # our own draws: the probe vectors and the measurement noise
    d = len(SHAPE)

    print(__doc__.split("\n\n")[0])
    print(f"\nHilbert tensor: shape {SHAPE}  ({np.prod(SHAPE):,} entries),  order d={d}")

    # ---- target + data ----------------------------------------------------------------------------
    A = hilbert_tensor(SHAPE)
    A_norm = float(np.linalg.norm(A))

    M = N_TRAIN + N_VAL
    ww_all = unit_probes(M, SHAPE, rng)                    # unit-norm probes (normalized rows!)
    b_clean = dense_apply(A, ww_all)
    b_rms = rms(b_clean)
    b_all = b_clean + NOISE_LEVEL * b_rms * rng.standard_normal(M)

    ww_train = [w[:N_TRAIN] for w in ww_all]
    ww_val = [w[N_TRAIN:] for w in ww_all]
    b_train, b_val = b_all[:N_TRAIN], b_all[N_TRAIN:]

    print(f"Measurements (applies): {N_TRAIN} train + {N_VAL} validation,  "
          f"{NOISE_LEVEL * 100:.0f}% noise.  (probe rows normalized to unit norm)\n")

    fwd_train, T_train, mdot = apply_operator(ww_train)
    fwd_val, _, _ = apply_operator(ww_val)

    # ---- rank-continuation sweep ------------------------------------------------------------------
    header = (f"{'level':>5} {'tucker / tt ranks':>24} {'DOF':>5} {'iters':>5} "
              f"{'train':>9} {'val':>9} {'true':>9} {'oracle':>9}")
    print(header)
    print("-" * len(header))

    # Start from the zero tensor; grow the rank by zero-padding (resize) -- no restarts, no nudges.
    X = t3.TuckerTensorTrain.zeros(SHAPE, *level_ranks(RANK_LEVELS[0], SHAPE))
    records = []
    for r in RANK_LEVELS:
        tucker_ranks, tt_ranks = level_ranks(r, SHAPE)
        dof = t3m.manifold_dim((SHAPE, tucker_ranks, tt_ranks))

        X0 = X.resize(SHAPE, tucker_ranks, tt_ranks)   # zero-pad the previous solution to this level
        X, stats = riemannian_newton_cg(X0, fwd_train, T_train, mdot, b_train)

        train_e = rms(fwd_train(X) - b_train) / b_rms
        val_e = rms(fwd_val(X) - b_val) / b_rms
        true_e = float(np.linalg.norm(X.to_dense() - A)) / A_norm
        oracle_e = oracle_relerr(A, tucker_ranks, tt_ranks)
        records.append(dict(level=r, dof=dof, val=val_e, true=true_e))

        rank_str = f"{tucker_ranks} {tt_ranks}"
        print(f"{r:>5} {rank_str:>24} {dof:>5} {stats['newton']:>5} "
              f"{train_e:>9.3e} {val_e:>9.3e} {true_e:>9.3e} {oracle_e:>9.3e}")

    # ---- model selection by validation ------------------------------------------------------------
    best = min(records, key=lambda rec: rec["val"])
    print("-" * len(header))
    print(f"\nNoise floor (relative): {NOISE_LEVEL:.1e}")
    print(f"Best ranks by validation error: level {best['level']}  "
          f"(val {best['val']:.3e}, true error {best['true']:.3e}, DOF {best['dof']}).")
    if best["level"] != RANK_LEVELS[-1]:
        print("Beyond this level the validation error rises while the training error keeps "
              "dropping -- the extra rank is fitting noise (overfitting) -- so rank continuation "
              "stops here.")


if __name__ == "__main__":
    main()
