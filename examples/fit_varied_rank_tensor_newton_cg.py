"""Fit a Tucker tensor train to "apply" probes of a tensor with **deliberately varied ranks**, using
Riemannian Newton-CG with the **condition-number rank continuation of Section 5.4.1** (t4s.pdf).

What this example adds over ``fit_hilbert_tensor_newton_cg.py``
--------------------------------------------------------------
That example fits a *Hilbert* tensor, whose matrix unfoldings all have a similar singular-value decay,
so **uniform** rank continuation (grow every Tucker rank and TT bond together: ``(r,...,r)``) is
near-optimal. Many tensors are not like that -- different unfoldings carry very different amounts of
information, and uniform continuation then either **over-fits the easy edges** (wasting degrees of
freedom, hence data) or **under-fits the hard ones** (never reaching the rank they need). This example
shows the alternative: choose *which* ranks to grow from the data, via the singular values of the
current iterate's unfoldings.

The target
----------
An order-4 cosine series with a **different number of frequencies per mode**::

    A[i,j,k,l] = sum_{p=0}^{P-1} rho^p * cos((p % n0) pi x_i) cos((p % n1) pi x_j)
                                       * cos((p % n2) pi x_k) cos((p % n3) pi x_l),

with ``(n0,n1,n2,n3) = (8,2,5,3)`` and a geometric coefficient decay ``rho^p``. Mode ``m`` cycles
through ``n_m`` distinct cosine frequencies, so its Tucker rank is ``n_m`` -- giving **Tucker ranks
(8,2,5,3)** -- while the cumulative coupling makes the **TT bonds (1,8,8,3,1)**. The decay ``rho`` gives
a genuine spectral tail (so held-out error has a clear minimum). The headline numbers (printed below):
the heterogeneous true ranks represent ``A`` essentially exactly with **~370 degrees of freedom**,
whereas the smallest *uniform* ranks that do as well (``r = 8`` everywhere) need **~1216** -- 3.3x more.

The method: condition-number rank continuation (Section 5.4.1)
-------------------------------------------------------------
After each fixed-rank fit we call :py:meth:`TuckerTensorTrain.continuation_ranks`, which:

  * takes the T3-SVD singular values of the converged iterate's unfoldings,
  * forms each edge's **condition number** ``kappa_i = sigma_1 / sigma_{rank_i}`` (large = the rank
    there is already "used up"; small = that edge is well conditioned and has room to grow),
  * grows only the **well-conditioned** edges -- those a factor ``tau`` below the worst edge -- so the
    ranks trend toward comparable conditioning (with a uniform-bump fallback to get started, an
    absolute conditioning guard, and "useless-rank removal" so the new ranks stay non-degenerate).

We then **warm-start** the next fit by zero-padding the converged cores to the new ranks
(:py:meth:`~TuckerTensorTrain.resize`), exactly as in the uniform example. We **terminate** when the
next model would become underdetermined (data divided by manifold dimension falls below ``TAU_DATA``),
and **select** the rank level with the smallest held-out (validation) error.

``tau``: this smooth target's conditioning spread across edges is only moderate (~5-8x at the levels
that matter), so we differentiate edges with ``tau = 3``; the paper's default ``tau = 10`` is more
conservative and suited to the wider spreads of random tensors. (Try ``TAU = 10`` below: continuation
then grows uniformly and matches the uniform sweep.)

We run **both** strategies on the same data and compare. The adaptive sweep discovers the heterogeneous
ranks -- freezing the rank-2 mode early while growing the rank-8 mode -- and reaches the noise floor
with fewer degrees of freedom than uniform continuation can afford within the data budget.

Run from the repo root:  ``python examples/fit_varied_rank_tensor_newton_cg.py``
(or with ``PYTHONPATH`` set to the repo root from elsewhere).
"""
import numpy as np

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.manifold as t3m
import t3toolbox.fitting as fitting


# --------------------------------------------------------------------------------------------------
# Problem configuration (tweak these)
# --------------------------------------------------------------------------------------------------
SHAPE        = (16, 16, 16, 16)   # the target's shape (order d = len(SHAPE))
NFREQ        = (8, 2, 5, 3)       # distinct cosine frequencies per mode -> Tucker ranks (8,2,5,3)
N_TERMS      = 24                 # number of cosine product terms (P)
RHO          = 0.85              # geometric coefficient decay rho^p (the spectral tail)
N_TRAIN      = 1800               # number of training applies
N_VAL        = 700                # number of held-out validation applies
NOISE_LEVEL  = 0.01               # measurement noise, as a fraction of the measurement RMS
SEED         = 0

# Rank-continuation knobs (Section 5.4.1)
TAU          = 3.0                # grow an edge only if its condition number is < kappa_max / TAU
N_CHUNK      = 1                  # rank increment per grown edge
TAU_DATA     = 2.0               # stop before data / manifold-dim drops below this (overdetermination)
MAX_LEVELS   = 25                 # safety cap on continuation rounds

# Optimizer knobs
MAX_NEWTON    = 30
GTOL_REL      = 1e-8              # stop a fit when ||grad|| <= GTOL_REL * ||grad_0||
CG_MAXITER    = 200


# --------------------------------------------------------------------------------------------------
# The target tensor and the (dense) ground-truth measurement operator
# --------------------------------------------------------------------------------------------------
def cosine_tensor(shape, nfreq, n_terms, rho):
    """Order-d cosine series with ``nfreq[m]`` distinct frequencies in mode m (-> Tucker rank nfreq[m]).

    A[i0,...] = sum_p rho^p * prod_m cos((p % nfreq[m]) * pi * x_m),  x_m = (arange(N_m)+0.5)/N_m.
    """
    xs = [(np.arange(N) + 0.5) / N for N in shape]
    A = np.zeros(shape)
    for p in range(n_terms):
        factors = [np.cos((p % nf) * np.pi * x) for nf, x in zip(nfreq, xs)]
        A = A + (rho ** p) * np.einsum("i,j,k,l->ijkl", *factors)
    return A


def dense_apply(A, ww):
    """Ground-truth applies of the dense tensor A against a W=(M,)-stack of probe-vector tuples.

    ww[m] has shape (M, N_m); returns shape (M,) -- the same multilinear contraction
    ``TuckerTensorTrain.apply`` computes structurally, used here only to generate the data.
    """
    res = np.einsum("i...,si->s...", A, ww[0])          # contract mode 0 for every sample
    for m in range(1, len(ww)):
        res = np.einsum("sj...,sj->s...", res, ww[m])   # contract the next free mode
    return res                                          # shape (M,)


def unit_probes(M, shape, rng):
    """M probe-vector tuples (a W=(M,) stack), each vector scaled to unit norm (normalizes every
    measurement's rank-1 row to unit Frobenius norm -- see ``fit_hilbert_tensor_newton_cg.py``)."""
    ww = [rng.standard_normal((M, N)) for N in shape]
    return [w / np.linalg.norm(w, axis=1, keepdims=True) for w in ww]


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


# --------------------------------------------------------------------------------------------------
# The measurement operator (apply).  Operator-agnostic: the optimizer only uses `forward`,
# `model_builder`, and `meas_dot`.  Swap these for entries / probe to change the demo.
# --------------------------------------------------------------------------------------------------
def apply_operator(ww):
    forward       = lambda Z: Z.apply(ww)   # works for a TuckerTensorTrain (point) or a T3Tangent (Jacobian)
    model_builder = lambda X, r: fitting.apply_model(t3m.MANIFOLD, X, ww, r)
    meas_dot      = lambda a, b: float(np.dot(np.asarray(a), np.asarray(b)))
    return forward, model_builder, meas_dot


# --------------------------------------------------------------------------------------------------
# Riemannian inexact Newton-CG with Armijo line search (identical to fit_hilbert_tensor_newton_cg.py)
# --------------------------------------------------------------------------------------------------
def _tangent_cg(H, rhs, base, tol, maxiter):
    """Solve H x = rhs in the tangent space at ``base`` (symmetric PSD Gauss-Newton, inexact stop)."""
    x = t3m.T3Tangent.zeros(base)
    res = rhs                                  # residual = rhs - H(0) = rhs
    p = res
    rs = float(res.corewise_inner(res))
    if np.sqrt(rs) <= tol:
        return x, 0
    i = 0
    for i in range(1, maxiter + 1):
        Hp = H(p)
        pHp = float(p.corewise_inner(Hp))
        if pHp <= 1e-30:                       # nonpositive curvature guard (GN is PSD; rarely hit)
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


def riemannian_newton_cg(X0, forward, model_builder, meas_dot, b,
                         max_newton=MAX_NEWTON, gtol_rel=GTOL_REL, cg_maxiter=CG_MAXITER,
                         c_armijo=1e-4):
    """Fit X to data b by minimizing 1/2 ||forward(X) - b||^2 on the fixed-rank manifold."""
    X = X0
    g0norm = None
    newton_iters = 0
    for it in range(max_newton):
        r = forward(X) - b
        model = model_builder(X, r)            # frame + base sweep, precomputed once and reused below
        base = model.base
        f = float(model.objective_value)       # = 1/2 ||r||^2

        g = model.gradient                     # Riemannian gradient Pi J^T r (already gauged)
        gnorm = float(g.corewise_norm())
        if g0norm is None:
            g0norm = gnorm if gnorm > 0.0 else 1.0
        if gnorm <= gtol_rel * g0norm:
            break
        newton_iters += 1

        H = model.gn_hessian                   # GN Hessian-vector product (symmetric PSD; sweep reused)
        eta = min(0.5, np.sqrt(gnorm / g0norm))                       # inexact-Newton forcing term
        p, _ = _tangent_cg(H, -g, base, tol=eta * gnorm, maxiter=cg_maxiter)

        slope = float(g.corewise_inner(p))
        if (not np.isfinite(slope)) or slope >= 0.0:
            p, slope = -g, -gnorm * gnorm

        alpha = 1.0                            # Armijo backtracking along the retraction curve
        X_trial = X
        for _ in range(40):
            X_trial = t3m.MANIFOLD.retract(alpha * p)
            r_trial = forward(X_trial) - b
            f_trial = 0.5 * meas_dot(r_trial, r_trial)
            if f_trial <= f + c_armijo * alpha * slope:
                break
            alpha *= 0.5
        X = X_trial

    return X, dict(newton=newton_iters)


# --------------------------------------------------------------------------------------------------
# Rank schedules / diagnostics
# --------------------------------------------------------------------------------------------------
def level_ranks(level, shape):
    """The simple *uniform* (minimal) rank schedule: Tucker ranks all ``level`` (capped by mode size),
    TT bonds ``(1, level, ..., level, 1)`` -- the baseline this example compares against."""
    d = len(shape)
    tucker_ranks = tuple(min(level, N) for N in shape)
    tt_ranks = (1,) + (level,) * (d - 1) + (1,)
    return tucker_ranks, tt_ranks


def oracle_relerr(A, tucker_ranks, tt_ranks):
    """Relative error of the best rank-(tucker,tt) approximation of A (dense T3-SVD truncation)."""
    Xr, _, _ = t3.TuckerTensorTrain.t3svd_dense(A, max_tucker_ranks=tucker_ranks, max_tt_ranks=tt_ranks)
    return float(np.linalg.norm(Xr.to_dense() - A)) / float(np.linalg.norm(A))


def _errors(X, A, A_norm, fwd_train, b_train, b_rms, fwd_val, b_val):
    return dict(
        train  = rms(fwd_train(X) - b_train) / b_rms,
        val    = rms(fwd_val(X) - b_val) / b_rms,
        true   = float(np.linalg.norm(X.to_dense() - A)) / A_norm,
    )


# --------------------------------------------------------------------------------------------------
# The two rank-continuation strategies
# --------------------------------------------------------------------------------------------------
def adaptive_continuation(shape, fwd_train, builder_train, mdot, b_train, fwd_val, b_val, A, A_norm,
                          b_rms, header):
    """Section 5.4.1 continuation: grow the well-conditioned edges, chosen from each iterate's spectra."""
    print("\nADAPTIVE rank continuation (Section 5.4.1, condition-number based, TAU = %g)" % TAU)
    print(header)
    print("-" * len(header))

    X = t3.TuckerTensorTrain.zeros(shape, *level_ranks(1, shape))   # start from the zero tensor, rank 1
    records = []
    for level in range(MAX_LEVELS):
        X, stats = riemannian_newton_cg(X, fwd_train, builder_train, mdot, b_train)
        tucker_ranks, tt_ranks = X.tucker_ranks, X.tt_ranks
        dof = t3m.manifold_dim((shape, tucker_ranks, tt_ranks))
        e = _errors(X, A, A_norm, fwd_train, b_train, b_rms, fwd_val, b_val)
        oracle_e = oracle_relerr(A, tucker_ranks, tt_ranks)
        records.append(dict(level=level, tucker=tucker_ranks, tt=tt_ranks, dof=dof, **e))
        print(_row(level, tucker_ranks, tt_ranks, dof, stats['newton'], e, oracle_e))

        new_tucker, new_tt = X.continuation_ranks(tau=TAU, n_chunk=N_CHUNK)
        if (new_tucker, new_tt) == (tucker_ranks, tt_ranks):
            print("  (stop: continuation_ranks returned unchanged ranks -- conditioning guard / maximal)")
            break
        next_dof = t3m.manifold_dim((shape, new_tucker, new_tt))
        if N_TRAIN / next_dof < TAU_DATA:
            print(f"  (stop: next ranks {new_tucker} {new_tt} -> DOF {next_dof}, "
                  f"data/DOF {N_TRAIN / next_dof:.2f} < TAU_DATA {TAU_DATA})")
            break
        X = X.resize(shape, new_tucker, new_tt)   # zero-padded warm start at the grown ranks
    return records


def uniform_continuation(shape, fwd_train, builder_train, mdot, b_train, fwd_val, b_val, A, A_norm,
                         b_rms, header):
    """The baseline: grow every rank together, ``(r,...,r)``, until the data budget runs out."""
    print("\nUNIFORM rank continuation (the baseline: grow all ranks together)")
    print(header)
    print("-" * len(header))

    X = t3.TuckerTensorTrain.zeros(shape, *level_ranks(1, shape))
    records = []
    for level in range(1, max(shape) + 1):
        tucker_ranks, tt_ranks = level_ranks(level, shape)
        dof = t3m.manifold_dim((shape, tucker_ranks, tt_ranks))
        if N_TRAIN / dof < TAU_DATA:
            print(f"  (stop: ranks {tucker_ranks} {tt_ranks} -> DOF {dof}, "
                  f"data/DOF {N_TRAIN / dof:.2f} < TAU_DATA {TAU_DATA})")
            break
        X, stats = riemannian_newton_cg(X.resize(shape, tucker_ranks, tt_ranks),
                                        fwd_train, builder_train, mdot, b_train)
        e = _errors(X, A, A_norm, fwd_train, b_train, b_rms, fwd_val, b_val)
        oracle_e = oracle_relerr(A, tucker_ranks, tt_ranks)
        records.append(dict(level=level, tucker=tucker_ranks, tt=tt_ranks, dof=dof, **e))
        print(_row(level, tucker_ranks, tt_ranks, dof, stats['newton'], e, oracle_e))
    return records


def _header():
    return (f"{'lvl':>3} {'tucker / tt ranks':>26} {'DOF':>5} {'its':>4} "
            f"{'train':>9} {'val':>9} {'true':>9} {'oracle':>9}")


def _row(level, tucker_ranks, tt_ranks, dof, its, e, oracle_e):
    rank_str = f"{tucker_ranks} {tt_ranks}"
    return (f"{level:>3} {rank_str:>26} {dof:>5} {its:>4} "
            f"{e['train']:>9.3e} {e['val']:>9.3e} {e['true']:>9.3e} {oracle_e:>9.3e}")


# --------------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(SEED)
    d = len(SHAPE)

    print(__doc__.split("\n\n")[0])

    # ---- target + its rank structure --------------------------------------------------------------
    A = cosine_tensor(SHAPE, NFREQ, N_TERMS, RHO)
    A_norm = float(np.linalg.norm(A))
    X_exact, _, _ = t3.TuckerTensorTrain.t3svd_dense(A / A_norm, rtol=1e-6)
    true_tucker, true_tt = X_exact.tucker_ranks, X_exact.tt_ranks
    dof_true = t3m.manifold_dim((SHAPE, true_tucker, true_tt))
    uni = max(max(true_tucker), max(true_tt))            # smallest uniform level that contains the target
    dof_uni = t3m.manifold_dim((SHAPE, *level_ranks(uni, SHAPE)))
    print(f"\nVaried-rank cosine tensor: shape {SHAPE}  ({np.prod(SHAPE):,} entries),  order d={d}")
    print(f"  true ranks (heterogeneous): tucker {true_tucker}, tt {true_tt}   -> {dof_true} DOF")
    print(f"  smallest uniform ranks that contain it: r={uni} everywhere       -> {dof_uni} DOF "
          f"({dof_uni / dof_true:.1f}x more)")

    # ---- data -------------------------------------------------------------------------------------
    M = N_TRAIN + N_VAL
    ww_all = unit_probes(M, SHAPE, rng)
    b_clean = dense_apply(A, ww_all)
    b_rms = rms(b_clean)
    b_all = b_clean + NOISE_LEVEL * b_rms * rng.standard_normal(M)
    ww_train = [w[:N_TRAIN] for w in ww_all]
    ww_val = [w[N_TRAIN:] for w in ww_all]
    b_train, b_val = b_all[:N_TRAIN], b_all[N_TRAIN:]
    print(f"\nMeasurements (applies): {N_TRAIN} train + {N_VAL} validation,  "
          f"{NOISE_LEVEL * 100:.0f}% noise.  (probe rows normalized to unit norm)")

    fwd_train, builder_train, mdot = apply_operator(ww_train)
    fwd_val, _, _ = apply_operator(ww_val)

    header = _header()
    args = (SHAPE, fwd_train, builder_train, mdot, b_train, fwd_val, b_val, A, A_norm, b_rms, header)
    rec_adaptive = adaptive_continuation(*args)
    rec_uniform = uniform_continuation(*args)

    # ---- model selection by validation, and the comparison ----------------------------------------
    best_a = min(rec_adaptive, key=lambda r: r["val"])
    best_u = min(rec_uniform, key=lambda r: r["val"])
    print("\n" + "=" * len(header))
    print(f"Noise floor (relative): {NOISE_LEVEL:.1e}")
    print(f"BEST adaptive: tucker {best_a['tucker']}, tt {best_a['tt']}  "
          f"DOF {best_a['dof']}  val {best_a['val']:.3e}  true {best_a['true']:.3e}")
    print(f"BEST uniform : tucker {best_u['tucker']}, tt {best_u['tt']}  "
          f"DOF {best_u['dof']}  val {best_u['val']:.3e}  true {best_u['true']:.3e}")
    print(f"\nAdaptive continuation reaches {best_a['true']:.1e} true error with {best_a['dof']} DOF; "
          f"uniform is limited to {best_u['true']:.1e} with {best_u['dof']} DOF within the same data "
          f"budget.\nThe condition-number scheme spent its degrees of freedom on the edges that needed "
          f"them (growing\nthe rank-{max(best_a['tucker'])} mode while freezing the rank-"
          f"{min(best_a['tucker'])} one), which uniform continuation cannot do.")


if __name__ == "__main__":
    main()
