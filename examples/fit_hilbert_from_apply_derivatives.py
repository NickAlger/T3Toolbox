"""Fit a Tucker tensor train to APPLY-DERIVATIVE data of a Hilbert tensor.

A worked example of Riemannian fixed-rank fitting from **symmetric directional derivatives** of the
all-modes ``apply`` operation -- the derivative analogue of ``examples/fit_hilbert_tensor_newton_cg.py``
(which fits from ordinary applies). Here we fit with **Manifold Cauchy SGD (MC-SGD)**, the stochastic
method of the T4S paper (Section 5.3.2): a minibatched, tuning-free Riemannian optimizer. The forward
``T3Tangent.apply_derivatives`` (the Jacobian ``J``) and its transpose ``apply_derivatives_transpose``
(``J^T``) drive it, so this is also the end-to-end integration test of the derivative pipeline.

The problem
-----------
The order-``d`` **Hilbert tensor** ``A[i0,...] = 1 / (1 + i0 + ... + i_{d-1})`` is smooth, positive, and
well approximated by low rank. We never see ``A`` directly.

    *Sampling: a few base points, many directions each.*  We sample ``N_X`` random base points
    ``X = (x0,...,x_{d-1})`` and, **for each ``X``**, ``N_P`` random perturbation directions
    ``P = (p0,...,p_{d-1})``.  So the perturbation stack is ``(N_P, N_X)`` and the frame stack starts as
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

The method -- Manifold Cauchy SGD (MC-SGD)
------------------------------------------
A first-order stochastic Riemannian method (T4S Section 5.3.2). Each iteration draws a fresh **minibatch
of base points** ``X`` (a few ``X``, with *all* their directions ``P`` -- base points are the scarce
resource, directions the cheap one), so the per-step work is a fraction of the full batch. On that
minibatch it forms the gauged stochastic gradient ``g = Pi J^T r`` (the forward ``Z.apply_derivatives``
works on a ``TuckerTensorTrain`` point or a ``T3Tangent`` direction; the transpose with
``sum_over_probes=True`` is ``J^T r``; ``Pi = MANIFOLD.project`` gauges it), then sets the step length by
the **Cauchy rule** ``alpha = ||g||^2 / ||J g||^2`` -- the exact 1D minimizer of the local Gauss-Newton
quadratic model along ``-g``. That costs one extra forward Jacobian-vector product ``J g`` and needs **no
learning-rate schedule** (the "Cauchy" in MC-SGD; "manifold" because we retract). The step is
``X <- retract(-alpha * g)``, and a fixed-rank stage stops when an exponentially-smoothed minibatch loss
stops decreasing. (``g`` and ``||J g||^2`` are exactly ``fitting.GaussNewtonModel.gradient`` /
``.gn_quadratic``; this example inlines them against the apply-derivative closures.)

Rank continuation starts from the zero tensor and grows rank by zero-padding (warm start); validation
picks the rank. Contrast ``fit_hilbert_tensor_newton_cg.py``, which uses deterministic full-batch
Newton-CG -- MC-SGD trades some final accuracy for cheap, tuning-free steps that scale to many samples.

Run from the repo root:  ``python examples/fit_hilbert_from_apply_derivatives.py``
"""
import time

import numpy as np

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.frame_variations_format as bvf
import t3toolbox.manifold as t3m
import t3toolbox.backend.sampling_derivatives as pd


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

# MC-SGD (Manifold Cauchy SGD) knobs -- the Cauchy step is tuning-free, so there is no learning rate.
# Minibatch = a few base points per step (with ALL their directions P). The paper's rule of thumb is ~10%
# of the samples, but with only N_X=10 base points that rounds to a single one, whose gradient is too
# noisy (it makes the stopping test trigger-happy); 2 base points smooths it enough that the paper's
# default stopping works, while staying a small minibatch. (3+ over-shrinks the epoch and destabilizes.)
N_X_BATCH    = min(N_X_TRAIN, max(2, N_X_TRAIN // 10))
MCSGD_MAXITER = 3000               # hard cap (the smoothed-loss criterion normally stops much sooner)
MCSGD_C_TAU  = 1.0                 # loss-smoothing timescale, in epochs (T4S 5.3.2 default)
MCSGD_C_T    = 3.0                 # plateau-detection lag, in epochs (T4S 5.3.2 default)


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
    sample stack). Each ``W``-element is ``dense_apply_derivatives`` (the exact subset expansion)."""
    W = ww[0].shape[:-1]
    b = np.zeros((order + 1,) + W)
    for idx in np.ndindex(*W):
        ww_m = [w[idx] for w in ww]
        pp_m = [p[idx] for p in pp]
        b[(slice(None),) + idx] = pd.dense_apply_derivatives(ww_m, pp_m, A, order)
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
    transpose = lambda r, frame: t3m.T3Tangent.apply_derivatives_transpose(
        np.asarray(r) / sv, ww, pp, frame, order, sum_over_probes=True)
    meas_dot = lambda a, b: float(np.sum(np.asarray(a) * np.asarray(b)))
    return forward, transpose, meas_dot


# --------------------------------------------------------------------------------------------------
# Manifold Cauchy SGD (MC-SGD), t4s.pdf Section 5.3.2
# --------------------------------------------------------------------------------------------------
# A tuning-free stochastic Riemannian method. Each iteration:
#   * draw a fresh minibatch of base points X (a few X, with ALL their directions P -- the cheap,
#     plentiful resource), so the per-step work is a fraction of the full batch;
#   * form the gauged stochastic gradient  g = Pi Jᵀr  on that minibatch (transpose, then the gauge
#     projection Pi = MANIFOLD.project) -- a T3Tangent at the current frame;
#   * set the step length by the *Cauchy rule*  alpha = ‖g‖² / ‖J g‖²  -- the 1D minimizer of the local
#     Gauss-Newton quadratic model along -g. It costs one extra forward Jacobian-vector product J g and
#     needs NO learning-rate schedule (this is the "Cauchy" in MC-SGD; "manifold" because we retract).
#   * step and retract:  X <- retract(-alpha * g).
# We stop a fixed-rank stage when an exponentially-smoothed FULL-batch loss stops decreasing (checked
# once per epoch -- the single-minibatch loss is too noisy a stop signal; see below).
# (The same g and ‖J g‖² are exactly what fitting.GaussNewtonModel.gradient / .gn_quadratic compute;
# we inline them here against the apply-derivative closures, which have no GaussNewtonModel yet.)
def manifold_cauchy_sgd(X0, ww, pp, b, order, s_vec, rng,
                        n_x_batch=N_X_BATCH, max_iter=MCSGD_MAXITER,
                        c_tau=MCSGD_C_TAU, c_t=MCSGD_C_T):
    n_x = ww[0].shape[1]                                  # base points (axis 1 of the (N_P, N_X) stack)
    iters_per_epoch = max(1, int(round(n_x / n_x_batch)))             # = n_s / |B|  (the N_P axis cancels)

    # The stopping signal is the FULL-batch loss, checked once per epoch: a single base point's minibatch
    # loss has too much base-point-to-base-point variance to detect the plateau (it false-triggers early).
    # One full forward (no transpose) per epoch is cheap. We exponentially smooth it (one sample/epoch) and
    # stop when it stops decreasing -- the paper's tuning-free criterion (T4S 5.3.2), on the clean signal.
    fwd_full, _, _ = apply_derivative_operator(ww, pp, order, s_vec)
    full_loss = lambda Z: 0.5 * float(np.mean((np.asarray(fwd_full(Z)) - b) ** 2))
    a_smooth = 1.0 - np.exp(-1.0 / c_tau)                 # EMA weight, timescale C_TAU epochs
    lag = max(1, int(round(c_t)))                         # plateau-detection lag, C_T epochs
    s_hist = []

    X = X0
    n_iter = 0
    for k in range(max_iter):
        n_iter = k + 1
        idx = rng.choice(n_x, size=min(n_x_batch, n_x), replace=False)   # fresh minibatch of base pts
        ww_B = [w[:, idx, :] for w in ww]                 # X-subset: stack (N_P, n_x_batch)
        pp_B = [p[:, idx, :] for p in pp]
        b_B = b[:, :, idx]
        fwd_B, T_B, mdot_B = apply_derivative_operator(ww_B, pp_B, order, s_vec)

        frame, _ = bvf.t3_orthogonal_representations(X)
        r_B = fwd_B(X) - b_B
        g = t3m.MANIFOLD.project(T_B(r_B, frame))          # gauged stochastic gradient  g = Pi Jᵀr
        gg = float(g.corewise_inner(g))                   # ‖g‖²  (HS, since g is gauged at an orth frame)
        if gg <= 1e-30:                                   # converged on this minibatch
            break
        Jg = fwd_B(g)                                     # one forward Jacobian-vector product  J g
        denom = float(mdot_B(Jg, Jg))                     # ‖J g‖² = gᵀ(JᵀJ)g  (the GN curvature along g)
        alpha = gg / max(denom, 1e-12 * gg)               # Cauchy step length (guarded against ‖Jg‖≈0)
        X = t3m.MANIFOLD.retract((-alpha) * g)

        if n_iter % iters_per_epoch == 0:                 # once-per-epoch full-batch stopping check
            L = full_loss(X)
            s = L if not s_hist else a_smooth * L + (1.0 - a_smooth) * s_hist[-1]
            s_hist.append(s)
            if len(s_hist) > lag and (s_hist[-1] - s_hist[-1 - lag]) > 0.0:
                break
    return X, dict(iters=n_iter)


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
    rng = np.random.default_rng(SEED)          # data: probe directions + measurement noise
    rng_opt = np.random.default_rng(SEED + 1)  # the optimizer's own minibatch draws (reproducible)
    d = len(SHAPE)

    print(__doc__.split("\n\n")[0])
    print(f"\nHilbert tensor: shape {SHAPE}  ({np.prod(SHAPE):,} entries),  order d={d}")
    print(f"Measurements: apply-derivatives orders 0..{ORDER}  "
          f"({N_X_TRAIN} train base pts + {N_X_VAL} val base pts, {N_P} directions each, "
          f"{NOISE_LEVEL*100:.0f}% noise).")
    print(f"Fit by Manifold Cauchy SGD: minibatch {N_X_BATCH} frame pt(s) (all {N_P} directions) per step, "
          f"tuning-free Cauchy step.\n")

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

    fwd_tr, _, _ = apply_derivative_operator(ww_tr, pp_tr, ORDER, s_vec)   # full-batch forwards: errors
    fwd_va, _, _ = apply_derivative_operator(ww_va, pp_va, ORDER, s_vec)   # only (MC-SGD minibatches itself)
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
    t_start = time.perf_counter()
    for r in RANK_LEVELS:
        tucker_ranks, tt_ranks = level_ranks(r, SHAPE)
        dof = t3m.manifold_dim((SHAPE, tucker_ranks, tt_ranks))
        X0 = X.resize(SHAPE, tucker_ranks, tt_ranks)              # warm start: zero-pad to the new ranks
        X, stats = manifold_cauchy_sgd(X0, ww_tr, pp_tr, b_tr_n, ORDER, s_vec, rng_opt)

        train_e = rms(fwd_tr(X) - b_tr_n) / b_tr_rms
        val_e = rms(fwd_va(X) - b_va_n) / b_va_rms
        true_e = float(np.linalg.norm(X.to_dense() - A)) / A_norm
        oracle_e = oracle_relerr(A, tucker_ranks, tt_ranks)
        records.append(dict(level=r, val=val_e, true=true_e, dof=dof))
        rank_str = f"{tucker_ranks} {tt_ranks}"
        print(f"{r:>5} {rank_str:>24} {dof:>5} {stats['iters']:>5} "
              f"{train_e:>9.3e} {val_e:>9.3e} {true_e:>9.3e} {oracle_e:>9.3e}")

    best = min(records, key=lambda rec: rec["val"])
    print("-" * len(header))
    print(f"\nNoise floor (relative): {NOISE_LEVEL:.1e}    (total fit time {time.perf_counter()-t_start:.1f}s)")
    print(f"Best ranks by validation error: level {best['level']}  "
          f"(val {best['val']:.3e}, true error {best['true']:.3e}, DOF {best['dof']}).")


if __name__ == "__main__":
    main()
