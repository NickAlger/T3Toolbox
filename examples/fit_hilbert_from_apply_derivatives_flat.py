"""Fit a Tucker tensor train to APPLY-DERIVATIVE data -- MC-SGD with FLAT ``(X, P)`` minibatches.

A variant of ``examples/fit_hilbert_from_apply_derivatives.py``. **Same problem, same optimizer
(Manifold Cauchy SGD, T4S Section 5.3.2); the only change is how minibatches are drawn.** The
reference example minibatches over **base points ``X``**: a few ``X``, each carrying *all* its
directions ``P``. Here we instead view the data as a **flat list of ``(X, P)`` pairs** and draw a
minibatch of those pairs **totally at random**, mixing freely across base points.

Why try this
------------
The sample stack is ``W = (N_P, N_X)``. The X-only scheme picks a subset of the ``N_X`` axis, so a
minibatch only ever sees a handful of distinct base points (its gradient is dominated by, and
correlated within, those few ``X``). Flattening to ``W = (N_P * N_X,)`` and sampling pairs at random
spreads each minibatch across (likely) *all* the base points, each contributing a random subset of its
directions. **Hypothesis:** this extra mixing decorrelates the per-sample contributions and lowers the
variance of the stochastic gradient -- which, for a Cauchy-step method, should mean steadier step
lengths and a more robust stopping signal. Untested; this example exists to see whether it holds here.

Nothing else changes. The flatten is a pure reshape of the *same* collected data (``N_X`` distinct base
points, ``N_P`` directions each), and the apply-derivative forward / transpose treat ``W`` as arbitrary
leading batch axes -- so ``apply_derivative_operator`` is reused verbatim and ``sum_over_probes=True``
still sums the whole minibatch into one ``J^T r``. To isolate the mixing effect we **match the per-step
sample count** to the reference (its 2 base points x ``N_P`` directions = 60 pairs), so the epoch length
(``round(n_pairs / |B|)``) comes out identical and the *only* difference is the minibatch composition.

The problem (unchanged from the reference example)
--------------------------------------------------
The order-``d`` **Hilbert tensor** ``A[i0,...] = 1 / (1 + i0 + ... + i_{d-1})`` is smooth and low-rank-
friendly; we never see ``A``, only **apply-derivative jets** along lines ``X + sP``:

    b = [ d^t/ds^t  A(x0 + s p0, ..., x_{d-1} + s p_{d-1}) |_{s=0} ]_{t=0..K}.

We sample ``N_X`` base points, ``N_P`` directions each, normalize per order (the essential conditioning
step), add noise, and fit with rank continuation (increase-by-1, zero-padded warm starts, validation-
picked rank). See ``fit_hilbert_from_apply_derivatives.py`` for the full annotation of the problem and
the Cauchy step; this file documents only the batching difference.

Run from the repo root:  ``python examples/fit_hilbert_from_apply_derivatives_flat.py``
"""
import time

import numpy as np

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.frame_variations_format as bvf
import t3toolbox.manifold as t3m
import t3toolbox.backend.probe_derivatives as pd


# --------------------------------------------------------------------------------------------------
# Problem configuration (tweak these) -- identical to the reference example
# --------------------------------------------------------------------------------------------------
SHAPE        = (12, 12, 12, 12)    # Hilbert tensor shape (order d = len(SHAPE))
ORDER        = 4                   # highest derivative order K (apply has degree d; orders 0..d carry it)
N_X_TRAIN    = 10                  # training base points X
N_X_VAL      = 5                   # held-out validation base points X
N_P          = 30                  # perturbation directions P sampled per base point X
NOISE_LEVEL  = 0.01                # measurement noise, fraction of the per-order RMS
RANK_LEVELS  = (1, 2, 3, 4, 5)     # rank-continuation schedule
SEED         = 0

# MC-SGD knobs. The minibatch is now a flat set of (X, P) PAIRS drawn uniformly at random from the
# N_P*N_X training pairs (vs the reference's "a few base points, all their directions"). We size it to
# match the reference's per-step sample count -- 2 base points worth of directions -- so the epoch length
# is identical and the only difference is the mixing. (Shrink BATCH_PAIRS to test smaller minibatches.)
N_X_BATCH_EQ  = 2                          # the reference example's base-point batch (for matching only)
BATCH_PAIRS   = N_X_BATCH_EQ * N_P         # = 60 random (X,P) pairs/step, 20% of the 300 training pairs
MCSGD_MAXITER = 3000                       # hard cap (the smoothed-loss criterion normally stops sooner)
MCSGD_C_TAU   = 1.0                        # loss-smoothing timescale, in epochs (T4S 5.3.2 default)
MCSGD_C_T     = 3.0                        # plateau-detection lag, in epochs (T4S 5.3.2 default)


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


def flatten_pairs(ww, pp, b):
    """Collapse the ``(n_p, n_x)`` sample stack to one flat ``(n_p*n_x,)`` axis of ``(X, P)`` pairs.

    A pure reshape -- the same samples, regrouped so a minibatch can mix across base points. ``ww``/``pp``
    go ``(n_p, n_x, N) -> (n_p*n_x, N)``; ``b`` goes ``(order+1, n_p, n_x) -> (order+1, n_p*n_x)``."""
    n_p, n_x = ww[0].shape[:2]
    ww_f = [w.reshape(n_p * n_x, w.shape[-1]) for w in ww]
    pp_f = [p.reshape(n_p * n_x, p.shape[-1]) for p in pp]
    b_f = b.reshape(b.shape[0], n_p * n_x)
    return ww_f, pp_f, b_f


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
# The measurement operator (per-order-normalized apply-derivatives) -- reused verbatim from the
# reference example; it treats the sample stack W as arbitrary leading axes, so a flat (X,P) stack
# "just works". The optimizer below uses only `forward`, `transpose`, `meas_dot`.
# --------------------------------------------------------------------------------------------------
def apply_derivative_operator(ww, pp, order, s_vec):
    """forward / transpose / meas_dot for the per-order-normalized apply-derivative map.

    Dividing the output (and the data) by ``s_vec`` (per order, broadcast over the sample stack ``W``)
    folds the per-order normalization into the operator, so ``transpose(r)`` applies ``J^T`` to
    ``r / s_vec``. ``sum_over_probes=True`` sums the whole ``W`` stack -> one tangent."""
    nW = ww[0].ndim - 1
    sv = s_vec.reshape((order + 1,) + (1,) * nW)               # broadcast over the W sample stack
    forward = lambda Z: np.asarray(Z.apply_derivatives(ww, pp, order)) / sv
    transpose = lambda r, base: t3m.T3Tangent.apply_derivatives_transpose(
        np.asarray(r) / sv, ww, pp, base, order, sum_over_probes=True)
    meas_dot = lambda a, b: float(np.sum(np.asarray(a) * np.asarray(b)))
    return forward, transpose, meas_dot


# --------------------------------------------------------------------------------------------------
# Manifold Cauchy SGD (MC-SGD), t4s.pdf Section 5.3.2 -- FLAT (X,P)-pair minibatching
# --------------------------------------------------------------------------------------------------
# Same tuning-free Cauchy-step method as the reference example; the minibatch is the difference. Each
# iteration:
#   * draw a fresh minibatch of `batch` (X, P) PAIRS uniformly at random from the n_pairs = N_P*N_X
#     flat training pairs -- mixed across base points (vs the reference's few-X-all-their-P scheme);
#   * form the gauged stochastic gradient  g = Pi Jᵀr  on that minibatch (transpose with
#     sum_over_probes=True sums the pairs, then Pi = MANIFOLD.project gauges) -- a T3Tangent;
#   * Cauchy step length  alpha = ‖g‖² / ‖J g‖²  (1D GN-quadratic minimizer along -g; one extra J·v);
#   * step and retract:  X <- retract(-alpha * g).
# Stop a fixed-rank stage when an exponentially-smoothed FULL-batch loss stops decreasing (checked once
# per epoch; the single-minibatch loss is too noisy a stop signal).
def manifold_cauchy_sgd(X0, ww, pp, b, order, s_vec, rng,
                        batch=BATCH_PAIRS, max_iter=MCSGD_MAXITER,
                        c_tau=MCSGD_C_TAU, c_t=MCSGD_C_T):
    n_pairs = ww[0].shape[0]                              # flat (X,P) pairs (axis 0 of the flattened stack)
    iters_per_epoch = max(1, int(round(n_pairs / batch)))            # = n_pairs / |B|

    # Stopping signal: the deterministic FULL-batch loss over all pairs, checked once per epoch and
    # exponentially smoothed; stop when it stops decreasing (T4S 5.3.2's tuning-free criterion).
    fwd_full, _, _ = apply_derivative_operator(ww, pp, order, s_vec)
    full_loss = lambda Z: 0.5 * float(np.mean((np.asarray(fwd_full(Z)) - b) ** 2))
    a_smooth = 1.0 - np.exp(-1.0 / c_tau)                 # EMA weight, timescale C_TAU epochs
    lag = max(1, int(round(c_t)))                         # plateau-detection lag, C_T epochs
    s_hist = []

    X = X0
    n_iter = 0
    for k in range(max_iter):
        n_iter = k + 1
        idx = rng.choice(n_pairs, size=min(batch, n_pairs), replace=False)   # fresh minibatch of (X,P) pairs
        ww_B = [w[idx, :] for w in ww]                    # pair-subset: stack (batch,), mixed across base pts
        pp_B = [p[idx, :] for p in pp]
        b_B = b[:, idx]
        fwd_B, T_B, mdot_B = apply_derivative_operator(ww_B, pp_B, order, s_vec)

        base, _ = bvf.t3_orthogonal_representations(X)
        r_B = fwd_B(X) - b_B
        g = t3m.MANIFOLD.project(T_B(r_B, base))          # gauged stochastic gradient  g = Pi Jᵀr
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
    print(f"Fit by Manifold Cauchy SGD: minibatch {BATCH_PAIRS} random (X,P) pairs per step "
          f"(of {N_P*N_X_TRAIN} training pairs), tuning-free Cauchy step.\n")

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

    b_tr_n = b_tr / sv_tr                                          # normalized data
    b_va_n = b_va / sv_va

    # Flatten the (N_P, N_X) sample stack to one (N_P*N_X,) axis of (X,P) pairs -- the only structural
    # change from the reference example. Everything downstream sees a 1D sample stack.
    ww_tr, pp_tr, b_tr_n = flatten_pairs(ww_tr, pp_tr, b_tr_n)     # (N_P*N_X_TRAIN, N), (order+1, *)
    ww_va, pp_va, b_va_n = flatten_pairs(ww_va, pp_va, b_va_n)

    fwd_tr, _, _ = apply_derivative_operator(ww_tr, pp_tr, ORDER, s_vec)   # full-batch forwards: errors
    fwd_va, _, _ = apply_derivative_operator(ww_va, pp_va, ORDER, s_vec)   # only (MC-SGD minibatches itself)
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
