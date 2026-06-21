"""Fit a Tucker tensor train to APPLY-DERIVATIVE data of a Hilbert tensor -- via the LIBRARY optimizers.

The library counterpart of ``examples/fit_hilbert_from_apply_derivatives.py``: the **same problem** (fit
the Hilbert tensor from symmetric apply-derivative jets by Manifold Cauchy SGD, with rank continuation),
but the optimizer is the library's ``t3toolbox.optimizers.mc_sgd`` over the ``apply_derivatives`` sampling
kind, instead of an inline loop. Three things the library carries that the reference inlines:

  * **The apply-derivatives operator** -- the sampling-kind string ``'apply_derivatives'`` bundles the
    forward jet Jacobian ``J``, its transpose ``Jᵀ``, and the per-order squared-norm reduction. No inline
    forward/transpose closures.
  * **The per-order normalization** -- a *residual weight* ``ω = 1/s_vec`` (``s_vec`` = the order-``t``
    RMS), passed as ``weight=ω``. The objective is ``½‖ω ⊙ (S(x) − y)‖²``, so the orders are balanced
    *inside* the kind and the data stays **raw** (no pre-normalization of ``y``).
  * **The minibatch** -- a custom ``draw`` that slices base points ``X`` (mirroring the reference: a few
    ``X`` with all their directions ``P``). One line; the flat default (a random subset across the whole
    sample stack) would be ``draw=None``.

Everything else -- the ``ω``-weighted objective, the Cauchy step, the manifold retraction, the absolute-
iteration stopping window -- lives in ``mc_sgd``. The rank continuation + validation stay here (a fitting
*facade* that picks geometry/optimizer/ranks is deferred). Cross-check: this recovers the tensor to the
same ~1% noise floor as the inline reference.

Run from the repo root:  ``python examples/fit_hilbert_from_apply_derivatives_topt.py``
"""
import time

import numpy as np

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.manifold as t3m
import t3toolbox.optimizers as topt
import t3toolbox.backend.probe_derivatives as pd   # only for the dense ground-truth derivative oracle


# --------------------------------------------------------------------------------------------------
# Problem configuration (matches the reference example)
# --------------------------------------------------------------------------------------------------
SHAPE        = (12, 12, 12, 12)    # Hilbert tensor shape (order d = len(SHAPE))
ORDER        = 4                   # highest derivative order K
N_X_TRAIN    = 10                  # training base points X
N_X_VAL      = 5                   # held-out validation base points X
N_P          = 30                  # perturbation directions P sampled per base point X
NOISE_LEVEL  = 0.01                # measurement noise, fraction of the per-order RMS
RANK_LEVELS  = (1, 2, 3, 4, 5)     # rank-continuation schedule
SEED         = 0

N_X_BATCH     = min(N_X_TRAIN, max(2, N_X_TRAIN // 10))   # base points per minibatch (>=2 at this toy scale)
MCSGD_MAXITER = 3000                                      # hard cap (the stopping rule normally stops sooner)


# --------------------------------------------------------------------------------------------------
# Target tensor + the dense ground-truth derivative measurements
# --------------------------------------------------------------------------------------------------
def hilbert_tensor(shape):
    """Dense Hilbert tensor A[i0,...] = 1 / (1 + i0 + ... + i_{d-1})."""
    grids = np.meshgrid(*[np.arange(N) for N in shape], indexing="ij")
    return 1.0 / (1.0 + sum(grids))


def unit_pairs_many_p(n_x, n_p, shape, rng):
    """``n_x`` unit-norm base points ``X``, each with ``n_p`` unit-norm directions ``P``; ``X`` replicated
    over the ``n_p`` axis so it shares ``P``'s ``(n_p, n_x)`` sample stack. Returns ``ww, pp`` (lists of
    ``d`` arrays of shape ``(n_p, n_x, N)``)."""
    def unit(*lead):
        vs = [rng.standard_normal(lead + (N,)) for N in shape]
        return [v / np.linalg.norm(v, axis=-1, keepdims=True) for v in vs]
    xx = unit(n_x)                                              # X: stack (n_x,)
    pp = unit(n_p, n_x)                                         # P: stack (n_p, n_x)
    ww = [np.broadcast_to(x[None], (n_p, n_x, N)).copy()        # X replicated -> stack (n_p, n_x)
          for x, N in zip(xx, shape)]
    return ww, pp


def dense_apply_derivatives(A, ww, pp, order):
    """Ground-truth apply-derivative jets of dense ``A``: shape ``(order+1,) + W`` (``W`` the sample stack)."""
    W = ww[0].shape[:-1]
    b = np.zeros((order + 1,) + W)
    for idx in np.ndindex(*W):
        b[(slice(None),) + idx] = pd.apply_derivatives_dense([w[idx] for w in ww], [p[idx] for p in pp], A, order)
    return b


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


def level_ranks(level, shape):
    d = len(shape)
    return tuple(min(level, N) for N in shape), (1,) + (level,) * (d - 1) + (1,)


def oracle_relerr(A, tucker_ranks, tt_ranks):
    Xr, _, _ = t3.TuckerTensorTrain.t3svd_dense(A, max_tucker_ranks=tucker_ranks, max_tt_ranks=tt_ranks)
    return float(np.linalg.norm(Xr.to_dense() - A)) / float(np.linalg.norm(A))


# --------------------------------------------------------------------------------------------------
# A custom minibatch draw: slice base points X (keep all their directions P). The library calls it with
# the optimizer's rng each step and feeds (sample_B, data_B) to the (jit-able) step. `data` carries a
# leading order axis (axis 0), so the base-point slice is over axis 2 of (order+1, N_P, N_X).
# (The flat default -- a random subset across the whole (N_P, N_X) stack -- is just draw=None.)
# --------------------------------------------------------------------------------------------------
def x_minibatch_draw(ww, pp, data, n_x_batch):
    n_x = ww[0].shape[1]                                        # base points (axis 1 of the (N_P, N_X) stack)

    def draw(rng):
        idx = rng.choice(n_x, size=min(n_x_batch, n_x), replace=False)
        return ([w[:, idx] for w in ww], [p[:, idx] for p in pp]), data[:, :, idx]
    return draw


# --------------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------------
def main():
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)          # data: probe directions + measurement noise
    rng_opt = np.random.default_rng(SEED + 1)  # the optimizer's minibatch draws (reproducible)
    d = len(SHAPE)

    print(__doc__.split("\n\n")[0])
    print(f"\nHilbert tensor: shape {SHAPE}  ({np.prod(SHAPE):,} entries),  order d={d}")
    print(f"Measurements: apply-derivatives orders 0..{ORDER}  "
          f"({N_X_TRAIN} train + {N_X_VAL} val base pts, {N_P} directions each, {NOISE_LEVEL*100:.0f}% noise).")
    print(f"Fit by library topt.mc_sgd over the 'apply_derivatives' kind (ω = 1/s_vec; X-slice draw of "
          f"{N_X_BATCH} base pt(s)/step).\n")

    A = hilbert_tensor(SHAPE)
    A_norm = float(np.linalg.norm(A))

    ww_tr, pp_tr = unit_pairs_many_p(N_X_TRAIN, N_P, SHAPE, rng)
    ww_va, pp_va = unit_pairs_many_p(N_X_VAL, N_P, SHAPE, rng)
    b_tr_clean = dense_apply_derivatives(A, ww_tr, pp_tr, ORDER)   # (order+1, N_P, N_X_TRAIN)
    b_va_clean = dense_apply_derivatives(A, ww_va, pp_va, ORDER)

    # per-order RMS scale (clean training data) -> the residual weight ω = 1/s_vec balances the orders
    s_vec = np.array([max(rms(b_tr_clean[t]), 1e-12) for t in range(ORDER + 1)])
    omega = 1.0 / s_vec
    sv_tr = s_vec.reshape((ORDER + 1,) + (1,) * (b_tr_clean.ndim - 1))
    sv_va = s_vec.reshape((ORDER + 1,) + (1,) * (b_va_clean.ndim - 1))

    # RAW noisy data (the kind applies ω internally; no pre-normalization)
    b_tr = b_tr_clean + NOISE_LEVEL * sv_tr * rng.standard_normal(b_tr_clean.shape)
    b_va = b_va_clean + NOISE_LEVEL * sv_va * rng.standard_normal(b_va_clean.shape)
    om_tr = omega.reshape((ORDER + 1,) + (1,) * (b_tr.ndim - 1))
    om_va = omega.reshape((ORDER + 1,) + (1,) * (b_va.ndim - 1))

    def weighted_relerr(X, ww, pp, b, om):     # relative error in the ω-normalized measurement space
        S = np.asarray(X.apply_derivatives(ww, pp, ORDER))
        return rms(om * (S - b)) / rms(om * b)

    header = (f"{'level':>5} {'tucker / tt ranks':>24} {'DOF':>5} {'iters':>5} "
              f"{'train':>9} {'val':>9} {'true':>9} {'oracle':>9}")
    print(header)
    print("-" * len(header))

    X = t3.TuckerTensorTrain.zeros(SHAPE, *level_ranks(RANK_LEVELS[0], SHAPE))
    records = []
    t_start = time.perf_counter()
    for level in RANK_LEVELS:
        tucker_ranks, tt_ranks = level_ranks(level, SHAPE)
        dof = t3m.manifold_dim((SHAPE, tucker_ranks, tt_ranks))
        X0 = X.resize(SHAPE, tucker_ranks, tt_ranks)             # warm start: zero-pad to the new ranks
        draw = x_minibatch_draw(ww_tr, pp_tr, b_tr, N_X_BATCH)
        X, stats = topt.mc_sgd(
            t3m.MANIFOLD, 'apply_derivatives', (ww_tr, pp_tr), b_tr, X0,
            rng_opt, N_X_BATCH * N_P,            # batch -- overridden by the custom draw (= its measurement count)
            order=ORDER, weight=omega, draw=draw, max_iter=MCSGD_MAXITER,
            # Stopping window, tuned for this toy scale: check the full-batch loss once per "epoch"
            # (N_X / N_X_BATCH iters) over a 3-check plateau. The library defaults (check_every=25,
            # plateau_lag=4) give a ~100-iteration window -- conservative for larger problems, but it
            # over-runs this tiny one by ~3x.
            check_every=max(1, N_X_TRAIN // N_X_BATCH), plateau_lag=3, smooth_tau=1.0)

        train_e = weighted_relerr(X, ww_tr, pp_tr, b_tr, om_tr)
        val_e = weighted_relerr(X, ww_va, pp_va, b_va, om_va)
        true_e = float(np.linalg.norm(X.to_dense() - A)) / A_norm
        oracle_e = oracle_relerr(A, tucker_ranks, tt_ranks)
        records.append(dict(level=level, val=val_e, true=true_e, dof=dof))
        print(f"{level:>5} {f'{tucker_ranks} {tt_ranks}':>24} {dof:>5} {stats['n_iter']:>5} "
              f"{train_e:>9.3e} {val_e:>9.3e} {true_e:>9.3e} {oracle_e:>9.3e}")

    best = min(records, key=lambda rec: rec["val"])
    print("-" * len(header))
    print(f"\nNoise floor (relative): {NOISE_LEVEL:.1e}    (total fit time {time.perf_counter()-t_start:.1f}s)")
    print(f"Best ranks by validation error: level {best['level']}  "
          f"(val {best['val']:.3e}, true error {best['true']:.3e}, DOF {best['dof']}).")


if __name__ == "__main__":
    main()
