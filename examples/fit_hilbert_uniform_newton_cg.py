"""Fit a UNIFORM Tucker tensor train to sampled "apply" probes of a Hilbert tensor.

The **uniform-layer** companion to ``examples/fit_hilbert_tensor_newton_cg.py``. Same problem, same
method (Riemannian inexact Newton-CG with rank continuation), but the fit runs on a
``UniformTuckerTensorTrain`` -- padded supercores + boolean rank masks, laid out so the tensor-train
sweeps compile to ``jax.lax.scan`` over the mode axis (the GPU/jit speed path; see the ``docs/uniform_*``
notes). This is the minimal on-ramp: read it next to the ragged example and note that **the only changes
are the two marked lines** -- the start point is uniform and the geometry is ``UNIFORM_MANIFOLD``.

The uniform pipeline (what actually differs from ragged)
--------------------------------------------------------
Everything is driven by the batteries-included optimizer :py:func:`t3toolbox.optimizers.newton_cg`, which
accepts **either** representation and infers it from ``x0``:

    x_opt, stats = optimizers.newton_cg(t3m.MANIFOLD,        'apply', ww, b, x0_ragged)     # ragged
    x_opt, stats = optimizers.newton_cg(ut3m.UNIFORM_MANIFOLD, 'apply', ww, b, x0_uniform)  # uniform  <-- this file

The uniform run returns a ``UniformTuckerTensorTrain``, fully packed, and (under ``use_jit=True`` with jax
inputs) its per-step kernel **compiles once** across Newton steps -- the masks are held loop-invariant and
only the supercores are traced. The uniform optimizer also requires a **minimal-rank frame**; the frontend
calls ``uniform_minimal(x0)`` transparently, so we never have to think about it here.

Rank continuation, uniform-style
--------------------------------
Continuation is unchanged in spirit -- start at rank 1 from the zero tensor and grow the rank by
zero-padding the converged cores -- but note **where each representation lives**. The uniform layer pins
the ranks with fixed masks, so we keep the cheap continuation bookkeeping (``resize`` / zero-pad) in the
**ragged** layer and drop into the **uniform** layer only for the fit itself (the hot loop):

    X = X.resize(SHAPE, tucker_ranks, tt_ranks)          # ragged: zero-pad the previous solution
    ux0 = UniformTuckerTensorTrain.from_t3(X)            # -> uniform for the fit
    ux, stats = optimizers.newton_cg(UNIFORM_MANIFOLD, 'apply', ww, b, ux0)
    X = ux.to_t3()                                       # -> ragged for the next resize

A zero-padded start is numerically rank-deficient but **structurally** minimal (its nominal ranks are
realizable), so ``uniform_minimal`` leaves it alone and the first gradient step grows the padding into the
new rank block -- exactly as in the ragged example (the fitted ranks come out minimal at every level, and
the errors match the ragged run to numerical tolerance). We track the held-out **validation** misfit and
pick the rank level that minimizes it.

Run from the repo root:  ``python examples/fit_hilbert_uniform_newton_cg.py``
"""
import numpy as np

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.manifold as t3m                     # for manifold_dim (the tangent-space DOF)
import t3toolbox.uniform_manifold as ut3m            # UNIFORM_MANIFOLD  <-- the uniform geometry
import t3toolbox.optimizers as optimizers


# --------------------------------------------------------------------------------------------------
# Problem configuration (tweak these)
# --------------------------------------------------------------------------------------------------
SHAPE        = (16, 16, 16, 16)    # the Hilbert tensor's shape (order d = len(SHAPE))
N_TRAIN      = 800                 # number of training applies
N_VAL        = 400                 # number of held-out validation applies
NOISE_LEVEL  = 0.01                # measurement noise, as a fraction of the measurement RMS
RANK_LEVELS  = (1, 2, 3, 4, 5, 6)  # rank-continuation schedule (level r below)
SEED         = 0
MAX_NEWTON   = 30


# --------------------------------------------------------------------------------------------------
# The target tensor and the (dense) ground-truth measurement operator
# --------------------------------------------------------------------------------------------------
def hilbert_tensor(shape):
    """Dense Hilbert tensor A[i0,...] = 1 / (1 + i0 + ... + i_{d-1})."""
    grids = np.meshgrid(*[np.arange(N) for N in shape], indexing="ij")
    return 1.0 / (1.0 + sum(grids))


def dense_apply(A, ww):
    """Ground-truth applies of the dense tensor A against a W=(M,)-stack of probe-vector tuples.
    ww[m] has shape (M, N_m); returns shape (M,)."""
    res = np.einsum("i...,si->s...", A, ww[0])
    for m in range(1, len(ww)):
        res = np.einsum("sj...,sj->s...", res, ww[m])
    return res


def unit_probes(M, shape, rng):
    """M probe-vector tuples, each vector scaled to unit norm (normalizes every measurement's rank-1
    row to unit Frobenius norm -- what makes the least-squares well-conditioned)."""
    ww = [rng.standard_normal((M, N)) for N in shape]
    return [w / np.linalg.norm(w, axis=1, keepdims=True) for w in ww]


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


def level_ranks(level, shape):
    """A feasible (minimal) rank schedule for continuation level r: Tucker ranks all r (capped by the
    mode size), TT bond ranks (1, r, ..., r, 1)."""
    d = len(shape)
    return tuple(min(level, N) for N in shape), (1,) + (level,) * (d - 1) + (1,)


def oracle_relerr(A, tucker_ranks, tt_ranks):
    """Relative error of the best rank-(tucker,tt) approximation of A (dense T3-SVD truncation)."""
    Xr, _, _ = t3.TuckerTensorTrain.t3svd_dense(A, max_tucker_ranks=tucker_ranks, max_tt_ranks=tt_ranks)
    return float(np.linalg.norm(Xr.to_dense() - A)) / float(np.linalg.norm(A))


# --------------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(SEED)
    d = len(SHAPE)

    print(__doc__.split("\n\n")[0])
    print(f"\nHilbert tensor: shape {SHAPE}  ({np.prod(SHAPE):,} entries),  order d={d}")

    # ---- target + data ----------------------------------------------------------------------------
    A = hilbert_tensor(SHAPE)
    A_norm = float(np.linalg.norm(A))

    M = N_TRAIN + N_VAL
    ww_all = unit_probes(M, SHAPE, rng)
    b_clean = dense_apply(A, ww_all)
    b_rms = rms(b_clean)
    b_all = b_clean + NOISE_LEVEL * b_rms * rng.standard_normal(M)

    ww_train = [w[:N_TRAIN] for w in ww_all]
    ww_val = [w[N_TRAIN:] for w in ww_all]
    b_train, b_val = b_all[:N_TRAIN], b_all[N_TRAIN:]
    print(f"Measurements (applies): {N_TRAIN} train + {N_VAL} validation,  "
          f"{NOISE_LEVEL * 100:.0f}% noise.  (probe rows normalized to unit norm)\n")

    # ---- rank-continuation sweep on the UNIFORM layer ---------------------------------------------
    header = (f"{'level':>5} {'tucker / tt ranks':>26} {'DOF':>5} {'newton':>6} "
              f"{'train':>9} {'val':>9} {'true':>9} {'oracle':>9}")
    print(header)
    print("-" * len(header))

    # The continuation iterate X stays RAGGED (cheap zero-pad bookkeeping); the FIT runs on the uniform layer.
    X = t3.TuckerTensorTrain.zeros(SHAPE, *level_ranks(RANK_LEVELS[0], SHAPE))
    records = []
    for r in RANK_LEVELS:
        tucker_ranks, tt_ranks = level_ranks(r, SHAPE)
        dof = t3m.manifold_dim((SHAPE, tucker_ranks, tt_ranks))

        X0 = X.resize(SHAPE, tucker_ranks, tt_ranks)         # ragged: zero-pad the previous solution
        ux0 = ut3.UniformTuckerTensorTrain.from_t3(X0)       # ------> UNIFORM start point   (diff #1)
        ux, stats = optimizers.newton_cg(
            ut3m.UNIFORM_MANIFOLD, "apply", ww_train, b_train, ux0, max_newton=MAX_NEWTON)  # UNIFORM geometry (diff #2)
        X = ux.to_t3()                                       # ragged again, for the next resize

        train_e = rms(X.apply(ww_train) - b_train) / b_rms
        val_e = rms(X.apply(ww_val) - b_val) / b_rms
        true_e = float(np.linalg.norm(X.to_dense() - A)) / A_norm
        oracle_e = oracle_relerr(A, tucker_ranks, tt_ranks)
        records.append(dict(level=r, dof=dof, val=val_e, true=true_e))

        rank_str = f"{tucker_ranks} {tt_ranks}"
        print(f"{r:>5} {rank_str:>26} {dof:>5} {stats['newton']:>6} "
              f"{train_e:>9.3e} {val_e:>9.3e} {true_e:>9.3e} {oracle_e:>9.3e}")

    # ---- model selection by validation ------------------------------------------------------------
    best = min(records, key=lambda rec: rec["val"])
    print("-" * len(header))
    print(f"\nNoise floor (relative): {NOISE_LEVEL:.1e}")
    print(f"Best ranks by validation error: level {best['level']}  "
          f"(val {best['val']:.3e}, true error {best['true']:.3e}, DOF {best['dof']}).")
    print("The fitted tensor is a UniformTuckerTensorTrain; the true errors match the ragged "
          "fit_hilbert_tensor_newton_cg.py run to numerical tolerance -- the uniform layer is a faster "
          "representation of the same computation.")


if __name__ == "__main__":
    main()
