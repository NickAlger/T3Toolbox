"""Fit a UNIFORM Tucker tensor train to PROBE-DERIVATIVE jets of a Hilbert tensor, with Newton-CG.

The showcase example for the uniform layer: it combines **three** features that each of the other
examples shows one of --

  * **derivative (jet) sampling** -- we fit from the symmetric directional derivatives of the vector-valued
    ``probe`` operation (the jet analogue of ``examples/fit_hilbert_uniform_newton_cg.py``, which fits
    ordinary applies), driving the full ``probe_derivatives`` Jacobian ``J`` / transpose ``Jᵀ``;
  * the **uniform layer** -- the fit runs on ``UniformTuckerTensorTrain`` supercores (the ``lax.scan``
    speed path), via the batteries-included ``optimizers.newton_cg`` with ``UNIFORM_MANIFOLD``;
  * **rank continuation** -- start at rank 1 and grow, picking the rank by held-out validation.

If you are new to the uniform layer, read ``examples/fit_hilbert_uniform_newton_cg.py`` first (the plain
apply fit) -- this example changes only the sampling operation (``'apply'`` -> ``'probe_derivatives'``,
plus the jet data + a per-order weight); the uniform pipeline and the continuation loop are identical.

The problem
-----------
The order-``d`` Hilbert tensor ``A[i0,...] = 1 / (1 + i0 + ... + i_{d-1})`` is smooth and low-rank-friendly.
We observe **probe-derivative jets**: for random frame probe vectors ``X = (x0,...,x_{d-1})`` and directions
``P = (p0,...,p_{d-1})``, each measurement is, for every free mode ``i``,

    b_i = [ dᵗ/dsᵗ  probe_i(A; x + s p) |_{s=0} ]_{t=0..K},

a length-``K+1`` jet per output coordinate (order 0 is the ordinary probe). A ``probe`` leaves one mode
free and contracts the other ``d-1``, so the restriction to the line ``X + sP`` has degree ``d-1``; orders
``0 .. d-1`` carry the whole line and higher orders vanish. We therefore fit orders ``0 .. d-1``.

    *Two normalizations* (both essential to conditioning): (1) unit-norm probe / direction vectors, so no
    sample dominates; (2) a **per-order weight** ``ω_t = 1 / RMS_t`` (RMS over the training data of order
    ``t``), passed straight to the fitting layer -- it weights the objective ``½‖ω ⊙ r‖²`` so the
    wildly-different-magnitude orders contribute comparably. (The ragged
    ``examples/fit_hilbert_from_apply_derivatives.py`` folds this into a custom operator; here the optimizer
    takes ``weight=ω`` directly.)

The method + rank continuation
------------------------------
Riemannian inexact Newton-CG on the fixed-rank manifold, via ``optimizers.newton_cg(UNIFORM_MANIFOLD,
'probe_derivatives', (ww, pp), data, ux0, order=K, weight=ω)``. Continuation keeps the cheap ``resize`` /
zero-pad bookkeeping in the **ragged** layer and drops into the **uniform** layer only for the fit (see
``fit_hilbert_uniform_newton_cg.py`` for why the zero-padded start is structurally minimal, so
``uniform_minimal`` leaves it alone and the gradient grows the new rank block). We track the validation
misfit and pick the rank that minimizes it.

Run from the repo root:  ``python examples/fit_hilbert_uniform_probe_derivatives_newton_cg.py``
"""
import numpy as np

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.manifold as t3m                     # for manifold_dim (the tangent-space DOF)
import t3toolbox.uniform_manifold as ut3m            # UNIFORM_MANIFOLD
import t3toolbox.optimizers as optimizers
import t3toolbox.backend.sampling_derivatives as pd     # dense ground-truth jets (data generation only)


# --------------------------------------------------------------------------------------------------
# Problem configuration (tweak these)
# --------------------------------------------------------------------------------------------------
SHAPE        = (10, 10, 10)        # Hilbert tensor shape (order d = len(SHAPE))
N_TRAIN      = 300                 # number of training (frame, direction) samples
N_VAL        = 150                 # held-out validation samples
NOISE_LEVEL  = 0.01                # measurement noise, fraction of the per-order RMS
RANK_LEVELS  = (1, 2, 3, 4)        # rank-continuation schedule
SEED         = 0
MAX_NEWTON   = 25
# probe leaves one mode free + contracts d-1, so the line X+sP has degree d-1: fit orders 0..d-1.
ORDER        = len(SHAPE) - 1


# --------------------------------------------------------------------------------------------------
# Target + dense ground-truth probe-derivative jets
# --------------------------------------------------------------------------------------------------
def hilbert_tensor(shape):
    grids = np.meshgrid(*[np.arange(N) for N in shape], indexing="ij")
    return 1.0 / (1.0 + sum(grids))


def unit_vectors(M, shape, rng):
    """M vector-tuples, each vector scaled to unit norm."""
    vv = [rng.standard_normal((M, N)) for N in shape]
    return [v / np.linalg.norm(v, axis=1, keepdims=True) for v in vv]


def dense_probe_derivative_jets(A, ww, pp, order):
    """Ground-truth probe-derivative jets of the dense tensor A, as the fitting-layer ``data``.

    Loops the exact per-sample dense jet (``backend.sampling_derivatives.dense_probe_derivatives``) over the
    W=(M,) sample stack and stacks it: returns ``len=d``, each ``(order+1, M, N_i)`` -- the probe-derivative
    residual layout ``(order+1) + W + (N_i,)``. (Data generation only; the fit never forms A.)"""
    M, d = ww[0].shape[0], len(ww)
    per = [pd.dense_probe_derivatives([w[s] for w in ww], [p[s] for p in pp], A, order) for s in range(M)]
    return [np.stack([per[s][i] for s in range(M)], axis=1) for i in range(d)]


def order_rms(data, order):
    """Per-order RMS over the whole training data (each ``data[i]`` is ``(order+1, M, N_i)``)."""
    return np.array([max(float(np.sqrt(np.mean(
        np.concatenate([np.asarray(z[t]).ravel() for z in data]) ** 2))), 1e-12)
        for t in range(order + 1)])


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


def level_ranks(level, shape):
    d = len(shape)
    return tuple(min(level, N) for N in shape), (1,) + (level,) * (d - 1) + (1,)


def weighted_misfit(pred, data, omega):
    """Relative per-order-weighted misfit ``‖ω⊙(pred-data)‖ / ‖ω⊙data‖`` over the probe (d-list) jets."""
    def wnorm(seq):
        return np.sqrt(sum(float(np.sum((omega[:, None, None] * np.asarray(z)) ** 2)) for z in seq))
    return wnorm([p - d for p, d in zip(pred, data)]) / wnorm(data)


# --------------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(SEED)
    d = len(SHAPE)

    print(__doc__.split("\n\n")[0])
    print(f"\nHilbert tensor: shape {SHAPE},  order d={d},  fitting probe-derivative orders 0..{ORDER}")

    # ---- target + data ----------------------------------------------------------------------------
    A = hilbert_tensor(SHAPE)
    A_norm = float(np.linalg.norm(A))

    M = N_TRAIN + N_VAL
    ww = unit_vectors(M, SHAPE, rng)                       # frame probe vectors X
    pp = unit_vectors(M, SHAPE, rng)                       # perturbation directions P
    data_clean = dense_probe_derivative_jets(A, ww, pp, ORDER)   # len=d, each (order+1, M, N_i)

    def split(seq, sl):
        return [z[:, sl] for z in seq]
    ww_tr, ww_va = [w[:N_TRAIN] for w in ww], [w[N_TRAIN:] for w in ww]
    pp_tr, pp_va = [p[:N_TRAIN] for p in pp], [p[N_TRAIN:] for p in pp]
    data_tr_clean, data_va_clean = split(data_clean, slice(0, N_TRAIN)), split(data_clean, slice(N_TRAIN, M))

    # per-order weight ω = 1 / RMS_t (from the clean training data); add noise scaled per order.
    s_vec = order_rms(data_tr_clean, ORDER)
    omega = 1.0 / s_vec
    data_tr = [z + NOISE_LEVEL * s_vec[:, None, None] * rng.standard_normal(z.shape) for z in data_tr_clean]
    print(f"Samples: {N_TRAIN} train + {N_VAL} validation,  {NOISE_LEVEL*100:.0f}% noise.  "
          f"per-order RMS = {np.round(s_vec, 3)}\n")

    # ---- rank-continuation sweep on the UNIFORM layer ---------------------------------------------
    header = (f"{'level':>5} {'tucker / tt ranks':>22} {'DOF':>5} {'newton':>6} "
              f"{'train (wtd)':>11} {'val (wtd)':>11} {'true':>9}")
    print(header)
    print("-" * len(header))

    X = t3.TuckerTensorTrain.zeros(SHAPE, *level_ranks(RANK_LEVELS[0], SHAPE))   # ragged continuation iterate
    records = []
    for r in RANK_LEVELS:
        tucker_ranks, tt_ranks = level_ranks(r, SHAPE)
        dof = t3m.manifold_dim((SHAPE, tucker_ranks, tt_ranks))

        X0 = X.resize(SHAPE, tucker_ranks, tt_ranks)         # ragged: zero-pad the previous solution
        ux0 = ut3.UniformTuckerTensorTrain.from_t3(X0)       # -> uniform for the fit
        ux, stats = optimizers.newton_cg(
            ut3m.UNIFORM_MANIFOLD, "probe_derivatives", (ww_tr, pp_tr), data_tr, ux0,
            order=ORDER, weight=omega, max_newton=MAX_NEWTON)
        X = ux.to_t3()                                       # -> ragged for the next resize

        train_e = weighted_misfit([np.asarray(z) for z in X.probe_derivatives(ww_tr, pp_tr, ORDER)],
                                  data_tr, omega)
        val_e = weighted_misfit([np.asarray(z) for z in X.probe_derivatives(ww_va, pp_va, ORDER)],
                                data_va_clean, omega)
        true_e = float(np.linalg.norm(X.to_dense() - A)) / A_norm
        records.append(dict(level=r, dof=dof, val=val_e, true=true_e))

        print(f"{r:>5} {f'{tucker_ranks} {tt_ranks}':>22} {dof:>5} {stats['newton']:>6} "
              f"{train_e:>11.3e} {val_e:>11.3e} {true_e:>9.3e}")

    # ---- model selection by validation ------------------------------------------------------------
    best = min(records, key=lambda rec: rec["val"])
    print("-" * len(header))
    print(f"\nBest ranks by validation error: level {best['level']}  "
          f"(val {best['val']:.3e}, true error {best['true']:.3e}, DOF {best['dof']}).")
    print("The fit ran entirely on the uniform layer (UniformTuckerTensorTrain supercores + lax.scan-ready "
          "sweeps) from probe-DERIVATIVE data, growing the rank under continuation.")


if __name__ == "__main__":
    main()
