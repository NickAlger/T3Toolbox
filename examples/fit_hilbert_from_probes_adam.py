"""Fit a Tucker tensor train to PROBE measurements of a Hilbert tensor -- corewise + our own Adam.

A worked example of **corewise** fixed-rank fitting from **probes** -- the richest of the three sampling
ops (one mode stays free per measurement, so each probe returns ``d`` vectors rather than a scalar) --
optimized by a **hand-written Adam** (a few lines, no dependency; the library carries no optax/scipy).
It rounds out the example set: ``apply`` (manifold Newton-CG), ``apply``-derivatives (manifold MC-SGD),
``entries`` (corewise L-BFGS), and now ``probes`` (corewise Adam). Three things on display:

  1. **Probes** -- ``X.probe(ww)`` leaves mode ``i`` free and contracts the rest, returning, per probe,
     ``d`` vectors (one per mode). Dense random probe vectors make this a *global, well-conditioned*
     measurement (contrast ``entries``, whose localized one-hot rows make a coherent tensor ill-posed).
  2. **Our own Adam** -- the standard adaptive first-order method, written here as ~10 lines over the flat
     core vector: per-coordinate first/second moment EMAs with bias correction. It is elementwise, so the
     flat update *is* per-core Adam. Dependency-free, runs on the numpy and jax paths, no memory leak.
  3. **Corewise stochastic fitting** -- minibatch over probes, step the raw cores with Adam (the corewise
     gradient is the plain ``J^T r``, no gauge projection). Adam tolerates the gauge-singular corewise
     Hessian for free (the gauge directions are flat valleys it simply coasts along).

Adam vs the Cauchy step (MC-SGD): Adam is **not tuning-free** -- it has a learning rate (here a single
constant ``LR``), unlike MC-SGD's curvature-set Cauchy step. That is the price of a generic adaptive
method that needs no Gauss-Newton ``J v`` product. (One could set Adam's scale from ``model.gn_quadratic``;
we keep this a *standard* Adam and just note the option.)

The problem
-----------
The order-``d`` **Hilbert tensor** ``A[i0,...] = 1 / (1 + i0 + ... + i_{d-1})`` is smooth and low-rank-
friendly. We never see ``A``; we observe ``M`` probes (dense random probe vectors, each scaled to unit
norm so every probe's vectors contribute comparably), split into a training set and a held-out validation
set, with a little noise. Because probe vectors are dense and random, the measurement is global and the
fit is well-posed from far fewer samples than entry-completion needs.

The method -- corewise Adam
---------------------------
We minimize ``f(X) = 1/2 sum_probes sum_modes ||X.probe(ww) - y||^2`` over the cores. Each step draws a
minibatch of probes, builds a corewise ``fitting.probe_model`` (objective + corewise gradient ``J^T r``),
and takes an Adam step on the flat core vector, with a **cosine learning-rate decay** so the late steps
settle (a constant rate leaves the iterate jittering and gives a noisier rank table). Rank by validation,
cold random restart per level -- corewise cannot warm-start (see ``fit_hilbert_from_entries_lbfgs.py`` /
``docs/entries_completion_findings.md``: the zero tensor has a vanishing Jacobian, and zero-padded
continuation freezes the new rank block).

Because probes are *global, well-conditioned* measurements, this fit is well-posed and the misfit drops
**monotonically** with rank toward the noise floor -- it does **not** overfit in the rank range shown, so
validation keeps selecting the largest rank (the opposite of the localized ``entries`` fit, which is
ill-posed and overfits). Adam is **first-order**, so it reaches a few-% misfit quickly but is slow to high
accuracy; the second-order ``apply`` example (Newton-CG) tightens to ~1% but costs more per step.

Run from the repo root:  ``python examples/fit_hilbert_from_probes_adam.py``
"""
import time

import numpy as np

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.manifold as t3m
import t3toolbox.fitting as fitting


# --------------------------------------------------------------------------------------------------
# Problem configuration (tweak these)
# --------------------------------------------------------------------------------------------------
SHAPE        = (12, 12, 12, 12)      # the Hilbert tensor's shape (order d = len(SHAPE))
N_TRAIN      = 200                    # training probes (each yields d vectors -> d*N values)
N_VAL        = 100                    # held-out validation probes
NOISE_LEVEL  = 0.01                  # measurement noise, as a fraction of the probe-output RMS
RANK_LEVELS  = (1, 2, 3, 4, 5, 6)    # rank schedule (each level fit independently; see the docstring)
SEED         = 0

# Adam knobs (NOT tuning-free: LR is a real hyperparameter). Minibatch = a subset of the probes.
ADAM_LR      = 2e-2
ADAM_BETAS   = (0.9, 0.999)
ADAM_EPS     = 1e-8
ADAM_BATCH   = 32                     # probes per minibatch
ADAM_MAXITER = 1200


# --------------------------------------------------------------------------------------------------
# Target tensor + probe sampling
# --------------------------------------------------------------------------------------------------
def hilbert_tensor(shape):
    """Dense Hilbert tensor A[i0,...] = 1 / (1 + i0 + ... + i_{d-1})."""
    grids = np.meshgrid(*[np.arange(N) for N in shape], indexing="ij")
    return 1.0 / (1.0 + sum(grids))


def unit_probes(M, shape, rng):
    """``M`` probe-vector tuples (a ``W=(M,)`` stack), each vector scaled to unit norm so no probe's
    free-mode output dominates. Returns a list of ``d`` arrays, each shape ``(M, N_i)``."""
    ww = [rng.standard_normal((M, N)) for N in shape]
    return [w / np.linalg.norm(w, axis=1, keepdims=True) for w in ww]


def dense_probe(A, ww):
    """Ground-truth probes of dense ``A``: for each mode ``free``, contract every *other* mode with its
    probe vector, leaving ``free`` open. Returns ``d`` arrays of shape ``(M, N_free)`` (the structural
    ``TuckerTensorTrain.probe`` computes the same thing without forming ``A``)."""
    d = len(ww)
    mode_axes = list(range(d))                       # A's axes 0..d-1
    w_axis = d                                       # the probe-stack axis
    out = []
    for free in range(d):
        ops = [A, mode_axes]
        for j in range(d):
            if j != free:
                ops += [ww[j], [w_axis, mode_axes[j]]]
        ops += [[w_axis, mode_axes[free]]]           # output axes: (M, N_free)
        out.append(np.einsum(*ops))
    return out


def rms_all(arrs):
    """RMS over the concatenation of a list of arrays (probe outputs have different free-mode sizes)."""
    ss = sum(float(np.sum(np.asarray(a) ** 2)) for a in arrs)
    n = sum(np.asarray(a).size for a in arrs)
    return float(np.sqrt(ss / n))


def probe_relerr(pred, data):
    """Relative misfit ||pred - data|| / ||data|| over all probe-output elements."""
    num = sum(float(np.sum((np.asarray(p) - np.asarray(dd)) ** 2)) for p, dd in zip(pred, data))
    den = sum(float(np.sum(np.asarray(dd) ** 2)) for dd in data)
    return float(np.sqrt(num / den))


# --------------------------------------------------------------------------------------------------
# Corewise Adam (our own; minibatch over probes, stepping the flat core vector)
# --------------------------------------------------------------------------------------------------
def random_start(tucker_ranks, tt_ranks, ww, data):
    """A small random corewise start, rescaled so the initial probes match the data magnitude. Corewise
    needs a *nonzero* start (the zero tensor has a vanishing Jacobian); scaling every core by
    ``scale**(1/2d)`` scales the multilinear tensor by ``scale`` with the cores kept balanced."""
    X = t3.TuckerTensorTrain.randn(SHAPE, tucker_ranks, tt_ranks)
    scale = rms_all(data) / max(rms_all(X.probe(ww)), 1e-12)
    tucker_cores, tt_cores = X.data
    c = scale ** (1.0 / (len(tucker_cores) + len(tt_cores)))
    return t3.TuckerTensorTrain(tuple(c * C for C in tucker_cores), tuple(c * C for C in tt_cores))


def corewise_adam(X0, ww, data, tucker_ranks, tt_ranks, rng,
                  lr=ADAM_LR, betas=ADAM_BETAS, eps=ADAM_EPS, batch=ADAM_BATCH, max_iter=ADAM_MAXITER):
    """Fit X to probe data by corewise Adam, minibatching over probes. Adam is elementwise on the flat
    core vector (== per-core Adam): first/second moment EMAs ``m``/``v`` with bias correction. Each step
    rebuilds the point, probes the minibatch, and takes the corewise gradient from ``probe_model``."""
    b1, b2 = betas
    n = ww[0].shape[0]                                    # number of probes (axis 0 of each (M, N_i))
    x = X0.to_vector()
    m = np.zeros_like(x); v = np.zeros_like(x)
    for k in range(max_iter):
        sel = rng.choice(n, size=min(batch, n), replace=False)       # fresh minibatch of probes
        ww_B = [w[sel] for w in ww]
        data_B = [dd[sel] for dd in data]
        X = t3.TuckerTensorTrain.from_vector(x, SHAPE, tucker_ranks, tt_ranks)
        pred_B = X.probe(ww_B)
        r_B = [np.asarray(pred_B[i]) - data_B[i] for i in range(len(ww))]
        g = np.asarray(fitting.probe_model(t3m.COREWISE, X, ww_B, r_B).gradient.to_vector(), dtype=float)

        t = k + 1
        m = b1 * m + (1.0 - b1) * g
        v = b2 * v + (1.0 - b2) * g * g
        mhat = m / (1.0 - b1 ** t)
        vhat = v / (1.0 - b2 ** t)
        lr_t = lr * 0.5 * (1.0 + np.cos(np.pi * k / max_iter))   # cosine decay -> settle in late steps
        x = x - lr_t * mhat / (np.sqrt(vhat) + eps)

    X = t3.TuckerTensorTrain.from_vector(x, SHAPE, tucker_ranks, tt_ranks)
    return X, dict(iters=max_iter)


# --------------------------------------------------------------------------------------------------
# Rank helpers (shared with the other Hilbert examples)
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
    np.random.seed(SEED)               # the random corewise starts (TuckerTensorTrain.randn) draw from here
    rng = np.random.default_rng(SEED)  # data: probe vectors + measurement noise
    rng_opt = np.random.default_rng(SEED + 1)  # Adam's minibatch draws
    d = len(SHAPE)

    print(__doc__.split("\n\n")[0])
    print(f"\nHilbert tensor: shape {SHAPE}  ({np.prod(SHAPE):,} entries),  order d={d}")

    A = hilbert_tensor(SHAPE)
    A_norm = float(np.linalg.norm(A))

    M = N_TRAIN + N_VAL
    ww_all = unit_probes(M, SHAPE, rng)
    y_clean = dense_probe(A, ww_all)                       # list of d arrays (M, N_i)
    y_rms = rms_all(y_clean)
    y_all = [yc + NOISE_LEVEL * y_rms * rng.standard_normal(yc.shape) for yc in y_clean]

    ww_tr = [w[:N_TRAIN] for w in ww_all]
    ww_va = [w[N_TRAIN:] for w in ww_all]
    y_tr = [y[:N_TRAIN] for y in y_all]
    y_va = [y[N_TRAIN:] for y in y_all]

    print(f"Measurements (probes): {N_TRAIN} train + {N_VAL} validation, {NOISE_LEVEL*100:.0f}% noise.  "
          f"(unit-norm probe vectors)")
    print(f"Fit on the COREWISE geometry by our own Adam (lr={ADAM_LR} cosine-decayed, "
          f"minibatch {ADAM_BATCH} probes).\n")

    header = (f"{'level':>5} {'tucker / tt ranks':>24} {'DOF':>5} {'iters':>5} "
              f"{'train':>9} {'val':>9} {'true':>9} {'oracle':>9}")
    print(header)
    print("-" * len(header))

    records = []
    t_start = time.perf_counter()
    for r in RANK_LEVELS:
        tucker_ranks, tt_ranks = level_ranks(r, SHAPE)
        dof = t3m.manifold_dim((SHAPE, tucker_ranks, tt_ranks))

        X0 = random_start(tucker_ranks, tt_ranks, ww_tr, y_tr)
        X, stats = corewise_adam(X0, ww_tr, y_tr, tucker_ranks, tt_ranks, rng_opt)

        train_e = probe_relerr(X.probe(ww_tr), y_tr)
        val_e = probe_relerr(X.probe(ww_va), y_va)
        true_e = float(np.linalg.norm(X.to_dense() - A)) / A_norm
        oracle_e = oracle_relerr(A, tucker_ranks, tt_ranks)
        records.append(dict(level=r, dof=dof, val=val_e, true=true_e))

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
