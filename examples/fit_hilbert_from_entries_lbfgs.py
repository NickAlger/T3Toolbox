"""Tensor completion: fit a Tucker tensor train to sampled ENTRIES of a Hilbert tensor.

A worked example of **corewise** fixed-rank fitting -- the Euclidean half of T3Toolbox's geometry
abstraction -- driven by **scipy's L-BFGS-B** through the library's flat-vector bridge. It is the
companion to ``fit_hilbert_tensor_newton_cg.py`` (apply data, *manifold* geometry, Riemannian
Newton-CG): same target tensor, a different information source (entries), the *other* geometry, and an
*external* quasi-Newton optimizer. Three things on display:

  1. **Entries / tensor completion** -- the most intuitive sampling op: we observe ``A`` at a handful of
     random multi-indices and "fill in" the rest by fitting a low-rank ``X`` with ``X[idx] ~ A[idx]``.
  2. **The corewise geometry** (``t3m.COREWISE``) -- optimize on the raw core parameters ``(U, G, G, G)``
     with the Euclidean metric and additive retraction (``cores += step``), rather than on the fixed-rank
     manifold. The corewise Gauss-Newton Hessian is gauge-singular, but first-order / quasi-Newton methods
     (L-BFGS, Adam) tolerate that: the gradient is gauge-orthogonal, so the gauge directions are harmless
     flat valleys (no damping needed -- contrast Newton on the cores, which would need regularization).
  3. **Driving the fit from an external optimizer via a flat array** -- ``TuckerTensorTrain.to_vector`` /
     ``from_vector`` map our cores to/from the 1-D vector scipy wants, and -- the crux -- the corewise
     gradient flattens (``model.gradient.to_vector()``) into the **same** coordinates as the point (both
     route through one ``t3_to_vector``). So ``scipy.optimize.minimize(..., jac=True)`` "just works". This
     is the reusable recipe for plugging a T3 fit into *any* flat-array optimizer (scipy, your own, ...).
     The library itself stays dependency-free; scipy is imported **only here, in the example**.

The problem
-----------
The order-``d`` **Hilbert tensor** ``A[i0,...] = 1 / (1 + i0 + ... + i_{d-1})`` is smooth, positive, and
well approximated by low rank. We observe it only at ``N`` random entries (a training set + a held-out
validation set), with a little noise, and complete it. Unlike applies / probes, an entry's measurement
"row" is a one-hot rank-1 tensor of unit norm already, so **no row-normalization is needed** here.

The method -- corewise L-BFGS-B
-------------------------------
We minimize the training misfit ``f(X) = 1/2 ||X.entries(idx) - y||^2`` over the cores. Each scipy
function evaluation rebuilds the point from the flat vector, samples its entries, builds a corewise
``fitting.entries_model`` (which gives the objective and the corewise gradient ``J^T r``, *no* gauge
projection), and hands scipy ``(f, grad_flat)``. scipy owns the L-BFGS update and the (robust, well-
tested) Wolfe line search -- exactly the machinery we don't want to reimplement.

    *Why a nonzero start (a real corewise/manifold difference).*  The manifold example starts from the
    *zero* tensor: ``t3_orthogonal_representations`` completes the rank-deficient frame with orthonormal
    directions, so the Jacobian there is nonzero and the first step "bumps off" onto the manifold.  On the
    cores there is no such completion -- at all-zero cores every single-core swap multiplies in a zero
    core, so ``J = 0``, the gradient vanishes, and L-BFGS cannot move.  So corewise needs a **nonzero
    start** (level 1: a small random tensor, rescaled so the initial prediction matches the data
    magnitude).

Rank selection (and why no warm-started continuation)
-----------------------------------------------------
We fit at ranks ``1..R`` and pick the level that minimizes the **held-out validation** misfit (``oracle``
is the best rank-``r`` dense T3-SVD -- a diagnostic only). Note what this does **not** do: the manifold
examples warm-start each level from the zero-padded previous solution, keeping every refit near its basin
and giving clean, monotone tables. **Corewise cannot.** A zero-padded warm start leaves the new rank block
at zero, where -- by the same vanishing-Jacobian argument above -- its gradient is exactly zero, so L-BFGS
freezes it (every higher level just reproduces the lower one). So each level here is an **independent cold
random fit**, which is noisier across levels (a level can land in a worse local minimum than its neighbour
-- the occasional bump in the table; validation still selects a good level). That asymmetry is itself a
lesson: **warm-started rank continuation is a genuine advantage of the manifold geometry.**

Why a sparse entry sample fits at all -- implicit regularization
----------------------------------------------------------------
Entries are a *weak, localized* source, and the Hilbert tensor is *coherent* (its mass sits at the low-
index corner, rarely hit by uniform sampling), so it is badly under-determined from a sparse entry sample.
A hard second-order solver (manifold Newton-CG) **overfits catastrophically** on this data -- it matches
the observed entries while blowing the unobserved corner up (true error several times the tensor norm).
Corewise L-BFGS does **not**: the over-parametrized cores, started small, bias it toward a **low-norm**
solution (the implicit regularization studied in over-parametrized matrix/tensor factorization), so it
recovers a sensible completion. This is a *converged* property, not early stopping -- running 10x longer
does not change it. At this 24% sample the completion bottoms out a few % off the true tensor (not the 1%
noise floor): from a sparse, localized sample the entries pin the tensor only that well, and more entries
close the gap. (MC-SGD on the manifold regularizes similarly; see the docs note.)

Run from the repo root:  ``python examples/fit_hilbert_from_entries_lbfgs.py``
"""
import time

import numpy as np
from scipy.optimize import minimize          # external optimizer -- imported ONLY in this example

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.manifold as t3m
import t3toolbox.fitting as fitting


# --------------------------------------------------------------------------------------------------
# Problem configuration (tweak these)
# --------------------------------------------------------------------------------------------------
SHAPE        = (12, 12, 12, 12)      # the Hilbert tensor's shape (order d = len(SHAPE))
N_TRAIN      = 3000                   # observed training entries (~14% -- a genuinely sparse sample)
N_VAL        = 2000                   # held-out validation entries
NOISE_LEVEL  = 0.01                  # measurement noise, as a fraction of the entry RMS
RANK_LEVELS  = (1, 2, 3, 4, 5, 6)    # rank schedule (each level fit independently; see the docstring)
SEED         = 0

# scipy L-BFGS-B knobs. We cap the iterations to keep the demo quick (~30-60s); the corewise fit
# regularizes (it converges to a bounded, low-norm solution -- it does NOT overfit with more iterations),
# so a modest cap loses a little accuracy but nothing essential.
LBFGS_MAXITER = 600
LBFGS_GTOL    = 1e-7                  # projected-gradient stopping tolerance


# --------------------------------------------------------------------------------------------------
# Target tensor + entry sampling
# --------------------------------------------------------------------------------------------------
def hilbert_tensor(shape):
    """Dense Hilbert tensor A[i0,...] = 1 / (1 + i0 + ... + i_{d-1})."""
    grids = np.meshgrid(*[np.arange(N) for N in shape], indexing="ij")
    return 1.0 / (1.0 + sum(grids))


def sample_indices(n, shape, rng):
    """``n`` distinct random multi-indices of ``shape``, as an int array of shape ``(d, n)`` -- the
    ``index`` format ``TuckerTensorTrain.entries`` / ``fitting.entries_model`` expect (mode ``i``'s
    indices in row ``i``)."""
    flat = rng.choice(int(np.prod(shape)), size=n, replace=False)
    return np.array(np.unravel_index(flat, shape))        # (d, n)


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


# --------------------------------------------------------------------------------------------------
# Corewise L-BFGS-B fit (the flat-vector bridge to scipy)
# --------------------------------------------------------------------------------------------------
def random_start(tucker_ranks, tt_ranks, index, y):
    """A small random corewise start (level 1), rescaled so the initial prediction's RMS matches the
    data's. Corewise needs a *nonzero* start (at all-zero cores the Jacobian vanishes -- see the module
    docstring). The tensor is multilinear in its ``2d`` cores, so scaling every core by ``scale**(1/2d)``
    scales the tensor by ``scale`` while keeping the cores balanced (good conditioning)."""
    X = t3.TuckerTensorTrain.randn(SHAPE, tucker_ranks, tt_ranks)
    scale = rms(y) / max(rms(X.entries(index)), 1e-12)
    tucker_cores, tt_cores = X.data
    c = scale ** (1.0 / (len(tucker_cores) + len(tt_cores)))
    return t3.TuckerTensorTrain(tuple(c * C for C in tucker_cores), tuple(c * C for C in tt_cores))


def corewise_lbfgs(X0, index, y, tucker_ranks, tt_ranks,
                   maxiter=LBFGS_MAXITER, gtol=LBFGS_GTOL):
    """Fit X to entries ``y`` at ``index`` by corewise L-BFGS-B (scipy), via the flat-vector bridge.

    Each scipy evaluation: rebuild the point from the flat core vector, sample its entries, build the
    corewise Gauss-Newton model, and return ``(f, grad_flat)``. The gradient is the *corewise* ``J^T r``
    (no gauge projection -- ``COREWISE.project`` is the identity); its ``to_vector()`` shares the point's
    flat layout, which is what makes the round-trip exact."""
    def objective_and_grad(x_flat):
        X = t3.TuckerTensorTrain.from_vector(x_flat, SHAPE, tucker_ranks, tt_ranks)
        r = np.asarray(X.entries(index)) - y                      # residual, shape (n_obs,)
        model = fitting.entries_model(t3m.COREWISE, X, index, r)   # corewise GN model at X
        f = float(model.objective_value)                          # 1/2 ||r||^2
        g_flat = np.asarray(model.gradient.to_vector(), dtype=float)   # corewise J^T r, point's layout
        return f, g_flat

    res = minimize(objective_and_grad, X0.to_vector(), method="L-BFGS-B", jac=True,
                   options=dict(maxiter=maxiter, gtol=gtol))
    X = t3.TuckerTensorTrain.from_vector(res.x, SHAPE, tucker_ranks, tt_ranks)
    return X, dict(iters=int(res.nit))


# --------------------------------------------------------------------------------------------------
# Rank continuation helpers (shared with the other Hilbert examples)
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
    rng = np.random.default_rng(SEED)  # our own draws: the sampled entries + the measurement noise
    d = len(SHAPE)

    print(__doc__.split("\n\n")[0])
    print(f"\nHilbert tensor: shape {SHAPE}  ({np.prod(SHAPE):,} entries),  order d={d}")

    A = hilbert_tensor(SHAPE)
    A_norm = float(np.linalg.norm(A))

    # ---- target + data: observe a fraction of the entries (train + held-out validation) -------------
    index_all = sample_indices(N_TRAIN + N_VAL, SHAPE, rng)        # (d, N_TRAIN+N_VAL)
    y_clean = A[tuple(index_all)]                                  # (N_TRAIN+N_VAL,)
    y_rms = rms(y_clean)
    y_all = y_clean + NOISE_LEVEL * y_rms * rng.standard_normal(y_clean.shape)

    idx_tr, idx_va = index_all[:, :N_TRAIN], index_all[:, N_TRAIN:]
    y_tr, y_va = y_all[:N_TRAIN], y_all[N_TRAIN:]

    frac = (N_TRAIN + N_VAL) / float(np.prod(SHAPE))
    print(f"Observed entries: {N_TRAIN} train + {N_VAL} validation  ({frac*100:.0f}% of the tensor),  "
          f"{NOISE_LEVEL*100:.0f}% noise.")
    print(f"Fit on the COREWISE geometry by scipy L-BFGS-B (random start per rank level).\n")

    header = (f"{'level':>5} {'tucker / tt ranks':>24} {'DOF':>5} {'iters':>5} "
              f"{'train':>9} {'val':>9} {'true':>9} {'oracle':>9}")
    print(header)
    print("-" * len(header))

    records = []
    t_start = time.perf_counter()
    for r in RANK_LEVELS:
        tucker_ranks, tt_ranks = level_ranks(r, SHAPE)
        dof = t3m.manifold_dim((SHAPE, tucker_ranks, tt_ranks))

        X0 = random_start(tucker_ranks, tt_ranks, idx_tr, y_tr)   # cold balanced random start, this level
        X, stats = corewise_lbfgs(X0, idx_tr, y_tr, tucker_ranks, tt_ranks)

        train_e = rms(np.asarray(X.entries(idx_tr)) - y_tr) / y_rms
        val_e = rms(np.asarray(X.entries(idx_va)) - y_va) / y_rms
        true_e = float(np.linalg.norm(X.to_dense() - A)) / A_norm
        oracle_e = oracle_relerr(A, tucker_ranks, tt_ranks)
        records.append(dict(level=r, dof=dof, val=val_e, true=true_e))

        rank_str = f"{tucker_ranks} {tt_ranks}"
        print(f"{r:>5} {rank_str:>24} {dof:>5} {stats['iters']:>5} "
              f"{train_e:>9.3e} {val_e:>9.3e} {true_e:>9.3e} {oracle_e:>9.3e}")

    best = min(records, key=lambda rec: rec["val"])
    print("-" * len(header))
    print(f"\n(total fit time {time.perf_counter()-t_start:.1f}s)")
    print(f"Best ranks by validation error: level {best['level']}  "
          f"(val {best['val']:.3e}, true error {best['true']:.3e}, DOF {best['dof']}).")
    print(f"True error bottoms out near {best['true']:.1%}, not the {NOISE_LEVEL:.0%} noise floor: from this\n"
          f"sparse, localized sample the entries fix the tensor only to that accuracy (a hard manifold\n"
          f"solver instead OVERFITS this data -- see the docs note). More entries close the gap.")


if __name__ == "__main__":
    main()
