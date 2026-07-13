"""Watch a Newton-CG fit converge, live -- the ``verbose=True`` diagnostic display for BOTH probe kinds,
so you can see the two table layouts side by side.

One synthetic low-rank Tucker tensor train is fit twice, each with ``optimizers.newton_cg(..., verbose=True,
val_sample=…, val_data=…)`` and a train/validation split:

  1. **plain probe** (kind ``'probe'``) -- a single data axis (mode), so the relative-error table puts
     **modes in columns and train/val in rows**;
  2. **probe derivatives** (kind ``'probe_derivatives'``) -- two data axes (mode × order), so the table
     puts **modes in rows, orders in columns, and train|val in each cell**.

Each Newton iteration prints a header line -- objective (and, when a residual weight ``ω`` is used, the
unweighted objective), gradient norm, CG iterations / tolerance / achieved residual / status
(``✓`` converged, ``⋯`` hit maxiter, ``⌇`` truncated on nonpositive curvature), line-search steps and step
length, the actual-vs-predicted reduction ``ρ``, and the wall time -- followed by the relative-error table
``‖S(x)_ij − y_ij‖ / ‖y_ij‖`` per ``(mode, order)`` block, for train and validation.

The per-iteration records are also returned in ``stats['diagnostics']`` (the same data, for plotting).
Everything here is a thin convenience over the **backend** display: a raw-``.data`` user gets the identical
output via ``backend.optimizer_display.make_newton_display`` + ``backend.optimizers.newton_cg`` (no frontend).

Run from the repo root:  ``python examples/fit_probe_display.py``
(or with ``PYTHONPATH`` set to the repo root from elsewhere).
"""
import numpy as np

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.manifold as t3m
import t3toolbox.optimizers as topt


# --------------------------------------------------------------------------------------------------
# Problem configuration
# --------------------------------------------------------------------------------------------------
SHAPE   = (10, 10, 10)       # target shape (order d = len(SHAPE))
TUCKER  = (2, 2, 2)          # true Tucker ranks
TT      = (1, 2, 2, 1)       # true TT bond ranks
M_TRAIN = 150                # training probe tuples
M_VAL   = 60                 # validation probe tuples
ORDER   = 2                  # highest derivative order (for the probe_derivatives fit)
MAX_NEWTON = 6
SEED    = 0


def unit_probes(M, rng):
    """M probe tuples (a W=(M,) stack), each vector scaled to unit norm."""
    ww = [rng.standard_normal((M, N)) for N in SHAPE]
    return [w / np.linalg.norm(w, axis=1, keepdims=True) for w in ww]


def main():
    np.random.seed(SEED)                       # TuckerTensorTrain.randn draws from the global rng
    rng = np.random.default_rng(SEED)
    A = t3.TuckerTensorTrain.randn(SHAPE, TUCKER, TT)
    x0 = t3.TuckerTensorTrain.zeros(SHAPE, TUCKER, TT)   # a zero start is valid on the manifold

    ww, wwv = unit_probes(M_TRAIN, rng), unit_probes(M_VAL, rng)

    # ---- 1. plain probe: one data axis (mode) -> modes in columns, train/val in rows ----------------
    print("=" * 100)
    print("1. PLAIN PROBE  (kind 'probe') -- single data axis: rel-err table has MODES IN COLUMNS, "
          "train/val in rows")
    print("=" * 100)
    y_tr = A.probe(ww)                                    # d per-mode data vectors (train)
    y_va = A.probe(wwv)                                   # ... and validation
    topt.newton_cg(t3m.MANIFOLD, 'probe', ww, y_tr, x0,
                   verbose=True, val_sample=wwv, val_data=y_va, max_newton=MAX_NEWTON)

    # ---- 2. probe derivatives: two data axes (mode x order) -> modes rows, orders cols, train|val cells
    print()
    print("=" * 100)
    print("2. PROBE DERIVATIVES  (kind 'probe_derivatives', order %d) -- two data axes: rel-err table has "
          "MODES IN ROWS, ORDERS IN COLUMNS, train|val cells" % ORDER)
    print("=" * 100)
    pp, ppv = unit_probes(M_TRAIN, rng), unit_probes(M_VAL, rng)   # one perturbation direction per probe
    jet_tr = [np.asarray(z) for z in A.probe_derivatives(ww, pp, ORDER)]    # d jets, (order+1)+W+(Ni,)
    jet_va = [np.asarray(z) for z in A.probe_derivatives(wwv, ppv, ORDER)]
    # a per-order residual weight ω = 1/‖data at that order‖ conditions the wildly-scaled orders (§4.6);
    # it makes the header show the unweighted objective alongside the weighted one.
    order_norm = np.sqrt(sum(np.sum(z ** 2, axis=tuple(range(1, z.ndim))) for z in jet_tr))  # (order+1,)
    omega = 1.0 / order_norm                                                # a row vector -> per-order
    _, stats = topt.newton_cg(t3m.MANIFOLD, 'probe_derivatives', (ww, pp), jet_tr, x0, order=ORDER,
                              weight=omega, verbose=True, val_sample=(wwv, ppv), val_data=jet_va,
                              max_newton=MAX_NEWTON)

    # the same per-iteration diagnostics are returned for plotting / inspection:
    print()
    print("stats['diagnostics'] holds one record per iteration; e.g. the final validation error table "
          "(rows=mode, cols=order):")
    print(np.array2string(np.asarray(stats['diagnostics'][-1]['val_err']),
                          formatter={'float_kind': lambda v: '%.1e' % v}))


if __name__ == "__main__":
    main()
