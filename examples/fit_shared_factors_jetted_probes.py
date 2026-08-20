"""Fit a GROUPWISE-SYMMETRIC tensor from noisy probe-derivative jets: shared vs unshared factors.

The showcase example for **shared Tucker factors** (SF-T3, ``docs/sharing.md``): when groups of
modes play symmetric roles, tying their Tucker factors (``sharing=``) removes parameters without
removing anything the target needs -- and under noise, the smaller model wins. We run the SAME
rank-continuation fit twice, differing ONLY in the geometry (``shared_manifold(sharing)`` vs plain
``MANIFOLD``), and compare what each achieves before overfitting.

The target
----------
A five-mode, **groupwise-symmetric** tensor built from two Hilbert tensors coupled by a random
matrix::

    T[i,j,k,n,o] = sum_{l,m} A[i,j,k,l] * B[l,m] * C[m,n,o],
    A[i,j,k,l] = 1/(1+i+j+k+l)   (Hilbert: symmetric in i,j,k),
    C[m,n,o]  = 1/(1+m+n+o)      (Hilbert: symmetric in n,o),

so ``T`` is symmetric under permutations *within* the mode groups ``{0,1,2}`` and ``{3,4}`` (but
not across them -- the groups even have different mode sizes). The matching partition is
``sharing = (0, 0, 0, 1, 1)``: two nontrivial groups, mirroring the symmetry. Because the target
IS groupwise symmetric, the shared manifold contains it -- tying costs no bias; it only removes
variance.

The data: probe-derivative jets
-------------------------------
We observe noisy **jets** of the vector-valued ``probe`` operation: for unit-norm probe vectors
``X`` and directions ``P``, each sample carries the directional derivatives of every
``probe_i(T; X + sP)`` at ``s = 0``, orders ``0..2`` (order 0 is the ordinary probe). A per-order
weight ``omega_t = 1/RMS_t`` makes the wildly-different-magnitude orders contribute comparably --
see ``examples/fit_hilbert_uniform_probe_derivatives_newton_cg.py`` for the jet-fitting mechanics;
this example changes the question, not the machinery.

The comparison: same data, two geometries
-----------------------------------------
Both runs use the identical noisy training data, validation split, per-order weight, adaptive
rank-continuation policy (``continuation_ranks`` + zero-padded ``resize`` warm starts, with the
``g0norm_newton`` pin of ``docs/rank_continuation.md``), and Newton budget. The only difference:

* **shared**   -- ``shared_manifold((0,0,0,1,1))``, with ``sharing=`` threaded through
  ``continuation_ranks`` / ``resize`` (a group's Tucker edges are ONE edge: one condition number
  ``kappa_g`` from the group spectrum, one growth decision group-wide, the group ceiling in
  useless-rank removal); every iterate keeps ONE factor array per group.
* **unshared** -- plain ``MANIFOLD``, the standard per-mode continuation.

What to expect: at every rank level the shared model has FEWER parameters (one Stiefel term per
group -- compare the DOF columns, ``manifold_dim(s, sharing=...)`` vs ``manifold_dim(s)``), so its
noise floor is lower and its validation error turns up (overfits) later. The gap is real but
bounded: TT-core parameters are never tied, so the advantage scales like the square root of the
DOF ratio. With this configuration the shared fit reaches ~25-35% lower true error at its
validation-selected rank, with ~40% fewer parameters.

Run from the repo root:  ``python examples/fit_shared_factors_jetted_probes.py``   (~1.5 minutes)
"""
import numpy as np

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.manifold as t3m
import t3toolbox.optimizers as optimizers
import t3toolbox.shared_geometry as sg
import t3toolbox.backend.sampling_derivatives as pd   # dense ground-truth jets (data generation only)


# --------------------------------------------------------------------------------------------------
# Problem configuration (tweak these)
# --------------------------------------------------------------------------------------------------
NI, L, M2, NO = 10, 8, 6, 8        # T = A(NI,NI,NI,L) . B(L,M2) . C(M2,NO,NO)
SHAPE          = (NI, NI, NI, NO, NO)
SHARING        = (0, 0, 0, 1, 1)   # the partition mirroring T's groupwise symmetry
ORDER          = 2                 # jet orders 0..2 (order 0 = the ordinary probe)
N_TRAIN, N_VAL = 12, 12            # (frame, direction) samples: few enough that noise matters
NOISE          = 0.05              # measurement noise, fraction of the per-order RMS
TAU            = 3.0               # continuation sensitivity (docs/rank_continuation.md)
MAX_LEVELS     = 6
MAX_NEWTON     = 30
SEED           = 0


# --------------------------------------------------------------------------------------------------
# Target + dense ground-truth jets
# --------------------------------------------------------------------------------------------------
def groupwise_symmetric_target():
    i, j, k, l = np.ogrid[0:NI, 0:NI, 0:NI, 0:L]
    A = 1.0 / (1.0 + i + j + k + l)                       # Hilbert: symmetric in (i, j, k)
    B = np.random.default_rng(12345).standard_normal((L, M2))   # fixed, independent of the data seed
    m, n, o = np.ogrid[0:M2, 0:NO, 0:NO]
    C = 1.0 / (1.0 + m + n + o)                           # Hilbert: symmetric in (n, o)
    return np.einsum('ijkl,lm,mno->ijkno', A, B, C)


def unit_vectors(M, shape, rng):
    vv = [rng.standard_normal((M, N)) for N in shape]
    return [v / np.linalg.norm(v, axis=1, keepdims=True) for v in vv]


def dense_jets(T, ww, pp, order):
    """Ground-truth probe-derivative jets of the dense target (data generation only)."""
    M, d = ww[0].shape[0], len(ww)
    per = [pd.dense_probe_derivatives([w[s] for w in ww], [p[s] for p in pp], T, order) for s in range(M)]
    return [np.stack([per[s][i] for s in range(M)], axis=1) for i in range(d)]


def order_rms(data, order):
    return np.array([max(float(np.sqrt(np.mean(
        np.concatenate([np.asarray(z[t]).ravel() for z in data]) ** 2))), 1e-12)
        for t in range(order + 1)])


def weighted_misfit(pred, data, omega):
    def wnorm(seq):
        return np.sqrt(sum(float(np.sum((omega[:, None, None] * np.asarray(z)) ** 2)) for z in seq))
    return wnorm([p - q for p, q in zip(pred, data)]) / wnorm(data)


# --------------------------------------------------------------------------------------------------
# One continuation run (the ONLY difference between the two runs is `geometry` + `sharing`)
# --------------------------------------------------------------------------------------------------
def continuation_fit(geometry, sharing, label, T, T_norm, ww_tr, pp_tr, data_tr,
                     ww_va, pp_va, data_va_clean, omega, n_eq):
    header = (f"{'':>9} {'tucker ranks':>16} {'tt ranks':>20} {'DOF':>5} "
              f"{'train (wtd)':>11} {'val (wtd)':>11} {'true':>10}")
    print(header)
    print("-" * len(header))
    X = t3.TuckerTensorTrain.zeros(SHAPE, (1,) * len(SHAPE), (1,) * (len(SHAPE) + 1))
    g0 = None
    records = []
    for level in range(MAX_LEVELS):
        kwargs = dict(max_newton=MAX_NEWTON)
        if g0 is not None:
            kwargs['g0norm_newton'] = g0          # pin the Newton reference across warm starts
        X, stats = optimizers.newton_cg(geometry, 'probe_derivatives', (ww_tr, pp_tr),
                                        data_tr, X, order=ORDER, weight=omega, **kwargs)
        if g0 is None:
            g0 = stats['history'][0]['gnorm']
        dof = t3m.manifold_dim((SHAPE, X.tucker_ranks, X.tt_ranks), sharing=sharing)
        tr_e = weighted_misfit([np.asarray(z) for z in X.probe_derivatives(ww_tr, pp_tr, ORDER)],
                               data_tr, omega)
        va_e = weighted_misfit([np.asarray(z) for z in X.probe_derivatives(ww_va, pp_va, ORDER)],
                               data_va_clean, omega)
        true_e = float(np.linalg.norm(np.asarray(X.to_dense()) - T)) / T_norm
        records.append(dict(level=level, tucker=X.tucker_ranks, tt=X.tt_ranks, dof=dof,
                            va=va_e, true=true_e, x=X))
        print(f"{label:>9} {str(X.tucker_ranks):>16} {str(X.tt_ranks):>20} {dof:>5} "
              f"{tr_e:>11.3e} {va_e:>11.3e} {true_e:>10.3e}")

        new_n, new_r = X.continuation_ranks(sharing=sharing, tau=TAU)
        if (new_n, new_r) == (X.tucker_ranks, X.tt_ranks):
            print(f"{'':>9} (stop: continuation returned unchanged ranks)")
            break
        if n_eq / t3m.manifold_dim((SHAPE, new_n, new_r), sharing=sharing) < 1.0:
            print(f"{'':>9} (stop: the next level would be underdetermined)")
            break
        X = X.resize(SHAPE, new_n, new_r, sharing=sharing)   # zero-padded warm start (tied if sharing)
    return records


# --------------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(SEED)
    print(__doc__.split("\n\n")[0])

    # ---- target -----------------------------------------------------------------------------------
    T = groupwise_symmetric_target()
    T_norm = float(np.linalg.norm(T))
    assert np.allclose(T, np.transpose(T, (1, 0, 2, 3, 4)))     # symmetric within group {0,1,2} ...
    assert np.allclose(T, np.transpose(T, (0, 1, 2, 4, 3)))     # ... and within group {3,4}
    print(f"\nTarget: shape {SHAPE}, groupwise symmetric under {SHARING} "
          f"(groups {{0,1,2}} and {{3,4}}; sizes differ across groups).")

    # The group spectrum on symmetric data (docs/sharing.md, 'What the group spectrum is'): within a
    # symmetric group the mode Grams are equal, so s_g = sqrt(k) * sigma elementwise and the group
    # condition number kappa_g EQUALS the per-mode one -- the shared edge competes fairly.
    x_true = t3.TuckerTensorTrain.t3svd_dense(T)[0].share(SHARING)
    _, sk_g, _ = x_true.t3svd(sharing=SHARING)
    _, sk_u, _ = x_true.t3svd()
    dev = float(np.max(np.abs(np.asarray(sk_g[0]) - np.sqrt(3.0) * np.asarray(sk_u[0])))
                / float(sk_g[0][0]))
    print(f"Group {{0,1,2}} spectrum check: max |s_g - sqrt(3)*sigma| / s_g[0] = {dev:.2e} "
          f"(the symmetric degeneration).\n")

    # ---- data: noisy jets, one split, one weight -- shared by BOTH runs ----------------------------
    M = N_TRAIN + N_VAL
    ww = unit_vectors(M, SHAPE, rng)
    pp = unit_vectors(M, SHAPE, rng)
    data_clean = dense_jets(T, ww, pp, ORDER)
    ww_tr, ww_va = [w[:N_TRAIN] for w in ww], [w[N_TRAIN:] for w in ww]
    pp_tr, pp_va = [p[:N_TRAIN] for p in pp], [p[N_TRAIN:] for p in pp]
    data_tr_clean = [z[:, :N_TRAIN] for z in data_clean]
    data_va_clean = [z[:, N_TRAIN:] for z in data_clean]

    s_vec = order_rms(data_tr_clean, ORDER)
    omega = 1.0 / s_vec
    data_tr = [z + NOISE * s_vec[:, None, None] * rng.standard_normal(z.shape) for z in data_tr_clean]
    n_eq = sum(z.size for z in data_tr)
    print(f"Data: {N_TRAIN} train + {N_VAL} validation jet samples, orders 0..{ORDER}, "
          f"{NOISE*100:.0f}% noise ({n_eq} scalar training equations).\n")

    # ---- the two continuation runs ------------------------------------------------------------------
    args = (T, T_norm, ww_tr, pp_tr, data_tr, ww_va, pp_va, data_va_clean, omega, n_eq)
    print(f"SHARED fit: shared_manifold({SHARING}) -- one Tucker factor per group, every iterate tied")
    rec_s = continuation_fit(sg.shared_manifold(SHARING), SHARING, 'shared', *args)
    print(f"\nUNSHARED fit: plain MANIFOLD on the same data")
    rec_u = continuation_fit(t3m.MANIFOLD, None, 'unshared', *args)

    # ---- model selection + the verdict --------------------------------------------------------------
    best_s = min(rec_s, key=lambda r: r['va'])
    best_u = min(rec_u, key=lambda r: r['va'])
    assert bool(np.all(np.asarray(best_s['x'].has_shared_tucker_factors(SHARING))))
    assert best_s['x'].data[0][0] is best_s['x'].data[0][1]     # ONE factor array per group

    print("\nBest level by validation error:")
    print(f"  shared:   ranks {best_s['tucker']} {best_s['tt']}  DOF {best_s['dof']:>4}  "
          f"val {best_s['va']:.3e}  true {best_s['true']:.3e}")
    print(f"  unshared: ranks {best_u['tucker']} {best_u['tt']}  DOF {best_u['dof']:>4}  "
          f"val {best_u['va']:.3e}  true {best_u['true']:.3e}")
    print(f"\nSame data, same noise, same continuation policy: tying the factors the target's "
          f"symmetry justifies\nreaches {100 * (1 - best_s['true'] / best_u['true']):.0f}% lower "
          f"true error with {100 * (1 - best_s['dof'] / best_u['dof']):.0f}% fewer parameters "
          f"(and the shared fit's factors\nstay exactly tied -- one array per group -- at every "
          f"iterate).")


if __name__ == "__main__":
    main()
