"""Fit a Hilbert tensor from noisy ``apply`` measurements, and show that **identity (Tikhonov)
regularization** denoises the fit -- with the regularization strength ``λ`` chosen automatically by
held-out validation (no peeking at the true tensor).

A spectral-decay target
-----------------------
The order-``d`` **Hilbert tensor** ``A[i0, ..., i_{d-1}] = 1 / (1 + i0 + ... + i_{d-1})`` is full rank
but has fast, smooth singular-value **decay**, so it is well approximated by low multilinear rank. We
fit a fixed rank ``(3,3,3)``; the best a rank-``(3,3,3)`` model can do is the ``t3svd`` truncation
**floor** (~0.008 relative error) -- the error you would get from the *whole* dense tensor. We never
see ``A``: we observe ``M`` scalar **applies** ``b_s = A.apply([w0_s, ..., w_{d-1}_s])`` for random
unit-norm probe vectors, add noise, and fit.

With a few hundred noisy applies the fit is *well determined* but not perfectly -- the rank constraint
already regularizes heavily, yet a handful of directions are poorly constrained, and the unregularized
Gauss-Newton fit pours the measurement noise into them. The recovered tensor sits well above the
truncation floor, and slightly above what a denoised fit achieves.

The fix
-------
Add ``ρ(X) = ½λ‖X‖²`` to the objective: ``½‖S(X) − b‖² + ½λ‖X‖²``. On the manifold this is the
Hilbert-Schmidt ridge (its Gauss-Newton term is ``+λI``, conditioning the poorly-constrained
directions), trading a little bias for reduced variance -- the classic bias-variance **U-curve**. Pass
it to any optimizer as ``regularizer=optimizers.IdentityRegularizer(λ)``; it composes with every
sampling kind, geometry, and the ragged/uniform layers (it acts on ``X``, not on the measurements).

Because the manifold's rank constraint does most of the regularizing, identity ``λ`` is a *secondary*
knob here -- the gain is modest (~1.2x), not dramatic. (Identity regularization earns its keep most in
severely ill-posed fits, where the unregularized fit is worse than useless; a sharper prior that
denoises the ill-conditioned directions *selectively* -- weighting cores by inverse unfolding singular
values, after Grasedyck & Kramer -- is the tool for a larger gain here, and is future work.)

**Choosing λ without seeing the answer.** You cannot measure the recovery error in practice (no true
tensor). So we split the measurements into a **training** set (to fit) and a held-out **validation**
set, sweep ``λ``, and keep the fit with the smallest *validation* error -- which tracks the recovery
error and so finds the U-curve's bottom.

The demonstration
-----------------
For each of ``N_SEEDS`` random draws: fit unregularized and across a grid of ``λ``, report the
recovered-tensor error ``‖X − A‖_F / ‖A‖_F`` and the validation error, pick ``λ`` by validation, and
compare -- against each other and against the ``t3svd`` floor.

Run from the repo root:  ``python examples/fit_hilbert_regularized.py``
(or with ``PYTHONPATH`` set to the repo root from elsewhere).
"""
import numpy as np

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.manifold as t3m
import t3toolbox.optimizers as topt


# --------------------------------------------------------------------------------------------------
# Problem configuration (tweak these)
# --------------------------------------------------------------------------------------------------
N          = 16                            # the Hilbert tensor is N x N x N
SHAPE      = (N, N, N)
TUCKER     = (3, 3, 3)                     # fitted Tucker ranks (t3svd floor ~0.008 -- see the header)
TT         = (1, 3, 3, 1)                  # fitted TT bond ranks
M_TRAIN    = 400                           # training applies -- well determined (manifold dimension 144)
M_VAL      = 400                           # held-out applies, for choosing λ (does NOT see the true tensor)
SIGMA      = 0.015                         # additive measurement noise
LAMBDAS    = (0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)   # the regularization grid (0 = unregularized)
SEED       = 0
N_SEEDS    = 6
MAX_NEWTON = 60


def hilbert_tensor(n, d=3):
    """The order-``d`` Hilbert tensor ``A[i,j,k] = 1 / (1 + i + j + k)`` on ``{0,...,n-1}^d``."""
    idx = np.indices((n,) * d).sum(axis=0)
    return 1.0 / (1.0 + idx)


def dense_apply(A, ww):
    """Ground-truth scalar applies of the dense tensor ``A``: contract every mode against ``ww``.
    ``ww[i]`` is ``(M, N_i)``; returns a length-``M`` vector (the same contraction ``.apply`` computes,
    used here only to synthesize the data)."""
    res = np.einsum("i...,si->s...", A, ww[0])
    for m in range(1, len(ww)):
        res = np.einsum("sj...,sj->s...", res, ww[m])
    return res


def _unit_probes(M, rng):
    """``M`` unit-norm probe tuples (one ``(M, N_i)`` per mode) -- unit rows keep the least-squares
    well-scaled (see fit_hilbert_tensor_newton_cg.py)."""
    ww = [rng.standard_normal((M, n)) for n in SHAPE]
    return [w / np.linalg.norm(w, axis=1, keepdims=True) for w in ww]


def fit_over_lambdas(Ad, A_norm, seed):
    """One draw: take noisy train + validation applies of the (fixed) Hilbert tensor ``Ad``, fit across
    ``LAMBDAS`` from a zero start. Returns a dict ``λ -> (recovery_err, validation_err, ‖X‖, misfit, reg)``
    where ``misfit`` / ``reg`` are the final objective split from ``stats['history']``."""
    rng = np.random.default_rng(seed)
    ww_tr = _unit_probes(M_TRAIN, rng)
    b_tr = dense_apply(Ad, ww_tr) + SIGMA * rng.standard_normal(M_TRAIN)
    ww_va = _unit_probes(M_VAL, rng)
    b_va = dense_apply(Ad, ww_va) + SIGMA * rng.standard_normal(M_VAL)
    b_va_norm = float(np.linalg.norm(b_va))

    x0 = t3.TuckerTensorTrain.zeros(SHAPE, TUCKER, TT)   # a zero start is robust on the manifold
    out = {}
    for lam in LAMBDAS:
        reg = topt.IdentityRegularizer(lam) if lam > 0 else None
        x, stats = topt.newton_cg(t3m.MANIFOLD, 'apply', ww_tr, b_tr, x0, regularizer=reg, max_newton=MAX_NEWTON)
        recov = float(np.linalg.norm(x.to_dense() - Ad)) / A_norm
        val = float(np.linalg.norm(x.apply(ww_va) - b_va)) / b_va_norm     # held-out; no access to A
        last = stats['history'][-1]                                        # final objective split
        out[lam] = (recov, val, float(np.linalg.norm(x.to_dense())),
                    last['misfit'], last['regularization'])
    return out


def main():
    print(__doc__.split("\n\n")[0])
    Ad = hilbert_tensor(N)
    A_norm = float(np.linalg.norm(Ad))
    md = t3m.manifold_dim((SHAPE, TUCKER, TT))
    x_svd, _, _ = t3.TuckerTensorTrain.t3svd_dense(Ad, max_tucker_ranks=TUCKER, max_tt_ranks=TT)
    floor = float(np.linalg.norm(x_svd.to_dense() - Ad)) / A_norm
    print(f"\nTarget: Hilbert tensor A[i,j,k]=1/(1+i+j+k), shape {SHAPE}, ‖A‖ ≈ {A_norm:.2f}.")
    print(f"Model: fixed rank tucker {TUCKER} / tt {TT} (manifold dimension {md}). Best a rank-{TUCKER} model")
    print(f"  can do, given the WHOLE dense tensor: t3svd floor = {floor:.4f} relative error.")
    print(f"Data: {M_TRAIN} noisy training applies + {M_VAL} validation applies; noise σ = {SIGMA}.")

    # ---- one representative seed: the bias-variance U-curve --------------------------------------
    res = fit_over_lambdas(Ad, A_norm, SEED)
    lam_recov = min(res, key=lambda l: res[l][0])        # recovery-optimal λ (needs the true A -- for reference)
    lam_val = min(res, key=lambda l: res[l][1])          # λ chosen by held-out validation (no A)
    print(f"\nSeed {SEED} -- sweep over λ (recovery error needs the true A; validation error does not):")
    print(f"  {'λ':>8} {'‖X‖':>8} {'recovery err':>13} {'validation err':>15}")
    for lam in LAMBDAS:
        recov, val, nx, _, _ = res[lam]
        tag = ''
        if lam == lam_recov: tag += '  <- best recovery'
        if lam == lam_val:   tag += '  <- validation picks this'
        print(f"  {lam:>8.0e} {nx:>8.2f} {recov:>13.3f} {val:>15.4f}{tag}")
    print(f"\n  λ=0 (unregularized) overfits the noise slightly: recovery error {res[0.0][0]:.3f}, above both the")
    print(f"  regularized fit and the t3svd floor ({floor:.4f}). Larger λ shrinks ‖X‖ and conditions the")
    print(f"  poorly-determined directions; too much over-shrinks -- the bias-variance U-curve. Held-out")
    print(f"  validation finds the bottom WITHOUT seeing A (it picks λ={lam_val:.0e}, "
          f"{'the recovery optimum' if lam_val == lam_recov else 'near the optimum'}).")

    # ---- the objective split at the selected fit (the misfit + reg breakdown) --------------------
    _, _, _, misfit, reg = res[lam_val]
    if reg is not None:
        print(f"\n  At λ={lam_val:.0e} the final objective splits as misfit {misfit:.3e} + reg {reg:.3e} "
              f"(= {misfit + reg:.3e});\n  the same split prints per-iteration under newton_cg(..., verbose=True) "
              f"and rides along in stats['history'].")

    # ---- robustness across seeds ----------------------------------------------------------------
    unreg, valsel, hits = [], [], 0
    for s in range(N_SEEDS):
        r = res if s == 0 else fit_over_lambdas(Ad, A_norm, SEED + s)   # reuse the representative seed's fit
        lr = min(r, key=lambda l: r[l][0]); lv = min(r, key=lambda l: r[l][1])
        unreg.append(r[0.0][0]); valsel.append(r[lv][0]); hits += (lv == lr)
    unreg, valsel = np.array(unreg), np.array(valsel)
    print(f"\nAcross {N_SEEDS} seeds:")
    print(f"  unregularized (λ=0)         : mean recovery error {unreg.mean():.3f}")
    print(f"  validation-selected λ       : mean recovery error {valsel.mean():.3f}   "
          f"({unreg.mean() / valsel.mean():.2f}x better)")
    print(f"  validation picked the recovery-optimal λ in {hits}/{N_SEEDS} seeds "
          f"(near-optimal in the rest).")
    print(f"\nOn the fixed-rank manifold the rank constraint is the primary regularizer, so identity λ is a")
    print(f"modest secondary denoiser -- but a genuine one, and validation sets it automatically.")


if __name__ == "__main__":
    main()
