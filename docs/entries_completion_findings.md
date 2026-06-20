# Fitting the Hilbert tensor — entries vs probes, and the geometry/optimizer interplay

*Findings from building two corewise fitting examples (2026-06-19/20):
[`fit_hilbert_from_entries_lbfgs.py`](../examples/fit_hilbert_from_entries_lbfgs.py) (entries → scipy
L-BFGS-B) and [`fit_hilbert_from_probes_adam.py`](../examples/fit_hilbert_from_probes_adam.py) (probes →
a hand-written Adam), plus an optimizer bake-off on the probe data. It became a small study of how **data
source × geometry × optimizer** interact — and why the same tensor is ill-posed to fit from one source and
trivial from another. Useful input for the G3 optimizer module. Companion to
[`mcsgd_apply_derivatives.md`](mcsgd_apply_derivatives.md); G3 plan in
[`geometry_refactor_plan.md`](geometry_refactor_plan.md).*

## TL;DR

- We added a **corewise + entries + scipy-L-BFGS-B** completion example, driven through the library's
  flat-vector bridge (`TuckerTensorTrain.to_vector`/`from_vector`; the corewise gradient flattens into the
  *same* coordinates as the point). The library stays dependency-free — scipy is imported only in the example.
- **Entries are a weak, localized source.** A coherent tensor (Hilbert: mass at the low-index corner) is
  badly under-determined from a sparse uniform entry sample — it needs **~10× more samples than applies**,
  and is ill-posed below ~48% of the tensor for a hard solver.
- **The optimizer/geometry is the regularizer.** A hard second-order solver (manifold Newton-CG) **overfits
  catastrophically** at sparse sampling (true error several × the tensor norm — it matches the observed
  entries and blows up the unobserved corner). **Corewise L-BFGS and manifold MC-SGD both recover** a
  sensible completion at the same sparse sample (true error a few %), via implicit low-norm bias. For
  L-BFGS this is a **converged** property, not early stopping (10× more iterations does not change it).
- **Corewise can't warm-start.** Zero-start fails (`J=0` at all-zero cores), and zero-padded continuation
  freezes the new rank block at a vanishing-Jacobian saddle — so corewise uses a **nonzero random init**
  and an **independent cold fit per rank level**, with validation selecting the rank. **Warm-started rank
  continuation is a genuine advantage of the manifold geometry** (an angle we hadn't appreciated before).
- **Probes are the well-conditioned counterpoint.** Dense random probe vectors make a *global* measurement,
  so the same tensor fits cleanly and **monotonically** with rank — no overfitting in range. *Same tensor,
  same geometry; the localized (entries) source overfits catastrophically, the global (probes) source not
  at all.*
- **Optimizer bake-off on probes (rank → 10).** Newton-CG and MC-SGD tie (true ~0.002, both overfit
  *gently* at L4); converged Adam trails (~0.004, overfits only at L9); the example's under-converged Adam
  stays monotone. Cold-start ≡ warm-start for Newton-CG here (start-independent on a well-conditioned
  problem). The **tuning-free Cauchy step (MC-SGD) matching the hard solver** is the headline for G3.

## 1. What we built

`examples/fit_hilbert_from_entries_lbfgs.py`: observe the Hilbert tensor (`shape (12,)*4`, 20 736 entries)
at a sparse random set of entries (3000 train + 2000 validation ≈ 24%), 1% noise; complete it by fitting a
fixed-rank `TuckerTensorTrain` on the **corewise** geometry, minimizing `½‖X.entries(idx) − y‖²` with
**scipy `L-BFGS-B`**. Each scipy evaluation rebuilds the point from the flat core vector, builds a corewise
`fitting.entries_model`, and returns `(objective, corewise-gradient-flattened)`. Rank by validation. Runs
in ~40 s. Status: **committed, clean.**

## 2. Entries are a weak, localized source (data-source conditioning)

An entry measurement's "row" is a one-hot rank-1 tensor (unit norm already — no row-normalization needed,
unlike applies/probes), so each measurement constrains a single coefficient. The Hilbert tensor is
**coherent**: its energy concentrates at the low-index corner (`A[0,0,0,0]=1`, decaying), which uniform
random sampling rarely hits — so the model is under-constrained exactly where the tensor is large.

Sampling threshold, from the **manifold Newton-CG gold standard** (best level by validation, true error):

| train entries | observed | best level | true error | verdict |
|---|---|---|---|---|
| 4000 | 29% | 1 | 1.8e-1 | ill-posed (val forces rank 1) |
| 8000 | **48%** | 4 | **7.0e-3** | well-posed (≈ noise floor) |
| 12000 | 68% | 4 | 5.2e-3 | |
| 16000 | 87% | 4 | 3.0e-3 | |

So entries completion of this coherent tensor needs **~48%** of the tensor to be well-posed for a hard
solver — versus the apply example, which recovers from **800 applies** (a few % of the entry count). *Data
source ↔ conditioning* is a real axis: applies (global, dense functionals) are sample-efficient; entries
(local, one-hot) are not.

## 3. Hard solvers overfit; SGD / over-parametrized L-BFGS regularize

At a sparse **24% (3000 train)** sample, the three optimizers diverge sharply (best true error by validation):

| optimizer | geometry | true error | what happens |
|---|---|---|---|
| Newton-CG | manifold (full 2nd-order) | **~5.9** at rank ≥ 3 → val falls back to rank 1 (0.18) | **overfits** — fits observed entries, blows up the unobserved corner |
| MC-SGD | manifold (stochastic Cauchy) | **0.063** | regularizes (stochastic low-norm bias) |
| corewise L-BFGS | corewise (over-parametrized) | **0.067** | regularizes (small-init low-norm bias) |

So on this ill-posed problem **the optimizer choice *is* the regularizer.** The hard Gauss-Newton solver
converges hard to a min-training-misfit solution that is wildly wrong off-sample; the stochastic and the
over-parametrized-small-init methods are biased toward low-norm solutions (the implicit regularization
studied in over-parametrized matrix/tensor factorization).

**The L-BFGS regularization is genuine, not early stopping.** Running it to near-full convergence vs the
iteration cap gives the same bounded result — **best true error 0.0544 (maxiter 20000, gtol 1e-12) vs
0.0554 (maxiter 2000)** — no drift toward the Newton-CG blowup. The over-parametrized corewise landscape
settles into a low-norm minimum.

At 48% (well-posed) all three are comparable (Newton-CG 0.007, MC-SGD 0.012, L-BFGS 0.017) — the
regularization only *matters* in the sparse regime. The example features 24% precisely so the effect shows.

## 4. Probes — the well-conditioned counterpoint, and an optimizer bake-off

The fourth example ([`fit_hilbert_from_probes_adam.py`](../examples/fit_hilbert_from_probes_adam.py),
corewise + a hand-written Adam) fits the same tensor from **probes** (dense random probe vectors, one mode
left free per measurement). Probes are **global, well-conditioned** — the opposite of entries — so the fit
is well-posed: the table drops **monotonically** with rank, no overfitting in range (validation just keeps
picking the largest rank). That contrast is the headline: *same tensor, same geometry; the localized
source overfits catastrophically, the global source not at all.*

To characterize the optimizers we ran a **bake-off** on identical probe data (200 train / 100 val probes,
1% noise), pushing rank to 10 — best true error by validation, and where the validation curve turns over
(= overfitting onset):

| method | start | converged | best lvl | best true | overfit onset |
|---|---|---|---|---|---|
| Newton-CG | warm | ✓ | 4 | **0.0018** | L4 |
| Newton-CG | cold (zeros) | ✓ | 4 | **0.0018** | L4 |
| MC-SGD | warm | ✓ (stoch) | 4 | **0.0025** | L4 |
| Adam | cold | ✓ (5000 it) | 9 | 0.0037 | L9 |
| Adam | cold | ✗ (1200 it, the example) | 6 | 0.017 | — (monotone) |

Five things fall out:

1. **Newton-type methods overfit at high rank — but gently.** Train keeps dropping while val/true turn up
   after level 4. Unlike the entries blowup (true 5.9), this is a mild, validation-detectable turnover (val
   0.0089 → 0.011 over L4→L10), *because the data is well-conditioned.* Overfitting is real for the hard
   solver, but its **severity is set by the data source**, not the optimizer.
2. **Start strategy doesn't matter for Newton-CG here.** Cold-started (zeros at each rank, no continuation)
   is **line-for-line identical** to warm-start — same 0.0018, same turnover. On a well-conditioned problem
   Newton-CG finds the same minimum regardless of start, so warm-start continuation buys **robustness on
   hard problems, not regularization here**. (Caveat: this is the *structured* zero-start; a random cold
   start is a separate question — it tends to hit bad local minima rather than overfit.)
3. **MC-SGD matches Newton-CG.** The tuning-free Cauchy step reaches 0.0025 (vs 0.0018), same level 4, same
   gentle turnover — at ~40–70 cheap iters/level. The first-order *stochastic* method essentially matches
   the hard second-order solver on this well-conditioned problem.
4. **Adam's "no overfit" was mostly under-convergence — but not entirely.** Run to convergence (5000 vs the
   example's 1200 iters) Adam reaches 0.0037 and *does* finally overfit, but ~5 ranks **later** than
   Newton-CG (L9 vs L4). That lateness is genuine implicit regularization (over-parametrized small-init
   bias) on top of the early-stopping effect.
5. **First-order trails second-order on accuracy.** Newton-CG/MC-SGD reach ~0.002; converged Adam ~0.004;
   under-converged Adam ~0.017. Curvature-aware steps reach high accuracy cheaply *when the problem is
   well-conditioned*; Adam's per-coordinate adaptive step is slower to the last digits and needs a learning
   rate + schedule (not tuning-free).

## 5. Corewise optimization mechanics (init + rank continuation)

Three corewise-specific facts, each a genuine contrast with the manifold geometry:

- **Zero-start fails (`J=0` at the origin).** The manifold path starts from the *zero* tensor:
  `t3_orthogonal_representations` completes the rank-deficient frame with orthonormal directions, so the
  Jacobian there is nonzero. On the **raw cores** there is no such completion — at all-zero cores every
  single-core swap multiplies in a zero core, so `J=0`, the gradient vanishes, and L-BFGS cannot move.
  Corewise needs a **nonzero start**: a small random tensor, rescaled (every core by `scale**(1/2d)`, which
  scales the multilinear tensor by `scale` while keeping the cores balanced) so the initial prediction
  matches the data magnitude.
- **Warm-started continuation freezes (the saddle).** The manifold examples warm-start each rank level from
  the zero-padded converged previous solution — robust, clean, monotone tables. Corewise **cannot**: a
  zero-padded warm start leaves the new rank block at zero, where (same vanishing-Jacobian argument) its
  gradient is exactly zero, so L-BFGS never grows it — every higher level just reproduces the lower one. A
  small random nudge to the new block was not enough to escape the saddle in practice.
- **So: cold independent fit per level + validation selection.** Each rank level is a fresh cold random
  fit. This is noisier across levels (a level can land in a worse local minimum than its neighbour — the
  occasional bump in the table; validation still selects a good level). We deliberately did **not** add
  restart-based polish.

**Lesson:** warm-started rank continuation is a real, previously-underappreciated advantage of the
**manifold** methods; the over-parametrized corewise chart pays for its flat additive retraction with a
degenerate continuation path.

## 6. The scipy bridge (ecosystem integration, library stays dependency-free)

The crux is layout consistency: `point.to_vector()` and `model.gradient.to_vector()` both route through the
one `backend.t3_operations.t3_to_vector`, so they live in the **same** `R^n` coordinates — `from_vector`
rebuilds the point, the corewise gradient is the flat `Jᵀr` in the matching layout, and
`scipy.optimize.minimize(..., jac=True)` "just works". This is the reusable recipe for driving a T3 fit
from *any* flat-array optimizer; scipy is imported only in the example, so the main library keeps no
scipy/optax dependency (per the own-vs-external decision: own Adam/Newton-CG/MC-SGD; bridge to scipy for
L-BFGS, where the value is its battle-tested Wolfe line search).

## 7. Architecture lessons for G3

- **The model hooks are the right optimizer interface.** Every loop here — MC-SGD, the Cauchy bake-off,
  Adam — was driven purely by `model.gradient` + `model.gn_quadratic` (the Cauchy step) + `geometry.retract`,
  with no inline closures. The fitting-model surface is sufficient for first-order, stochastic, and
  quasi-Newton optimizers alike.
- **MC-SGD (tuning-free Cauchy step) is the workhorse.** It matched manifold Newton-CG on the
  well-conditioned probe fit (0.0025 vs 0.0018) at a fraction of the cost, and regularizes the ill-posed
  entries fit — strong evidence to make it a first-class `optimizers.py` citizen, not a footnote.
- **Adam is a *corewise option*, not the default.** It needs a learning rate **and** a schedule (cosine
  decay was essential), is slower to high accuracy, and only its under-converged form avoids overfitting.
  Useful as the dependency-free over-parametrized first-order method, but it does not displace MC-SGD/Newton.
- **Three orthogonal axes the optimizer module must respect:** *data source ↔ conditioning* (entries weak,
  probes/applies strong), *geometry ↔ continuation* (manifold warm-starts, corewise cold-starts), *optimizer
  ↔ regularization* (hard solvers overfit ill-posed fits; stochastic / over-parametrized / under-converged
  regularize). **Overfitting severity is data-source-set**, so validation rank selection is essential and —
  for well-conditioned sources — works cleanly off a gentle turnover.
- **Corewise defaults differ from manifold:** nonzero init (not zero), cold per-level (not warm
  continuation). A "geometry-agnostic" `optimizers.py` must still let these init/continuation policies vary
  by geometry.

## 8. Open questions / future work

- **Disentangle the regularization source.** We saw *hard manifold* overfit and *over-parametrized
  corewise / stochastic manifold* regularize, but did not cleanly separate **optimizer** (L-BFGS/SGD
  implicit bias) from **geometry** (over-parametrization). The missing cells: does a *corewise Newton* (hard,
  over-parametrized) overfit, and does a *manifold L-BFGS* (quasi-Newton, minimal frame) regularize? Filling
  the 2×2 would tell us whether the cure is the chart or the step rule.
- **Coherent-tensor sampling.** Uniform entry sampling is weak for a coherent tensor; leverage/importance
  sampling (bias toward the high-energy corner) should cut the ~48% requirement substantially. Worth a note
  for users completing coherent tensors.
- **A non-degenerate corewise warm start.** Is there a continuation that activates the new rank block
  without the saddle freeze (e.g. re-orthogonalize between levels, or seed the new block by fitting the
  current residual)? Would restore warm-start continuation for corewise.
- **Restart-based robustness.** Best-of-K cold restarts per level would smooth the table; we skipped it as
  polish. Quantify the variance / how many restarts buy a monotone table.
- **At scale.** Everything here is the toy `12^4` at a single seed. Whether the sample thresholds and the
  regularization margins hold at realistic scale is unproven (the recurring caveat).
- **Recoverable accuracy vs sample fraction.** The completion bottoms out a few % off at 24% and ~0.7% at
  48%; the scaling of recoverable accuracy with sample fraction (and rank, and coherence) is unmapped.

## 9. Pointers

- Entries example: [`examples/fit_hilbert_from_entries_lbfgs.py`](../examples/fit_hilbert_from_entries_lbfgs.py)
  (`corewise_lbfgs`, `random_start`).
- Probes example: [`examples/fit_hilbert_from_probes_adam.py`](../examples/fit_hilbert_from_probes_adam.py)
  (`corewise_adam`).
- The manifold companion (apply data, Newton-CG): [`examples/fit_hilbert_tensor_newton_cg.py`](../examples/fit_hilbert_tensor_newton_cg.py).
- MC-SGD method + its own findings: [`mcsgd_apply_derivatives.md`](mcsgd_apply_derivatives.md).
- The geometry abstraction (`MANIFOLD`/`COREWISE`, the model hooks): [`geometry_refactor_plan.md`](geometry_refactor_plan.md).
- The three sampling ops (entries/apply/probe) and their costs: [`entries_apply_probe.md`](entries_apply_probe.md).
