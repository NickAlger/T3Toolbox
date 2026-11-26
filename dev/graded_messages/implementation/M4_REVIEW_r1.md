SIGN OFF

# M4 review, round 1: the transposes typed

Reviewer (the sub-arc's second), 2026-11-23. Reviewed: the uncommitted M4 on `graded-messages` (over `3d0ed1a3`),
together with `M4_HANDOVER.md`, against plan §5 (decision Q2), the M4 row of §7, and `M_ARC_DECISIONS.md` 24.

My scratch work is in `scratchpad/m2rev2/`:
- `check_m4_tangent.py`: the drawn-batch gradient on the tangent transposes;
- `gate_m4/`: the gate logs (Sphinx output deleted);
- the round-trip copies, deleted.

**Verdict: SIGN OFF.**
- I regenerated the scripted rename from HEAD myself, and its round trip is byte for byte.
- The prose follows Q2.
- Every `c` line carries the right type.
- The law tests are independent and have teeth.
- The drawn-batch gradient holds on the routes I checked independently.
- I ran the gate steps that the implementer did not run, and all pass.

Two minor points follow.

## 1. The round trip and the regeneration

Built on two fresh `git archive`s of HEAD (`t3toolbox`, `tests`, `examples`), with the committed `m4/rename.py` and
`m4/prose.py` copied in.

**The rename.**
- `rename.py forward` on HEAD writes an edit log **identical** to the committed `m4/edit_log.tsv` (`cmp`; 275 edits).
- It prints the three pre-existing `c` sites in `t3_apply_ambient_transpose`. I read them: the local
  `c = xnp.asarray(residual)` becomes `c = xnp.asarray(c)`, which is correct.
- `inverse` on the forward result reverts 275 edits and gives HEAD back byte for byte under `t3toolbox/` and `tests/`
  (`diff -r`).

**The prose.** `prose.py` applied after the forward run reproduces the working tree exactly in `t3toolbox/` and
`tests/`. The only differences are the three hand edits that are not scripted, and I read each:
- the new test file;
- `tests/_typing_laws.py` (the new movers);
- the two examples' two comment lines.

So the diff a reader must review by eye is exactly `hand.diff`, plus those hand edits.

## 2. The prose against Q2 and §5

**The argument and its description.** Every one of the 42 takes `c`; no docstring of the 42 mentions `residual`
(AST scan). Each states, or points to one that states, the plan's text:
- "an element of `Y` … paired with `z` slot by slot";
- "carries no metric, and none is applied here";
- for the gradient of a weighted least-squares fit, pass `data_metric(r, weight, minibatch_scale)` or
  `batch.data_metric(r)`.

**`J^T r` → `J^T c`,** with "the Gauss-Newton gradient `J^T(M r)` when `c = data_metric(...)`", at the plan's sites
and at the two `project` docstrings ("a bare adjoint `J^T c` (with `c = M r`: the misfit's raw gradient)").

**Elsewhere.**
- The remaining `J^T r` mentions are in `sampling_derivatives.py`, the old layer, which is left alone by design.
- The corewise Hessians define `phi(x) = <c, z(x)>` with "the pairing of `Y` with `Y*`, slot by slot: no metric".
  That is the right object: the corewise transpose is `∇phi`, and the Hessian its derivative.
- P13's right and wrong adjoint doctests keep their code, and their prose now says `(c, ww)`.

**The examples' comment** is correct, and I checked it against plan §1:
- the measurement law is `r_d[k] = k! r_t[k]`;
- so a `c` in derivative coordinates converts by `taylor_to_derivatives`, `c_t[k] = k! c_d[k]`;
- the code (`apply_transpose(frame, jet, taylor_to_derivatives(r / sv))`) does exactly that.

## 3. The type on each `c` line (all 42, by AST)

| Family | Type on the `c` line |
| --- | --- |
| probe: `tv_`, `t3_ … corewise_{transpose,hessian}`, the frontend `T3Tangent`, `TuckerTensorTrain` (ragged) | `P*⊗V_i, len=d, elm_shape=F+W+[K]+C+(Ni,)` (the Hessians without `[K]`) |
| probe, uniform | the same, with "; or packed" |
| apply (all forms) | `P*, shape=F+W+[K]+C` (the Hessians without `[K]`) |
| entries (all forms) | `ℝ, shape=W+[K]+C` (the Hessians without `[K]`) |
| **ambient**: probe (backend and frontend) | `V_i, len=d, elm_shape=W+C+(Ni,)` |
| **ambient**: apply, entries (backend and frontend) | `ℝ, shape=W+C` |

- **The ambient ones are right.** They take no `order`, so `P* = ℝ`, and the probe element `P*⊗V_i` is then `V_i`
  (plan §9.0's table at `F = ()`). Their output is `E*`, CP factors with `c_i` in slot `i`, which the docstrings say.
- **The shapes are unchanged.** Comparing old and new lines: only the type replaced "of the type of z" (and
  "; no weight").

## 4. The law tests: independence and teeth

**The movers.** They live in `tests/_typing_laws.py`, and none calls the library's transposes:
- `frame`;
- `variations_dual`;
- `packed_mode_dual`;
- `entries_dual`;
- `ambient` (the dense tensor times `⊗Δ_i`, applied to the CP output summed by the test's own `einsum`).

**The input law** runs on every one of the 42. Each moves the right way:
- for probe, `c` by `Δ_i`;
- for entries, `c` by `∏Δ_i[n_i]` (the index fixed, as a basis vector);
- the cores and the Hessian's directions by `Δ_i^-1`;
- the `X*` blocks by `Δ_i`.

The moved frame is not orthonormal, so the frontend calls run under `safety.unsafe()`. That is right: the transposes
are metric-free.

**The jet law** runs on the 24 with jets, at `(2,)` and `(1,1)`, with `ℓ` distinct per variable, as plan 9.3 requires.

**Teeth.** The implementer's mutants cover the faults the laws are designed to catch:
- `c` moved as `V_i*` (the probe residual where `c` is meant);
- `c` moved as a jet;
- the seed not flipped;
- the ambient factor on one mode;
- a library scale on the apply ambient transpose.

**What the laws cannot catch** is stated in the file: they are homogeneity tests, blind to a constant factor and to the
`(c, ww)` swap. That is plan 9.3. The adjoint doctests and the gradient check below are the primary guards.

## 5. The drawn-batch check

**The implementer's test.** `TestTheGradientOfAWeightedFit` checks the frontend corewise transposes on a `take`-drawn
batch (`γ = 2`) with a per-entry weight including zeros:
- `<J^T(batch.data_metric(r)), ẋ>` against a central difference of `γ · oracle.misfit_full` (dense, independent),
  one core at a time;
- exact, since the misfit is quadratic along one core;
- 18 subtests to 1e-9.

`r` comes from the library's forward, but a wrong forward would break the agreement with the dense misfit, so the
check is not circular. The implementer's mutant (`r` passed in place of `batch.data_metric(r)`) fails it.

**My complement (`check_m4_tangent.py`): the tangent transposes.** `T3Tangent.*_transpose(frame, s,
batch.data_metric(r))` paired with a random `p` (`corewise_inner`), against the central difference of the dense
`½ γ Σ ω²(S(X + tP) − y)²` with `P = p.to_dense()`.
- Cases: probe at `(2,)` and `()`, apply at `(2,)` and `(1,)`, entries.
- The batch is drawn (`γ = 7/3`), with a per-entry weight including zeros.
- 15 checks; the worst relative difference is 3.4e-12.

## 6. The gate

**The suite and the tally.**
- The implementer's suite log (`m4impl/gate/suite_v.log`) is newer than every changed file: 1,965 passed.
- My own tally against M3's log: 1,966 → 1,971 nodes, and exactly the 5 new `test_transposes_typed` nodes differ (42,
  42, 48 and 18 subtests, plus one plain). Every renamed keyword call keeps its outcome and its count.

**The steps not run before, run by me now** (`run_gate.sh -n 2 doctest-modules doctest-quickstart doctest-pages
examples sphinx`, one after another; logs in `scratchpad/m2rev2/gate_m4/`):

| Step | Result | Seconds | Lowest MemAvailable (MB) |
| --- | --- | --- | --- |
| module doctests | PASS, 325 passed | 56 | 10,208 |
| `getting_started.rst` | PASS, 0 failed | 3 | 11,484 |
| pages (47 files) | PASS, 0 failed | 19 | 11,434 |
| examples | PASS, 14, 0 failed (the two edited ones included) | 570 | 11,143 |
| Sphinx `-W` | PASS, 0 warning or error lines | 1,012 | 11,032 |

So the reworded docstrings and the two edited examples are now tested.

## Minor

1. **The corewise Hessians' parameter text** reads "for the **gradient** of a weighted least-squares fit pass
   `data_metric(r, …)`". For a second-order transpose, say "for the curvature term of a weighted least-squares fit
   (`Kp` at `c = M r`), pass …", as plan §1 writes it. That is one sentence in each of the six Hessian docstrings.
   It can go in M5, which edits these signatures anyway.
2. **A comment column one off.** In nine backend signatures (`tv_`, `t3_ … corewise_transpose` and
   `t3_ … corewise_hessian` in `apply.py`, `entries.py` and `probing.py`), the `#` of the `c` line is one column
   right of its neighbours'.
   - It was already so on the `residual` line at HEAD, not introduced here.
   - M4 realigned the type column but not this one.
   - M5 rewrites these comments; realign them there.

Nothing is left running. Sphinx's output and my round-trip copies are deleted. The working tree is as I found it,
apart from this file.
