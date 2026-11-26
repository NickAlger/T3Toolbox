# Review of `DATA_METRIC_PLAN.md` (first draft), round 1

Reviewer, 2026-11-21. Read only; checked against the tree at `aaee0b04` (`graded-messages`). Two light scripts were
run (findings 5 and 6). `file:line` refers to that tree.

## Verdict: NOT YET

The direction is right. Decision 1 is T-a of the audit (the transposes take `c` in `Y`, and the weight lives in the
objective). Decision 2 is R-1 widened to elementwise. The typing in section 1 is sound, and the flip really is a
change of representation. But the plan is a sketch where the design is hardest:

- **The weight's reading rule** is the central object of decision 2, and the plan leaves it as a question.
- **The public `W`** has a signature that cannot place its weight on a residual that carries the `K`/`C` stacks. It
  omits the minibatch scale. As drafted, it cannot live where the plan puts it (an import cycle).
- **Three silent wrong answers** wait in the conversions the plan lists as one-liners: `prepare_derivative_data`'s
  factorials, `pack_batch`'s padding, and `take`.
- **One claim is too strong.** The derivative-coordinates trap "cannot arise", yet two examples in the repository
  run into it.
- **The blast radius is undercounted.** The reference module, the dense fitting oracle and `data_preparation` are
  missing, the function count is 18 where it should be 21 + 21, and two earlier decisions of the maintainer are
  reversed without being named.

Findings 1 to 3 are blocking. The rest are should-fix or minor.

---

## The typing (what holds)

- **Slot-by-slot pairing on Taylor coefficients: the right identification.**
  - It is the Euclidean pairing in the library's chosen data coordinates. That is the dual basis, with no metric in
    it.
  - It is the pairing the adjoint doctests already use. The audit's section 3.0 checks it at `(2,)` and `(1, 1)`.
  - `jet_to_dual_jet` is `flip ∘ W` (`jets.py:413-420`), and the flip only turns the pairing into a corner
    coefficient.
  - What matters is that the pairing is fixed by the coordinates, never by the weight. The plan says so.
- **"Keep all data in Taylor coordinates" holds for the library's own outputs, but not for every public entry
  point.** See finding 4.

---

## Findings

### 1. BLOCKING: the reading rule of an elementwise weight is unspecified, and the obvious one is wrong

**The gap.** Plan §2.A says the weight "broadcasts to the shape of `y`" and calls the old block forms broadcast cases,
written as `F + (1,)*len(W) + (1,)`. It then leaves open how the old forms are read ("unambiguous by rank?", Q2).

**Evidence.**
- `y`'s layout puts `F` **first**: apply `F+W` (`fitting.py:769`), probe `F+W+(Ni,)` per mode.
- The block forms are therefore **left-aligned** (prefix) broadcasts. The engine already reads them that way:
  `weight_F.reshape(weight_F.shape + (1,)*(ndim - m))` (`sampling_steps.py:970,1053`; `jets.py:290`).
- numpy's own broadcasting is **right-aligned**. Read as numpy broadcasting, the apply weight `F = (3,)` at
  `order=(2,)` with `W = (3,)` would weigh the samples, not the slots, silently.
- **The stacks sit between `W` and the mode edge.** The engine weighs `r` of shape `F+W+C+(Ni,)` (`r_step`,
  `sampling_steps.py:1006-1016`) and `Jp` of shape `F+W+K+C+(Ni,)`. A probe weight that carries the mode edge needs
  ones **inserted before the last axis**, not appended. Appending them, as `_squared_weights` and `_modes` do
  today, misaligns as soon as `C` or `K` is non-empty.
- **Rank alone is ambiguous for probe.** At `m = 0`, a weight of rank 2 reads as `W = (W1, W2)` with no edge, or as
  `W = (W1,)` plus the edge, unless the engine is told `len(W)`.

**What would resolve it: one stated rule.** The reviewer recommends this one, because it keeps every old form
valid with no conversion:

- A weight is read **per array of `y`** (probe: per mode, a tuple, or the `d`-leading packed array, as `jets._arrays`
  already reads it).
- Each array has one of two forms:
  - the **block form**: rank `m`, the jet axes `F` only;
  - the **y-shaped form**: exactly `y`'s rank (apply `F+W`, entries `W`, probe `F+W+(Ni,)`, the edge always present).
- In both forms each axis is either 1 or `y`'s size.
- Then:
  - nothing else is legal;
  - the old forms `(d,)+F`, `F` and `()` are the block form, read as today;
  - the rank alone tells the two forms apart, with no `len_W` needed (probe: rank `> m` means the edge is there);
  - stacks are inserted before the edge (probe) or appended (apply, entries).
- `check_weight` checks the rule against `y`, so it needs `y` (or `sample`): a signature change (finding 9).
- The trap that remains, to document like the existing transposed-square-weight trap (`jets.py:1248-1251`): a
  per-sample apply weight written as a bare `W` with `len(W) == m` reads as the block form if the sizes coincide.

This settles Q2 (accepted, by this rule) and decides how `W` places a weight on a stacked `r` (finding 2).

### 2. BLOCKING: the public `W`'s signature, scale and home

**(a) Stacks.**
- `weigh(kind, order, r, weight)` must accept what the transposes take: `F+W+[K]+C(+Ni)`
  (`probing.py:1033`; the oracle's optional `r` carries `C`).
- Under finding 1's rule it can, with no lengths passed. Under the draft it cannot.
- State it in the signature comment, and test `W` at `K ≠ ()` and `C ≠ ()`.

**(b) The minibatch scale is part of `W`.**
- O-21 puts the scale **in the batch's data inner product**: `<a, b> = scale * <a, omega^2 b>`
  (`TARGET_DECISIONS.md:371-380`; `fitting.py:938-944`).
- A `W` that returns `omega^2 r` therefore makes M2's own test, "`J^T (W r)` equals the oracle's `g`", false on
  every drawn batch.
- Either `W` takes the batch (`data_metric(kind, order, batch, r)` = `scale * omega^2 * r`), or it takes
  `minibatch_scale=` explicitly.
- Recommend the batch form: the objective's data are the batch, and that is the objective-dependence line.
- The M2 test must cover a drawn batch.

**(c) Home.**
- `backend/fitting.py` imports `sampling_variables` (`fitting.py:37`), which imports `sampling_steps`
  (`sampling_variables.py:320`).
- If the engine's seed calls a `W` defined in `backend/fitting.py`, the import graph has a cycle.
- The one implementation must live at or below `jets.py`, where `jet_to_dual_jet` already lives. `backend/fitting.py`
  and `t3toolbox.fitting` re-export it.
- The plan should say where it is defined and where it is exported.

**(d) "One implementation" is larger than the seed.** `omega^2` is applied in six places:

- `w_tilde_step` (through `jet_to_dual_jet`);
- `f_step` (`sampling_steps.py:1052-1053`);
- `Jp_norm_sq_step` (`:969-970`);
- `f_resolution` (`fitting.py:935`);
- `dual_jet_to_jet` (`jets.py:498-507`; `W^{-1}`);
- the frontend safe-mode zero check (`fitting.py:275`, through `_squared_weights`).

Every one of them must go through **one private alignment helper**: the weight broadcast to a value under finding 1's
rule. The public `W` is that helper times the scale. `f = 1/2 <r, W r>`, `Jp_norm_sq = <Jp, W Jp>` and
`f_resolution`'s `|omega y|^2 = <y, W y>` then share it.

### 3. BLOCKING: the questions have no recommendations, and the real open choices are missing

- **Q1, the names.** Give a recommendation with a reason.
  - `c` is the pre-N16 apply/entries name, and the Hessian docstrings already say `c`.
  - `covector` is self-documenting in keyword calls.
  - Note: `dual` collides with "dual jet", the turned form (D3). That is an argument against it.
- **Q2.** Answered by finding 1. Give it as the recommendation.
- **Q3 (`omega` or `omega^2`).** Recommend keeping `omega`:
  - `equalizing_weight` returns `1/norm` (`data_preparation.py:203`);
  - `prepare_derivative_data` multiplies `omega` by `k!`, not `(k!)^2` (`jets.py:657-659`);
  - a noise level gives `omega = 1/sigma`;
  - a zero weight already drops a block;
  - the display, the docs and P16's pinned translation all speak `omega`.
- **Missing questions (each needs a recommendation):**
  - (i) The display semantics. See finding 8.
  - (ii) `W` takes the batch or a scale. See finding 2b.
  - (iii) The stacked `y`/weight with `C`. LOG:1240 flags it "to discuss with the data-space design". Recommend: out
    of scope, CHANGELOG line, stated in the plan.
  - (iv) The reversal of the exact-shape rule. See finding 9.

### 4. SHOULD FIX: "the trap D4 cannot arise" is too strong

**The plan's claim.** §1: "Derivative data enter only through `prepare_derivative_data` ... no element of `Y` is
ever held in derivative coordinates".

**Why it is false.** That holds for the library's outputs. Users hold `Y` elements in derivative coordinates
whenever they compose an objective by hand with the public converters:

- `examples/fit_hilbert_from_apply_derivatives.py:142-144` does it: a derivative-coordinate forward
  (`taylor_to_derivatives(...)/sv`), whose transpose correctly converts the covector with `taylor_to_derivatives`.
  That is the contragredient direction the audit derived (D4).
- `examples/fit_hilbert_from_apply_derivatives_flat.py:137-139` does the same.

These examples stay after P17. The same class of trap covers probe data under `rescale_coordinates` (covectors of
`V_i'`, audit D6).

**What would resolve it.**
- Restate the claim: the library never produces a `Y` element in derivative coordinates, and a hand-built objective
  in other data coordinates converts its covector the contragredient way.
- Put that sentence in `W`'s docstring and in `derivatives_to_taylor` / `taylor_to_derivatives`.
- Either rewrite the two examples onto `prepare_derivative_data` plus `W`, or keep them and cite the rule there.
- Name the choice (document, or the audit's optional `covector_derivatives_to_taylor`) with a recommendation.

### 5. SHOULD FIX: `prepare_derivative_data` and `derivative_weight_to_taylor_weight` put the factorials on the wrong axes for any new form

**Evidence.**
- `_taylor_weight` multiplies by `_factorials(order)` with numpy's right-aligned broadcasting (`jets.py:255-263`).
  Its docstring states "the jet axes are the last `len(order)` axes of the weight for every kind" (`jets.py:660`).
- In a y-shaped weight the jet axes come first (after the mode). Run: an apply weight `ones((3, 3))` meant as
  `F+W` at `order=(2,)` comes back `[[1,1,2]]*3`. The factorials land on the samples, silently. They should be
  `[[1,1,1],[1,1,1],[2,2,2]]`.
- `derivative_weight_to_taylor_weight(weight, order)` has **no `kind`**, so it cannot tell a leading mode axis from a
  leading jet axis. It needs `kind` (and finding 1's rule), a public signature change.
- A block form with a size-1 jet axis must come back with full `F`.

**What would resolve it.** Specify this in the slice that introduces the rule (M1, not M3). Test it against a hand
conversion with distinct `F` and `W` sizes **and with equal sizes**, the case that hides the bug.

### 6. SHOULD FIX: uniform padding. `pack_batch` would corrupt a broadcast weight, and the engine does not mask the weight

**Evidence.**
- `pack_if_ragged` pads a size-1 mode edge to `N` with zeros. Run: per-mode weights of shape `(3, 1)` packed to
  `N = 5` come back `[2, 0, 0, 0, 0]` per row, so the weight is now on entry 0 only. That is a silent wrong answer
  if "pack with zero" is done naively.
- The engine masks `y` on entry (`_UNIFORM_MASKS['y']`, `sampling_variables.py:1366`), but `weight` is not in that
  table. A user-packed weight with a full `N` axis and garbage (nan) padding gives `nan * 0 = nan` in `f` and `g`.
- `testing_strategy.md` requires robustness to garbage-padded inputs.

**What would resolve it.**
- `pack_batch` packs only a weight whose edge has full size, and leaves a size-1 axis unpadded.
- Add `'weight': (('N', None),)` to `_UNIFORM_MASKS`, applied only when the edge has full size. With that,
  `pack_batch`'s padding value no longer matters.
- `f_resolution`'s `|omega y|` on packed probe data unpacks or masks the weight as it does `y`
  (`fitting.py:928-929`).
- Tests: a garbage-padded weight; a size-1-edge weight on the packed layout against ragged.

### 7. SHOULD FIX: `take` must reshape the weight's `W` axes even when nothing is gathered

**Evidence.**
- `take` flattens `W` to one axis `(b,)` (`fitting.py:980-1021`), and today it passes the weight through.
- A y-shaped weight has `len(W)` axes. After the draw it must have one, even when every `W` axis is 1, else the
  drawn batch fails finding 1's rule against the drawn `y`.
- A weight that is partly broadcast (`W = (5, 7)`, weight `W`-part `(5, 1)`) cannot be flat-gathered until it is
  broadcast to the full `W`.

**What would resolve it.** Specify the order:
1. broadcast the weight's `W` part to the full `W`, only when it is present and some axis is not 1;
2. flatten;
3. gather, with the `W` start at `m` (ragged) or `1+m` (packed);
4. an all-ones `W` part becomes `(1,)`.

Under jit `b` is static, so this stays shape-static. Test a partly broadcast weight in the exact-enumeration
unbiasedness test.

### 8. SHOULD FIX: the display changes meaning, and the plan says it does not

**Evidence.**
- With block weights the per-block ratio `‖r_b‖/‖y_b‖` is invariant under a block-constant weight (audit D9;
  `optimizer_display.py:27-30`): the display's error **was** the weighted per-block error.
- With elementwise weights it no longer is. A sample dropped with weight 0 (an outlier, missing data) still
  dominates the unweighted table while the fit ignores it.
- The docstrings "a block is an entry of the weight, so the errors have the weight's shape"
  (`optimizer_display.py:29-30, 165, 423, 584`; `data_preparation.py:137`; `sampling_steps.py:1068`) and the error
  message at `optimizer_display.py:186` become false either way: the block shape is no longer the weight's shape.

**What would resolve it.** A question for the maintainer, with a recommendation:

- (a) Keep it unweighted (O-21/O-42 as decided) and document the change.
- (b) Report `‖omega⊙r_b‖/‖omega⊙y_b‖`. That equals today's numbers for every block-form weight, and it respects
  masks. It needs weighted block sums as new optional oracle outputs.

The reviewer leans to (b). Either way, rename "the weight's shape" to "the block shape `(d,)+F | F | ()`" throughout.

### 9. SHOULD FIX: two decisions of the maintainer are reversed, and the plan does not say so

**Evidence.**
- 2.5 and 6.7 decided **"exact shape only"**, with no dimension of 1 broadcast (`TARGET_DECISIONS.md:1326-1332,
  2307-2317`), and `check_weight` enforces it (`jets.py:1291-1298`).
- `tests/test_jets.py:847-877` asserts the rejections. Under finding 1's rule several become legal (`'a jet dimension
  of 1'` for apply, for instance), while others must still raise. Each case of that table needs a ruling.
- The engine's stack check exists because "a size-1 axis would broadcast into a silent wrong answer"
  (`TARGET_DECISIONS.md:758-761`, "strongly agree"). For the weight, a size-1 axis is now the intended meaning.

**What would resolve it.**
- Say that decision 2 supersedes the exact-shape rule.
- `check_weight` checks "each axis 1 or `y`'s size" against `y`: the signature becomes `(kind, order, y, weight)`
  or takes the sample, so its callers change too (`check_batch` at `fitting.py:958`; `test_data_preparation.py:100-101`;
  `test_dispatch.py:823`; `test_jax_absent.py:182`).
- The weight stays outside the engine's stack check, as now (`sampling_variables.py:1097`). Say so.

### 10. SHOULD FIX: the test oracles are missing from the plan

**Evidence.**
- `backend/sampling_definitions.py` has its own weighting (`_weight_squared`, `_dual_jet`, `:655-697`) and a
  `weight` parameter on about 20 public reference functions (block form).
- It is **the** oracle of the engine's by-definition tests (PART1_PLAN §6, "a random weight, against the reference"),
  with an independence test (`test_sampling_definitions.py`, `independence_violations`).
- `tests/oracle_sweeps/oracle.py:103-139` (`weighted_residual`) is the dense fitting oracle, still on the old
  `omega[mode, order]` form.

**What would resolve it.**
- Extend both, **independently**: the reference must not call the new `W`.
- In M1, the engine's weighted `f`, `g`, `JtJp`, `Jp_norm_sq` and `Kp` against the reference over the jet matrix,
  with y-shaped weights, `K` and `C`.
- In M3, fits against the extended dense oracle. Without this, M1's "against a dense oracle" has no oracle that takes
  the new weight.

### 11. SHOULD FIX: the count and the call sites

**The functions.**
- Backend: **21**, not 18. The three `t3_*_ambient_transpose` also take `residual` (`probing.py:1071`, `apply.py:82`,
  `entries.py:94`), and the plan's frontend list includes the ambient ones.
- Frontend: **21** (TT 9, UTT 6, `T3Tangent` 3, `UT3Tangent` 3). The audit's "15" is an arithmetic slip.
- In all: **42 functions, about 205 mentions of `residual`** in their bodies. Counted with an AST script over the
  new-form names, excluding `_from_sweep` and `*_derivatives*`.
- The plan names 5 docstrings. The prose to retype is wider: for example `manifold.py:760` "The residuals live in the
  forward probe space", and the `sampling_variables` module text at `:121-122` and `:872`.

**The keyword calls.**
- Tests: **123** `residual=` calls (`test_uniform_probing` 26, `test_frontend_sampling` 26, `test_dispatch` 24,
  `test_tucker_tensor_train` 20, `oracle_sweeps/o1_ragged` 9, `o1_uniform` 7, `test_manifold` 6, `test_fitting` 5).
- Library doctests: **18**.

**The other changes to list.**
- `data_preparation` (`data_block_norms` and `equalizing_weight` docstrings; under finding 1 they stay valid block-form
  producers);
- the `LeastSquaresBatch` field comments and docstring (`fitting.py:764, 772`);
- the engine's `weight` parameter docs (`sampling_variables.py:1691-1692`) and the step comments;
- `jets.py`'s module docstring (`:51-56`, "no broadcast");
- the frontend `fitting` module text (`:45-47`);
- `test_jets` (187 weight references, the rejection table), `test_optimizer_display`, `test_least_squares_oracle`,
  `test_fitting`, `test_data_preparation`;
- `test_dispatch` / `test_jax_absent`, for the new public `W` (jit, and numpy in numpy out).

**What to leave alone, stated in the plan:**
- the old kinds' `_apply_weight` / `_weight_matrix` and `*_derivatives_transpose` / `_from_sweep` (deleted in P20);
- the bridge (block forms are unchanged under finding 1);
- `t3_inner_product_corewise_*` (a T3 operand, not `Y`).

### 12. MINOR: P17's translation needs no change

Under finding 1 the translation's outputs are already block forms, read as today (`optimizers.py:177-197`). Its 19
pins stay valid. Mapping them to a broadcast form (plan §2.A, last bullet) is churn on code P17 deletes, so drop the
bullet. P17's `point_value_and_gradient` is unaffected if `W` takes the scale (finding 2b), and its docstring can show
`J^T (W r)`.

### 13. MINOR: do not pre-empt decision 4

"`jet_to_dual_jet` is split into the public `W` and the private flip" reads as if `jet_to_dual_jet` disappears. Keep
its signature and behavior, implemented as `flip ∘ W`, and leave its public status (and D3, the double flip) to
decision 4. The same goes for `dual_jet_to_jet` as `W^{-1} ∘ flip`.

### 14. MINOR: jit, shared factors, and the engine's role

- **jit.** The weight stays a batch leaf (`fitting.py:782-788`): a new value does not recompile, but a new **shape**
  does (block form to y-shaped, or size 1 to full). Add that case to P16's compile test.
- **Shared factors.** No interaction: the weight is data-side, sharing parameter-side. Say so in one line.
- **The engine's role.** §1 says the objective-free maps "contain no data metric", while the engine takes `weight`.
  Add one sentence: the engine's `weight` is the oracle's channel, and no objective-free public function passes it.

### 15. SHOULD FIX: the slicing

**Problems.**
- M1 mixes the rule (jets), the engine and the public `W`. It omits the reference (finding 10) and
  `prepare_derivative_data` (finding 5), which is the same rule.
- M2's only stated oracle, "`J^T (W r)` equals the oracle's `g`", is the engine against itself. It needs a dense
  gradient of `1/2 <r, W r>` as well, on a drawn batch (finding 2b).

**Suggested order:**

| Slice | Contents | Tested against |
| --- | --- | --- |
| **M1** | The rule and `W` (jets level): the alignment helper, `W` with the scale, `check_weight(…, y, …)`, `jet_to_dual_jet` / `dual_jet_to_jet` through it, the factorial conversions with `kind` | hand-computed arrays with equal and distinct `F`/`W` sizes, `K`/`C` stacks; the extended rejection table |
| **M2** | The engine and the reference: `f`/`Jp_norm_sq`/seed through the helper; `_UNIFORM_MASKS['weight']`; `sampling_definitions` extended independently | the reference, over the jet matrix; garbage-padded weight; uniform against ragged |
| **M3** | The fitting layer: batch docs, `take`, `pack_batch`, `f_resolution`, the display (as ruled), `data_preparation` docs | the extended dense oracle; exact-enumeration unbiasedness with a partly broadcast per-sample weight; an entries per-sample fit |
| **M4** | The typed transposes: the rename and the docstrings, alone (mechanical, 42 functions, 141 call sites) | the existing adjoint doctests renamed; `J^T(W r)` against a dense gradient on a drawn batch |

---

## Summary for the writer

1. Write the reading rule: per array of `y`, block (rank `m`) or y-shaped (full rank, probe always with the edge),
   each axis 1 or `y`'s size, stacks inserted before the edge.
2. Fix `W`:
   - it handles stacks by that rule;
   - it takes the batch (or the scale);
   - it is defined at the `jets` level and re-exported by both fitting modules;
   - it is the one helper under all six `omega^2` sites.
3. Recommend an answer to every question.
4. Add the missing questions: the display, the scale, the stacked `C`, and the reversal of 2.5/6.7.
5. Specify the axis positions in the conversions, `pack_batch`, `take` and the uniform mask.
6. Restate D4.
7. Add the reference module and the dense oracle as independent oracles.
8. Correct the counts.
9. Reslice M1 to M4 as in the table.
