AGREE

# Review of the revision of `PART1_PLAN.md` for the maintainer's answers (O-1 to O-28)

Reviewer: Claude, 2026-11-16. Branch `graded-messages` at `de10ad5`. The library, `tests/`, `docs/` and `examples/`
are unchanged since `4d38df7`.

Every answer is carried out, in every place where the plan uses it. I found no leftover that contradicts an answer.
No blocking or should-fix finding remains. Six minor items remain (39 to 44 below). The writer can fix them without
another round.

## What I checked, and how

- **The answers.** I read everything in `TARGET_DECISIONS.md` from the note "The names of Part 1, and the form of
  the merged sampling functions" to "What remains, in groups", and the corrected lines of "The two parts" (the 5.3
  and 2.9 lines, small items 2 and 3). I also read rows 4.2 and 8.3 as amended (lines 1101 and 1884).
- **The revised plan, in full,** and `REVISION_ANSWERS.md`.
- **Leftovers.** I searched the whole plan for text that contradicts an answer: `**held`, "only the scan", "all
  sites at once", "benchmark", "safe mode", `safety.`, "proposed" or "open point" next to an answered point, and
  every slice reference.
- **Callers, by my own searches**, over `t3toolbox/`, `tests/`, `examples/`, `docs/` and every module and page
  doctest, using `ast` scans and greps:
  - the library's `dense_probe`, as against the local helpers of the same name;
  - the `dense_*_derivatives` calls and the local helpers;
  - `_canonical_weight`;
  - every page doctest that calls an optimizer or a factory;
  - the fitting API of each of the 14 examples;
  - the backend import graph for the new uniform residuals.
- **O-35.** I checked it against `scripts/review/bad_scaling_example.py` and the 6.7 notes.
- **`get_backend`.** I read it (`backend/common.py:315`), for how `jet_contract` would choose its scan.

## The answers, one by one

| Answer | State | Where in the plan |
| --- | --- | --- |
| N16 and the ambient transposes (18:35) | Carried out | 2.4 "Changed" and its table; 2.5 table, with the ambient static methods, default off. P12 moves the backend ambient transposes and their four callers; P13 moves the frontend's 19 calls. P20's grep now also finds no `sum_over_probes` or `ztildes` |
| `dense_probe` under N16, `dense_apply`, `dense_entries` (O-27, O-28) | Carried out | 2.4 sketches; P2, with the 12 callers; P13, the examples and `test_probe_derivatives.py`; the reference module's `z` goes through them; no frontend wrapper |
| O-1, O-2 (no benchmark gate) | Carried out | Section 1 "No benchmark in Part 1", the seam "The benchmark"; section 9 "What it is for"; section 10 "on the tests alone". No benchmark gate remains anywhere |
| O-3 | Carried out | 2.2, 2.4, P11, P12 |
| O-4 | Carried out | 2.4 (the protocol); P15 (the frontend geometries unchanged); 2.5 (`COREWISE` passes `(B, G)`) |
| O-5 | Carried out | 2.5 and 7.1: a supplied `start_frame` or `fitting.local_model(frame=)`, `L` against `R`, once, in safe mode. A library-built frame, centers and `U` are not checked. Tests in P16 |
| O-6 | Carried out | 2.1's import rule; 2.3's paragraph "gives no guarantees"; 2.4 (the frame residuals never raise); 4 "Checkers, not guarantees"; P3 pins that no backend module imports `safety`; P5 "no check of the engine depends on the safety mode"; 7.1 rewritten; the backend `LocalModel`, `local_model` and the optimizers check nothing numerical |
| O-7 | Carried out | 2.4 protocol; P14; P16 (exact mode reads `optional_outputs`) |
| O-8 | Carried out | 2.2 (`HeldVariables`: slots from the graph, dictionary-like, `with_values`, no shape check, a pytree only where jax is importable); 2.3 (`held=None`); P5, with its own test file and the test without jax; N26 amended; the protocol types `held` as any object |
| O-9 | Carried out | 2.5 sketch; P16; the `frame=` keyword is proposed in 11.3 |
| O-10 | Carried out, one wording slip (finding 39) | 2.4: `over_orders` with no default, both forms, the engine passes `'scan'` on uniform and `'loop'` on ragged; `GroupLengths` keeps five fields; P1 tests both forms at orders 1 to 16 |
| O-11 | Carried out | 2.3 step 4 counts only the floating inputs the closure reads; P10 tests the integer index and the unread `y` |
| O-12, O-13 | Carried out | 2.2, 2.3, P8 (with a test that a corewise call wanting `nu_L` raises) |
| O-14 | Carried out | 2.4; section 8, the bad-scaling example computes the estimates inline |
| O-15 | Carried out | Section 8: `docs/paper_to_code.md`; `CLAUDE.md` keeps a pointer; the `.tex` note archived in P19 |
| O-16 and the reach (20:15) | Carried out | 2.5 "What else `t3toolbox.fitting` holds"; `TuckerTensorTrain.unscale_coordinates`; P3 |
| O-17 | Carried out | P16 part 3: the laps `frame`, `oracle`, `companion`, `gradient`, `curvature`; the `PhaseSeconds` doctest |
| O-18, "Checking in the two tiers" | Carried out | 2.2 and 2.3 step 1: the engine checks only unknown names, missing arguments, held shapes, and operand ranks in `jet_contract`. `check_sample`, `check_data`, `check_weight` and `check_batch`; `_canonical_weight` and `_check_ragged_sample` go |
| The two users | Carried out | 7.5 rewritten; no frontend engine; the reference module and the dense functions backend only; writing an oracle is a stated exception; the backend-only doctest and the frontend-only fit test (P16) |
| O-19 | Carried out | Section 1 table and "Time at order 0"; 2.3 step 5; 2.4 "The strings in Part 1" (no `S`); 4; P11; 7.2; the seam "The sites" |
| O-20 | Carried out | 2.4 (the backend divides); 2.5 (the frontend raises in safe mode); P3 tests both |
| O-21 | Carried out | 2.4 "The minibatch scale" (`f`, `g`, `JtJp`, `Jp_norm_sq`, `Kp`, `f_resolution` only); P14 tests the identities and that the other outputs are unscaled |
| O-22 | Carried out | 2.4, through `phi(x) = <c, z(x)>`; P12 tests against automatic differentiation at orders 0 and 2 |
| O-23 | Carried out | Section 8 |
| O-24 | Carried out | P16 with the translation; P17 deletes it; 5.1; the table of 5.4; section 6 |
| O-25 | Carried out | 5.1; deleted in P20 |
| O-26 | Carried out | P12; section 10 (the README line) |

## Leftovers searched

- **None that contradicts an answer.** Every "O-n" for an answered point is a citation of the answer, not a call to
  decide it.
- **Places that describe today's behaviour** speak in the past or of Part 2. "All sites at once" appears only as
  today's plain kernels and as Part 2's seam. "Benchmark" appears only as Part 2's.
- **Safe mode.** Every remaining mention is the frontend's, or a test that the backend does not depend on it.
- **Slice references** are consistent with P0 to P20: every P19 is the documentation, every P20 the deletion.

## The new slices

- **P16 and P17.** Each can be reviewed. The split is the one O-24 decides. The suite stays green after P16:
  - the five examples and five page doctests that call a factory move in P16, which the translation cannot cover;
  - the seven examples and the page doctests that call an optimizer in the old form run through the translation
    until P17.

  My own scan agrees with the assignment, except that `fit_per_mode_weight_probes.py` names `probe_model` in its
  docstring only. `fit_hilbert_from_apply_derivatives.py` and `_flat.py` call no optimizer.

  After P17, nothing calls the translation or `_canonical_weight`: the library's callers of `_canonical_weight` go
  with the factories in P16, and the three in `tests/test_fitting.py` are rewritten in P16. T3Polynomial runs
  through the translation until P17, and through the dry port after it.
- **The slices that gained work** remain reviewable, and nothing is used before it is built:
  - P2's dense functions come before the reference module that calls them, in the same slice, and before P6's
    bridge row and P13's examples;
  - P3's checkers come before P14's `check_batch`;
  - P4's residuals come before the cached property and P16's frontend check;
  - P5's `HeldVariables` comes before P14's `query`;
  - `pack_batch` builds on today's `pack_sample` and `pack_data`.

## The inventories, sampled

- **`dense_probe`.** 12 calls of the library's function, all positional:
  - eight doctests in `backend/probing.py`, at lines 133, 148, 163, 708, 727, 1857, 1873 and 1889;
  - `manifold.py:735`, `uniform_manifold.py:575`, `tests/test_manifold.py:524`, `tests/test_tucker_tensor_train.py:3166`.

  Seven files define a local `dense_probe`: three tests, `o1_common.py` and three examples. Confirmed.
- **`dense_*_derivatives` in examples.** The plan's "eleven calls" mixes two things. There are five calls of the
  library's functions (`pd.dense_apply_derivatives` once in each of three examples, `pd.dense_probe_derivatives`
  once in each of two). The three apply examples also define a local helper named `dense_apply_derivatives` and call
  it twice each. See finding 40.
- **`dense_*_derivatives` in tests.** `tests/test_probe_derivatives.py` has 9 calls. Confirmed.
- **`_canonical_weight`.** The account lists `t3toolbox/fitting.py` 783, 846 and 876. There is also 959
  (`probe_derivatives_model`). It goes with the factories in P16 like the others, so the plan is unaffected.
- **The uniform residuals.** `ufv_operations` can host them: no module they need reaches `ufv_operations`, so they
  create no cycle.

## The new open points (11.2), and the writer's "unclear or in conflict"

- **O-29** (`check_batch` for an oracle the user wrote): open. The answer puts `check_batch` on the least-squares
  oracle only. Answerable as stated.
- **O-30** (`held` of a user's oracle under `use_jit`): open, and a real consequence of the model being a pytree.
  Answerable as stated.
- **O-31** (the geometry's consistency member): open. No answer names it. Answerable as stated.
- **O-32** (rows 4.2 and 8.3 against O-5 and O-6): **no longer a question.** Both rows now carry the amendment of
  19:30, which says what the plan does. See finding 41.
- **O-33** (which paper-to-code maps move): open, answerable.
- **O-34** (the uniform `unscale_coordinates`): open, answerable.
- **O-35** (the shape of `scales`): the plan may state it. See finding 42.
- **Nothing open is missing from the list.** The translation's handling of positional arguments is a gap in the
  plan's description, not an open question (finding 43).
- **The writer's notes on `TARGET_DECISIONS.md`** (the 5.3 row, small items 2 and 3) are already corrected in the
  file as it stands.

## Findings

### 39. Minor. The plan describes `'scan'` by the layout, and also says `jet_contract` knows nothing of the layout (2.4, lines 332 to 338)

Line 332 says `'scan'` is "a real `lax.scan` with jax on the uniform layout, a Python loop on numpy or on the ragged
layout". The next paragraph and O-10's answer say that `jet_contract` "knows nothing of the layout" and reads only
whether its operands are jax.

- What `jet_contract` must do: take its `xscan` from `get_backend(True, use_jax)`, so that `'scan'` is a real
  `lax.scan` on jax and a loop on numpy, as the answer says.
- What the engine does: it only ever passes `'scan'` on the uniform layout, so the engine's behaviour is the same
  either way.

Fix line 332 to the answer's wording, and say which `get_backend` call `jet_contract` makes.

### 40. Minor. The examples' local helpers carry a removed name (P13; section 10's gate)

`examples/fit_hilbert_from_apply_derivatives.py:116`, `_flat.py:110` and `_topt.py:74` define a local function
named `dense_apply_derivatives`. It is one of the 129 removed names, and P20's gate greps `examples/` for it. P13
should replace these helpers with `dense_apply`, which takes the `W` stack itself, so the per-sample loop goes too.
Also restate the count as five library calls plus the local helpers.

### 41. Minor. O-32 is answered; drop it (11.2; 5.2 "Needs" of P16; 7.1 "The compiled phases")

Rows 4.2 and 8.3 now carry the amendments of 19:30 (`TARGET_DECISIONS.md` lines 1101 and 1884). They say exactly
what the plan does: the frontend checks a supplied frame once, before it calls the backend; the backend optimizers
check nothing. Move O-32 to 11.1 as answered by the amended rows, remove it from P16's "Needs", and cite the rows in
7.1.

### 42. Minor. O-35 can be stated rather than asked (2.4; P3 "Needs")

The 6.7 decision of 18:32 made "the rescaling of a mode's coordinates" and its inverse library functions, as the
prototype `scripts/review/bad_scaling_example.py` wrote them. The prototype's `scale` is `len=d, elm_shape=(Ni,)`:
lines 70 and 78, applied per coordinate at lines 72 and 80. Its problem has coordinate scales spread over four
decades within a mode (line 104), which one number per mode cannot undo. The earlier `(d,)` was the plan's own
sketch, not a decision.

So the per-coordinate shape follows the decision. State it with the prototype as the source, and keep at most a
line asking for confirmation, not a dependency of P3.

### 43. Minor. The translation must remap positional arguments (P16 part 4)

Old calls pass the sample and the data by position, as in `topt.adam(t3m.COREWISE, 'probe', ww, pdata, x0c, rng,
20, max_iter=50)` (`docs/fitting_and_optimization.md:70`). In the new signature these land in `batch`, `x0` and
beyond. Say that the translation takes the old positional arguments as the old signature names them, not only that
"a string where the oracle goes marks the old form".

### 44. Minor. Form

- **Section 7.5** quotes a whole paragraph of the maintainer. One sentence and the citation would do, as the brief
  asks.
- **The status line** still carries the history of the rounds (finding 38 of the N16 review).

The record of the answers is otherwise confined to 11.1, and the body reads as a plan.

## What is sound (keep it)

- **Each answer lands everywhere it bears**, with its citation; 11.1 is a clean index of where.
- **The two tiers are now concrete:** checkers in the backend, enforcement in the frontend, `check_batch` and
  `pack_batch`, the reach from `t3toolbox.fitting`, and the frontend-only fit test.
- **The P16 and P17 split** keeps every example and page in exactly one slice, with the reason for each.
- **The corewise Hessian is now one definition with the transpose,** and P12 tests both against automatic
  differentiation.
