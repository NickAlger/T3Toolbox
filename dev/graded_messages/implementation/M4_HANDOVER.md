# M4 handover: the transposes typed

Implementer, 2026-11-23, on `2bc53249`; the CI fix `66bdbc1c` is in that base. Built from `M4_DESIGN_NOTE.md` and
Claude's rulings:
- **D1:** the type goes on the `c` line now; every other line waits for M5.
- **D2:** P19's upgrade note reads "`residual=` → `c=`; positional calls unaffected; no shim".

No separate design review was held. Nothing is committed.

## 1. What changed

**Step A, the scripted rename** (`m4/rename.py`; its edit log `m4/edit_log.tsv`):
- 275 edits in 18 files, by rule:

  | Rule | Edits | What |
  | --- | --- | --- |
  | A1 | 88 | the argument `residual` of the 42 functions and its uses in their bodies |
  | A2 | 128 | the `residual=` keyword calls in tests: every one; the plan counted 123 lines |
  | D1 | 17 | the doctest `residual=` lines |
  | D2 | 36 | the ` ``residual`` ` literals in the 42 docstrings |
  | E | 6 | the `'residual'` strings of `test_sampling_functions.py`'s signature table |

- **How each call is found.** A2 locates a call by AST: the callee is one of the 42 by name, or a `getattr` whose name
  string contains `_transpose` or `_corewise_hessian`, or one of three reviewed sites (`CALL_SITES`). The run left no
  `residual=` anywhere under `t3toolbox/` or `tests/`.
- **One pre-existing name.** `t3_apply_ambient_transpose`'s local `c = xnp.asarray(residual)` becomes
  `c = xnp.asarray(c)`. It was reviewed by eye, and is correct.
- **The round trip.** On a `git archive` of HEAD: forward, then inverse by replaying the log, gives HEAD back byte for
  byte under `t3toolbox/` and `tests/`. A fresh forward run on HEAD reproduces the committed edit log exactly
  (`cmp`), so the reviewer can regenerate the scripted part and diff it.

**Step B, by hand** (as explicit edits in `m4/prose.py`; the hand-only diff is `scratchpad/m4impl/hand.diff`, 14
files):
- **The 42 `c` lines (D1).** Each gets its type, and its type column is realigned (A1 shortened the name by 7):
  - `P*⊗V_i, len=d, elm_shape=...` for probe;
  - `P*, shape=...` for apply;
  - `ℝ, shape=...` for entries;
  - the ambient ones: `V_i` for probe, `ℝ` otherwise, since they have no jets.

  "of the type of z" and "; no weight" are gone from these lines.
- **The parameter text.** Every docstring of the 42 either states it, or points ("as there") to one that does: "`c`
  is an element of `Y` (`P*⊗V_i` per mode / `P*` / `ℝ` per sample), paired with `z` slot by slot. It carries no
  metric, and none is applied here: for the gradient of a weighted least-squares fit pass
  `data_metric(r, weight, minibatch_scale)` (or `batch.data_metric(r)`), which is `r` itself with no weight and no
  scale."
- **`J^T r` → `J^T c`,** with "the Gauss-Newton gradient `J^T(M r)` when `c = data_metric(...)`" where a gradient is
  meant. The sites:
  - the plan's: `manifold.py` probe and apply transposes; `uniform_manifold.py` probe; `tucker_tensor_train.py` the
    probe and apply corewise transposes; `uniform_tucker_tensor_train.py` the apply corewise transpose;
  - the ambient ones;
  - the two `project` docstrings (`manifold.py`, `uniform_manifold.py`).
- **Residual wording.**
  - `phi(x) = <residual, z(x)>` becomes `<c, z(x)>` ("the pairing of `Y` with `Y*`, slot by slot: no metric").
  - "turned inside (a dual jet)" becomes "the engine turns `c` into its dual-jet form inside".
  - The probe ambient formulas use `c_i`.
  - `sampling_variables.py` (the held-`Jp` route) and `HeldVariables` say "the transposes compute `J^T c` (they pass
    no weight)".
- **P13's right/wrong adjoint doctests** are kept. Their prose says "the sample, then `c`" / "`(c, ww)` runs", and
  their code is unchanged.
- **The two examples** (`fit_hilbert_from_apply_derivatives{,_flat}.py`) gain a two-line comment on the
  contragredient rule (`c_t[k] = k! c_d[k]`, by `taylor_to_derivatives`); their code is unchanged.

**`tests/_typing_laws.py`.** New input-law movers:
- `frame` (`D_i Δ_i^-1`);
- `variations_dual` (`V_i Δ_i`);
- `packed_mode_dual`, a packed block times `Δ_i` padded with ones;
- `entries_dual` (`× ∏ Δ_i[n_i]`);
- **`ambient`**, the part left from M1: a dense tensor times `⊗Δ_i`, applied to the CP output summed.

The module text now describes them.

## 2. Tests

**`tests/test_transposes_typed.py`** (new).
- `TestTheSignatures`:
  - each of the 42 (listed explicitly, by owner and name) has `c` and no `residual`;
  - `residual=` raises `TypeError` (no shim). The old layer's names are documented as the deliberate exceptions.
- `TestTheInputLaw.test_all_42`:
  - every one of the 42 against the input law, through an adapter that moves the ragged arguments and packs inside
    for the uniform ones;
  - order `(2,)` for probe and apply, `()` for entries and the ambient ones;
  - `rtol 1e-11`; the frontend calls run under `safety.unsafe()`, since the moved frame is not orthonormal.
- `TestTheJetLaw`: the 24 with jets at `(2,)` and `(1, 1)` (48 subtests). `c` moves by `P*` (`ℓ^-k`), the inputs by
  `P`, and the output is fixed.
- `TestTheGradientOfAWeightedFit`:
  - setup: a drawn batch (`take`, `γ = 2`) with a weight per entry, zeros included;
  - check: `<J^T(batch.data_metric(r)), ẋ>` against a central difference of `γ · oracle.misfit_full`, one core at a
    time (exact: the misfit is quadratic in `t`);
  - every kind, probe and apply at `(2,)`; 18 subtests, to `1e-9`.

**Mutation pass** (`scratchpad/m4impl/mutants.sh`, each on a fresh copy):

| Mutant | Result |
| --- | --- |
| test side: the probe `c` moved as a measurement (`V_i*`) | 15 failures (input law) |
| test side: `c` moved as a jet (`P`, not `P*`) | 49 (jet law) |
| helper: the ambient factor on mode 0 only | 7 (the six ambient subtests) |
| library: the seed not flipped (a dual jet read as a jet) | 61 (jet law, gradient check) |
| library: `t3_apply_ambient_transpose` scales `c` by `Σ w_0²` | 3 (the apply ambient ones) |
| test side: the gradient check given `r` in place of `batch.data_metric(r)` | 18 (every core of every kind) |

## 3. The gate

**The suite**, `pytest tests/ -n 2 -v` (under `timed.py`): **1965 passed, 5 skipped, 1 xfailed, 92,475 subtests
passed**, 36:00. Lowest MemAvailable 3.6 GB (`scratchpad/m4impl/gate/suite_v.log`).

**The node tally** against M3's `-v` log (`m0/tally.py`; `scratchpad/m4impl/tally_m3_vs_m4.txt`): 1,966 → 1,971 nodes.
Exactly 5 nodes differ, all new (`test_transposes_typed.py`: 42, 42, 48, 18 subtests and one plain). Every other node
has M3's outcome and subtest count, the 128 renamed keyword calls included, so no number changed.

**The other steps** (module doctests, `getting_started.rst`, the pages, the examples, Sphinx) were started detached
and **stopped unfinished** when the report was due, so that nothing is left running. They are still to be run:
`run_gate.sh -n 2 doctest-modules doctest-quickstart doctest-pages examples sphinx`. Already checked after step A:
the module doctests of the nine changed sampling modules, 134 passed. The prose of step B has not yet been run as
doctests.

## 4. Departures and choices

- **A2 counts 128 keywords.** The plan's 123 counted lines; five lines hold two calls. Every `residual=` is converted.
- **The six `'residual'` strings** in `test_sampling_functions.py`'s signature table are renamed too (rule E). That
  table pins the backend signatures, so leaving them would make it fail.
- **The ambient transposes' type** is `V_i` for probe and `ℝ` for apply and entries. They take no `order`, so
  `P* = ℝ`.
- **The law adapters** move ragged arguments and pack inside the uniform calls. The input law on a uniform call is
  then the law on its real blocks, with `Δ` padded with ones; the padding is zero by the masks.
- **The prose was applied by a script of explicit, exactly-matched edits** (`m4/prose.py`), not by free editing. It
  is kept as the record; the hand-only diff is `hand.diff`.

## 5. For P19, and found outside the slice

- **P19's upgrade note (D2):** "`residual=` → `c=` in the 42 transposes and Hessians; positional calls unaffected; no
  shim."
- **Untouched, as the plan says:**
  - the old layer: `SamplingKind.transpose`/`corewise_hessian`, the four `utv_*_from_sweep` helpers, and
    `sampling_derivatives.py`'s "dual jet" prose;
  - the Hessians' names (decision 4).
- **Pages.** No page names `residual=`. `docs/transposes.md` says "residual" for the transposes' argument five times
  (`:20, 36, 209, 210, 214`); that is page prose, so it goes on P19's list.
