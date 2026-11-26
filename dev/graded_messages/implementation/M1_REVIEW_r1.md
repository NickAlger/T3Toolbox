NOT YET

# M1 review, round 1: `data_metric`, `blocks_to_weight`, the converters, the laws, the grammar

Reviewer, 2026-11-22, on the uncommitted tree over `cb3c0fdc` and `M1_HANDOVER.md`. Checked against:
- the plan (§2, §3, §4, 9.0 to 9.3, the M1 row);
- Claude's rulings (LOG, "M1 designed");
- my design review.

My scripts and outputs are in `scratchpad/m1rev/` (`check_dm.py`, `mutants.sh`, `mutants.out`).

**Verdict: NOT YET, for two small should-fix findings.** One is a test gap that a surviving mutant shows; the other
is an accidental deletion. Everything else holds:
- the pairing, the axis check and the dtype agree with my own construction bitwise;
- the bridge to today's block weighting is bitwise, including mixed precision;
- the law helper and its controls catch my mutants;
- the converters are retyped as decided;
- the 37 nodes of the tally are exactly the expected ones;
- no existing number changes.

## 1. `data_metric`, against my own construction (`check_dm.py`)

My oracle expands the weight with `np.repeat` along each size-1 axis (no broadcasting), then computes
`(ω·ω)·r` and, if given, `γ·(…)`.

- **Random sweep: 400 cases, all bitwise equal** (`np.array_equal`, with shape and dtype). The cases cover:
  - the kinds: apply, entries, probe ragged (tuple weights and mode-leading array weights), and probe packed;
  - `m ∈ {0,1,2}`, with 0 to 2 axes each in `W` and `C`;
  - every axis of the weight drawn at 1 or at full size;
  - the scale `None`, a float, or a 0-d array.
- **The structural raises, 11 cases.** All raise:
  - a `K`-stacked `r` with a weight;
  - a weight that would grow `r` (an `r` with `W` of 1 against a weight with `W` of 5);
  - a weight axis that is neither 1 nor `r`'s size;
  - a ragged `r` with a mode-leading weight whose mode axis is 1 (tuple and list `r`);
  - a ragged `r` with a weight of the wrong mode count (tuple and array weights);
  - a packed `r` with a ragged weight;
  - a scale of shape `(4,)` or `(1,)`;
  - a missing argument.

  With `weight=None`, a `K`-stacked `r` gives `γ r`.
- **The dtype grid** (numpy): the lower precision of `r` and `ω`. A float64 scale, as a Python float or a 0-d array,
  never raises it.
- **jax.** In eager mode and under `jax.jit`, x64 off: a float32 `r` with numpy or jax weights and scales gives
  float32. A rank mismatch raises under jit.

My design-review findings 1 (exact mode count) and 2 (the axis check), and minor point 1 (a scalar scale), are
implemented as asked. The docstring carries Q7: Taylor coordinates, the recipe, and the contragredient rule. Its
example checks the weighted gradient `J^T(data_metric(r, ω, None))` against a central difference.

## 2. The bridge to today's block weighting: bitwise

I compared `flip(data_metric(r, blocks_to_weight(kind, order, blocks, r), None))` with today's
`jet_to_dual_jet(kind, order, r, blocks)`, and `_omega_squared` with `_squared_weights`, the divisor of
`dual_jet_to_jet`:
- 144 cases: orders `()`, `(2,)`, `(1,1)` and `(2,1)`; `(W, C)` of `((4,),())`, `((3,),(2,))` and `((2,2),(2,))`;
- probe ragged, probe packed, apply and entries;
- the precision pairs f64/f64, **f32 `r` with f64 blocks**, and f64 `r` with f32 blocks.

**All bitwise equal.** The implementation is right. But the suite does not pin the mixed-precision case
(finding 1).

## 3. Mutants (`mutants.sh`, against `test_data_metric`, `test_type_laws` and `test_jets`)

| Mutant | Result |
| --- | --- |
| L1: the scale applied twice | killed (89 failed) |
| L3: the mode-0 weight used for every mode (mode-leading array) | killed (11) |
| **L4: `ω²` squared in the higher precision, then cast** | **SURVIVES** (finding 1) |
| L5: a packed `y`'s stack count off by one in `blocks_to_weight` | killed (6) |
| L6: probe blocks transposed before the reshape | killed (11) |
| L7: a tuple weight's modes reversed | killed (42) |
| L8: the batch method weighs `self.y` instead of `r` | killed (8) |
| H1: `law_error` always 0 | killed by the negative controls (2) |
| H2: the turned law not turned | killed (13) |
| H3: `P*` equal to `P` | killed (30) |
| H4: no distinct-`ℓ` check | killed by the vacuous-settings test |
| H5: `tucker` the identity | killed (9) |
| H6: the entries law the identity | killed (1) |

These are different from the implementer's 15, which I did not rerun. The negative controls do their job: H1 is
caught only by them, which is the point of having them.

## 4. The tally: 37 nodes, all expected

My own node-by-node parse of `scratchpad/m0impl/m0_suite_v.log` against `scratchpad/m1impl/m1_suite_v.log`. No file
in `t3toolbox/` or `tests/` is newer than the M1 log.

| What | Nodes |
| --- | --- |
| new in `test_data_metric.py` | 16 |
| new in `test_type_laws.py` | **10** (the handover says 9; minor 2) |
| new in `test_dispatch` (`test_jit_data_metric_and_blocks_to_weight`, 98 subtests) | 1 |
| deleted: `TestFrontendDualJetToJet` | 7 |
| `test_every_public_function_has_a_doctest`: 11 → 13 (the two new public names) | 1 |
| `test_an_object_of_the_other_type…`: 5 → 4 | 1 |
| `test_the_jet_shape_is_a_tuple…`: 7 → 6 | 1 |

I read the two `TestDualJet` hunks: each drops exactly its `fitting.dual_jet_to_jet` call, together with the two
frontend-only order checks. The jit node of `test_jit_jets_and_data_preparation` keeps its count.

The deleted frontend class's backend facts stay covered elsewhere:
- `test_the_backend_inverse_divides_by_a_zero_weight_and_raises_nothing` in `test_jets`;
- the silent division in the `test_jax_absent` script.

## 5. No existing number changed

- In `jets.py`, the diff to existing functions is docstrings only: seven hunks, the code of the converters and
  helpers untouched. The new section is purely additive.
- In `backend/fitting.py`, only the method is added.
- In the frontend `fitting.py`, the converter re-export, the frontend `dual_jet_to_jet` and its O-20 check are
  removed, as ruled (decision 4).
- The engine, the oracle, the checkers and `prepare_derivative_data` are untouched.
- The suite passes with every pre-existing node's subtest count unchanged, apart from the three listed above.

## 6. The converters, the method, the grammar

- **The converters.**
  - `jet_to_dual_jet` is typed `Y* -> Y` turned (the turn after the data metric at `γ = 1`, weight per block), with
    the `weight=None` reading `Y -> Y` turned.
  - `dual_jet_to_jet` is the inverse.
  - Both say "for the engine's `w_tilde` and held tilde variables, never for a transpose".
  - The division by zero is the only behaviour, and nothing checks it.
  - "Seed of a transpose" is corrected to the engine's adjoint sweeps.

  This is 9.2 and r4's minor point 2 as decided.
- **The method.** `jets.data_metric(r, self.weight, self.minibatch_scale)`, with a docstring stating the rank raise
  on block weights. That raise is pinned by `test_a_batch_with_a_weight_per_block_raises_by_rank` (ruling 3).
- **The grammar** (`signature_style.md`): type first, then `len`, then the shape; the role name first, with a colon;
  the tag after the shape; untyped signatures unchanged; the exemplar is `data_metric`. As planned. See minor 1 for
  one convention left implicit.
- **Departures.**
  - `_typing_laws.py`: correct; `_typing` is shadowed by CPython's built-in module.
  - The input law's `ambient` waits for M4: agreed, since it has nothing to test until then.
  - The jit coverage as a new node: right, it keeps the old node's count.

## Findings

### 1. SHOULD FIX: the bitwise bridge test does not cover mixed precision, and a precision mutant survives

**Evidence.** Mutant L4 squares `ω` in its own precision and casts afterwards: `(ω64·ω64)→f32` in place of
`(ω64→f32)²`. It passes all three test files. I confirmed that it changes numbers: with it, `check_dm.py`'s bridge
fails on every f32-`r`/f64-weight case. The two roundings differ in the last bit.

**Why it matters.**
- `_omega_squared` is "the one place `ω²` is formed, in the lower precision of the two" (plan §3).
- M2's gate rests on "every existing weighted test reproduces its pre-M2 numbers". The dual-jet divisor's underflow
  semantics are defined "as computed", in that precision.

The precision test checks the dtype only, and the bitwise test uses float64 throughout.

**Resolve.** Add mixed-precision cases to `test_through_the_metric_it_is_the_old_block_weighting_bitwise`: a float32
`r` with float64 blocks, and the reverse, on at least apply and ragged probe. With that, L4 is killed.

### 2. SHOULD FIX: an unrelated comment block was deleted from `t3toolbox/fitting.py`

**Evidence.** The deletion of the frontend `dual_jet_to_jet` also removed the six-line section header that followed
it, `# Layer dispatch for the model. ONE LocalModel serves both representations; …`, which explains
`_is_uniform_frame` and its two siblings (visible in `git diff t3toolbox/fitting.py`, just before `def
_is_uniform_frame`). This is the same kind of accident as the `interpolated_derivatives` deletion the handover
reports.

**Resolve.** Restore the block verbatim, and check the other deletions of this slice for the same accident. I found
no other: the `jets.py` deletions are docstring rewrites, and the `test_jets` deletions are the class and the
frontend calls.

### 3. MINOR

1. **Unicode or ASCII in the grammar.** The rule's examples write `⊗` and `*` (`# P*⊗V_i, …`), while the code's own
   comments spell the Greek letters out in ASCII (`# omega, …`, `# gamma: …`, `# M r: Y, gamma * omega**2 * r`), and
   older docstrings write the tensor product as `(x)` (`w0 (x) ... (x)`; the law helper too). M5 will type about
   100 signatures, so say once in `signature_style.md` which form the comments use (for example: `⊗`, `ℝ` and `*`
   as the symbols of the type, and Greek letters spelled out). That keeps M5 uniform.
2. **The handover's node count.** "9 in `test_type_laws.py`" should be 10. The total of 37 is right.
3. **`test_jax_absent`'s unused `safety` import,** which the handover notes. It is harmless; remove it if you touch
   the file anyway.

## What would resolve this round

Findings 1 and 2. Both are small and local. I will check them by:
- rerunning mutant L4 against the extended test;
- reading the restored block against `cb3c0fdc`.

No gate rerun is needed for me beyond the slice's own test files and the module doctests of `fitting.py`.
