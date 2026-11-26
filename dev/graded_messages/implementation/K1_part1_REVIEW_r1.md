SIGN OFF

# K1 part 1, build review, round 1

Reviewer 4 (Claude), 2026-11-26, 01:05 to 01:20 CT. I judged the working tree of `k1-wip` against `HEAD`
(`49db8e37` = part 0's `78a34d77` plus my preparation files, differing from it in `dev/` only). My plan, written before the
build: `k1/REVIEWER4_PLAN.md`; my tools: `k1/reviewer_tools/{ref_dense_pins,ref_independence,part1_textcheck,
audit_c_attack,ref_mutations}.py`. Scratch: `scratchpad/reviewer4/`. I read the code first and the handover after.

**Verdict: SIGN OFF.** No blocking or should-fix finding. Two minor notes for later parts (§9). Each of my plan's
questions is answered by the build: the `g` pin is a pairing done in loops (Q1), the law test covers all 17 functions
that build the seed (Q2), and the both-weights case is in the engine comparison (Q3). The note's 124-column signature
line became a 112-column line with a `#   None = ones` continuation (Q4). Q5 (mode agreement) is consistent with the
plan (§9).

## 1. My dense pins against the built reference (the check K1 rests on)

`PYTHONPATH=<repo> python -B k1/reviewer_tools/ref_dense_pins.py --api new`: **1,968 checks, 0 failed.**
`--jax`: **25 checks, 0 failed.** No library code is in the oracle side: a dense model built by loops.

**What passed**
- Every variable: `w_tilde` (turned), `f`, `r_block_sumsq`, `Jp_norm_sq`, and `g`, `JtJp` **entry by entry of every
  V and H block**. Each entry is the loop pairing of `M r` (or `M Jp(p)`) with the sampled dense tangent of its unit
  direction.
- The cases:
  - all three kinds;
  - orders `()`, `(2,)` (`d = 3 = n + 1`, `F = W = (3,)`, every `N_i = 3`) and `(2, 1)` (`F = W = (3, 2)`);
  - input jets of three jet sizes per mode, one slot beyond the box;
  - tails on both end bonds;
  - one degenerate configuration: order `(0,)`, `N = (3, 1, 2)`, `W = (1,)`, where "exact" is 1.
- The weight patterns: none; block; full exact; full with every `W` axis 1; full with one `W` axis of 1; both; both with
  `W` of 1; both with one `W` axis of 1; for entries, a block as a Python float, alone and with a full weight.
- **The law:** each of the 16 seed-reading functions (`g`, `Kp`, the six tilde and four tilde-dot variables,
  `V_tilde`, `H_tilde`, `V_tilde_dot`, `H_tilde_dot`) equals itself with `w_tilde=` **my loop seed**. It is checked
  in every case above. The 16 functions that read no weight give the same result with and without both weights.
- **The refusals**, each in `f`, `r_block_sumsq`, `Jp_norm_sq`, `w_tilde`, `g` and `JtJp`:
  - block weights: `(d,)` at `(n,)` (the E3b coincidence), `F` alone, `(d,)+F+(1,)`, the old `blocks_to_weight` form,
    `F+(1,)`, `(d,)+F` for apply, `(1,)` and `W` for entries: `ValueError`;
  - a block weight as a tuple or a list: `TypeError`;
  - full weights: an edge of 1, a mode-leading array with a mode axis of 1, `d-1` arrays, a first jet axis of 1, every
    structural axis 1, rank too high or too low, a scalar for entries;
  - `y` with every `W` axis 1, one `W` axis 1, an edge or a jet axis of 1 (also in `r`).
- **jax:** jax block and full weights, a numpy block with a jax full weight, a jax block alone. Jax arrays come out,
  and the numbers are numpy's.

**What it does not cover**
- A `K` or a `C` stack: the reference has neither, by design.
- Corewise frames (the law and the pins use a manifold frame with independent stores; the part's own tests cover
  corewise frames).
- `Kp`'s value is not pinned by my loops, only through the law: `Kp(y, b, f) == Kp(w_tilde = my seed)`. `Kp` given a
  seed is unchanged code, pinned by the part's existing finite-difference tests.
- float32.

**INFO:** the reference accepts a probe full weight whose modes differ in their pattern of size-1 `W` axes (read mode
by mode; see §9).

## 2. Independence

`ref_independence.py --api new`:
- **(1) static:** the reference reads only `probing.dense_probe`, `apply.dense_apply` and `entries.dense_entries` off
  its imports (and `common` by its star import).
- **(2) a profile hook over all 37 functions with both weights:** the library functions entered outside the reference
  are `dense_probe`, `dense_apply`, `dense_entries`, `probing._dense_jet` and `_dense_contract_one_mode`,
  `contractions._read_order`, and `common`'s dispatch helpers. **Nothing of `jets`, the engine, the steps or the
  fitting layer.**
- **`b**2 * f**2` is formed by the reference's own code:** `_weight_squared` (`sampling_definitions.py:725-747`)
  squares the block row and reshapes it left-aligned (`row.shape + (1,) * (len(shape) - m)`), then multiplies by the
  full weight squared. The full weight is checked by its own `_check_data_space`, with the `W` axes as `may_be_one`.

**The pins' oracles.** `TestTheTwoWeights.test_against_brute_force` (`tests/test_sampling_definitions.py:611-671`):
- `z`, `Jp`, `Jq` come from `sample_dense_jet`, the tests' own route, over `t3_to_dense` / `tv_to_dense`.
- `r`, the seed, `f`, the block sums, `Jp_norm_sq` and **the pairings `pair_g = Σ b² f² r Jp(p)` and
  `pair_JtJp = Σ b² f² Jp(p) Jq`** are `np.ndindex` loops. They go through the local `weight2(i, index)`, which reads
  `block[i][index[:m]]` and the full weight at the entry by the module-level `at`. No reference function is on the
  expected side.
- **Q1 is settled:** `g` is pinned by the loop pairing, not by `d/dt sdef.f` (handover §4.4).

**The tests my scan marked READ:**
- `weight2`, and `TestJetsTwoWays.at` / `TestApplyAndEntries.at`, are name collisions with the module-level brute-force
  `at` (`:561`): false positives.
- `test_every_name_that_reads_the_seed_reads_both_weights`: its seed is written by loops from the tests' own dense `z`.
  The reference is on both sides, but the expected side is the reference *given* that independent seed. That is the
  law, and right.
- `TestKp...gradient`: a finite-difference law of the reference on itself, unchanged by part 1.
- `test_sampling_{apply,corewise,variables}`: the engine against the reference. The reference gets
  `block_weight=problem.block_weight(...)` (the blocks), the engine `problem.data(...)` (the `blocks_to_weight`
  placement of the same blocks).

**Point 8 of the brief.** `TestOutputsAgainstDense` (`:456`, `:519`) and the `Kp` second-derivative test (`:1409`)
build their expectation with `P.data(kind, case)[1]`, the `jets.blocks_to_weight` placement, then `Jets.weighted`
(equal-rank broadcasting, asserted).
- This is not a correlation risk: `blocks_to_weight` is a reshape of `jets`, the reference never calls it, and the
  two codes differ.
- The two-weight reading is pinned by `TestTheTwoWeights` without it.
- Leaving it to part 5 is right in substance. I would do it in part 2, since part 2 is where `Problem.data` stops
  being the engine's input: after that its second element serves only these three expectations, and a loop placement
  of a dozen lines makes the reference's tests free of `jets` for their expectations. Either part is fine, provided
  the part's scope list says so (handover §5 does, for part 5).

## 3. The text

`part1_textcheck.py reference` (each unit of the reference against `78a34d77` after the intended substitutions):
- **Exactly the intended sites, and no others:**
  - all 37 signatures (`y`, `block_weight` with its `#   None = ones` continuation, `full_weight`);
  - the 17 `_the_seed(...)` calls;
  - `_dual_jet` in `JtJp`;
  - `_weight_squared` in `f`, `r_block_sumsq`, `Jp_norm_sq`.
- The residual differences are confined to 17 units, all intended:
  - the module docstring;
  - `_r`, `_check_data_space`, `_weight_squared`, `_dual_jet`, `_the_seed`, and the new `_check_block_weight`
    (prose §6.7);
  - `w_tilde`, the six tilde examples, `JtJp`, `f`, `r_block_sumsq`, `Jp_norm_sq` (prose §6.2 to §6.6).

  I read each residual hunk. The lines removed are the old weight's text, and nothing else.
- **Line wraps:** the body lines of `_dual_jet`, `JtJp`, `f`, `r_block_sumsq` and `Jp_norm_sq` are wrapped, because the
  substitution took them past 120. The module docstring line `Pass ``y`` (and the weights) for the seed` is 136
  columns. It was already over 120 before (129), and it was not re-flowed (ruling 33).
- **Columns:** every public signature keeps its one `#` column (61). The continuation `#   None = ones` is in that
  column in all 37 functions. The `block_weight` line is 112 columns in all 37. The module has 3 lines over 120, as
  before.
- `part1_textcheck.py prose`: **all 8 "After" blocks of `prose_edits.md` §6 found** (whitespace-normalized; `...` and
  `<to be pasted>` as wildcards).
- **The doctests:** `--doctest-modules` on the reference passes 37 of 37, so the pasted outputs are those of a run. The
  blank line before every `>>>` block is kept: a scan finds the same single exception-output case as before.
- **Python 3.9:** `ast.parse(..., feature_version=(3, 9))` passes on every changed `.py` file.

## 4. The weights audit

- **My 14 planted conversions** (`audit_c_attack.py k1/diff_audit.py`), against the final audit: **14 of 14 as a sound
  audit should be.** Every wrong conversion FAILs, every right one PASSes. A dropped weight passes and is listed
  `W-GONE`. A string key FAILs through its `BTW-UNPAIRED`.
- **The rerun:** `diff_audit.py HEAD --part 1 --scope-map TestTheWeightIsOfTheTypeOfZ=TestTheTwoWeights`. It
  reproduces `k1/part1_audit.txt` **byte for byte**, RESULT PASS. Without `--scope-map`, the renamed class's 7 sites
  show as `W-GONE`. The flag is in the handover's command; noted for whoever reruns it.
- **The one `W-GONE`:** `TestAgainstTheReference.test_the_engine_is_the_reference`. Its `weight=` reaches the reference
  through `**reference`, which the class's `weights` method builds. The audit pairs that construction:
  `B-PAIRED tests/test_engine_data_space.py:142` and its `F-REASON` line.
- **The 40 reason lines** (25 B, 15 F), each read against its site:
  - **No apply or entries block weight is passed as a full weight.** In `test_sampling_{apply,entries}` the
    reference's weight is now `block_weight=problem.block_weight(...)`, the blocks themselves. The engine gets
    `problem.data(...)`, their `blocks_to_weight` placement.
  - The apply and entries full weights are all in `TestAgainstTheReference.weights` and `TestTheTwoWeights`. There
    they are genuinely full: per entry; per sample written out; with `W` of 1; the full half of "both".
  - The one case where a full weight carries a block's meaning: the pattern "edge and first jet axis, written out" for
    apply. It is a weight constant off the first jet axis, passed deliberately as a written-out full weight (M2's
    pattern kept), beside the separate "per slot: a block weight" pattern. The reason line says so. Right.
  - The docstring B sites: the old `(3,1,1)` and `(3,3,1,1)` became `rng.random(3)` and `rng.random((3, 3))` (the
    same draws). Correct block forms.
  - The F sites: the old `(3,7,1)` per-sample weight is written out by `np.broadcast_to`. Correct.
  - `test_sampling_stacks.py:311`: `weight_c` is element `c` of a per-entry `box+W+C+(N_i,)` weight. Full, right.
- **The string keys:** `Problem.arguments` / `TestKp.jax_problem` read `args['block_weight']`. `arguments` builds it
  from `self.block_weight(kind, case)` (the blocks), so the key carries the right object.

## 5. The rewritten tests

- **`TestAgainstTheReference`** (`tests/test_engine_data_space.py`; round 2's finding 12), pattern by pattern. Each
  maps to the same metric on both sides:

  | pattern | reference gets | engine gets |
  |---|---|---|
  | per slot | block `(d,)+F` / `F` / `()` | `blocks_to_weight` of that block |
  | one per mode | a per-mode value broadcast over `F` | `blocks_to_weight` of it |
  | per entry, per sample, edge and first jet axis | a written-out full weight | the same array |
  | with `W` of 1 | the full weight with its `W` axes 1 | the same array |
  | **a block and a full weight** | the pair | `blocks_to_weight(b)[i] * f` mode by mode, which squared is `b² f²` to rounding |

  - `y` constant in `W` is written out by `np.broadcast_to` and handed to both sides.
  - The engine path is the old one-weight `jets.data_metric`, independent of the reference.
  - It passes at `TOL`, so the both-weights case holds to rounding (Q3).
- **The law test** (`test_every_name_that_reads_the_seed_reads_both_weights`) covers 17 names over probe at `(3,)` (a line)
  and `(1, 2)` (a plane), apply at `(3,)` and entries, with both weights (a random block, a full weight with `W` of 1). Would it catch a function's own slip in its call?
  - My mutants **M6** (`Kp` passes `None` for the block) and **M7** (`nu_L_tilde` swaps the two): both caught (§6).
  - The implementer's two (`nu_tilde` swapped, `V_tilde_dot` drops the full weight) are caught too (its record).
- **`Problem.full_weight` → `Problem.block_weight`:**
  - its four callers in `test_sampling_{apply,corewise,entries,variables}.py` follow;
  - no `.full_weight(` call is left in `tests/` (grep);
  - the suite and my run of the changed files pass.
- **`Jets.weighted`** (my plan's item 4) keeps equal-rank broadcasting and **asserts** equal rank. It is handed only
  the placed table, so it never right-aligns a block row.
- **`oracle.py`'s `weighted_residual_two` / `weighted_two`** place the block row by a loop over the slots
  (`_block_factor`) and write out the full weight by `np.repeat` (`expand`). The module imports only `math` and
  `numpy`. They are pinned in `test_dense_oracle.TestTwoWeights` against an `np.ndindex` loop. Their `expand(y)`
  still accepts size-1 axes in `y`: an oracle may be more liberal than the rule. No caller yet (part 4).

## 6. Mutations

**My eleven** (`ref_mutations.py`, the patches written from the built text), each run against my pins and, on a
second pass, against the part's `TestTheTwoWeights` and `TestAgainstTheReference`:

| # | mutation (in `sampling_definitions.py`) | my pins | the part's tests |
|---|---|---|---|
| M1 | the block row right-aligned (`(1,)*k + row.shape`) | caught: 66 numeric failures (at `()` and `(2,)`, where every size coincides and so nothing fails by shape), then a shape error at `(2, 1)` | caught, 53 failed |
| M2 | the block row of the next mode, `block_weight[(i + 1) % d]` | caught, 286 failures | caught, 66 failed |
| M3 | the block table transposed (`.T[i]`) | caught: 66 numeric, then a shape error | caught, 35 failed |
| M4b | the block row unsquared | caught, 610 | caught, 132 |
| M4f | the full weight unsquared | caught, 810 | caught, 204 |
| M5 | both weights applied after the turn in `_dual_jet` | caught, 408 | caught, 167 |
| M6 | `Kp` passes `None` for the block in its own `_the_seed` call | caught, 31 (the law rows of `Kp`) | caught, 4 (the law test) |
| M7 | `nu_L_tilde` swaps the two weights in its own call | caught (a structural refusal) | caught, 4 (the law test) |
| M8 | `y` allowed a size-1 `W` | caught, 70 (the refusals) | caught, 3 |
| M9 | the full weight allowed an edge of 1 | caught, 18 | caught, 1 |
| M10 | a block weight as a sequence accepted (the `TypeError` dropped) | caught, 84 | caught, 3 |
| M11 | the exact check written as "no 1 on a structural axis" | caught (the degenerate configuration refused) | caught, 4 (`test_an_object_larger_than_z_where_z_has_a_size_of_one_raises`, order `(0,)`) |

Every mutant is caught twice, by my independent pins and by the part's own tests. The unmutated part's tests pass
(§7).

**The implementer's record** (`k1/part1_mutations.txt`): eleven, all caught. I did not rerun them separately: my M1,
M4b, M5, M8, M9 and M10 are the same intents on the same lines (right-aligned, unsquared, after the turn, `y` with a
`W` of 1, a structural 1 in the full weight, the block unchecked), each rerun here against the part's tests.

## 7. The tally and the gate

- **The `-v` log is valid.** `k1/part1_suite_v.log.gz` ended at 01:04:24 after 32:51, so it started about 00:31:33.
  The last library or test file was written at 00:29:46 (`tests/test_sampling_definitions.py`; the reference at
  00:29:11). Only `dev/` files were written after the start. So I did not rerun the suite.
- **My own tally** of `k1/part0_suite_v.log.gz` against `k1/part1_suite_v.log.gz`:
  - 1,887 → 1,890 passed; 5 skipped and 1 xfailed in both;
  - gone: `TestTheWeightIsOfTheTypeOfZ`'s four tests, which reappear under `TestTheTwoWeights` with the same names;
  - new: `TestTheTwoWeights.test_every_name_that_reads_the_seed_reads_both_weights` and `test_dense_oracle`'s two
    `TestTwoWeights` tests;
  - moved: `test_least_squares_oracle::test_reachable_from_the_frontend`, from `TestTheBatchConstructor` to
    `TestTheOracle`. That file is unchanged by part 1; part 0's log predates the fix made after part 0's sign-off
    (ruling 32), as the handover says. That makes 10 nodes, as the handover says.
- **My own run** of part 1's eight changed test files (`-n 2`): **152 passed, 17,933 subtests passed** (3 min 19 s).
- **The reference's module doctests:** 37 passed.
- **Not run by me:** the full gate (no page, example or Sphinx input changes in part 1), the slow tier, the CI floor.
  The 3.9 syntax check above is static.

## 8. The departures (handover §4)

1. The two-line `block_weight` comment: the orchestrator's point. It is aligned and fits the line limit (§3). Agree.
2. The class renamed `TestTheTwoWeights`, with its four test names kept: agree. The tally maps them, and the audit
   needs `--scope-map` for it (§4).
3. `Problem.full_weight` renamed `Problem.block_weight`: agree. The old name would now hand the reference a form it
   refuses. Every caller followed.
4. `g` pinned by the loop pairing: agree; this was my Q1.
5. Two patterns added to `TestAgainstTheReference`, and the refusals left to `TestTheTwoWeights`: agree. The engine's
   refusals are part 2's.
6. The oracle reads `m` off the block weight: fine, since `m` is needed only to place a block.
7. Three expectations still use the `blocks_to_weight` placement: acceptable (§2, point 8). I recommend replacing it
   in part 2 rather than part 5.
8. The mutation list is the implementer's own: fine. Mine adds the wrong mode, the transpose, a dropped block in one
   function and the degenerate "exact".

## 9. Notes for later parts (not findings against part 1)

- **Part 2: the mode agreement.**
  - My pins show the reference reads a probe full weight mode by mode, so it accepts modes whose size-1 `W` patterns
    differ.
  - The plan puts the agreement rule on the engine, the checkers, `pack_batch` and the batch, not on the reference.
    The module docstring claims only "its sample axes 1 or full", which is true of the reference.
  - Consequence: part 2's engine refusal of disagreeing modes cannot be checked against the reference (which would
    accept the input). Test it as a refusal on its own.
- **Part 2 or 5:** `Problem.data`'s `blocks_to_weight` placement in the reference's own tests (§2, point 8).
- **Rerunning audit (c)** needs `--scope-map TestTheWeightIsOfTheTypeOfZ=TestTheTwoWeights` while the renamed class is
  in the part's diff.
