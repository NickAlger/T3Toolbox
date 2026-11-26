NOT YET

# K1 part 2, prose review, round 1

Reviewer 4 (Claude), 2026-11-26, 01:25 to 01:30 CT. I reviewed `k1/prose_edits.md` §3 (3.1 to 3.11) against the code at
`31e3d100` and against the note's §4.2 and §4.3. The work was static: scripts that import the library, `ast` scans and
greps; no test run.

**Verdict: NOT YET.** Two findings should be fixed before the build, both small:
- the dtype rule in `jets._metric`'s docstring (finding 1);
- the `Rule` rows' layout, which can and should be decided now (finding 2).

The rest is minor. Most of the prose is accurate and says no more than the engine will enforce:
- the parameter entries;
- "The engine checks what it needs";
- the uniform-layout paragraph (the full weight always masked, the block weight never, no promise beyond finite
  padding);
- the steps;
- the bridge's comment.

Every "before" is today's text. A script found each changed line of every before-block in the four modules at
`31e3d100`: six blocks match byte for byte, four line for line (their docstring indentation was stripped in the
markdown). I checked the inline before-texts by hand.

## Findings

### 1. `jets._metric`'s docstring states a dtype rule that is neither today's nor the absorbing rule (should fix)

- **Place:** §3.9: "Each square is formed in the lowest floating precision of `r_X` and that weight (the absorbing rule);
  ... then `* r_X`; `gamma` last, cast to the result's dtype".
- **Evidence:**
  - Today's `data_metric` (`jets.py`, its loop) first **casts `r_X` itself** to the lowest floating precision of `r`
    and the weight (`r_X = xnp.asarray(r_X, dtype=dtype)`), then squares the weight in it.
  - Measured at `31e3d100`: `data_metric(r float64, w float32, None).dtype` is **float32**, and so is
    `data_metric(r float32, w float64, None)`.
  - The drafted rule, read literally, squares each weight in the lowest precision of `r` and that weight, and leaves
    `r` as it is. With `r` float64, a float32 block and a float64 full weight it gives **float64**.
  - So does a float32 block weight alone. A block passed as the old `blocks_to_weight(b)` gives float32 today, so the
    battery's bitwise single-weight claim (note §3, check 5, which compares dtypes too) would fail.
  - Reviewer 3's prenote 3a asked exactly this: the rule must read `r`, `b` and `f` together.
- **Resolve:** state the rule as the definition:
  - the result's dtype is the lowest floating precision among `r_X` and the weights given (as `data_metric` does
    today; on jax no more than jax's own float);
  - `r_X` and each given weight are cast to it, and each square is formed in it;
  - `block**2 * full**2` only when both are given; then `* r_X`; `gamma` last, cast to that dtype.

  One sentence also belongs in the note's §4.3, which says "forms `b*b` and `f*f` each in the lowest floating
  precision of `r` and the weights": the same ambiguity.
- **For the build review:** one battery row, or a test, with a float32 block weight against float64 data, and its
  dtype.

### 2. The `Rule` rows: the table already has its overflow form; decide it in the prose, not in the build (should fix)

- **Place:** §3.4: "the rows keep their columns where the new tuple fits ... where it does not the build re-aligns the
  later columns of that block of rows".
- **Evidence:** `sampling_variables.py` has 69 `Rule(` rows, and all of them share one set of columns across the
  tuples (`_PLAIN_ROWS`, `_RESIDUAL_ROWS`, `_DOT_ROWS`, `_DOT_SUM_ROWS`, ...; e.g. `:385` and `:415` have identical
  columns).
  - The `reads` column is 31 characters wide (`('r', 'weight'),` plus its padding).
  - `('r', 'block_weight', 'full_weight'),` is 37 characters and `('Jp', 'block_weight', 'full_weight'),` 38, so
    none of the four rows fits: this is known now, not at the build.
  - Re-aligning "that block of rows" would give the module two column layouts for one table.
  - The table already has a form for a `reads` tuple that does not fit: the row stops after `reads`, and the later
    columns continue on the next line, aligned in their own columns (`:393-406`, e.g. `_raw.eta_D_dot` at `:405-406`).
- **Resolve:** write the four rows (`:385-387`, `:415`) now in that existing form:

  ```python
    Rule('f',              ('whole', 'f_step'),             ('r', 'block_weight', 'full_weight'),
                                                                                           _NONE,             False),
  ```

  No other row moves. The twin rows are derived (they are in the module docstring's table, not in code), so nothing
  else is affected. The module docstring's markdown table needs no alignment.

### 3. Two after-texts are not written in full (minor)

- **Place:** §3.4, `HeldVariables` ("that line re-filled with the one before and after it if it exceeds the column"),
  and `_check_stacks` (`:1111-1112`).
- **Evidence:**
  - The new `HeldVariables` line is 148 columns, so a re-fill is forced. `HeldVariables` is a public class, and the
    slice's rule is that public docstrings are written in full before the build.
  - `_check_stacks`' second line (`:1112`) is already 151 columns. The new sentence is longer, and the prose does not
    say which lines result.
- **Resolve:** write the after-lines of both, as for every other paragraph.

### 4. §3.2, the block weight "one array": a number is accepted for entries (minor)

- **Place:** §3.2, `block_weight`: "one array, the same on both layouts, never a tuple or a list".
- **Evidence:**
  - The engine's check is `np.shape(block_weight) == ()` for entries, so a Python float passes.
  - The reference's text says so ("a Python number for entries", `sampling_definitions.py` `_check_block_weight`).
- **Resolve:** "one array (a number for entries), the same on both layouts, never a tuple or a list (a `TypeError`)".

### 5. §3.6, the steps' comment keeps "each axis 1 or z's size" for `y` and the full weight (minor)

- **Place:** §3.6: "A data-space object (y, the full weight, r, Jp) is of the type of z: each array of z's rank, each
  axis 1 or z's size (the engine checks y and the weights on entry)".
- **Evidence:** this is what the steps may assume (true, since exact is a special case), but it reads as the rule,
  which is now narrower: `y` 1 only on `C`, the full weight only on `W` and `C`.
- **Resolve:** "each axis `z`'s size or, on a stack axis, 1 (the engine checks `y` and the weights on entry)".

### 6. §3.11 misses one red node and cites one stale line (minor)

- **`tests/test_block_sums.py::TestAgainstDenseSums::test_engine_and_oracle`** puts the engine call and the oracle
  queries in one subtest.
  - Its `y` forms "shared by W" and "edge 1" (`:106-117`, `y_forms`) and its weights "per sample, a zero" (jet axes
    of 1) and "per block" (`blocks_to_weight`, an edge of 1 for probe) are refused from part 2: by the engine
    directly, and by the oracle through the bridge.
  - Part 2 rewrites the engine half. The oracle half of the same subtest stays red until part 4 (item 3 names only
    `TestSharedData`).
  - The engine assertion comes first in the subtest, so part 2's engine is still checked there before the oracle
    raises. The account should name the node, and the expected-red record will show it per subtest.
- **Stale citation:** item 4's example `tests/backend/test_optimizer_display.py:425` is now `:439`
  (`np.ones((1,) * m + (40, 1))`: a jet axis and an edge of 1).

## My scan: engine and `computed_names` calls with `weight=` (asked)

An `ast` scan of every call of `sampling_variables(` or `computed_names(`: in `tests/`, `t3toolbox/` (code, and
doctests parsed with `doctest`), `examples/`, and the doctests of `docs/*.md` and `*.rst`.

- **Direct `weight=`** in 13 test files:

  | file | calls |
  |---|---|
  | `test_block_sums` | 2 |
  | `test_contractions_sharding` | 2 |
  | `test_dispatch` | 7 |
  | `test_engine_data_space` | 16 |
  | `test_least_squares_oracle` | 4 |
  | `test_sampling_apply` | 1 |
  | `test_sampling_corewise` | 2 |
  | `test_sampling_entries` | 6 |
  | `test_sampling_graph` | 2 |
  | `test_sampling_identities` | 5 |
  | `test_sampling_stacks` | 1 |
  | `test_sampling_typed` | 1 |
  | `test_sampling_variables` | 3 |

- **Through `**`:** `test_sampling_uniform` builds `dict(..., weight=self.weight, ...)` at `:180`. The other `**`
  calls carry no weight key: `test_sampling_functions` and `test_held_variables` contain no "weight" at all. The
  engine's own doctests pass none.
- **The library:** only the oracle (`backend/fitting.py:561`, the bridge).
- **The examples and the pages** call neither function.
- **So the list is complete with the implementer's two additions** (`test_contractions_sharding`,
  `test_sampling_graph`). `test_sampling_apply` and `test_sampling_entries` are covered by the note's "the same engine
  tests" (its part-1 row names them). `test_held_variables`, in the note's part-2 list, has no weight and needs no
  change.

## §3.11's account, checked against the code

Items 1 to 4 hold:
- `prepare_derivative_data` places the `k!` table with `blocks_to_weight` (`jets.py:1178`).
- The translation builds probe weights with `blocks_to_weight` (`t3toolbox/optimizers.py:216-217`, `:237`).
- Apply and entries `blocks_to_weight` outputs have `F` exact and size-1 `W`/`C` only, so they are legal full weights
  with the same metric.

**A path not refused that would run with other numbers:** none in the shapes. The rule only narrows, and the bridge
sends `block_weight=None`. The one way to get other numbers is finding 1's dtype rule, if it were built as drafted
(a float32 weight with float64 data). That is why it should be fixed now.

## Checked and fine

- §3.1, §3.2, §3.3: the signature comments are the note's verbatim. The entries for `y` (exact but `C`; one `C`
  pattern) and `full_weight` (exact but `W` and `C`; one `W`/`C` pattern; never `K`, by rank) are the note's table
  exactly. "Agree in everything but the edge" is right for both objects.
- §3.4: the `reads` lines (112 and 114 columns). The table rows change only their `reads` cell; the `Type` column is
  kept. The held-`Jp` paragraph breaks one line, with no cascade. The uniform paragraph drops the "edge of 1" clauses,
  which no longer apply.
- §3.5's checkers carry the rationale of today's text (the right-alignment trap of `z - y`).
- §3.7 and §3.8:
  - each re-fill is forced: `w_tilde_step`'s first line would be 126 columns, and its second has no room;
  - `_with_K` keeps its line 2 byte for byte;
  - "the weights applied before the turn" is the definition my dense model checks.
- §3.9's alignment ("left-aligned: its axes are the first of `r_X`'s, every axis after them 1", the whole table
  against a packed `r_X`) agrees with my dense model, `k1/reviewer_tools/ref_dense_pins.py`.
- §3.10: the bridge's comment. Nothing else anticipates parts 3 to 5.
- Nothing true and useful of today's text is lost that I could find, except as noted in finding 5.

## What part 2's build review must check first

1. `jets._metric`'s dtype, as finding 1 resolves it, by a float32 block with float64 data. Then the battery's engine
   rows bitwise, dtypes included.
2. The engine against the reference (part 1's anchor):
   - `ref_dense_pins.py` covers the reference only;
   - the engine comparisons now pass the pair to both sides;
   - with `K` and the coinciding sizes, a block weight left-aligned against `Jp` with `K` (`_with_K` must not touch
     it).
3. The refusals at the engine's entry, cell by cell of the note's §4.2 table. The modes' `W` pattern is tested as a
   refusal on its own, since the reference accepts it.
4. Uniform: junk in the padding of `y` and the full weight. It must be finite until K5, and the mask is a product.
5. `test_block_sums.test_engine_and_oracle` (finding 6) and the per-subtest expected-red record.
6. The `Rule` rows in the form of finding 2. A whitespace-insensitive comparison of the rest of the table is unchanged.
