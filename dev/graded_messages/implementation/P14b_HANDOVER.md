# P14b: `computed_names` in place of `keep=` — handover

Slice P14b, built to `TARGET_DECISIONS.md` ("The hold list: `computed_names`, not `keep=`"), `P14b_DESIGN_NOTE.md`
and the coordinator's rulings:
- the signature is identical to the engine's, `sample` included;
- `_closure_facts` is shared by the engine and `computed_names`;
- the oracle passes one shared dictionary to both;
- there is no test that `keep` is gone.

`frame_form` is unchanged. Written by the implementer, 2026-11-21, against `graded-messages` at `0e71fec8`. Nothing is
committed.

## 1. What changed

**The engine** (`backend/sampling_variables.py`):
- **`keep=` removed:** from the signature, the docstring and the return. `sampling_variables` returns exactly `want`
  and reuses `held`.
- **`_closure_facts(kind, frame, want, order, y, variations, core_directions, Vc, held)`** is the one reading of the
  call's facts and its closure, and both the engine and `computed_names` call it.
  - **It checks** the kind, the jet shape, `want`'s type and names, `held`'s type, the frame's form (`frame_form`), a
    direction's form on a corewise point, and, from the closure, a wanted name's missing argument.
  - **It reads** the call type, `held`'s filled slots (and `held:Jp`), the presence of `y`, `Vc`, `variations` and
    `core_directions`, and whether the sample is `'w'`.
  - **It returns** the closure.
  - The engine's array-reading checks (the entries index, packing, the setting, the stacks, the held shapes) stay in
    the engine, after it.
- **`computed_names(kind, frame, sample, want, order=(), y=None, weight=None, variations=None, core_directions=None,
  Vc=None, sum_over_samples=True, held=None) -> frozenset`** is public and in `__all__`.
  - The signature is the engine's; `sample`, `weight`, `order`'s value and `sum_over_samples` are accepted and not
    read.
  - It returns the public names of `_closure_facts`' closure, so held names (never computed) and `'w'` are excluded.
  - It raises the engine's errors with the engine's messages.
  - Its doctest shows a line-search `('f',)` call against `('f', 'g')`, with what `g` adds, and the oracle's pattern
    (one dictionary, the free hold-list names added and split off).

**The oracle** (`backend/fitting.py`, `LeastSquaresOracle.query`) builds one dictionary `args` for both calls.
1. It sets `free = computed_names(kind, frame, sample, engine_want, **args)`, and
   `extras = HOLD ∩ free - engine_want`.
2. It makes one engine call for `engine_want + extras`.
3. It returns `held` as the `held` passed in plus `HOLD ∩ free`, taking the values of both the extras and any wanted
   hold-list name.

## 2. Tests

**`tests/test_least_squares_oracle.py`:** P14's `test_keep` is replaced by three tests, and `TestQuery`'s
held assertion now reads `computed_names`.
- **`test_computed_names_is_what_the_engine_runs`.** The engine's `_run` is wrapped to record the rules it runs. The
  public names it ran must equal `computed_names`, which must be a `frozenset` disjoint from `held` with no private
  name. This is checked over more than 150 calls (asserted in the test):
  - every kind;
  - the four layouts and call types (ragged and uniform, manifold and corewise);
  - eight want patterns: `('f',)`, `('f', 'g')`, `('z',)`, `('JtJp',)`, `('Jp_norm_sq',)`, `('Kp',)`,
    `('g', 'JtJp', 'Kp')` and `('r', 'f')`;
  - the routes: plain, `Vc` (probe `z`), `JtJp` from a held `Jp` with no direction, and held `r` and `mu`.
- **`test_computed_names_raises_as_the_engine`:** seven cases (an unknown name, a missing `y`, `variations` on a point,
  entries with jets, an unknown kind, a bad frame, `want` as a string). Each raises the engine's exception type with
  an identical message.
- **`test_the_oracle_holds_what_is_free`,** for the three kinds and four want patterns:
  - the engine rules the oracle runs are exactly `computed_names(want)`: holding adds no work;
  - `held.keys() == HOLD ∩ computed_names`;
  - each held value is bitwise a fresh engine call's;
  - a later query passes the held arrays back as the same objects, and adds exactly `HOLD ∩ computed_names(('g',),
    held=held)`.

**`tests/test_dispatch.py`:** the `keep=` lines are replaced by an assertion that the oracle's held slots under jit
equal the eager ones.

## 3. Mutation pass (scratch copy, deleted)

6 mutants, all killed by `tests/test_least_squares_oracle.py`:
- `computed_names` returning private names, ignoring `held`, or ignoring `Vc`;
- the oracle holding only the extras (it drops a wanted hold-list name);
- the oracle dropping the held passed in;
- **the oracle adding the whole hold list to `want`** (computing to hold). This one survived the first pass: `held`
  was filtered by `free`, so neither the values nor `held` changed. It is killed by the added assertion that the
  oracle's engine call runs exactly `computed_names(want)`.

## 4. Environments and gate

- **`t3toolbox`**: the new file passes (17 tests, 285 subtests), and the module doctests pass.
- **`tttt`**: `test_least_squares_oracle` and `test_sampling_variables` pass (32), and so do the doctests of the two
  modules (6).
- **The full gate at `-n 2`**: below.

The full gate at `-n 2` (logs in `scratchpad/gate_P14b2`) passed every step:
- **suite**: 1857 passed, 5 skipped, 1 xfailed, 90,794 subtests, in 2118 s; peak 8.1 GB; **lowest MemAvailable
  3902 MB**;
- **doctest-modules**: 320;
- **quickstart**: pass;
- **pages**: 47;
- **examples**: 14;
- **sphinx**: 0 warnings.

## 5. Added after the gate: `frame_shape` (the maintainer's request) — NOT YET RUN

- **The reader.** `sampling_variables.frame_shape(frame) -> tuple of int`, public and in `__all__`, beside
  `frame_form`, which is unchanged. It reads the mode sizes from the last axis of each Tucker core on ragged, and the
  static `shape` element (`frame[-2]`) on uniform, and raises as `frame_form` does. Its doctest covers all four
  forms.
- **The oracle** unpacks a packed probe `y` with `frame_shape(frame)` in place of `frame[4]` / `frame[2]`.
- **The test.** `test_frame_shape` in `tests/test_least_squares_oracle.py` covers the four forms, unstacked and
  stacked, uniform padded to `N = 7` (the static shape, not `N`), with plain `int` entries, and bad structures raising.

**Run on the coordinator's "go":**
- `tests/test_least_squares_oracle.py` passes on both environments: 18 tests on each (293 subtests on `t3toolbox`).
- The module doctests of `sampling_variables.py` and `fitting.py` pass on both (7 each), including `frame_shape`'s,
  with no output to fix.
- The gate's `sphinx` step passes with 0 warnings (logs in `scratchpad/gate_P14b_sphinx`).

## 6. For P16

**`aligned_padded_frame` returns `(frame, centers)`,** the old centers zero-padded. This reverses the P15 ruling, and
supersedes `P15_HANDOVER.md` §1 and §4, which say it keeps returning a frame only. The note is recorded with the
others in `TARGET_DECISIONS.md`, "The notes carried to P16, settled", item 3.

## 7. The plan brought in line with 2026-11-21 (`PART1_PLAN.md`; minimal edits to the slices' text)

1. **The oracle's text (after section 2.2's sketches):** the hold-list names the call computes anyway are reported by
   `computed_names(kind, frame, sample, want, **arguments)`, in place of an engine `keep=`, and the engine returns
   exactly `want`.
2. **Section 2.4's table of new names:** a row for `computed_names`, `frame_form` and `frame_shape`
   (`sampling_variables`).
3. **P14:** one sentence that the engine gains `frame_form`, `frame_shape` and `computed_names`; the test of `held`
   reads "the names of the hold list in `computed_names` of the call".
4. **P16:**
   - item 1: the aligned padded frame returns `(frame, centers)`, the old centers zero-padded (zeros at the cone
     start), with a test of the consistency residual at a padded start;
   - item 2: the held-across-draws test, and `pack_sample` and `pack_data` with `(kind, order, ...)`;
   - item 3: the optimizer adds `eps |rho|` to the oracle's `f_resolution`.
