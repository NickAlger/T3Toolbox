# Reviewer handoff, data-metric sub-arc (written 2026-11-22, after `M2_part2_REVIEW_r1.md`)

For a successor reviewer, in case this reviewer's context runs out. The record of each slice is in its own files
(`M*_DESIGN_REVIEW.md`, `M*_REVIEW_r*.md`); this note gives only what is not obvious from them.

## Where things stand

- M0 and M1 are committed.
- M2 is built in parts on `m2-wip`:
  - part 1 signed off (`M2_part1_REVIEW_r2.md`);
  - part 2 at `NOT YET`, round 1: one should-fix, the join across the modes in `_pack_data_space` is untested. Rerun
    the packing mutant (below) once a test is added.

## Method that has worked

- **An independent brute force for every numerical claim**, with the weight expanded by `np.repeat`, never by
  broadcasting. Scripts are in `scratchpad/m2rev/`:
  - `check_ref.py`: the reference module;
  - `check_p2.py`: the engine with `C`, `K`, equal sizes (`C = W = F = N_1 = 3`), and uniform against ragged with
    mixed edges.

  Rerun `check_p2.py` after each later part; it uses only engine names.
- **Mutants on a scratch copy** (`cp -r t3toolbox tests pyproject.toml` into `scratchpad/m2rev/new`, then mutate a
  copy). Run only the relevant test files at `-n 2`. The pattern is in `scratchpad/m2rev/mutants_p1.sh`.
- **The red lists.** Parse both progress and summary lines of the `-v` log (`FAILED|SUBFAILED[...]|ERROR`, and the
  `FAILED tests/... - ...` summary) and compare the set with the listed ids. Then tally every `^E ` exception line:
  anything that is not the stated cause must be read, since it may be a downstream consequence.
- **The tallies.** Compare node outcome counts between consecutive `-v` logs, which are in `scratchpad/m2impl*/`.
  First check that no file in `t3toolbox/` or `tests/` is newer than the log.
- **The battery** (`implementation/m2/battery.py`). The pre-M2 record is `scratchpad/m2impl/battery_old.npz`; the
  base tree is `scratchpad/m2impl/base`, equal to `85c6e54d`. It is deterministic, and I re-recorded it myself with 0
  bytewise differences. The final comparison is part 5's.

## Open items to watch in M2 parts 3 to 5

- **Part 3:**
  - M1's bitwise bridge test must be re-pinned against a written-out old formula: cast, then square, then reshape to
    `F + (1,)*rest`; the four float32/float64 pairings. Then M1's L4 mutant (square before the cast) must still fail.
  - The factorials go on the leading jet axes; check them with distinct and equal sizes.
- **Part 4:**
  - `take` with a partly broadcast weight, checked against index loops;
  - `f_resolution` over `z`'s shape (`y` broadcast to `r`; correction 3);
  - `pack_batch` must never zero-pad an edge of 1;
  - the oracle tests' batches feed the engine directly too.
- **Part 5:**
  - the battery `--compare` under design-review finding 2: bitwise by default; `f`, `Jp_norm_sq`, `f_resolution` to
    1e-13; jit 1e-12; optimizer runs with exact counts and stop reasons and 1e-10 iterates;
  - the translation's 19 pins;
  - the diff audit: converted tests change only `y` and `weight`.
- **M3:** on stacked models with a shared `y`, the display ratio is already off by `√|C|` today: `r_block_sumsq` sums
  over `C`, `y_block_sumsq` counts `y` once. The `y` broadcast fixes it.

## Laptop rules

- `-n 2` at most.
- Never run tests at the same time as the implementer.
- Never wait with `pgrep -f` on your own pattern.
- Leave nothing running.
