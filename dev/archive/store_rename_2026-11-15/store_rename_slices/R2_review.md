# R2 review — the scripted pass (plan draft 24, §7 row R2, §5)

Reviewer, 2026-11-11. Reviewed:
- the working tree at `7a74a3b2` (R1 committed), which holds R2's scripted pass, uncommitted;
- `R2_handoff.md`;
- `R2_pass1.diff`.

Every check below is my own, run with HEAD's committed tools on a separate worktree and my own `v2026.4.0` battery
reference, independently of the implementer's runs. My scripts and outputs are in the session scratchpad under
`rv2/`, not in the repo.

## (a) Re-applying the approved table to a clean HEAD, then diffing against the working tree

- A `git worktree` at HEAD (`7a74a3b2`). HEAD's committed `dev/store_rename_apply.py --dry-run` regenerates a hit table
  that is byte-identical to the committed `R1_hit_table.tsv`. It is also identical to the table I approved in R1's
  third pass (4,896 hits, 283 keys, 0 mismatches, bijection errors or conflicts).
- My pass-2 tree against the working tree, file by file:
  - 77 of the 81 rewritten files are byte-identical;
  - the other 4 differ only by the **11 listed §4.6 hand edits**: the messages at `padded_frames.py:102` and
    `uniform_padded_frames.py:196, 225, 657, 682`, and the test regexes at `test_padded_frame.py:405, 409, 760` and
    `test_uniform_padded_frame.py:303, 320, 551`;
  - no other file differs. The only other modified files are in `dev/`: the new `--apply` code, and the plan, which
    the writer is editing.
- **The hand edits say the right thing.**
  - The messages name the Tucker store "down" exactly where they read it: `down_widths`, `counts[0]` and the `mD`
    masks.
  - The regexes match the messages after B2's `_STORE_NAMES = ('down', 'up', …)`. The test that corrupts `up[1]` /
    `masks[1]` now expects "up".
- All 81 files keep their line counts (in-place token replacement), and `git diff --check` is clean.

## (b) The pass-1 sentinel diff

- `R2_pass1.diff` equals my own pass-1 tree line for line: 3,054 `+` lines, 0 mismatches. Of these, 78 are unchanged
  context lines that `difflib`'s replace blocks carry. All 4,896 sentinels are in place.
- **Scoping, checked mechanically over every sentinel:**
  - none inside a §4A fence (column ranges honoured);
  - none on a doctest expected-output line (module docstrings and the docs pages);
  - none on a DNT name. The applied table is the one I proved disjoint from `dnt.txt` in R1, and K14 passes.
- **Read by hand.** I read all 41 non-exact STRING hits and all 12 doctest-comment hits in context.
  - The reprs swap their labels and their `%` arguments together (`UT3Frame(… nD=%d, nU=%d …)`).
  - The f-string reprs read `tucker_ranks={self.frame.down_ranks}`.
  - The test dicts' `'nU'`/`'nD'` keys swap with their reads.
  - The shape messages (`=? (d,) + stack_shape + (nD, N)`) are right in the new meaning.

## (c) Battery, gates, suite, doctests, docs, floor env

- **Battery** (working tree under test, my own reference): **PASS (bitwise)**, 2,190 arrays and 856 texts, B9's
  cross-version loads included, with `NEW=True`, `LETTERS_NEW=False`.
- **Gates** (HEAD's `store_rename_check.py --stage R2 --realized` on the working tree, with my pass-1 tree):
  - K0, K8b, K9-converse and K11 PASS (K11 also finds no sentinel inside a fence in pass 1);
  - K13 PASS (fences skipped), and K14 PASS (1,393 DNT rows; every `contract`/`einsum` string identical);
  - K15 and K15-realized PASS;
  - K17 PASS (891 checked), K17-shape-frame 0 and K17-shape-converse 0. K17-shape-point reports 5, exactly the
    corewise kernels that item 12's generic `n` covers in R4a/R4b.
  - The other K-checks are R3–R5 to-do lists.
- **Suite** at `-n 3` (t3toolbox env): **1,338 passed, 4 skipped, 48,697 subtests passed, 0 failed** (21 min). The
  one warning is the library's intended adam-on-manifold warning.
- **Module doctests**, the quickstart and the docs pages (CI's three commands): **241 module doctests passed**, `getting_started.rst`
  passed, and every `docs/*.md` and `docs/contributor/*.md` page except `doctest_style.md` passed. Static check: of the 3,441 module doctest
  examples, **no expected output changed**; only input lines changed, through the renames.
- **`sphinx -W --keep-going`**: **exit 0, no warning** (1,444 pages).
- **Floor env** (`tttt`: Python 3.9, numpy 1.22, jax 0.4.30), the 27 touched test modules plus
  `test_oracle_sweep.py`, in three serial shards of plain pytest: **1,066 passed, 5 skipped, 0 failed**. The shards
  were 300 passed and 1 skipped; 234 passed and 2 skipped; 532 passed and 2 skipped. Without `pytest-subtests`, a
  failing `subTest` still fails its test.

## (d) Every diagram and equation block

All 301 inventoried blocks were compared by me, byte for byte, between HEAD and the working tree (each column-ranged
block located by its text on its line):
- **all 300 fenced blocks are byte-identical**;
- the one block that changed is `FVF-1302`, the unfenced companion doctest of `fv_to_t3`. R2's letter rows rename it
  by design: the Tucker frame cores become `D0…D2`, the TT store `U0…U2`, and its expected outputs do not change.
  The diagrams beside it (`FVF-1264`, `FVF-1271`) keep their old letters until R4a, the planned interim state.

K11 agrees (300 fenced spans byte-identical to `v2026.4.0`, located by text).

## (e) Interim states

- The `update` letter API is old everywhere, untouched: the same 101 lines in HEAD and in the tree carry
  `'ULRO'`/`'U' in update`/`'O' in update`. Only the NAMEs beside them were renamed, and consistently. For example,
  the old `'U' in update` (the Tucker store) now sits beside the new Tucker names `PhiD`/`XiD`.
- Nothing was "fixed" beyond the 11 hand edits:
  - no realignment (the ±2-column drift is R4a's);
  - the fenced blocks keep their old letters and widths;
  - free-prose store words such as "per-mode up rank", "up cores untouched" and the subtest label `'a down width'`
    wait for R4a (K9);
  - stale comment labels (`# O.`, `# Q.`, `# dU_tilde`) wait for R4b.

## (f) Spot checks

- **Twenty swapped width comments** (random sample, seed 20261111), judged by the shape rule. All 20 are right:
  - `down_*`, the left/right cores' middle and the TT variations' middle carry `nD`;
  - `up_*`, the Tucker variations' rows and the up-side edge variables (`etas`, `dxi_tildes`) carry `nU`;
  - the corewise `_apply_tangent_assemble_step` line is on item 12's list.
- **Ten renamed names against the physical arrays**, at runtime on the working tree with unequal stores
  (`xS`: down `(3, 6, 2)`, up `(2, 4, 2)`; `UT3Frame.from_t3frame(F, 8, 7, 5, 6, 3)`). All 10 PASS:
  - `T3Frame.down_tucker_cores` is slot 0 with shape `(n, N)`, and `up_tt_cores` is slot 1, TT-shaped;
  - `down_ranks == (3, 6, 2)` and `up_ranks == (2, 4, 2)`;
  - `UT3Frame.down_tucker_supercore` is `(3, 7, 8)` and `up_tt_supercore` is `(3, 6, 5, 3)`, with `nD == 7`,
    `nU == 5`;
  - `masks.down_mask`/`up_mask` count the Tucker/up stores;
  - `UT3Variations`' Tucker rows are `nU`, and its down/up ranks match the frame's.

## Findings

No MAJOR, no MINOR.

**NIT.** `R2_pass1.diff` is not a pure `-U0` diff: `difflib`'s replace blocks carry 78 unchanged context lines (for
example, blank lines between two changed `def` lines). Every `+` line still equals the pass-1 tree, so the diff is
correct and reviewable. If it is kept, one sentence in the handoff saying so is enough; `git diff --no-index -U0` would
give the minimal form. This is optional.

**For the record (no action):** whether to commit the 532 KB `R2_pass1.diff` is the session's call (handoff §1). The
writer is recording draft 24's three missing rows (`dG1→H1`, the TT twins' `dG_tilde(s)→H_tilde(s)`; handoff §6);
the script already applies them, and I approved them in R1's third pass.

## For the maintainer's review stop (R2)

1. **The only non-scripted change: the 11 §4.6 hand edits** (handoff §3). Five backend messages now call the Tucker
   store "down"; six test regexes follow them.
2. **What the interim state looks like**, in two places:
   - `frame_variations_format.py:156–175`: the `T3Frame` fields and `up_ranks`/`down_ranks`. The NAMEs are renamed;
     the fenced field comments keep their old letters and drift ±2 columns (R4a realigns them).
   - `fv_to_t3` (`:1262–1322`): the companion doctest is in the new letters beside diagrams still in the old ones.
3. **A silent swap at work.** `F.down_ranks == (3, 6, 2)` (the Tucker store) and `F.up_ranks == (2, 4, 2)` on the
   battery's unequal fixture. `UT3Frame.from_t3frame(F, 8, 7, 5, 6, 3)` gives `nD == 7`, `nU == 5`. The battery is
   bitwise on slack frames.
4. **The shape rule in code.** `manifold.py`'s five transposes now return `V, H` (tier C-VH, 243 hits, e.g.
   `utv_sampling.py`, `probing.tv_probe_transpose`).
5. **If he wants every token touched:** `R2_pass1.diff`, by file.

## Verdict

VERDICT: APPROVE
