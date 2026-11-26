# Reviewer carryover (K arc), written before K1's design review

Reviewer 2 (Claude), 2026-11-25, about 21:15 CT. I reviewed K0's build: `K0_REVIEW_r1.md` and `_r2.md`, both SIGN
OFF; K0 is committed as `282b46a9`. I stopped here, at a clean point, before reading `K1_DESIGN_NOTE.md`: my context,
which holds the orientation and all of K0, would not carry K1's design rounds and its part reviews. This page holds
what a successor needs that the slice records do not. Read it with:
- `K_BRIEF.md`, then `BRIEF.md` ("What the reviewer does");
- `k0/REVIEWER2_PLAN.md` (my tools and my checklist, a model for K1's);
- `K0_REVIEW_r1.md` (what a build review established, and how);
- **`k1/REVIEWER_PRENOTES.md`** (my view of K1's risks, written before the note; the orchestrator asks the review to
  answer it point by point).

The predecessor's K0 carry-over, which this page replaces, is in git history (`282b46a9`).

## Where things stand

- **K0: done**, committed and pushed. Nothing of K0 is open.
- **K1: its design note is ready and unreviewed** (`K1_DESIGN_NOTE.md`, 1,242 lines, pinned to `282b46a9`). Its
  working files are in `k1/`: `k1_sites.py`, `sites_before_k1*.md`, `signatures*.{py,md}`, `check_math.py`,
  `kwonly_check.py`, and `prose_edits.md` (a partial draft). The orchestrator's request for review round 1 lists what
  it wants. In short:
  - the pre-notes answered point by point;
  - the design against `PACKING_PLAN.md` §2, §4.3, §4.4, §5, §7, §12-14 and `PACKING_PLAN_NOTES.md` §3d, §7;
  - the note's three amendments of the parts, checked by experiment: an old name deleted with its last library
    caller; a temporary `full_weight=batch.weight` in the oracle between parts 2 and 4; the translation moved to
    part 4. In particular, **is there a part boundary at which the tree runs and is silently wrong?**
  - independent dense checks of `M = γ b² f²`, the left alignment, `k!` at one and two jet variables, and `<y, M y>`
    with `y`'s `C` axes of 1;
  - the structural rule as stated at each enforcing place;
  - the batch's hand-written keyword-only `__init__` (Python 3.9), its static aux, `dataclasses.replace`, old
    positional calls;
  - Part 0 as a pure refactor;
  - an own scan of the site lists, including `getattr` and duck-typed readers;
  - the prose split;
  - the open points O1 to O4 and the decisions D1 to D15.
- Do not run `check_math.py` as evidence: write your own checks (the orchestrator's instruction).

## My tools (scratch: `/tmp/claude-1000/-home-nick-repos-T3Toolbox/7a03b046-9177-4c43-923a-b31595edb27e/scratchpad/reviewer2/`)

The scratch folder may be cleared. Each tool is short; rewrite it from this description if it is gone.

| Tool | What it does | Reuse for K1 |
|---|---|---|
| `unitdiff.py OLD NEW [--diff] [--comments] [files]` | every unit (module docstring, top-level or class-level statement, method; decorators, comments, blank lines inside) identical / changed / deleted / added, by exact text; comment blocks between units kept / gone / new | yes: every kept function's text, signature columns included |
| `refs.py ROOT [NAMES_JSON] [--docs]` | every occurrence of a set of names, tokens classified (code / doctest / string / docstring / comment / page), code hits resolved by `ast` to the owning module; run-time names (`getattr`/`hasattr` with a computed name, `%`/format/f-string templates) | adapt: K1's removed names (`blocks_to_weight`, `check_weight`, `pack_data`, `_pack_data_space`, `data_metric`, the old helper names) as a names file `{'deleted': {module: [names]}}` |
| `sim.py SRC OUT` | the mechanical cut of a frozen name list | K0 only |
| `prose_check.py ROOT [PROSE_MD]` | each indented "After" block of a `prose_edits.md` present EXACT (relative indentation kept) in its file; each "Before" absent | yes, on `k1/prose_edits.md` (the file path comes from the last `path:lines` in a heading or bullet) |
| `imports.py ROOT [nojax]` | imports every module, jax present or hidden; asserts the tree imported is ROOT's | yes |
| `roles.py PAGE...` | every `:py:` role resolved by import plus `getattr` (Sphinx `-W` is blind: `nitpicky` is off) | yes |
| `tally2.py OLDLOG NEWLOG OLDROOT NEWROOT` | per-node outcomes and subtest counts of two `-v` logs, each difference labelled by the static status of the test's text; an unchanged test whose record moved is flagged. It stops at the report sections (the FAILURES section echoes captured subtest lines). Its totals equalled pytest's on K0's logs | yes |
| `helper_users.py OLD NEW` | unchanged tests reached through a changed or deleted helper (same file; class-scoped; imported helper modules; transitive) | yes |
| `doctest_count.py ROOT` | docstrings with examples by `DocTestFinder` (pytest's count differs: 325 → 320 at K0, which is the one to state) | optional |
| `mut.py copy/step/patch/same` and `run_mut.sh NAME "MUTATION" TESTS...` | scratch copies; a step scaled by `1 + eps` (rebinding the module attribute: the engine looks steps up by name); exact single-occurrence patches from a JSON spec; a check that the copy differs from its source only by the mutation; the runner prints the failed subtests | yes |

Snapshots there: `head/` (the tree at `3a480e7c`, library as at `3375d213`) and `built/` (K0 as reviewed). Both can go
once K1 starts; take a fresh `git archive` of `282b46a9` as K1's "before".

## Lessons from K0 worth keeping

- **Compare code tokens, not diffs, when only prose should change.** An `ast` with docstrings blanked, plus tokens
  without comments and strings, settled K0's round 2 without a test run.
- **A `-v` log's validity for the final tree:** compare the files' mtimes (with seconds) against the run's start
  (its end less its duration). Then run only what changed after it.
- **Negative controls with several patches:** remove each patch in turn and see which rows go red. At K0 each
  masking layer alone zeroes the padding, so all three patches are needed, and the re-mask patch only on the tangent
  rows.
- **Floors that scan the library**, measured at K0 with the test modules' own functions:

  | Floor | At K0 |
  |---|---|
  | vocabulary `> 60` | 84 |
  | call sites `> 90` | 113 |
  | subscript literals `> 100` | 208 |
  | jet literals `>= 29` | 38 |
  | sharding cases `> 150` | 228 |
  | backend modules `> 30` | 47 |

  Re-measure in any slice that removes contractions (K5 above all: the call-site floor is the nearest).
- The dense reference (`sampling_definitions._weight_squared`) forms `ω²` itself and imports no `jets`. Keep it that
  way through K1 (pre-notes §1c).

## Carried to later slices

- **K5:**
  - point the hygiene negative control's three patches (`_UNIFORM_MASKS`, `_mask_variable`,
    `ufv_masking.ufv_apply_variations_masks`) at K5's masking points;
  - add the apply and entries cases of `TestKpAtTTSlack` before changing the masking (`K_ARC_DECISIONS.md` 11);
  - re-measure the floors above.
- **P17:** `sum_over_probes` in `tests/_derivative_data.py` and its callers (ruling 10); `chunk_size` in the
  translation; `tests/oracle_sweeps/oracle.py:146`'s "until the old kinds go".
- **P19:** the pages' prose naming deleted code (`K0_HANDOVER.md` §5 lists them; `signature_style.md` and
  `reading_signatures.md` quote `compute_mu`); `test_uniform_curvature.py`'s docstring naming
  `uniform_least_squares_problem` (a P16 leftover).
