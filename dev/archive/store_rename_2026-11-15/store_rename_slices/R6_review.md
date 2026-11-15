# R6 review: acceptance (store rename, 2026.5.0)

Reviewer, 2026-11-13. Reviewed: the working tree on `e659d6ab` (HEAD; code and docs as R5's `6062f3d2`), which holds
R6 uncommitted:

- the alignment sweep: 215 whitespace-only lines in 50 files at first, 277 lines in 56 files after Revision 1;
- `R6_handoff.md`, `R6_alignment_sweep.md` and `r6_align_scan.py`;
- `/home/nick/repos/T3Polynomial/scripts/g01_g0_common.py`, uncommitted in that repo.

Read first: `CLAUDE.md`, the HANDOFF top block, plan draft 28 (§5, §6, §7 row R6), `R6_handoff.md`,
`R6_alignment_sweep.md`, R1–R5's reviews, and `t3toolbox/tucker_tensor_train.py` (the style reference).

Every check below is my own run. Heavy jobs ran one at a time. Scripts and outputs are in the session scratchpad under
`r6rev/`, not in the repo.

## Verdict in one paragraph

The rename is accepted. On the final tree:

- every K gate passes at stage R6;
- the battery is bitwise against the tag, in both environments;
- the suite, the slow tier, the module and page doctests, `sphinx -W` and all fourteen examples pass;
- no expected doctest output changed.

I re-ran the K14/K15 chain from the tag through R6 and reproduced the handoff's table exactly.

The one real finding was in the alignment sweep (MINOR 1): its scan paired adjacent lines only, so it missed columns
split by a comment-less line. The implementer fixed every item in Revision 1, and I re-verified the fix.

The T3Polynomial change is right, and its suite passes. What remains are NITs and items for the session.

## MAJOR

None.

## MINOR

### MINOR 1 — the sweep paired adjacent lines only (fixed in Revision 1, verified)

`r6_align_scan.py` pairs consecutive lines, and a doctest line with the one after its output. A column split by a
comment-less code line was therefore invisible, both as a near-miss and as drift. So the first pass's claims ("0
near-misses left", "0 tag alignments broken by the arc") held for adjacent pairs only.

My bridge scan, `r6rev/bridge.py`, reuses the R6 scan's feature extraction and pairs lines across one or two
comment-less, non-blank lines. It found 74 pairs 1–2 columns apart in the 109 files. `r6rev/bridge2.py` compared each
pair with HEAD and with the tag. Among them:

- **The sweep split three columns that were one at HEAD.**
  - `test_padded_frame.py:1041`: "of a pair" stayed at 89 while the rest of its sentence (`:1038–1039`) moved to 90.
  - `manifold.py:1089–1095`: J5's re-pad stopped at the comment-less `frame:` line, so one column became 57 and 59.
  - `test_sharing.py:556`.
- **The arc broke three tag alignments.** These are exactly the lines Nick's rule covers ("every block the rename
  touched is re-aligned"):
  - `fv_conversions.py:207`: "1. Tucker down-orth" ended up 2 columns off steps 2–4 (`up_r` → `down_r`).
  - `geometry.py:129` (`up[0]` → `down[0]`).
  - `padded_frames.py:297`: R2's `up` → `down` moved `:294` onto `:292`'s column, leaving `:297` behind.
- **Twins of the sweep's own fixes were left.**
  - `t3_svd.py:720`: `dense_t3svd`'s "graded-spectrum tensor" stayed at 56 under a block at 77. It is the third twin
    of J8's `:510` and `:620`.
  - `test_dispatch.py:1809, 1811`: the twin of `:1769–1771`.
- **Staircases inside one doctest or statement group**, e.g. `tv_operations.py:807–813` at 80/81/81/82.

**Revision 1, verified:**

- **Whitespace only.**
  - `git diff -w` is empty, and every changed `.py` file has HEAD's token stream.
  - Only 20 string tokens changed, all docstrings.
  - Every changed docstring line and every changed comment differs only in the gap before `#` (`r6rev/comcheck.py`:
    0 other differences).
- **No inventoried block was touched.** My re-run of every gate on the final tree passes, K10/K10b, K11 and K12
  (301/301) included, and so does `store_rename_diagram_check.py all`.
- **The scans are clean.**
  - My bridge scan now lists 19 pairs, all at different indents (a statement against a nested block, arguments
    against a return). None of them shared a column at HEAD or at the tag.
  - The adjacent NEAR pairs are only J1's 24 type pairs and J4's 6 `=` pairs.
  - Drift: the 2 deliberate J1 moves.
- **Read by eye:** A1–A4 and the signature blocks, e.g. `tucker_tensor_train.py:4479–4486`, `test_fitting.py:675–679`,
  `fv_operations.py:632–753`, `tv_operations.py:807–813` and `test_padded_frame.py:1038–1043`, `:1091–1093`.
- **The reports now state the adjacent/bridged distinction:** `R6_alignment_sweep.md` (Revision 1 section, J3b, the
  fix table at 277 rows) and `R6_handoff.md` §2. `r6_align_scan.py --bridge` reproduces the listing.

**The judgements J1–J9 against the style reference.**

- J1 (long names stick out with one space) is the reference's own habit: `tucker_tensor_train.py:3644`,
  `sum_over_probes:` in `probe_ambient_transpose`.
- J2 (stick-outs of 3 or more columns, left) also matches the reference. I read the 3-column cases (19), e.g.
  `fitting.py:241`, `ranks.py:372`, `optimizers.py:345`, `tucker_tensor_train.py:4315`, and none reads as a slip. The
  three-line comment at `test_padded_frame.py:1091–1093` was the exception; Revision 1 fixed it.
- J5's re-pads grow seven lines by 1–2 characters, to at most 139. No line is absurdly long.
- J6, J7, J8 and J9 read right.

I eyed well over 30 realigned blocks in all; every one now aligns the way the reference aligns.

## NIT

- **N1 — the plan and the HANDOFF disagree on when to commit the T3Polynomial change.**
  - Plan §7 row R6 (with §2) and `R6_handoff.md` §3 say it is "committed only after the release".
  - The HANDOFF top block says to commit it in that repo after R6, and push.
  - Either is harmless: the change renames two locals, and the function behaves the same against both library
    versions (see "My runs"). The session should pick one and record it.
- **N2 — `release_notes_2026.5.0.md` will need re-cutting at the release commit.** It equals the CHANGELOG
  `[2026.5.0]` section today. The rest of the 2026.5.0 bundle (the timing feature, the GN→Newton switch, the glossary
  pass) will add to that section before the tag.
- **N3 — a direct `--prev v2026.4.0` K14/K15 run fails by construction.** It reports 122 DNT tokens "lost", 3,237
  mixed hunks and K15-cross 231, because R2's in-place renames merge with R3/R4's hunks. Its contract/einsum strings
  check is still clean (0 files differ). The chain is the right gate, as the handoff says. I record this only so that
  nobody reads a tag-based run as a regression.
- **N4 — records in `R6_handoff.md` §1.3.**
  - The rendered-HTML paragraph says the hits are "in three pages". My wider pattern set (K1, K2, K3-docs, K4, K6,
    "outer-orthogonal", `Oᵢ`) also hits `entries_apply_probe.html`, at `entries_apply_probe.md:260–273` (`Oᵢ`, inside
    its §8 paper ↔ code map). On the release-notes page it also hits the past releases' CHANGELOG records (`[2026.4.0]`'s
    `('UL', 'RO')`, `[2026.2.0]`'s "Tucker up-SVD"). All are carved out (plan §5.3).
  - The "sources changed … the sweep's 20 comment realignments" row predates Revision 1 (now 37). §2 already says
    that the §1 runs predate Revision 1.
  - Optional.

## My runs (final tree unless noted)

| check | result |
|---|---|
| `git diff -w` (code, tests, docs, examples) | empty; 277 lines in 56 files, 0 NAME hunks against HEAD |
| `store_rename_check.py --stage R6 --prev HEAD`, on the final tree | every gate **PASS**: K0–K9 with K9-docs, K10/K10b (37 + 23), K11, **K12 301/301**, K13, K14, K15 0 hunks, K15-cross, K17 with the three shape checks; K16 REPORT 21 |
| K16 sign-offs | the 21 forms are exactly R5 handoff §2.2's table (#1–#4 hold the 10 PFC forms, #5–#15 the rest), signed off in `R5_review.md` (#15 explicitly); no `.tex` changed since `6062f3d2` |
| K14/K15 chain from the tag (fresh `git archive` of each commit; R2/R3 against the hit table without its 24 `R4a-row` rows) | tag→R2 K15 / K15-realized / K14 **PASS**; R2→R3 **36** hunks, R3→R4a **40**, R4a→R4b **36**, all listed, 0 unlisted; K14 PASS at every step (291 / 24 / 26 DNT tokens added by listed new code); R4b→R5 and R5→R6 0 hunks |
| battery, `t3toolbox` env, against R1's `battery_ref` | **PASS (bitwise)**: 2,190 arrays, 856 texts, 0 errors |
| battery, floor env `tttt` (py3.9 / numpy 1.22 / jax 0.4.30): my own tag extract `b9-save` + `run --b9`, then the tree `run --b9`; compared with the 3.11 Python | **PASS (bitwise)**: 2,190 arrays, 856 texts, 0 errors |
| suite, `pytest tests/ -q -n 3` (started on the first-pass tree; Revision 1 landed mid-run, token-identical apart from comment gaps) | **1,347 passed, 4 skipped, 48,724 subtests**, 0 failed (20 min 26 s); one warning, the intended adam-on-manifold one |
| slow tier, serial, `T3TOOLBOX_SLOW_TESTS=1` | **66 passed, 13,173 subtests** (14 min 45 s); two adam-on-manifold warnings, intended |
| module doctests, `pytest --doctest-modules t3toolbox` | **241 passed** |
| `python -m doctest docs/getting_started.rst`; every `docs/*.md`, `docs/contributor/*.md` but `doctest_style.md` | pass |
| expected outputs against the tag (static: `ast` docstrings for `.py`, whole text for `.md`/`.rst`, every non-`dev/` file changed since the tag) | **3,177 examples at the tag, 3,177 in the tree, 0 wants changed** |
| `sphinx -W --keep-going -b html` | **build succeeded**, exit 0 |
| rendered HTML (tags stripped; `_sources`/`_static` skipped) for K1, K2, K3-docs, K4, K6, `outer-orthogonal`, `Oᵢ` | hits on 4 pages, all carved out: `glossary.html` (the 2026.4.0 alias groups, N17's "outer-orthogonal"), `batching_and_stacking.html` §8 and `entries_apply_probe.html` §8 (paper ↔ code maps), `release_notes.html` (the upgrade section and past releases' CHANGELOG records); 0 in any API page |
| every example (CI's loop) | **14 of 14 PASS** |
| `release_notes_2026.5.0.md` against CHANGELOG `[2026.5.0] — unreleased` | equal byte for byte after stripping boundary newlines, as the 2026.4.0 precedent |
| version | `pyproject.toml`, `CITATION.cff` and the `__init__` fallback all read 2026.4.0 (unchanged, as planned) |
| §5.2 unequal-store assertions | present in all six named test modules |
| T3Polynomial `g01_g0_common.py` | the two lines agree: `down, up, left, right = frame` and `return tuple(down), tuple(up), …`. One line alone would hand positions 0 and 1 back swapped. HEAD's and the tree's `zero_base` give identical arrays on `xS`'s unequal-store frame (positions 0, 1 pass through; `left[-1]`, `right[0]` zeroed). No other renamed API use in that repo (grep). |
| T3Polynomial suite, from its root, `PYTHONPATH=T3Toolbox`, `-n 2` | **69 passed, 33 skipped, 547 subtests**; 1 error, the known `ufl` collection of `scripts/s02_probe_smoke_test.py` |

## Plan §5/§7 acceptance list

| item | status |
|---|---|
| §5.1 battery bitwise, current env | met (mine) |
| §5.1 battery in the floor env (§7 R6) | met (the implementer's and mine) |
| §5.2 unequal-store assertions | met (R3; present, and the suite passes) |
| §5.3 K0–K17 clean on code, docs and notes; K12 301/301; K16 signed off | met (mine, final tree) |
| §5.3 K14 DNT identity against the tag | met, through the chain (mine) |
| §5.4 suite at `-n 3`, current env | met (mine) |
| §5.4 suite, floor env | met (the implementer's: 1,346 passed, 5 skipped; the extra skip by design) |
| §5.4 slow tier, serial (the release gate) | met (mine) |
| §5.4 module doctests and every page doctest, both envs | met: current env mine, floor env the implementer's; Revision 1 changes only comment gaps on doctest source lines |
| §5.4 expected outputs unchanged | met (mine, static) |
| §5.4 dispatch rows (`_STEP_STATICS` at `test_dispatch.py:2418`), the import-graph test | met (in the suite) |
| §5.4 `sphinx -W` | met (mine) |
| §5.4 `gh run list` after the push | **open, the session's**: after the R6 commit |
| §5.4 PDFs | met at R5; no `.tex` changed since |
| §6 docs and release notes; root `release_notes_2026.5.0.md` = the CHANGELOG section | met; re-cut at the release (N2) |
| §6 the 601-row docs pass, then K2–K9 again | **after R6**, by plan (the 2026.5.0 bundle) |
| §7 R6: every example runs | met (mine) |
| §7 R6: the T3Polynomial follow-up prepared, its suite green | met; when to commit is N1 |
| §7 R6: HANDOFF | **open, the session's** |
| the alignment sweep (Nick 2026-11-12; R6's report) | met after Revision 1 (MINOR 1) |

## Agreed

The implementer agrees with the review. N4 is applied and verified: `R6_handoff.md` §1.3 now lists
`entries_apply_probe.html`'s §8 `Oᵢ` and the past releases' CHANGELOG records as carved out, and its realignment count
reads "20 in the first pass, 37 after Revision 1". The tree is unchanged by it (`git diff -w` empty, 277 lines in 56
files). N1–N3 are the session's.

VERDICT: APPROVE
