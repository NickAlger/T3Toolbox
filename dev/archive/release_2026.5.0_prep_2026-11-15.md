# The 2026.5.0 release commit — prepared as text (2026-11-15)

Drafted on HEAD `7807b238` (the glossary/docs pass `8ad6dea3` committed; the 17 diagram hand fixes still in flight).
Nothing here is applied. Precedents read: the 2026.4.0 release commit `e364fe01` (tag `v2026.4.0`) and the 2026.3.0
one `ad578f63`; `CHANGELOG.md` `[2026.5.0]` as it stands; `W4_review.md` and `W3_handoff.md` §2–§5 for the figures.

**Preconditions.** (a) The hand-fix agent's commit has landed: it edits `t3toolbox/`, `docs/` and
`dev/store_rename_slices/diagrams_inventory.md`, and it may add a CHANGELOG line. (b) Nick has passed the timing
FINAL STOP and the switch W4 review stop; `dev/HANDOFF.md` still lists both as pending. (c) No other heavy job is
running: the laptop runs one of suite, docs build or examples at a time, `-n 3` at most.

## 1. Version strings

There are four sites. `docs/conf.py:16–18` reads `release` from `pyproject.toml` with `tomllib`, so it needs no edit.
No test asserts the version, and `release.yaml` keys off the tag. Line numbers are as of `7807b238`.

| file:line | before | after |
|---|---|---|
| `pyproject.toml:14` | `version = "2026.4.0"` | `version = "2026.5.0"` |
| `CITATION.cff:18` | `version: 2026.4.0` | `version: 2026.5.0` |
| `t3toolbox/__init__.py:56` | `    __version__ = '2026.4.0'` | `    __version__ = '2026.5.0'` |
| `CHANGELOG.md:6` | `## [2026.5.0] — unreleased` | `## [2026.5.0] — 2026-11-15` |

The CHANGELOG heading uses an em dash (U+2014), as every earlier heading does, not ` - `. Use the date Nick tags; if
that is not 2026-11-15, change the heading and re-cut the release notes (§3).

`CITATION.cff` has no `date-released` field (2026.4.0 did not add one). `t3toolbox.__version__` reads the
installed metadata first. An old editable install therefore reports a stale version (the HANDOFF saw 2026.0.0). Use
the fresh-venv wheel check (§5, step 8) for the version, not the dev env.

## 2. The CHANGELOG intro for `[2026.5.0]`

This replaces `CHANGELOG.md` lines 8–10 (the rename-only intro). The sections below it stay as they are, in this
order: Changed — breaking / Changed / Fixed / Added (the 2026.3.0 order). `### Fixed` stays. The intro is 22 lines
and ≤ 115 columns.

```markdown
Three headlines. **The store rename** (breaking): the frame stores `up` and `down` swap names, so that a store is
named for the side of its edge that its cores occupy, as `left` and `right` always were. Some names keep their
spelling and now mean the other store, so stale code need not raise. Every number is bit-identical; names, letters
and messages change. Upgrade notes: `docs/release_notes.md`, "Upgrading from 2026.4.0". **Phase timing for
`newton_cg`** (`timing=True`): each row's wall time split into phases (the local model's build, the gradient, CG,
the trials, the callback, the rest), with jax's compile wait beside each, at a cost within the run-to-run noise
(user guide: `docs/fitting_and_optimization.md` §4.13). **Starting with Gauss-Newton and switching to the exact
Hessian** (`hessian='switch'`): `newton_cg` switches for good once the exact model has predicted the objective
clearly better on consecutive steps (§4.14). Both additions are opt-in; no default changes.

**Measured** (the switch; A100, the uniform layer under `use_jit` in x64, `gtol_rel=1e-3` on the default `'zero'`
reference, at most 300 Newton steps a level, the Darcy rank-continuation ladder of the downstream T3Polynomial
project, 2026-11-14). On 10 points × 100 directions (six levels to rank 24) from the library's own start frame, the
switch took 223 Newton steps and 6,303 CG iterations in 177 s (net of compile and display), against 267 / 10,751 /
275 s for `hessian='exact'` and 1,625 / 24,682 / 427 s for Gauss-Newton with the line search (the default), which
stopped at the step cap on 4 of the 6 levels; in Gauss-Newton apply-equivalents at measured cost, 19,794 against
33,309 and 24,682. It switched on every level after 5–8 Gauss-Newton steps, its tracking took 3 % of its time, and
at the level the validation error selected its errors were no worse than Gauss-Newton's. The same pair on a laptop
CPU agreed: 219 against 277 Newton steps, 13,068 against 16,777 s. **It is not a uniform win:** restarted from the
aligned frame (`padded_from`), the exact Hessian took 231 Newton steps to the switch's 303, and at 20 points × 1000
directions (eleven levels, inner TT rank to 50) the two were even at both starts (switch / exact 287 / 291 Newton
from the library start, 308 / 288 from the aligned), with the same best test error.
```

**Where each figure comes from** (all are in W3, and none was recomputed here). The library-start table is W3 §2.2,
the same as §4.14's table: switch 223 / 206 + 6,097 / AE 19,794 / 177 s; exact 267 / 10,751 / 33,309 / 275; GN
1,625 / 24,682 / 24,682 / 427, at CAP on 4 of 6. `switched_at` 5, 8, 8, 7, 6, 5 (§2.5). Tracking 3.3 % of net wall
(§4.3). The errors at the selected level 4 are val_fsr 0.3697 vs 0.3736 and test p50 0.0675 vs 0.0689, in the W4
review's M2 wording. The laptop pair is §2.1 (net 13,068 / 16,777). The aligned start is §2.3 and §5.1. The
20 × 1000 runs are §2.4 and §5.2: best test p50 0.0449–0.0450 in all four runs. The 2026.4.0 intro was the model for
the style.

**One choice for Nick.** The timing overhead figures (−4 % to +2 % laptop, −1.4 % to +3.1 % A100) are already in
the `### Added` timing entry. Following the W4 review (a feature's own cost goes in its entry), the intro only says
"within the run-to-run noise". To put the figures in the intro anyway, end **Measured** with: "Phase timing on
against off differed by −4 % to +2 % on the laptop and −1.4 % to +3.1 % on the A100, within the run-to-run noise."
That makes 24 lines.

**Known caveat, stated in the docs and not in the intro.** GN's counts are lower bounds because it stopped at the
cap. The intro says that it stopped at the cap and leaves out the "lower bound" wording.

## 3. Re-cutting `release_notes_2026.5.0.md`

**The rule, inferred and checked.** The file is the body of the CHANGELOG `[2026.5.0]` section: every line after the
`## [2026.5.0] — …` heading and before the next `## [` heading, with leading and trailing blank lines removed. The
command below rebuilds the untracked `release_notes_2026.4.0.md` **byte for byte** from
`git show v2026.4.0:CHANGELOG.md`. `release_notes_2026.3.0.md` is the same cut but keeps one blank line at each end
(the raw `awk` output). 2026.4.0's form is the later one, so use it. No script in `dev/` or `scripts/` cuts these
files, and `docs/release_notes.md` does not need a cut: it `{include}`s the whole CHANGELOG from line 5.

Run it from the repo root after the **last** CHANGELOG edit, i.e. after the date stamp and the intro:

```bash
awk '/^## \[2026\.5\.0\]/{f=1;next} /^## \[/{f=0} f' CHANGELOG.md \
  | sed -e '/./,$!d' | sed -e :a -e '/^\n*$/{$d;N;ba' -e '}' > release_notes_2026.5.0.md
head -1 release_notes_2026.5.0.md      # "Three headlines. **The store rename** ..."
grep -c '^## ' release_notes_2026.5.0.md   # 0
```

The current file is stale. It is the R6-era cut: 91 lines, without `### Fixed` or `### Added`. Today's section is
166 lines, before the new intro.

**Tracked or not: untracked**, by convention. `release_notes_2026.3.0.md` and `release_notes_2026.4.0.md` sit
untracked in the root. They are not gitignored; they are left out by practice. The 2026.4.0 commit message says so:
"The GitHub release text is release_notes_2026.4.0.md (untracked, as for 2026.3.0)." The file is the GitHub Release
body, which Nick pastes. Never stage it.

## 4. Archiving the plans

**The precedents.**
- 2026.3.0's release commit (`ad578f63`) renamed the plan to `dev/archive/<plan>_<date>_shipped.md` and prepended a
  SHIPPED banner.
- The same commit moved the arc's review folder whole to `dev/archive/curvature_impl_review_2026-10-29/`, archived a
  HANDOFF snapshot as `dev/archive/handoff_2026-10-29_pre-2026.3.0.md`, and rewrote the HANDOFF.
- The 2026-11-09 sweep (`a2e26ca8`) moved reviews flat into `dev/archive/` and slice folders to
  `dev/archive/<slices>_2026-11/`.
- 2026.4.0's release commit archived nothing (see §6 e).

The proposal below follows 2026.3.0: each plan becomes `_shipped` at the archive's top level, and the rest of its arc
goes into one dated folder with the file names unchanged.

```bash
# store rename (84 tracked files)
git mv dev/store_rename_plan_2026-11-10.md dev/archive/store_rename_plan_2026-11-10_shipped.md
mkdir -p dev/archive/store_rename_2026-11-15
git mv dev/store_rename_plan_review_r*.md dev/store_rename_inventory_*.md dev/store_rename_sample_edits*.md \
       dev/store_rename_letter_sites.md dev/store_rename_pqu_inventory.md dev/store_rename_dnt.txt \
       dev/store_rename_keeps.txt dev/store_rename_apply.py dev/store_rename_battery.py dev/store_rename_check.py \
       dev/store_rename_diagram_check.py dev/up_down_naming_blast_radius.md dev/store_rename_slices \
       dev/archive/store_rename_2026-11-15/
# phase timing (9)
git mv dev/newton_timing_plan_2026-11-10.md dev/archive/newton_timing_plan_2026-11-10_shipped.md
mkdir -p dev/archive/newton_timing_2026-11-15
git mv dev/newton_timing_plan_review_r1.md dev/newton_timing_plan_review_r2.md dev/newton_timing_slices \
       dev/archive/newton_timing_2026-11-15/
# GN-to-Newton switch (11)
git mv dev/gn_newton_switch_plan_2026-11-13.md dev/archive/gn_newton_switch_plan_2026-11-13_shipped.md
mkdir -p dev/archive/gn_newton_switch_2026-11-15
git mv dev/gn_newton_switch_plan_review_r1.md dev/gn_newton_switch_plan_review_r2.md \
       dev/gn_newton_switch_source_notes_2026-11-13.md dev/gn_newton_switch_slices dev/archive/gn_newton_switch_2026-11-15/
```

**Notes on the moves.**
- A directory `git mv` also carries the untracked `store_rename_slices/R2_pass1.diff` (532 KB, regenerable) and the
  ignored `battery_ref/battery.npz` and `__pycache__/`. Do not `git add` the moved folder afterwards, because that
  would stage the diff. Deleting the diff first is Nick's call.
- The archived `store_rename_*.py` tools hard-code `dev/store_rename_slices` and stop working from the archive.
  That is expected: they are records, like 2026.3.0's archived checks. No test or CI job imports them.

**Banners, prepended** (the 2026.3.0 form):
- Rename: "> **SHIPPED in 2026.5.0 (tag `v2026.5.0`, <date>); archived at the release.** The durable record is the
  CHANGELOG `[2026.5.0]` "Changed — breaking", `docs/release_notes.md` "Upgrading from 2026.4.0" (users),
  `docs/frame_variations.md` § Four rank stores (the convention) and `docs/glossary.md`. The reviews, inventories,
  tools and slice records R1–R6 are in `dev/archive/store_rename_2026-11-15/`. Paths below are as they were while the
  plan was live."
- Timing: the same form, with the durable record `docs/fitting_and_optimization.md` §4.13 and
  `docs/contributor/scan_body_principles.md` (the clock slot), and the reviews and P1–P3 records in
  `dev/archive/newton_timing_2026-11-15/`.
- Switch: the same form, with the durable record §4.14, `docs/contributor/curvature_internals.md` (the Gauss-Newton
  view) and `docs/contributor/deferred_and_rejected.md` (I-36; what 2026.5.0 left out). The reviews, the source notes
  and W1–W4 are in `dev/archive/gn_newton_switch_2026-11-15/`. This makes the ledger's claim at
  `deferred_and_rejected.md:225` ("archived under `dev/archive/` with the 2026.5.0 release") true.

**After the moves.** No shipped file points into these live paths. A grep of `t3toolbox/`, `tests/`, `examples/`,
`docs/`, `README.md`, `CHANGELOG.md` and `CLAUDE.md` finds only §6 e's test docstring.

**Also recommended** (Nick's call; not among the three named plans):
- **HANDOFF.** `cp dev/HANDOFF.md dev/archive/handoff_2026-11-15_pre-2026.5.0.md`, then rewrite `dev/HANDOFF.md` to
  the post-release state. That state is: the audit items; the 69 deferred docs rows; a jitted tracking product as a
  future option; I-36 open at the aligned start; the post-2026.3.0 backlog.
- **The glossary pass.** Archive its records to `dev/archive/glossary_docs_pass_2026-11-15/`:
  `glossary_candidates_{A,B,C}_*.md`, `glossary_review_r{1,2,3}.md`, `glossary_docs_pass_2026-11-15.md`,
  `glossary_docs_pass_review_2026-11-15.md`, and the hand-fix record once it is committed. Keep
  `glossary_naming_decisions.md` and `_status.md` live, because they hold the 69 open rows.
- **2026.4.0's records** (§6 e). Move `aligned_padded_frame_impl_plan_2026-11-09.md` to `_shipped`. Move the six
  reviews, `aligned_padded_frame_slices/` and `padded_frame_cone_judgment_calls_2026_11.md` to
  `dev/archive/aligned_padded_frame_2026-11-10/`.
- **This file.** Move it to `dev/archive/release_2026.5.0_prep_2026-11-15.md` in the release commit, as 2026.3.0
  archived its `REL_2026.3.0_handoff.md`.

## 5. Release checklist, in order

S = the session, N = Nick.

1. **S, edits.**
   - §1, §2, §4.
   - The README and `docs/index.rst` mentions (§6 i). Both earlier release commits mentioned the headline there.
   - `CLAUDE.md` "Current state" (§6 c).
   - Any optional items Nick accepts (§6 g, h).
   - Re-cut the release notes last (§3).
2. **S, suite.** `PYTHONPATH=$PWD <env-python> -m pytest tests/ -q -n 3`. `<env-python>` is the `t3toolbox` env
   (py3.11, numpy 2.4.6, jax 0.10.2), the CI current-leg twin.
3. **S, slow tier.** Required, and run serially:
   `T3TOOLBOX_SLOW_TESTS=1 PYTHONPATH=$PWD <env-python> -m pytest tests/test_oracle_sweep.py tests/backend/test_uniform_curvature.py -q -p no:cacheprovider`.
4. **S, floor suite.** Single process, on the `tttt` env (py3.9, numpy 1.22.4, jax 0.4.30), as 2026.4.0 did:
   `PYTHONPATH=$PWD <floor-python> -m pytest tests/ -q`.
5. **S, doctests, on both envs.**
   - `pytest --doctest-modules t3toolbox -q`
   - `python -m doctest docs/getting_started.rst`
   - `python -m doctest $(ls docs/*.md docs/contributor/*.md | grep -v doctest_style.md)` (48 pages)
   - `docs/chunking.md` by hand on the floor env (CLAUDE.md).
6. **S, docs build.** `python -m sphinx -W --keep-going -b html docs <scratch>`, which takes about 18 min. It must
   exit 0 with no warning. Check that the rendered pages say 2026.5.0 (`index.html`, `release_notes.html`).
7. **S, examples.** `for f in examples/*.py; do PYTHONPATH=$PWD <env-python> "$f" || echo FAIL $f; done`. All 14
   need optax and scipy, as CI's examples job does.
8. **S, wheel smoke** (REL-2, as at 2026.4.0).
   - `python -m build --outdir <scratch>`, then `twine check`.
   - The wheel holds the 55 package modules and nothing else. No module was added since `v2026.4.0`.
   - In a fresh numpy-only py3.11 venv, it reports 2026.5.0, fits, and warns and runs eager on `use_jit`.
   - Smoke `newton_cg(hessian='switch', timing=True)` there too: the new paths.
9. **S, commit.** Stage by name and never use `-a`. Stage: the four version files, `CHANGELOG.md`, `README.md`,
   `docs/index.rst`, `CLAUDE.md`, `dev/HANDOFF.md` with its archive snapshot, the edited moved plans (their banners),
   any accepted optional files, and this file's archived copy. The `git mv` renames are already staged. Never stage
   `release_notes_*.md`, `*.npz`, `R2_pass1.diff`, `dev/check_t3svd_retraction_second_order.py` or
   `dev/graded_messages_handoff/`. Message in the house style of `e364fe01`:

   ```
   release: T3Toolbox 2026.5.0 — the store rename, phase timing and the Gauss-Newton-to-exact switch

   Version 2026.5.0 in pyproject.toml, CITATION.cff and the __init__ fallback;
   CHANGELOG `[2026.5.0] — <date>` (Changed — breaking / Changed / Fixed /
   Added, with the three headlines and the measured paragraph in the intro;
   the upgrade steps in docs/release_notes.md "Upgrading from 2026.4.0");
   docs/index.rst and the README mention the switch and the timing.

   What ships: BREAKING -- the frame stores up and down swap names (a store
   is named for the side of its edge its cores occupy): same-spelling silent
   swaps (up_ranks / down_ranks, the frame weights and masks, nU / nD) and
   renames that raise (down_tucker_cores, up_tt_cores, the supercores, the
   probing and jet keywords); `update` takes a tuple of the store letters
   'D', 'U', 'L', 'R'; every number bit-identical; letters D / U / L / R,
   V / H, B / G in the docs, docstrings and the six math notes. Opt-in:
   newton_cg(timing=True) (NewtonInfo.timing, stats['timing'], the display's
   time line and closing table, the members' timing keyword,
   common.tree_block_until_ready); newton_cg(hessian='switch') with
   switch_tau / switch_k / switch_cg_forcing_power / switch_track, the
   per-row model and the switch_* fields, stats['switched_at'], the
   display's sw segment. Fixed: the 'cone' converged line; eight messages.
   The glossary and docs-consistency pass (505 of 599 rows).

   Measured (A100, uniform use_jit, x64, gtol_rel 1e-3 on 'zero',
   T3Polynomial's Darcy ladder, 2026-11-14): 10 x 100 from the library
   start, Newton / CG / net s: switch 223 / 6,303 / 177 vs exact 267 /
   10,751 / 275 vs GN 1,625 / 24,682 / 427 (at the cap on 4 of 6 levels);
   from the aligned start exact 231 vs switch 303 Newton; 20 x 1000 even
   (287 / 291 and 308 / 288); the laptop pair 219 / 277 Newton, 13,068 /
   16,777 s. Timing on against off: -4 % to +2 % (laptop), -1.4 % to
   +3.1 % (A100), within the noise.

   Verification at this commit: <from steps 2-8: tests / subtests (-n 3) on
   py3.11 / numpy 2.4.6 / jax 0.10.2; the floor's count; the slow tier;
   module doctests, getting_started and the 48 doc pages on both envs;
   sphinx -W clean; examples 14/14; the wheel's 55 modules, twine check, the
   numpy-only venv's checks>. Downstream: T3Polynomial adopted the rename
   (e1078cb) and ran the benchmarks on c97c754f (90dbd17, dbda0de). dev/:
   <the archive moves>. The GitHub release text is
   release_notes_2026.5.0.md (untracked, as for 2026.3.0 and 2026.4.0).

   Co-Authored-By: <the model(s) that did the work; 2026.4.0 carried Claude Opus 5.5 and Claude Fable 5.1> <noreply@anthropic.com>
   Claude-Session: <the session line>
   ```
10. **S, push.** `git push`.
11. **S, CI.** `gh run list --limit 5` until every job on the commit is green: both test legs, examples, and
    build-sphinx-docs.
12. **N, tag.** An annotated `v2026.5.0` on the release commit, with the message "T3Toolbox 2026.5.0 — the store
    rename, phase timing and the Gauss-Newton-to-exact switch" (the form `git tag -n1` shows for 2026.4.0). Then push
    the tag. One option: tag `v2026.5.0-rc1` → TestPyPI first, as 2026.3.0 did, the last breaking release. 2026.4.0
    skipped the rc.
13. **N, PyPI gate.** `release.yaml` builds and runs `twine check`; `publish-pypi` waits for Nick's approval of the
    `pypi` environment.
14. **S, check the published package.** In a fresh venv against real PyPI: `pip install t3toolbox==2026.5.0`, the
    version, a fit, the jax-absent warning.
15. **N, GitHub Release.** Created from the tag, with `release_notes_2026.5.0.md` as the body. The session cannot
    create releases.
16. **S, after the release.** Refresh HANDOFF and CLAUDE.md if the release commit did not, then commit and push.

## 6. Inconsistencies found

a. **The CHANGELOG date line** uses the em dash, `## [2026.5.0] — 2026-11-15`, not ` - ` (§1).

b. **`release_notes_2026.5.0.md` is stale.** It is R6's cut (§3).

c. **`CLAUDE.md` "Current state" is stale** (line 410). It names 2026.3.0 as the latest release and never mentions
   2026.4.0 (tagged 2026-11-10). 2026.3.0's release commit updated it; 2026.4.0's did not. Proposed lead: "2026.5.0
   is RELEASED (<date>) — the store rename (breaking), opt-in phase timing and `hessian='switch'`; 2026.4.0
   (2026-11-10) the aligned padded frame; 2026.3.0 shipped 2026-10-29 …"

d. **The HANDOFF's first heading is stale.** It still reads "⚠ IN PROGRESS (2026-11-02): the padded frame / cone
   reference feature" (2026.4.0 work), and its top block lists two review stops as pending (§4: snapshot and rewrite).

e. **2026.4.0's arc records were never archived.** These are `dev/aligned_padded_frame_impl_plan_2026-11-09.md`,
   reviews r1–r6, `dev/aligned_padded_frame_slices/` (25 tracked files) and
   `dev/padded_frame_cone_judgment_calls_2026_11.md`. `tests/backend/test_padded_frame.py:6` points at the live plan
   path. If the plan is archived, that docstring line becomes ``The plan is
   ``dev/archive/aligned_padded_frame_impl_plan_2026-11-09_shipped.md``; …``. It is the only shipped file that points
   into a live `dev/` path.

f. **The timing entry's "is 2026.4.0's" is slightly off** (`CHANGELOG.md:118`). It says "every output, the display
   included, is 2026.4.0's (clock readings aside)". `### Fixed` changes a converged `'cone'` row's display line and
   eight messages, so outputs are not exactly 2026.4.0's. Proposed: "… is the untimed run's (clock readings aside)".

g. **`docs/release_notes.md` "Upgrading from 2026.4.0" covers only the rename.** 2026.4.0's upgrade section listed
   its opt-in additions. The timing plan's rule is to add a line only if the section lists additions, and it does
   not, so nothing is required. An optional closing paragraph, before line 85: "**Also new, and opt-in.**
   `newton_cg(timing=True)` splits each Newton row's wall time by phase (`fitting_and_optimization.md` §4.13), and
   `newton_cg(hessian='switch')` starts with Gauss-Newton and switches to the exact Hessian (§4.14). Without them,
   every number a run returns is 2026.4.0's."

h. **`docs/contributor/verification_records.md` has no 2026.5.0 section.** 2026.3.0 and 2026.4.0 each have one, and
   2026.4.0's release commit added its acceptance table there. The switch table lives only in §4.14 and the ledger.
   Optional: a short "## The GN-to-Newton switch (2026.5.0)" section with W3 §2.2–§2.4's totals.

i. **The README and `docs/index.rst` mention neither new feature.**
   - README after line 116 ("such a frame."), add: "`newton_cg(..., hessian='switch')` starts with Gauss-Newton and
     switches to the exact Hessian once the exact model predicts the objective clearly better, and `timing=True`
     splits each Newton row's wall time into phases."
   - `docs/index.rst:13–15`: "(Newton-CG with the Gauss-Newton or the exact Riemannian Hessian, or Gauss-Newton
     switching to the exact one within a run; residual weighting, regularization and rank continuation included,
     with an aligned start frame and a matching Newton reference after each rank increase, and optional per-phase
     timing)".

j. **`docs/padded_frame_cone.tex:51` has `\author{T3Toolbox 2026.4.0}`,** though R5 converted the note to 2026.5.0's
   letters. Recommend leaving it ("introduced in") until the note's deferred-row rebuild, which rebuilds the PDF
   anyway.

k. **No dangling notes.** The "reserved" `tracking` phase and the W3 placeholder are gone. No other "reserved",
   "placeholder" or "unreleased" remains in the optimizer modules or their docs. The only "unreleased" left is
   `CHANGELOG.md:6`.

## 7. As executed (2026-11-15, the implementer; nothing committed)

On HEAD `51782061` (the hand fixes committed). The CHANGELOG had not changed since `7807b238`; every line number in
§1 still held (the intro was lines 8–10; the timing wording line 118 became 137 once the intro went in). The §2
figures were re-checked against `W3_handoff.md` §2.1–§2.5, §4.3 and §5.1–§5.2: all match.

**Edited** (for the session to `git add` by name; the 147 `git mv` renames are already staged):
`pyproject.toml`, `CITATION.cff`, `t3toolbox/__init__.py`, `CHANGELOG.md`, `README.md`, `docs/index.rst`,
`docs/release_notes.md`, `docs/contributor/verification_records.md`, `tests/backend/test_padded_frame.py`,
`dev/HANDOFF.md`, the new `dev/archive/handoff_2026-11-15_pre-2026.5.0.md`, this file
(`dev/archive/release_2026.5.0_prep_2026-11-15.md`, moved with a plain `mv`: it was untracked), and the three
bannered plans at their new paths (`dev/archive/store_rename_plan_2026-11-10_shipped.md`,
`newton_timing_plan_2026-11-10_shipped.md`, `gn_newton_switch_plan_2026-11-13_shipped.md`). `CLAUDE.md` (§6 c) is the
session's and was not touched. `release_notes_2026.5.0.md` was re-cut and stays untracked.

**What was done.**
- §1: the four sites; the heading `## [2026.5.0] — 2026-11-15` (U+2014).
- §2: the 22-line intro as written, without the optional timing sentence; ≤ 115 columns.
- §6 f: "is the untimed run's (clock readings aside)"; the two lines rewrapped to stay ≤ 115 columns.
- §6 g: the "Also new, and opt-in" paragraph before "## Upgrading from 2026.3.0".
- §6 h: "## The GN-to-Newton switch (2026.5.0)" at the end of `verification_records.md`: the W3 §2.2–§2.3 totals
  (10 × 100, both starts, all five arms) and §2.4 (20 × 1000, switch and exact, both starts), and five bullets
  (`switched_at` 4–10 everywhere, tracking 0.44–3.3 % of net wall on the GPU and 0.24 % on the laptop, the GN cap
  caveat, the partial GN-TR ladder, the laptop pair, the cross-process GPU rounding and T1).
- §6 i: README (the `newton_cg` bullet) and `docs/index.rst:13–16`.
- §4: deleted the untracked `dev/store_rename_slices/R2_pass1.diff` (532 KB, regenerable), then every `git mv`.
  Store rename 84 files, timing 9, switch 11, 2026.4.0's records 33 (plan `_shipped` and `aligned_padded_frame_2026-11-10/`:
  reviews r1–r6, `aligned_padded_frame_slices/`, the judgment calls), the glossary pass 10
  (`glossary_docs_pass_2026-11-15/`: candidates A–C, reviews r1–r3, the pass record and its review, `hand_fixes_2026-11-15.md`
  and `hand_fixes_review_2026-11-15.md`). Total 147 renames, R100. The ignored `battery_ref/` and `__pycache__/` moved
  with their folders. No moved folder was `git add`ed. `glossary_naming_decisions.md` and `_status.md` stay live.
- Banners in the 2026.3.0 form (tag date 2026-11-15). The switch banner also names `verification_records.md`
  (h adopted).
- §6 e: `tests/backend/test_padded_frame.py:6` points at `dev/archive/aligned_padded_frame_impl_plan_2026-11-09_shipped.md`.
  It is wrapped onto two lines because the new path made the line 131 columns. After the moves, the grep of shipped
  files for the moved live paths is empty.
- HANDOFF: the snapshot is byte-identical to HEAD's. The rewrite follows the 2026.3.0 shape: the next arc, where we
  are, open items, the release gates, the machine rules, the backlog, and the standing questions and constraints.
  The "NEXT ARC AFTER THE 2026.5.0 TAG" block is verbatim (diffed against `923178ff`'s addition).

**Deviations and judgment calls.**
1. §6 g uses the page's own link form, [`fitting_and_optimization.md`](fitting_and_optimization.md) §4.13, which is
   the precedent at `release_notes.md:300`, rather than plain backticks. The wording is otherwise the prep's.
2. 2026.4.0's plan moved to `_shipped` **without** a banner, because §4 names banners for the three 2026.5.0 plans
   only. Add one if wanted ("SHIPPED in 2026.4.0 (tag `v2026.4.0`, 2026-11-10); archived at the 2026.5.0 release").
3. Not edited, out of scope, and recorded in the HANDOFF instead:
   - The live glossary files still point at pre-archive paths (`glossary_naming_decisions.md:5, 31`, and
     `_status.md:3`'s link to the pass record).
   - The status file still marks as deferred the 7 in-block rows that the hand fixes did (its lines 74, 746–748,
     750, 752 and 757). The real open count is 62 math-note rows.
4. The suite ran with `-p no:cacheprovider` added. The verification ran before the §3 re-cut, which edits only the
   untracked notes. A full disk stopped the work after steps 2–8 had passed, and the session's scratch was cleared.
   Step 8 was re-run on the unchanged tree, with the same results; the counts below for steps 2–7 are from the
   first pass.
5. In the wheel smoke the exact target has zero residual, so the switch does not fire there (`switched_at` None;
   Gauss-Newton is quadratic). A noisy target was added, and on it the switch fired. The jax-absent warning is
   issued once per call site per process (`_JAX_WARNED`), so a second `use_jit` call in one process is silent by
   design.

**Counts.**
- Step 2, the suite on the `t3toolbox` env (py3.11.15, numpy 2.4.6, jax 0.10.2), `-n 3`:
  - 1,407 passed, 5 skipped, 49,058 subtests, in 25:00.
  - 1 warning: Adam on MANIFOLD in the oracle sweep (expected).
- Step 3, the slow tier, serial: 66 passed, 13,173 subtests, in 17:04 (2 warnings, the same Adam ones).
- Step 4, the floor suite on `tttt` (py3.9.19, numpy 1.22.4, jax 0.4.30), single process: 1,406 passed, 6 skipped, in
  45:09 (4 Adam warnings).
- Step 5, doctests, the same on both envs:
  - 241 module doctests;
  - `getting_started.rst` 81/81;
  - the 48 pages, clean (rc 0);
  - `docs/chunking.md` by hand on `tttt`, 29/29.
- Step 6: sphinx 9.0.4, `-W --keep-going`, exit 0 with zero warnings, in 16:44. `index.html` and
  `release_notes.html` render 2026.5.0, and the heading reads `[2026.5.0] — 2026-11-15`.
- Step 7: the 14 examples all exit 0, in 7:30 (the slowest is `fit_shared_factors_jetted_probes.py`, 146 s).
- Step 8, the wheel:
  - sdist and wheel built; `twine check` PASSED on both;
  - the wheel holds exactly the 55 tracked `t3toolbox/*.py` modules plus dist-info;
  - no module was added since `v2026.4.0`.
- Step 8, a fresh py3.11.15 venv with only numpy 2.4.6 and the wheel:
  - `__version__` 2026.5.0 and no jax;
  - Gauss-Newton fit, apply, rank 2: rel. err 8.05e-11, 12 Newton steps; exact: 6.26e-10, 17 steps;
  - `use_jit=True` warns ("jax was requested but is not installed -- running on numpy instead") and runs eager;
  - `newton_cg(hessian='switch', timing=True)`: `stats['hessian']` is `'switch'`; `stats['timing']` has all ten
    keys and 14 of 14 phases timed, with `compile` None on numpy;
  - on a 1 %-noise target the switch fired: `switched_at` 12, then 12 Gauss-Newton rows and 2 exact rows, with
    tracking 0.047 s of 0.42 s.
  - The venv and the wheel directory were deleted afterwards.
- §3: `release_notes_2026.5.0.md` re-cut after every CHANGELOG edit:
  - 185 lines, first line "Three headlines. …", no `## ` heading;
  - Changed — breaking / Changed / Fixed / Added;
  - the same command rebuilds `release_notes_2026.4.0.md` byte for byte from `v2026.4.0`.

**The commit message's paragraph, filled in:**

```
Verification at this commit: 1,407 tests (5 skipped) / 49,058 subtests on
py3.11 / numpy 2.4.6 / jax 0.10.2 (-n 3) and the serial slow tier (66 /
13,173); 1,406 tests (6 skipped) on the floor env py3.9 / numpy 1.22.4 /
jax 0.4.30 (single process); 241 module doctests, getting_started (81/81)
and all 48 doc pages on both envs, docs/chunking.md by hand on the floor
(29/29); sphinx -W --keep-going clean (renders 2026.5.0); examples 14/14;
sdist and wheel built, `twine check` passes, the wheel holds the 55
package modules and nothing else, and a fresh numpy-only venv (py3.11,
numpy 2.4.6, no jax) installs it, reports 2026.5.0, fits (apply, rank 2:
Gauss-Newton 8e-11, exact 6e-10), warns-and-runs-eager on `use_jit`, and
runs newton_cg(hessian='switch', timing=True) eager (every phase timed; on
a noisy target the switch fired, switched_at 12). Downstream: T3Polynomial
adopted the rename (e1078cb) and ran the benchmarks on c97c754f (90dbd17,
dbda0de). dev/: the three plans archived as _shipped with SHIPPED banners,
their reviews and slice records in dev/archive/{store_rename,
newton_timing,gn_newton_switch}_2026-11-15/; 2026.4.0's plan _shipped and
its records in dev/archive/aligned_padded_frame_2026-11-10/ (the test
docstring re-pointed); the glossary-pass and hand-fix records in
dev/archive/glossary_docs_pass_2026-11-15/; the HANDOFF snapshotted to
dev/archive/handoff_2026-11-15_pre-2026.5.0.md and rewritten to the
post-release state; this prep archived with its as-executed record. The
GitHub release text is release_notes_2026.5.0.md (untracked, as for
2026.3.0 and 2026.4.0).
```
