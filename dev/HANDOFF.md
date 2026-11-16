# T3Toolbox — current handoff

_Updated 2026-11-15, at the 2026.5.0 release; "The next arc" updated the same evening, when the design work moved into
`dev/graded_messages/`._ The previous handoff (the store rename R1–R6, phase timing P1–P3, the
switch W1–W4, the glossary/docs pass, the hand fixes, and the 2026.4.0 arc's state blocks) is archived verbatim at
`dev/archive/handoff_2026-11-15_pre-2026.5.0.md`; older ones are `handoff_2026-10-29_pre-2026.3.0.md`,
`handoff_2026-08-21_optimization_layer.md`, `handoff_2026-08-20_pre-2026.1.0.md` and
`handoff_2026-07-12_1.0_complete.md`, all in `dev/archive/`. What each release contains is the CHANGELOG; why it is
the way it is, `docs/`. This file is where-we-are + what's next, nothing else.

## Released: 2026.5.0 (tag v2026.5.0 on 4463231f, 2026-11-15; PyPI published the same day)

The release commit is 52843bfc; 4463231f is a tests-only follow-up (CI's floor leg had been red since c1a42c24 with
four failures nobody watched: an `np.bool_` subtest label xdist cannot serialize on numpy 1.x, and three tests whose
trajectories depended on a TIE — see the backlog). The published package installs in a fresh venv and reports 2026.5.0.
The GitHub Release body is `release_notes_2026.5.0.md` (untracked). Records: `dev/ci_floor_fix_2026-11-15.md`.

**Backlog added at the release:**
- **Trust-region radius tie (library question).** Under Gauss-Newton with `step_control='trust_region'` the DEFAULT
  radius equals the length of CG's first step bitwise, so rounding (BLAS; eager vs jit) decides whether row 0 is a
  boundary stop and whether the radius doubles; the whole trajectory follows. A default that cannot tie (e.g. 2x the
  Cauchy step) would make row 0 platform-independent. A behaviour change: its own slice, Darcy benchmarks, CHANGELOG.
- **CI discipline.** Check both CI legs after EVERY push; the local `tttt` env (numpy 1.22, single-process) does not
  reproduce CI's floor leg (numpy 1.26, xdist). Pins for a CI-like local env are in `dev/ci_floor_fix_2026-11-15.md`.
  Structural tests drive their events deterministically; subtest params are builtins.
- **Disk hygiene.** Clear the session scratch after each slice (18 GB of review copies filled the 64 GB root on
  2026-11-15); review agents delete their repo copies when done.

## The next arc

**THE ARC IN PROGRESS (Nick, 2026-11-15): the "graded messages" refactor of the whole sampling layer.** Major, BREAKING,
intentionally so. Nick's ruling: design from FIRST PRINCIPLES; never preserve existing behaviour as a crutch; today's
behaviour is one candidate, never the default; decide the TARGET before any sequencing; go slow.

**Where it lives.** `dev/graded_messages/` (tracked since 2026-11-15, 18:04; it was the untracked handoff package
`dev/graded_messages_handoff/` before, and any such copy on a machine is superseded: Nick confirmed this for the
laptop copy on 2026-11-15). `main` holds the snapshot of that
moment. **From then on the work, that folder included, continues on the branch `graded-messages`** (Nick, 2026-11-15:
"future commits will be on a branch. The reason for using a branch is the fact that this refactor will be major and
severely breaking"); it is merged when it is done. Look on the branch for the live state. No library code has been
changed yet.

**Read, in this order:** its `README.md`; `TARGET_DECISIONS.md`, the working agenda (its sections "Where this stands"
and "What remains, in groups" are the state and carry the counts; the rows are being settled one at a time with
Nick, group A "jets and the misfit" first); `TYPE_CATALOG.md` (the types of every sampling variable, arrows on
the edges, a type checker for every contraction); then `DESIGN_AFTER_REVIEW.md` and `REVIEW_DECISIONS.md` (the log).
The older documents there carry status notes and are history.

**Settled so far, in brief** (the rows of `TARGET_DECISIONS.md` are authoritative): one public backend function
`sampling_variables`, with `SamplePlan`/`SampleScan` hooks and explicit per-variable functions; jets are Taylor
coefficients everywhere, with derivative mode as one adapter at the outer level of fitting; order 0 is the same functions
with the order axis absent (an optional axis `T`); the plain and the jetted sampling function are ONE function,
`probe(ww, pp=None, order=0)`, and the fitting layer has three kinds with an `order` in place of six; data and residuals
are of the type of the output, in forward order, and the tilde variables are stored turned (reversed) inside; the misfit's
default weight is ones, with the weight shaped `(d,) + T`; preparation of data and weights is done by standalone
functions with opt-in keywords. All of this is USER-VISIBLE and overrides the first proposal's "no user-visible
convention has to change".

**Planned after the target is settled** (from the handoff written at the release; confirmed by Nick on 2026-11-15, 18:32): an agent
re-running the folder's checks against the tag and scouts reading the library's own records -> a writer/critic in-depth
plan (two stages; token map + gates reusing the rename machinery; exact-equality test cleanup as a pre-slice; a
performance gate on the Darcy ladders before any hand kernel is deleted) -> sample-edit batches -> implementer/reviewer
slices. Open with Nick: stage 1 (plan wiring, no kernel change) as its own release? The folder's check suites
(`scripts/run_all.sh`, `scripts/review/run_review_checks.sh`) give their recorded output against 2026.5.0.

## Where we are

**2026.5.0 — tag pending; the session flips this line after the tag** (to "2026.5.0 SHIPPED to PyPI (2026-11-15)").
2026.4.0 was tagged 2026-11-10 (`v2026.4.0` on `e364fe01`), 2026.3.0 shipped 2026-10-29, 2026.2.0 2026-08-25, 2026.1.0
2026-08-20, 2026.0.0 2026-07-13. The release commit bumps the four version sites (`pyproject.toml`, `CITATION.cff`,
the `__init__` fallback, the CHANGELOG heading `[2026.5.0] — 2026-11-15`), writes the CHANGELOG intro (three
headlines and the measured paragraph), adds the switch section to `docs/contributor/verification_records.md` and the
"Also new, and opt-in" paragraph to `docs/release_notes.md`, and archives the arcs' records (below). The prep and its
as-executed record, with every verification count, is `dev/archive/release_2026.5.0_prep_2026-11-15.md` (§7).
What remains: push and CI green (both test legs, examples, the docs build) → the `v2026.5.0` tag (Nick) → approve
the `pypi` environment → a clean-venv check against real PyPI (version, a fit, the jax-absent warning) → the GitHub
Release (Nick) with `release_notes_2026.5.0.md` as the body (untracked in the root, as for 2026.3.0 and 2026.4.0;
cut from the CHANGELOG by the archived prep's §3 command).

Verification at the release commit (2026-11-15): 1,407 tests (5 skipped) / 49,058 subtests (`-n 3`) and the serial
slow tier (66 / 13,173) on the `t3toolbox` env; 1,406 tests (6 skipped) on the floor env, single process; 241 module
doctests, `getting_started.rst` (81/81) and all 48 doc pages on both envs, `docs/chunking.md` by hand on the floor
(29/29); `sphinx -W --keep-going` clean, rendering 2026.5.0; the 14 examples; the wheel smoke (`twine check` PASSED;
the 55 modules and nothing else; a fresh numpy-only py3.11 venv: 2026.5.0, Gauss-Newton and exact fits, the
jax-absent warning, `hessian='switch', timing=True` eager, the switch firing on a noisy target).

**What 2026.5.0 shipped** (2026-11-10 → 15; each arc a writer/critic plan, then implementer/reviewer slices):
- **The store rename** (BREAKING; R1–R6, plan draft 29). The frame stores `up` and `down` swap names; letters D / U /
  L / R, V / H, B / G; every number bit-identical. Durable record: the CHANGELOG "Changed — breaking",
  `docs/release_notes.md` "Upgrading from 2026.4.0", `docs/frame_variations.md` § Four rank stores,
  `docs/glossary.md`. Build record: `dev/archive/store_rename_plan_2026-11-10_shipped.md` and
  `dev/archive/store_rename_2026-11-15/` (reviews r1–r26, the inventories and sample edits, the tools, the slice
  records `store_rename_slices/R1–R6` with the diagram inventory). The tools (`store_rename_apply.py`, `_check.py`,
  `_diagram_check.py` and the slice folder's scanners) hard-code `dev/store_rename_slices` and do not run from the
  archive; the next arc's "gates reusing the rename machinery" must re-point or copy them. The ignored battery
  reference (`store_rename_slices/battery_ref/`) moved with the folder.
- **Phase timing** (`newton_cg(timing=True)`; P1–P3). Durable: `docs/fitting_and_optimization.md` §4.13,
  `docs/contributor/scan_body_principles.md` (the clock slot). Build:
  `dev/archive/newton_timing_plan_2026-11-10_shipped.md` and `dev/archive/newton_timing_2026-11-15/`.
- **The Gauss-Newton-to-exact switch** (`newton_cg(hessian='switch')`; W1–W4, W3 the Darcy benchmark run downstream:
  T3Polynomial `90dbd17`, `dbda0de`). Durable: §4.14, `docs/contributor/curvature_internals.md` (the Gauss-Newton
  view), `docs/contributor/verification_records.md` (the benchmark totals), the ledger
  `docs/contributor/deferred_and_rejected.md`. Build: `dev/archive/gn_newton_switch_plan_2026-11-13_shipped.md` and
  `dev/archive/gn_newton_switch_2026-11-15/` (W3's full tables are `gn_newton_switch_slices/W3_handoff.md`).
- **The glossary and docs-consistency pass** (`8ad6dea3`: eight new glossary entries, 505 of 599 work-list rows) and
  the 29 hand fixes (`51782061`). Records: `dev/archive/glossary_docs_pass_2026-11-15/` (the scout candidates, the
  glossary reviews r1–r3, the pass record and its review, the hand-fix record and its review). Live:
  `dev/glossary_naming_decisions.md` (the decisions and the work list) and `dev/glossary_naming_decisions_status.md`
  (per-row status). Their pointers to the pass record and the rename plan name the pre-archive paths.
- **2026.4.0's build record**, archived at this release:
  `dev/archive/aligned_padded_frame_impl_plan_2026-11-09_shipped.md` and `dev/archive/aligned_padded_frame_2026-11-10/`
  (plan reviews r1–r6, slices N1–N10, the judgment calls).

Downstream: T3Polynomial adopted the rename (`e1078cb`) and ran the switch benchmarks on `c97c754f` (`90dbd17`,
`dbda0de`).

## Open items from the 2026.5.0 arcs

- **The deferred docs rows.** `dev/glossary_naming_decisions_status.md` tallies 599 rows: 505 done, 25 done earlier,
  69 deferred. Of the 69, the 7 inside hand-rewritten blocks were fixed by the hand fixes (`51782061`), but the file
  still marks them deferred (its lines 74, 746–748, 750, 752 and 757); re-mark them when next touching it. The other
  62 are math-note rows, each waiting for its note's rebuild (the `.tex` and its PDF). `docs/padded_frame_cone.tex:51`
  still has `\author{T3Toolbox 2026.4.0}`; leave it to that note's rebuild.
- **Audit items** (from the 2026.4.0 arc and the rename): canonical bases / sign canonicalization at the
  representation level (D-42/D-49: representations of one point in different bases, e.g. at singular-value ties, give
  different hint-free starts); the code's own `H` for companion centers (sharing / alignment) against the
  frame-variations `H`; the shared note's `\dot V_i`, `\dot W_i`, D-shaped by the shape rule but not store letters; a
  `functools.partial`-free uniform loop (D-47; the binding was accepted in that one instance and sets no precedent);
  the `ranks`-depends-on-sharing smell.
- **A jitted tracking product: a future option only** (G-4, Nick 2026-11-13: "no need for more elaborate options at
  this time"). On the A100 the eager product is dispatch-bound (about 80 ms at every rank, 12–32 Gauss-Newton applies;
  about 2 on a CPU), yet at most 3.3 % of a switch run's net wall. It is the candidate if a problem tracks much longer.
  The switch's other deferred items (switching back, a delayed tracking start, a product-free trigger, a secant `S`,
  carrying the state across calls, public helpers, the lift diagnostics on Gauss-Newton-phase rows) are in the ledger.
- **I-36 is open at the aligned start.** From the library start the switch beat both pure modes (223 Newton steps
  against exact's 267 and Gauss-Newton's 1,625; 177 s net against 275 and 427). From the aligned start
  (`padded_from`) exact took 231 against the switch's 303, most of it at level 4. At 20 × 1000 the two were even at
  both starts. Which mode a level should start in stays open. W3 §5.1's hypothesis (the forced Gauss-Newton rows move
  the iterate and the carried-over trust radius off the exact path) is untested, and one ladder does not show whether
  level 4 is systematic. The 20 × 1000 Gauss-Newton trust-region and `tracking` arms never finished (GN-TR at the
  library start reached level 4), so the "promising" bar at that scale is open too.
- 2026.4.0's leftover: whether to rerun the laptop Newton comparison on the shipped aligned start (the pre-release
  g06 measurements stand as the reference unless Nick asks).

## Release gates (run all before a tag)

As run for 2026.5.0 (the archived prep's §5 has the exact commands):
- **Full suite:** `PYTHONPATH=$PWD <env-python> -m pytest tests/ -q -n 3` (per-test seeded, order-independent).
- **Slow tier — REQUIRED, run SERIALLY:** `T3TOOLBOX_SLOW_TESTS=1 PYTHONPATH=$PWD <env-python> -m pytest
  tests/test_oracle_sweep.py tests/backend/test_uniform_curvature.py -q -p no:cacheprovider`.
- **Floor suite**, single process, on the floor env.
- **Doctests and docs, on both envs:** `pytest --doctest-modules t3toolbox`; `python -m doctest
  docs/getting_started.rst`; every `docs/*.md` and `docs/contributor/*.md` page except `doctest_style.md` (48 pages);
  `docs/chunking.md` by hand on the floor env; `python -m sphinx -W --keep-going -b html docs <out>` (about 18 min;
  zero warnings; check the rendered version).
- **Examples:** all 14 (`examples/*.py`; they need optax and scipy).
- **Two local environments** (conda; run each env's python directly): `t3toolbox`, py3.11 / numpy 2.4.6 / jax 0.10.2
  (the CI current-leg twin), and `tttt`, py3.9 / numpy 1.22.4 / jax 0.4.30 (the floor).
- **Wheel smoke** (`dev/archive/release_plan_2026-07-13.md` REL-2): build, `twine check`, the wheel's file list (the
  55 package modules and nothing else), a fresh numpy-only py3.11 venv (the version, a fit, the jax-absent warning,
  the release's new paths). `t3toolbox.__version__` reads the installed metadata first, so an old editable install
  reports a stale version: read the version from the fresh venv. A release is: bump the four sites, stamp the
  CHANGELOG, re-cut `release_notes_<version>.md` (untracked), smoke, gates, commit, CI green → tag → approve `pypi` →
  clean-venv check → GitHub Release.

## Machine rules

- **The 13 GB laptop:** pytest-xdist at `-n 3` at most (`-n 8` swapped; jax workers reach 3–4 GB each), `-n 2` if
  anything else runs. Never two heavy jobs at once (2026-11-13: a `-n 3` suite beside a sphinx build hit 0 GB
  available): the suite, the docs build, the floor-env suite, the examples and any battery run one after another. This
  binds the session and every agent.
- **Commits:** stage by name, never `-a`. Never stage `release_notes_*.md`, `*.npz`,
  `dev/check_t3svd_retraction_second_order.py`, or a leftover `dev/graded_messages_handoff/` (superseded by the tracked
  `dev/graded_messages/`; commits to that folder go on the branch `graded-messages`). End each message with the `Co-Authored-By:` line of the model that did the work and the
  `Claude-Session:` line. Check `gh run list` after each push.

## Next up — the backlog (open since 2026.3.0)

**From the exact-Hessian arc** (each also in the ledger `docs/contributor/deferred_and_rejected.md`):
- **float32 end to end.** A float32 fit comes back float64: 38 default-dtype constructors in eight backend modules sit
  upstream of the jet kernels (which preserve float32 since 2026.3.0). A ready follow-up patch is
  `dev/archive/curvature_impl_review_2026-10-29/slices/float32_end_to_end_followup.patch` (checked 2026-10-29 over the
  six kinds × both layouts × manifold / corewise / shared × both Hessians; float64 bit for bit). Its own slice: each scan
  init must take the dtype of ALL its scan's operands, which wants a mixed-dtype jax test per site.
- **Legacy RNG** (backlog from the cancelled R): move the random constructors off the global `np.random` stream to
  explicit `Generator` / jax-key arguments (the E2 and O2 flakes were this).
- **I-36 — the Newton-schedule A/B** (Gauss-Newton vs exact as the continuation default; Nick leans exact with the warm
  start). Empirical; the per-call override hooks exist (and T3Polynomial's per-level `newton_kwargs` schedule).
  Inputs: `dev/archive/curvature_impl_review_2026-10-29/shared_exp/shared_vs_unshared_exact.md` and
  `.../nonlinearity_vs_tied_factors.md`, and since 2026.5.0 the switch benchmark's GPU ladders (all numbers in
  `verification_records.md`). Still open at the aligned start: see "Open items from the 2026.5.0 arcs" above.
- **The S9 tied-projection clip.** The uniform tied *projection* (`ufv_share_tucker_variations`, Gauss-Newton code)
  clips at the padded dims and does not mask padded `s_g` levels (17 % off ragged at a near-singular point the
  equivalence contract exempts; not robust to garbage in a hand-built companion). Needs its own bit-identity run.
- **The leftover-chunk overshoot.** On current XLA a remainder chunk is allocated on top of the full one, so
  `max_chunk_size_within` can overshoot its budget ~2×; `_wchunked_sum` doubles the peak at a near-full remainder.
- **The S7b follow-ups:** carry the operand scale through `normalized` / `sum_tangents` / `unstack_tangents`; a
  `UT3Tangent.__truediv__`; a faster `corewise_stack_norm` (safe-mode `±` is 4–6× slower since S7b; the optimizers run
  on `.data` and are unaffected).
- Smaller: `fv_share_tucker_variations`'s mixed-backend inference; optional `_blm.hvp` routing for the frontend exact
  `hessian(p)`; the `docs/conf.py` xref variant for the dead `:py:func:` links; the six
  `u_*_corewise_transpose_vs_ragged` oracle-harness checks (excluded since Phase D, repairable); `docs/pad_safe_svd.tex`
  still cites the pre-archive `dev/review_2026-08-22/` path (fix + PDF rebuild). Downstream, not ours: T3Polynomial's
  `x01` script has a pre-existing `NameError` (`K`), noted at S14.

**Carried forward:**
- The Goal-1 **`fit(...)` facade** — auto geometry / optimizer / ranks / `x0` + rank continuation and validation
  ("standard user, no fiddling"); the largest single item, deferred. The Newton-schedule hooks are its seam.
- The **Grasedyck–Kramer `SingularValueRegularizer`** and weighted `+` / `−` / scale / `⊙`; **base-point-as-tangent**
  as a public op (promote the regularizer's private `_manifold_point_tangent`); **sharded fitting**
  (`dev/sharded_fitting_plan.md`); **data-shaped residual weighting** (deferred 2026-09-03 — pick up from
  `dev/data_shaped_residual_weight_reviews.md`, not the plan's v1).
- The 2026.2 recon backlog, still open at last look (re-verify before picking one up; the archived handoff has the
  detail): frontend access to `edge_condition_numbers` / `t3_tie_tucker_factors`; uniform `share`; `UT3Weights` /
  `UT3FrameWeights` `reverse` / `stack`; `make_newton_display`'s backend uniform validation packing; the frontend-only
  `chunk_size='auto'` derivation; the `shared_geometry.py` signature pass; mechanical signature items (17 `regularizer:
  typ.Any`, the `(order+1)+W` tuple comma, `T3Weights`' `len=d+1`); `t3_stack`'s docstring; `core_shapes` vs
  `get_core_shapes`; the R2-12 / R2-13 review leftovers; a default-path doctest pass; trimming `test_dispatch` jit time.
- The **toolbox reference paper** (independent of releases): `dev/paper_scope.md`. The standalone grouped-contractions
  library: PARKED. Zenodo DOI (Nick, later); the dead `gh-pages` branch (optional). Sweep candidates: the DONE plans
  `dev/newton_display_plan.md` and `dev/per_mode_weighting_plan.md`.

## Standing open questions

- **`dev/OPEN_QUESTION_extension_surface.md`** — what extension the fitting layer is meant to support (opened
  2026-08-21; 2026.3.0 added the sixth `SamplingKind` operation and the four exact-mode geometry members to its
  surface). Standing, not a thread: it stays in `dev/` until answered.

## Standing constraints

Project-wide conventions live in **CLAUDE.md**; contributor conventions and decision records in the rendered
**Contributor guide** (`docs/contributor/`). Operational ones worth repeating: the docs build stays at **zero
warnings** (`sphinx -W` in CI); doctest outputs are **run-and-pasted, never hand-written**, and CI runs every
`docs/*.md` / `docs/contributor/*.md` page; a markdown doctest needs a **blank line before the closing fence**; a doctest
must not pin a stop reason or step count at the resolution floor (BLAS-dependent — assert a fact with a wide margin);
subTest params must be primitives under xdist.
