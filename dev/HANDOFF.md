# T3Toolbox — current handoff

_Updated 2026-11-15, at the 2026.5.0 release; "The next arc" updated the same evening, when the design work moved into
`dev/graded_messages/`, at the pause of 2026-11-16, 04:46, and at the pause of 2026-11-17, 15:14, after slice P4 of the implementation, at the pause of 2026-11-17, 16:30, with the plan for the generalized jets complete, and on 2026-11-19 for the move of the work to Claude Code on Nick's laptop (on the branch `graded-messages`)._ The previous handoff (the store rename R1–R6, phase timing P1–P3, the
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
severely breaking"); it is merged when it is done. Look on the branch for the live state. Library code has been
changed on the branch since 2026-11-16 (slices P0 to P4 of Part 1); `main` has none of it.

**START HERE (updated 2026-11-19, 19:30; Claude Code on Nick's laptop, branch `graded-messages`).** Nothing lives
in a session: everything is in files.

- **Where it stands (2026-11-21): the sub-arc P12 to P15 is COMMITTED and pushed (P15 = `70489fe6`); the work is at
  the THIRD PLANNED PAUSE, before the switch (P16).** What the sub-arc did: P12 the backend sampling functions on the
  engine (+ `*_corewise_tangent`), P13 the frontend methods merged and the `*_derivatives*` methods removed, P14 the
  least-squares oracle and batch (additive; the engine gained `keep=` and the public `frame_form`), P15 the backend
  geometry's `frame(x)` returning `(frame, centers)`. Per-slice record and every ruling: `LOG.md`. **At the pause, decided:** `keep=` is
  replaced by a pure `computed_names` (TARGET_DECISIONS, "The hold list"; slice P14b before P16). **Under discussion:**
  `frame_form`. **Raised for the
  pause:** `examples/fit_hilbert_from_apply_derivatives.py` about 11 s instead of 5.8 s (the engine at the jet in place
  of the old derivative kernels; Part 2). **P16 IS IN PROGRESS ON THE LOCAL BRANCH `p16-wip`** (created from
  `graded-messages` after P14b, `959eb70b`; never pushed; the parts are committed there as they are signed off, then
  squashed into one commit on `graded-messages`; to resume: `git checkout p16-wip`, read `LOG.md` there). P14b is
  committed. P16, the switch, design-first, by
  a FRESH pair of agents (the maintainer, 2026-11-21; `BRIEF.md`), committed in parts on a local side branch `p16-wip`
  and squashed into one commit on `graded-messages`. The pause's decisions: `TARGET_DECISIONS.md`, notes of 2026-11-21. Notes carried to P16 are in `P14_HANDOVER.md` and `P15_HANDOVER.md`
  (`eps|rho|` in the resolution; a test that `held` is not carried across MC-SGD draws; `pack_sample`/`pack_data` new
  forms; at a padded-start frame the oracle runs without `Vc`). **To resume after a lost session:** read `LOG.md`'s
  last entries, then the slice in hand's `P<n>_DESIGN_NOTE.md` / `_DESIGN_REVIEW.md` / `_HANDOVER.md` /
  `_REVIEW_r*.md`; the agents `implementer` and `reviewer` (Opus 5.5, high) may be resumable by name; if not, start
  fresh ones on `BRIEF.md`, `LOG.md` and those files. Uncommitted work in the tree belongs to the slice in hand.
- **The design-first rhythm** (from P5): before each slice's code, the implementer writes a design note and the
  reviewer critiques it; Claude settles it into the plan; then build, review rounds, commit. Design notes and reviews
  are beside the slices in `dev/graded_messages/implementation/`.
- **The gate runs at `-n 2`** since P9b (memory); the suite is about 1,800 tests and 31 minutes.
- **What P4a to P4d built:** `jet_contract` over one jet group `J` of any number of axes (`GroupLengths(J, W, K,
  C)`, 29 strings); `affine_jet` and `index_to_one_hot`; the dense functions on input jets (`dense_probe`,
  `dense_apply`, `dense_entries(T, index)`); the reference module `sampling_definitions` in several jet variables;
  `jets` and `data_preparation` in the jet shape, with their frontend reach. Nothing of P1 to P4 takes `pp` or an
  integer order any more. The per-slice record is `dev/graded_messages/implementation/LOG.md`.
- **The agents:** two Opus 5.5 agents on high effort, an implementer and a reviewer, kept from slice to slice
  (Nick's choice); Claude relays, rules on findings, looks each slice over and commits. Each slice: an independent
  oracle by the reviewer, mutation passes by both, the full gate at `-n 3`. Rounds: P4a 2, P4b 2, P4c 1, P4d 1.
- **CI:** the draft PR #26 runs both legs on every push (never merge it before Parts 1 and 2 are done). It found
  two floor bugs from P1/P2 that had never been run in CI (fixed: jax 0.4.30's `jnp.einsum` default
  `optimize='optimal'` is exponential in the operand count; numpy `subTest` labels that xdist cannot send on numpy
  1.x). Green on the floor since `3ccff5da`.
- **The laptop baseline** is `dev/graded_messages/implementation/BASELINE_laptop.md`; the suite at `-n 3` peaks near
  10 GB of the 13, so nothing may run beside it.
- **Read, in this order:** (1) `CLAUDE.md`; (2) this section down to the end of the block "STATE AT THE PAUSE OF
  2026-11-17" (the blocks after it are older states, kept as history); (3)
  `dev/graded_messages/implementation/ORCHESTRATION.md`: how a slice is run, what was particular to the cloud
  session, section 0 (the process since the laptop), and the notes carried to later slices;
  (4) `dev/graded_messages/PART1_PLAN.md`: the head, section 2.4 (`jet_contract`), section 5 (the slices; P5 is
  next), section 11.1 (Nick's answers) and 11.4 (the names), section 12 (the process); (5)
  `dev/graded_messages/implementation/BRIEF.md` (the agents' brief) and `LOG.md` (the record, slice by slice);
  (6) as reference: `dev/graded_messages/TARGET_DECISIONS.md` (the decisions; amended for the jets on 2026-11-19,
  see the banner at its top; its notes of 2026-11-17 and 2026-11-19 are the newest decisions) and
  `SAMPLE_VARIABLES.md` (every variable's name, definition and shape);
  `dev/graded_messages/REVIEW_DECISIONS.md` is the log of decisions in order (entries 116 to 122 are the jets).
- **Nick's standing rules for this arc** (all in force; the first block of "THE ARC IN PROGRESS" above, and
  `TARGET_DECISIONS.md`): reason from first principles; compatibility with today's behaviour is never a reason;
  keep it simple and avoid state; code statically readable, with literal contraction strings; the backend gives
  checkers and no guarantees, the frontend enforces; each of the two users (frontend, backend) can do everything
  in their own layer; a new group or abstraction of weight is raised with him before it enters a plan; no reliance
  on small CPU timings; tolerance tests for numerical things and exact tests for structural ones; read the
  library's own records before proposing; go slowly. No release until Part 1 and Part 2 are both done.

**The reading order of 2026-11-15, for the design documents** (history; the START HERE block above is the order for
the work now): its `README.md`; `TARGET_DECISIONS.md`, the working agenda (its sections "Where this stands"
and "What remains, in groups" are the state and carry the counts; the rows are being settled one at a time with
Nick); `SAMPLE_VARIABLES.md`, THE reference for the names and definitions of the sampling variables (the notation Nick
locked in on 2026-11-15, 23:23: a letter for the side contracted, `mu`/`nu`/`xi`/`eta`, a basis written only when it is
not the usual one, then `_tilde`, `_dot`; `NOTATION_HISTORY.md` when an older text is read); `TYPE_CATALOG.md` (the
types of every sampling variable, arrows on the edges, a type checker for every contraction); then
`DESIGN_AFTER_REVIEW.md` and `REVIEW_DECISIONS.md` (the log). The older documents there carry status notes and are
history.

**STATE AT THE PAUSE OF 2026-11-17, 15:14 (after slice P4; the current state, and it supersedes the blocks below
where they differ).** All of this is on the branch `graded-messages`; `main` is untouched since the release.

- **The target was settled and split into two parts** (2026-11-16): Part 1 is what is computed (names, conventions,
  interfaces); Part 2 is performance (the plan object, chunking, contraction order). No release until both are done.
- **Part 1 has an approved plan**, `dev/graded_messages/PART1_PLAN.md` (Nick, 2026-11-16, 21:16), in slices P0 to P20.
  How it is carried out is its section 12 and `dev/graded_messages/implementation/BRIEF.md`: per slice an implementer
  and a reviewer (Fable 5.2, medium effort) go back and forth until the reviewer signs off; the orchestrating session
  looks it over and commits (authored by NickAlger, Claude as co-author); major issues go to Nick. The per-slice record
  is `implementation/LOG.md` with each slice's handover, reviews and responses beside it.
- **Built, reviewed, committed and pushed: P0 to P4.** P0 the baseline and the gate script
  (`implementation/run_gate.sh`; steps run ONE AT A TIME, never sphinx beside tests); P1 `jet_contract` and
  `GroupLengths` in `backend/contractions.py`; P2 the reference module `backend/sampling_definitions.py` and the dense
  functions; P3 `backend/jets.py`, `backend/data_preparation.py`, the checkers and the frontend's re-exports in
  `fitting.py`; P4 the frame checkers (`fv_frame_consistency_residual`, `fv_centers_consistency_residual`, their
  uniform twins, the cached `consistency_residual` on both frame classes). The suite is at 1,632 passed, 5 skipped,
  1 xfailed (the xfail is the contraction-order marker carried to Part 2). Nothing of the old sampling layer is
  rerouted or deleted yet: P5 to P11 build the engine, P12 to P17 switch the layers over, P20 deletes.
- **P5 is NOT started, on purpose.** Nick asked for a planned pause after P4, and decided at it to **generalize the
  jets first** (2026-11-17, log entries 116 to 120 of `dev/graded_messages/REVIEW_DECISIONS.md`; the notes "One jet
  group, of any number of axes" and "The public interface is generalized from the start" in `TARGET_DECISIONS.md`):
  one jet group, letter `J`, of any number of axes (jets in several variables), box truncation, an explicit output
  shape; `T` and `A` leave the contraction strings; `order` becomes a jet shape (a tuple of maximum orders); the
  inputs are jets with sizes of their own (a line is two slots, a curve more) and `pp` leaves every signature, a
  helper building the input of a line; everything of the type of the output has the jet axes; the entries kind is of
  order 0 only (a helper to one-hot vectors, and apply for anything with a derivative; `entries_derivatives` goes).
- **The plan for that generalization is COMPLETE and awaits Nick's review of the whole (pause of 2026-11-17, 16:30).**
  An Opus writer and an Opus critic produced it in three rounds (`dev/graded_messages/reviews/jet_group_plan/`: the
  brief, `ACCOUNT.md` with the design on one page and every choice, `REVIEW_r1..r3.md`, `RESPONSE_r1..r2.md`); the
  critic agreed. It is in `dev/graded_messages/PART1_PLAN.md` (commit 48a1af4): four new slices between P4 and P5,
  **P4a** (`jet_contract(subscripts, *operands, lens, order, over_orders)` over the one group `J`,
  `GroupLengths(J, W, K, C)`, the 41 strings become 29), **P4b** (`affine_jet(ww, *directions)`,
  `index_to_one_hot(index, shape)`, the dense functions without `pp`), **P4c** (the reference module in several
  variables; entries without `order`), **P4d** (`jets` and `data_preparation` in the jet shape), and every later
  slice brought in line. Nick answered all its open points (O-36 to O-47; plan section 11.1, log entries 121 and
  122) and approved its names (section 11.4): `order` is a tuple only; no default of `order` on `LeastSquaresOracle`
  (required for every kind, `()` for entries), `()` the default on the sampling functions; every slot along a jet
  axis of the uniform layout is data (no mask); the ambient transposes have no jets and none are planned; the
  display's table has one column per multi-index and can be turned off (`error_table=True`, taken as approved with
  the other names and TO BE CONFIRMED at his review, as is its third place, `format_newton_iter`).
- **NEXT, and only when Nick says so** (he paused the work for a couple of days on 2026-11-17 at 16:26, to review
  the updated plan and because of his token budget): (1) his sign-off of the plan as a whole, with any changes;
  (2) Claude amends the decision records the account's section 7 lists (`TARGET_DECISIONS.md` rows 2.1, 2.3 to 2.9,
  5.4, 6.2, 6.6 and the notes named there; `SAMPLE_VARIABLES.md` sections 4, 6 and 7) -- NOT yet done: the notes
  of 2026-11-17 in `TARGET_DECISIONS.md` hold the decisions meanwhile, and the latest note wins; (3) P4a to P4d by
  the usual pair (Fable 5.2, medium), one slice per commit (the scheduled check-ins of the cloud session are not
  needed where the session does not stop; none is scheduled); (4) P5.
- **From P4, answered by Nick (2026-11-17, 15:18):** the normalization and reduction of the centers' residual, ok;
  `UT3Frame.is_consistent` thresholding against `max(1, ||T||)`, ok; `(x - y).norm()` losing accuracy across gauges
  is noted for consideration AFTER this refactor (in the backlog below).
- **Carried to Part 2** (the section of that name in `TARGET_DECISIONS.md`): the contraction order inside one step of
  `jet_contract` on numpy; map-type operations on uniform one site at a time; the benchmark against 2026.5.0.

**State at the pause of 2026-11-16, 04:46** (Nick: "let's pause. We will pick up tomorrow"): 41 of the 56 rows decided,
11 remain. Groups A (jets and the misfit), B (the frame and the value) and C (what a kind is, and the public surface)
are finished. HELD by Nick for one sitting on performance hooks and defaults: 5.3, 5.5 (the contraction order is in the
plan; granularity, the field's form and the helpers open) and all of group E (7.2, 7.3, 7.5, 7.6, 7.7). OWED: a
prototype for 5.4 (terms generic over several sample axes, `K`, `C` and the order group). NEXT, one question per
message: 8.3 (the optimizers' contract with a model), 8.4 (what the display reads), 9.3 (the reference when old
behaviour is not it). A Claude doc mirrors the folder, tab by tab, and is in step at this pause.

**Settled so far, in brief** (the rows of `TARGET_DECISIONS.md` are authoritative): one public backend function
`sampling_variables`, with `SamplePlan`/`SampleScan` hooks and explicit per-variable functions; jets are Taylor
coefficients everywhere, with derivative mode as one adapter at the outer level of fitting; order 0 is the same functions
with the order axis absent (an optional axis `T`); the plain and the jetted sampling function are ONE function,
`probe(ww, pp=None, order=0)`, and the fitting layer has three kinds with an `order` in place of six; data and residuals
are of the type of the output, in forward order, and the tilde variables are stored turned (reversed) inside; the misfit's
default weight is ones, with the weight shaped `(d,) + T`; preparation of data and weights is done by standalone
functions with opt-in keywords. Added in the session of 2026-11-15/16: the weight is an ARGUMENT of the engine (with
the data `y`) and acts on `f`, `g`, `JtJp`, `Kp` and `Jp_norm_sq`, so that they are derivatives of each other, and a
bare transpose takes a seed and no weight; transposes SUM over the samples by default (the reverse of today's
`sum_over_probes=False`); a frame is the four stores `(D, U, L, R)` with OPTIONAL centers `(Vc, Hc)` beside it, the value
read off the frame when they are given; frame consistency is a safe-mode precondition checked only where needed and
never through a dense tensor; frames support arbitrary tails (in the functions this arc touches); three trial-point
policies as an optimizer option; a kind is CLOSED in the sampling layer (three strings plus an order) and the open
extension point is an `Oracle` in the fitting layer (a frozen record: at a frame, return the wanted ones of `f`, `g`,
`JtJp`, `Jp_norm_sq`, and optionally `Kp`, `Jp`, `r`, as blocks before projection); `LocalModel` is the geometry, the
oracle and the regularizer at one frame and still the holder of local information (`residual`, `jacobian(p)` stay); the
projection, the lift and the exact-Hessian assembly stay in the geometry; one set of step functions for the ragged and
the uniform layout; the dtype rule is absorbing (jax over numpy, lower precision over higher). All of this is USER-VISIBLE and overrides the first proposal's "no user-visible
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

**From the graded-messages arc, for after it** (Nick, 2026-11-17, 15:18: "Note for consideration later, after this
refactor"):
- **An accurate norm of a difference.** `(x - y).norm()` loses accuracy when `x` and `y` are in different gauges: the
  error grows like `eps ||T||^2` (measured in slice P4: a consistent frame of norm `3.7e6` read `3.9e-10` through the
  subtraction and `4e-15` through the new route). The frame checkers avoid it by bringing each term to left-orthogonal
  form before the difference is taken (`_tt_difference_norm` and its helpers in `backend/fv_operations.py`, private).
  To consider: a public function for the norm of a difference of two T3s by that route, and whether `-` followed by
  `norm` should use it. Record: `dev/graded_messages/implementation/P4_HANDOVER.md` section 4.1 and `P4_REVIEW_r1.md`.

- **For future consideration: the low-rank structure of per-sample blocks** (Nick, 2026-11-20: "worth filing away").
  With `sum_over_samples=False` the engine materializes each sample's `g`/`JtJp`/`Kp` block dense, `W x |core|` in
  memory. But each per-sample block is low rank: without jets, a sample's `V_tilde` is a rank-1 outer product
  (`eta_tilde_s (x) w_s`) and its `H_tilde` a sum of three rank-1 three-way products (`mu_tilde (x) xi (x) nu` and its two
  siblings); with jets, short sums of such terms over slot pairs. The information is about `W x (sum of edge sizes)`,
  and the per-sample edge variables the engine already returns hold it in factored form. It is the same structure the
  ambient transposes exploit when their sum folds `W` into the CP rank. Many per-sample statistics need no dense
  blocks: directional derivatives `<g_s, p>` come from `r` and `Jp`; norms from the edge variables. **The oracle does
  not ask for unsummed blocks** (Nick: wait for a use case; his lean even then is to add hooks another way than through
  the oracle, the decision to be made on the facts at that time).
- **Before the release, after Parts 1 and 2: consider this backlog** (Nick, 2026-11-20: "let's make sure we take a
  moment to consider the backlog. We may make another Part 3 from the backlog before release").
- **Re-gauging for corewise optimization** (Nick agreed, 2026-11-20). Gauge drift (scale traded between neighbouring
  cores, or between `B_i` and `G_i`) degrades conditioning over long corewise runs. Recommended: an option on the corewise
  optimizers (or the corewise retraction) with a period and two modes. `'orthogonalize'`: the canonical form (what
  practitioners do), optionally followed by spreading the norm evenly over the cores, since orthogonalization alone
  puts all the scale in one core and the Hessian blocks then differ by about `||T||^2`; Adam's moments must be reset
  after it (a rotation has no diagonal image), L-BFGS pairs can be transformed exactly (`s' = A s`, `y' = A^-T y`).
  `'balance'`: a positive scalar per core equalizing norms, a diagonal change of coordinates that carries Adam's and
  L-BFGS's state exactly, essentially free and parallel (no latency-bound sweep of small factorizations on a GPU),
  fixing scale drift but not ill-conditioned bond matrices. Cost: one orthogonalization is about `max(n, r) / (c W)`
  gradients (`c` 3 to 6), so it is cheap unless the minibatch is below the rank; frequency tunes it. Defaults to be
  chosen on real problems (Darcy ladders), not CPU microbenchmarks.
- **A test cut held in reserve: the four-site half of `TestJetsTwoWays::test_every_variable`**
  (`tests/test_sampling_definitions.py`; Nick, 2026-11-20: "leave the tests in for now", the suggestion recorded). The
  test checks the 24 jetted variables of the reference module (`z`, `Jp` and the 22 edge variables) against an
  independent route: the same variable with no jets, at the input jets evaluated on two real node grids, interpolated
  into Taylor coefficients. It runs every jet case on three sites, and two cases (the line at order `(3,)` and the
  mixed modes) on the four-site problem `long()`. *Why it could go:* on four sites it is the most expensive part of
  the file; and the four-site problem's variables are also checked by `TestRecurrences` (each grade's recurrences),
  `TestReadouts`, and `TestOutputsAgainstDense` (`z` and `Jp` against the dense functions, at every jet case
  including the order-3 line), all of which keep running. *What would be lost:* the only check of the 22 EDGE
  variables against an independent route at full degree. On three sites a line's order-3 output has a structurally
  zero top coefficient (`z[i]` contracts `d - 1 = 2` line inputs, degree 2), so an error confined to the highest
  slots of the shared jet machinery (the roots-of-unity grid, the clipping) could pass on three sites and is caught
  there only on four; the outputs through the same machinery would still be checked. The reviewer of P4c and the
  implementer both named this as the one cut they would accept. *Measured savings* (2026-11-20, single process): the
  test takes 122 s on the floor env (`tttt`) and 67 s on `t3toolbox`; without the four-site half 48 s and 32 s: **about
  74 s on the floor and 35 s on the current env**, of the file's 224 s and 217 s. In the `-n 3` suite it is one test on
  one worker, so the wall-clock saving there is up to about that much, less when other tests fill the wait.

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
