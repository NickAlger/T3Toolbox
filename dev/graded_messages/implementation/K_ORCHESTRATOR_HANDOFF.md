# The packing sub-arc (K0 to K7): handoff to the orchestrator

Written by Claude on 2026-11-25, at the end of the planning session, for the same session after a compaction (or for
a new one). **You are the orchestrator of this arc** (the maintainer, Nick, asked for it). Nothing of the arc is
built. Everything you need is in files; this page says what they are, what the maintainer asked, and what to watch.

## 0. RESUME HERE (written 2026-11-26, 01:50 CT, at the session's stop)

The rest of this page was written before the arc started and is still the background (sections 2 to 6). **What has
happened since, and where to pick up:**

- **K0 is done**: `282b46a9` on `graded-messages`, pushed, CI green on both legs.
- **K1 is being built on the side branch `k1-wip`** (pushed to the remote as a backup; CI does not run on it). Its
  design is approved (`K1_DESIGN_NOTE.md`, revision 2, with `k1/prose_edits.md`). **Parts 0 and 1 are reviewed,
  signed off and committed there** (`78a34d77`, `31e3d100`). Parts 2 to 5 are not built: 2, the engine and its
  steps; 3, `jets`' public surface; 4, the fitting layer and the translation; 5, the frontend's names, the
  deletions, the pages, and the full gate. When part 5's gate passes, the branch is squashed into one commit on
  `graded-messages`.
- **The next step is part 2's build.** Its public docstrings are written in full and signed off after two rounds
  (`k1/prose_edits.md` §3, the one text the build uses; `K1_part2_PROSE_REVIEW_r1.md`, `_r2.md`,
  `K1_part2_PROSE_RESPONSE_r1.md`). The order of the build and what its review checks first are in the two
  carry-over notes. Part 2 ends **red by design** (the list is §3.11 of the prose file), recorded by a run, per
  subtest. Then, before each of parts 3, 4 and 5 is built: its prose round (`K_ARC_DECISIONS.md` 28).
- **The agents do not survive a new session.** Start a fresh implementer on `K_IMPLEMENTER_CARRYOVER.md` and a fresh
  reviewer on `K_REVIEWER_CARRYOVER.md`, each with the orientation the maintainer asks for, `K_BRIEF.md` first. An
  agent lasts about one part after its orientation: ask each for its remaining context after every turn and rotate
  at part boundaries. The reviewer orients while the implementer builds and writes its own checks before it sees the
  build (this found real things every time).
- **Read first:** `LOG.md` from "The packing sub-arc (K0 to K7) starts" to the end; `K_ARC_DECISIONS.md` whole (41
  entries and the maintainer's instructions during the arc: the `typ.Any` audit after K7, with
  `sampling_definitions` exempt; "don't worry about the token cost"); then this page's sections 3 to 6.
- **Waiting for the maintainer:** nothing blocks K1. At K2's design: whether the full weight gets public checkers of
  its own (`K_ARC_DECISIONS.md` 29). After the arc: the discussion of the decision log, and the `typ.Any` audit.
- **The working tree** is on `k1-wip`; the untracked files at the repository root (`release_notes_2026.*.md`,
  `dev/check_t3svd_retraction_second_order.py`) are the maintainer's and are left alone. Disk: about 8 GB free.

## 1. Where things stand (at the arc's start, 2026-11-25)

- Branch `graded-messages`; the plan and its records were committed on 2026-11-25 (`git log -1` shows the commit, "dev:
  the packing sub-arc planned"). Not pushed at the time of writing: ask the maintainer before the first push, or push
  with K0 as the process below does.
- **The plan is signed off** by its critic (four rounds) and by the maintainer ("I sign off on the design"):
  `dev/graded_messages/PACKING_PLAN.md`, revision 5, 972 lines. Eight slices, K0 to K7, before P17.
- The M arc's logged decisions were ruled in the same session (`M_ARC_DECISIONS.md`, last section). Still pending with
  the maintainer, and not this arc's: the sampling audit's decisions 3 (corewise placement) and 4 (the name
  "Hessian"). After K7: the small `from_t3` slice, then P17, P18, P19 (P20 is K0 now).
- **After K7, with the maintainer (his request, 2026-11-25): an audit of every `typ.Any` in the sampling layer and
  what is downstream**, case by case: why it is there, and whether it can be made strict (`sampling_definitions` is
  exempt, his ruling). The handovers of K1 to K7
  list the ones each slice leaves (`K_BRIEF.md` §5); the count before K1 is in `K_ARC_DECISIONS.md`.
- Untracked files to leave alone: `release_notes_2026.*.md`, `dev/check_t3svd_retraction_second_order.py`.

## 2. Read these on resuming, in this order

1. This file.
2. `dev/graded_messages/PACKING_PLAN.md`, whole. It is the specification. §7 has the slices; §12 to §14 the decisions.
3. `dev/graded_messages/PACKING_PLAN_NOTES.md` §7 to §9 (the maintainer's review, with his words). The earlier
   sections are the planning session's rulings in order; the plan already carries them.
4. `implementation/BRIEF.md` and `implementation/ORCHESTRATION.md`: the process documents of the P and M slices (the
   agents' brief; what the orchestrating session does per slice; the gate; lessons learned the hard way). **In the
   planning session these were read by a sub-agent, not firsthand: read them yourself.** `run_gate.sh` is the gate.
5. As models: `implementation/M2_DESIGN_NOTE.md` and `M2_DESIGN_REVIEW.md` (a slice's plan and its critique),
   `M2_part4_HANDOVER.md` (a handover), `M_ARC_DECISIONS.md` (the decision log the maintainer discusses afterwards),
   the tail of `LOG.md` (the M arc's entries).
6. `reviews/packing_plan/REVIEW_r4.md` (the critic's last notes for the builders) and `REVIEW_r1.md` §1 (its
   experiments; the scripts are kept in `packing_audit/critic_experiments/`).
7. For K0: `packing_audit/K_MEASUREMENTS.md`, section M1.

Also re-read the root `CLAUDE.md` (house rules) and the memory index. Two memories bind this arc: no direct library
edits outside the process (the process is now authorized: the agents implement, you do not), and "comments are
content".

## 3. The process the maintainer set (2026-11-25, his words)

"The implementation will take the normal approach: an orchestrator oversees two Opus 5.5 agents on high effort, an
implementer and a reviewer, both of which are instructed to get familiar with the library before starting, and to use
judgement and first principles thinking rather than blindly following rules. Each slice includes a plan phase and an
implementation phase. In the plan phase the implementer writes a more detailed plan about the specific slice, the
reviewer critiques it, and they go back and forth until the reviewer signs off. Then they send it to the
orchestrator, who can approve or send it back. Once approved, the implementer implements the slice, the reviewer
reviews their implementation, and they go back and forth. Once the reviewer signs off, they send it to the
orchestrator who either approves and initiates the next slice, or sends it back for more work. The orchestrator is
advised to use their judgement for decisions along the way, except for big decisions or major unforseen problems
which they can pause and talk to me about. The orchestrator logs all decisions, and then after the entire arc is
complete, discusses it with me. I would like you to be the orchestrator."

**Precedence (the maintainer, 2026-11-25): "my instructions here in chat supercede those in BRIEF.md and
ORCHESTRATION.md wherever they conflict."** The quotation above is the process. `BRIEF.md` and `ORCHESTRATION.md`
are the older practice: use them for what the quotation does not say (the gate, the artifacts, the lessons), and
where they differ from it (for example a rule to be followed where he asks for judgement, or a different sequence
of approvals), the quotation wins. Tell the agents the same in their brief.

How that maps onto the existing practice (to be checked against `ORCHESTRATION.md` when you read it; the mapping
below is Claude's, and yields to the quotation too):

- **Per slice:** `K<n>_DESIGN_NOTE.md` (the implementer) -> `K<n>_DESIGN_REVIEW_r<k>.md` (the reviewer), rounds until
  the reviewer signs off -> **you approve or send back** -> the build -> `K<n>_HANDOVER.md` -> `K<n>_REVIEW_r<k>.md`
  and `K<n>_RESPONSE_r<k>.md`, rounds until the reviewer signs off -> **you approve or send back** -> you write the
  `LOG.md` entry, commit (staging files by name, one commit per slice), push, and read both CI legs (draft PR #26;
  `gh run list --branch graded-messages`). Large slices (K1, K4, K5) in reviewed parts on a local side branch,
  squashed; the branch may be wrong between parts if the squash is right (the maintainer, 2026-11-24).
- **The agents:** two, started with the Agent tool, `model: opus`, `effort: high` (the maintainer's explicit
  instruction), kept from slice to slice with SendMessage while their context holds; when one fills, it writes a
  carry-over note and you start a replacement on it, and log it. Their brief must say: read `BRIEF.md`; get familiar
  with the library first (methods of `TuckerTensorTrain` traced into the backend, the engine
  `backend/sampling_variables.py`, and the modules their slice touches); use judgement and first-principles thinking,
  not rules followed blindly; stop and report when the plan cannot be done as written.
- **Agents never commit, push or switch branches. The implementer and the reviewer never run tests at the same time.**
  The gate runs at `-n 2` (a 13 GB laptop; the suite peaks near 10 GB): one heavy run at a time, never Sphinx beside
  tests. Expected-red lists are recorded by a run, never predicted.
- **Your log:** create `implementation/K_ARC_DECISIONS.md` with the first decision you make, one numbered entry each
  (the decision, the reason, where it is recorded), as `M_ARC_DECISIONS.md` does; larger rulings also in `LOG.md`. The
  maintainer discusses it after the arc.
- **When to pause for the maintainer** (my reading of "big decisions or major unforeseen problems"): a departure from
  a ruling in `PACKING_PLAN_NOTES.md` or from the design of the plan; a public name the plan does not have; a change
  of the slice order or of the scope (sampling and what is downstream of it); a silent wrong answer found in released
  behaviour; anything that would need a new principle. Small or clear things: decide, log, go on.
- **Budget.** The maintainer is on a Max plan and watches token use (memory: agent budget rules). He has authorized
  this arc's two agents. Do not add agents beyond the pair (and a replacement when one fills) without telling him;
  measure what K0 costs and tell him, so he can judge the rest.

## 4. The slices, with what to watch in each

The plan's §7 is the specification; these are the points the critic's four rounds raised that a builder can miss.

| Slice | Content | Watch |
|---|---|---|
| **K0** | Delete the old kernels, the old kinds, the bridge (the plan's P20, moved) | Begin with a fresh inventory. The orphan scan of `K_MEASUREMENTS.md` M1(a2) reads top-level definitions only: it lists `_BATCH_FIELDS` as unreferenced, which the batch's pytree registration uses. Tests that stay but use an old kernel as their oracle must be re-based on an independent one, or deleted with a reason. The gate is the full gate, the slow tier, both CI legs, and no code reference to a deleted name; not P20's grep as written (the pages' prose is P19's, `chunk_size` in the translation is P17's). |
| **K1** | The weights (`block_weight`, `full_weight`), `y` exact, the metric split by kind, the kind and the order into the batch | **Part 0 first:** the move of `kind` and `order` and the batch's methods, a refactor with a bitwise battery, reviewed apart. Its sites outside `backend/fitting.py` are listed in the plan (the display reads `oracle.order` and would print positional labels silently; the default draw; the `Oracle` protocol's text; the translation). The oracle's constructor checks (O-44, O-46) move to the batch's. Then bottom-up: the dense reference first, so its errors cannot correlate with the engine's. Optimizer runs are compared on gauge-invariant quantities. The largest slice. |
| **K2** | `backend/packing.py`, additive; `frame_stack_shape` public; the `PerSite` `TypeVar` | CI's Sphinx build is where a constrained `TypeVar` is first rendered; the fallback is in the plan. The old checkers stay until K3. |
| **K3** | The ragged layer takes sequences only (the sample, the data, the entries index as a tuple) | The ragged `entries` methods must stop calling `_as_index` here (it turns `(1, 2, 3)` into one array); `index_to_one_hot` takes the tuple; `pack_batch` gains its `pack_index` branch; the ragged functions refuse a uniform object. |
| **K4** | The frontend: `PackedVectors` (field `mode_sizes`), the uniform methods, the frontend batch class | Before the backend refuses anything. The refusal of the backend's batch at a frontend entry starts here. The design note must measure the tests that mix the tiers (`tests/_fits.py:232`, `tests/test_dispatch.py`, `tests/test_jax_absent.py`, `tests/test_least_squares_oracle.py`, `tests/oracle_sweeps/*`) and the short cross-references to `LeastSquaresBatch` in the docs. The translation builds the frontend batch from here and packs at `N = x0.N`, its validation data too. |
| **K5** | The uniform backend takes packed only; masking by selection; the data-space steps and the shape reads as single operations; the program-size gate | Selection: the zero has the array's dtype; comparisons with the tree before are on values, not bytes (signed zeros). The gate measures the optimized program (`jax.jit(f).lower(...).compile().as_text()`) at `d = 3` and `5`, never `1` or `2`. `ut3_unscale_coordinates` masks its output. |
| **K6** | The remaining `kind`-switched helpers, ragged only; `aligned_padded_frame` | Small. Each new function against the one it replaces, bitwise. |
| **K7** | The guards: hostile fixtures by default; the junk-in coverage meta-test | The surface it enumerates is sampling and what is downstream, not every uniform `__all__`. |

## 5. The rulings, in one screen (the notes have the words)

- **The definition:** the container is the form (a tuple or a list of `d` arrays is ragged; one array is packed).
  **No mixing. No guessing** (the function's name or an explicit argument says what the container cannot).
- No sampling function packs or unpacks. The ragged layer takes sequences only, the entries index included (a tuple
  of `d` integer arrays; lists accepted; no broadcasting between them).
- **G1:** the mode sizes are checked on the ragged form (backend checkers, plain arguments `shape`, `stack_shape`);
  automatic in the frontend, where a `PackedVectors` carries its `mode_sizes`.
- **G2:** liberal in, canonical out, for sampling and what is downstream. The engine masks everything it reads by
  selection (`where`), the sample included. Contract: data may hold anything in its padding; cores must be finite.
- **G3:** one operation per function, one return type; the engine (and `computed_names`) the only exemption; a `kind`
  argument is split when the computation or the structural check of its arguments depends on the kind; each typed
  face fixes its layer.
- **The weights:** `M = gamma * block_weight**2 * full_weight**2`. A size of 1 only on stack axes (`W`, `C`), never on
  structural ones (the mode, the jet slots, the edge); `y` exact on `W`; the modes of a probe object of the type of
  `z` agree in everything but the edge. No `sample_weight` yet.
- **The batch knows its kind** (`kind`, `order` static fields; `check`, `take`, `n_measurements`, `data_metric` its
  methods); `LeastSquaresOracle()` keeps `query`. **The frontend has its own batch class** of the same name, of
  `PackedVectors`, handing the backend `.data`; messages and `repr` name the module.
- **Helpers:** `probe_data_metric` and `affine_jet` are parametric over the forms; every other probe helper is ragged
  only (prepare ragged, pack last). Names are kind first.
- **The per-mode loops:** on the uniform layer a map over the modes is one array operation; the conversions and the
  dense reference are exempt.
- **Deferred:** the option to skip the entry masking, to Part 2; the uniform library's padding audit, the upstream
  G3 cleanups and a `sample_weight`, to the backlog after Part 2.
- From the M arc's rulings, binding here too: bitwise agreement of optimizer paths is welcome, not required; a side
  branch may be wrong between parts; comments are content (a scripted edit of comments is checked for ASCII diagrams
  and aligned signature columns; when in doubt, review).

## 6. What the planning session taught

- **A design made in one pass in conversation has seams.** Each round of the critic found real ones: a slice order
  that would have turned the frontend red; a wrapper class that broke the shared batch; a kind-free metric that could
  not check its block weight; consumers of fields that had moved. Expect the same of a slice's design note, and have
  the reviewer run small experiments against its claims, as the critic did.
- **Claims about the code are checked in the code,** with `file:line` at a stated commit and counts with their
  method. The plan's are at `3375d213`; every slice re-measures on its own tree.
- **The maintainer** leads with principles and wants the reasoning and a recommendation before a question; brief,
  plain prose; pros and cons on both sides when he is undecided. He does not want to micromanage the order of work.
- **Environment:** `/home/nick/miniconda3/envs/t3toolbox/bin/python` (py3.11, numpy 2.4, jax 0.10) with
  `PYTHONPATH=/home/nick/repos/T3Toolbox`; the floor env `tttt` (py3.9, numpy 1.22, jax 0.4.30) for reproducing floor
  failures; `gh` is authenticated. The gate: `dev/graded_messages/implementation/run_gate.sh` (see its header and
  `ORCHESTRATION.md`).

## 7. The first steps after reading

1. Tell the maintainer you are back up to speed, in a few lines, and what you will do first.
2. Start the pair on K0: the orientation (read-only), then K0's design note and its review.
3. Open `K_ARC_DECISIONS.md` and the `LOG.md` entry for the arc's start.
