# Implementer carry-over (K arc)

Written 2026-11-25 by the K0 implementer, after K0's design revision 2, so that a successor can build K0 without the
conversation. The records are authoritative: `K0_DESIGN_NOTE.md` (revision 2), `K0_DESIGN_REVIEW_r1.md`,
`K0_DESIGN_RESPONSE_r1.md`, `k0/prose_edits.md`, `k0/inventory_before_k0.md`. Here is only what they do not say.

## Tools in `k0/`

- `k0_inventory.py`: run from anywhere with the env's python. Plain run = the report; `--freeze` rewrites
  `k0_names.json` (**do not run it on K0's tree**: the names come from the tree before K0); `--check` is the gate
  (exit 0 when no reference is left; today 1,277). `STRING_ALLOWLIST` is near the end of the file; `DEAD` near the top
  (the two dead scan bodies). Its "Imports ... unused after K0" and "Prose of kept code" sections are the build's
  checklists.
- `simulate.py OUTDIR`: a rough K0 library (deleted statements cut). Good for the import check and the scan counts
  of the interpreter tests (`test_contractions_interpreter.scan_vocabulary` with `PYTHONPATH=OUTDIR:<repo>/tests`).
- `tally.py BASE NEW`: per-node comparison; it reads the two renames of note §3.5.
- `experiments/`: E1-E5 and `mutate_step.py STEP EPS` (run inside a scratch copy's root; it appends a wrapper to
  `t3toolbox/backend/sampling_steps.py`, so restore that file after each mutant).

## Prototypes already written (scratch, may be gone after a reboot)

`scratchpad/implementer/proto_test_curvature.py` (a copy of today's file with one test edited) holds the re-based
`TestPlainKindsTwoHole.test_residual_weights` (forward = `probing.t3_probe` / `bapply.t3_apply` /
`bentries.t3_entries`; `w2 = r` for no weight, `[ri * wi**2 ...]` for probe's per-mode row): it passes on today's
library. E3 (`experiments/e3.py`) is the body of the re-based jvp rows, ragged and uniform (the uniform `ut3s.*`
return `(d, K, ...)`; move the K axis to the front as the existing `_check` expects). E1c is the body of the hygiene
negative control. E4 and E5 are the oracles of the two `test_sampling_definitions` / `test_tucker_tensor_train`
re-bases (E5 still uses `taylor_to_derivatives`: the note rules factorials inline instead).

## Things learned that are easy to trip on

- Background runs: `setsid` forks, so `$!` is not the pytest pid; find it with `ps -eo pid,args | grep "[p]ytest ..."`.
  The baseline took 38:50 at `-n 2`; the slowest tail is `test_dispatch` and `test_sampling_uniform`.
- The baseline was run on a `git archive` of HEAD in scratch with `python -B` (the copy is deleted).
- `tests/test_dispatch.py` turns on `jax_enable_x64` at import; under xdist that leaks into other tests of the worker.
- The uniform tangent faces re-mask twice (engine exit mask, then `_remasked_transpose_output`); the corewise faces
  once. Any negative control on the uniform transposes must remove both layers.
- In the engine, removing one mask family in general position changes nothing (layered masking); controls only bite
  at TT slack or with every mask removed.
- `ut3_*_corewise_transpose`'s probe `c` is packed by the face with `pack_if_ragged`; the engine's held `Jp` must be
  packed for probe on the uniform layout.
- `git status` shows `K_ARC_DECISIONS.md` modified and `k0/REVIEWER_PRENOTES.md`: not the implementer's.

## K0 build: where it stands (updated as it goes)

- 2026-11-25, build step 1 done: every test edit of the note (§3) is in the working tree, and `k0/prose_edits.md`
  has the final `fitting` docstring (round 3). Tools: `k0/edit_tools.py` (removes defs by name),
  `k0/unused_imports.py` (imports newly unused against HEAD), `k0/delete_library.py` (the mechanical library cut; run
  once, refuses a library that differs from b5125f76).
- Recorded step: `k0/logs/pre_deletion_edited_tests_v.log.gz`: the edited files on the untouched library, 1 failed
  (only the new absence test, expected), 519 passed, 2 skipped, 37,798 subtests. Then the slow tier of the edited
  `test_uniform_curvature.py`: `k0/logs/pre_deletion_slow_uniform_curvature.log`.
- Next: `delete_library.py`; the hand rewrites of `prose_edits.md`; the kept-comments diff check; `--check`; imports
  both ways; the gate (`-v` suite gzipped in `k0/`, then the other steps by name, then `slow`); tally; mutations;
  `K0_HANDOVER.md`.
- Build done (2026-11-25): library deleted and rewritten, gate green (the suite's one failure, a third scan floor, fixed
  and its file rerun), tally accounted, mutations rerun; `K0_HANDOVER.md` written. Logs: `k0/k0_suite_v.log.gz`,
  `k0/tally_baseline_vs_k0.txt`, `k0/logs/`. Nothing running. The third floor waits for the orchestrator's confirmation.

## K1 (the second implementer, from 2026-11-25 evening)

- The plan phase's record is `K1_DESIGN_NOTE.md` (complete for review) with `k1/prose_edits.md` (first draft; its
  status line says what is in full). Tools in `k1/`: `k1_sites.py` (stdlib `ast` + `doctest` scan of every K1 site in
  five places; 5 s; `--sites` for the full list, `--json F`), `signatures.py` (verbatim headers; rerun for the
  handover's "after"), `check_math.py` (the dense checks of note §3; all pass), `kwonly_check.py` (the batch's
  keyword-only `__init__` on a model class; passes on the t3toolbox env and on `tttt`, the 3.9 floor).
- Outputs kept: `k1/sites_before_k1.md`, `k1/sites_before_k1_full.md`, `k1/signatures_before_k1.md` (rerun on K0's
  commit `282b46a9`: identical).
- Not yet written (build-phase, part 0 first): `k1/battery.py` (note §7, after `m2/battery.py`), `k1/diff_audit.py`
  (after `m2/diff_audit.py`), the `-v` baseline log of the starting tree.
- `ast.get_source_segment` is quadratic on long files (it splits the text at every call): `k1_sites.py` slices the
  lines itself; a first version took minutes.
- 2026-11-25 ~21:10 CT: design round 1 answered (`K1_DESIGN_RESPONSE_r1.md`; note revision 2; prose revision 2 with
  parts 0 and 1 in full). Rulings to remember: `flat_draw(batch, batch_size)`; the frontend guards of a drawn and a
  validation batch's `(kind, order)` are part 0's (not K4's); no sentinel default for `sample`; D4 confirmed; a part's
  build starts only when its public docstrings are in full and reviewed (parts 2 to 5 need a prose round each).
- Next, once the note is signed off: write `k1/diff_audit.py` (note §7.4 (a)-(c)) and `k1/battery.py` (§7), record
  the battery `old` on `282b46a9` in scratch (`git archive`), then build part 0.

### Part 0 build log (k1-wip, from dfb9585f)

- 21:30 CT: round 2 folded (`K1_DESIGN_RESPONSE_r2.md`). `k1/battery.py` written; the `old` record taken on a
  `git archive 282b46a9` (t3toolbox + tests) in `scratchpad/implementer2/old_tree`, kept in
  `scratchpad/implementer2/battery/old.{npz,json}` (3 MB; NOT in the repository; regenerable in 10 s:
  `cd old_tree && PYTHONPATH=$PWD python -B .../k1/battery.py <prefix> old`; two runs are bitwise identical).
  `k1/diff_audit.py` written with audits (a) and (b); (c) (the weights) is to be added before part 1.
- ~21:35 CT: part 0's library, test, example and page edits are in the working tree (k1-wip, uncommitted). Scripted
  steps: `k1/part0_rewrite.py` (constructions; log `k1/part0_rewrite_log.txt`), `k1/part0_methods.py` (the moved
  methods; log `k1/part0_methods_log.txt`), the rest by hand (the 23 MANUAL sites of the first log, the no-draw tests,
  the oracle/batch-constructor tests, the guards' tests). The audit (b): `k1/part0_audit.txt` (PASS). The battery
  `old` vs `part0`: **bitwise, 9,156 keys** (`k1/battery_old_vs_part0.txt`; the part0 record in scratch).
  A scripted wrap of long lines went wrong (it rewrapped pre-existing lines) and was undone by joining back against
  HEAD; check `git diff` for stray reflows if in doubt.
- ~21:50 CT: evidence so far: battery bitwise (above); audit (b) PASS and the line audit (`k1/part0_audit.txt`: no
  tolerance changed; assertions changed only in the new guard / constructor tests); kept-text check
  (`k1/part0_kept_text.txt`); mutation script written (`k1/part0_mutations.py`, not yet run). A targeted run of the
  20 changed test files is going (log `scratchpad/implementer2/p0_targeted.log`). Then: the full `-v` suite (`-n 2`,
  detached, gzipped into `k1/`), the tally against `k0/k0_suite_v.log.gz`, doctest-modules, quickstart, pages,
  examples, the mutations, the handover (`K1_part0_HANDOVER.md`, draft written).
- ~22:10 CT: the targeted run of the 20 changed test files found 9 failures, all test sites the scripts had missed
  (a subclass constructed through a variable, `cls('probe', ())`; `topt._setup(geom, oracle, batch, x0)` x3;
  `_NoResolutionOracle('apply', ())` in the frontend file; the `flat_draw` argument-name pin; a jitted test of mine
  summing a list): fixed, and those tests rerun green. The full `-v` suite (`-n 2`) started at ~22:10, log
  `scratchpad/implementer2/part0_suite_v.log` (to be gzipped into `k1/part0_suite_v.log.gz`).
- ~22:55 CT: full `-v` suite green: 1,887 passed, 5 skipped, 1 xfailed, 90,148 subtests (`k1/part0_suite_v.log.gz`);
  tally vs K0 in `k1/tally_k0_vs_part0.txt` (one node had moved class by mistake: fixed after the run, the file to be
  rerun). Gate steps (`scratchpad/implementer2/gate_part0`): doctest-modules 321 passed, quickstart, pages (47) pass;
  examples: `fit_hilbert_from_probes_adam.py` failed (my construction line clobbered its `batch` size variable):
  renamed to `drawn` in it and in the optax twin, to rerun. Sphinx running. After it: rerun the two examples and
  `tests/test_least_squares_oracle.py`, the mutations (`k1/part0_mutations.py`), the battery again (annotation-only
  edits came after the first part0 record: `minibatch_scale`'s type, `_setup`/`_frontend_draw` annotations), the
  audit, then the handover.
- ~23:05 CT: part 0's evidence complete but the slow tier (running, `scratchpad/implementer2/gate_part0_slow`): the
  two examples rerun green, the moved test and `TestCheckBatch` rerun green, the battery re-recorded bitwise, the
  audit PASS, all 7 mutations caught, Sphinx 0 warnings. `K1_part0_HANDOVER.md` written (only the slow tier's line is
  pending). Next after the report: the reviewer's review of part 0; then, before part 1's build, part 1's prose is
  already in full (`prose_edits.md` §6) and `diff_audit.py` needs its audit (c) (the weights).
- 23:22 CT: **part 0 built, handover written, every step green** (slow tier 52 passed, 13,076 subtests). Waiting for
  the review of part 0. Nothing running.

## K1: where it stands for a successor (written 2026-11-26 ~00:20 CT by the second implementer)

**Done:** the design (note revision 2 + round 2's findings folded; signed off and approved) and **part 0** (built,
reviewed SIGN OFF, the orchestrator's four fixes made: `K1_part0_RESPONSE_r1.md`). Part 0 is to be committed to
`k1-wip` by the orchestrator. **Next: part 1** (the dense reference `sampling_definitions` and the test oracles),
note §2.2 part 1, §4.1, §6.1-§6.3; its prose is written in full in `k1/prose_edits.md` §6 (part 1's reference texts)
and needs no further prose round.

**Tools in `k1/` (each with a header saying how to run it):**
- `k1_sites.py`: the site scan (stdlib, 5 s). `signatures.py`: verbatim headers. `check_math.py`: the dense checks.
- `battery.py OUT FORM` with FORM `old | part0 | new`, and `--compare A B --expect part0|new`. The `_Api` class is the
  only place the trees differ; for part 1 onward the `new` form uses `block_weight=`/`full_weight=` (already written;
  the engine's keywords too, `engine_weights`). The `old` record: `scratchpad/implementer2/battery/old.{npz,json}`
  (regenerate: `git archive 282b46a9 t3toolbox tests | tar -x -C <empty dir>`, then from that dir
  `PYTHONPATH=$PWD python -B <repo>/dev/graded_messages/implementation/k1/battery.py <prefix> old`; deterministic).
  Parts 1 to 4 compare their own engine/oracle rows as they become `new`-shaped; the full `old` vs `new` comparison is
  part 5's (note §7.2). Note: until part 4 the batch still has the one field `weight`, so a mid-K1 tree is neither
  `part0` nor `new` for the batch rows: run only the rows a part changes, or extend `_Api` with an intermediate form.
- `diff_audit.py BASE --part 0 --out FILE`: audits (a) the lines and (b) the kind/order pairing. **Audit (c), the
  weights (note §7.4 (c)), is not written yet and is needed from part 1 on:** pair each removed
  `blocks_to_weight(k, o, b, y)` with a `block_weight=b` in the same scope (one assignment deep on the old side);
  pair each old `weight=` keyword with the new call of the same callee in the scope, classify it
  (`block_weight=` / `full_weight=` / removed); require a reason line in `k1/full_weight_reasons.tsv` for every new
  `full_weight=` (mark apply and entries sites); list new `block_weight=` sites whose old value was not a
  `blocks_to_weight` output. Reuse `scopes_of`, `_Walker` (add a `kw_weight` record to `Scope.calls`, which already
  holds the keywords). The reviewer (round 2 of the design) listed its blind spots: new-side indirection, positional
  weights to test helpers, string keys.
- `kept_text_check.py BASE`: every changed statement of the touched library modules, header changed or not.
- `tally.py BASE_LOG NEW_LOG`: per-node counts; its `RENAMED` table is part 0's (edit it per part).
- `part0_rewrite.py`, `part0_methods.py`: the scripted rewrites of part 0 (models for part 1's keyword moves).
- `part0_mutations.py SCRATCH`: the mutation runner (a list of (label, file, old, new, tests)); copy it per part.
- `part0_floor_smoke.py`: the batch on both Python environments.

**Pitfalls met (do not repeat):**
- **Never let a script re-wrap lines it did not create.** My `wrap_long.py` (scratch) rewrapped pre-existing long
  lines because of a `| {l}` in its membership test; it was undone by joining lines back against `HEAD`. Wrap only
  the lines the part added (`git diff -U0` `+` lines), by hand where a comment is involved, and **never re-flow a
  docstring paragraph to change one sentence** (the maintainer's rule, repeated by the orchestrator): split only the
  line that holds the change so the following lines stay byte for byte; a whole re-wrap is fine only where an inserted
  sentence forces it.
- Run the examples: a construction line I added in two examples named its result `batch`, shadowing the example's
  minibatch size; only the examples step caught it.
- A test-local subclass of the oracle constructed through a variable (`cls('probe', ())`) and `topt._setup(geom,
  oracle, ...)` calls escaped the scripted rewrites: a targeted run of the changed files first finds such sites fast.
- Do not introduce helper lambdas to shorten construction lines: they hide constructions from audit (b) (it flagged a
  COUNT mismatch); wrap the explicit calls instead.
- `pytest` under xdist: the gate's `run_gate.sh wait LOGDIR 170` is the way to wait (foreground commands are
  backgrounded after ~3 min); `timeout 175 tail --pid=PID -f /dev/null` works for a plain background run.
- The `typ.Any` rule (`K_BRIEF.md` §5; `sampling_definitions.py` exempt): each handover lists the `typ.Any`s left in
  the signatures it touched.
- Times measured: full `-v` suite at `-n 2` 34 min; doctest-modules 1 min; pages 20 s; examples 9 min; Sphinx 12.5 min;
  slow tier 24 min.
