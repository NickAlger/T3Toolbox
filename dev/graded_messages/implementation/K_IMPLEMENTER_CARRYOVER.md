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
