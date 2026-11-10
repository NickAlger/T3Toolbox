# N8 review: the frontend twins (review stop)

Reviewer: Claude Opus 5.5, 2026-11-10. I reviewed the uncommitted working tree on top of 3054a0a6 against:
- the plan: §3.6, §3.7, §5.7, D-8, D-9, D-21, D-35, §6 T15 and T16, §12;
- `N8_handoff.md`;
- the session's CI rules: the `-W` docs build, dead or shadowed imports, suspicion of 1e-14 cross-path tolerances,
  and `-n 3` at most.

## Summary

Correct, and in the house split: the frontend holds descriptions, doctests, validation and class dispatch, and the
backend stays bare. My own checks:
- **Every twin is bitwise equal to its backend call.** That covers `T3Frame.padded`, `UT3Frame.padded` (with its split
  and masks), `fv_aligned_padded_frame` and `ufv_aligned_padded_frame` (frames and movements), unshared and shared,
  at C = () and (2,).
- **The frontend `newton_cg` with `padded_from` + `'cone'` equals the backend route bitwise.** Cores, `stats` and
  history (times excluded) match on both layers, unshared and shared.
- **Every hint and old-point error names its conversion.**
- **The safe-mode tied check** raises in safe mode and runs under `safety.unsafe()`.
- **Static checks.** No shadowed imports, no duplicate definitions, and no import cycle anywhere in the package.

There are no MAJOR or MINOR findings. The judgment calls are acceptable, with one maintainer item that matters to
users: eager-jax alignment accumulates thousands of compiled executables per rank level.

## Style (checked first)

- **The frontend carries the description, doctests, validation and dispatch.** The backend is untouched in N8.
  - `T3Frame.padded` and `UT3Frame.padded` are staticmethods returning the frame, and `(frame, split)` on the
    uniform layer.
  - The loop twins, `T3Tangent.align_padded` / `UT3Tangent.align_padded`, and `oversampled_ranks` on both tensor
    classes each make one backend call and rewrap.
  - Every twin has a user-level doctest.
- **The class dispatch on the hint is explicit and closed.** It is decided by `isinstance` on the layer's own classes.
  - Ragged: a `TuckerTensorTrain` → the T3 projection; an ndarray of `x_old`'s shape → the dense projection; a
    uniform object (recognized by `.to_t3`) → a `TypeError` naming `.to_t3()`; anything else → a `TypeError` asking
    for `project=`.
  - Uniform: a `UniformTuckerTensorTrain` with `x_old`'s `N` → the uniform projection; a wrong `N` → a `ValueError`
    naming `from_t3(..., N=)`; a ragged object (recognized by `.tucker_cores`) → a `TypeError` naming the
    conversion; anything else → a `TypeError`.
  - Recognizing the other layer by an attribute rather than its class avoids a sibling frontend import (§3.7). The
    only shape check is the dense hint's validation, never a dispatch.
- **The old point, one rule** (`_old_frame_data` / `_old_point_data`):
  - the layer check;
  - a structurally zero point (any Tucker or interior TT rank 0) is the empty frame, with no reduction (D-21); a mixed
    uniform stack raises;
  - otherwise the reduction to minimal ranks (the ragged three methods; `uniform_minimal`), then the orthogonal
    representation.
  - Safe mode checks tied factors under `sharing` and `x0.allclose(padded_from)`, and is skipped under `unsafe()`.
- **`newton_cg` mirrors the fourteen keywords exactly.** The N7 exemption is gone and the mirror test passes
  unexempted.
  - Only set options are forwarded, so without them the call is unchanged.
  - A corewise geometry with any of the three raises.
  - `start_frame` is class-checked, in safe mode orthonormal and representing the reduced `x0`, and converted under
    `use_jit`.
  - `gradient_descent`'s signature is unchanged (§5.7), and its docstring covers the `cone_*` pass-through.
- **Imports.**
  - The AST scan of all of `t3toolbox/`: no name both imported and defined, no duplicate top-level definition.
  - New frontend edges: `optimizers` → `frame_variations_format` and `uniform_frame_variations_format`;
    `uniform_frame_variations_format` → `backend.uniform_fitting`.
  - The whole-package graph (T18's file-level graph, all modules) has no cycle. The frontend `optimizers` already
    imported both manifold modules, so the sibling imports add no new kind of dependency.
- **No plan labels** in the shipped diff. The one plan reference is a code comment in the TEST file.

## Correctness

- **Twin ≡ backend, bitwise.** These are my checks (`n8_check.py`), all `True`:
  - `T3Frame.padded` against `fv_padded_frame` on the reduced old frame;
  - `fv_aligned_padded_frame` (T3 hint) against `padded_frames.fv_aligned_padded_frame` with the T3 projection;
  - `UT3Frame.padded` against `ufv_padded_frame` (supercores, masks, split);
  - `ufv_aligned_padded_frame` against the backend loop at the default level widths (NaN tails compared equal);
  - each of the four in four variants: unshared and shared `(0,1,1,0)`, at C = () and (2,). 16 in all.
- **`newton_cg` frontend against backend.** I ran `newton_cg(geom, 'probe', ..., padded_from=x_old,
  g0_reference='cone', padded_max_passes=3, cone_max_passes=3)` against `backend.newton_cg(setup.problem,
  setup.x0_cores, padded_from=<x_old reduced>, ...)`. Final cores bitwise, `stats` and every history row equal
  (times excluded): ragged and uniform, unshared and shared. 4 of 4.
- **The messages** (my probes): a uniform hint on the ragged layer; an unknown hint; a dense hint of the wrong shape;
  a ragged hint on the uniform layer; a wrong `N`; the wrong layer for either frame twin. Each raises the type above
  and names the fix.
- **Safe mode.** An untied `x_old` with `sharing=(0,0,1)`: `ValueError` naming `x_old.share(sharing)` in safe mode;
  it runs under `safety.unsafe()`.
- **Judgment calls.**
  1. **One test file** instead of the plan's seven: accepted. The twins are one feature across five modules with
     shared cross-layer and zero-point setups, and the keyword mirror stays in `test_optimizers_frontend.py`.
  2. **A structurally zero `padded_from` is passed through, not canonicalized: accepted.** The plan's "convert
     first" exists to skip the reduction and the orthogonalization, and both are skipped. The members apply the same
     D-21 rule, and T14 (h)'s frontend half shows both spellings give the same iteration-0 frame bitwise.
  3. **`gradient_descent` via `**kwargs`: accepted** (§5.7).
  4. **`jax.clear_caches()` after the heavy tests: an acceptable remedy and placement, with a note.**
     - The cause is measured: 8,000–9,500 mappings per eager-jax alignment test, against the 65,530 limit.
     - Clearing at cleanup cannot affect the compile-count assertions. They measure `_cache_size()` or monitoring
       deltas within one test (N3b's T13 (a), N6's T13 (b, c)). The lru-cached `_jitted` wrappers survive and only
       their programs go.
     - Per-test `addCleanup` on the eight `TestDispatch` methods (the rest of that class should keep its caches) and
       `setUp` on the four classes is the right granularity. A module-wide autouse fixture would clear after
       hundreds of unrelated rows.
     - A small decorator could replace the twelve `addCleanup` lines (NIT 1); not required.
     - The real lesson is for users, not tests (maintainer item 1).

## Findings

### 1. NIT: the twelve `self.addCleanup(_release_jax_programs)` lines could be one decorator or mixin

`tests/test_dispatch.py`.

**Fix.** Optional: `@_releases_jax_programs` on the eight methods and a `_ReleaseJaxPrograms` mixin for the four
classes. The behaviour stays the same.

### 2. NIT: the wrong-`N` message suggests a repack that fails on a stacked hint

`ufv_aligned_padded_frame`'s wrong-`N` message suggests `from_t3(hint.to_t3(), N=...)`.

**Problem.** For a STACKED uniform hint, `to_t3()` returns a nested tree, not one `TuckerTensorTrain`, so the
suggested call fails.

**Fix.** Optional: say "repack it to N=… (per element when stacked)", or point at the frontend conversion that
handles stacks if one exists.

## What I ran (env `t3toolbox`: py3.11, numpy 2.x, jax 0.10.2; `-n 3` at most)

1. **`n8_check.py`:** the 16 twin-against-backend checks, the 4 `newton_cg` frontend-against-backend checks, the 7
   message probes and the safe/unsafe pair. Results above.
2. **The AST scan** for shadowed or duplicate definitions over `t3toolbox/`: clean. The whole-package import-graph
   cycle check: none.
3. **Tests.** TESTS_PLACEHOLDER

## For the maintainer (review stop after N8)

1. **Eager-jax alignment is a process-lifetime resource problem for users, not only for the test suite.**
   - Each eager-jax alignment at a new structure compiles hundreds of executables. N3 measured 932 at d = 4, and the
     implementer measured 8,000–9,500 process memory mappings per alignment test. A new rank level is a new
     structure.
   - So an eager-jax rank-continuation run that aligns at every level could pass `vm.max_map_count` (65,530) after
     roughly seven levels and abort with `LLVM ERROR: Unable to allocate section memory!`. That is the failure
     `scan_body_principles.md` records.
   - `use_jit=True` compiles 2–4 programs per level (ragged) or one (uniform).
   - Recommendation for N9's "jit or eager" section: with jax inputs, use `use_jit=True` for the alignment, and
     `newton_cg`'s `use_jit` forwards it. Optionally, the loops could warn once when called eagerly on jax inputs.
2. **T15 lives in one file** (judgment call 1). Update the plan's §6 to name it.
3. **The structurally-zero rule is implemented as "skip, pass through"** rather than "convert to canonical rank-0"
   (judgment call 2). Behaviour is equal and tested; only the plan's wording differs.
4. **Basis dependence of the hint-free start** (your N7 ruling: document in N9). The twins inherit it: `T3Frame.padded`
   and `UT3Frame.padded` of the same orthonormal `x_old` differ in span. N9's text should say so where the twins are
   documented.

## Counts

MAJOR 0, MINOR 0, NIT 2.

VERDICT: APPROVE
