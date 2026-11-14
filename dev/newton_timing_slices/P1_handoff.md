# P1 handoff: the synchronization helper, the loops' timed twins, the members' `timing` (Newton timing, 2026.5.0)

Implementer's record for the session. Base: HEAD `7a045cb3` (the plan at draft 3; Q-1 = (a), not read by P1).
Nothing is committed. Scope: plan §7 row P1, exactly —

- `common.tree_block_until_ready`;
- the loops' private twins (`_fv_aligned_padded_frame`; `_ufv_aligned_padded_frame` with the kernel's `clock` slot);
- the members' `timing=False` keyword (ragged and uniform `aligned_padded_frame`);
- the `Geometry` protocol docstring;
- tests T6 (the helper's rows, the compiled programs) and T7 (member level); T10.

Not pulled forward (P2): the records, `NewtonInfo.timing`, `_PhaseClock` / `_CompileSeconds`, `_local_model_phases`,
`_padded_start`, `_cone_reference(timing)`, the `inspect.signature` guard that decides whether to pass `timing` to a
member, everything in `newton_cg`. (P3): the display, the frontend keyword, docs pages, CHANGELOG.

**Changed:** `t3toolbox/backend/{common,padded_frames,uniform_padded_frames,geometry,optimizers}.py`,
`tests/backend/test_padded_start.py`, `tests/test_dispatch.py`, `tests/test_jax_absent.py` (round 1). **New:** this file.

## 1. What was built

| where | what |
|---|---|
| `backend/common.py` | `tree_block_until_ready(tree) -> None`, in `__all__` beside `tree_to_jax`: `jax.block_until_ready(tree)` when `jax.tree_util.tree_leaves(tree)` holds a jax array, else nothing (numpy, no jax installed). Documented eager-only. |
| `backend/padded_frames.py` | `_aligning_pass(..., clock=None)`: with a list, the pass's seconds from its entry to a `tree_block_until_ready((frame, movement))` at its end are appended; without one, the old call (no sync, no `perf_counter`). The body of `fv_aligned_padded_frame` moved verbatim into `_fv_aligned_padded_frame(<the same arguments>, timing=False) -> (frame, movements, pass_seconds)`; the public function keeps its signature and docstring and returns `_fv_aligned_padded_frame(..., timing=False)[:2]`. With `timing`, the start frame is synchronized once before pass 1 and every `_aligning_pass` gets the clock; the early returns (zero passes, nothing to complete) give `()`; `timing=False` gives `None` (sliced off by the public loop). |
| `backend/uniform_padded_frames.py` | The pass statics gain a last slot `clock` (the truncating pass's `statics[:5] + (target,) + statics[6:]` and the `at(update)` lambda carry it). `_aligning_pass` unpacks it and clocks as the ragged one does, syncing `(F[:4], movement)`. `_ufv_aligned_loop_kernel(..., use_jit, clock=None)` raises `ValueError` when `clock is not None and use_jit` (first statement), syncs `(start cores, the NaN movement record)` before pass 1, and its partial-binding note now names the slot and why it is eager-only. `ufv_aligned_padded_frame`'s body moved into `_ufv_aligned_padded_frame(..., timing=False)`; the clock is `[]` only for `timing and not jitted`; the jitted call is today's (six positional arguments, no clock) so its program and cache key are unchanged; third element `None` under `use_jit`, `()` on the early returns. |
| `backend/geometry.py` | Both `aligned_padded_frame` members: `timing: bool = False` after `use_jit`. Off: today's call to the public loop (the same keywords, now gathered in one `options` dict). On: the private twin with `timing=True`, three elements. Return annotations are `Union[(frame, movements), (frame, movements, seconds)]`. |
| `backend/optimizers.py` | The `Geometry` protocol docstring: the member's optional `timing=False` keyword, its third element (`None` when it cannot read the passes, as under the uniform whole-loop jit), and the call rule: the optimizers pass it only to a member whose signature names it (`**kwargs` does not count). |
| `tests/backend/test_padded_start.py` | `TestMemberTiming` (T7, numpy): two elements off (keyword absent or `False`), three on, frame and movements bitwise; one positive float per pass over four cases that cover every pass kind (oversampled, truncating, pairs, the odd remainder's final `('D', 'L')`, simultaneous, an early stop, the zero tensor); `()` on zero passes (members) and on nothing-to-complete (the private twins); the start frame is the first synchronization (identity, via a mocked `tree_block_until_ready` and a recording start constructor), passes + 1 synchronizations, pass k's sync holds pass k's movement (ragged), none when off; the public loops and `padded_start` have no `timing` parameter and return two; the kernel's `ValueError`. Module docstring extended. |
| `tests/test_dispatch.py` | `TestDispatch.test_tree_block_until_ready_dispatch` (T6 helper rows): waits exactly on trees with a jax leaf (jax, mixed with numpy / scalars / None / a string, a jax frontend `T3Frame`), not on numpy / empty / `None` / numpy cores; returns `None`; leaves untouched (jax absent: `test_jax_absent.py`, round 1). `TestPaddedStartSeam.test_members_and_padded_start` gains the eager-jax T7 case (a timed member call at the untimed call's arguments: bitwise equal, jax leaves, one float per pass). New `TestAlignmentTiming.test_timing_adds_no_program` (T6 programs + T7 `use_jit`): off then on at one structure, the ragged jitted step's and the uniform whole-loop program's `_cache_size()` unchanged, results equal, uniform third element `None`, ragged one float per pass. |

## 2. Judgement calls

1. **The ragged clock is a slot of `_aligning_pass` too**, not timing code at the loop's four call sites. It mirrors
   W-6's uniform design (one place every pass runs, an optional host list), keeps the loop body's diff to one argument
   per call, and makes "each pass is [synced start, synced end]" one statement in one function on both layers. The
   ragged pass function is module-private and not traced (the jitted object is the step inside it), so no cache-key
   or trace question arises.
2. **The private twins keep the public signature, defaults included,** with `timing: bool = False` appended. The plan
   writes `(..., timing: bool)`; after defaulted arguments a required one would have to be keyword-only, and the
   members call the twin with exactly the keywords they pass the public loop today. Their argument comments are not
   duplicated (the first line says "the arguments of fv_aligned_padded_frame, then timing"): a second unverified copy
   of thirteen contracts is the drift the signature guide warns about. The return block is fully commented.
3. **`timing=False` on a twin returns `None` as the third element**, `timing=True` with no pass returns `()`. So `None`
   always means "not measured" and `()` "measured, zero passes", the same reading as the plan's uniform-jit `None`.
4. **The uniform start-frame sync includes the NaN movement record** (`(frame[:4], movements)`), built between the start
   frame and pass 1, so nothing dispatched before pass 1 can run inside its interval. The `_record` write between
   passes stays unsynchronized (tiny; it lands in the next pass's interval, the plan's §2.4 caveat).
5. **The `use_jit` + `clock` guard is the kernel's first statement**, before `rebuild_static`; the test calls it with
   placeholder arguments to show it raises before reading anything.
6. **Member return annotations:** `typ.Union[2-tuple, 3-tuple]`, following `tt_left_orthogonalize`'s
   `return_variation_cores` precedent, rather than the plan sketch's bare `typ.Tuple` (the signature guide: annotate
   what Python can express). The two-tuple keeps today's per-element comments; the timed alternative is one line.
7. **The member docstrings do not mention `newton_cg(timing=True)`**, which does not exist until P2; the protocol
   docstring states the call rule as "the optimizers pass `timing` only to a member whose signature names it", which
   is true at P1 (none passes it) and is the rule P2 implements. P2/P3 may add the cross-reference.
8. **Test placement.** The numpy member cases are in `test_padded_start.py` (that file is numpy-only by its docstring);
   every jax case (eager jax and `use_jit`) is in `test_dispatch.py`, where x64 is on, as T14 (i) was. The eager-jax
   case is folded into the existing `TestPaddedStartSeam.test_members_and_padded_start` as one extra call at the same
   structure, because a separate eager-jax test cost 80 s (eager uniform jax compiles per operation, and every class in
   that file clears jax's caches after each test); folded, it compiles nothing new. That test gained a docstring
   (it had none) naming both checks.
9. **`test_padded_start.py` imports `test_uniform_padded_frame` as a module** for `loop_case` (the precedent:
   `test_uniform_padded_frame` imports `test_padded_frame` the same way; a module binding is not re-collected).
10. **No `docs/` change.** §6's `aligned_padded_frame.md` paragraph and the CHANGELOG are P3 by §7's slice table;
    `scan_body_principles.md`'s note on the partial binding is left as it is (the plan extends only the kernel's
    docstring). Recommended for P3: one clause there naming the eager-only clock slot, so the exception stays written
    down in both places that describe it.

## 3. Where the plan could not be followed literally

Nothing blocked. Two readings, both recorded above: the twins' `timing` has a default (call 2), and "the early returns
give `()`" is applied to the ragged twin too (the plan states it in the uniform bullet; the ragged loop has the same
two early returns). The T7 rows that need `newton_cg(timing=True)` (row 0's `alignment_passes` / `reference_passes`;
the fixed-signature and `**kwargs` custom members with `alignment_passes is None`) are P2's: the guard they test is in
`newton_cg` / `_padded_start`, which P1 does not touch. T6's `_jitted(_cg_core)` row is likewise P2's.

## 4. Verification

One job at a time, current env `t3toolbox` (py3.11, numpy 2.4.6, jax 0.10.2) unless stated; logs and scripts in the
session scratchpad `p1/`.

| check | result |
|---|---|
| **Non-breaking, bitwise against HEAD** (`7a045cb3`, a temporary worktree, now removed): `p1/battery.py` run in both trees, `p1/compare.py` (dtype, shape and `array_equal(equal_nan)` on every leaf; the history's `wall_time` dropped). numpy: the ragged and uniform public loops on all seven `TestLoopAgainstRagged` cases plus each at `max_passes=0`; both members (old point and zero tensor), `padded_start`, `newton_cg` (`padded_from` + `'cone'`; exact + trust region + `padded_from`; plain) and `gradient_descent(g0_reference='cone')`, each unshared and shared (`(0, 1, 1, 0)`), both layers. jax (x64): the ragged loop eager and `use_jit`, the uniform loop `use_jit` and eager, `newton_cg(use_jit=True, padded_from, 'cone')` on both layers | **numpy 56 entries (548 arrays, 956 floats), 0 mismatches; jax 6 entries, 0 mismatches.** A planted one-ulp change is caught |
| new tests | `TestMemberTiming` 5 passed (18 subtests, 3.4 s); `test_tree_block_until_ready_dispatch` and `TestAlignmentTiming` pass (16.8 s); `TestPaddedStartSeam.test_members_and_padded_start` 53.0 s -> 54.1 s with the eager-jax timing case |
| full suite, `pytest tests/ -q -n 3` | **1,354 passed, 4 skipped**, 48,751 subtests, 15 min 44 s (R6's 1,347 + the 7 new tests) |
| module doctests, `pytest --doctest-modules t3toolbox -n 3` | **241 passed** |
| `python -m sphinx -W --keep-going -b html docs <scratch>` | **exit 0, no warning**; `backend/common/tree_block_until_ready` has its page; the private twins stay out of the reference |
| floor env `tttt` (py3.9.19, numpy 1.22.4, jax 0.4.30), single process: `test_padded_start`, `test_padded_frame`, `test_uniform_padded_frame`, `test_dispatch` | **200 passed, 1 skipped** (21 min 57 s); the skip is the pre-existing `test_eager_jax_compiles_nothing_twice` (no `unregister_event_duration_listener` on 0.4.30). The new `use_jit` cache-size checks run there too (0.4.30 has `_cache_size`) |

Two lines of `test_padded_start.py` were re-wrapped (a comment, a dict literal) while the suite ran; that file was rerun
afterwards in the current env (29 passed) and the floor run used it. No `docs/*.md` page changed, so the page doctests
were not rerun.

## 5. Round 1 (review `P1_review.md`: REVISE, three MINOR test fixes and one NIT; no library code path changed)

| finding | change |
|---|---|
| **M1** `test_timing_adds_no_program` could pass vacuously | `self.skipTest("this jax has no jit _cache_size()")` when either program lacks `_cache_size` (the `test_compile_count` precedent); per layer, `_release_jax_programs()` first (checked: `jax.clear_caches()` empties a wrapper's `_cache_size()` on 0.10.2 and 0.4.30), then `assertGreater(size, 0, 'the untimed run did not use this program')` after the untimed call, so the check is strict even when another test in the worker compiled that structure without clearing; the cache-size equality is now unconditional |
| **M2** the simulated jax-absent row could not fail | dropped from `test_dispatch.py` (its docstring points to the new home); `tests/test_jax_absent.py`'s jax-hidden subprocess asserts `common.tree_block_until_ready(...) is None` on a mixed tree `(np.ones(2), 3, None, 'D')` and on a point's cores; its module docstring names the row. Checked on a scratch copy: deleting the `jax_available and` guard now fails with `NameError: name 'jax' is not defined` |
| **M3** the uniform branch checked only shapes | `test_the_start_frame_is_synchronized_first` also checks, for every pass k, `np.array_equal(np.asarray(trees[k + 1][1]), record[k])` (by value: the record is a `where` copy); docstring says so. Checked on a scratch copy: the mutation "synchronize the input cores before the step, nothing after" now fails the uniform subtest |
| **N1** | the uniform `_aligning_pass` docstring: "One write is not synchronized: the movement record's (`_record`, one `where` between passes), which on jax may land in the next pass's interval." |

Rerun (single process, the reviewer's request; the reviewer is running sphinx and the floor env): `TestMemberTiming`,
`test_tree_block_until_ready_dispatch`, `TestAlignmentTiming`, `test_jax_absent.py`: **8 passed, 27 subtests**
(29 s). No full suite, sphinx or floor run this round. The plan-text items the review lists (a draft-4 CHANGES row for
P1's readings; the `scan_body_principles.md` clause added to P3's §6 list) are the session's.
