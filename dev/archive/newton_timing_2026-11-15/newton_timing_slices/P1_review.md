# P1 review: the synchronization helper, the loops' timed twins, the members' `timing` (Newton timing, 2026.5.0)

Reviewer's record. Base HEAD `7a045cb3`; P1 in the working tree, uncommitted. Read: `CLAUDE.md`; the HANDOFF top block;
the plan (draft 3) and its reviews r1, r2; `P1_handoff.md`; the switch plan §3; `signature_style.md`,
`scan_body_principles.md` (the partial-binding note), `parameters_not_closures.md`; `TuckerTensorTrain.resize` and its
backend, `tt_left_orthogonalize` (the `Union` return precedent); the whole diff and the code around it (both loops,
both kernels' bodies, both members, the protocol docstring).

## Round 1

### What I ran (my own, one job at a time, current env py3.11 / numpy 2.4.6 / jax 0.10.2 unless stated)

| check | result |
|---|---|
| **Bitwise vs HEAD**, my own battery (scratchpad `p1_review/battery.py`, `compare.py`) on a worktree of `7a045cb3` and on the working tree. It keeps container types (tuple vs list), dict key order, dtypes and float bits (`tobytes`, so `-0.0` and NaN payloads count), and the `make_newton_display` text with only the `%.2fs` wall-clock cell masked. numpy: the public ragged and uniform loops on all seven `TestLoopAgainstRagged` cases, each as given, `alternate=False`, `max_passes=0`, `tol=0, max_passes=7`, plus the ragged nothing-to-complete call; the two frontend twins; both members (default, `tol=0` 6 passes, the zero tensor, `max_passes=0`) unshared and shared `(0, 1, 1, 0)` on both layers; `padded_start`; `newton_cg` (`padded_from` + `'cone'`; exact + trust region + `padded_from`; plain), each with its display records and text; `gradient_descent(g0_reference='cone')`. jax x64: both loops eager and `use_jit`, alternate on and off; both members under `use_jit`; `newton_cg(use_jit=True, padded_from, 'cone')` on both layers | **numpy 124 entries, 1,115 arrays, 0 mismatches; jax 12 entries, 141 arrays, 0 mismatches.** A planted one-ulp change in one uniform member output is caught |
| The T7 member cases really cover every pass kind | working ranks `((5, 6, 5, 6), (1, 5, 9, 6, 1))` differ from the target, so `n_over = 2`: 'two-colour, final pass' runs 2 oversampled + truncating + one pair + the final `('D', 'L')` = 6 passes, 'simultaneous' 4, the zero tensor 5; one float per pass, 14-55 ms each, both layers |
| `tree_block_until_ready` on both envs (jax 0.10.2 and the floor's 0.4.30) | mixed `(jax, numpy, int, None, str)`, `()`, `None`, a bare `str`, a dict and a list: no error. Called on a tracer inside `jit` and inside `grad`: no error, nothing waited (both versions skip tracers), so the docstring's "a traced leaf has nothing to wait for" is accurate |
| full suite, `pytest tests/ -q -n 3` | **1,354 passed, 4 skipped**, 48,751 subtests (17 min 37 s); the one warning is the pre-existing adam-on-manifold one |
| module doctests, `pytest --doctest-modules t3toolbox -n 3` | **241 passed** |
| **Mutations** (scratchpad `p1_review/mutate.py`, on a copy of the tree, one at a time, the targeted tests rerun) | caught: no start sync (ragged; uniform); the ragged pass synchronizing its input before the step; the clock appended twice; the public loop returning three; the kernel's raise removed. **Missed: the uniform pass synchronizing its INPUT cores before the step instead of its output after it (M3); the helper without its `jax_available` guard (M2)** |

### Plan fidelity

Every §7 P1 row is built as specified, and nothing from P2/P3 is pulled forward: no record, no `NewtonInfo` change,
no `_padded_start` / `_cone_reference` change, no `newton_cg` keyword, no `inspect.signature` guard, no docs page, no
CHANGELOG. `optimizers.py` changes only the protocol docstring. Nothing is renamed; the members' `options = dict(...)`
is a call refactor that passes today's keywords with today's values (the battery confirms bitwise).

- `common.tree_block_until_ready`: in `__all__` beside `tree_to_jax`, the §3.3 signature, the jax-absent guard.
- Ragged: the start frame synchronized before pass 1; each pass [synced start, synced end]; the early returns `()`.
- Uniform: `clock` the last statics slot, carried by the truncating pass's splice (`statics[6:]`) and by `at`; every
  pass runs through `_aligning_pass` (`_oversampled_body`, the truncating pass, both passes of `_pairs_body`,
  `_final_pass`, `_single_body`); the kernel's `ValueError` is its first statement; the start cores and the NaN record
  synchronized before pass 1; the jitted call is today's six positional arguments; the partial-binding note names the
  clock slot and why it is eager only (r1 M3 (a)-(c) all met).
- Members: `timing=False` after `use_jit`; off = today's call to the public loop; on = the private twin.
- Protocol docstring: lists the keyword, says a member need not have it, and states the call rule exactly as W-3
  narrowed it ("only to a member whose signature names it (a `**kwargs` catch-all does not count)"). This is the
  documented rule P2's guard must implement; it matches §2.6.

**The judgement calls (the brief's five, and the rest of the handoff's ten).** All sound; none needs to go back.

1. *The same clock slot in both loops* (call 1). Sound: one statement of "[synced start, synced end]" per layer, in the
   one function every pass runs through; the ragged pass function is a host function, never traced (the jitted object
   is the step inside it), so no cache-key question arises.
2. *The twins keep the public defaults with `timing=False` appended* (call 2). Sound. The plan's `timing: bool`
   (required) after defaulted arguments would have to be keyword-only; with a default, `None` means "not measured",
   which is also what the uniform jit returns. Not repeating the thirteen argument comments on the twins follows the
   signature guide's own "second, unverified copy" warning; the return blocks are fully commented.
3. *`Union` return annotations on the members* (call 6). Sound: "annotate what Python can express", with the
   `tt_left_orthogonalize` precedent for a flag-dependent arity. The timed alternative is one line rather than one
   element per line; expanding it would repeat the two commented elements above it, so I accept the compression.
4. *No `newton_cg(timing=True)` reference yet* (call 7). Sound: the protocol docstring's rule is true at P1 (no
   optimizer passes `timing`) and is the rule P2 implements. P3 adds the cross-reference.
5. *The eager-jax case folded into the existing seam test* (call 8). Sound: it reuses the seam test's compiled
   programs (a separate test cost 80 s because every class there clears jax's caches), and it checks what T7 asks
   (bitwise equal to the untimed call, jax leaves, one positive float per pass).

Calls 3 (`None` vs `()`), 4 (the NaN record in the start sync), 5 (the raise first), 9 (the module import) and 10 (no
docs) are sound too. **Plan text (the session, not a code change):** record in a draft-4 CHANGES row the readings P1
made: the twins' `timing: bool = False`; `()` from the ragged twin's early returns too; the members' `Union`
annotation (§3.3 sketches `-> typ.Tuple`); the ragged `_aligning_pass` carries the clock as well. And add to P3's §6
list the one clause in `docs/contributor/scan_body_principles.md` that the handoff recommends (its partial-binding
paragraph should name the eager-only clock slot, so the exception is written down in both places that describe it).

### Non-breaking

Confirmed by my own battery (above): with `timing=False` every public loop, frontend twin, member, `padded_start`,
`newton_cg` (stats, history rows, display records and text) and `gradient_descent` output is bitwise HEAD's, container
types and key order included; the public loops still return 2-tuples (also when the early returns fire); the members
return two elements with the keyword absent or `False`. The jitted uniform program is the untimed one: the jitted call
never receives `clock` (its default `None` binds inside the Python function, so the trace is today's), and the
`use_jit` battery entries are bitwise HEAD's.

### Correctness of the clocks

- Ragged: the start frame is synchronized, then every pass reads `perf_counter` at entry (after the previous pass's
  synchronized end, or the start's) and appends after `tree_block_until_ready((frame, movement))`. The host work between
  passes (`_peak`'s `np.asarray`, which synchronizes the movements anyway; the list append) is in no pass. Clean.
- Uniform eager: the same, with one exception the code does not state (NIT 1): the movement record's write
  (`_record`, one `where` over `(max_passes,)+C`) is dispatched after a pass's synchronized end and is never waited on,
  so on jax it can land inside the next pass's interval. It is microseconds, and it is the plan's §2.4 caveat, which the
  handoff (call 4) records; the docstrings claim [synchronized start, synchronized end] without it.
- `tree_block_until_ready` waits only when a leaf is a jax array, handles mixed and empty trees on both jax versions,
  and is a no-op on tracers (checked above).
- The clock cannot reach a jitted kernel silently: with `use_jit=True` the kernel raises before reading anything, and a
  list passed through `jax.jit` as a dynamic argument would arrive as `[]` (an empty pytree), which also raises.
- Parameters-not-closures: no geometry field changed; `timing` is a call argument, so no hash/eq or aux changes. Scan
  bodies: no new body; the clock rides the existing, documented partial binding, eager only.

### House style

Signatures shape-annotated and aligned (the twins' blocks, the new `clock` arguments in their own blank-delimited
group in the kernel, the members' `timing` rows); the long trailing comments are within the files' existing practice.
`time` imported in alphabetical order; `__all__` updated; no debugging residue; docstrings say what each piece does and
when it does nothing. Nothing to change.

## Findings

### MINOR

**M1. `TestAlignmentTiming.test_timing_adds_no_program` can pass vacuously** (`tests/test_dispatch.py`). The check is
`size = program._cache_size() if hasattr(program, '_cache_size') else None` and then `if size is not None:
assertEqual(...)`. Two holes: (a) nothing asserts that `program` is the wrapper the member actually called. If a later
change made the member's jitted path use another wrapper (other `static_argnums`, another memo), `program` would stay at
its old size (0 after `_release_jax_programs`), the test would pass, and an extra compile in the timed run would go
unseen; (b) without `_cache_size` the assertion silently disappears. The file's precedent (`test_compile_count`,
~line 2425) does both properly: `self.skipTest("this jax has no jit _cache_size()")` and `assertGreater(counts[0], 0)`.
**Fix:** skip explicitly when `_cache_size` is missing, and after the untimed call
`self.assertGreater(size, 0, 'the untimed run did not use this program')`.

**M2. The jax-absent row of `test_tree_block_until_ready_dispatch` cannot fail.** It sets `common.jax_available =
False` and calls the helper on numpy cores; but a numpy tree never reaches `jax.block_until_ready`, and `jax` is still
importable in the test process, so deleting the `jax_available and` guard leaves the row green. The guard's job is
to keep the helper from touching `jax` when jax is not installed (numpy-only is a supported configuration).
**Fix (recommended):** move the row to the house's real jax-absent test, `tests/test_jax_absent.py`, whose subprocess
hides jax (`sys.modules['jax'] = None`): one call there, e.g. `assert common.tree_block_until_ready((np.ones(2), 3,
None, 'D')) is None`, fails with a `NameError` if the guard goes. Drop the simulated row from `test_dispatch.py` (or,
if you prefer to keep it there too, patch both names with `unittest.mock.patch.object(common, 'jax_available', False)`
and `unittest.mock.patch.object(common, 'jax', None)`, so a missing guard raises on `None.tree_util`; that also
replaces the hand-written save/restore).

**M3. On the uniform layer nothing checks that a pass ends synchronized on its own output.** In
`test_the_start_frame_is_synchronized_first` the uniform branch checks only the shape of each pass's synchronized
tree (`len(t) == 2 and len(t[0]) == 4`). A mutation that synchronizes the pass's *input* cores before the step, and
nothing after it, keeps every test green (the clock then measures dispatch on jax, which is exactly what the
synchronization exists to prevent). The ragged branch catches the same mutation through `t[1] is m`. **Fix:** in the
uniform branch also check that pass k's synchronized movement is pass k's movement, by value against the record:
`np.array_equal(np.asarray(trees[k + 1][1]), np.asarray(out[1])[k])` for each pass k (identity is not available,
since the record is a `where` copy). With that, the mutation fails on shape and a stale movement fails on value.

### NIT

**N1. State the uniform record write's exception where the clock is described.** In the uniform `_aligning_pass`
docstring (or the kernel note), one clause: the movement record's write between passes is not synchronized (one
`where`; on jax it may land in the next pass's interval). Then the code says what the handoff says.

VERDICT (round 1): REVISE (three MINOR test fixes, one NIT docstring clause; no library code-path change needed)

## Round 1, continued: the runs finished after the findings went out

| check | result |
|---|---|
| `python -m sphinx -W --keep-going -b html docs <scratch>` | **exit 0, no warning**; `autoapi/t3toolbox/backend/common/tree_block_until_ready.html` rendered (the build ran on the round-0 tree; round 1 changed only a private docstring and tests) |

## Round 2 (the implementer's `P1_handoff.md` §5)

What changed: tests only, plus N1's docstring clause. The library's only change since my bitwise battery is those two
lines of the uniform `_aligning_pass` docstring (checked by a file-by-file diff of `t3toolbox/` against my round-0
copy), so the battery's 0 mismatches stand.

| finding | resolution | checked by me |
|---|---|---|
| M1 | explicit `skipTest` without `_cache_size` (the `test_compile_count` pattern); per layer the caches are cleared, then `assertGreater(size, 0, 'the untimed run did not use this program')`, then the unconditional equality | new mutation, the uniform jitted call through a fresh `jax.jit` wrapper instead of the memoized program: **caught** (`assertGreater` fails) |
| M2 | the simulated row dropped from `test_dispatch.py`; `tests/test_jax_absent.py`'s jax-hidden subprocess asserts the helper returns `None` on a mixed tree and on a point's cores; its module docstring names the row | guard deleted: **caught**, `NameError: name 'jax' is not defined`; the unmutated control passes in the same environment |
| M3 | the uniform branch checks pass k's synchronized movement against record slot k by value, for every pass | "synchronize the input cores before the step, nothing after": **caught** |
| N1 | the uniform `_aligning_pass` docstring: "One write is not synchronized: the movement record's (`_record`, one `where` between passes), which on jax may land in the next pass's interval." | read: accurate, and it matches the plan's §2.4 caveat |

Mutation re-run on a fresh copy of the round-1 tree (`p1_review/mutate2.py`): **9 of 9 caught** (the eight of round 1
plus M1's).

Round-2 runs (mine, one job at a time, on the round-1 tree):

| check | result |
|---|---|
| touched files, current env, `pytest tests/backend/test_padded_start.py tests/test_dispatch.py tests/test_jax_absent.py -n 3` | **111 passed**, 1,481 subtests (10 min) |
| floor env `tttt` (py3.9, numpy 1.22.4, jax 0.4.30), single process (no xdist there): `test_padded_start`, `test_padded_frame`, `test_uniform_padded_frame`, `test_dispatch`, `test_jax_absent` | **201 passed, 1 skipped** (24 min); the skip is the pre-existing `test_eager_jax_compiles_nothing_twice` (0.4.30 cannot unregister a monitoring listener). `TestAlignmentTiming` runs there, not skipped (0.4.30 has `_cache_size`); the jax-hidden subprocess passes |

Earlier results that still hold: the full suite (1,354 passed) and the module doctests (241) ran on the round-0 tree;
round 1 changed library code only in a private docstring, and the touched test files were rerun above. The session's
pre-commit full suite covers the rest.

**For the session (plan text, not code):** a draft-4 CHANGES row recording P1's readings (the twins'
`timing: bool = False`; `()` from the ragged twin's early returns; the members' `Union` annotation; the ragged
`_aligning_pass` carries the clock too), and the `scan_body_principles.md` clause naming the eager-only clock slot
added to P3's §6 list.

**Files for the P1 commit:** `t3toolbox/backend/{common,geometry,optimizers,padded_frames,uniform_padded_frames}.py`,
`tests/backend/test_padded_start.py`, `tests/test_dispatch.py`, `tests/test_jax_absent.py`,
`dev/newton_timing_slices/P1_handoff.md`, `dev/newton_timing_slices/P1_review.md`.

VERDICT: APPROVE
