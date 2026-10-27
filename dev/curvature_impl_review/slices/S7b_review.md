# S7b review (I-38: the safe-mode gauge check scaled by the operands). Verdict: APPROVE-WITH-FIXES (fixes applied, uncommitted)
Base main @ 0fd3a83; implementer tree uncommitted (13 files, unchanged count). Scripts: review/s7b/
(check_ruling.py, check_trace.py, bitident.py, timing*.py, prof.py, mutants.py + mutants_out.txt, gate_*.txt). No blocking finding. No code defect in the library.
Every pre-existing test is unchanged (`git diff tests/` has no `-` lines). sampling_derivatives.py, test_chunk_size_estimator.py, dev/, CLAUDE.md untouched.

## Does it implement the ruling? Yes. (check_ruling.py)
(i) S0b-style cancellation (agree 1e-8, 1e-15 noise): self-rel 2.7e-8, operand-scaled 3.2e-16, `M.norm` passes. (ii) tiny ungauged leaf (1e-12 raw): 0.23, raises.
(iii) 1e-6*max(|a|,|b|) violation injected into a difference, `(a-b)+v` or `(a+v)-b`: 2.8e-7, raises; 1e-8*s still raises (2.4e-9), 1e-9*s passes (3.1e-10).
(iv) non-cancelling `a+b`: residual bitwise = the self-relative value. Recursion is REQUIRED: without it `|c|*||a||` = `||c a||` is a no-op, and the
non-recursive `max(||H_fd||, ||Hp||)` fails the S0b repro at h <= 1e-6 (handoff: gauge(H_fd) self-rel 6.5e-9). x06 spells `(gp-gm)*(1/(2t))`, the docs `/ (2*h)`: both keep the seed.

## Judgment calls
1. FLOOR `max(own, scale)`: AGREE, no escalation. One step: `||a+-b|| <= 2 max(||a||,||b||)`, so own > scale only for a constructive sum, by <= 2x, and there
   the floor IS the pre-S7b check. Every cancellation (own << operands) is identical to the literal rule, so (ii)/(iii) hold. The literal replacement would be
   STRICTER than pre-S7b on constructive chains, by up to the number of summands: `(a+a)+(a+a)` gives literal 1.0e-15 vs floor 2.6e-16 (4x), i.e. new false
   failures the ruling never asked for. "Never stricter than self-relative" is the property Nick wants: S7b removes false failures; it must not add any.
2. Record only with `checks_active`: AGREE. Verified nothing recorded under jit (traced operands; closed-over recorded tangent times a Python scalar, caught by
   the probe; stack_tangents), grad (traced scalar; traced variations) or unsafe(). No `float()`/host sync on the record path; eager jax gives a jax seed.
   GAP: mutant M3 (gate on safe mode only, ignore the trace) SURVIVED the implementer's suite (incl. test_dispatch). FIXED with a new test (below).
3. A scaled leaf stays a leaf: AGREE (identical under the floor; saves a norm per scalar multiply).
4. Scope: AGREE. Ragged `-t` and `t / s` route through `*` and propagate (M5/M6 killed). normalized / sum_tangents / unstack_tangents give leaves:
   conservative, follow-up. Pre-existing gap: `UT3Tangent` has no `__truediv__` (ragged has one), so a uniform FD must spell `* (1/(2h))`.
5. Non-scalar `c` records nothing: AGREE. 6. Extra docs: AGREE, with the fixes below.
Seed mechanics: `vars(out)['_gauge_scale'] = ...` writes the same `__dict__` slot cached_property fills (frozen dataclass OK, no lock on py3.13, fresh
`out`). copy.copy/deepcopy keep it (same values, valid); `.copy()`, `dc.replace`, to_jax/to_numpy and a pytree round trip drop it (leaf, conservative).
No new field/leaf/aux; `==`/hash raise (tested). The seed is one K+C array per result, freed with the object; computed EAGERLY (no operand references kept).

## Findings
F1 MEDIUM (inherent to the ruling; doc overclaim FIXED; framing for Nick). Recursion through `/ (2h)` makes `H_fd - Hp` lenient: its scale is
   max||g(+-h)||/(2h), 1.6e6*||Hp|| at h=1e-5 on the test_fitting problem. A 1e-4-relative gauge defect injected into Hp passes `M.norm(H_fd - Hp_bad)` at
   h <= 1e-5 (1e-6 passes even at h=1e-3), while `M.norm(Hp_bad)` raises (check_ruling.py (vi)). This is the honest roundoff (H_fd's noise is eps*||g||/h), but
   tol 1e-9 >> eps widens it by the same 1/h. The FD comparison itself still exposes such a defect (||D||/||Hp|| >= the defect). The docs said "a tangent
   ungauged in its own right is still caught"; now they say "when checked itself" and that the yardstick scales with `/ (2h)`.
   For Nick: "Under I-38 the gauge check of an FD difference certifies the gauge only to 1e-9 * ||grad||/h, the FD's own noise floor. To certify hessian(p)'s
   gauge, check hessian(p) itself. Accept as documented (recommended), or cap the recorded scale's growth under scalar multiplication?" Rec.: accept.
F2 MINOR (perf, safe mode only; DOCUMENTED). Each safe-mode `+`/`-` computes two `corewise_stack_norm`s eagerly. Leaf add: 55->314 us at 3.2k dof (~6x),
   175->663 us at 113k dof (~4x). Recorded*scalar and unsafe() unchanged. One norm costs 134 us at 3.2k dof, so corewise_stack_norm is itself slow; a faster
   norm would cut most of this. The cost is acceptable: optimizer loops run on backend `.data`, and `LocalModel.hessian`/regularized `gn_hessian` pay 2 norms
   per call next to a matvec. Nit: `T3Tangent.allclose` now does 4 norms (its `self - other` records).
F3 MINOR (test maintenance; FIXED). The bitwise "frozen b6ad78b body" tests pinned the float summation order, so a harmless refactor would fail them. They
   are now reference-oracle tests (allclose rtol 1e-13), plus bitwise `None == default == leaf` (one code path). The one-time bitwise check was rerun out of
   tree (bitident.py, separate processes vs `git archive 0fd3a83`): 342 arrays (tv + leaf + utv/uleaf, 90 uniform with forced padding) differ in 0.
F4 MINOR (FIXED). The new uniform tests sat after `if __name__ == '__main__'`; the guard moved to the end. test_manifold.py has the same pattern from earlier
   slices; left as is.
F5 NIT. `scale: typ.Optional[NDArray]` also takes floats (`typ.Union[float, NDArray, None]`). Negative scale is harmless (floor). NaN gives a raise.

## Fixes applied (uncommitted)
tests/test_manifold.py:1238 `test_not_recorded_under_a_trace` (kills M3) | tests/backend/test_tv_operations.py:140,175 + tests/test_uniform_manifold.py:1322,1384
(F3) | tests/test_uniform_manifold.py:1505 main guard (F4) | docs/numerical_contracts.md:75-80,115-117 (F1 wording, F2 cost) | docs/contributor/
precompute_and_caching.md:52-55 (cost) | CHANGELOG.md:131-133 ("recursively", "never stricter", "computed directly", "when checked itself") | t3toolbox/manifold.py:604-606 (F1 docstring).

## Mutants (mine; tests = the 4 S7b classes, + test_dispatch for M3)
KILLED: M1 tv floor->replacement, M1b utv floor->replacement, M2 `__sub__` drops seed, M4 uniform seed ignores masks (garbage-padding test), M5 ragged
`__neg__` drops seed, M6 `__truediv__` drops seed. SURVIVED then KILLED: M3 record under a trace (after the new test). The floor-vs-replacement
difference is caught only by the backend tests (scale=0 and 0.5*own must be bitwise the default), and one level suffices.

## Plan / doc amendments for S13
Plan S7b row: record the floor (`max(own, operand scale)`), recursion through `*`/`/`, scaled leaf = leaf, scope (+ - * / neg stack; normalized / sum / unstack
give leaves), the F1 leniency and the F2 cost. verification_records: S0b repro before/after (handoff numbers) + bitident 342/0.
Follow-ups (DEFERRED): seed propagation for normalized/sum_tangents/unstack_tangents; `UT3Tangent.__truediv__` parity; a faster `corewise_stack_norm`.

## Gates (after fixes)
All green. Targeted, 6 files, -n 2: 301 passed / 1544 sub (pre-fix 300). test_dispatch serial: 49 / 36 sub. Module doctests: 213. Doc pages and quickstart: OK.
sphinx -W: exit 0. FLOOR LEG (py 3.11.17 / numpy 1.26.4 / jax 0.4.30; manifold + uniform_manifold + tv_operations + dispatch): 216 passed / 588 sub,
new trace test included. Full suite -n 2: dispatch + backend 318 / 2096 sub; the rest 745 passed, 1 skipped / 42621 sub (1 warning, the pre-existing
adam-on-manifold one). Slow tier, serial: 5 passed / 12790 sub (3 pre-existing warnings). `git diff --check` is clean. Tree left uncommitted.
