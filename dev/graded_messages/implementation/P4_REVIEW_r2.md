SIGNED OFF

# P4, review round 2

Reviewer, 2026-11-17, on the working tree after `P4_RESPONSE_r1.md` (library files last changed 14:42:12, before
the gate in `/tmp/gate_P4_r1` started at 14:43:02). No blocking or should-fix finding remains. Three minor notes,
none of which needs another round.

## The nine findings of round 1

| # | Round 1 | Now |
| --- | --- | --- |
| 1 | A norm that overflows read 0.0 | Fixed. My round-1 script: float32 at norm `1e20` and `1e25`, and float64 at `1e160`, read `2e-7` to `3e-7` (float64 `3e-15`) when consistent, `9.99e-4` with `R[0]` off by `1e-3`, `0.333` with it off by a half; the centers likewise. Consistent frames at norm `1e170` and `1e200` read `3e-15` where round 0 gave `nan` |
| 2 | The `1` of `max(1, .)` unpinned | Fixed. The floor `1e-3` now fails 18 cases |
| 3 | The guard's test hung | Fixed. Zeroing removed, and `all` to `any`: each fails in two seconds |
| 4 | Cost | Fixed, and cheaper: see below |
| 5 | `nan` or `inf` in uniform padding | Fixed in the code. A `nan` or an `inf` in a padded slot leaves the residual as it was |
| 6 | The `LinAlgError` branch | Fixed. One stack element at `1e200` gives `nan` there and `1.4e-15`, `1.6e-15` for the others, on numpy; the same pattern on jax |
| 7 | Scalar promotion under numpy 1.x | No Python float meets an array any more; an unstacked float32 frame is tested. Still not run on numpy 1.x (none here); CI's floor leg will |
| 8 | Wall-clock assertions | Removed |
| 9 | Duplicated helpers | Each names its twin |

## The new route (shared `D`), verified again with my own code

It is still the residual the variables page and the plan define. `D[i] = R[i]^T Q[i]` with `Q[i]` of orthonormal
rows, so a T3 over `D` is a TT over `Q` and its norm is the TT's; at a site with a Tucker center, `Vc[i]` and
`D[i]` are factored stacked, so `T_V[i]` and the base point stand on one basis there, and the denominator is still
the norm of `T(D, L)`.

- **Against my dense einsum:** 216 hand-made cases (non-orthogonal stores, tails, stacks, unequal ranks, Tucker
  rank above the mode size, each family alone): worst relative difference `5.2e-15`.
- **Accuracy over the norm** (20 seeds, `d = 5`): frame at most `1.6e-14`, centers at most `1.9e-14`, from norm
  `4e-6` to `4e14`. The literal route still reads `1.2e-9` at `4e6` and `0.35` at `4e14`.
- **float32:** numpy `2e-7` to `4e-7`; jax, eager and under `jit`, `4e-7` to `1.6e-6`; norms `1e-30` to `1e30`,
  with and without tails.
- **The trap the implementer names:** `D[2]` scaled by `1e-100`, `1e-8`, `1e8`, `1e100` with the cores above
  it scaled back, at tensor norms `1e-5`, `1`, `1e7`, with and without tails: frame at most `6.2e-15`, centers at
  most `1.4e-14`; a `Vc` off by `1e-6` reads `1.0000e-6` at that site and at its neighbour.
- **Rank-deficient cores:** `9e-16` and `2.4e-15`. **Neighbouring TT cores scaled by `1e150` and `1e-150`:**
  `1.1e-14`. **A stack with norms `1e-12` to `1e12`, a zero element and one inconsistent element:** each element
  right; the zero element reads `0` (frame) and `7e-16` (centers, absolute, where the dense value is 0).
- **The uniform twins:** 792 comparisons with the ragged functions on the real cores and with my dense
  definition, prefix and gappy masks, ranks drawn per stack element, tails: `8.5e-15`. Garbage against clean
  padding bit for bit; a mask widened by one slot changes the answer; a traced mask is refused.
- **The guard:** `nan`, `inf`, `-inf` in every store and both centers at every site, numpy, eager jax and `jit`:
  54 cases, none unexpected.
- **Cost** (`d = 8`, `N = 2000`, `n = 60`, `r = 40`): frame residual 0.34 s (round 0: 0.97 s) where building the
  frame takes 0.14 s; centers 4.5 s (round 0: 13.6 s).

## Minor notes (no change required for sign-off)

**1. The balancing of `D[i]` is not pinned by a test.** With `scale_D` replaced by ones, every test passes. It is
a real mutant: on a frame whose `D[2]` is scaled by `1e8` the centers of a consistent frame then read `1.3e-8`,
and by `1e100`, `1e84`. The library's own frames have `D` at unit scale, so only a hand-made frame meets it. One
case with a scaled `D[i]` in `TestRaggedCentersResidual` would pin it.

**2. A zero stack element reads `7e-16` in the centers' residual, not `0`.** The stacked factor of a zero `Vc[i]`
and a unit `D[i]` leaves rounding in the `Vc` block. Absolute and far below any threshold; the frame residual
reads exactly 0.

**3. The response's count for finding 2 differs from what I saw** (it says the floor `1e-3` fails 21 cases; my run
shows 18 failed). The mutation is caught either way.

## Mutations (36 on the round-2 code, in a scratch copy)

Caught: the norm by a plain sum of squares; `nan` not set; the floor; no balancing; balancing of `D` only; either
scale not given back; the base point not refactored at the site; a wrong factor under `Hc`; `R` or `L` on the
wrong side of a center; padding zeroed by multiplication, or kept; every error caught (both levels); all `nan`
on failure; the last `where` removed (both checkers); no guard for a zero core; the denominator from the other
store; either tail; the pre-orthogonalization removed; the singular values dropped; the first site in place of
the maximum; sum in place of difference; the guard broken two ways (as failures, not hangs); `np` on a tracer
in three places.

Not caught: note 1. One more was a no-op of my own making.

## What I ran

- My round-1 scripts again, and one new (`n_scale.py`), all in `/tmp/p4rev`.
- `run_gate.sh -n 2 -l /tmp/gate_P4_review_r2 quick` on the new file, the two frame test files,
  `test_import_graph.py`, `test_jax_absent.py`, `tests/backend/test_fv_operations.py` and
  `test_dispatch.py::TestDispatch::test_jit_backend`: 167 passed, 619 subtests. The uniform dispatch test
  (`test_jit_uniform_frame_helpers`) passed in the mutation baseline.
- Module doctests of the four touched modules: 35 passed.
- The implementer's gate (`/tmp/gate_P4_r1`, all steps PASS, 1,631 passed, 5 skipped, 1 xfailed) started after
  the last change to any library or test file; I did not rerun it.
- Nothing under `t3toolbox/backend/` imports `safety`; the frame classes changed only in docstrings since round 1.
- The tree is as I found it apart from this file; the scratch copy is removed; nothing is running.
