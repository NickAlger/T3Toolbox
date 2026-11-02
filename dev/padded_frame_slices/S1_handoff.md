# S1 handoff: pinned primitives, the rank function, HISTORICAL 1-2

Implementer, 2026-11-02. Plan: `dev/padded_frame_cone_impl_plan_2026-11-02.md` §8 row S1 (§4.1, §5.1, §5.4, §6 T1 /
T5(h) / T10 / T12). Nothing committed.

## Files touched

| file | lines | what |
|---|---|---|
| `t3toolbox/backend/linalg.py` | 5-14 (module docstring), 20 (`import contractions`), 35-36 (`__all__`), 830-996 (new section) | `_check_pinned_operands` (private), `pinned_completion`, `pinned_leading_columns` |
| `t3toolbox/backend/ranks.py` | 11 (`import operator`), 22 (`__all__`), 178-250 | `oversampled_ranks`, beside `compute_minimal_ranks` |
| `tests/backend/test_linalg.py` | 4, 22-26 (docstring), 31-37 (imports), 206-371 | T1: `TestPinnedCompletion`, `TestPinnedLeadingColumns` |
| `tests/backend/test_padded_start.py` | new file | T5(h), the rank function's rows: `TestOversampledRanks` |
| `tests/test_dispatch.py` | 146-164 | T10 (primitives): `test_pinned_primitives_dispatch` |
| `tests/test_contractions_interpreter.py` | 354-358 | T12 (partial): `'Cia,Cib->Cab': ''`, `'Cia,Cab->Cib': ''` |

## The new functions

- **`linalg.pinned_completion(pinned, seeds, k)`**: the `k` columns continuing the orthonormal `pinned` (`C+(n, s)`),
  taken from `seeds` in order. It is the sign-fixed reduced Householder QR of `[pinned | seeds[:k]]` (missing seeds are
  zero columns), columns `s..s+k-1`. No threshold anywhere. `k = 0` returns `C+(n, 0)` without factorizing; `s + k > n`,
  a mismatched `C` / `n` or a negative `k` raises `ValueError`.
- **`linalg.pinned_leading_columns(pinned, A, k)`**: the leading `k` left singular directions of
  `A - P(PᵀA)` (two `contract` calls, HISTORICAL 1-2), passed through `pinned_completion`, so rank `< k` is completed
  deterministically. `A` with no columns becomes the completion of zero seeds.
- **`ranks.oversampled_ranks(shape, tucker_ranks, tt_ranks, p, sharing=None)`**: the elementwise maximum of the input
  and `compute_minimal_ranks(shape, n+p, (1, r1+p, ..., 1), sharing)`. It takes integer sequences, returns Python-int
  tuples, is structurally minimal for a minimal input, is never below the input, and keeps one rank per sharing group.

## Numbers reproduced

- Doctests: completion residual `‖[P|Q]ᵀ[P|Q] − I‖ < 1e-14` for random, in-span, all-zero and too-few seeds; the
  planted direction recovered with `|⟨u, U₀⟩| > 1 − 1e-12`; the rank-1 content completed to `< 1e-14`.
- Rank function: `(3,10,4), (2,3,2), (1,2,2,1)` gives `((3,4,3),(1,3,3,1))` at p=1 and `((3,8,4),(1,3,4,1))` at p=5,
  and the input unchanged at p=0. The non-minimal `(5,3,2)` with p=1 gives `((5,4,3),(1,3,3,1))`. Shared
  `(8,3,8), (2,1,2), (1,1,1,1)` with `('a','b','a')` and p=1 gives `((3,2,3),(1,2,2,1))`; unshared it gives `((2,2,2),…)`.
- T1, measured margin: the worst completion residual over 300 random `(n ≤ 12, s, k)` cases × {random, zero,
  in-span} seeds × both primitives is 2.0e-15, against the 1e-14 bound. `s = 0` with the keyed Gaussian equals
  `_haar_rows(...).T` to 1e-14 (four `(n, N, key, seed)` cases). The planted subspace has projector distance `< 1e-12`.
  `C = (2,)` and `(2, 3)` equal per-element calls to 1e-14, a rank-deficient element included. Float32 in gives
  float32 out (residual ≤ 1e-5). Seeds are honoured in Gram–Schmidt order (each column to 1e-14).
- T10: numpy in gives numpy out; jax in gives jax out (eager); `jax.jit(fn, static_argnums=2)` gives pure jax and
  agrees with numpy to 1e-12 (x64); jitted `k = 0` has shape `(2, 7, 0)`.

## Gates

- New / touched modules: `test_contractions_interpreter.py`, `test_contractions_sharding.py`, `backend/test_linalg.py`,
  `backend/test_ranks.py` and `backend/test_padded_start.py`: **81 passed, 5,106 subtests**. The dispatch row passes.
- Doctests: `--doctest-modules t3toolbox/backend/linalg.py t3toolbox/backend/ranks.py`: **15 passed**.
- Full suite `-n auto`: **1,187 passed, 4 skipped, 47,562 subtests, 1 warning** (8 min 40 s).

## Judgment calls

1. **T5(h) placement.** The rows go in the plan's T5 file, `tests/backend/test_padded_start.py`, created now; S6 extends
   it. Two parts of T5(h) are deferred, because the objects they compare against do not exist yet: the frontend
   `TuckerTensorTrain.oversampled_ranks` agreement (S9) and the geometry member agreement (S5).
2. **Boundary bonds in `oversampled_ranks`.** The maximum is taken over all `d+1` TT entries. An input with boundary
   bonds 2 (an unsquashed T3 sum) therefore keeps 2, which is "never below the input" taken literally. The return
   comment reads "boundary bonds as the input's (1 for a squashed T3)" instead of the plan's "boundary bonds stay 1".
3. **Validation in `oversampled_ranks`.** `p` must be a non-negative int; a bool or float raises `ValueError`. Ranks pass
   through `operator.index`, so a float rank is a `TypeError`. The length checks are repeated before the `+p`
   construction so their messages stay clear. Stacked rank arrays are not accepted.
4. **Primitive checks.** `C` and `n` must match exactly; there is no broadcasting, and the caller broadcasts (§4.3's
   `broadcast(Gᵀ)`). `k` must be a non-negative int. The checks are shared through a private `_check_pinned_operands`.
   The output dtype is `result_type(pinned, seeds)`.
5. **The dispatch row uses generic `A` only.** A degenerate seed's Householder completion is determined by roundoff
   inside the reflectors: deterministic for a given backend, but not the same across numpy and jax. So jax == numpy is
   asserted only on generic data. The degenerate cases are numpy-only in T1.
6. **Docstring extras.** The `pinned_completion` docstring adds the Gram–Schmidt-order reading and the caveat that
   `pinned` is assumed orthonormal and not checked. Both are tested or stated; neither changes the §5.1 contract.
7. **Commit trailer.** It names Claude Fable 5.1, as instructed by the brief and HANDOFF (the committing session's
   model). This implementer agent ran as Claude Opus 5.5.

## Draft commit message

```
padded frame S1: pinned completion primitives, ranks.oversampled_ranks, HISTORICAL 1-2

backend/linalg.py: pinned_completion (sign-fixed Householder completion of an
orthonormal block from ordered seeds; any seeds, no threshold; C vectorized) and
pinned_leading_columns (the pinned truncated SVD: leading left singular directions
of (I - PP^T)A, completed). backend/ranks.py: oversampled_ranks, the one home of
the working-rank rule (+p, structurally minimal, never below the input, sharing
aware). Tests: T1 (test_linalg), T5(h) rank rows (new test_padded_start), T10
primitive dispatch rows, T12 HISTORICAL 'Cia,Cib->Cab', 'Cia,Cab->Cib'.
Additive; no existing behaviour changes.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HmCnX876F5amZcDYUZPZqp
```

## Round 2 (after `S1_review.md`, REVISE: two MINORs, NITs)

- **MINOR 1.** The `pinned_leading_columns` docstring is reworded. Columns beyond the projected content's rank come
  from the SVD's trailing singular vectors, re-orthogonalized by the QR. They are always a valid orthonormal
  completion ⊥ `P`, deterministic for a given input and backend, but `A` does not determine them: roundoff-level
  changes to `A` (e.g. content inside `span(pinned)`) can rotate them by O(1), and backends may differ there.
  **Note for S3-S5:** gauge-invariant or cross-layer projector-equality tests (T3(h)/(j), T4(e), T5(f)) must use hints
  with at least `k` directions per slot. Rank-deficient hints may only be asserted *valid* (as in T3(a)).
- **MINOR 2.** A jax validity subtest is added to `test_pinned_primitives_dispatch`. Both primitives run under jit
  with x64 on all-zero, too-few, in-span and no seeds (`C = (2,)`). It asserts a completion residual ≤ 1e-14 and makes
  no comparison with numpy.
- **NITs.** "the first seed's" now reads "seed ``j``'s". The `oversampled_ranks` docstring gains a sentence: boundary
  bonds above 1 are kept, and the interior caps are computed as for boundary bonds 1, which is conservative. The draft
  commit message is unchanged (the trailer is settled by the orchestrator).
- **Gates re-run.** `backend/test_linalg.py`, `backend/test_padded_start.py` and `backend/test_ranks.py`: 63 passed,
  2,024 subtests. `test_dispatch.py -k pinned`: passed (subtests included). Doctests of `linalg.py` and `ranks.py`:
  15 passed. The full suite was not re-run (per the coordinator); the changes since round 1 are a docstring and one
  added test.
