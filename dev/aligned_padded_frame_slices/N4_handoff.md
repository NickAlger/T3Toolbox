# N4 handoff: the uniform start

Slice N4 of `dev/aligned_padded_frame_impl_plan_2026-11-09.md` (§8): `linalg.pad_safe_pinned_completion` (§3.4, §5.2),
`ufv_masking.ufv_frame_widths`, and the new `backend/uniform_padded_frames.py` with `ufv_empty_frame` and
`ufv_padded_frame` (§3.4, §4.5, §5.3). Tests T7, T8, T17 and the T16 rows. Implementer: Claude Opus 5.5, 2026-11-09,
on top of N3 (cccc6f7c). Everything is uncommitted; nothing outside the repository was touched.

## Files

| file | change |
|---|---|
| `t3toolbox/backend/linalg.py` | `+ pad_safe_pinned_completion` (in `__all__`, doctests: no twin). |
| `t3toolbox/backend/ufv_masking.py` | `+ ufv_frame_widths` (in `__all__`, doctest). |
| `t3toolbox/backend/uniform_padded_frames.py` | **new**: `ufv_empty_frame` (doctest), `ufv_padded_frame` (one-line docstring: it gets a twin in N8), the `UFrameData` alias (used in the signature), private host helpers. Imports: `common`, `linalg`, `ranks`, `sharing`, `ufv_masking`, `fv_operations` (`ZERO_FRAME_SEED`), `padded_frames` (`_haar_seed_rows`, which §4.2 says is shared with the uniform start). T18's allowed set holds. |
| `tests/backend/_uniform_padded_frame_testing.py` | **new**: the scatter helper (the brief's name; the plan calls it `_uniform_padded_scatter.py`). `scatter_frame` scatters a ragged frame at its own four widths, or a padded frame in block layout (gappy in BOTH blocks); finite garbage in every junk entry, written into the clean layout's own arrays. Also `stores_of`, `stack_frames`, `element`, `ragged` (via `ut3frame_to_t3frame`), `junk_is_zero`. |
| `tests/backend/test_linalg.py` | T7 (`TestPadSafePinnedCompletion`, 6 tests); module docstring extended. |
| `tests/backend/test_uniform_padded_frame.py` | **new**: T8, T17, structural errors, a round trip of the helper. |
| `tests/test_dispatch.py` | four T16 rows (below). |

No HISTORICAL strings: none of the new code contracts anything. The QR, permutations, masks and placements are
`take_along_axis`, `where`, `concatenate` and reshapes.

## The code

- **`pad_safe_pinned_completion(pinned, pinned_mask, seeds, seed_mask, row_mask)`.**
  - The masks are host bool (`require_concrete_masks`), broadcast to the common stack `lead`.
  - The per-element count check `sum(seed_mask) <= sum(row_mask) - sum(pinned_mask)` raises a `ValueError`.
  - The inputs are masked at entry; `M = [pinned | seeds]`.
  - Host permutations: columns `[real pins | selected seeds, in order | the rest]` (stable argsort of a key), rows
    real-first.
  - One reduced QR, sign-fixed by `diag(R) >= 0` (as `pinned_completion`); the rows are un-permuted.
  - Output column `j` is QR column `s_real + cumsum(seed_mask)_j − 1`, times `seed_mask_j`.
  - Junk rows are bitwise zero: they are zero and trail, so the reflectors of the selected columns (all at positions
    `< n_real`) never touch them.
- **`ufv_frame_widths(N, n, r) = (min(n, N), min(n, N, r²), r, r)`.**
- **`ufv_empty_frame(shape, N=None, stack_shape=(), use_jax=False)`.** The §5.3 table. The masks are
  `ufv_make_frame_masks` of the rank-0 stores at widths `(0, 0, 1, 1)`.
- **`ufv_padded_frame(old_frame_data, tucker_ranks, tt_ranks, sharing=None, seed=ZERO_FRAME_SEED) -> (frame, (Na, Rb))`.**
  1. The old counts come from the mask sums, per element. A layout check (`compute_orthogonal_representation_ranks`
     of the up and left counts) must reproduce the down and right counts.
  2. The target ranks may be a sequence or a `(d,)+C` / `(d+1,)+C` host int array (per element). Boundary TT
     ranks must be 1. The completion counts `k = target − old` must be ≥ 0.
  3. Widths: `Na = max(U°, O°)` and `Rb = max(L°, R°)` (padded); `Ka = max(kU, kD)`, `Kb = max(kL, kR)`.
  4. The old frame is masked once (`ufv_apply_frame_masks`) and zero-padded to the split. Output masks are
     `[old mask, padded with False | prefix(k)]`.
  5. **Tucker:** one batched `pad_safe_pinned_completion` over the sharing groups. Seeds are each group's keyed
     Haar rows (`_haar_seed_rows(Ka, N_first, first, seed)`; the first `kU` rows equal the ragged draw). Row mask:
     the shape prefix. Members take their group's slice, so tied factors are bitwise equal. A group whose members
     differ in old up mask or Tucker count raises.
  6. **Left (cores `0..d−2`), right (cores `1..d−1`), down (all):** one `_store_completion` per store, batched over
     its modes and the stack.
     - Outside units are placed exactly.
     - The rest are completed by `pad_safe_pinned_completion` on the old block's sub-grid. The pins are the old
       columns; the seeds are the inside coordinate units at columns `t..count−1`.
     - Carriers `left[d−1]` and `right[0]` keep their old content, padded.

### Judgment calls

1. **The coordinate order without a cumsum.**
   - Issue: the brief (and the prototype) order the real rows by a ragged key built from cumsums of the masks.
   - Finding: a mutation that replaced the cumsum ranks with the raw slot indices survived, and so did one that
     replaced the real count with the slot width. Both are equivalent mutants: the masks keep the real slots in
     order (`ut3frame_to_t3frame` reads them in mask order), so the ragged row-major order of the real rows IS
     their flat slot order.
   - Decision: `_units` takes the True rows in flat order (a stable argsort of `~rows`), and the cumsum key is
     gone.
   - Why: simpler, with identical results. T8 compares this against the ragged start, which still uses
     `_coordinate_seed_order`.
   - Consequence: the uniform start does not call `_coordinate_seed_order`; §4.2 expects it to be shared.
2. **jit vs numpy: tolerance, plus bitwise structure.**
   - Issue: the brief says "bitwise under jit vs numpy"; the plan's T8 says ≤ 1e-12; the house rule allows bitwise
     assertions only for structural facts.
   - Measurement: on this machine, numpy, eager jax and jit agree BITWISE on every case I tried (d = 3 and 4,
     gappy, garbage).
   - Decision: the T16 row asserts that the old block (a masked copy) is bitwise equal, that the junk is exactly 0
     under jit, and that the completions agree to ≤ 1e-12.
   - Why: jit-vs-numpy QR bits depend on the two LAPACK builds (numpy's and jaxlib's), and the house rule allows
     bitwise only for structural facts. **For Nick** if he wants bitwise everywhere.
3. **Where the jit check lives.** In the T16 row in `tests/test_dispatch.py` (x64 is on there), not in the T8
   file, which stays numpy-only as the testing strategy says. `jax.experimental.enable_x64` is not a stable API
   in jax 0.10.
4. **Extra T8 cases beyond the plan's 8:**
   - sharing; d = 1;
   - the zero tensor padded with garbage (split `(2, 3)`);
   - the zero tensor at d = 1, the only configuration with real old rows but no old column. It is needed to kill
     the `has_old`-ignored mutation;
   - an old frame that is itself in block layout (gappy masks in both of its blocks): the next increase.
5. **`ufv_frame_widths` dispatch.** It takes three ints, as §5.3 specifies. Its T16 row checks ints in,
   ints out, usable as static shapes under jit (there are no arrays to dispatch).
6. **A dead check removed.** "A count exceeds the coordinates its unfolding offers" cannot fire. The real rows of
   an unfolding bound its target store, so the remainder after the outside units always fits inside, and
   `pad_safe_pinned_completion`'s count check stands behind it anyway.

## Verification

- **Against the ragged start, through the scatter** (`TestStartAgainstRagged`): 11 cases plus the two extra
  tests. Real parts ≤ 1e-12; measured worst 2.8e-17, most cases exactly 0.
  - Junk is exactly 0.
  - Output masks equal the independent expectation exactly, and are host numpy.
  - Garbage changes nothing (bitwise, equal layouts).
  - A `C = (2,)` stack with different old and target ranks per element reads back element by element.
- **T7:** the real block equals `pinned_completion` ≤ 1e-14 over 5 shapes. Also covered:
  - offset and scattered seed masks;
  - per-element masks in a stack;
  - garbage giving the bitwise clean result;
  - degenerate seeds (in span, zero) completed;
  - the count rejection.
- **T17:** 40 random structures × 3 padding regimes (`n > N`, `r² < n`), d = 1..4, exact.
- **T16 rows** (`tests/test_dispatch.py`):
  - `test_pad_safe_pinned_completion_dispatch`: numpy → numpy, jax → jax eager, and jit with the masks closed
    over, ≤ 1e-12 with exact junk; per-element masks; a traced mask gives the actionable error.
  - `test_frame_widths_dispatch`.
  - `test_uniform_empty_frame_dispatch`: `use_jax`, jit, masks host.
  - `test_uniform_padded_frame_dispatch`: three cases (gappy with garbage; the zero tensor shared; `C = (2,)` with
    per-element ranks); numpy, eager jax and jit; masks host; a traced mask raises. Eager jax costs about 2.5 s
    per case.
- **Mutations, 17, all killed:**
  - sign fix; entry masks (pins, seeds); row permutation; column order; count check;
  - old frame unmasked; inside before outside; units in reversed or column-major order; `has_old`;
  - output up mask; group, layout, below-old and boundary checks; Haar key.
  - The two cumsum mutants are equivalent (judgment call 1).
- **Doctests:** `linalg`, `ufv_masking`, `uniform_padded_frames`.

## Full suite

`PYTHONPATH=$PWD python -m pytest tests -q -n auto`, on the final code: **1,271 passed, 4 skipped, 48,376
subtests passed** in 9 min 25 s (N3: 1,245 passed). The one warning is the existing `adam` one in the optimizer
tests.

## Plan problems

- The helper's file name: the plan (§7 and the N4 row) says `tests/backend/_uniform_padded_scatter.py`; the brief
  says `_uniform_padded_frame_testing.py`. I used the brief's.
- §4.2 says `_coordinate_seed_order` is shared with the uniform start; the uniform start does not need it
  (judgment call 1).
- The N4 row of §8 does not list `tests/test_dispatch.py`, but T16 and the maintainer's ruling require rows for the
  four new public functions; they are added.
