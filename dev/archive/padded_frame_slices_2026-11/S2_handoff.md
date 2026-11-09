# S2 handoff: `fv_padded_frame`, the hint-free path

Implementer, 2026-11-02. Plan §8 row S2; §3.2, §4.2–4.3, §5.2, I-11, I-32; tests T2 (§6) and the hint-free T10 row.
Nothing committed.

## ⚠ Note for the orchestrator first

The S1 commit `b14f56eb` does **not** contain `tests/test_dispatch.py`. S1's T10 row (`test_pinned_primitives_dispatch`,
including the round-2 jax-validity subtest) is still an uncommitted working-tree change. The working-tree diff of that
file therefore holds both S1's row (lines 146–176) and S2's (lines 1953–1972). Stage it with S2, or as an S1 fixup first.

## Files touched

| file | lines | what |
|---|---|---|
| `t3toolbox/backend/fv_operations.py` | 9-10 (module docstring); 40 (`__all__`); 383-385 (`fv_zero_point_frame` "See also"); 438-747 (new section) | `_frame_stores`, `_empty_old_frame`, `_embed`, `_broadcast_C`, `_outside_first` (private), `fv_padded_frame` |
| `tests/backend/test_padded_frame.py` | new, 345 lines | T2(a)–(f), structural errors, template-values-not-read |
| `tests/test_dispatch.py` | 1953-1972 | T10: `test_padded_frame_dispatch` (hint-free) |

## The new functions

- **`fv_padded_frame(frame, old_frame=None, hint=None, groups=(), seed=ZERO_FRAME_SEED)`**: the frame of `X_old` at the
  template's larger structure. The old block is embedded as a copy, with zero carriers. The new slots are filled with
  a keyed Haar Tucker completion (`pinned_completion` against `U°_g`; `_haar_rows` at `s_g = 0`) and outside-first
  coordinates, then an in-block completion from coordinate seeds for the TT and down cores. It is `C`-vectorized,
  backend-inferred and jit-traceable. `hint` is not `None` → `NotImplementedError` (S3).
- **`_frame_stores`**: reads `C` and the stores `(N, nU, nD, rL, rR)` from core shapes. It checks neighbour consistency
  and admissibility; each failure is a `ValueError` naming the store and mode.
- **`_empty_old_frame`**: the zero tensor's old frame (Tucker stores 0, bond stores `(1, 0, …, 0, 1)`).
- **`_embed`**: `pad_or_truncate` into the leading slots.
- **`_outside_first`**: the generic hint-free completion of one unfolding. Its input is the old core arranged
  `C+(row1, row2, old cols)`. The `L` / `R` / `O` cores map onto it by `moveaxis`.

## Numbers reproduced

- **T2(a):** `old_frame=None` equals `fv_zero_point_frame` for d = 1, 3, 4, unshared and two shared partitions, seeds 0
  and 5. The down / left / right cores match **bitwise**, the Haar factors to ≤ 1e-14, and a group's members are one
  array (`is`). A stacked `C = (2, 3)` template gives the zero-point frame broadcast, to 1e-14.
- **T2(b):** Tucker-only, TT-only, joint, joint d = 4, d = 1 and no-increase cases all give valid frames:
  - orthogonality residual ≤ 1e-14;
  - both reconstructions within 1e-13 relative of `X_old`;
  - the old block a bitwise copy, with zeros elsewhere in the old columns and rows;
  - carriers zero on every new index.

  No increase returns the old frame bitwise. The shared case (`share((0, 0, 1))`, group rank 4 → 5) gives one array
  per group and a valid frame.
- **T2(c):** the in-block columns equal `pinned_completion(old unfolding, coordinate seeds)` to 1e-14 in three cases:
  - d = 4, bond 2 grows while `(rL_1, nU_1)` stay;
  - a non-minimal old point at bond 1;
  - the down core when the middle Tucker rank grows at fixed bonds.

  Two calls agree to 1e-15. The outside units are exact, in row-major order.
- **T2(d):** `kU → kU + 1` gives leading rows equal to 1e-14; the other modes are bitwise unchanged.
- **T2(e):** `C = (2,)` and `(2, 3)`, an in-block case included, equal per-element calls to 1e-14. Each element is
  valid.
- **T2(f):** random sign gauges (Tucker rows, left / right bonds, down rows; boundaries +1) on a joint case and two
  in-block cases (TT and down): the assembled projectors (`UᵀU`, the dense left / right interfaces, the dense down
  environments) agree to ≤ 1e-12. In the in-block TT case they are exactly 0.0.
  - Sensitivity probe (not in the suite): swapping the in-block seeds for a fixed Gaussian gives a projector
    difference of **0.32** on the same case, so the test detects a non-covariant seed rule (cf. P17).
- **T10:** numpy in gives numpy out, jax in gives jax out, eager. `jax.jit(..., static_argnames=('groups', 'seed'))`
  equals the eager numpy result to 1e-12 (x64): at the zero tensor with a shared partition and seed 3, and at an old
  frame with an in-block `L_1` and a Haar-completed `U_3`.
- **Doctest:** the zero-point identity (coordinate cores bitwise, Haar to < 1e-14), and at a resized point: shapes,
  the old block a bitwise copy, orthogonality < 1e-14, both reconstructions < 1e-13 relative.

## Gates

- `tests/backend/test_padded_frame.py`, `tests/backend/test_fv_operations.py`, `tests/test_contractions_interpreter.py`:
  **32 passed, 2,272 subtests**. `tests/test_dispatch.py -k padded_frame`: passed.
- Doctests, `--doctest-modules t3toolbox/backend/fv_operations.py`: **4 passed**.
- Full suite `-n auto`: **1,203 passed, 4 skipped, 47,604 subtests, 1 warning** (11 min 14 s).

## Judgment calls

1. **An old block with no old columns counts as empty** (`_outside_first`: `c == 0` → every coordinate is outside).
   The plan's rule "outside `{a < sL, n < s}`" exists to keep the units off the old columns, and with none there is
   nothing to avoid. It matters at `s = 0, d = 1`: the down core's bond block `(1, 1)` is "inside" the zero tensor's
   boundary stores. The literal rule would send it through a QR, instead of the identity prefix that T2(a) checks
   bitwise. For `d > 1` this changes nothing.
2. **Validation beyond the plan's list.**
   - `_frame_stores` also checks each core against its neighbours (a consistent frame) for both the template and the
     old frame.
   - It checks the old frame's admissibility.
   - It range-checks the group indices.

   The messages name the store and mode. A hint raises `NotImplementedError` before any validation.
3. **Stack broadcasting materializes copies** (`xnp.array(xnp.broadcast_to(...))`), so no read-only views are
   returned. At `C = ()` a group's factor is one array object for all members, as in `fv_zero_point_frame`.
4. **dtype:** the old frame is cast to the template's dtype. The "old block a bitwise copy" claim holds when the
   dtypes agree.
5. **Docstring:** the hint section is a one-line placeholder until S3. `fv_zero_point_frame` gains a "See also" line,
   and the module docstring mentions the new function, as §5.2 allows. The function body of `fv_zero_point_frame` is
   unchanged.
6. **The T2(c) "TT-only increase at bond 1"** needs a non-minimal old point: for a minimal one `sL_1 = s_0` already
   saturates. The test also covers the minimal d = 4 bond-2 case, the natural in-block case.
7. **Extra test:** the template's values are never read (noisy template, bitwise-equal output).

## Draft commit message

```
padded frame S2: fv_padded_frame, the hint-free path

backend/fv_operations.py: fv_padded_frame (beside fv_zero_point_frame): the
frame of X_old at a zero-padded rank increase -- the old block embedded as a
copy, zero carriers, the new slots completed hint-free (keyed Haar Tucker
completion against U_old; outside-first coordinates, then an in-block
completion from coordinate seeds, covariant under a sign gauge). Structural
validation (consistent / admissible template and old frame, old stores <=
template, stack C, groups); C vectorized; jit-traceable. A hint raises
NotImplementedError until S3. old_frame=None reproduces fv_zero_point_frame.
Tests T2(a)-(f) (new test_padded_frame), the hint-free T10 dispatch row.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HmCnX876F5amZcDYUZPZqp
```
