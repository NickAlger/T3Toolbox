# S4 handoff: the uniform conversion, `UniformManifoldGeometryOps.padded_frame`

Implementer, 2026-11-02. Plan §8 row S4; §3.3, §4.8 (item 7: no twin), §5.4 (the uniform `padded_frame` member);
tests T4(a)–(d) and the uniform rows of T10. Nothing committed.

## Files touched

| file | lines | what |
|---|---|---|
| `t3toolbox/backend/geometry.py` | 62 (import `ut3_conversions`); 716-767 | the member `UniformManifoldGeometryOps.padded_frame`, after `zero_point_frame` |
| `t3toolbox/backend/fv_operations.py` | 818-822 (docstring only) | the raw-uniform recipe now names the hint's unpack (`ut3_to_t3`) and points to the member |
| `tests/backend/test_padded_frame.py` | 21-27 (docstring), 45-50 (imports), 707-882 (T4) | `TestUniformMember` and helpers |
| `tests/test_dispatch.py` | 1973-1995 | T10: `test_uniform_padded_frame_dispatch` |

No `ufv_padded_frame`; `ufv_operations.py` is untouched.

## The member

**`UniformManifoldGeometryOps.padded_frame(frame, old_frame, hint, seed=None)`** runs host-side and eagerly. It requires
concrete masks everywhere, and the old frame and the hint must have the template's stack `C` (else a structural
`ValueError`). Per stack element (`np.ndindex(*C)`), it:
1. unpacks the template and the old frame with `ut3frame_to_t3frame`, and the hint with `ut3_to_t3` (through its
   masks, gappy ones included);
2. calls `fv_padded_frame(..., self.groups, seed)`;
3. repacks with `t3frame_to_ut3frame` at the template's padded `N`, `nU`, `nD`, `rL`, `rR`;
4. checks that the repacked masks equal the template's element masks (else a structural `ValueError`, the
   `ufv_zero_point_frame` check).

It then restacks the supercores (`xnp.stack` on axis 1, reshaped to the template's supercore shapes) and returns them
with the template's own masks object. Only real blocks are read. `seed=None` means `ZERO_FRAME_SEED`. The docstring
says it is built through the ragged constructor, host-side and eager.

## Numbers reproduced

- **T4(a)** — the real blocks equal `fv_padded_frame` on the unpacked inputs to ≤ 1e-14, in eight cases:
  - hint-free at s = 0 and at s > 0;
  - a `from_t3` hint (H + X_old) at s > 0, and H alone at s = 0;
  - a **`utv_to_ut3` gradient hint**, whose TT bond masks a test confirms are not prefixes (gappy);
  - shared with a hint;
  - a **non-minimal `x_old`**, hint-free and with a gradient hint.

  The output frames are orthonormal to ≤ 1e-14. `old_frame=None, hint=None` equals `ufv_zero_point_frame`, unshared
  and shared: the down, left and right supercores bitwise, `up` to 1e-14, the masks the same objects.
- **T4(b):** `out[5] is template[5]` in every case, and every padded supercore entry is exactly `0.0`.
- **T4(c):** NaN in the padding of the template, the old frame and the hint leaves the output NaN-free, with real blocks
  equal to the clean run's to 1e-14.
- **T4(d):** a varying-rank stack (two elements with different old and new ranks, a common pad, and hints padded to
  `n = 8`, `r = 6`) equals per-element member calls to 1e-14, hint-free and hinted. The template's per-element masks
  differ and are carried through.
- **Structural:** an old frame with another stack is a `ValueError`.
- **T10:** numpy in gives numpy out. Jax supercores in give jax supercores out, with the masks host numpy and the
  template's own, hint-free and hinted.

## Gates

- `test_padded_frame.py`, `test_geometry.py`, `test_fv_operations.py`: **60 passed, 346 subtests**.
  `test_dispatch.py -k uniform_padded_frame`: passed.
- Doctests, `geometry.py` and `fv_operations.py`: **4 passed** (`geometry.py` has no module doctests; the member has
  none, as `zero_point_frame` beside it has none).
- G-acyclic re-run (the new `geometry → ut3_conversions` edge): no cycle, no function-level import.
- Full suite `-n auto`: **1225 passed, 4 skipped, 1 warning, 48009 subtests passed in 626.74s (0:10:26)**

## Judgment calls

1. **Per-element indexing instead of `ufv_unstack` / `ufv_stack`.** The member slices `(slice(None),) + idx` on the
   supercores and masks, and restacks with `xnp.stack` + `reshape`. That is the same operation, without building a
   nested tree. The reshape only splits the flattened stack axis 1 back into `C`, so no non-stack axis moves.
2. **A stack check on the old frame and the hint.** Element indexing needs a matching `C`, so the member raises a
   structural `ValueError` naming the input. `fv_padded_frame` already checks each element's internal structure.
3. **The recipe edit** in `fv_padded_frame`'s docstring is documentation only. It completes the three-call recipe the
   coordinator asked for (the hint's unpack, and a pointer to the member). No S1–S3 code changed.
4. **T4(e)** — the native uniform old frame's gauge equivalence, and a gappy `x_old` from uniform `+` — needs
   `padded_structures` and belongs to S5 by the slice table. It is not in this slice.
5. **No doctest on the member.** It is a geometry member like `zero_point_frame`, whose neighbours carry none. The
   behaviour is pinned by T4 and the dispatch row.

## Draft commit message

```
padded frame S4: the uniform conversion, UniformManifoldGeometryOps.padded_frame

backend/geometry.py: the uniform padded_frame member -- built through the
ragged constructor, host-side and eager: per stack element unpack the
template / old frame (ut3frame_to_t3frame) and the hint (ut3_to_t3, gappy
masks included), fv_padded_frame with the geometry's groups, repack at the
template's padded sizes (t3frame_to_ut3frame) with the template's masks
checked and returned; no ufv_padded_frame twin. fv_padded_frame's docstring
completes the raw-uniform recipe. Tests T4(a)-(d) (real blocks vs ragged to
1e-14, masks the template's, zero padding, NaN padding never read,
varying-rank stack), the uniform T10 dispatch row.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HmCnX876F5amZcDYUZPZqp
```
