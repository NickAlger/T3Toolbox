# Pad-Safe SVD Packet

Batched SVD of zero-padded matrices whose null-space singular vectors would
otherwise contaminate the pad region. Solves it exactly, tolerance-free,
with static shapes (one jit compile for all mask patterns), at
`O(N M^2 + M^3)` cost independent of pad counts.

## Contents

| File | Role |
|---|---|
| `ALGORITHM.md` | **Start here.** Direct spec of the final algorithm: contract, 4 steps, why each works, invariants, caveats. |
| `pad_safe_svd_jax.py` | Static-shape JAX implementation -- **the one to port**. Includes a jitted `lax.scan` sanity check over heterogeneous masks. |
| `pad_safe_svd.py` | NumPy reference (dynamic shapes, more readable). Same contract; includes an adversarial before/after demo. |
| `tests/test_pad_safe_svd.py` | Pytest suite: all invariants, edge cases, adversarial spectra, single-compile checks. `pytest tests/ -q` |
| `padded_svd.pdf` / `.tex` | Full 10-page writeup: five methods (A-D) with derivations, error analysis of the approximate variants, the worked examples, and how the final algorithm emerged. Reference material, not required reading. |

## Integration notes (for Claude Code)

1. Read `ALGORITHM.md` first; treat it as the spec. The PDF is background.
2. Port `pad_safe_svd_jax.pad_safe_svd_jax` into the codebase. It is pure
   `jax.numpy`, jit/vmap/scan-safe, ~40 lines. Keep the structure:
   - the row permutation before the QR is **load-bearing**, not cosmetic;
   - the constant `c = 4 * ||A||_F` has a deliberate safety margin --
     do not change it to `2 * ||A||` (documented regression);
   - the augmented SVD's right factor is intentionally discarded;
   - `Omega` is drawn once per shape `M` from a fixed seed, outside jit.
3. Copy the invariant checks from `tests/` into the codebase's test suite,
   especially the **bitwise** `== 0.0` assertions and the jit
   `_cache_size() == 1` check -- these are the guards that will catch a
   backend or refactor regression (e.g. a non-Householder QR).
4. Interface: `U, S, V = pad_safe_svd_jax(A_pad, row_pad, col_pad, Omega)`
   with boolean masks; the first `m = M - col_pad.sum()` triplets are the
   SVD of the unpadded matrix, in original coordinates; the rest carry
   `sigma = 0` and are safe to ignore.
5. Requirements: `n >= m` per instance (asserted in the NumPy version;
   in JAX, validate on the host side if inputs are untrusted) and `N >= M`
   (transpose first for wide padded shapes).
6. If the codebase runs float32: the bitwise guarantees are unaffected;
   loosen the orthonormality/sigma tolerances in tests to ~1e-5.

## Quick start

```bash
python pad_safe_svd.py          # NumPy demo: naive-vs-safe, n>=M and n<M
python pad_safe_svd_jax.py      # jitted scan over heterogeneous masks
pytest tests/ -q                # full invariant suite
```
