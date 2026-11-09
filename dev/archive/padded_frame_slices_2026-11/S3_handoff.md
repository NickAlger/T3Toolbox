# S3 handoff: the hinted path of `fv_padded_frame`

Implementer, 2026-11-02. Plan §8 row S3 (review stop follows); §2.3, §4.4–4.7, §4.10, §5.2; tests T3, T10 (hinted
jit), T12; G-acyclic; P1/P3 re-measured. Nothing committed.

## Files touched

| file | lines | what |
|---|---|---|
| `t3toolbox/backend/fv_operations.py` | 26-27 (imports `t3_orthogonalization`, `tt_orthogonalization`); 560-721 (new private helpers); 782-812 (docstring: the hinted completion, hint structural errors); 860-873 (the hinted branch); 908-926 (doctest: two hints) | `_unfold`, `_check_hint`, `_hint_coordinates`, `_tucker_completion`, `_pinned_left_sweep`, `_down_completion`; the S2 `NotImplementedError` removed |
| `tests/backend/test_padded_frame.py` | 1-26 (docstring), 27-37 (imports), 350-693 (T3) | oracle helpers + `TestHintedValid`, `TestHintCoordinates`, `TestDenseOracle`, `TestMatrixCase`, `TestHintStructuralErrors` |
| `tests/test_contractions_interpreter.py` | 359-374 | HISTORICAL 3–12 (`''` each), incl. `'Cazb,Cuz->Caub'` (plan #11, new) |
| `tests/test_dispatch.py` | 1973-2009 | T10: `test_padded_frame_hinted_dispatch` (jit x64 + float32) |

## The new functions (private)

- **`_check_hint`**: the hint's structural checks per §3.2: a `(tucker, tt)` pair, `d` cores each, the template's
  stack `C`, mode sizes, and the bonds and Tucker ranks. A wrong stack is reported as the caller's job: combine several
  hints into one T3 first.
- **`_hint_coordinates`** (§4.4):
  - squashes the hint's tails, then down-orthogonalizes it (`W`, `Ztt`) and builds the left / right chains `ZL` / `ZR`;
  - expresses the old Tucker block in the hint's coordinates (`A_i`);
  - returns the old interfaces in the chains' coordinates (`ΩL` / `ΩR`, by zippers) and the chains' gauge `ZZ`.
- **`_tucker_completion`** (§4.5): the HOSVD-style two-sided step per mode, in ambient coordinates. Per group it
  concatenates the members' contents and calls `pinned_leading_columns` off `U°_g`, which yields one array per group.
- **`_pinned_left_sweep`** (§4.6): the two-sided pinned left TT-SVD. The right chain is the same sweep run on the
  `tt_reverse`d problem (`ZL`, `ΩL[::-1]`, `B[::-1]`, `R°`).
- **`_down_completion`** (§4.7): `G2`, the `ΛL` / `ΛR` zippers, then the environment on the **new** Tucker slots,
  pinned off the embedded `O°_i`.
- **`fv_padded_frame`**: checks the hint, requires template boundary bonds of 1, casts the hint to the template's
  dtype, then runs the four steps. The output is cast to the template's dtype, one cast per array, so a group's members
  stay one array.

## Numbers reproduced

- **T3(h) dense per-step oracle:** every new direction of every step (Tucker per group, `L_k`, `R_k`, `O_i`) is
  checked against its dense definition with Z = H at a projector distance < 1e-9. A check counts only when the oracle
  determines the direction (singular value above 1e-8·‖Z‖, gap above 1e-6·‖Z‖). Two kinds of hint:
  - **Rich hints** (H above the target ranks, H + X_old; and H itself at s = 0): every slot is required to be
    determined (`r == min(count, columns)`). Four structures (joint d = 3, joint d = 4, in-block d = 4, Tucker-only),
    plus shared, plus `C = (2,)` per element.
  - **Gradient hints:** compared where determined.

  Robustness probe (not in the suite): 10 seeds × 4 structures gave a worst distance of **3.5e-14**, with all 360
  slots determined.
- **T3(c) matrix fixed point** (d = 2, s = 1 → 3): the new slots return to the top-2 singular subspaces of `N`
  (< 1e-10).
- **T3(j) matrix case:** with H + X_old, the new slots are the top-2 subspaces of `P_u⊥ H P_v⊥` (< 1e-10). With the raw
  H the frame is valid and its captured norm ‖P_{T_F}H‖ is at least the library frame's.
- **T3(a), (d), (i), (j)-validity:** valid frames (orthogonality ≤ 1e-14, base point ≤ 1e-13, old block bitwise) for:
  - gradient hints at the library frame and at a padded frame;
  - a zero gradient;
  - rank-1, zero and zero + X_old hints;
  - low-rank external hints and T3 sums (boundary bonds 2), at s > 0 and s = 0 (base point exactly 0);
  - the truncation (a working-frame gradient, the target template);
  - d = 1.
- **T3(b):** for gradient hints, `A`, `ΩL[:d]` and `ΩR[1:]` have orthonormal rows ≤ 1e-13. For an arbitrary T3 hint
  their norms are ≤ 1 + 1e-13.
- **T3(e) sharing:** `up[0] is up[1]`; the group oracle uses the members' concatenated contents.
- **T3(f):** a stacked `C = (2,)` call equals per-element calls to 1e-12.
- **T3(g):** structural errors for mode sizes, bond mismatch, Tucker-rank mismatch, another stack (the message names
  the caller's reduction), too few cores, not a pair.
- **T10:** jax in gives jax out. The jitted result matches the eager numpy one gauge-invariantly through the dense
  `P_{T_F}Y`: rtol 1e-12 in x64, and 1e-5 for a float32 run whose outputs are all float32.
- **P1 / P3 re-measured through the function** on c11 / c13 structures (random `X_old`, gradient hints, 4 passes from
  both the library and the hint-free start):
  - orthogonality ≤ 1.2e-15 and base point ≤ 1.3e-15 (P1: 2.2e-15 / 1.6e-15);
  - `max|YYᵀ − I|` over `A`, `ΩL`, `ΩR` ≤ 3.9e-15 (P3: 3.3e-15).
- **G-acyclic:** an AST scan of every import node in the 41 backend modules, module- and function-level, plus the
  modules they reach. **No cycle through the backend**, and **no function-level library import** in the backend.

## Gates

- `test_padded_frame.py`, `test_fv_operations.py`, `test_contractions_interpreter.py`: **46 passed, 2,633 subtests**.
  `test_dispatch.py -k padded_frame`: passed. `test_contractions_sharding.py` picks up the new strings automatically
  (full suite).
- Doctests, `--doctest-modules t3toolbox/backend/fv_operations.py`: **4 passed**. The `fv_padded_frame` example now
  covers a gradient hint and an arbitrary T3 hint.
- Full suite `-n auto`: **1,218 passed, 4 skipped, 47,987 subtests, 1 warning** (10 min 59 s).

## Judgment calls

1. **Signs are backend-dependent, so cross-backend checks are gauge-invariant.** A hinted new column's sign follows
   its SVD vector's, so numpy and jax (and float32 vs float64) can differ by column signs. The x64 run happened to
   agree elementwise. The test compares the dense `P_{T_F}Y`, as §6's preamble prescribes for frame claims.
2. **The output is cast to the template's dtype.** Library helpers (`tt_squash_tails`, the zippers) build float64
   `ones` under jax x64, which promoted a float32 run. The cast keeps the §3.2 contract; one cast per array preserves
   the shared group factor's identity.
3. **A new structural check:** the hinted path requires template boundary bonds `rL[0] = rR[d] = 1` (the sweep's ones
   carry). A library frame always has them.
4. **HISTORICAL #11.** `'Cazb,Cuz->Caub'` is a new string, as the plan's table says; the reused pair is
   `'Czn,Cun->Czu'` / `'Cazb,Czu->Caub'`.
5. **Oracle scope.**
   - Direction claims use hints with at least `k` directions per slot (S1's lesson). Gradient-hint oracle runs compare
     only determined directions. A d = 4 in-block case has a gradient hint with no normal content at one mode, a true
     degenerate slot.
   - The `O_i` slots can need more directions than the new Tucker slots supply (`kD > kU`). The oracle then requires
     `min(count, columns)`.
6. **G-acyclic** was run as a script (scratchpad), not added as a test, because the plan calls it a gate, not a unit
   test. Recording it in `docs/contributor/verification_records.md` is left to the orchestrator; that file is outside
   this slice's list.
7. **A slip, caught:** the first hinted insertion landed in `fv_zero_point_frame`, matched by an identical anchor. The
   dense oracle exposed it (Tucker distances of 1.4, because the hint was being ignored). It is fixed, and the diff
   leaves `fv_zero_point_frame` untouched (`git diff` shows no change there).

## Draft commit message

```
padded frame S3: the hinted path of fv_padded_frame (the pinned T3-SVD of a hint)

backend/fv_operations.py: _hint_coordinates (tail squash, down- and chain-
orthogonalization, the old interfaces in the hint's coordinates),
_tucker_completion (HOSVD-style, two-sided, per sharing group, ambient),
_pinned_left_sweep (two-sided pinned TT-SVD; the right chain on the reversed
problem), _down_completion (the environment on the new Tucker slots) -- all
C-stacked named contractions, no threshold. HISTORICAL strings 3-12. Tests T3
(dense per-step oracle, matrix fixed point, external hints, degenerate hints
valid, sharing, stack C, structural errors), the hinted T10 jit row (x64 and
float32, gauge-invariant). P1/P3 re-measured; G-acyclic holds.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HmCnX876F5amZcDYUZPZqp
```
