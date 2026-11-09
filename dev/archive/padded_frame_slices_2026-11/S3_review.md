# S3 review (round 1): the hinted path of `fv_padded_frame` — review stop

Reviewer, 2026-11-02. Scope: the working-tree diff against `2562264a`:

- `backend/fv_operations.py`: `_unfold`, `_check_hint`, `_hint_coordinates`, `_tucker_completion`,
  `_pinned_left_sweep`, `_down_completion`, and the hinted branch with its docstring and doctest;
- `tests/backend/test_padded_frame.py` (T3);
- `tests/test_contractions_interpreter.py` (HISTORICAL 3–12);
- `tests/test_dispatch.py` (the hinted T10 row).

Checked against plan §2.3, §2.4, §4.4–4.7, §4.10, §5.2, §6 T3 / T10 / T12, and §12.

## Overall

This is a faithful and careful implementation of the pinned T3-SVD. Every step matches §4.4–4.7 line by line,
and the mathematics holds:

- the right projections are taken in the right bases;
- the pins are held fixed;
- the R sweep is the exact mirror;
- the down step works on the new slots only.

Three independent probes back this up. The output depends only on the ambient tensor of the hint (invariant under
a non-orthogonal regauge of the hint's cores and under redundant rank). The `X_old + H` exactness claim holds to
roundoff against a dense oracle that does not share code with the implementation. The cross-backend row is
determined on its data.

The tests do what T3 asks. Bitwise comparisons are used only for structural facts, and direction claims are made
only on content-determined slots.

I found no BLOCKER, MAJOR or MINOR. The NITs below are optional, and one observation is offered for Nick at this
review stop. **I approve.**

## Findings

1. **NIT — `tests/backend/test_padded_frame.py`, `TestMatrixCase.test_raw_external_hint_captures_at_least_the_library_frame`.**
   T3(j) states "the captured norm `‖P_{T_F} H‖` is at least the library frame's" generally, but the test checks it
   only at `d = 2`. My probe ran 10 seeds × the four `HINTED_CASES` × {raw `H`, `H + X_old`} = 80 cases, with `H`
   at Tucker ranks below the target. The padded frame never captured less than the library frame beyond roundoff:
   the minimum ratio was `1 − 8e-16`, which is a tie, and those ties come from the Tucker-only and in-block
   structures. The median ratio was 1.19. Optionally extend the assertion to the `HINTED_CASES`, with an explicit
   relative slack of `1e-12`. If you leave it at `d = 2`, that is acceptable, because the claim is not a theorem
   beyond the matrix case. If so, say so in the docstring.
2. **NIT — `fv_operations.py`, layout.**
   - The hint-cast line in the hinted branch (`hint = (tuple(xnp.asarray(...` …) is 129 characters long.
   - `_down_completion` is followed by a single blank line before `def fv_padded_frame`; the module's convention is
     two.
3. **Observation for Nick (review stop; not a defect).** `_down_completion` takes the new down rows only from the
   hint's content on the **new** Tucker slots, so `Anew` has `kU_i` columns. For a **minimal** `x_old` this always
   suffices. Minimality gives `sD_i = s_i`, so `kD_i = min(nU_i, rL_i·rR_{i+1}) − s_i ≤ kU_i`, and at `s = 0`,
   `kD_i = nD_i ≤ nU_i`. On a rich hint every new down row is therefore content-determined, which the oracle
   confirms. `kD_i > kU_i` arises only for a non-minimal `x_old`, the case implementer judgment call 5 mentions.
   There the extra down rows are the Householder completion of zero seeds. That is valid and deterministic, but it
   is neither content-driven nor the coordinate-seed convention the hint-free path uses for its sign covariance.
   The frontend reduces `x_old` structurally (§3.6), so this concerns only direct backend callers. It belongs in
   the docstring's caveats, or in R5/E-list wording at S10, rather than in the code.
4. **For the orchestrator.**
   - (a) Record G-acyclic in `docs/contributor/verification_records.md`. Implementer judgment call 6 left it
     unrecorded; I spot-checked that `t3_orthogonalization` and `tt_orthogonalization` do not import
     `fv_operations`.
   - (b) The commit trailers: as in S2, use the model(s) actually used.

## Checked and found correct

- **`fv_zero_point_frame` is byte-identical** to its version at `2562264a`. I compared the extracted function
  source at the commit and in the working tree, and they are equal. The anchor slip is fully undone.
- **`_hint_coordinates` (§4.4).**
  - `tt_squash_tails` runs first, so a T3 sum with boundary bonds 2 is accepted; this is tested.
  - Then come the down-orthogonalization `(W, Ztt)` and the two chains `ZL` / `ZR`.
  - `A_i = U°_i W_iᵀ`, and `Lh` / `Rh` use the reused strings.
  - `ΩL`, `ΩR` and `ZZ` are built by zippers.
  - `ZZ_{k}` is the exact change of coordinates `ZL_{≥k} = ZZ_k · ZR_{≥k}`: the right part of the left-orthogonal
    chain lies in the row space of `Z` at that bond, which `ZR_{≥k}` spans. So
    `Ht_i = ZL_i · ZZ_{i+1}` is the mixed-canonical center in `ZL_{<i}` ⊗ `W_i` ⊗ `ZR_{>i}`.
- **Tucker (§4.5).**
  - `E_i = O°_i ×ΩL_i ×ΩR_{i+1}` is `X_old`'s down rows in exactly those coordinates. The removal
    `N = Ht − (Ht Eᵀ) E` is the column-side projection, an exact projection when the Ω have orthonormal rows. T3(b)
    checks that for gradient hints.
  - The SVD compression preserves the Gram, and `Y = Wᵀ Uq sq` is the content in ambient coordinates.
  - Per group the members' `Y` are concatenated and pinned off `U°_{min g}`, giving one array per group.
  - `B_i = U_i W_iᵀ`, and its first `s_i` rows are `A_i`.
- **Left sweep (§4.6).**
  - The chain is `ZR`, right-orthonormal except its first core, so the SVD of the
    `(rL_k nU_k) × zr_{k+1}` matrix is the SVD of the full unfolding.
  - `Mb` is the hint in `Φ_k ⊗ U_k` against `ZR_{>k}`.
  - The column side removes `X_old`'s **right** interface at `k+1`, built from `R°`, in `ZR` coordinates.
  - The row side is pinned off the **embedded** `L°_k` in the `(rL_k, nU_k)` unfolding.
  - The carry uses `Mb`, not `Mp`; P10 says the rotation corrections are inert.
  - The carrier `L_{d−1}` is embedded only.
- **Right sweep = the exact mirror.** I traced the index maps:
  - `tt_reverse(ZL)` is right-orthonormal except its first core.
  - `ΩL[::-1][k'+1] = ΩL[d−k'−1]` is the old **left** interface at the mirrored bond, in the same coordinates
    (`tt_reverse` only swaps axes).
  - `tt_reverse(R°)` with `rR[::-1]` gives pins `R°_{d−1−k'}` transposed, embedded at `(rR_{d−k'}, nU, ·)`.
  - The reversed carrier is `R_0 = embed(R°_0)`.

  The new rows are taken in the `(b, n)` order of the reversed problem rather than the hint-free `(n, b)` order.
  That is a basis permutation, irrelevant to the hinted subspaces.
- **Down (§4.7).**
  - `G2 = Ztt ×₂ B` is the hint TT in the new Tucker basis.
  - `ΛL` / `ΛR` are zipped against the final `L` / `R`. They never touch the carriers: `ΛL_i` uses `L_{<i}` and
    `ΛR_{i+1}` uses `R_{>i}`.
  - `env` gives the coordinates in `Φ_i ⊗ U_i ⊗ Ψ_{i+1}`, `[..., s_i:]` restricts to the new slots, and the result
    is pinned off the embedded `O°_i` in the `(a, b)` space.
- **Exactness statement (X_old + H).** `TestDenseOracle.test_external_hint_plus_x_old`, `test_shared` and
  `test_stacked` pass the hint `H + X_old` but run the oracle with `Z = H` and require every slot to be determined.
  That is exactly the §2.3 claim that adding `X_old` changes no normal content. In the matrix case the claim gives
  the top-2 singular subspaces of `P_u^⊥ H P_v^⊥`, and the test checks it to `< 1e-10`.
- **Gauge of the hint (probe).**
  - Regauging every Tucker rank axis and every interior bond of the hint by random **non-orthogonal** invertible
    matrices leaves the dense hint the same to 7e-15. The assembled frame projectors change by at most **3.0e-14**
    over the four `HINTED_CASES`.
  - A redundant-rank hint (`H + X_old + X_old − X_old`) gives projectors equal to 9e-15.

  So the construction is a function of the ambient tensor, as §2.3 claims. The output's column signs follow LAPACK.
  That makes the gauge-invariant `P_{T_F} Y` comparison in T10 the right choice (judgment call 1).
- **Oracle independence.** `oracle_distances` is fully dense:
  - it uses `einsum` on the dense `Z`, dense interfaces of the old and new frames, and its own `numpy` SVD;
  - it shares no code with `_hint_coordinates`, the zippers, the sweeps or `pinned_leading_columns`;
  - its pins are rebuilt by its own `embed`.

  As §6 T3(h) prescribes, it conditions each step on the constructed frame's earlier choices (`Φ_k`, `U_k`,
  `Ψ_{k+1}`). This is still a complete check by induction:
  - `U` is conditioned only on the old frame;
  - `L_0` and `R_{d−1}` need no prior choice;
  - every later `L_k` / `R_k` is checked given verified predecessors;
  - `O` is checked given verified `L`, `U` and `R`.

  The determination rule (σ > 1e-8‖Z‖, gap > 1e-6‖Z‖) together with `require_full` makes every rich-hint slot
  count.
- **Determinedness of the T10 data (probe).** For the dispatch row's structure and seed, both hints
  (external + `X_old`, gradient) have **no undetermined slot** (oracle `r = count` everywhere, max distance 4e-15).
  The cross-backend comparison is therefore not exposed to roundoff choices (S1 finding 1).
- **Sharing, stack, dispatch, dtype.**
  - Sharing gives one factor per group; the cast preserves identity through its `id` map.
  - Stacking is vectorized through `C`-group contractions and the batched primitives. `_unfold` merges only
    non-stack axes.
  - The backend is inferred from `(frame, old_frame, hint)`.
  - The hint and the output are cast to the template dtype (judgment call 2). This is right: under x64, jax helpers
    promote float32.
  - The new structural check (template boundary bonds 1, judgment call 3) is sound; it also answers my S2 NIT 2 for
    the path that needs it.
- **Tests vs T3.**
  - (a) gradient hints at the library and padded frames, and a zero gradient: all valid.
  - (b) orthonormal overlaps (`1e-13`) and contractions (`≤ 1 + 1e-13`).
  - (c) matrix fixed point (`< 1e-10`).
  - (d) truncation.
  - (e) sharing (`is` plus the group oracle).
  - (f) stack equals per-element (`1e-12`).
  - (g) six structural errors, each naming the store; the stack message names the caller's reduction.
  - (h) the dense oracle at d = 3, 4, unshared and shared, `C = ()` and `(2,)`.
  - (i) `d = 1`.
  - (j) external hints valid at `s > 0` and `s = 0` (base point exactly 0, a structural fact); rich hints pass the
    oracle; the matrix top-k holds.
  - Degenerate hints (rank 1, zero, zero + `X_old`) are asserted valid only.

  All tolerances are explicit; there are no `allclose` defaults.
- **HISTORICAL 3–12.** The ten strings match the plan's §4.10 table letter for letter, each `''`.
  `'Cazb,Cuz->Caub'` (#11) is new, and the reused pair is as stated (judgment call 4).
- **Docstring and doctest.** The hinted section covers §5.2's points:
  - the four steps;
  - the counts structural, with no tolerance;
  - degenerate hints valid, with the trailing directions not determined (S1's wording);
  - the hint kinds;
  - the exactness caveat and the `+ X_old` remedy;
  - the cost and compression.

  The doctest covers plan items (c) and (d): a gradient hint via `tv_to_t3`, and an arbitrary T3.
- **No behaviour change elsewhere.** The hint-free path is untouched apart from passing `hint` to
  `tree_contains_jax`. `fv_zero_point_frame` is identical. The existing tests pass.

## Runs

- `tests/backend/test_padded_frame.py`, `tests/test_contractions_interpreter.py` and
  `tests/test_contractions_sharding.py`: **47 passed, 3,495 subtests**. The sharding test picks up the new strings.
- `test_dispatch.py -k padded_frame`: 2 passed, 4 subtests.
- `fv_operations` doctests: 4 passed.
- A stacked **and** shared hinted call (`C = (2,)`, groups `((0, 1), (2,))`) keeps the group identity and equals the
  per-element calls to 0.0 (probe6).
- Probes: `/tmp/claude-1000/-home-nick-repos-T3Polynomial/3eb9e6ad-d789-494e-ac17-23291348eda0/scratchpad/probe4.py`
  (hint gauge, redundant rank, T10 determinedness) and `.../probe5.py` (captured norm at d = 3, 4).

VERDICT: APPROVE
