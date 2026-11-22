NOT YET

# Review of `DATA_METRIC_PLAN.md` section 9 (lightweight typing), round 3

Reviewer, 2026-11-22. I read §9 at `68705f49` and ran two scripts:

- `scratchpad/r2/laws.py`: the two laws on 8 functions at orders `()`, `(2,)` and `(1, 1)`, with injected faults;
- `scratchpad/r2/entries_law.py`: the entries kind under the input rescaling.

**What holds up.** The typing table is right in substance, and the jet law is exact and catches what it claims to
catch. Its correction about the Taylor/derivative mix-up is also right.

**What has to change:**

- **9.2's premise is false:** the turned form of `P*` stays public in the backend engine.
- **9.3 claims the input law catches the probe residual/sample swap. It does not, and nor does the jet law;** the
  swap is an exact symmetry of both laws.
- **The parameter-side law is open in the plan.** It is derived and verified below, so the writer can put it in.

Findings 1 and 2 are blocking. The rest are should-fix or minor.

---

## The laws, verified

**The jet law, `w_k -> λ^|k| w_k` with one `λ_j` per jet variable.** At `(2,)`, `λ = 1.7`; at `(1, 1)`, `λ = (1.7, 0.6)`.

| Function or injected fault | `(2,)` | `(1,1)` |
| --- | --- | --- |
| `t3_probe`: `P` law (`z_k -> λ^|k| z_k`) | 3e-16 | 3e-16 |
| `tv_probe`: `P` law | 3e-16 | 3e-16 |
| `t3_probe_corewise_transpose`, with `c` under the `P*` law (`λ^-|k|`): output invariant | 4e-16 | 9e-16 |
| `tv_apply_transpose`: invariant | 5e-16 | 3e-16 |
| `t3_probe_corewise_hessian`: invariant | 3e-16 | 8e-16 |
| FAULT: missing or double flip | **2.6** caught | **1.6** caught |
| FAULT: a flip on one jet axis only | | **0.67** caught |
| FAULT: `c` moved as a jet (`P` law): the jet/dual-jet swap | **7.0** caught | **1.4** caught |
| FAULT: Taylor/derivative mix (`taylor_to_derivatives(c)` inside) | 4e-16, **not caught** (as §9 corrects) | 9e-16, not caught |
| FAULT: `c` and the sample swapped | 5e-16, **not caught** | 8e-16, not caught |

**The input law, `V_i -> Δ_i V_i`, with the cores contragredient (derived in finding 3).**

| Function or injected fault | `()` | `(2,)` | `(1,1)` |
| --- | --- | --- | --- |
| `t3_probe`: `z_i -> Δ_i^-1 z_i` | 2e-16 | 5e-16 | 6e-16 |
| `t3_apply`: invariant | 7e-17 | 3e-16 | 4e-16 |
| `tv_probe` (frame and variations transformed) | 3e-16 | 3e-16 | 4e-16 |
| `t3_probe_corewise_tangent` | 2e-16 | 5e-16 | 5e-16 |
| `t3_probe_corewise_transpose`: the `X'` law | 4e-16 | 9e-16 | 7e-16 |
| `tv_probe_transpose`: the `X'` law | 4e-16 | 5e-16 | 4e-16 |
| `t3_probe_corewise_hessian`: the `X'` law | 5e-16 | 6e-16 | 5e-16 |
| `tv_apply_transpose`: the `X'` law | 8e-16 | 3e-16 | 1e-15 |
| FAULT: `c` moved as `V_i*` (a residual where `c` is meant) | **1.7** caught | **3.5** caught | **2.7** caught |
| FAULT: `c` and the sample swapped | 9e-16, **not caught** | 6e-16, not caught | 4e-16, not caught |

**Entries obeys a different input law.** Its outputs transform as `z -> z / ∏_i Δ_i[n_i]`; with the same `Δ`, they
are not invariant.

---

## Findings

### 1. BLOCKING: 9.2. The turned form stays public in the backend engine, so "one public representation of `P*`" is false

**Evidence.**
- `w_tilde` and every tilde variable are public engine variables in the turned form:
  - they can be wanted, and `HeldVariables` holds them ("turned", `sampling_variables.py:223`; table rows `:56-98`);
  - `sampling_definitions` takes `w_tilde=` turned in its public reference functions (`:857, :926`, and others);
  - tests use `w_tilde` in 8 files (`test_sampling_definitions` 35, `test_sampling_apply` 15, `test_sampling_graph`
    9, and others).
- If `jet_to_dual_jet` and `dual_jet_to_jet` become private, a backend user who holds or seeds `w_tilde` loses the
  only converter. That fails the backend/frontend razor. The conversion is easy to get wrong: a missing flip passes
  every shape check (the table above).

**What would resolve it.** The recommendation, to put to the maintainer:
- **Frontend: one representation.** Remove both functions from `t3toolbox.fitting` and its `__all__`; nothing in the
  frontend takes a turned form.
- **Backend: the two functions stay public in `backend/jets`,** retyped as the converters between slot coordinates
  and the engine's turned coordinates. Their docstrings say: "for `sampling_variables`' `w_tilde` and held tilde
  variables; never for a transpose (D3)".
- 9.2 is then a *frontend* settlement of decision 4. It needs its own question in §8 (finding 7).

**This also supersedes O-20.** O-20 is the maintainer's decision that the frontend `dual_jet_to_jet` raises in safe
mode on a zero weight (`fitting.py:273-281`). The plan should name it.

**The blast radius,** counted at `68705f49`:

| Where | References |
| --- | --- |
| Library: `backend/jets.py` | 16 + 10 (definitions, docstrings) |
| Library: frontend `fitting.py` | 7 + 13 (the export, the frontend `dual_jet_to_jet`, the module text `:45-47`) |
| Library: module doctest lines (`>>>`) across the two files | 11 |
| Library: `sampling_steps.py` (the engine's call; it stays) | 2 |
| `test_jets` (`TestDualJet`; `TestFrontendDualJetToJet`, which goes with O-20) | 78 |
| `test_dispatch` | 16 |
| `test_jax_absent` | 7 |
| Examples and `docs/` pages | 0 (one prose mention of "dual jet" in `docs/contributor/ambient_derivative_transpose_note.md`) |

The prose "turned inside (a dual jet)" in the transposes' docstrings (about 70 lines across the library) is retyped by
M4 anyway.

### 2. BLOCKING: 9.3. The input law does not catch the residual/sample swap, and nor does the jet law

**The claim.** §9.3 says the input law "catches a `V`/`V*` swap (the probe residual/sample trap)".

**Why it does not.** Both arguments of a probe transpose are `V_i`-typed: `c ∈ P*⊗V_i` and the sample
`w ∈ P⊗V_i`. So the input law moves them alike (9e-16, table above). The jet law tells `P` from `P*`, but the swapped
function is covariant under `λ -> 1/λ`:
- `T(λ^{k} w, λ^{-k} c)` is `T` at `1/λ`, which is exact;
- so the swap passes the jet law too (5e-16 at `(2,)`, 8e-16 at `(1,1)`).

No scaling law can see this swap. That agrees with LOG: "a swapped call is a valid transpose of a different
problem". The right/wrong adjoint doctests stay the only guard.

**What each law does catch, verified:**
- **The jet law:** a missing, double or one-axis flip, and a jet/dual-jet swap. It needs `λ_j ≠ 1`, distinct per jet
  variable, and some `n_j ≥ 1`; it is vacuous at `order = ()`.
- **The input law:** treating `c` as `V_i*` (a probe residual `r ∈ P⊗V_i*` handled where `c ∈ P*⊗V_i` is meant)
  inside library code, and wrong co- or contravariance in anything that changes `V_i` coordinates
  (`rescale_coordinates`, `t3_unscale_coordinates`).
- **Both** are homogeneity tests: necessary, not sufficient. A wrong constant, sign or binomial factor passes them.
  The dense oracles stay primary. Say so in one sentence.

### 3. SHOULD FIX: the parameter-side law of the input rescaling, derived and verified (it is open in 9.3)

The point is `x = (B, G)` with `x(w) = … B_i w_i …`. Keep every represented value fixed while `w_i -> Δ_i w_i`
(`Δ_i` diagonal, positive, `(N_i,)`):

| Object | Type | Law |
| --- | --- | --- |
| input jets `w_i` | `P⊗V_i` | `Δ_i w_i` |
| Tucker cores `B_i` (`nD_i x N_i`) and their directions `Ḃ_i` | `X` | `B_i Δ_i^-1` |
| TT cores `G` and `Ġ` | `X` | unchanged |
| frame: the down store `D_i`; variations `V_i` | `X` | `D_i Δ_i^-1`, `V_i Δ_i^-1` |
| frame: `U`, `L`, `R`; variations `H` | `X` | unchanged |
| corewise blocks returned (`g_B`, `K_B`) | `X'` | `g_B Δ_i` (contragredient to `Ḃ`: `<g_B Δ, Ḃ Δ^-1> = <g_B, Ḃ>`) |
| TT blocks returned (`g_G`, `H̃`) | `X'` | unchanged |
| frame-form blocks returned (`Ṽ_i`) | `X'` | `Ṽ_i Δ_i` |
| probe `z_i`, `y_i`, `r_i`, `Jp_i` | `P⊗V_i*` | `Δ_i^-1 ·` |
| probe `c_i` | `P*⊗V_i` | `Δ_i ·` |
| apply `z`, `c` | `P`, `P*` | unchanged |
| entries `z` | | `/ ∏_i Δ_i[n_i]` (an index is a basis vector, not a vector that transforms) |
| probe weight `ω_i` (§2 full form, with the edge) | | `ω_i Δ_i`, so `f` is invariant. A block form follows only for `Δ_i = λ_i I`. This is audit D6 restated: with a block weight, rescaling a mode's coordinates changes the loss. |

**Caveats to state with the law:**
- The transformed frame is no longer orthonormal. The law holds for the **metric-free** maps (forwards, tangents,
  transposes, Hessians, `W`). It does not hold for `MANIFOLD.project`, `retract`, `inner`, `frame(x)` or `t3svd`: the
  Hilbert-Schmidt metric on `E` is not invariant under a non-orthogonal `Δ`. Raw backend arrays only, as §9.3 says.
- On a shared geometry, `Δ_i` must be equal within each sharing group.
- On the uniform layout, `Δ` is arbitrary on the padding, which the masks zero.
- **The two public instances of the laws:**
  - `rescale_coordinates` / `t3_unscale_coordinates` are the input law with `Δ` = the scales;
  - `rescale_direction` is the jet law, one `λ` per variable (O-40).

  Their tests should be law tests, and D8 (rescaling the direction changes the weight) is the jet law applied to `ω`
  (finding 4).

### 4. SHOULD FIX: the typing table, against §1 and for completeness

- **Two notations for one thing.**
  - §1 writes `J: X -> Y'`, `W: Y' -> Y`, with `'` for the dual and `Y'` the measurement space.
  - §9 writes `P` for the measurement jets and `P*` for their dual, with `*` for the dual and also `X'`.
  - So `Y' = P` (apply), `⊕_i P⊗V_i*` (probe), and `Y = P*`, `⊕_i P*⊗V_i`: the unprimed `Y` is the starred `P*`.
  - That inversion sits exactly where the typing is meant to help. Write the dictionary into §9 and use one dual
    symbol throughout (question below).
- **Rows missing from the table:**
  - **entries:** `index -> P` with `m = 0`, i.e. `R`, per sample; transposes `R -> X'`;
  - **the second-order transposes:** `P* × X -> X'` (resp. `⊕_i P*⊗V_i × X -> X'`), bilinear;
  - **the ambient transposes:** `P* -> E'` with `E = ⊗_i V_i*` (the tensor) and `E' = ⊗_i V_i` (the CP factors
    `c·⊗w`, which scale by `Δ_i`). `from_canonical` of the result is the Hilbert-Schmidt identification `E' ≅ E`, the
    one place a metric enters silently on the parameter side (audit G). The type column should say so;
  - **the dense functions:** `E × input jets -> Y'`;
  - **`W`:** `Y' -> Y`, and `ω`'s own law: `λ^-|k|` under the jet law, `Δ_i` on the probe edge;
  - **the oracle's outputs:** `r`, `Jp`: `Y'`; `g`, `JtJp`, `Kp`: `X'`; `f`: `R`.
- **The input jets' box is `J_in_i`, not `F`.** They are jets on their own box, clipped to `F` (`min(J_in_i, F)`), so
  `P⊗V_j` should read "jets on `J_in_j`". The jet law is unaffected.
- **`Hom_P(P⊗V_i, P) = P⊗V_i*` is right,** checked: apply is `P`-linear in `w_i` under the truncated product. But it
  uses the `P`-bilinear jet-product pairing, while `P*` is the real dual with the slot pairing. Both are written `*`.
  Say which is meant where.
- **Sign slip.** A flipped dual jet scales as `λ^-(n-k)`, not `λ^(n-k)`.

### 5. SHOULD FIX: 9.1's scope and where the types live

- **The style rule changes.** The type goes before `len`, which changes `signature_style.md`'s micro-grammar ("`len`
  first", `docs/contributor/signature_style.md:74-89`). That is a durable contributor rule, so the doc changes in the
  slice that first applies it, not only in P19.
- **The scope is about 100 signatures:**
  - the 42 transposes;
  - the forwards and tangents, ragged and uniform, backend and frontend (about 30);
  - the dense functions (3), the `jets` helpers (about 12), `W`, and the oracle.
- **The routing rule.** `SAMPLE_VARIABLES.md` is under `dev/`, so a type column there does not reach users. Put it in
  the engine's own variable table (`sampling_variables.py`, module docstring), or have P19 promote it.
- **The tilde variables** are typed as turned `P*` (the engine form). Finding 1 depends on that being explicit.
- **Settle the notation and the grammar in M1's design note,** before any slice writes types.

### 6. MINOR: the slicing (9.4)

**What is right:**
- 9.2 and the law helper in M1 (with `W`);
- M2 applies the laws to the engine (seed, `JtJp`, `Kp`);
- M4 applies them to all 42 transposes.

**What to add:**
- M1 tests `W` itself under the laws: given `ω` moved by its law, `W` maps the `P` law to the `P*` law.
- M1 or M3 tests `rescale_coordinates` and `rescale_direction` as the law instances (finding 3).
- 9.1's ~100 signature comments would double M4, which is the "mechanical" slice. Either keep 9.1 strictly per slice,
  or make it an M5 of comments only.
- The law helper needs the full table of finding 3, entries included, in M1's design note.

### 7. SHOULD FIX: questions to add to §8, each with a recommendation

9. **9.2: where the dual-jet converters are public.**
   - (a) **Frontend-private, backend-public,** retyped as the engine-form converters (recommended: the razor, since
     `w_tilde` and the tilde variables are public in the backend).
   - (b) Private everywhere, with `w_tilde` also made engine-internal: a larger change, to the engine and the
     reference module.
   - (c) Unchanged (decision 4 stays open).
   - Either (a) or (b) supersedes O-20.
10. **One dual symbol.**
    - Recommended: the maintainer's own, `'` with `Y`/`Y'`. Write `Y' = P⊗V_i*` in the comments, so the
      measurements are primed. Then `P' `, `V_i'`, `E'` replace `P*`, `V_i*`.
    - The alternative is `*` everywhere, with §1 rewritten to match.
11. **The scope of 9.1.** Backend and frontend, as drafted (recommended: the razor, since backend users read the
    backend signatures), or the frontend only.

---

## Summary for the writer

1. Rewrite 9.2 as frontend-only, with backend converters retyped (or put the choice as Q9). Name O-20.
2. Correct 9.3's catch list: the swap is invisible to both laws; say what each law catches.
3. Insert the parameter law (finding 3), with its caveats and with entries.
4. Complete the table and reconcile it with §1's `Y`/`Y'`; fix the sign.
5. Place 9.1's grammar and the type column; add Q9 to Q11.
