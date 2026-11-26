# P7 (Engine C): design note

Implementer, 2026-11-20. This note covers only what is new on top of P6. Nothing has been run. **[open]** marks a
choice for the orchestrator.

## 1. The dot rows (the frame form, `SAMPLE_VARIABLES.md` section 5.2)

All four have `carries_K=True` and need `{'variations'}`. Their reads are the frame stores, the plain variables, and
`V` and `H` as two new arguments, read per site.

| Variable (basis) | Driver, start | Reads, in step order | Terms and strings |
| --- | --- | --- | --- |
| `xi_U_dot` (`U`, leaf) | site | `V`, `w` | `'JWo,KCio->JWKCi'`, the leaf's jet shape |
| `mu_R_dot` (`R`) | left, zeros | `R`, `mu`, `xi`, `xi_U_dot`, `U`, `H` | `'JWKCa,Caib,JWCi->JWKCb'` (carry), `'JWCa,KCaib,JWCi->JWKCb'` (`H`), `'JWCa,Caib,JWKCi->JWKCb'` (`U` over `xi_U_dot`) |
| `nu_L_dot` (`L`) | right, zeros | `L`, `nu`, `xi`, `xi_U_dot`, `U`, `H` | the three mirror `nu` strings |
| `eta_D_dot` (`D`) | site | `mu_R_dot`, `R`, `mu`, `H`, `L`, `nu_L_dot`, `nu` | `'JWKCa,Caib,JWCb->JWKCi'` (`R`), `'JWCa,KCaib,JWCb->JWKCi'` (`H`), `'JWCa,Caib,JWKCb->JWKCi'` (`L`) |

- **The sum over the sites** where the direction is put in is the recurrence itself: each step adds the terms putting
  it at site `i` (`H[i]`, or `U[i]` over `xi_U_dot[i]`), and the carry holds the earlier sites. So a sweep starts at
  zeros.
- **Bonds and starts.** The bond read at a site follows the letter (`_BOND_OFFSET`). The start is sized by the swept
  store (`R` for `mu_R_dot`, `L` for `nu_L_dot`), and with `carries_K=True` it has the `K` axes, `F + W + K + C +
  (size,)`.
- **[open] The two names with both forms (O-12).** In P7 the frame-form rows are the public `nu_L_dot` and
  `eta_D_dot`, and `Jp` reads them. In P8, the raw rows come first under `needs={'core_directions'}`. A call with
  both arguments must then give `Jp` the frame form while the public names mean the raw form. So in P8 the frame
  rows become private (`_frame.nu_L_dot`, ...) and `Jp`'s reads move with them. I recommend leaving that rename to
  P8 rather than adding private rows now that nothing in P7 needs.

## 2. `Jp`, `Jp_norm_sq`, the seed of `JtJp`

- **`Jp`:** site, `carries_K=True`, reads `eta_D_dot`, `D`, `eta`, `V`. Its value is
  `z_term(eta_D_dot, D) + z_term(eta, V)`, with the strings `'JWKCi,Cio->JWKCo'` and `'JWCi,KCio->JWKCo'`. It has
  the shape `F + W + K + C + (N_i,)`.
- **`Jp_norm_sq`:** whole, reads `Jp` and `weight`. It is `sum_i sum(weight_i^2 * Jp_i^2)` over the jet axes, `W`
  and `N`, keeping `K + C`, as `f_step` does (with the same `0.5`-free dtype handling).
- **The seed of `JtJp`:** a private row `_Jp.w_tilde` reads `('Jp', 'weight')` through `w_tilde_step`: the same
  `jet_to_dual_jet`, which takes `F + W + [K] + C`.

## 3. The private twins (the `Jp`-seeded adjoint, row 3.3)

- **The naming rule:** `twin(name) = '_Jp.' + name.lstrip('_')`. That gives `_Jp.w_tilde`, `_Jp.xi_tilde`,
  `_Jp.mu_tilde`, `_Jp.nu_tilde`, `_Jp.eta_tilde_top`, `_Jp.V_tilde` and `_Jp.H_tilde`. The public `JtJp` is
  `g_step` over `_Jp.V_tilde` and `_Jp.H_tilde`: it is `g`'s twin.
- **Derived, not written by hand.** `_twins(graph)` maps each row on the path from `w_tilde` to `g` to its twin:
  the same step, `carries_K=True` and the same start, with each tilde read renamed by the rule and the seed's `r`
  replaced by `Jp`.
- **Reuse.** The rows reuse P6's steps unchanged, with the call's record. Their strings already put `K` on the tilde
  operands.
- **The letter of a twin**, for `_BOND_OFFSET`, is read after the prefix.
- **Choice by the closure.** Nothing new: `JtJp` reads the twins, and they read `Jp`, which needs `variations`.
  Wanting `g` and `JtJp` runs both families over the same plain sweeps.
- **The test:** every row on the path from `w_tilde` to `V_tilde`, `H_tilde` and the short node has a twin, made by
  the rule, with the same step and `carries_K=True`.
- **[open] Two rows get no twin:** the full `eta_tilde` (only wanted publicly, which a private twin can never be)
  and `g` (`JtJp` is its twin). I propose the twin test say so, against the coordinator's "every public tilde row".

## 4. Shapes and `HeldVariables`

- **Where `K` appears.** In the table it is decided by the making row's `carries_K`, never by shape:
  - `J + W + K + C + (size,)` for `xi_U_dot` (leaf sizes, `nU`), `mu_R_dot` (bond, `rR`), `nu_L_dot` (bond, `rL`),
    `eta_D_dot` (site, `nD`) and `Jp` (mode, `N`);
  - `K + C` for `Jp_norm_sq`;
  - `JtJp` as `g`, with `K` in front of each block.
- **New slots** (public): `xi_U_dot`, `mu_R_dot`, `nu_L_dot`, `eta_D_dot`, `Jp`, `Jp_norm_sq`, `JtJp`. The twins are
  private, so they are not slots.
- **The held check** gains `K` from the call: a held `Jp` must have the call's `K`.

## 5. `GroupLengths` and `K`

- **Where it is read:** `K = V[0].shape[:V[0].ndim - 2 - len(C)]`, with `C` off `D[0]` as now. The call's record has
  `K = len(K)`; rows with `carries_K=False` get `K = 0`. The setting gains `K` for the shapes and the starts.
- **No new check.** `jet_contract`'s rank check covers `H` and the other `V[i]`: a wrong count raises there, and
  unequal sizes fail in the contraction. That is "what it needs to run".

## 6. The tests

- **By definition, with `K = ()` and `K = (2,)`:** every new name against the reference, which has no `K`, applied
  to each `(V[k], H[k])` and stacked. They run over the jet matrix, three problems and a nonuniform weight. For
  `JtJp` the reference's own function is used.
- **Identities, to rounding:**
  - the adjoint, `<Jp, omega^2 r>` (summed over every slot) `= <p, g>`;
  - `<p, JtJp> = Jp_norm_sq`, per `K` element;
  - **[open]** `JtJp` equals `g` with `Jp` held as `r` (no `y` needed). That holds per `K` element: a held `r` has no
    `K`.
- **The bridge:** `ProbeDerivativesKind.forward((V, H), (ww, pp), frame, sweep)` is `t!` times the new `Jp`.
  `.transpose` given `c_t = weight_t^2 Jp_t / t!` (forward) is the new `JtJp`, the same conversion as `g`. Both run at
  `K = ()` and `(2,)`.
- **The twin test** and the hand-derived step counts: `('JtJp',)` runs each tilde step twice when `g` is also wanted.
- **`test_dispatch`:** jit `('Jp', 'JtJp', 'Jp_norm_sq')` with `K = (2,)`.

## 7. Changes to the plan I would make

1. **The twin set** is the path that `JtJp` reads, without the full `eta_tilde` and `g` (section 3).
2. **The twin name rule** `'_Jp.' + name.lstrip('_')`, the letter read after the prefix.
3. **The frame-form names stay public in P7**, and P8 makes them private when it adds the raw rows (section 1).
4. **The `JtJp`-equals-`g` identity is per `K` element.**
