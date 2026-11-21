# P8 (Engine D): design note

Implementer, 2026-11-20. This note covers only what is new on top of P7. Nothing has been run. **[open]** marks a
choice for the orchestrator.

## 1. The rows

**Raw dots.** All have `carries_K=True` and need `{'core_directions'}`. `D_dot` and `L_dot` are two new arguments.
They reuse P7's steps, which were written for both forms:

| Row | Driver, start | Step, reads |
| --- | --- | --- |
| `xi_dot` | site | `xi_dot_step`: `D_dot`, `w` (a leaf) |
| `mu_dot` | left, zeros | `mu_dot_step`: `L`, `mu`, `xi`, `xi_dot`, `L`, `L_dot` |
| `_raw.nu_L_dot` | right, zeros | `nu_dot_step`: `L`, `nu_L`, `xi`, `xi_dot`, `L`, `L_dot` |
| `_raw.eta_D_dot` | site | `eta_dot_step`: `mu_dot`, `L`, `mu`, `L_dot`, `L`, `_raw.nu_L_dot`, `nu_L` |

These are the terms of `SAMPLE_VARIABLES.md` section 5.2, raw table: the direction is `L_dot` over `D`, or `L` over
`D_dot`, and every other core is `L`.

**The `L`-basis tilde rows** (`carries_K=False`, P6's steps):
- `nu_L_tilde`: right, zeros, `nu_tilde_step` over `L`, `nu_L`, `xi`, `xi_tilde`;
- `eta_D_tilde`: `eta_tilde_step` over `mu_tilde`, `mu`, `L`, `nu_L_tilde`, `nu_L`, full, run when wanted;
- its short node `_eta_D_tilde_top`, `eta_tilde_top_step` over the same, read by the corewise gradient (section 3).

**The tilde-dot rows** (`carries_K=True`, needing `core_directions`; the six terms of section 5.4, three new steps):
- `xi_tilde_dot`: `xi_tilde_dot_step`, `'JWCo,KCio->JWKCi'` over `D_dot`, `w_tilde`. A new step: `K` sits on the core,
  not on the seed.
- `mu_tilde_dot`: left, zeros, `mu_tilde_dot_step`. Its six terms are `mu_term` of `(mu_tilde_dot, L, xi)`,
  `(mu_dot, L, xi_tilde)`, `(mu_tilde, L, xi_dot)`, `(mu, L, xi_tilde_dot)`, `(mu_tilde, L_dot, xi)` and
  `(mu, L_dot, xi_tilde)`. `K` is on the dotted operand, so three strings of the `mu` family, all in `STRINGS`.
- `nu_L_tilde_dot`: right, zeros, `nu_tilde_dot_step`, the six mirror terms with `nu_L`, `_raw.nu_L_dot` and
  `nu_L_tilde`.
- `eta_D_tilde_dot`: `eta_tilde_dot_step`, six `eta_term`s, full, run when wanted.
- `_eta_D_tilde_dot_top`: the short node (section 5).
- `V_tilde_dot`: the corner of `V_term(_eta_D_tilde_dot_top, w)`, read at `a - 1` as in P6, plus
  `V_term(_raw.eta_D_dot, w_tilde)` with the string `'JWKCi,JWCo->KCio'`.
- `H_tilde_dot`: nine readouts of `(mu, xi, nu_L)`, the tilde on one and the dot on one: three with both on one
  variable (its tilde-dot), six with them on two. `K` is on the dotted operand, so the three `H` strings with `K`.
- `Kp`: `g_step` over the two blocks.

The raw tilde-dot rows have no twins: `Kp` is seeded by the residual only, row 3.3.

## 2. O-12 in practice

The public routes become, in order (the precedence):

```
Rule('nu_L_dot', identity, ('_raw.nu_L_dot',), {'core_directions'}, True)     # P8, in front
Rule('nu_L_dot', identity, ('_frame.nu_L_dot',), {'variations'}, True)        # P7
```

and the same for `eta_D_dot`.

**A call with both directions:**
- `Jp`, `Jp_norm_sq` and `JtJp` read `_frame.*`, the frame form along `(V, H)`;
- the public `nu_L_dot` and `eta_D_dot` are the raw form along `(D_dot, L_dot)`;
- `Kp` reads `_raw.*`.

So both families of sweeps run, once each. With `core_directions` alone, the public names are raw, and a want of
`Jp` falls to the raw route below.

**A raw route for `Jp`** (my addition, needed by section 3): `Jp` gains a second row,
`z_term(_raw.eta_D_dot, D) + z_term(eta_D, D_dot)`, needing `core_directions`, after the frame route. That is the
base point's direction (`subtree_gauge_check.py`'s `Jp_base`). On a manifold call it is reached only without
`variations`.

**The `K` of the call** is read off `V[0]`, or off `D_dot[0]` when there are no `variations`. All of `V`, `H`, `D_dot`
and `L_dot` must share it, by P7's check extended. Two direction stacks of different `K` in one call raise.

## 3. The corewise call

**Structure.** A ragged frame of two stores, `(B, G)`, binds the stores `D = B` and `L = G`. There are no `U` or `R`,
and `present` gets `'_corewise'`. The sizes the shapes read come from them: `rR` and `rL` from `G`, `nU` and `nD`
from `B`.

**The graph is derived:** `_corewise(graph)` is applied once at import:
1. drop every row that reads `U`, `R`, `V`, `H` or `Vc`, a `_frame.*` node, or a row dropped before it (transitively),
   or that needs `'_manifold'`;
2. rename the names and the reads by one declared table:

| On a manifold call | On a corewise call |
| --- | --- |
| `nu_L` | `nu` |
| `eta_D` | `eta` |
| `nu_L_tilde` | `nu_tilde` |
| `eta_D_tilde` | `eta_tilde` |
| `nu_L_dot` | `nu_dot` |
| `eta_D_dot` | `eta_dot` |
| `nu_L_tilde_dot` | `nu_tilde_dot` |
| `eta_D_tilde_dot` | `eta_tilde_dot` |
| `_raw.nu_L_dot` | `_raw.nu_dot` |
| `_raw.eta_D_dot` | `_raw.eta_dot` |
| `_eta_D_tilde_top` | `_eta_tilde_top` |
| `_eta_D_tilde_dot_top` | `_eta_tilde_dot_top` |

3. derive the `Jp` twins from the result, by P7's rule.

**The routes that differ** are rows of the manifold graph needing `{'_corewise'}`, placed first:
- `V_tilde` reads `_eta_D_tilde_top`, `eta_D`, `w`, `w_tilde`;
- `H_tilde` reads `mu_tilde`, `mu`, `xi_tilde`, `xi`, `nu_L_tilde`, `nu_L`.

A manifold call never takes them; the derivation renames them into the corewise `g`.

**What follows from the derivation:**
- **One sweep each.** The corewise graph has one `nu`, the renamed `nu_L`, and the manifold `nu`, through `R`, is
  dropped.
- **The names a corewise call accepts** are its public made names: section 9's table with no basis in any name. The
  frame forms (`xi_U_dot`, `mu_R_dot`) are dropped, and `nu_L` or `eta_D_dot` are unknown names, so O-13 falls out
  of the derived graph.
- **Held slots and shapes.** A corewise slot is the base name's: `nu`, `nu_dot`, `nu_tilde_dot`, ... The
  corewise-only names (`nu_dot`, `eta_dot`, `nu_tilde_dot`, `eta_tilde_dot`) become slots of `HeldVariables`. The
  slot rule is then "every public computed variable of every graph".

**[open] The direction on a corewise call.** With the frame-form rows dropped, `Jp`, `JtJp`, the dots and `Kp` all read
`core_directions`. A corewise call given `variations` raises ("a corewise point has no frame form; pass the
direction as `core_directions`"). The least-squares oracle (P14) then passes the corewise geometry's tangent as
`core_directions`. The alternative, accepting `variations` as the raw direction on a corewise call, gives one argument
two meanings. I recommend the first.

## 4. Masks (P11)

P8 leaves P11 one table, `_DIRECTION_STORE = {'D_dot': 'D', 'L_dot': 'L'}`, beside the inputs. P11 masks each
input with the masks of the store it names: the frame's `D` and `L` on a manifold call, the point's `B` and `G` on a
corewise call, which is the same object there. It re-masks `Kp`'s blocks with the same table. Nothing in P8 reads it
but a test that it names a store.

## 5. `Kp`'s readouts, the short form, the cost

- **The corner, as `g`'s.** `V_tilde_dot` reads only the top `a` slots of `eta_D_tilde_dot`, as `V_tilde` does. Its
  short node is worth it more than P6's: six full products per site against `2 · 6 · prod(a)` readouts. For a line at
  order `n`, `6(n + 1)` contractions under `'scan'` against 12. It needs one new literal for `K` on the core,
  `'JWCa,KCaib,JWCb->WKCi'`; the other `K` placements are P6's two.
- **The cost** against today's two-hole kernel (`t3_probe_corewise_derivatives_hessian`), which also forms only two
  rows of its `dxi_tilde_dot`:
  - the same term counts: `6 + 6` tilde-dot sweep terms, `6 · a` for the short node, and 9 three-operand readouts for
    `H_tilde_dot` per site;
  - each term at P1's contraction counts.

  I expect parity at a line input. It is not measured: the benchmark is Part 2's.

## 6. The tests

- **By definition:** every new name against the reference (its `Kp`: finite differences on numpy), on the jet
  matrix, at `K = ()` and `K = (2,)` element by element, on a manifold frame with both directions and with each alone.
  The corewise names against the reference on `(B, G, G, G)`, through section 9's table.
- **Identities, to rounding:**
  - a corewise call at `(D, L)` against the manifold call: `nu` = `nu_L`, `z` = `z`, `g` = the corewise `g`;
  - along the undamped lift of a gauged `(V, H)` on a library frame (`subtree_gauge_check.py`'s construction): the raw
    and frame `nu_L_dot`, `eta_D_dot` and `Jp` are equal;
  - `Kp` against a central difference of `g` on the corewise call, along `(B_dot, G_dot)` with the seed held: the
    gradient with `w_tilde` held at `x ± h·x_dot`.
- **Counts:** a corewise call wanting `('z', 'g')` runs `xi_step`, `mu_step`, `nu_step` and `eta_step` `d` times each,
  one sweep of each. A corewise want of `nu_L`, `eta_D_dot` or `xi_U_dot` raises as unknown.
- **The structure:** `_corewise` against the rename table; the corewise graph has no reads of `U` or `R`; every
  public corewise name is a slot.
- **The bridge:** `ProbeDerivativesKind(weight=None).corewise_hessian(core_directions, c, (ww, pp), frame, sweep,
  (D, L))` with the forward `c_t = weight_t^2 r_t / t!` equals the new `Kp`. The old objective
  `sum_t c_t S^(t) = sum_t weight_t^2 r_t S_t` is the new one, so the second derivatives agree. To be confirmed by
  script.

## 7. Changes to the plan

1. A raw route for `Jp` (section 2), needed by the corewise call.
2. On a corewise call the direction is `core_directions` only (section 3, [open]).
3. Short nodes for `eta_D_tilde` (read by the corewise gradient) and for `eta_D_tilde_dot` (read by `Kp`), with one
   new literal.
4. The corewise graph derived at import from the manifold one: the rename table above, and the rows needing
   `'_corewise'`. The stores `D` and `L` are bound to `B` and `G`, so no store is renamed.
