# P8 design review: the implementer's note on Engine D

Reviewer, 2026-11-20, on `P8_DESIGN_NOTE.md` (read only; nothing run). Against `SAMPLE_VARIABLES.md` sections 5.2,
5.4, 6 and 9, the plan's sections 2.2 to 2.4 (the oracle's `query`, O-12, O-13) and P8 to P11, and rows 4.5, 8.2 and
8.3. Weighed by simplicity and performance, with values equal to rounding taken as the same.

**Overall: sound and well reasoned.** The mathematics of the tilde-dot grade is right, term by term. The corewise
graph, derived by drop and rename, does give "each sweep once" and O-13 by construction.

One finding the derivation must handle: P7's `needs={'variations'}` on `JtJp` would make `JtJp` unreachable on a
corewise call.

The open point (the direction on a corewise call) does not change a decided row. It does change the plan's sketch
of the oracle (section 2.4) and P14, and it should be settled there by one sentence. I recommend the note's option,
with the mapping in the least-squares oracle.

---

## 1. The tilde-dot grade: agree. The mathematics is right.

**`mu_tilde_dot`'s six terms.** A tilde-dot variable is the sum over the place of the direction and the place of the
seed. At the step to bond `i + 1`:
- the direction is in the carry, on the core of site `i` (`L_dot[i]`), or in the leaf of site `i` (`xi_dot[i]`);
- the seed is in the carry or in the leaf of site `i` (`xi_tilde[i]`).

That is 3 × 2 = 6 places. When both are in the leaf of site `i`, it is `D_dot[i]` applied to the seed, which is
`xi_tilde_dot[i]`, the one product that is not of two known variables. The six:

| Direction in | Seed in | Term |
| --- | --- | --- |
| the carry | the carry | `(mu_tilde_dot, L, xi)` |
| the carry | the leaf | `(mu_dot, L, xi_tilde)` |
| the leaf | the carry | `(mu_tilde, L, xi_dot)` |
| the leaf | the leaf | `(mu, L, xi_tilde_dot)` |
| the core | the carry | `(mu_tilde, L_dot, xi)` |
| the core | the leaf | `(mu, L_dot, xi_tilde)` |

That is section 5.4, exactly. `nu_L_tilde_dot` is the mirror with `nu_L`, `_raw.nu_L_dot` and `nu_L_tilde`. `K` sits
on the dotted operand, which gives the three `mu` strings with `K` on the carry, the core and the leaf. All are in
`STRINGS`.

**`xi_tilde_dot`** is `'JWCo,KCio->JWKCi'`, `K` on `D_dot` and none on the residual's seed. Right, and it is a
string of the 29 (the `xi` seed family). A new step is justified, since `xi_tilde_step` puts `K` on the seed.

**`H_tilde_dot`'s nine readouts.** The block with `L[i]` taken out: the tilde on one of `(mu, xi, nu_L)` and the dot on
one of them.
- Three with both on one variable: `mu_tilde_dot`, `xi_tilde_dot`, `nu_L_tilde_dot`.
- Six with them on two.
- None with the direction on the core removed. At site `i` "the direction can only be `D_dot[i]`" (section 6), which
  is in `xi_dot` and `xi_tilde_dot`.

So nine (section 6). `K` on the dotted operand gives the three `H` strings with `K`.

**`V_tilde_dot`** is the corner of `V_term(eta_D_tilde_dot, w) + V_term(eta_D_dot, w_tilde)`: the first through the
short node at the corner `a - 1`, as P6; the second `'JWKCi,JWCo->KCio'`, `K` on the raw dot, none on the seed.
Right (section 6).

**The short node of `eta_D_tilde_dot`.**
- Each kept slot is the six `eta_term`s at a lowered corner. All six are full × full products.
- The readouts keep `W`, with `K` on the first operand, the core or the third. That is P6's two literals and the new
  `'JWCa,KCaib,JWCb->WKCi'`.

**The cost claim holds:**

| Form | Contractions per site | Slot-pairs per site |
| --- | --- | --- |
| full `eta_D_tilde_dot`, `'loop'` | `6(n + 1)` | `6(n + 1)(n + 2)/2` |
| full `eta_D_tilde_dot`, `'scan'` | `6(n + 1)` | `6(n + 1)²` |
| short node, a line | `6 · 2 = 12` readouts | `6(2n + 1)` |

So the saving is larger than P6's, at every order above 1, as the note says.

## 2. O-12 with both directions, and `K`: agree, with the "same `K`" rule made explicit

- **The two route rows of each shared name**, raw in front (needing `core_directions`) and frame behind (needing
  `variations`), are what I proposed for P7. With both arguments the public names are raw, which is O-12. `Jp`,
  `Jp_norm_sq` and `JtJp` read `_frame.*`, and `Kp` reads `_raw.*`. Both families run once each.
- **The raw route of `Jp`,** `z_term(_raw.eta_D_dot, D) + z_term(eta_D, D_dot)`, is the derivative of
  `z = z_term(eta_D, D)` along `(D_dot, L_dot)` at the base point. Right, and needed by the corewise call.
  - On a manifold call it is reached only without `variations`.
  - It is then the raw direction's output. That is legitimate (it is `Jp_base` of `subtree_gauge_check.py`) and
    useful to a user.
- **`JtJp` on a manifold call with only `core_directions`** stays unreachable, by P7's `needs={'variations'}`.
  - That is right: `JtJp`'s blocks are the frame's (`U`-basis `V_tilde`, `H` in `L`/`R`).
  - Seeding them from a raw `Jp` would give `J_frameᵀ(ω² J_raw p)`, a mixed product with no use. State it in the
    docstring.
- **One `K` for the call, shared by `V`, `H`, `D_dot` and `L_dot`.** Agree, and say why in the plan:
  - the record has one `K`, and the curvature term pairs element `k` of the direction with element `k` of its lift
    (the exact Hessian `JtJp + Kp` applied to one stack of directions);
  - two stacks of different `K` in one call have no meaning the oracle uses;
  - so a mismatch is a structural error, with the operands named. P7's check, extended to the four.

  A user who wants different stacks makes two calls.

## 3. The corewise call: agree with the derivation, with one finding and two safeguards

**"Each sweep once" by construction: yes.**
- Dropping every row that reads `U`, `R`, `V`, `H`, `Vc` or a `_frame.*` node, transitively, removes the manifold's
  `nu` (through `R`), `eta` (through `U`), `nu_tilde`, `eta_tilde`, `_eta_tilde_top`, `xi_U_dot` and `mu_R_dot`, and
  the routes reading them.
- The rename then maps `nu_L` to `nu`, `eta_D` to `eta`, and so on, into names nothing occupies any more.
- So the corewise graph has one `nu` sweep (through `L = G`), one `mu`, and one family per grade.

**O-13 falls out: yes.** `nu_L`, `eta_D_dot` and `xi_U_dot` are not made by any corewise row, so they are unknown names
there, and section 9's basis-free names are exactly the corewise graph's public names.

**Finding A (must handle): rewrite `needs` as well as names and reads.**
- P7 gave the twin `JtJp` `needs={'variations'}`, so that its missing-argument message is clear. On a corewise call
  the direction comes as `core_directions` (section 4), so the derived `JtJp` would need an argument the call never
  has: `JtJp` would be unreachable.
- The same holds for any row whose `needs` names `variations` but which survives the drop: the public frame routes
  are dropped, but `JtJp` reads no `V` or `H` directly, so it survives.
- *Resolve:* the derivation maps `needs` too, `'variations'` to `'core_directions'`, in the same declared table. Add
  a test that every public corewise name is reachable on a corewise call with `core_directions` and `y`.

**Safeguard 1: the rename must not collide.** After the drop, no surviving row may already have a name that is a
rename target (`nu`, `eta`, `nu_tilde`, `eta_tilde`, `_eta_tilde_top`, ...). Today the drop guarantees it, but a
later row (P9's apply kind, a new route) could break it silently. Make `_corewise` assert it at import, with the
colliding name, and a test that the derived graph has each name's rows from one source.

**Safeguard 2: the twins after the rename.** P7's `_JP_PATH` lists manifold names (`nu_tilde`, `_eta_tilde_top`).
After the rename, the corewise rows with those names are the renamed `nu_L_tilde` and `_eta_D_tilde_top`, which is
what the corewise adjoint needs. But it holds by coincidence of spelling.
- Derive the corewise twins from the corewise graph with the path read through the same rename, or define the path
  by role (the seed, the tilde sweeps, the short node, the blocks) rather than by name.
- A test that the corewise `JtJp` equals the corewise `g` with `Jp` held as `r`, per `K` element, catches either way
  going wrong.

**The corewise routes of `V_tilde` and `H_tilde`** (needing `{'_corewise'}`, placed first in the manifold graph, renamed
into the corewise `g`) are the right mechanism: the same as P5's call-type facts. A manifold call never takes them.

## 4. The open point: the direction on a corewise call

**What the note proposes.** On a corewise call the direction is `core_directions` only; `variations` raises; the
least-squares oracle passes the corewise geometry's tangent as `core_directions`.

**Does it change a decided row? No.**
- Row 4.5 says the sampling layer "takes the lifted direction `(D_dot, L_dot)` for `Kp`" and never knows the
  geometry. That is unchanged.
- Row 8.2 says the oracle returns its values "in the engine's vocabulary". Unchanged.
- Row 8.3 fixes the optimizers' contract and the five phases, not the oracle's arguments. Unchanged.

**What it does change is the plan.**
- Section 2.4's sketch of `Oracle.query` comments `variations: (V, H): JtJp, Jp_norm_sq, Jp` and
  `core_directions: Kp`, for every geometry.
- The geometry-generic `LocalModel` (P16) passes the tangent `p` as `variations` and its lift as `core_directions`.

On the corewise geometry, then:
- the tangent is the raw direction `(dB, dG)` itself;
- the lift is the identity;
- the frame form on `(B, G, G, G)` (`U = R = L = G`, `V = dB`, `H = dG`) *is* the raw form.

So today's library passes the corewise tangent as variations at the frame `(B, G, G, G)`, and gets the right numbers.

**The options:**
- **(a) The note's:** the engine refuses `variations` on a corewise call, and `LeastSquaresOracle.query` maps it: on
  a corewise frame (two stores) it passes the direction as `core_directions`. The geometry-generic `LocalModel` and
  the optimizers change nothing, and in the engine one argument has one meaning. The cost is one rule in the oracle,
  which already passes the frame through and can read its structure (layout first, then the count, as the engine
  does).
- **(b)** The engine accepts `variations` on a corewise call as the raw direction (the corewise frame's form is the
  raw form). No mapping anywhere. But a call with both arguments then needs two raw families (one per argument) to
  keep `Jp` along `variations` and `Kp` along `core_directions`. Those are the same array for the corewise lift, but
  the engine cannot know it without an identity test.
- **(c)** `LocalModel` passes the direction by geometry. That puts the call type in the optimizer layer, against
  8.3's geometry-agnostic optimizers.

**I recommend (a).** It is the simplest in the engine, it keeps O-12's "one name, one meaning" sharp, and the
mapping lives in the one oracle that is built on the engine.
- *Write it in the plan's section 2.4*, under `query`: "on a corewise frame the oracle passes the direction as
  `core_directions`: the point has one form, the raw one; the lift of a corewise geometry is the identity".
- *Add it to P14's tests:* a corewise `JtJp` and `Kp` from one query.

It is a plan change of the size Claude rules on, not a decided row. If the coordinator wants the maintainer's eye on
it, the question is: "on a corewise point, is the direction argument of the engine `core_directions` (the oracle
maps it), or `variations` read as raw?"

**The engine's message** when `variations` is given on a corewise call should name the fix: "a corewise point has
one form of the direction: pass it as `core_directions`".

## 5. The mask attach point `_DIRECTION_STORE`: agree, and make it cover every direction operand

`{'D_dot': 'D', 'L_dot': 'L'}` says what the plan's step 2 requires: `core_directions` takes the frame's `D` and `L`
masks, not the point's (a superset at TT slack). On a corewise call the store is the point, which is the frame there.

*Suggest:* one table for every input with a direction, so P11 has a single place:
- `V` and `H` map to the frame's variation masks (`ufv_variation_masks`), `D_dot` and `L_dot` to the stores' masks;
- the centers `Vc` (the variation mask of `V`) are in the same table.

A test then checks that every argument a row reads that is core-shaped appears in it.

## 6. The tests: agree; the bridge's conversion, derived

**The bridge.** The old `corewise_hessian(core_directions, c, (ww, pp), frame, sweep, (D, L))` is the second derivative
in the cores, along `core_directions`, of `Φ(x) = Σ_t <c_t, S^(t)(x)>`, with `S^(t)` the old derivative outputs,
`S^(t) = t! z_t(x)`. So `Φ = Σ_t t! <c_t, z_t>`.

With `c_t = ω_t² r_t / t!`, in forward order, `Φ = Σ_t ω_t² <r_t, z_t(x)>`, `r` held. That is the engine's
`<seed, z(x)>` with the seed held (the corner pairing of the turned seed is the forward weighted sum). `Kp` is that
second derivative applied to `(D_dot, L_dot)`. So the two agree exactly in the mathematics, by the same step as `g`
in P4c and P6.

The note's "to be confirmed by script" is right as a check; the conversion is not a fit.

**`Kp` against a central difference of `g` (corewise, seed held).**
- `g` with the seed held is the gradient of `<seed, z(x)>`. `z` has degree `d` in the cores, so `g` has degree
  `d - 1` along a line in the cores.
- A central difference is exact (to rounding) only up to degree 2. For `d ≥ 4` it has an `O(h²)` error.
- Use either a Richardson pair (`h` and `h/2`), or `d = 3` problems for the exact comparison and a tolerance scaled
  by `h²` for `d = 4`. Otherwise the test will be loose or flaky.

**The other identities** (corewise at `(D, L)` against the manifold call; raw against frame along the undamped lift of
a gauged `(V, H)`) are the plan's, and right.

**Counts.** A corewise `('z', 'g')` runs each of `xi`, `mu`, `nu`, `eta` `d` times. Add `('Kp',)` on a corewise call:
one `nu` sweep, the raw dot sweeps once, no frame rows. That pins "each sweep once" for the largest closure.

## 7. Other points

- **`HeldVariables`.** The corewise-only names (`nu_dot`, `eta_dot`, `nu_tilde_dot`, `eta_tilde_dot`) as slots: right,
  by "every public computed variable of every graph".
  - Their shapes come from the base name's table entry with the corewise sizes. `nu` on a corewise call reads `rR`
    from `G`, which is `rL`, as the note says.
  - Pin that a held corewise `nu` of the `L` sizes passes and one of another size raises.
- **P9.** Apply's `nu_tilde[d]` starts at the seed (P6 review finding A, ruled). On a corewise call that is the
  renamed `nu_L_tilde`'s start. Make sure the rename carries a row's `start` field when it names a node.
- **Dtype.** `D_dot` and `L_dot` count when read (O-11). Nothing new.
- **What I will check on the built P8:**
  - every new name per `K` element against the reference;
  - `Kp` against my own dense second derivative: the dense objective `<ω² r, probe(T(x + ε x_dot))>` by my
    roots-of-unity oracle, by a Richardson second difference, or by the coordinate Hessian on a `d = 3` problem;
  - the corewise call at `(D, L)` against the manifold call;
  - the raw and frame forms along a lift;
  - the step counts for corewise `('Kp',)`;
  - mutants: one of the nine `H` terms dropped, `xi_tilde_dot` with `K` on the seed, the short node's third literal
    swapped, the rename table missing an entry, `needs` not rewritten.
