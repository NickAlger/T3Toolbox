# Types of the sampling variables

**Status.** A catalog asked for by the maintainer on 2026-11-15 (04:57): "Let's catalog the types for all edge
variables (jet vs dual jet), where are inner products vs dual pairings, and where any Riesz maps are occurring."
It is Claude's analysis of the explicit program in `scripts/review/jet_steps_check.py`. Nothing here is a
decision. It bears on target decisions 2.4 (what a tilde jet is), 2.5 (the misfit on Taylor coefficients), 2.3
(the derivative-mode adapter) and 3.6 (whether the loss is inside the sampling layer).

**Checked** by `scripts/review/type_catalog_check.py`; see section 7. Covered: the probe kind, ragged cores, one
sample axis, order 2. Not covered: apply and entries, the uniform layout, the `K` and `C` stacks.

**Names updated** to `SAMPLE_VARIABLES.md` (the notation decided on 2026-11-15, 23:23) on 2026-11-16. Sections 13
and 14 moved to `NOTATION_HISTORY.md`. Section 7 of `SAMPLE_VARIABLES.md` is the reference for types and shapes.

## 1. The spaces

Every variable is an element of (a space of the network) tensor (a space of the order).

| Space | What it is |
| --- | --- |
| `N` | The mode space at a site, of dimension `Ni` |
| `XD`, `XU` | The Tucker-rank spaces of the `D` family and of the `U` family of the frame |
| `AL`, `AR` | The bond spaces of the `L` family and of the `R` family |
| `R` | Jets of order `n`: polynomials in `s`, truncated above `s^n`. A ring |
| `R*` | Dual jets: linear functionals on `R` |

On the corewise frame `(B, G, G, G)` there is one Tucker-rank space and one bond space.

A vector of a space pairs with a covector of the same space, with no inner product. The cores are real: they
have no order factor. Each index of a core is a slot of one space, vector or covector, and every contraction
inside a term joins a vector with a covector slot.

## 2. The variables

| Grade | Variable | Network factor | Order factor |
| --- | --- | --- | --- |
| plain | `w` | vector of `N` | jet |
| plain | `xi` | vector of `XD` | jet |
| plain | `mu` | vector of `AL` | jet |
| plain | `nu` | covector of `AR` | jet |
| plain | `nu_L` | covector of `AL` | jet |
| plain | `eta` | covector of `XU` | jet |
| plain | `z`, the data `y`, the residual `r` | covector of `N` | jet |
| dot | `xi_U_dot` | vector of `XU` | jet |
| dot | `mu_R_dot` | vector of `AR` | jet |
| dot | `nu_L_dot` | covector of `AL` | jet |
| dot | `eta_D_dot` | covector of `XD` | jet |
| dot | `Jp` | covector of `N` | jet |
| tilde | `w_tilde` (the seed) | vector of `N` | dual jet |
| tilde | `xi_tilde` | vector of `XD` | dual jet |
| tilde | `mu_tilde` | vector of `AL` | dual jet |
| tilde | `nu_tilde` | covector of `AR` | dual jet |
| tilde | `nu_L_tilde` | covector of `AL` | dual jet |
| tilde | `eta_tilde` | covector of `XU` | dual jet |
| tilde-dot | `xi_tilde_dot` | vector of `XD` | dual jet |
| tilde-dot | `mu_tilde_dot` | vector of `AL` | dual jet |
| tilde-dot | `nu_L_tilde_dot` | covector of `AL` | dual jet |
| tilde-dot | `eta_D_tilde_dot` | covector of `XD` | dual jet |
| blocks | `V_tilde`, `H_tilde` | covector on the space of `V`, of `H` | none: real |
| blocks | `V_tilde_dot`, `H_tilde_dot` (`Kp`) | covector on the space of `B`, of `G` | none: real |

The dot rows are for the manifold frame with the direction `(V, H)`. The tilde-dot rows are for the base point
with the raw core direction, as `Kp` is computed; there the dot variables are `xi_dot` in `XD`, `mu_dot` in `AL`
(these two names meant the frame's dots until 2026-11-15, 23:23), `nu_L_dot` a covector of `AL`, and `eta_D_dot` a
covector of `XD`.

Two rules summarise the table.

- **The letter fixes the network factor's variance, in every grade.** `w`, `xi` and `mu` are vectors. `eta` and
  `nu`, in either basis, and `z` are covectors.
- **The grade fixes the order factor.** Plain and dot are jets. Tilde and tilde-dot are dual jets.

So the tilde changes the order factor and nothing else.

**Why, and where the reversal comes from.** A tilde variable is a multiplier of the constraint that defines its
partner. The multiplier of a variable in `S (x) R` lives in the dual, `S* (x) R*`. On every edge the two partners,
in the same basis, are in dual network spaces: `xi` and `eta_D`, `mu` and `nu_L` in the base point's bases;
`xi_U_dot` and `eta`, `mu_R_dot` and `nu` in the frame's other bases (`SAMPLE_VARIABLES.md` section 8); `w` and
`z`. So `S*` is the letter's own kind of space, and the network factor needs no identification at all. In the order
factor there is no such partner: both partners are jets, so the multiplier is a dual jet. The duality of the
network is absorbed by the partner rule. The duality of the order has nothing to absorb it. That leftover dual is
the reversal.

## 3. What happens to the order factor

| Where | Operation | In forward order | With representing jets |
| --- | --- | --- | --- |
| Every term of a plain or dot step | The product of the ring: `R x R -> R` | convolution | convolution |
| Every term of a tilde or tilde-dot step. Exactly one operand is a dual jet | The action of the ring on its dual: `R x R* -> R*` | correlation | convolution |
| The block readouts: `V_term` and `H_term` in the tilde and tilde-dot grades | A dual pairing, `R* x R -> real`: one dual jet against the product of the jets | sum over slots | top coefficient |
| The misfit `f`, `r_block_sumsq`, `Jp_sumsq` | An inner product on `R`, with the weights `omega^2` on its diagonal | sum over slots | same |
| The seed: `omega^2 r`, and `omega^2 Jp` for `JtJp` | The Riesz map of that inner product, `R -> R*` | multiply slot `t` by `omega[t]^2` | the same, then the flip |

The order index is contracted away only in the last three rows. Inside the sweeps it is never contracted.

The start values fit: a plain sweep starts from the unit of the ring (ones at order 0, zero above), and a tilde
sweep of the probe kind starts from the zero functional.

## 4. What happens to the network factor

| Where | Operation |
| --- | --- |
| Every contraction over a bond, a Tucker-rank or a mode index, inside every term | A dual pairing of a vector with a covector. No inner product |
| The sum over the samples at a block readout | A dual pairing over the sample index |
| The misfit `f`, `r_block_sumsq`, `Jp_sumsq` | The dot product on `N`, for each sample: an inner product |
| The seed | The Riesz map of that dot product: a covector of `N` (the residual) becomes a vector of `N` (the seed). The identity on the array |
| Outside the sampling layer: the orthogonality of the frame, the gauge projection, the lift, the retraction, the inner products of the Krylov solver | Inner products on the spaces of the cores |

Inside the sampling layer, no inner product of `XD`, `XU`, `AL` or `AR` is used anywhere.

## 5. The Riesz maps, and two maps that only look like one

| Map | From, to | What it is | Where |
| --- | --- | --- | --- |
| 1. The data Riesz map | residual or `Jp` (jet, covector of `N`) to seed (dual jet, vector of `N`) | The map of the misfit's inner product. Positive definite. On the array: multiply by `omega^2` | The seed step. The only Riesz map in the sampling layer. Used for `g` and for `JtJp` |
| 2. The core Riesz map | a block (covector on the cores) to a gradient (a vector of the cores) | The map of the dot product of the cores. The identity on the array | The fitting layer and the geometry: before the gauge projection |
| 3. The flip. Not a Riesz map | dual jet to representing jet | The isomorphism of `R*` with `R` given by the top-coefficient form. That form is symmetric and nondegenerate but not positive: it is not an inner product | Only if tilde variables are stored as representing jets: once after map 1 at the seed, and undone by the top coefficient at the readouts |
| 4. Taylor to derivative coordinates. Not a Riesz map | jet to jet, dual jet to dual jet | A change of basis. Jets are multiplied by `t!`, dual jets by `1/t!` | The derivative-mode adapter |

Two consequences for open questions.

- **2.5.** The misfit is exactly a choice of inner product on data jets. The weight is that choice.
- **2.3.** Derivative mode is another inner product of the same kind: the weights `omega[t]^2 t!^2`, with the data
  divided by `t!`. It needs nothing below the seed step.

## 6. At the API seam

| Object | Type | Remark |
| --- | --- | --- |
| The output of `probe` | jet, covector of `N` | |
| The residual argument of a transpose, `ztildes` | dual jet, vector of `N` | It pairs slot by slot with the output. That is a dual pairing written in the dual basis: forward order, and no metric. This is the maintainer's lean of 04:44 |
| `y` and `weight` given to `sampling_variables` | data of the type of `z`; a metric | With these, map 1 is applied inside the sampling layer. Whether it belongs there is 3.6 |
| A held tilde variable, such as `mu_tilde` | dual jet | Stored turned, as a representing jet (2.4, decided on 2026-11-15) |

## 7. The check

`scripts/review/type_catalog_check.py` runs the explicit program of `jet_steps_check.py`. A type is a rule of
transformation, so it can be tested.

- **Part A, the network factor.** Every space `N`, `XD`, `XU`, `AL`, `AR` gets an arbitrary invertible change of
  basis, with condition numbers up to 4. The frame, the directions, the input and the seed are transformed
  according to their types. Each of the 30 arrays of the program then agrees with the catalog's rule applied to the
  original (worst relative difference 1e-15, in every grade). This passes only if no inner product of a network
  space is used.
- **Part B, the one inner product.** With an arbitrary change of basis of `N`, copying `omega^2 r` into the seed
  is wrong by 7e-01, and the misfit changes by 2e-01. With an orthogonal one, both agree to 1e-15.
- **Part C, the order factor.** Under `s -> c s`, slot `t` of every jet is multiplied by `c^t` (4e-16). A
  functional in forward order has slot `t` multiplied by `c^-t`. The stored tilde variables, as representing
  jets, have slot `t` multiplied by `c^-n c^t`: they move like jets, up to the factor that the generator `[s^n]`
  carries (6e-16). The blocks are unchanged (4e-16).

## 8. The maintainer's picture: arrows on the edges (2026-11-15, 05:20)

The maintainer, in his words: "I am thinking about this like arrows on each edge, which can point either way.
Arrows can only be contracted into a core if they point towards it, and the thing that comes out the other side
is an arrow pointing away. On the internal edges, an object is neither plain nor dual. It is plain for the thing
it is pointing towards, and dual for the thing it is pointing away from. The notion of plain vs dual only takes a
definite meaning in the exterior, when something comes into or out from the network completely. There, the w are
plain objects (pointing in), and the z and r are dual objects (pointing out). The r is a dual object from the
perspective of the network and points the wrong way, and therefore cannot be put back into the network as is; it
must be converted to a plain object by a Riesz map. For the unjetted version, this means applying the weights.
For the jetted version, this also means reversing the jets."

Claude's reading of it against this catalog. Not a decision.

- **It corrects section 2 in one respect.** The words "vector" and "covector" on an internal edge are a choice of
  orientation, not a fact. What is a fact is that the two directions on an edge are dual to each other. The
  catalog's first rule is better said as: the letter fixes the direction in which a variable travels along its
  edge. Whether that is with the arrow of the edge or against it is fixed by the basis, once the cores carry
  arrows (section 9).
- **The order index is one more leg on every edge.** Its "core" is the product of the ring: the tensor that is 1
  when `t = i + j`. As the term functions use it, two jets point in and one points out. A residual in forward
  order points out on that leg. The reversal turns it round.
- **Where the order leg differs from a bond.** A TT core is used in all three orientations, by three term
  functions: `mu_term`, `nu_term` and `eta_term`. So no arrow on a bond ever has to be turned inside the network.
  The order leg has one function, the convolution, with one orientation. Stored reversed, that one function
  serves, because with the top-coefficient pairing the tensor becomes symmetric in its three legs: it is 1 when
  `i + j + k = n`. Stored in forward order, the order leg needs its second orientation, the correlation, exactly
  as a bond needs `nu_term` beside `mu_term`.
- **Weights and reversal are one map with two sources.** The weights come from the misfit, and are a choice. The
  reversal comes from the network's own rule for combining orders, and is the same for every misfit. Together
  they are the representative of the misfit's functional in the network's own pairing:
  `sum_t omega[t]^2 a[t] b[t] = [s^n] ( flip(omega^2 a) b )`. In that sense the whole map is a Riesz map, as the
  maintainer says; map 3 of section 5 is its part that does not depend on the misfit.
- **Under this picture the order factor has no definite type inside the network either.** With representing
  jets, every internal object is a jet flowing one way or the other. "Dual jet" in section 2 is a statement made
  from the exterior. Part C of the check agrees: the stored tilde variables move like jets under `s -> c s`, up
  to one scalar that belongs to the exterior readout.

## 9. Arrows on the cores: every contraction type checked (2026-11-15, 05:26)

The maintainer: "Maybe this can make more sense if the frame cores are also given orientations. For example, L
cores Laib have a: left-to-right arrow, i: up arrow, b: left-to-right arrow, and R cores Raib have right-to-left
instead for a,b. Then every contraction can be type checked."

Carried out in `scripts/review/arrow_type_check.py`. Claude's work, not a decision.

**The rule.** Every edge has one arrow. Every leg of every array, core or variable, is labelled with its edge
and a sign: `+` where the arrow enters the array, `-` where it leaves. A contraction is well typed when the two
legs are the two ends of one arrow: the same edge, opposite signs. A sum is well typed when the legs agree.

**The arrows of the frame.** Every arrow points away from its store and towards the hole.

| Core | Where it sits | Its legs |
| --- | --- | --- |
| `D` | below the Tucker edge | `o`: the bottom-to-top arrow enters. `i`: the bottom-to-top arrow leaves |
| `L` | left of the hole | `a`: enters. `i`: enters (bottom-to-top). `b`: leaves. Left-to-right |
| `R` | right of the hole | `b`: enters. `i`: enters (bottom-to-top). `a`: leaves. Right-to-left |
| `U` | above the Tucker edge | `a`: enters (left-to-right). `b`: enters (right-to-left). `i`: leaves, top-to-bottom |
| `H`, a hole | | every arrow enters |
| `V`, a hole | | every arrow enters |

The maintainer gave the rows for `L` and `R`. The rows for `D`, `U`, `V` and `H` are Claude's continuation of the
same rule.

**The arrows are the gauge conditions.** In each frame core, the legs where an arrow enters are the ones summed
in its orthogonality condition, and the one leg where the arrow leaves is the free one: `L` over `a, i`; `R` over
`i, b`; `D` over `o`; `U` over `a, b`. The sampling contractions do not use orthogonality, and they would type
check under any orientation of the edges. This orientation is the one that records what the geometry uses.

**The result.** The whole program type checks: 424 contractions over network indices and 176 sums, in the 22
step functions, at order 2 with `d = 4`. The seven term functions were replaced by typed ones; the step functions
were not touched.

**With the arrow, against the arrow.** With fixed arrows on the cores, the flow of the computation is a second
thing: a variable flows with the arrow of its edge or against it.

| Flows | Variables |
| --- | --- |
| With the arrows | `w`, `xi`, `mu`, `nu`, `eta`, and the tilde of each: `w_tilde`, `xi_tilde`, `mu_tilde`, `nu_tilde`, `eta_tilde`; the raw form's dots `xi_dot`, `mu_dot`, and `xi_tilde_dot`, `mu_tilde_dot` |
| Against the arrows | `nu_L`, `z`; the frame's form's dots `xi_U_dot`, `mu_R_dot`, `nu_L_dot`, `eta_D_dot`, and `Jp`; `nu_L_tilde`; the raw form's `nu_L_dot`, `eta_D_dot`, and `nu_L_tilde_dot`, `eta_D_tilde_dot` |

So what flows against the arrows is exactly what carries a basis in its name, and the outputs `z` and `Jp`;
everything else flows with them, the raw form's dots included. (Until 2026-11-16 this paragraph said that what
flows with the arrows has not yet met a hole, and what flows against them has passed a hole or is on its way out;
the raw form's dots, which flow with the arrows, are the exception to that.) Each block is the outer product of
variables that flow with the arrows into a hole, and its legs are the legs of the hole's direction with every sign
flipped: `V_tilde` fits `V`, `H_tilde` fits `H`, and the `Kp` blocks fit the raw core directions.

**One change to the first statement of the picture.** "Arrows can only be contracted into a core if they point
towards it" describes the flow. With fixed arrows on the cores it does not hold as a typing rule: `nu_L` enters
`L` through the leg where `L`'s arrow leaves. The typing rule is the two ends of one arrow.

**What the checker rejects.**

| Call | Why it is rejected |
| --- | --- |
| The residual put back as it is, `xi_term(D, r)` | On the mode edge both legs have sign `+`: the arrow points the wrong way |
| Weights applied but the order not reversed | The order leg is `out`: it cannot enter the product |
| `mu` swept through an `R` core | Two different edges, an `L` bond and an `R` bond |
| `nu` swept through an `L` core | Two different edges |
| `nu_L` where `nu` is wanted, in `eta_term(mu, U, nu_L)` | Two different edges |
| A bond variable from the neighbouring bond | Two different edges |
| `mu + mu_R_dot` | A sum of unlike types |

The first two rows are the maintainer's statement of 05:20: `r` points the wrong way and must be converted, by
the weights and, for jets, by the reversal.

**A naming hazard.** The library's names say where a store sits: the left cores are left of the hole, the down
cores are below the Tucker edge. The arrow of a store points the other way: the left cores carry left-to-right
arrows, the down cores carry bottom-to-top arrows. The library's CHANGELOG records that in 2026.4.0 "up" and "down" named
the direction the edges point, and that the meaning was then swapped to the position. Arrows in the documentation
would put that other meaning back beside the names.

## 10. The convention for the arrows (2026-11-15, 05:39)

**Agreed by the maintainer:** the change to the first rule in section 9. "I agree with the first rule change,
this is right." The typing rule is the two ends of one arrow; flowing with or against the arrow is a separate
property.

**The one convention, decided** (target decision 9.5). The maintainer's lean: "inward pointing arrows towards a
core should mean dual indices (contracting puts a vector 'in'), and outward facing arrows from a core should mean
plain (contracting pairs the mode with an external functional)." Claude agreed (below), and he confirmed at
05:47: "All good except the name suggestion."

**Claude's view: agree.** Said in one line: an arrow shows the way a vector travels, out of a plain index and
into a dual index.

- It is the convention of a matrix as a map: the input index is the dual one. `w` is plain and points in; `z` and
  `r` are dual and point out, as in section 8.
- It is what the checker of section 9 already encodes: `+` (the arrow enters) is a dual index, `-` (the arrow
  leaves) is a plain one. So the variables that flow with the arrows are plain, and those that flow against them
  are dual.
- It gives the gauge conditions one form. As a map along its arrows, from its dual legs to its plain leg, every
  frame core `M` satisfies `M M^T = I`. Checked on the library's own four orthogonalization methods in
  `scripts/review/gauge_arrow_check.py`: `D` 9e-16, `L` 4e-16, `R` 9e-16, `U` 7e-16. Section 9 stated this from a
  review note; it is now tested.
- "Plain" and "dual" here are about the spaces of the edges. A direction `H` has three dual indices and is still
  a vector of the space of the cores; its block `H_tilde` has three plain indices and is a covector of that space.

**Is it conventional?** It is the convention of index notation and of tensors read as maps from a domain to a
codomain, with arrows running from the domain to the codomain; TensorKit.jl is built on that reading. It is not
universal. TeNPy's documentation labels the other way round: a leg that goes into the tensor stands for a vector
space, and one that goes out for the dual. So the convention has to be stated once, wherever arrows are used.

**The names.** Claude suggested letting the four direction words belong to the names only, and describing
arrows only relative to the hole. **Rejected by the maintainer** (05:47): "Describing directionality with respect
to diagrams (ascii in the documentation here and tikz in papers) is immensely valuable to humans, even if there
is the risk of misinterpreting or switching it (as I have already done). It's a risk that is worth it in my
opinion." So arrows are described by the way they point in the diagram, and the names keep saying where a store
sits.

**Forms on edges; parked for another arc.** The maintainer: "The diagonal weight matrices between cores on a
normal T3 could be given meaning as (pseudo-)inner products, and the plain T3 is interpreted as having secret
identity matrices along each internal edge. This suggests that orthogonality for weighted networks should be done
with respect to weighted inner products, though this is a task for another arc."

- In the typed picture a form on an edge is a tensor with two legs of the same kind: two dual legs for a form on
  vectors, two plain legs for a form on functionals. It is what makes a contraction of two like legs well typed.
- Three such forms are already in this catalog: the weights `omega^2` on the exterior mode legs (an inner
  product), the flip on the order leg (a pseudo-inner product: nondegenerate, not positive), and the identities
  hidden in each `M M^T = I`.
- The orthogonality conditions are where forms on edges are used. The sampling contractions do not use them
  (section 4). So weights on the internal edges would change the gauge conditions and leave the sampling terms as
  they are.
- Parked. Orthogonalization is outside the scope of this refactor (scope guidance 1.1).

**The wording for the direction of an arrow** (the maintainer, 05:49 and 05:50): "left-to-right" and
"right-to-left" along a bond, "bottom-to-top" and "top-to-bottom" along a Tucker edge. Not "rightward", "leftward",
"up" or "down": those four words alone stay with the stores. The tables of section 9 are written this way.

## 11. The tilde jet as a multiplier; arrows of the base point (2026-11-15, 15:49)

The maintainer: "Regarding the tilde jet; I think we have a clear picture now. It is an adjoint variable with
respect to a Lagrangian, and the reversal of order within the jet comes in because it is a dual object which
points the opposite way on its edge. Are there any more subtleties here I'm not seeing?"

**Six points Claude raised in answer.** None changes the picture.

1. **"The opposite way" is on the order leg only.** On its network edge a tilde variable points the same way as
   its letter: it is opposite to its partner, whose multiplier it is. On the order leg both partners are jets, so
   the multiplier points against them there. That is the reversal.
2. **Which Lagrangian.** On the manifold frame the tilde sweeps are the adjoint of the linear program
   `(V, H) -> Jp`, whose variables are the dot variables of the frame's form. So the tilde of a letter is the
   multiplier of the dot of its partner: `xi_tilde` of `eta_D_dot`, `eta_tilde` of `xi_U_dot`, `mu_tilde` of
   `nu_L_dot`, `nu_tilde` of `mu_R_dot`. The library's present names say the same (`deta_tilde`, `dxi_tilde`).
   Section 9 shows it as arrows: the tilde variables flow with the arrows and the dot variables of the frame's
   form against them. On the corewise frame, for the gradient of the value, the partner is the plain variable
   (`mu_tilde` of `nu_L`, which a call on the corewise frame itself calls `nu`), which is what
   `tilde_adjoint_check.py` checked at order 0. Checked now with jets in
   `scripts/review/tilde_multiplier_check.py` at order 3: perturbing each dot variable changes
   `F = <phi, Jp>` by the top-coefficient pairing with its tilde variable (relative error from 5e-15 to 1e-11).
   The slot-by-slot pairing is wrong (errors from 0.2 to 30).
3. **Two arrays where there was one name.** The functional at the API seam is in forward order. The seed inside
   is that functional turned round. `w_tilde` has to name one of them.
4. **A tilde jet is tied to the order.** A plain jet of order `n`, cut at order `m`, is the plain jet of order
   `m`. A tilde jet is not: when the order changes, every slot moves. This bears on 6.5.
5. **The weights are indexed in forward order.** After the turn, slot `t` of the seed carries `omega[n-t]^2`.
   It matters only to something that reads a tilde variable slot by slot.
6. **The tilde of an affine jet is not affine.** `xi` has two coefficients; `xi_tilde` has all of them. So a
   tilde step has full products where a plain step has two-term ones. This belongs to 2.9.

**Arrows of the base point.** The maintainer: for `D, U, L, R, V, H` the directions are clear. "For the basepoint
cores, we can either have all arrows pointing inward and imagining that there is a hidden identity matrix dual
pseudo-inner product on each edge, or we can choose a gauge convention which is used unless otherwise stated. I
think the gauge can be bottom to top and left to right, so that all edges are pointing towards the last TT core.
There is a clear known exception to gauge correctness, which is the last L core and first R core; these are
inward pointing like H."

Claude's view: the gauge convention, bottom-to-top and left-to-right.

- It is the rows for `D` and `L` in section 9: the base point is `(D, L)`. The checker of section 9 already types
  the raw core directions this way, and the whole `Kp` family type checks with it.
- The other choice, a hidden form on every edge, is the picture for the weighted networks that are parked.
- **Typing and gauge correctness are two things.** The last `L` core and the first `R` core are typed like every
  other `L` and `R` core. What fails for them is `M M^T = I`: they carry the norm. Checked in
  `gauge_arrow_check.py`: both are off by 2e+03 on a random tensor, where the others hold to 1e-15.
- "Inward pointing like `H`" is exact when the tail has dimension 1, because the leg where the arrow leaves is
  then trivial. With a tail of dimension above 1 that leg is real and is closed by the boundary vector of ones.
  Tails are question 4.4.
- Not checked: whether a padded frame on the uniform layout is a further exception, with `M M^T` a projector.

## 12. Target decision 2.4, decided (2026-11-15, 16:10)

The maintainer's statement: the residual is the same type of object as the output `z`; the user supplies the data
in the order in which the function would have produced it; the dualization comes from the misfit. With three
precisions he agreed to:

- **The inner product of the misfit has no reversal.** It is `sum_t omega[t]^2 r[t]^2`, slot by slot. The
  reversed form `[s^n](r r)` is not positive. The reversal is in the map from `r` to the seed, which writes the
  functional `<r, .>` as a jet that can enter the network: `flip(omega^2 r)`.
- **The sign is the library's:** `r = S(x) - data`.
- **Two seams.** In fitting only the data and the weight cross, both of the type of `z`. In the frontend
  transposes the argument is a residual of the type of `z` in forward order; the function dualizes it by the
  plain inner product and the turn. `w_tilde` names the seed.

Every tilde variable is stored turned, including those a user holds, and pairs with a jet by the top coefficient
of the product. In the words of section 5: map 1 and map 3 are applied together, once, at the seed.

## 13. The letters by edge, and the variables without names (2026-11-15, 22:11)

Deprecated. Claude's analysis of options A and B for the dots of `xi` and `mu` along the raw core direction stood
here. Superseded on 2026-11-15 at 23:23 by target decision 9.6; the section is now `NOTATION_HISTORY.md` section 5.

## 14. A variable is a contracted subnetwork in a basis; one dot (2026-11-15, 22:24 to 22:27)

Deprecated. Claude's analysis of the naming from first principles, with the second scheme written out at 22:41,
stood here. Superseded on 2026-11-15 at 23:23 by target decision 9.6, which took that scheme; the parts adopted are
in `SAMPLE_VARIABLES.md`. The section is now `NOTATION_HISTORY.md` section 6.
