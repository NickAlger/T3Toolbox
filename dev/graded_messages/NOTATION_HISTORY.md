# Notation history: deprecated names and naming ideas

> **Deprecated. Everything in this file is superseded and kept as the record.** The notation of the sampling
> variables was decided by the maintainer on 2026-11-15 at 23:23 (target decision 9.6). The reference for every
> name is `SAMPLE_VARIABLES.md`; where this file disagrees with it, that page wins.

This file holds the names this folder used before 2026-11-15, 23:23, and the naming ideas that led to the
decision, moved here on 2026-11-16 from the documents where they stood. Each moved passage says where it came
from (file, section, date) and is copied word for word. Two things are changed in a moved passage and nothing
else: its headings are one level lower, so that they sit under the sections of this file; and an editor's gloss
may stand in square brackets, marked `[ed.: ...]`, where an old name would otherwise be unreadable. A section
number inside a moved passage refers to the document it came from. In each place a passage left, one or two
lines say what was there and point here.

What stays where it was: `PROPOSAL.md`, `API_IMPACT.md`, `WORKED_EXAMPLE.md`, `EVALUATOR.md` and
`EXECUTION_PLANS.md` are snapshots and keep their old names, with a notice in their status notes;
`REVIEW_DECISIONS.md` is a dated log and keeps the words of each entry's time (its header says how to read
them); the reviewer reports under `reviews/` are never edited.

Contents:

1. Timeline of the names
2. `gamma`: "an auxiliary variable, not a `nu`" (`DESIGN_AFTER_REVIEW.md` section 3.5)
3. The variables without a name (`DESIGN_AFTER_REVIEW.md` section 4.3; `TARGET_DECISIONS.md`, notes to section 6)
4. `zeta` (2026-11-15, 22:11) (`TARGET_DECISIONS.md`, notes to section 6)
5. The dot collision: options A and B (`TARGET_DECISIONS.md`, notes to section 6; `TYPE_CATALOG.md` section 13)
6. The naming problem widened: the core and the leg; eight letters, or four letters and a basis
   (`TARGET_DECISIONS.md`, notes to section 6; `TYPE_CATALOG.md` section 14)
7. Prototype identifiers renamed on 2026-11-16

## 1. Timeline of the names

Written for this file on 2026-11-16 (the editor's summary, not a moved passage).

- **Phase 1, 2026-11-13/14** (`PROPOSAL.md`, sections "Edge variables" and "Correspondence with current names",
  at `89ac203f`). Letters `w`, `z`, `xi`, `eta`, `mu`, `nu`, and `phi` for the value of apply and entries. Grades
  by suffix: none, `_t` (one core replaced by its variation), `_z` (one residual inserted), `_tz` (both). The
  base letter was to fix "the edge and direction". `PROPOSAL.md` keeps these names; its status note maps them.
- **2026-11-14, review decision 8** (`REVIEW_DECISIONS.md` entry 8). The grades became `_dot`, `_tilde`,
  `_tilde_dot`; the right edge variable swept through `L` got the letter `gamma`; the site rules became the seven
  `*_term` functions and the per-site functions `*_step`. A `_dot` meant the frame's dot, along the direction
  `(V, H)`.
- **2026-11-15, 21:57.** Six variables without a name were listed: the down variable through the base point's
  cores and its three grades, and the dots of `xi` and `mu` along the raw core direction (section 3).
- **22:11.** `zeta` was decided for the down variable through the base point's cores (section 4). Options A and
  B for the dots of `xi` and `mu` were set out (section 5).
- **22:24 to 22:41.** The naming was rethought from first principles: a variable is a contracted subnetwork in a
  basis; two schemes, eight letters or four letters and a basis; the second written out as a table (section 6).
- **23:02.** Diagrams of seven variables in the four-letters-and-a-basis scheme (`REVIEW_DECISIONS.md` entry
  56).
- **23:23, decided** (target decision 9.6; `REVIEW_DECISIONS.md` entry 57): four letters and a basis.
  `SAMPLE_VARIABLES.md` is the reference.
- **2026-11-16.** The folder was brought into line; this file was made; the live prototypes were renamed
  (section 7).

**The names in the three periods.** As seen from a call on a manifold frame. In a call on the corewise frame
itself the decided names carry no basis (`SAMPLE_VARIABLES.md` section 9). "Frame's form" is a dot along the
direction `(V, H)`; "raw form" is a dot along a raw core direction at the base point. The middle column is
`SAMPLE_VARIABLES.md` section 10; the last column is read from the name table of `PROPOSAL.md`.

| Decided (2026-11-15, 23:23) | In this folder, 2026-11-14 to 23:23 | Phase 1 (`PROPOSAL.md`) |
| --- | --- | --- |
| `mu` | `mu` | `mu` |
| `mu_dot` (raw form) | no name; the prototype's `c_mu_dot` | `mu_t`, at the corewise frame |
| `mu_tilde` | `mu_tilde` | `mu_z` |
| `mu_tilde_dot` | `mu_tilde_dot` | `mu_tz` (corewise only) |
| `mu_R_dot` (frame's form) | `mu_dot`, "the frame's `mu_dot`" | `mu_t`, in a manifold frame |
| `nu` | `nu` | `nu` |
| `nu_tilde` | `nu_tilde` | `nu_z` |
| `nu_L` | `gamma` | no letter of its own: "`nu` through `L`" (`EVALUATOR.md`); `nu` at the corewise frame |
| `nu_L_dot` | `gamma_dot` (raw form), and `nu_dot`, "the frame's `nu_dot`" (frame's form) | `nu_t`, in either |
| `nu_L_tilde` | `gamma_tilde` | `nu_z`, at the corewise frame |
| `nu_L_tilde_dot` | `gamma_tilde_dot` | `nu_tz` (corewise only) |
| `xi` | `xi` | `xi` |
| `xi_dot` (raw form) | no name; the prototype's `c_xi_dot` | `xi_t`, at the corewise frame |
| `xi_tilde` | `xi_tilde` | `xi_z` |
| `xi_tilde_dot` | `xi_tilde_dot` | `xi_tz` (corewise only) |
| `xi_U_dot` (frame's form) | `xi_dot`, "the frame's `xi_dot`" | `xi_t`, in a manifold frame |
| `eta` | `eta` | `eta` |
| `eta_tilde` | `eta_tilde` | `eta_z` |
| `eta_D` | `zeta` (22:11 to 23:23); before 22:11 no name (written inline in `z_step`) | `eta`, at the corewise frame |
| `eta_D_dot` | `zeta_dot` and the prototype's `c_eta_dot` (raw form), and `eta_dot`, "the frame's `eta_dot`" (frame's form) | `eta_t`, in either |
| `eta_D_tilde` | `zeta_tilde` | `eta_z`, at the corewise frame |
| `eta_D_tilde_dot` | `zeta_tilde_dot`; the prototype's `eta_tilde_dot` | `eta_tz` (corewise only) |
| `w`, `w_tilde`, `z`, `Jp` | the same | `w`, `w_z`, `z` (and `phi` for apply and entries), `z_t` (and `phi_t`) |
| `V_tilde`, `H_tilde`, `V_tilde_dot`, `H_tilde_dot` | the same | `dB_z`, `dG_z`, `dB_tz`, `dG_tz` |

## 2. `gamma`: "an auxiliary variable, not a `nu`"

**Moved from `DESIGN_AFTER_REVIEW.md`, section 3.5**, as it stood on 2026-11-14 (commit `e2e8fc6`), on
2026-11-16. Superseded on 2026-11-15 at 23:23 by target decision 9.6: the right of a bond in the `L` basis is
`nu_L`, the same side as `nu` in the base point's basis. The facts of the section (who reads it, the library's
`nu_c`, that it coincides with `nu` at a trial point) stay in `DESIGN_AFTER_REVIEW.md` section 3.5 under the new
name. The decision itself is entry 8 of `REVIEW_DECISIONS.md`, which stays in place.

#### 3.5 `gamma`

`nu` means the frame's right edge variable, swept through `R`, and nothing else. The right edge variable swept
through `L`, the base point's own cores, is a different variable: **`gamma`**. The maintainer's view is that it is
an auxiliary variable, not a `nu`. Its residual-grade companion is `gamma_tilde`.

- `gamma` is read by `z` and by `Kp`. `gamma_tilde` is read by `Kp`. Both are public: they can be wanted and held.
- The library writes `gamma` as `nu_c`, with `nu_c = T nu` at a manifold frame (`curvature_internals.md`). The
  proposal's matrix `S` in that identity becomes `T`.
- At a trial point the call uses the point's own cores as the frame, so `gamma` and `nu` coincide there.
- The name is free in code. In the docs, lowercase gamma is a scalar in one example of `padded_frame_cone.tex`, and
  capital `Gamma_i` is the mode Gram matrix in the sharing documents.

## 3. The variables without a name

### 3.1 "Variables with no public name" (`DESIGN_AFTER_REVIEW.md`, section 4.3)

**Moved from `DESIGN_AFTER_REVIEW.md`, section 4.3**, as it stood on 2026-11-14 (commit `e2e8fc6`), on
2026-11-16. Superseded: every variable can be wanted and held (target decision 6.4, confirmed on 2026-11-15 at
21:57), and every variable has a name (target decision 9.6, 2026-11-15, 23:23). Which application of the
transpose a tilde name refers to is target decision 3.3.

Variables with no public name:

- The tilde variables seeded by `Jp`. They change with every product and stay internal to `JtJp`. The tilde
  variables that can be wanted or held are always the ones seeded by the residual.
- The `eta`-like variable through `L` that only `z` reads.
- The dot variables of `Kp`, which belong to the lifted direction.

### 3.2 "The variables that still have no name" (`TARGET_DECISIONS.md`, notes to section 6, 21:57)

**Moved from `TARGET_DECISIONS.md`, notes to section 6** (commit `e2e8fc6`), on 2026-11-16. The note was
written at 21:57 and 21:59 on 2026-11-15; its sub-bullets were added at 22:11 and 22:24 to 22:41. It is split
here by topic, in its own order: this section has its head and its first three sub-bullets; sections 4, 5 and
6 have the rest. All six variables were named at 23:23 (target decision 9.6): rows 1 to 4 are `eta_D`,
`eta_D_dot`, `eta_D_tilde`, `eta_D_tilde_dot`; rows 5 and 6 are `xi_dot` and `mu_dot`. The argument that carries
the raw core direction is still without a name (`SAMPLE_VARIABLES.md` section 11).

- **The variables that still have no name** (2026-11-15, 21:57). The maintainer: "Can you please list the
  variables that still need names, and some suggestions? One possible idea is to recycle names that are in use
  in the current codebase but will be freed up, such as sigma and tau which form a nice pair". And at 21:59:
  "Trying to keep consistency with the t4s paper's notation through this refactor is an impossible task. We do
  not need to attempt it." Listed by Claude; nothing decided.
  - **What they are.** Every one is a variable of the corewise frame `(B, G, G, G)` at the base point, seen
    from inside a call on a manifold frame. In a call on the corewise frame itself they have the ordinary names.
    `gamma` is the first of this kind: it is the corewise frame's `nu`.

    | # | The variable | In a corewise call it is | Read by | Stand-in in `jet_steps_check.py` |
    | --- | --- | --- | --- | --- |
    | 1 | the down variable through the base point's cores, `eta_term(mu, L, gamma)` | `eta` | `z` | written inline in `z_step` |
    | 2 | its dot along the raw core direction | `eta_dot` | `Kp`, Tucker block | `c_eta_dot` |
    | 3 | its tilde | `eta_tilde` | nothing on a manifold frame; the corewise gradient | none |
    | 4 | its tilde-dot | `eta_tilde_dot` | `Kp`, Tucker block | `eta_tilde_dot`, which is a misnomer there |
    | 5 | the dot of `xi` along the raw core direction | `xi_dot` | `Kp` | `c_xi_dot` |
    | 6 | the dot of `mu` along the raw core direction | `mu_dot` | `Kp` | `c_mu_dot` |

  - **Rows 1 to 4 are one missing letter**, the partner of `gamma`. **Rows 5 and 6 are a collision**: `xi` and
    `mu` are shared by the two frames, but their dots are not, because the two frames have different
    directions: the tangent direction `(V, H)` in the frame's coordinates, and a raw core direction at the base
    point. `gamma_dot` and the four tilde-dot variables have no collision, since they exist at the base point
    only.
  - **Also without a name:** the raw core direction itself, as an argument (`(B_dot, G_dot)` in the prototype,
    `core_directions` in the library; the design document notes it as unnamed).

## 4. `zeta` (2026-11-15, 22:11)

**Moved from `TARGET_DECISIONS.md`, notes to section 6** (commit `e2e8fc6`), on 2026-11-16: the sub-bullet of
22:11 of the note in section 3.2 above. Superseded at 23:23 (target decision 9.6): `zeta` is retired, and the
down variable through the base point's cores is `eta_D`, the trunk above a Tucker edge in the `D` basis.

  - **`zeta`, decided** (2026-11-15, 22:11). The maintainer: "zeta is great!" The down variable through the
    base point's cores is `zeta`, with the grades `zeta_dot`, `zeta_tilde`, `zeta_tilde_dot`; `z = D . zeta`. The
    base point has the four letters `xi`, `mu`, `gamma`, `zeta`. The prototype's `eta_tilde_dot` is
    `zeta_tilde_dot`.

## 5. The dot collision: options A and B (2026-11-15, 22:11)

Superseded at 23:23 by target decision 9.6. Neither option was taken: the two dots of `mu` are one side in two
bases, `mu_dot` (the `L` basis, raw form) and `mu_R_dot` (the `R` basis, frame's form), and likewise `xi_dot` and
`xi_U_dot`. `sigma` and `tau` are not used.

**From `TARGET_DECISIONS.md`, notes to section 6** (commit `e2e8fc6`), moved on 2026-11-16: the sub-bullet that
follows section 4 above.

  - **Rows 5 and 6, two options, not decided.** A: a mark for the raw direction on the base point's dots. B:
    letters for the frame's tangent variables (`sigma` and `tau` for its two environments, as in today's code),
    so that `_dot` means the raw core direction only. The maintainer asked how these sit within the type
    catalog. The answer is section 13 of `TYPE_CATALOG.md`: the collision is a difference of edge; a different
    edge getting a different letter is the catalog's precedent (`nu` and `gamma`); B carried through needs a
    letter for each of the frame's four tangent variables, and makes a letter fix the edge and the direction in
    every grade.

**From `TYPE_CATALOG.md`, section 13** (commit `e2e8fc6`), moved whole on 2026-11-16. Claude's analysis of
2026-11-15, 22:11.

### 13. The letters by edge, and the variables without names (2026-11-15, 22:11)

The maintainer accepted `zeta` for the down variable through the base point's cores ("zeta is great!"), and asked
of the two ways to name the dots of `xi` and `mu` along the raw core direction (target decisions, notes to
section 6): "How do these options sit within the naming concepts we explored in the type catalog document?"
Claude's analysis. Not a decision.

**The catalog has three coordinates, and its two rules use two of them.** A variable's type is an edge (`AL`,
`AR`, `XD`, `XU`, `N`), a direction on that edge (vector or covector), and an order factor. Rule 1 says the letter
fixes the direction. Rule 2 says the grade fixes the order factor. Nothing in a name fixes the edge. Grouping the
checked catalog of `type_catalog_check.py` by edge and direction:

| Edge | Vector | Covector |
| --- | --- | --- |
| `AL`, an `L` bond | `mu`, `mu_tilde`, `mu_tilde_dot`, and the dot of `mu` along the raw core direction | `gamma`, `gamma_tilde`, `gamma_dot`, `gamma_tilde_dot`, and the frame's `nu_dot` |
| `XD`, a `D` edge | `xi`, `xi_tilde`, `xi_tilde_dot`, and the dot of `xi` along the raw core direction | `zeta` in every grade, and the frame's `eta_dot` |
| `AR`, an `R` bond | the frame's `mu_dot`, and nothing else | `nu`, `nu_tilde` |
| `XU`, a `U` edge | the frame's `xi_dot`, and nothing else | `eta`, `eta_tilde` |

**What the table says about the collision.**

- The two rules do not see it: both dots of `mu` are vectors and jets.
- It is a difference of edge. The dot of `mu` along the raw core direction is on the edge of `mu`, so it has the
  type of what was differentiated, and `mu + mu_dot` is well typed. The frame's `mu_dot` is on an `R` bond; the
  checker of section 9 rejects `mu + mu_dot` for it. The same holds for `xi`.
- The catalog already has this situation once: `nu` and `gamma` travel the same way on different edges, and they
  were given different letters. `eta` and `zeta` are the second case.
- Two slots of the table have no letter of their own: the vector on an `R` bond and the vector on a `U` edge.
  They are the partners of `nu` and of `eta`, and only the frame's tangent variables live there.
- The frame's `nu_dot` and `eta_dot` have exactly the types of `gamma` and `zeta`.

**The two options in these terms.**

- **A, a mark for the raw direction.** "Which direction a variable is linear in" is not one of the catalog's
  coordinates. The mark would record that, and leave the difference of edge, which is one, unspoken.
- **B, letters for the frame's tangent variables.** A different edge getting a different letter is the catalog's
  own precedent. Carried through, the frame's four tangent variables each get a letter: on the `R` bond, vector
  (the proposal's frame `mu_dot`; the `sigma` of today's code); on the `L` bond, covector (frame `nu_dot`;
  today's `tau`); on the `U` edge, vector (frame `xi_dot`); on the `D` edge, covector (frame `eta_dot`). The
  first and third are the two slots without a letter. The second and fourth share the types of `gamma` and
  `zeta` and differ from `gamma_dot` and `zeta_dot` only in the direction they are linear in; a letter for a role
  where the type is shared also has a precedent, `z` and `Jp`.
- **What B gives.** Rule 1 becomes stronger: a letter fixes the edge and the direction, in every grade. Today it
  cannot say that, because `xi` is on `XD` and `xi_dot` on `XU`. The grades `dot` and `tilde-dot` then exist at
  the base point only, along the raw core direction, and `_dot` has one meaning.
- **The multiplier pairings of section 11 under B.** The tilde of a letter is the multiplier of the tangent
  variable on the same edge, travelling the other way: `nu_tilde` of the `R`-bond tangent variable, `mu_tilde`
  of the `L`-bond one, `eta_tilde` of the `U`-edge one, `xi_tilde` of the `D`-edge one.
- **What B leaves to settle.** Two more letters. And which spelling a call on the corewise frame accepts, since
  there the edges merge and each tangent variable equals a dot: the question already open for `nu` and `gamma`.

## 6. The naming problem widened (2026-11-15, 22:24 to 22:41)

Superseded at 23:23 by target decision 9.6, which took the second scheme below ("four letters and a basis").
Parts of `TYPE_CATALOG.md` section 14 were adopted into `SAMPLE_VARIABLES.md`: "Three things identify a variable"
and the table of 22 of 32 (its sections 2 and 3), "One dot" (its section 5.2), the core and leg as the
definition of a plain variable (its section 5.1), and the written-out table (its section 10). "Eight letters" was
not taken. In the passages below "the frame's `mu_dot`" is `mu_R_dot`, "the frame's `xi_dot`" is `xi_U_dot`,
"the frame's `nu_dot`" and `gamma_dot` are `nu_L_dot`, "the frame's `eta_dot`" and `zeta_dot` are `eta_D_dot`,
`gamma` is `nu_L` and `zeta` is `eta_D`; a bare `mu_dot` or `xi_dot` in section 14 already means the raw form.

**From `TARGET_DECISIONS.md`, notes to section 6** (commit `e2e8fc6`), moved on 2026-11-16: the last sub-bullet
of the note in section 3.2 above.

  - **The naming problem widened** (2026-11-15, 22:24 and 22:27). The maintainer asked for the minimal
    information that identifies a variable, proposed naming by the core and leg a variable comes out of
    (`mu` as `L2`), and suggested seeing an edge variable as the fully contracted subnetwork on one side of an
    edge. He set precedent aside: "This is all still in the planning stage, so there is no precedent to worry
    about." Claude's analysis is section 14 of `TYPE_CATALOG.md`, checked by
    `scripts/review/subtree_gauge_check.py`. A variable is identified by the side contracted (four kinds), the
    basis of the cut edge (two on a manifold frame), and what was differentiated. Along the lifted direction the
    frame's dots and the base point's dots are one tangent vector: the frame's `nu_dot` is `gamma_dot` and its
    `eta_dot` is `zeta_dot`, on the arrays. So rows 5 and 6 are one side in two bases, as `nu` and `gamma` are,
    and not two directions. Two schemes follow (eight letters; or four letters and a basis). Not decided.

**From `TYPE_CATALOG.md`, section 14** (commit `e2e8fc6`), moved whole on 2026-11-16, with the table written out
at 22:41. Claude's analysis of 2026-11-15, 22:24 to 22:41. Its "Not decided" was true until 23:23.

### 14. A variable is a contracted subnetwork in a basis; one dot (2026-11-15, 22:24 to 22:27)

The maintainer: "Let's allow ourselves the freedom to explore broader concepts for the naming scheme. This is all
still in the planning stage, so there is no precedent to worry about. What is the minimal amount of information
that is required to identify a variable? Can they be identified by which core they come out of, which direction
they go, and whether they are tilde or not? For example, in this case mu would become L2 since it comes out of
index 2 (L_aib, a is index zero, i is index 1, b is index 2) of the L core." And at 22:27: "One idea is to
consider the edge variables as being as corresponding to fully contracted subnetworks on one side of the edge."
Claude's analysis. Not a decision. Checked by `scripts/review/subtree_gauge_check.py`.

**Three things identify a variable.**

1. **Which side of which edge is contracted.** The network is a tree, so a cut edge has two sides. There are four
   kinds: the left of a bond, the right of a bond, the leaf below a Tucker edge, the trunk above it. (At a mode
   edge the leaf is the input `w` and the trunk is the output `z`.) This is a property of the tensor and of the
   samples. It does not depend on a gauge.
2. **Which basis the cut edge is written in.** A manifold frame gives each cut edge two: the base point's own
   (the stores `L` and `D`) and the frame's other one (the stores `R` and `U`). The corewise frame has one.
3. **What was differentiated:** the cores along the direction (dot), the inputs along the seed (tilde), or both.

That is 4 x 2 x 2 x 2 = 32 possibilities, of which 22 occur:

| Side contracted | In the base point's basis (`L`, `D`) | In the frame's other basis (`R`, `U`) |
| --- | --- | --- |
| The left of a bond | `mu`: plain, dot, tilde, tilde-dot | the frame's `mu_dot`: dot only |
| The right of a bond | `gamma`: plain, dot, tilde, tilde-dot | `nu`: plain, tilde |
| The leaf below a Tucker edge | `xi`: plain, dot, tilde, tilde-dot | the frame's `xi_dot`: dot only |
| The trunk above a Tucker edge | `zeta`: plain, dot, tilde, tilde-dot | `eta`: plain, tilde |

The first column is the base point's complete family: four sides in four grades. The second column is what the
orthogonal frame adds. This is the table of section 13 with its rows explained: an edge of section 13 is a basis
here, and a direction of travel is a side.

**One dot.** Sections 2 and 13 treated the frame's dot variables (along the tangent direction `(V, H)`) and the
base point's dots (along a raw core direction) as two families. Along the lifted direction they are one tangent
vector, and the check shows it on the arrays, with the library's own frame, gauge projection and undamped lift:

- Where both are in the same basis they are the same array: the frame's `nu_dot` is `gamma_dot`, and the
  frame's `eta_dot` is `zeta_dot`.
- Where they are in different bases they are one subnetwork in two bases, joined by the lift's own factors:
  the frame's `mu_dot` and `mu_dot`; the frame's `xi_dot` and `xi_dot`. The same factors join the plain pairs
  `nu` and `gamma`, `eta` and `zeta`.

So the collision of rows 5 and 6 (target decisions, notes to section 6) is not between two directions. It is the
situation of `nu` and `gamma` again: one side, two bases.

**Two limits of that statement.**

- With slack ranks a twin relation holds in one direction only. On the checked frame the last bond has 4
  dimensions in the `L` basis and 3 in the `R` basis; the frame's `mu_dot` is the image of `mu_dot`, but not the
  other way round, because `mu_dot` is free in the dimension that the right side annihilates.
- With curvature damping above zero the lifted direction is not exactly the tangent, so the dots that `Kp` uses
  are then along a slightly different direction. Not examined.

**The maintainer's naming by the core and the leg a variable comes out of.**

- For the plain variables it is exact and unique: `mu` comes out of leg `b` of `L`, `gamma` out of leg `a` of
  `L`, `zeta` out of leg `i` of `L`, `nu` out of leg `a` of `R`, `eta` out of leg `i` of `U`, `xi` out of leg `i`
  of `D`, and `z` out of leg `o` of `D`. It explains at once why `gamma` is not `nu`.
- For a dotted variable it is not unique: the frame's `mu_dot` is a sum of terms that come out of `R`, of `H`
  and of `U`. What stays unique is the side and the basis.
- So the core and leg are the right definition of a plain variable, to be given in the table of variables. As
  the name itself, a digit after a store letter also reads as a site number.

**Two schemes that follow from the table.** Neither is decided.

- **Eight letters: a letter for each side in each basis.** `mu`, `gamma`, `xi`, `zeta` for the base point's
  column and `nu`, `eta` and two new letters for the frame's column. The two new ones are exactly the variables
  that exist only as dots, the twins of `mu` and `xi`: `sigma` and `tau` fit there as a pair.
- **Four letters and a basis: a letter for each side, and the store of the basis where it is not the usual
  one.** `mu`, `nu`, `xi`, `eta` name the four sides, as they did at the start. The usual basis of each is the
  store it travels away from: `L` for `mu`, `R` for `nu`, `D` for `xi`, `U` for `eta`. The other four are
  `nu_L` (today's `gamma`), `eta_D` (`zeta`), `mu_R` (the frame's `mu_dot`) and `xi_U` (the frame's `xi_dot`).
  What this buys: a name can be read without a table; a term function's name agrees with what it returns
  (`nu_step` through `L` returns `nu_L`); and a call on the corewise frame, which has one basis, uses the four
  letters with no suffix, so the question of two spellings for one array there does not arise.

**The second scheme written out** (asked for by the maintainer at 22:41). Every variable, with its name so far
in this folder and the name of the array in the library's kernels today (`backend/probing.py`; the jetted
kernels add `_jets`). Not decided.

| Four letters and a basis | Side, basis | Name so far in this folder | In the library today |
| --- | --- | --- | --- |
| `mu` | left of a bond, `L` | `mu` | `mus` |
| `mu_dot` | left of a bond, `L` | none (row 6; the prototype's `c_mu_dot`) | `mus_dot`, in the two-hole kernel |
| `mu_tilde` | left of a bond, `L` | `mu_tilde` | `tau_tildes` |
| `mu_tilde_dot` | left of a bond, `L` | `mu_tilde_dot` | `tau_tildes_dot` |
| `mu_R_dot` | left of a bond, `R` | the frame's `mu_dot` | `sigmas` |
| `nu` | right of a bond, `R` | `nu` | `nus` |
| `nu_tilde` | right of a bond, `R` | `nu_tilde` | `sigma_tildes` |
| `nu_L` | right of a bond, `L` | `gamma` | `nus`, re-swept through the point's cores in the two-hole kernel |
| `nu_L_dot` | right of a bond, `L` | `gamma_dot`, and the frame's `nu_dot` | `nus_dot` in the two-hole kernel, and `taus` |
| `nu_L_tilde` | right of a bond, `L` | `gamma_tilde` | `sigma_tildes`, in the two-hole kernel |
| `nu_L_tilde_dot` | right of a bond, `L` | `gamma_tilde_dot` | `sigma_tildes_dot` |
| `xi` | leaf, `D` | `xi` | `xis` |
| `xi_dot` | leaf, `D` | none (row 5; the prototype's `c_xi_dot`) | `xis_dot`, in the two-hole kernel |
| `xi_tilde` | leaf, `D` | `xi_tilde` | `deta_tildes` |
| `xi_tilde_dot` | leaf, `D` | `xi_tilde_dot` | `deta_tildes_dot` |
| `xi_U_dot` | leaf, `U` | the frame's `xi_dot` | `dxis` |
| `eta` | trunk, `U` | `eta` | `etas` |
| `eta_tilde` | trunk, `U` | `eta_tilde` | `dxi_tildes` |
| `eta_D` | trunk, `D` | `zeta` (row 1) | `etas`, when `t3_probe` runs on the point's own cores |
| `eta_D_dot` | trunk, `D` | `zeta_dot` (row 2), and the frame's `eta_dot` | `etas_dot` in the two-hole kernel, and `detas` |
| `eta_D_tilde` | trunk, `D` | `zeta_tilde` (row 3) | `dxi_tildes`, in the corewise transpose |
| `eta_D_tilde_dot` | trunk, `D` | `zeta_tilde_dot` (row 4; the prototype's `eta_tilde_dot`) | `dxi_tildes_dot` |

- **Unchanged:** `w`, `w_tilde`, `z`, `r`, `Jp`, `f`, `g`, `JtJp`, `Kp`, and the blocks `V_tilde`, `H_tilde`,
  `V_tilde_dot`, `H_tilde_dot`.
- **Two names change their meaning.** `mu_dot` and `xi_dot` meant the frame's variables in this folder. Here
  they are the base point's, and the frame's become `mu_R_dot` and `xi_U_dot`.
- **Two pairs of names become one.** The frame's `nu_dot` and `gamma_dot` are `nu_L_dot`; the frame's `eta_dot`
  and `zeta_dot` are `eta_D_dot`. They are the same arrays along the lifted direction with no damping.
- **The multiplier pairings.** The tilde of a variable is the multiplier of the dot of the other side of the
  same edge, in the same basis: `mu_tilde` of `nu_L_dot`, `nu_tilde` of `mu_R_dot`, `xi_tilde` of `eta_D_dot`,
  `eta_tilde` of `xi_U_dot`. This is section 11's list with the names of this table.
- **On the corewise frame** there is one basis, so the names are the four letters and their grades with no
  suffix; `nu` there is this table's `nu_L`.
- The library today uses one name for two variables in several rows (`nus`, `sigma_tildes`, `etas`,
  `dxi_tildes`), telling them apart by which cores were passed.

## 7. Prototype identifiers renamed on 2026-11-16

The live prototypes in `scripts/review/` were renamed to the decided notation on 2026-11-16. Their recorded
output, `scripts/review/expected_review_output_3dc15d9.txt`, was recorded again: only labels changed, and every
number is the same. The dated records among the scripts (listed in `README.md`) keep their names.

| Script | Old identifier | New identifier |
| --- | --- | --- |
| `jet_steps_check.py` (and the string key in `arrow_type_check.py`) | the step function `gamma_tilde_dot_step` | `nu_tilde_dot_step` |
| `jet_steps_check.py` | the parameters `gamma_tilde_dot`, `gamma_dot`, `gamma_tilde`, `gamma` of `gamma_tilde_dot_step`, `eta_tilde_dot_step` and `H_tilde_dot_step`, and `gamma` of `z_step` | `nu_tilde_dot`, `nu_dot`, `nu_tilde`, `nu` (a step function's parameters carry no basis) |
| `jet_steps_check.py`, `type_catalog_check.py`, `subtree_gauge_check.py`, `order0_assessment.py` | `gamma` | `nu_L` |
| `jet_steps_check.py`, `type_catalog_check.py` | `gamma_tilde`, `gamma_tilde_dot` | `nu_L_tilde`, `nu_L_tilde_dot` |
| `jet_steps_check.py`, `type_catalog_check.py` | the variable `eta_tilde_dot` (a misnomer: it was the trunk in the `D` basis) | `eta_D_tilde_dot` |
| `jet_steps_check.py`, `type_catalog_check.py` | `gamma_dot`, `c_eta_dot` (the raw form) | `raw_nu_L_dot`, `raw_eta_D_dot` |
| `subtree_gauge_check.py` | `gamma_dot`, `zeta_dot` (the raw form) | `raw_nu_L_dot`, `raw_eta_D_dot` |
| `subtree_gauge_check.py` | `zeta` | `eta_D` |
| `jet_steps_check.py`, `type_catalog_check.py` | `c_mu_dot`, `c_xi_dot` | `mu_dot`, `xi_dot` |
| `jet_steps_check.py`, `type_catalog_check.py`, `tilde_multiplier_check.py`, `order0_assessment.py` | `xi_dot`, `mu_dot`, `nu_dot`, `eta_dot` (the frame's form) | `xi_U_dot`, `mu_R_dot`, `nu_L_dot`, `eta_D_dot` |
| `subtree_gauge_check.py` | `f_xi_dot`, `f_mu_dot`, `f_nu_dot`, `f_eta_dot` | `xi_U_dot`, `mu_R_dot`, `nu_L_dot`, `eta_D_dot` |
| `kind_readout_check.py` | `xi_dot`, `mu_dot` (the frame's form) | `xi_U_dot`, `mu_R_dot` |
| `arrow_type_check.py` | the reject row "`gamma` used where `nu` is wanted", "a sum of `mu` and `mu_dot`" | "`nu_L` used where `nu` is wanted", "a sum of `mu` and `mu_R_dot`" |

`raw_nu_L_dot` and `raw_eta_D_dot` are a local spelling of the scripts, not notation: the raw form's `nu_L_dot`
and `eta_D_dot` where a program also holds the frame's.

The step functions keep their names, apart from `gamma_tilde_dot_step`, and every one of them is named by letter
and grade, with basis-free parameters. A step function serves both bases: the basis of what it returns follows
from the cores it is given (`nu_step` given `L` cores returns `nu_L`; `mu_dot_step` given `(R, U, H)` and the
frame's `xi_U_dot` returns `mu_R_dot`, given `(L, L, G_dot)` and the raw `xi_dot` it returns `mu_dot`). Decided by
the maintainer on 2026-11-16 at 00:23: "There is no need for extra step functions if they literally do the exact
same things with different arguments."
