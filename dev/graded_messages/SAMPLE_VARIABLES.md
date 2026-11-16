# The sampling variables

**Status.** This page is the reference for the names and the definitions of the sampling variables. The
notation was decided by the maintainer on 2026-11-15 (23:23), after he had seen it drawn: "This is excellent and
further confirms that this notation is the right one. I want to lock in this new notation." It replaces the
names used in this folder until then and the names in the library today; section 10 maps both to the new ones.

Where another document of this folder disagrees with this page about a name, this page wins. Older names and
the ideas that led here are kept in `NOTATION_HISTORY.md`, marked as deprecated.

**Checked** by `scripts/review/sample_variables_check.py`: the recurrences of section 5 against the library, and
every definition "as a subnetwork" against its recurrence. Covered: the probe kind, ragged cores, numpy, one
sample axis, orders 0 and 2. Not covered: apply and entries, the uniform layout, the `K` and `C` stacks.

## 1. The network

A Tucker tensor train with `d` sites, numbered `0` to `d-1`, sampled at input vectors.

```
 bond 0     bond 1     bond 2            bond d
   |--[TT 0]--|--[TT 1]--|--  ...  --[TT d-1]--|          the TT cores
        |          |                     |                Tucker edge i, below TT core i
     [Tu 0]     [Tu 1]               [Tu d-1]             the Tucker cores
        |          |                     |                mode edge i, below Tucker core i
      w[0]       w[1]                 w[d-1]              the inputs
```

- **Bond `i`** is the edge to the left of site `i`, for `i = 0 .. d`. Site `i` lies between bond `i` and bond
  `i+1`. Bonds `0` and `d` are the two ends.
- **Tucker edge `i`** joins TT core `i` to Tucker core `i`. **Mode edge `i`** joins Tucker core `i` to the input.

**The stores of a manifold frame** `(D, U, L, R)`, at site `i`:

| Store | What it is | Legs, in order | Sizes |
| --- | --- | --- | --- |
| `D[i]` | the down Tucker core | Tucker edge, mode edge | `(nD_i, N_i)` |
| `L[i]` | the left TT core | bond `i`, Tucker edge, bond `i+1` | `(rL_i, nD_i, rL_(i+1))` |
| `R[i]` | the right TT core | bond `i`, Tucker edge, bond `i+1` | `(rR_i, nD_i, rR_(i+1))` |
| `U[i]` | the up TT core | bond `i`, Tucker edge, bond `i+1` | `(rL_i, nU_i, rR_(i+1))` |

- **The base point** is `(D, L)`: the tensor itself is every `L[i]` over its `D[i]`.
- **A direction at the frame** is `(V, H)`. `V[i]` has the legs of a Tucker core with the Tucker edge in the `U`
  basis, `(nU_i, N_i)`: it sits under `U[i]`, in place of `D[i]`. `H[i]` has the legs of a TT core,
  `(rL_i, nD_i, rR_(i+1))`: it sits over `D[i]`, with `L` cores to its left and `R` cores to its right.
- **A raw core direction at the base point** is `(D_dot, L_dot)`: `D_dot[i]` has the shape of `D[i]` and
  `L_dot[i]` the shape of `L[i]`. It is the direction the cores of the base point move in. On the corewise frame
  `(B, G, G, G)` the same thing is `(B_dot, G_dot)`.
- **The corewise frame** `(B, G, G, G)` has one basis on each edge. See section 9.

**Two bases on each cut edge.** A manifold frame writes each bond in two bases, the `L` basis (size `rL_i`) and
the `R` basis (size `rR_i`), and each Tucker edge in two, the `D` basis (size `nD_i`) and the `U` basis (size
`nU_i`). The `L` and `D` bases are the base point's own.

## 2. What identifies a variable

A sampling variable is the contraction of the subnetwork on one side of an edge. Three things identify it.

1. **The side that is contracted.** There are four kinds, and each has a letter.
2. **The basis the cut edge is written in.** It is named by a store: `L` or `R` for a bond, `D` or `U` for a
   Tucker edge.
3. **The grade: what was put into the subnetwork.** Nothing (plain); the direction at one site (dot); the seed in
   place of one input (tilde); both (tilde-dot).

| Letter | Side contracted | It travels | Usual basis | Other basis |
| --- | --- | --- | --- | --- |
| `mu` | the left of a bond: sites `0 .. i-1` | left-to-right | `L` | `R` |
| `nu` | the right of a bond: sites `i .. d-1` | right-to-left | `R` | `L` |
| `xi` | the leaf below a Tucker edge: Tucker core `i` and its input | bottom-to-top | `D` | `U` |
| `eta` | the trunk above a Tucker edge: everything except that leaf | top-to-bottom | `U` | `D` |

The usual basis of a letter is the store it travels away from.

**A name** is `<letter>[_<basis>][_tilde][_dot]`, with the index in brackets after it.

- The basis is written only where it is not the usual one: `nu_L`, `eta_D`, `mu_R`, `xi_U`.
- `_tilde` comes before `_dot`: `nu_L_tilde_dot`.
- `mu[i]` and `nu[i]` sit on bond `i`, for `i = 0 .. d`. `xi[i]` and `eta[i]` sit on Tucker edge `i`, for
  `i = 0 .. d-1`.

**In a formula or a drawing** the same name is written with the Greek letter, a tilde or a dot above it, and the
basis and the index as subscripts: `nu_L_dot[3]` is the letter ν with a dot above it and the subscript `L,3`;
`mu_tilde[2]` is μ with a tilde above it and the subscript `2`. Where a name has both, the tilde is next to the
letter and the dot is above the tilde. The outputs and the blocks are written the same way: `z[2]` is z with the
subscript `2`, and `H_tilde[2]` is H with a tilde above it. Code and plain text keep the spelling with
underscores.

A mode edge has one basis, so `w`, `w_tilde`, `z` and `Jp` never have a basis in their names.

## 3. The taxonomy

Four sides, two bases, four grades: 32 possibilities, of which 22 occur.

| Side contracted | In the base point's basis (`L`, `D`) | In the frame's other basis (`R`, `U`) |
| --- | --- | --- |
| The left of a bond | `mu`, `mu_dot`, `mu_tilde`, `mu_tilde_dot` | `mu_R_dot` |
| The right of a bond | `nu_L`, `nu_L_dot`, `nu_L_tilde`, `nu_L_tilde_dot` | `nu`, `nu_tilde` |
| The leaf below a Tucker edge | `xi`, `xi_dot`, `xi_tilde`, `xi_tilde_dot` | `xi_U_dot` |
| The trunk above a Tucker edge | `eta_D`, `eta_D_dot`, `eta_D_tilde`, `eta_D_tilde_dot` | `eta`, `eta_tilde` |

- The first column is the base point's complete family: four sides in four grades.
- The second column is what the orthogonal frame adds. `mu_R` and `xi_U` occur only with a dot: no plain variable
  is formed there.
- Beside these 22 there are the input `w`, the seed `w_tilde`, the outputs and the blocks (section 6). Their
  names did not change.

## 4. The seven terms

Every contraction is one of seven terms. At order 0 each is one `einsum`; at a higher order it is the truncated
product over the order axis of the same contraction (target decision 2.9).

| Term | Contraction | What it makes |
| --- | --- | --- |
| `xi_term(B, w)` | `'io,Wo->Wi'` | a leaf: a Tucker core applied to an input |
| `mu_term(mu, G, xi)` | `'Wa,aib,Wi->Wb'` | the left of the next bond |
| `nu_term(G, xi, nu)` | `'aib,Wi,Wb->Wa'` | the right of the previous bond |
| `eta_term(mu, G, nu)` | `'Wa,aib,Wb->Wi'` | a trunk |
| `z_term(eta, B)` | `'Wi,io->Wo'` | an output: a trunk through a Tucker core |
| `V_term(eta, w)` | `'Wi,Wo->io'` | a block with the legs of a Tucker core |
| `H_term(mu, xi, nu)` | `'Wa,Wi,Wb->aib'` | a block with the legs of a TT core |

`B` stands for any Tucker core and `G` for any TT core. A TT core has the legs `a` (its left bond), `i` (its
Tucker edge) and `b` (its right bond); a Tucker core has the legs `i` and `o` (its mode edge).

## 5. The definitions

Each variable is defined twice: by a recurrence, and as a subnetwork. The check confirms that the two agree.

**Step functions.** One step of a recurrence is a step function. A step function is named by letter and grade
only, and serves both bases: the basis of what it returns follows from the cores it is given. `nu_step` given `R`
cores returns `nu`, and given `L` cores returns `nu_L`. `mu_dot_step` given `(R, U, H)` returns `mu_R_dot`, and
given `(L, L, L_dot)` returns `mu_dot`. The maintainer (2026-11-16, 00:23): "There is no need for extra step
functions if they literally do the exact same things with different arguments."

### 5.1 Plain

| Variable | Recurrence | As a subnetwork |
| --- | --- | --- |
| `xi[i]` | `xi_term(D[i], w[i])` | `D[i]` applied to the input `w[i]` |
| `mu[i+1]` | `mu_term(mu[i], L[i], xi[i])`, from `mu[0] = 1` | the `L` cores of sites `0 .. i`, each over its `D` core and its input |
| `nu[i]` | `nu_term(R[i], xi[i], nu[i+1])`, from `nu[d] = 1` | the `R` cores of sites `i .. d-1`, each over its `D` core and its input |
| `nu_L[i]` | `nu_term(L[i], xi[i], nu_L[i+1])`, from `nu_L[d] = 1` | the same with `L` cores in place of `R` |
| `eta[i]` | `eta_term(mu[i], U[i], nu[i+1])` | `U[i]` with `mu[i]` on its left bond and `nu[i+1]` on its right bond: `L` cores to the left, `R` cores to the right, and the leaf of site `i` left out |
| `eta_D[i]` | `eta_term(mu[i], L[i], nu_L[i+1])` | the same with `L[i]` in place of `U[i]` and `L` cores to the right: every TT core is an `L` core |

A plain variable can also be said to come out of one leg of one core: `mu` out of leg `b` of `L`, `nu` out of leg
`a` of `R`, `nu_L` out of leg `a` of `L`, `xi` out of leg `i` of `D`, `eta` out of leg `i` of `U`, `eta_D` out of
leg `i` of `L`. That is the store and the basis in its name.

### 5.2 Dot: a direction put in at one site

A dotted variable is a sum of terms. In each term the direction sits at one site of the contracted side, and the
sum is over that site. The direction is one tangent vector, and it can be given in two forms.

**The frame's form: the direction is `(V, H)`.** At the site where it sits, the direction is either `H` over the
`D` core, or `U` over `V` in place of the `D` core. (The direction's own core is `H` in the first case and `V` in
the second.) The TT cores before that site (to its left) are `L` cores,
and the TT cores after it are `R` cores.

| Variable | Recurrence | As a subnetwork |
| --- | --- | --- |
| `xi_U_dot[i]` | `xi_term(V[i], w[i])` | `V[i]` applied to the input `w[i]` |
| `mu_R_dot[i+1]` | `mu_term(mu_R_dot[i], R[i], xi[i]) + mu_term(mu[i], H[i], xi[i]) + mu_term(mu[i], U[i], xi_U_dot[i])`, from `0` | the left of bond `i+1` with the direction at one of the sites `0 .. i` |
| `nu_L_dot[i]` | `nu_term(L[i], xi[i], nu_L_dot[i+1]) + nu_term(H[i], xi[i], nu[i+1]) + nu_term(U[i], xi_U_dot[i], nu[i+1])`, from `0` | the right of bond `i` with the direction at one of the sites `i .. d-1` |
| `eta_D_dot[i]` | `eta_term(mu_R_dot[i], R[i], nu[i+1]) + eta_term(mu[i], H[i], nu[i+1]) + eta_term(mu[i], L[i], nu_L_dot[i+1])` | the trunk above Tucker edge `i` with the direction at one site. At site `i` itself it can only be `H[i]`; at any other site it is `H`, or `U` over `V`. When the direction is at another site, the TT core of site `i` is `R[i]` if the direction is to its left and `L[i]` if it is to its right |

Why the bases come out as they do: a bond to the right of the direction is in the `R` basis, because the cores
after the direction are `R` cores; a bond to its left is in the `L` basis. So the left of a bond that contains
the direction ends in the `R` basis (`mu_R_dot`), and the right of a bond that contains it begins in the `L`
basis (`nu_L_dot`). In `eta_D_dot` the open Tucker leg belongs to `R[i]`, `H[i]` or `L[i]`, all of which carry
the `D` basis there. In `xi_U_dot` the open leg is that of `V`, which is in the `U` basis.

**The raw form: the direction is `(D_dot, L_dot)` at the base point.** At the site where it sits, the direction
is either `L_dot` over the `D` core, or `L` over `D_dot` in place of the `D` core. Every other TT core is an `L`
core.

| Variable | Recurrence | As a subnetwork |
| --- | --- | --- |
| `xi_dot[i]` | `xi_term(D_dot[i], w[i])` | `D_dot[i]` applied to the input `w[i]` |
| `mu_dot[i+1]` | `mu_term(mu_dot[i], L[i], xi[i]) + mu_term(mu[i], L_dot[i], xi[i]) + mu_term(mu[i], L[i], xi_dot[i])`, from `0` | the left of bond `i+1` with the direction at one of the sites `0 .. i` |
| `nu_L_dot[i]` | `nu_term(L[i], xi[i], nu_L_dot[i+1]) + nu_term(L_dot[i], xi[i], nu_L[i+1]) + nu_term(L[i], xi_dot[i], nu_L[i+1])`, from `0` | the right of bond `i` with the direction at one of the sites `i .. d-1` |
| `eta_D_dot[i]` | `eta_term(mu_dot[i], L[i], nu_L[i+1]) + eta_term(mu[i], L_dot[i], nu_L[i+1]) + eta_term(mu[i], L[i], nu_L_dot[i+1])` | the trunk above Tucker edge `i` with the direction at one site. At site `i` itself it can only be `L_dot[i]` |

**One dot.** `nu_L_dot` and `eta_D_dot` appear in both tables. They are one variable: when `(D_dot, L_dot)` is
the lift of `(V, H)`, with no damping, the two forms give the same arrays (`scripts/review/subtree_gauge_check.py`).
The other four are one side in two bases: `mu_dot` and `mu_R_dot` are the left of a bond with the direction in
it, in the `L` and in the `R` basis; `xi_dot` and `xi_U_dot` are the leaf with the direction in it, in the `D`
and in the `U` basis. Two limits: with slack ranks `mu_R_dot` is the image of `mu_dot` but not the other way
round; and with curvature damping above zero the lifted direction is not exactly the tangent (not examined).

### 5.3 Tilde: the adjoint variables

The tilde variables are the adjoint variables of the transpose. They are the library's `deta_tilde`,
`tau_tilde`, `sigma_tilde` and `dxi_tilde` under new names, computed by the same recurrences. Part D of the check
compares them with the library's arrays, one by one.

**The seed.** The transpose takes one residual-like vector for each output: `w_tilde[k]` for mode `k`, of the
size of `z[k]`. The `d` of them together are called the seed on this page. They are the library's `ztildes`.

**One variable of each kind, combining all the modes.** There is one `mu_tilde`, one `nu_tilde`, one `xi_tilde`
and one `eta_tilde`: one array per bond or site, exactly as for the plain variables. There is not one set per
mode. `mu_tilde[i]` is a sum over the modes `k` to the left of bond `i`; the term of mode `k` is the network of
`mu[i]` with the input `w[k]` replaced by the residual-like vector `w_tilde[k]`. The recurrence carries that sum
along the chain in one sweep (the library's `_tau_tilde_step`: the carry through `L` and `xi`, plus `mu` through
`L` and `deta_tilde`). A drawing of a tilde variable shows one term of the sum; the variable is the whole sum.

| Variable | Recurrence | As a subnetwork |
| --- | --- | --- |
| `xi_tilde[i]` | `xi_term(D[i], w_tilde[i])` | `D[i]` applied to the seed `w_tilde[i]` |
| `mu_tilde[i+1]` | `mu_term(mu_tilde[i], L[i], xi[i]) + mu_term(mu[i], L[i], xi_tilde[i])`, from `0` | `mu[i+1]` with the seed in place of one of the inputs `0 .. i` |
| `nu_tilde[i]` | `nu_term(R[i], xi[i], nu_tilde[i+1]) + nu_term(R[i], xi_tilde[i], nu[i+1])`, from `0` | `nu[i]` with the seed in place of one of the inputs `i .. d-1` |
| `nu_L_tilde[i]` | the same through `L`, with `nu_L` | `nu_L[i]` with the seed in place of one of the inputs `i .. d-1` |
| `eta_tilde[i]` | `eta_term(mu_tilde[i], U[i], nu[i+1]) + eta_term(mu[i], U[i], nu_tilde[i+1])` | `eta[i]` with the seed in place of one input other than input `i` |
| `eta_D_tilde[i]` | `eta_term(mu_tilde[i], L[i], nu_L[i+1]) + eta_term(mu[i], L[i], nu_L_tilde[i+1])` | `eta_D[i]` with the seed in place of one input other than input `i` |

**The transpose is applied twice in a Newton step, to two different seeds.** With the weighted residual as the
seed it gives the gradient `g`. With the weighted `Jp` as the seed it gives `JtJp`. These are two applications of
one map, as in the library today, where the kind's `transpose` is called once with the residual and once with
`Jp`. Each application has its own tilde variables while it runs, and each is linear in its seed. Target
decision 3.3 is only about names: when one call computes both `g` and `JtJp`, which of the two applications does
the name `mu_tilde` refer to. It is not a question about the algorithm.

### 5.4 Tilde-dot: both

At the base point, with the raw form of the direction. Each term has the direction at one site and the seed in
place of one input, and the sum is over both. The two may be at the same site: `L_dot` over a `D` core that is
applied to the seed, or `L` over `D_dot` applied to the seed.

| Variable | Recurrence | As a subnetwork |
| --- | --- | --- |
| `xi_tilde_dot[i]` | `xi_term(D_dot[i], w_tilde[i])` | `D_dot[i]` applied to the seed `w_tilde[i]` |
| `mu_tilde_dot[i+1]` | six terms: `mu_term` of `(mu_tilde_dot, L, xi)`, `(mu_dot, L, xi_tilde)`, `(mu_tilde, L, xi_dot)`, `(mu, L, xi_tilde_dot)`, `(mu_tilde, L_dot, xi)`, `(mu, L_dot, xi_tilde)` at site `i` | the left of bond `i+1` with the direction at one of the sites `0 .. i` and the seed in place of one of the inputs `0 .. i` |
| `nu_L_tilde_dot[i]` | the six mirror terms with `nu_term` and `nu_L` | the right of bond `i`, likewise |
| `eta_D_tilde_dot[i]` | six terms: `eta_term` of `(mu_tilde_dot, L, nu_L)`, `(mu_dot, L, nu_L_tilde)`, `(mu_tilde, L, nu_L_dot)`, `(mu, L, nu_L_tilde_dot)`, `(mu_tilde, L_dot, nu_L)`, `(mu, L_dot, nu_L_tilde)` | the trunk above Tucker edge `i` with the direction at one site and the seed in place of one input other than input `i` |

## 6. Outputs and blocks

Their names did not change. They are written here with the new names of what they read.

| Name | Definition | What it is |
| --- | --- | --- |
| `z[i]` | `z_term(eta_D[i], D[i])` | the value: the whole network at the base point, with mode edge `i` open |
| `Jp[i]` | `z_term(eta_D_dot[i], D[i]) + z_term(eta[i], V[i])` | the direction's output: the whole network with the direction at one site and mode edge `i` open |
| `V_tilde[i]` | top coefficient of `V_term(eta_tilde[i], w[i]) + V_term(eta[i], w_tilde[i])` | a block with the shape of `V[i]`: the whole network with the leaf core under `U[i]` taken out, and the seed in place of one input, which may be input `i` |
| `H_tilde[i]` | top coefficient of `H_term(mu_tilde[i], xi[i], nu[i+1]) + H_term(mu[i], xi_tilde[i], nu[i+1]) + H_term(mu[i], xi[i], nu_tilde[i+1])` | a block with the shape of `H[i]`: the whole network with the TT core of site `i` taken out (`L` cores to its left, `R` cores to its right), and the seed in place of one input |
| `V_tilde_dot[i]` | top coefficient of `V_term(eta_D_tilde_dot[i], w[i]) + V_term(eta_D_dot[i], w_tilde[i])` | a block with the shape of `D[i]`: the network at the base point with `D[i]` taken out, the direction at one site and the seed in place of one input, which may be input `i`. At site `i` the direction can only be `L_dot[i]` |
| `H_tilde_dot[i]` | top coefficient of nine `H_term`s of `mu[i]`, `xi[i]`, `nu_L[i+1]`: the tilde on one of the three and the dot on one of the three. Three terms have both on the same variable (its tilde-dot); six have them on two different ones | a block with the shape of `L[i]`: the network at the base point with `L[i]` taken out, the direction at one site and the seed in place of one input. At site `i` the direction can only be `D_dot[i]` |

- `g = (V_tilde, H_tilde)` with the seed made from the weighted residual `r = z - y`.
- `JtJp` is the same pair of blocks with the seed made from the weighted `Jp`.
- `Kp = (V_tilde_dot, H_tilde_dot)`, at the base point, with the raw form of the direction. The two blocks keep
  the letters `V` and `H` for "the Tucker block" and "the TT block", although the cores taken out are `D[i]` and
  `L[i]`.
- A block is summed over the samples. "Top coefficient" is the readout of a tilde jet (target decision 2.4).
- On the corewise frame the gradient is `V_term(eta_D_tilde, w) + V_term(eta_D, w_tilde)` and the three
  `H_term`s with `nu_L`: the same formulas in the base point's basis. (These are the names as seen from a call on
  a manifold frame. In a call on the corewise frame itself they have no basis: section 9.)

**What each output reads.**

| Output | Reads |
| --- | --- |
| `z` | `xi`, `mu`, `nu_L`, `eta_D` |
| `g`, `JtJp` | `xi`, `mu`, `nu`, `eta`, and `xi_tilde`, `mu_tilde`, `nu_tilde`, `eta_tilde` |
| `Jp` | `xi`, `mu`, `nu`, `eta`, and `xi_U_dot`, `mu_R_dot`, `nu_L_dot`, `eta_D_dot` |
| `Kp` | `xi`, `mu`, `nu_L`; `xi_tilde`, `mu_tilde`, `nu_L_tilde`; `xi_dot`, `mu_dot`, `nu_L_dot`, `eta_D_dot`; `xi_tilde_dot`, `mu_tilde_dot`, `nu_L_tilde_dot`, `eta_D_tilde_dot` |

`eta_D_tilde` is read only by the gradient on the corewise frame, where it is spelled `eta_tilde`.

## 7. Types and shapes

**Three rules give the type of a variable from its name.**

- **The letter gives the variance.** `mu` and `xi` are vectors. `nu` and `eta` are covectors. (`w` is a vector;
  `z` and `Jp` are covectors.)
- **The basis gives the space.** `L` and `R` are the two bases of a bond; `D` and `U` the two of a Tucker edge.
- **The grade gives the order factor.** Plain and dot are jets. Tilde and tilde-dot are dual jets, stored with
  the order turned (target decision 2.4).

**The arrows** (target decision 9.5; `scripts/review/arrow_type_check.py`). A variable whose name has no basis
travels with the arrows of the frame: it is in the basis of the store it travels away from. A variable whose name
carries a basis travels against them, and so do the outputs `z` and `Jp`.

**The shape** of a variable at index `i` is `T + W + [K] + C + (size,)` on the ragged layout, with a leading
`(d,)` on the uniform layout.

| Group | Meaning |
| --- | --- |
| `T` | the order group: `(order+1,)`, or `()` at order 0 |
| `A` | the order group of an affine variable, in place of `T`: `(2,)`, or `()` at order 0. The affine variables are the input `w` and the leaves `xi`, `xi_dot`, `xi_U_dot`. The seed is a full dual jet, so `xi_tilde` and `xi_tilde_dot` carry `T` |
| `W` | the sample stack |
| `K` | the stack of directions. Present on every variable that depends on the direction: the dot and tilde-dot grades, `Jp`, and a tilde variable whose seed was made from `Jp` |
| `C` | the stack of base points |
| `size` | the size of the cut edge in the variable's basis: `rL_i`, `rR_i`, `nD_i` or `nU_i` |

| Variable | Sits on | Basis, size | Variance | Order factor | Shape at index `i` |
| --- | --- | --- | --- | --- | --- |
| `mu` | bond `i` | `L`, `rL_i` | vector | jet | `T + W + C + (rL_i,)` |
| `mu_dot` | bond `i` | `L`, `rL_i` | vector | jet | `T + W + K + C + (rL_i,)` |
| `mu_tilde` | bond `i` | `L`, `rL_i` | vector | dual jet | `T + W + C + (rL_i,)` |
| `mu_tilde_dot` | bond `i` | `L`, `rL_i` | vector | dual jet | `T + W + K + C + (rL_i,)` |
| `mu_R_dot` | bond `i` | `R`, `rR_i` | vector | jet | `T + W + K + C + (rR_i,)` |
| `nu` | bond `i` | `R`, `rR_i` | covector | jet | `T + W + C + (rR_i,)` |
| `nu_tilde` | bond `i` | `R`, `rR_i` | covector | dual jet | `T + W + C + (rR_i,)` |
| `nu_L` | bond `i` | `L`, `rL_i` | covector | jet | `T + W + C + (rL_i,)` |
| `nu_L_dot` | bond `i` | `L`, `rL_i` | covector | jet | `T + W + K + C + (rL_i,)` |
| `nu_L_tilde` | bond `i` | `L`, `rL_i` | covector | dual jet | `T + W + C + (rL_i,)` |
| `nu_L_tilde_dot` | bond `i` | `L`, `rL_i` | covector | dual jet | `T + W + K + C + (rL_i,)` |
| `xi` | Tucker edge `i` | `D`, `nD_i` | vector | jet, affine | `A + W + C + (nD_i,)` |
| `xi_dot` | Tucker edge `i` | `D`, `nD_i` | vector | jet, affine | `A + W + K + C + (nD_i,)` |
| `xi_tilde` | Tucker edge `i` | `D`, `nD_i` | vector | dual jet | `T + W + C + (nD_i,)` |
| `xi_tilde_dot` | Tucker edge `i` | `D`, `nD_i` | vector | dual jet | `T + W + K + C + (nD_i,)` |
| `xi_U_dot` | Tucker edge `i` | `U`, `nU_i` | vector | jet, affine | `A + W + K + C + (nU_i,)` |
| `eta` | Tucker edge `i` | `U`, `nU_i` | covector | jet | `T + W + C + (nU_i,)` |
| `eta_tilde` | Tucker edge `i` | `U`, `nU_i` | covector | dual jet | `T + W + C + (nU_i,)` |
| `eta_D` | Tucker edge `i` | `D`, `nD_i` | covector | jet | `T + W + C + (nD_i,)` |
| `eta_D_dot` | Tucker edge `i` | `D`, `nD_i` | covector | jet | `T + W + K + C + (nD_i,)` |
| `eta_D_tilde` | Tucker edge `i` | `D`, `nD_i` | covector | dual jet | `T + W + C + (nD_i,)` |
| `eta_D_tilde_dot` | Tucker edge `i` | `D`, `nD_i` | covector | dual jet | `T + W + K + C + (nD_i,)` |

The input `w[i]` is `A + W + (N_i,)` and the seed `w_tilde[i]` is `T + W + [K] + C + (N_i,)`, with `K` when it
was made from `Jp`. The outputs: `z[i]` is
`T + W + C + (N_i,)` and `Jp[i]` is `T + W + K + C + (N_i,)`. A block has the shape of the core it stands for,
the stack `C` included: `V_tilde[i]` is `C + (nU_i, N_i)` and `H_tilde[i]` is `C + (rL_i, nD_i, rR_(i+1))`. It has
`K` in front where it depends on the direction: always for `V_tilde_dot` and `H_tilde_dot`, and for `V_tilde` and
`H_tilde` when the seed was made from `Jp`. It has no `T` and no `W`: a block is summed over the samples and read
at the top coefficient.

## 8. The multiplier pairings

The tilde of a variable is the multiplier of the dot of the other side of the same edge, in the same basis.

| Tilde variable | Is the multiplier of | The edge, and its basis |
| --- | --- | --- |
| `mu_tilde` | `nu_L_dot` | a bond, `L` |
| `nu_tilde` | `mu_R_dot` | a bond, `R` |
| `xi_tilde` | `eta_D_dot` | a Tucker edge, `D` |
| `eta_tilde` | `xi_U_dot` | a Tucker edge, `U` |

The same statement for the direction: at a cut bond, the direction's sampled network is `mu_R_dot` paired with
`nu` (the direction is on the left) plus `mu` paired with `nu_L_dot` (the direction is on the right). At a cut
Tucker edge it is `xi_U_dot` paired with `eta` plus `xi` paired with `eta_D_dot`.

## 9. On the corewise frame

The corewise frame `(B, G, G, G)` has one basis on each edge, so a call on it uses the four letters and their
grades with no basis in the name. Its `nu` is what a call on a manifold frame calls `nu_L`, its `eta` is
`eta_D`, and its dots are those of the raw form. Whether such a call also accepts the spellings with a basis
is not decided.

## 10. Old names to new

"In this folder" is the name used in the documents and prototypes of this folder before 2026-11-15, 23:23. "In
the library today" is the array's name in the kernels of `backend/probing.py`; the jetted kernels add `_jets`.

| New name | Side, basis | In this folder until now | In the library today |
| --- | --- | --- | --- |
| `mu` | left of a bond, `L` | `mu` | `mus` |
| `mu_dot` | left of a bond, `L` | no name; the prototype's `c_mu_dot` | `mus_dot`, in the two-hole kernel |
| `mu_tilde` | left of a bond, `L` | `mu_tilde` | `tau_tildes` |
| `mu_tilde_dot` | left of a bond, `L` | `mu_tilde_dot` | `tau_tildes_dot` |
| `mu_R_dot` | left of a bond, `R` | `mu_dot`, "the frame's `mu_dot`" | `sigmas` |
| `nu` | right of a bond, `R` | `nu` | `nus` |
| `nu_tilde` | right of a bond, `R` | `nu_tilde` | `sigma_tildes` |
| `nu_L` | right of a bond, `L` | `gamma` | `nus`, re-swept through the point's cores in the two-hole kernel |
| `nu_L_dot` | right of a bond, `L` | `gamma_dot`, and `nu_dot`, "the frame's `nu_dot`" | `nus_dot` in the two-hole kernel, and `taus` |
| `nu_L_tilde` | right of a bond, `L` | `gamma_tilde` | `sigma_tildes`, in the two-hole kernel |
| `nu_L_tilde_dot` | right of a bond, `L` | `gamma_tilde_dot` | `sigma_tildes_dot` |
| `xi` | leaf, `D` | `xi` | `xis` |
| `xi_dot` | leaf, `D` | no name; the prototype's `c_xi_dot` | `xis_dot`, in the two-hole kernel |
| `xi_tilde` | leaf, `D` | `xi_tilde` | `deta_tildes` |
| `xi_tilde_dot` | leaf, `D` | `xi_tilde_dot` | `deta_tildes_dot` |
| `xi_U_dot` | leaf, `U` | `xi_dot`, "the frame's `xi_dot`" | `dxis` |
| `eta` | trunk, `U` | `eta` | `etas` |
| `eta_tilde` | trunk, `U` | `eta_tilde` | `dxi_tildes` |
| `eta_D` | trunk, `D` | `zeta`; before 22:11 no name (written inline in `z_step`) | `etas`, when `t3_probe` runs on the point's own cores |
| `eta_D_dot` | trunk, `D` | `zeta_dot`, and `eta_dot`, "the frame's `eta_dot`"; the prototype's `c_eta_dot` | `etas_dot` in the two-hole kernel, and `detas` |
| `eta_D_tilde` | trunk, `D` | `zeta_tilde` | `dxi_tildes`, in the corewise transpose |
| `eta_D_tilde_dot` | trunk, `D` | `zeta_tilde_dot`; the prototype's `eta_tilde_dot` | `dxi_tildes_dot` |

**Unchanged:** `w`, `w_tilde`, `z`, `r`, `Jp`, `f`, `g`, `JtJp`, `Kp`, the blocks `V_tilde`, `H_tilde`,
`V_tilde_dot`, `H_tilde_dot`, the stores `D`, `U`, `L`, `R`, the direction `(V, H)`, and the seven term
functions.

**Four things to watch when reading an older text.**

- **Two names changed their meaning.** `mu_dot` and `xi_dot` used to mean the frame's variables. They now mean
  the base point's, in the `L` and `D` bases. The frame's are `mu_R_dot` and `xi_U_dot`.
- **Two pairs of names became one.** The old `nu_dot` (the frame's) and `gamma_dot` are both `nu_L_dot`. The old
  `eta_dot` (the frame's) and `zeta_dot` are both `eta_D_dot`.
- **Two letters are retired.** `gamma` is `nu_L`. `zeta` is `eta_D`.
- **The library today uses one name for two variables in several rows** (`nus`, `sigma_tildes`, `etas`,
  `dxi_tildes`), telling them apart by which cores were passed.

## 11. Not decided

Settled since this page was first written: the raw core direction is `(D_dot, L_dot)` (the maintainer,
2026-11-16, 00:30; it had been `(B_dot, G_dot)` here, and is `core_directions` in the library, the first argument
of the kind's `corewise_hessian`); and step functions are named by letter and grade (section 5).

- **A call on the corewise frame** (section 9). The maintainer considered this principle (2026-11-16, 00:03):
  "The engine is designed for the manifold setting. When corewise operations are required, this is done by having
  the caller of the engine put in the B and G cores, and it is their responsibility to track the altered meaning
  of the variables. The engine remains manifold-focused." After the pros and cons he set it aside (00:23): "The
  against is more convincing, particularly the extra sweep cost. This will be a real performance hit and is
  unacceptable." The cost meant: with `(B, G, G, G)` put in, the engine does not know that `nu` and `nu_L` are
  one array, and sweeps twice when both `z` and `g` are wanted. So a call on a corewise frame must not sweep
  twice. How the engine is told, and which spellings such a call accepts, are open.
- **Which application of the transpose a tilde name refers to**, when one call computes both `g` and `JtJp`
  (target decision 3.3; section 5.3).
- What the dots that `Kp` reads are when the curvature damping is above zero (section 5.2).
- In a formula the index is written as a subscript after the basis, as in `L,3`. This is how the drawings were
  made and shown; the maintainer asked for the basis as a subscript and did not speak of the index.
