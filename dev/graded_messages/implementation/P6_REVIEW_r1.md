SIGN OFF

# P6 review, round 1: Engine B, the tilde grade

Reviewer, 2026-11-20, on the working tree of `graded-messages` (the implementer's uncommitted changes on
`9a94b887`). No blocking and no should-fix finding; two minor ones.

The tilde grade is right:
- it agrees with the P4c reference;
- `g` agrees with a transposed **dense** Jacobian I built coordinate by coordinate, with no engine or reference code;
- the short node is the slice `[c - a + 1, c]` of the full `eta_tilde` on every axis, mixed sizes included;
- the bridge's conversion of the old adjoint variables follows from the definitions, not merely from fitting.

## The six points

**1. The short node and `V_tilde`.** `eta_tilde_top_step` computes `a = min(J_in_i, F)` from `w[i]`'s static shape and
forms exactly the slots `c - a + 1 + j`, `j` in the box of `a`. Each is the two readouts at that lowered corner,
stacked row-major.

`V_tilde_step` then reads the stack at the corner `a - 1`, the leaf's jet shape. That is the identity
`Σ_k eta_tilde[c - k] w[k] = coefficient (a - 1) of S * w` from my design review, axis by axis.

Checked three ways:
- **By the step itself.** I called `eta_tilde_top_step` on the engine's own `mu_tilde`, `mu`, `nu_tilde`, `nu` and
  compared it with the slice of the engine's full `eta_tilde`, at every site. The cases include the jet matrix's
  modes `(3, 2)`, `(1, 1)`, `(2, 3)` at `(1, 2)`, mixed sizes at `(2,)`, and `(1, 1, 1)`. Agreement to 1e-15.
- **Through `g`, against the dense Jacobian** (point 3).
- **Two mutants of my own,** both caught by the slice's tests:
  - the box taken from `order` instead of the input;
  - the input term read at the full corner instead of `a - 1`.

**2. The bridge's conversion of the old adjoint variables, derived.** The old library's objective in derivative form
is `L = Σ_t <c_t, y^(t)>`, with `y^(t) = t! z_t`, so `L = Σ_t t! <c_t, z_t>`. The forward-seed conversion of P4c sets
`t! c_t = ω_t² r_t`, which makes `L` the engine's pairing.

For a variable `X` of the network:
- *The old adjoint* is the derivative of `L` with respect to `X` in **derivative** form, `X^(r) = r! X_r`. So it is
  `(1/r!) ∂L/∂X_r`.
- *The new adjoint:* the engine's tilde variable that multiplies `X` (`SAMPLE_VARIABLES.md` section 8) is stored
  turned, and the pairing is the corner coefficient. So `∂L/∂X_r` is that tilde variable at slot `n - r`.
- *Hence:* old adjoint at slot `r` = new multiplier at slot `n - r`, divided by `r!`.

Applied to the old names (section 10):
- `deta_tilde` (of `deta` = `eta_D_dot`) is `xi_tilde`;
- `tau_tilde` (of `tau` = `nu_L_dot`) is `mu_tilde`;
- `sigma_tilde` (of `sigma` = `mu_R_dot`) is `nu_tilde`;
- `dxi_tilde` (of `dxi` = `xi_U_dot`, two slots for a line) is `eta_tilde`, with `u! = 1` for `u = 0, 1`.

That is exactly the conversion the bridge asserts, and the pairings make it the only candidate, so the implementer's
script finding it alone among five is what the mathematics predicts.

**3. `g` against the derivative of `f`, and against the dense Jacobian.**
- The implementer's `TestGradient` (the central difference of `f` along the reference's `Jp`) is the adjoint identity.
- My own check is independent of both the engine and the reference:
  - I built the dense tangent tensor of each **coordinate** direction, a unit entry of `V_j` (with `U_j` over it, `L`
    to the left, `R` to the right) or of `H_j`, through `TuckerTensorTrain(...).to_dense()`.
  - I probed it at the input jets with my roots-of-unity oracle, and paired it with `ω² r`, `r` from the oracle's
    `z`.
  - That gives every entry of `g = Jᵀ ω² r`. It matches the engine's `g` to 1e-15 at `()`, at `(2,)` with mixed
    sizes, and at `(1, 2)` with the plane and with the jet matrix's mixed modes, on two seeds.

**4. The generic twin.**
- It imports `numpy` and `itertools` only, and the AST test pins it.
- It computes `V_tilde` from `out['dB'][i]['z'][corner]`, its own `gmul` on the full `E` (`eta_tilde`) and the inputs
  lifted to the box. So the long way round, as asked.
- Its value is taken at `(D, L, L, L)`, the corewise form of the base point, which is where `z` is defined (`z` at
  the base point `(D, L)`). Its adjoint is on the manifold frame. Departure 4 is right.

**5. `HeldVariables`' 18 slots** are the public computed variables:
- P5's eight;
- `f` and `r_block_sumsq`;
- `w_tilde`, `xi_tilde`, `mu_tilde`, `nu_tilde`, `eta_tilde`, `V_tilde`, `H_tilde`;
- `g`.

`_check_form` checks a held value recursively against `_expected`: one array, a tuple over the index, or `g`'s pair.
The tests cover wrong forms of `f`, `r_block_sumsq` and `g`, and the implementer's mutant (`g`'s `H` checked with
`V` shapes) is caught.

**6. "`g` bit-identical whether or not `eta_tilde` is wanted".** Minor finding 1:
- Under the maintainer's rule the thing to pin is *structural*: the route of `V_tilde` does not depend on `want`.
  That is better asserted on the route itself, exactly: the closure for `('g',)` and for `('g', 'eta_tilde')` take
  the same `V_tilde` row, and the hook records the same `eta_tilde_top_step` calls. The values then compare to
  tolerance.
- The bitwise assertion passes today and is not wrong, but it tests a stronger property than the design promises
  (determinism of the same arithmetic). A later legitimate change could break it, for example a reordered summation
  in P11's scan, while the route stays the same.
- *Resolve:* replace it with the route assertion plus a tolerance on the values, or keep the bitwise one and add the
  route assertion beside it.

## The departures

All five are sound:
1. **`start` on the row** (my finding A). A start by node name is left to P9, and an undeclared start raises.
2. **`w_tilde_step` for the probe kind:** the kind's own seed step is the natural place. P9 adds apply's, which seeds
   the end bond, not the inputs.
3. **The new strings written with `K`:**
   - `STRINGS` lists what the library writes, guard (ii), so the plain readout of the plan, which no call site
     writes, is rightly not in it.
   - The family `eta_top` with the two `W`-keeping readouts, and the pinned counts that follow, are consistent.
4. **The twin's value at `(D, L, L, L)`:** right (point 4).
5. **The bridge's `n = 0` against `probing`'s plain functions:** right. `sampling_derivatives` has no order 0, and
   `t3_probe` and `tv_probe_transpose` given `ω² r` are the old order-0 pair.

## Minor finding 2

`tests/bridge/__pycache__/` exists in the untracked directory `tests/bridge/`. Commit `tests/bridge/` by its file
name, not the directory (`ORCHESTRATION.md`, "`git add` by name").

## What I ran (single process)

- **`p6_check.py`** (mine), seeds 0 and 1. It checks:
  - every new name against the reference;
  - the short node against the slice;
  - `g` against the transposed dense Jacobian;
  - `g` with and without `eta_tilde` wanted, to tolerance.

  It runs on seven jet cases (the jet matrix and `(1, 1, 1)`), `d = 3`, and `d = 4` with tails whose end bonds
  differ. Worst error 2.6e-15.
- **Current env:** the four engine test files, the bridge, `test_jet_contract`, `test_import_graph` and the module
  doctests of `sampling_variables` and `sampling_steps`: 102 passed, 1 xfailed, 15,207 subtests.
- **Floor env (`tttt`):** the four engine test files and the bridge, 46 passed.
- **Mutants:** `mutate_p6.py`, two, both caught; the copy deleted.

The working tree is as I found it, apart from this file. Nothing of mine is left running.
