SIGN OFF

# P7 review, round 1: Engine C, the dot grade of the frame form

Reviewer, 2026-11-20, on the working tree of `graded-messages` (the implementer's uncommitted changes on `3cae91e5`).
No blocking and no should-fix finding; one minor.

The frame-form dots, `Jp`, `Jp_norm_sq` and `JtJp` are right:
- **against the P4c reference**, element by element over a stack of two directions;
- **`Jp` against a dense tangent tensor** of my own, built from the definition of a tangent vector and probed by my
  roots-of-unity oracle, with no engine or reference code;
- **the per-`K` identities** hold to rounding;
- **a held public `nu_L_dot` or `eta_D_dot` never reaches `Jp`**;
- **the structural `K` check** catches a size-1 `K` on one `H[i]`.

Built to my design review: private frame nodes, public route rows, the `K` check, `_base`.

## My checks (`p7_check.py`, seeds 0 and 1)

The cases: `d = 3`, and `d = 4` with tails whose end bonds differ, at four jet cases (`()`; `(3,)` with a line; the
jet matrix's mixed modes at `(1, 2)`; mixed sizes at `(2,)`), a nonuniform weight, `K = (2,)`:

- **The seven new names** (`xi_U_dot`, `mu_R_dot`, `nu_L_dot`, `eta_D_dot`, `Jp`, `Jp_norm_sq`, `JtJp`) for each
  direction `k` against the reference called with `(V[k], H[k])`. Worst 8.6e-15.
- **`Jp` from a dense tangent tensor.**
  - The tangent tensor is `Σ_j` (`H_j` with `L` cores to its left, `R` to its right, `D` below) plus (`U_j` over
    `V_j`, the same sides), each through `TuckerTensorTrain(...).to_dense()`.
  - It is probed at the input jets on a grid of roots of unity, one einsum per node.
  - It agrees with the engine's `Jp` for each `k`. This checks the four dot recurrences end to end, and the bases of
    each leg, independently of both the engine and the reference.
- **The identities, per `k`, to rounding:**
  - `<Jp_k, ω² r> = <p_k, g>`;
  - `<p_k, JtJp_k> = Jp_norm_sq_k`;
  - `JtJp_k = g` with `Jp_k` held as `r` and no data.
- **The frame node design, by behaviour.** A held public `nu_L_dot` and `eta_D_dot`, changed (`2x + 1`), leave `Jp`
  unchanged.
- **The `K` check.** `H[1]` with `K = (1,)` beside `V[0]` with `K = (2,)` raises a `ValueError` naming `H[1]`. That
  is the silent broadcast my design review found, now an error.

Together with my P6 check of `g` by the transposed dense Jacobian, this closes `JtJp`: it is `g`'s adjoint seeded by
`ω² Jp`, `Jp` is right against the dense tangent tensor, and the identity `JtJp_k = g(r = Jp_k)` holds.

## The departures

1. **`Jp_norm_sq` has `carries_K=True`.** Right. The flag means "this row runs with the call's `K`", and
   `Jp_norm_sq` keeps `K + C` (step 7). The shape table reads `K` from the flag, so it must be `True`.
2. **The `K` check also counts each operand's total axes.** Right. A prefix test alone passes a core with one axis too
   many or too few when `K + C` happens to be its prefix. The count makes "a wrong count" an error at the check, with
   the operand named, not deep in `jet_contract`.
3. **`JtJp` has no `needs` of its own.** Minor finding 1 below.
4. **The bridge checks the frame dots through `Jp` and `JtJp`**, not through `compute_sigma_jets` and its kin.
   Acceptable:
   - The dots enter `Jp` linearly, and `.forward` and `.transpose` are the old interface the users had.
   - The dots themselves are pinned by the definition tests against the reference and, independently, through my
     dense `Jp`.
   - The old kernels' derivative-normalized inputs would need their own conversion with nothing gained.
   - The bridge's docstring says so.

## Minor finding

**1. The missing-argument message for `JtJp` names a chain of private nodes.** Measured:
`'Jp' needs the argument variations, which was not given (needed through JtJp -> _Jp.V_tilde -> _Jp.eta_tilde_top
-> _Jp.mu_tilde -> _Jp.xi_tilde -> _Jp.w_tilde -> Jp)`.
- It is correct, and a user can read the cause, but the six private names are noise.
- `Jp_norm_sq`'s message (`through Jp_norm_sq -> Jp`) and `nu_L_dot`'s (`'nu_L_dot' needs the argument variations`)
  are clean.
- *Resolve:* give the twin of `g` the need explicitly, `needs={'variations'}` on `JtJp` (one argument in
  `_jp_twins`). The closure then stops at `JtJp` with "'JtJp' needs the argument variations". Nothing else changes:
  every path to `Jp` needs `variations` anyway.

## Form

- **The dot steps** are named by grade and serve both forms (P8's raw rows will pass `L`, `L_dot`, `L` and `nu_L`).
  - The argument names and comments say which core is which: `G_left`, `G_right`, `G_up`, `H`.
  - The strings put `K` on exactly the direction's operand.
- **`identity_step`** is the one trivial step, as proposed.
- **The starts** take `KC`, the stacks after `W`.
- **`_jp_twins`** derives the twins mechanically, with nothing hand-written but the docstring table. `_base`,
  `_letter` and `_table_name` read through the prefix, so `'_Jp.mu_tilde'` gets letter `mu`.

## What I ran (single process)

- `p7_check.py`, two seeds: all OK.
- The missing-argument messages, by a throwaway script (deleted).
- **Current env:** the four engine test files, the bridge, `test_import_graph`, and the module doctests of
  `sampling_variables` and `sampling_steps`: 62 passed, 2,170 subtests.
- **Floor env (`tttt`):** the four engine test files and the bridge, 52 passed.

The implementer's 17 mutants include the five I listed in the design review. I made no further mutants: my oracle
checks above test the same paths from outside.

The working tree is as I found it, apart from this file. Nothing of mine is left running.
