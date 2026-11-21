SIGN OFF

# P4c review, round 1: the reference module in several variables

Reviewer, 2026-11-19, on the working tree of `graded-messages` (the implementer's uncommitted changes on
`3ccff5da`). No blocking and no should-fix finding. Two notes for Claude: one on the plan's wording of the bridge,
for P6, and one on test time.

I checked the module by a route of my own that shares no code with it. Each variable is computed from its
recurrence, every step a truncated box product by P4a's reviewed `jet_contract`, and compared with the module's
evaluate-and-transform. They agree to 2e-15.

## The five points of the brief

**1. The bridge to the old kernels: the conversion is right, and for the right reason.** Derived from the
definitions, not from fitting:
- **The forward map.** The old kernels return derivatives: `y_old^(t) = d^t/ds^t probe(X + sP)|_0 = t! z_t`, with
  `z_t` the Taylor coefficient. So `z_new = z_old / t!` slot by slot, which is what the test asserts.
- **The transpose.** The old transpose is the transpose of the old linear map `p -> (t! (Jp)_t)_t`, so
  `old_T(c) = Σ_t t! J_tᵀ c_t`. The new `g` is the transpose seeded by the dual jet: the pairing is the corner
  coefficient of the product with the seed stored turned, which is `Σ_t ω_t² r_t (Jp)_t`. So
  `g = Σ_t J_tᵀ ω_t² r_t`.
- **So the conversion is** `old_T(c) = g` exactly when `c_t = ω_t² r_t / t!`, in **forward** order. The turn
  belongs to the new seed's *storage* (the corner pairing). The old kernel did its own binomial correlation and
  takes its residual forward. So turning `c`, or multiplying by `t!` instead of dividing, must fail, as the
  implementer found.
- **Consistency with the weights:** an old derivative-mode objective `½ Σ ω_d,t² (t! r_t)²` is the new
  `½ Σ ω_t² r_t²` exactly when `ω_t = t! ω_d,t`, which is `derivative_weight_to_taylor_weight`. The conversion is
  consistent with it.

The test (`TestAgainstTheOldKernels`) asserts exactly this, on three problems at orders 1 to 3, with a nonuniform
weight. That weight is what makes the `t!`, the squaring and the turn each detectable.

*Note for Claude (not a finding):* the plan's P4c text and section 5.1's bridge say "the conversions (`t!` per slot,
the turn)". For the old transpose the right conversion is `ω² r / t!`, forward, with no turn; the turn is inside
the new `g`. The bridge of P6 (`tests/bridge/test_old_kernels_bridge.py`) will need the same conversion. One
sentence in P6's slice text would save its implementer the derivation.

**2. The node count: exact everywhere, tight for apply only.** From the degrees:
- An input clipped to the box has degree at most `a_j - 1` in `s_j`, and the seed `n_j`. A direction carries no
  `s`.
- **Probe:** a plain or dot variable is a product of at most `d` inputs, so of degree at most `d (a_j - 1)`. A
  tilde or tilde-dot variable puts the seed in place of one input, so at most `(d - 1)(a_j - 1) + n_j`.
- **Apply:** the seed multiplies the ones at bond `d`, so `nu_tilde[0]` carries the seed and all `d` inputs:
  `d (a_j - 1) + n_j`.
- So `M_j = d (a_j - 1) + n_j + 1` exceeds every degree, and the DFT on `M_j` roots of unity is exact. It is also at
  least `n_j + 1`, so the box can be read.
- **Measured** (a scratch copy with `M_j` one less, against my recurrence route): probe stays exact at `(2,)`,
  `(3,)` and `(1, 2)`; apply breaks at all three, in `nu_tilde[0]`, as derived. The implementer's account agrees.

**3. The declared jet sizes.** `test_the_shapes_are_those_of_the_table` asserts them for every name on two problems
over the whole jet matrix:
- `w` as given;
- the leaves at `min(J_in_i, F)`;
- everything else `F`;
- `r_block_sumsq` the weight's shape;
- the blocks the cores' shapes.

My route checks the shapes of the eight variables it computes, exactly, at mixed sizes (inputs above, at and below
the box, and a mode `(1, 1)`).

**4. The trims lose no distinct coverage.**
- **`TestJetsTwoWays` on the four-site problem** keeps the line at `(3,)` (m = 1) and the mixed modes (m = 2, with
  sizes that differ per mode and per axis, and clipping on one axis). The dropped cases there:
  - the plane is the mixed modes' special case;
  - the curve and the mixed sizes at `(2,)` are m = 1 cases with per-mode sizes, which the mixed modes contain
    per axis.

  The three-site problem still runs every jet case.
- **`TestKp`'s jax comparison** drops the line at `(1,)` (the same code path as `(3,)`) and the plane (the mixed
  modes are the richer m = 2). Kp's other three tests still run the plane.

**5. Independence.**
- `independence_violations` and the two tests built on it are byte-identical to `HEAD` (diffed).
- The module imports only `math`, `typing`, `dataclasses`, `numpy`, the three dense modules and `common`.
- `_read_jet_shape` uses `(int, np.integer)` minus `bool`, where `contractions._read_order` uses
  `numbers.Integral`: the same set for every input a caller can give. A test compares them on 15 inputs.

## Independent checks

- **The recurrences by box products** (`defs_by_box.py`, seeds 0, 1 and 2).
  - *What:* the plain variables `xi`, `mu`, `nu`, `eta` and the tilde variables `xi_tilde`, `mu_tilde`,
    `nu_tilde`, `eta_tilde`, computed from their recurrences in `SAMPLE_VARIABLES.md` (sections 5 and 5.3), every
    step one `jet_contract` call. That code is committed and reviewed, and the reference module never touches it.
  - *Coverage:*
    - on a frame of four unrelated stores `(D, U, L, R)`, so each basis is in its place;
    - probe and apply, `d` = 3 and 4;
    - a line at `(2,)`; mixed sizes at `(3,)`; a curve at `(2,)`; the plane at `(1, 2)`; the jet matrix's modes
      `(3, 2)`, `(1, 1)`, `(2, 3)` at `(1, 2)`; a second variable of order 0 at `(2, 0)`; and `(1, 1, 1)`.
  - *Result:* every array is equal to the module's in shape, and in value to 2.1e-15 at worst.
- **The grid tightness**, by the mutant `M_j - 1`, on the same route: as stated in point 2.
- **Mutations** (`mutate_p4c.py`, a scratch copy, `python -B`, deleted after), against the whole
  `tests/test_sampling_definitions.py`. All four caught:
  - the corner pairing without turning the seed back;
  - the weight applied after the turn, so on the wrong slots;
  - the nodes conjugated (`exp(-2πi k / M)`);
  - the inputs not clipped before evaluation.

  The implementer's 17 cover the rest (16 caught, the one missed equivalent).
- **Runs** (single process):
  - `tests/test_sampling_definitions.py` and `test_jax_absent.py`: 72 passed, 5,518 subtests, 193 s.
  - The module doctests of `sampling_definitions`: 37 passed, from real runs.

## Form

- The signature of all 37 functions has the shape comments, `order=()` after `i`, and the keyword-only arguments.
  `_Setting`'s fields carry their shape comments.
- Dispatch is inferred from the arrays (`get_backend(False, tree_contains_jax(...))`).
- Host constants (the powers of the nodes) are numpy and are cast once by `_complex` to the computation's precision
  and module.
- No `jets` or `contractions` import.

## A note on time (not a finding)

The file takes 193 s here and, by the handover, 224 s on the floor (double P4b's). The largest part is
`TestJetsTwoWays` (about 67 s). If CI time matters later, the four-site half of it is the one cut I would accept:
the mixed modes there could go, keeping the line at `(3,)`, since the three-site problem covers the mixed modes.

The working tree is as I found it, apart from this file. The scratch copies are deleted, and nothing of mine is
left running.
