NOT YET

# M2 part 1 review, round 1: the independent oracles and the pre-M2 battery

Reviewer, 2026-11-22, on the uncommitted part 1 on `m2-wip` (over `441acae9`) and `M2_part1_HANDOVER.md`. My
scripts and outputs are in `scratchpad/m2rev/`:
- `check_ref.py`: the reference against my brute force;
- `mutants_p1.sh` and `mutants_p1.out`: my mutants;
- `r1demo.py`: the surviving mutant shown.

**Verdict: NOT YET, for one should-fix test gap (finding 1).** It is small. Everything else holds:
- the reference's full-form reading agrees with my own brute force, and all its refusals fire;
- `oracle.py`'s new functions are right and pinned;
- the engine-against-reference tests changed only the reference's `weight=` and keep every outcome and count;
- the battery is deterministic bytewise, its base is `85c6e54d` exactly, and its `--compare` expectations are those
  of my design review's finding 2.

I agree with Claude's ruling on the scope: the reference module is part 1's subject, and nothing in the library
calls it.

## 1. The reference against my own brute force (`check_ref.py`)

A random frame `(D, U, L, R)`, `W = (3,)`, a direction `(V, H)`. The data `y` and the weight are drawn with every
axis at random either 1 or `z`'s size. Probe weights are given alternately as a tuple and as one mode-leading array,
whose edge is 1 so that the modes agree. My expansion uses `np.repeat`, not broadcasting:
- `f = ½ Σ ω²(z − y)²`;
- `Jp_norm_sq = Σ ω² Jp²`;
- `w_tilde = flip(ω²(z − y))`,

with `z` and `Jp` taken from the reference's own dense functions.

- **Values: 36 cases** (probe and apply; orders `()`, `(2,)`, `(1,1)`; 6 draws each). All agree, `f` and
  `Jp_norm_sq` to `1e-12`, `w_tilde` to `1e-13`.
- **The refusals: 11 of 11 raise.**
  - probe: the block weight `(d,)+F`, another rank, an axis neither 1 nor the size, a mode-leading weight with a
    mode axis of 1, `d−1` arrays, a `y` without `W`, a `y` with a mode axis of 1, a `y` larger than `z`;
  - apply: the block weight `F`, a tuple weight, a weight larger than `z`.
- **No `jets` import** in `sampling_definitions` (an AST scan), so its independence holds.

## 2. `oracle.py`

I read `expand`, `_modes`, `S_taylor`, `weighted_residual_full` and `misfit_full`:
- `expand` writes size-1 axes out by `np.repeat` and refuses anything else;
- `S_taylor` divides `_poly_derivs`' derivatives by `t!`;
- `misfit_full` is `½ Σ (ω(S_taylor − y))²`.

All are correct, and additive: the old forms are untouched. `test_dense_oracle.py` pins them with no library call.
The old-form bridge (`ω_t = w_t · t!`) is the right identity for derivative data.

## 3. The engine-against-reference tests, and the tally

- **The diffs** of `test_sampling_variables`, `_apply`, `_entries`, `_corewise` and `_stacks` change only the
  reference call's `weight=`, to `Problem.full_weight` or `blocks_to_weight`, plus one import. The engine keeps its
  blocks.
- **The other test edits** in `test_sampling_definitions`, which I read: `Jets.weighted` now reads the full form by
  equal-rank broadcasting; the block-sum shape assertions now name the block shape; the `(3, 0)` case puts the
  size-1 jet axis after the mode axis; the old-kernel test passes the full form to the reference.
- **My own tally** of M1's `-v` log against part 1's: 11 nodes differ, all expected.
  - 10 are new: 7 in `test_dense_oracle` and 3 in `TestTheWeightIsOfTheTypeOfZ`.
  - The bridge test's 10 → 40 is M1's round-1 widening, made after M1's log.
- **Subtest counts per file:** `test_sampling_variables` 2362, `_apply` 4003, `_entries` 733, `_corewise` 2168,
  `_stacks` 2530, all unchanged. `test_sampling_definitions` goes 5586 → 5744, exactly the 155 + 3 new ones.

So no comparison was dropped, and the old engine now pins the new reading over the jet matrix. One dependency to
know: the reference's weight in these tests is built by `blocks_to_weight`, which is library code. M1 pinned it
against loops, so this is acceptable. The brute-force class does not use it for its own expectations.

## 4. Mutants

The implementer's nine are not rerun. My nine, against `test_dense_oracle`, `test_sampling_definitions` and
`test_sampling_variables`:

| Mutant | Result |
| --- | --- |
| **R1: the reference accepts a data-space axis where `z`'s axis is 1 (`a not in (1, b) and b != 1`)** | **SURVIVES** (finding 1) |
| R2: `r = z + y` | killed (889) |
| R3: a mode count of 1 accepted (broadcast over the modes) | killed (2) |
| R4: `f` without the half | killed (231) |
| R5: `ω⁴` | killed (1007) |
| O1: `misfit_full` without the half | killed (58) |
| O2: `oracle`'s weight applied squared | killed (53) |
| O3: `S_taylor` divides by `(t!)²` | killed (9) |
| O4: `expand` accepts any size | killed (2) |

## 5. The battery

- **Determinism.** I compared `battery_old.npz` with `battery_old2.npz` bytewise, key by key: 25,468 arrays, 0
  differences.
- **The base.** `scratchpad/m2impl/base/t3toolbox` equals `git archive 85c6e54d t3toolbox` (`diff -rq`).
- **Coverage.**
  - The engine: every public variable through `computed_names`; three kinds; manifold and corewise; ragged and
    uniform, with a ragged `y` into the uniform engine, the path whose packing changes in part 2; five jet shapes;
    `K`; `C` with a shared `y`; `sum_over_samples` both ways.
  - The oracle: all ten outputs, on full and drawn batches, with a packed batch on the uniform layout.
  - Derivative data.
  - Optimizer runs in both call forms.
  - One jitted run.
- **`--compare` follows my finding 2:**
  - bitwise by default (so `z`, `Jp`, `r_block_sumsq`, `y_block_sumsq` are included);
  - `f`, `Jp_norm_sq`, `f_resolution` to `1e-13`;
  - the jitted run to `1e-12`;
  - the optimizer runs: iteration counts and stop reasons exact, losses and iterates to `1e-10`, with the knife-edge
    rule stated.
- **One minor gap.** The `mc_sgd` runs record only `x` and `n_iter`: their stats have no `losses` or stop reason under
  those names. Store their loss history under whatever key `mc_sgd` uses, if it has one, so that the comparison
  sees more than the endpoint.

## Findings

### 1. SHOULD FIX: the reference's "neither 1 nor `z`'s size" check is untested where `z`'s axis is 1

**Evidence.** Mutant R1 lets a data-space axis of any size stand against a size-1 axis of `z`, which is a weight or
`y` larger than `z`. It survives all three files. It is not an equivalent mutant (`r1demo.py`):
- an apply problem with `W = (1,)`, a one-sample batch, and a weight of shape `(3, 5)`: the real reference raises;
- under R1 it is accepted, and `f` is 36115.17 where the true value is 7223.03, five times too large.

Every refusal in `test_a_weight_of_another_form_raises` and `test_data_of_another_form_raises` (and in my
`check_ref.py`) uses an axis of `z` larger than 1. A size-1 axis of `z` is common: a one-sample batch, an order-0 jet
axis, a stack of 1.

**Resolve.** Add a refusal, for the weight and for `y`, where `z`'s axis is 1 and the object's is larger. For
example, the `W = (1,)` batch above, or a jet axis of 1 at order `(0,)`, against an object of size 3 there. With it,
R1 is killed.

### 2. MINOR

1. Store the `mc_sgd` runs' loss history in the battery (section 5).
2. The brute-force class says it covers "sizes equal" through `small`, where `F = W = N_1 = (3,)`. Keep that problem
   in the class when part 2 extends it to `C`: the equal-size case is the one that hides a transposed axis.

## What resolves this round

Finding 1. I will rerun R1 against the added refusal.
