NOT YET

# P5 review, round 1: Engine A

Reviewer, 2026-11-20, on the working tree of `graded-messages` (the implementer's uncommitted changes on
`94613e3f`; the four documentation files left aside, as the coordinator asked).

One should-fix finding, a test gap on a check the plan requires. Two minor ones. The engine is right: it agrees with
the P4c reference and with a dense oracle of my own to 3e-15, on the current environment and on the floor. A held
variable is really used. The step counts match my own reading of the graph. The floor's dtype fix holds.

## What is confirmed

- **Against the plan, the design sketch and my design review** (with the coordinator's rulings):
  - the rows are `Rule(name, step, reads, needs, carries_K)`;
  - routes are chosen by `needs`, the precedence written in the docstring and tested;
  - a held variable never changes the route;
  - the pseudo-argument `'_manifold'` is in `present`;
  - the bond offset is one table by letter (`_BOND_OFFSET`);
  - the start of a sweep is sized from the end bond of the row's own store, tested on a frame whose `L` and `R` end
    bonds differ;
  - the one layout-aware join is `_with_end`, over `xprepend` and `xappend`;
  - bodies are memoized on `(step_name, lens, order, form)` and close over nothing per site, the leaf's order read
    off its operand inside the step;
  - `w` is not a slot, and `HeldVariables`' slots are the computed variables another row reads;
  - option (b) for `HISTORICAL`, with guard (ii) worded "in `STRINGS`";
  - the structural tests I asked for: every `step_name` resolves, and the namespaces are disjoint.
- **The checks are section 2.3's step 1 and nothing more.** No `safety` import.
- **Form.**
  - The steps are one site's math, one literal string each, the final strings with `C` written, through the module
    attribute.
  - Dispatch is inferred from every array input, and host constants are cast once.
  - `f_step` makes its `0.5` in the sum's dtype and returns an array.

## The departures

1. **`carries_K`:** what I asked for. Agree.
2. **`z_step` as the `z_term` alone:** right, since `eta` and `eta_D` are nodes of their own, and the two routes
   need exactly the term. Agree.
3. **A held variable that a row reads comes back cast; one not read comes back as given.**
   - Defensible: one not read is not a counted input (O-11), like `w`.
   - But the rule is two-sided and the user must know which case they are in: a wanted held `mu` comes back float32
     or float64 depending on whether `eta` is also wanted.
   - A simpler rule: *a held variable comes back exactly as it was handed in, always; the computation uses the cast
     copy.* That is one sentence, and it removes the identity question too.
   - Minor finding 2: I recommend it, but either is sound.
4. **`HeldVariables` refuses `==` with its own message:** agree. `ExplicitEquality`'s message names `allclose`,
   which a record of slots does not have.
5. **`NotImplementedError` for the parts not built:** agree, as a temporary marker of P6 to P11.
   - The order of the checks is right: an entries call with jets raises its `ValueError` first, as step 1 says.
   - Each later slice should delete its `NotImplementedError` and a test that it raised.

## Findings

**1. Should fix. The held-shape check is pinned on one slot only.**
- *Place:* `tests/test_sampling_graph.py` (the held-shape test) against `sampling_variables._check_held`.
- *Evidence:* the mutant `for name in held.keys()[:1]:` (check only the first filled slot) passes all four new test
  files: 36 passed, 772 subtests. A `HeldVariables` with a good `mu` and a wrong-shaped `nu` then reaches the steps
  unchecked, and fails deep inside `jet_contract`, or not at all if the rank happens to fit. The plan's step 1:
  "every held variable it is handed has the shape of the table".
- *Resolve:* one test with two or more filled slots, the wrong one not first (and the wrong one also unread, since
  the check covers every filled slot whether read or not). For example `HeldVariables(mu=good_mu, nu=bad_nu)`
  wanting `('eta_D',)`, so that `nu` is not read.

**2. Minor. The cast of a held variable that a row reads is not pinned (departure 3).**
- *Evidence:* the mutant that does not cast a read held value passes every new test. A float64 held `mu` in a
  float32 computation then makes `eta` float64 on numpy, against the absorbing rule (row 5.6).
- *Resolve:* either one assertion now (a held float64 `mu`, a float32 frame and sample: `eta` comes back float32),
  or an explicit item in P10's dtype matrix: "a held variable of another precision counts, and is cast". The
  coordinator ruled the matrix P10's, so the second is enough if it is written into the plan's P10.

**3. Minor. `docs/conf.py` and the three files with `y: of the type of z`** are being changed by the implementer
under the coordinator's ruling. I will read them when they are final. This is not a finding against the engine.

## Independent checks (`p5_check.py`, mine)

- **Every name against the P4c reference.** All ten names (`xi`, `mu`, `nu`, `nu_L`, `eta`, `eta_D`, `z`, `r`, `f`,
  `r_block_sumsq`), with a nonuniform weight:
  - eight jet cases: `()`, `(1,)` and `(3,)` with a line, the plane at `(1, 2)`, the jet matrix's modes `(3, 2)`,
    `(1, 1)`, `(2, 3)` at `(1, 2)`, a curve at `(2,)`, mixed sizes at `(2,)`, and `(1, 1, 1)`;
  - on `d` = 3, and `d` = 4 with tails whose `L` and `R` end bonds differ at both ends.

  The shapes are exact; the worst relative error is 2.7e-15 (current), 2.9e-15 (floor).
- **`z`, `r`, `f`, `r_block_sumsq` against my own dense oracle**, independent of the reference: the dense tensor of
  `(D, L)` (`TuckerTensorTrain(D, L).to_dense()`, tails squashed by ones as the engine closes them) probed at the
  input jets on a grid of roots of unity, one einsum per node. `r = z - y`, `f = ½ Σ ω² r²` and the block sums are
  written from their formulas. All agree.
- **A held variable is used.**
  - The setup: `mu` deliberately modified at one bond (`mu[2]` plus noise) and held.
  - `nu` is unchanged.
  - `eta[i]`, `eta_D[i]` and `z[i]` equal what the recurrences give from the modified `mu`, and change only at
    site 2.

  This shows the held value is read, which "held equals recomputed" cannot show.
- **`jet_contract` calls**, against my own reading of the graph:

  | Want | Calls | What runs |
  | --- | --- | --- |
  | `z` | `5d` | `xi`, `mu`, `nu_L`, `eta_D`, `z` |
  | `z` with `Vc` | `5d` | `xi`, `mu`, `nu`, `eta`, `z`; no `nu_L` |
  | `eta`, `eta_D` | `6d` | |
  | `mu` | `2d` | |
  | `z`, `mu` held | `4d` | |
  | `f` | `5d` | |
  | `w` | 0 | |

  All as predicted.
- **`z` through `Vc` equals `z` through `D`** on a consistent frame from `t3_orthogonal_representations`.
- **Safe mode against `safety.unsafe()`:** bit-identical.
- **jit and float32.**
  - `jax.jit` of `('f',)` on a float32 jax frame gives a jax float32 array of shape `()`, equal to numpy to 1e-4.
  - On numpy, float32 in gives an `ndarray` of shape `()` in float32, on the floor env too. That confirms the fix
    of `0.5 *` promotion and the scalar return.

## Mutations (`mutate_p5.py`, a scratch copy, `python -B`, deleted after)

Run against the four new test files. They complement the implementer's 26.

| Mutant | Result |
| --- | --- |
| the leaf clipped one slot short (`min(a, n) - 1`) | caught |
| the right sweep's start sized by the left end bond | caught |
| `f`'s half a Python float, run on the floor env | caught (the floor fix is pinned) |
| the held-shape check on the first slot only | **missed** (finding 1) |
| a read held value not cast | **missed** (finding 2) |

## What I ran (single process)

- Current env: the four new files, `test_jax_absent`, `test_contractions_interpreter` and `test_dispatch`'s new and
  P1 to P4 rows: 114 passed, 4,656 subtests. The module doctests of `sampling_variables` and `sampling_steps`: 2
  passed.
- Floor env (`tttt`: py3.9, numpy 1.22, jax 0.4.30): the four new files and `test_dispatch -k sampling_variables`,
  38 passed; `p5_check.py`, all OK.

The working tree is as I found it, apart from this file. The scratch copy is deleted, and nothing of mine is left
running.
