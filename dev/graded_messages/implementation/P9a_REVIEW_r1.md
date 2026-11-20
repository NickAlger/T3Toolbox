SIGN OFF

# P9a review, round 1: Engine E, the apply kind on both call types

Reviewer, 2026-11-20, on the working tree of `graded-messages` (the implementer's uncommitted changes on `c0bae86b`).
No finding.

The apply kind agrees with dense constructions of my own for its value, gradient, direction and curvature, on a frame
whose `L` and `R` end bonds differ at both ends. The probe graphs are unchanged row for row. A garbage `Vc` is read
by nothing. The departures are sound.

## My checks (`p9a_check.py`; seed 0 on the current env, seed 1 on the floor)

The problem: `d = 3`, with tails where `rL_0 = 2 ≠ rR_0 = 3` and `rL_d = 2 ≠ rR_d = 4`, at `()`, `(3,)` with input
sizes `(2,)`, `(3,)`, `(1,)`, and the jet matrix's mixed modes at `(1, 2)`; a nonuniform weight `F`; `K = (2,)`. Each
check below shares no code with the engine or the reference:

| Variable | Checked against | Worst |
| --- | --- | --- |
| `z` | the full contraction of `TuckerTensorTrain(D, L).to_dense()` with every input jet, by my roots-of-unity oracle | 8.5e-15 overall |
| `g` | the transposed dense Jacobian | |
| `Jp` | the dense tangent tensor | |
| `Kp` | the dense second derivative | |

- **`z`.** The full contraction, closed by the `rL` ones at both ends (the dense tensor squashes its tails).
- **`g` by the transposed dense Jacobian,** every coordinate of every `V_j` and `H_j`. The tangent tensor has `L` cores
  to the left, `U` over `E` or `E` itself, and `R` cores to the right. Its right end is closed by the `R` store's
  ones, `rR_d = 4`, which is what `close_step` of `mu_R_dot` and the seed's start on `nu_tilde` must reproduce. They
  do.
- **`Jp` from the dense tangent tensor,** per `k`.
- **`Kp` by the dense second derivative** of `Φ(x) = <ω² r, apply(T(x))>`, `r` held, entry by entry. It is the `ε¹`
  coefficient of a degree-`2d` polynomial, interpolated exactly. This covers the tilde-dot rows of section 0 of the
  design (`nu_dot_step` over `nu_L_tilde`, `eta_dot_step` over the tilde forms, `H_tilde_step` with the dot
  arguments).
- **A `Vc` of the wrong shapes** passed to the same calls: no error, and the results above unchanged. Read by nothing,
  as ruled.

Worst relative error over all of it: 8.5e-15 (current), 8.1e-15 (floor).

## The probe graphs, row for row

I extracted `t3toolbox/` at `HEAD` with `git archive` and dumped both versions' `_GRAPHS[('probe', '_manifold')]` and
`[('probe', '_corewise')]`, every row as a tuple of all six fields. The two dumps are identical (97 rows), so the
block split changed no probe row, and no probe call can have changed.

## The departures

1. **Every whole step takes `kind`,** `identity_step` and `g_step` included. Right: one uniform signature for the
   whole driver, and `_modes` keeps the per-kind difference (the mode axis, the weight row) in one helper rather than
   in every step.
2. **`_manifold(rows, kind)`**, so both call types run the same path check and add their twins the same way. Right:
   the twins were the one step the manifold graph did not share with the corewise derivation.
3. **The scratch `_read_setting` call in a test gained the kind:** a test-only consequence of `_Setting` now holding
   the kind.

## The mutant note

`test_deps_and_the_renames_carry_the_start` adds two synthetic rows to apply's rows before deriving the corewise
graph:
- `_from_eta`, a sweep starting from the `U`-basis `eta`, which must be dropped on a corewise point;
- `_from_nu_L_tilde`, starting from `nu_L_tilde`, whose start must be renamed `nu_tilde`.

**Meaningful, not contrived.** It tests the contract of `_corewise` itself, that a start node is a dependency in the
drop and a name in the rename, with the smallest rows that exercise it. No real row does yet: apply's start node
`w_tilde` is neither dropped nor renamed, which is why the two mutants were equivalent. Without this test the
mechanism would be unpinned until a later kind relies on it. The assertions read the derived rows, not the code path,
so they would survive a rewrite of `_corewise` that keeps its contract. The rest of the test (the real twin's start
`_Jp.w_tilde`, the corewise `nu_tilde` reading `L` and starting from `w_tilde`, the closure reaching the seed through
the start) pins the real rows.

## The rest, confirmed

- **The names comparison** (`test_sampling_apply.py`), engine against `sdef.__all__` per kind and call:
  - probe manifold equal;
  - apply manifold equal but the four zero names;
  - corewise the section-9 spellings;
  - `_ZERO_NAMES` exactly.

  It passes, and a zero name raises with its reason on both call types.
- **`nu_tilde` against the jet product `w_tilde · nu`.** Tested by the implementer, and implied by my dense `g` and
  `Kp`, which read it.
- **The seed start** is `w_tilde ⊗ 1(rR_d)` (`start_node`): pinned by my dense `g` on the tails frame and by the
  implementer's three seed mutants.

## What I ran (single process)

- **Current env:** `test_sampling_apply`, the four engine files, the corewise file, the bridge, `test_import_graph`,
  `test_jet_contract`, and the module doctests of `sampling_variables` and `sampling_steps`: 139 passed, 1 xfailed,
  23,742 subtests.
- **Floor env (`tttt`):** `test_sampling_apply`, `test_sampling_graph`, the corewise file and the bridge, 52 passed;
  `p9a_check.py`, OK.
- The graph comparison against `HEAD`, through a temporary extraction (deleted).

The working tree is as I found it, apart from this file. Nothing of mine is left running.
