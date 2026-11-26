NOT YET

# M2 part 2 review, round 1: the engine on the rule

Reviewer, 2026-11-22, on the uncommitted part 2 on `m2-wip` (over `4c61fece`, which contains the unreviewed checkpoint
`fe7c7250`) and `M2_part2_HANDOVER.md`. My scripts are in `scratchpad/m2rev/`: `check_p2.py`, the brute force, and
the mutants run inline.

**Verdict: NOT YET, for one should-fix test gap (finding 1).** The library change is right:
- the engine agrees with my own brute force, including `C`, `K`, the equal-sizes problem and uniform against
  ragged with mixed edges;
- every one of the 194 expected-red ids is red for the stated cause and no other (I read the tracebacks);
- the reasoning about the equivalent mutant and the weight mask is correct.

## 1. The engine against the plan, the rulings and my design review

- **`r_step`** is `z - y` by position, the old alignment gone. That is right only because of `_check_data_space`,
  which refuses a `y` without `C` on a stack. Correction 1 is implemented as ruled:
  - static;
  - run on the objects as given, before packing;
  - ragged edges `N_i`, packed `N`;
  - an exact mode count;
  - a tuple for apply or entries refused;
  - never `K`.
- **`_weighted_sum`** is `Σ v · data_metric(v, ω, None)` over the jet axes, `W` and the edge, keeping `[K] + C`.
  `f_step` and `Jp_norm_sq_step` use it: the plan's §3 table.
- **The seed.** `w_tilde_step` is `flip(data_metric(r, _with_K(ω), None))` per mode, no longer `jet_to_dual_jet`.
  `_with_K` inserts `lens.K` size-1 axes at `lead + m + len(W)`, `lead` 1 for a mode-leading array, my minor 1. On the
  `r` route the row runs with `K = 0`, so nothing is inserted.
- **The uniform packing.** `_pack_data_space` takes, for each axis of `F + W + C`, the join across the modes. The edge
  stays 1 when every mode's is 1; otherwise a full edge is zero-padded to `N` and an edge of 1 is broadcast, never
  padded.
- **The masks.** The probe `y` is masked always (correction 2); the probe weight only when its edge is `N`; apply's
  and entries' never.
- **The `None`-guard fix** in the weight mask is correct. Without it, every uniform probe call with `y` and no weight
  read `np.shape(None)[-1]`.

## 2. My brute force (`check_p2.py`)

The engine's own `z` and `Jp` stand in for the dense tensors; part 1 pinned them, and M2 does not touch them. The data
and the weight are drawn per mode with every axis at random 1 or full size, `C` included, and expanded by
`np.repeat`.

- **Ragged with `C`: 56 cases, all within `1e-12`.**
  - The settings: `C = (2,)` with `W = (3,)`; and the equal-sizes case `C = W = F = N_1 = (3,)`, where a transposed
    axis cannot fail by shape.
  - Probe, apply and entries; orders `()`, `(2,)`, `(1,1)`.
  - Compared: `f` per element of `C`, `r`, `w_tilde`, `Jp_norm_sq` with `K = (2,)` (the weight constant over `K`),
    and `<g, p_k> = Σ ω² r Jp(p_k)` per element of `K + C`.
  - `f` has shape `C`.
- **Uniform against ragged: 12 cases, all within `1e-12`.** A packed frame (`UT3Frame.from_t3frame`) against the
  ragged one, with `C = (2,)`. Ragged `y` and `ω` whose edges are at random 1 or `N_i`, mode by mode, and whose other
  axes have random size-1 patterns that differ between modes.

`TestAgainstTheReference` covers the jet matrix and the weight patterns against the reference, on `small`, the
equal-sizes problem. Together with the stacks tests and `TestStackedDataAreSeparateFits` (data per element of `C`),
the new tests are the right set.

## 3. The expected-red list: verified

- **The set.** I parsed the `-v` log, both progress lines and summary lines. The failing set is exactly the 194 listed
  ids (`set ==`), so no red id is missing and none is listed in error.
- **The cause.** Every exception line in the log:
  - 657 are `ValueError: sampling_variables: the weight … / y … of the type of z`, the new check: 511 the weight, 146
    `y`;
  - 7 are not. I read each, and each is a downstream consequence of subtests that raised at the check:
    - `test_sigma_min_…`: `bond_wins` stays 0 when every subtest raised;
    - `test_retract_…`: `m` is never bound;
    - `test_the_records`: the covered set is empty;
    - `test_curvature_damping_…`: `assertRaisesRegex` received the check's `ValueError` instead of the expected one;
    - the three sweeps' "Lists differ";
    - `test_jax_absent`'s missing `JAX_ABSENT_OK`.

  So all 194 have the one cause. Make the list's header exact: about 7 ids fail at the test level through a
  consequence, not at the check itself. Minor 1.
- **The tally.** My own parse of part 1's log against part 2's: every node that differs and is not in the red list is
  new (the 11 of `test_engine_data_space.py`, and part 1's round-1 refusal test). No converted engine node changed
  outcome.

## 4. The implementer's reasoning

- **"Squared before the cast" is equivalent in the engine: correct.** `sampling_variables` casts every floating input
  it reads, the weight included, to the call's dtype on entry (`_cast`, step 4), and `r` and `Jp` are formed in that
  dtype. Inside `data_metric` the pair then already shares one dtype, and the cast in `_omega_squared` is trivial. The
  mixed-precision order is pinned where it can occur, in a direct `data_metric` call (M1's bitwise test).
- **The weight mask: correct, and the overflow test is a good find.** `r`, `Jp` and a held variable have zero
  padding, so `ω² · 0 = 0` for ordinary finite garbage, and the mask is redundant there. A finite padding value whose
  square overflows (`1e200`) gives `inf · 0 = NaN` without it. Keeping the mask, as plan §4 asks, with a test that says
  why, is right.
- **The gauge fix in `TestTheUniformEdgeOfOne`: correct.** `UNIFORM_MANIFOLD.frame(ux)` and `MANIFOLD.frame(x)` are
  gauge-equivalent, not equal (`docs/uniform_equivalence_contract.md`, "Gauge-carrying operations"). `g`'s blocks
  carry the gauge, so the test must pack the ragged frame.

## 5. Mutants of my own

Against `test_engine_data_space.py` and `test_sampling_stacks.py`. The implementer's 14 were not rerun.

| Mutant | Result |
| --- | --- |
| `_with_K` with `lead = 0` for a mode-leading array | killed (4) |
| apply/entries `_weighted_sum` misses the last `W` axis | killed (139) |
| **`_pack_data_space` takes `F + W + C` from mode 0, not the join across the modes** | **SURVIVES** (finding 1) |

## Findings

### 1. SHOULD FIX: the join across the modes in `_pack_data_space` is untested

**Evidence.** The mutant that takes the packed body `F + W + C` from mode 0 survives both files. With it, a ragged
probe `y` or `ω` on the uniform layout whose modes have different size-1 patterns breaks. For example, mode 0 per
slot `F + (1,) + (1,)` and mode 1 per sample `(1,) + W + (1,)`. My `check_p2.py` uniform section, which draws such
patterns, fails under it: `broadcast_to` raises. The failure is loud, not silent, but the join is the packing rule
plan §4 states ("each axis of `F + W + C` becomes the join across the modes"), and no test exercises it.

**Resolve.** In `TestTheUniformEdgeOfOne` (or beside `test_mixed_edges`), add a case whose modes differ in their
size-1 axes of `F + W + C` as well as in their edges: per slot in one mode, per sample in another, per entry in a
third. Compare uniform with ragged on `f` and `g`, and pack it through `_pack_data_space`. The mutant must then fail.

### 2. MINOR

1. **The expected-red header** says "every error inside them is the same check". Write instead: about 7 ids fail at
   the test level through a consequence of subtests that raised at the check (the four named above and the three
   sweeps' collected lists, plus `test_jax_absent`'s subprocess). The cause is the same, but say how it shows.
2. **The checkpoint `fe7c7250` and this part are one review unit.** This review covers both, and the squash should
   keep them together.

## What resolves this round

Finding 1. I will rerun the packing mutant against the added case.
