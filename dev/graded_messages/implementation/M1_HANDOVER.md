# M1 handover: `data_metric`, `blocks_to_weight`, the converters, the laws, the grammar

Implementer, 2026-11-22. Built on `M1_DESIGN_NOTE.md`, `M1_DESIGN_REVIEW.md` and Claude's rulings:
- `_with_K` is moved to M2;
- edge case (a) raises;
- the batch method raises on block weights, pinned by a test;
- the frontend converters leave `fitting` entirely;
- M0's `-v` log is the reference.

Review findings 1 to 4 and minor points 1 to 5 are folded in. Nothing committed.

## 1. What changed

**`t3toolbox/backend/jets.py`.** A new section, "The data metric", with these functions, all in `__all__`:

- **`data_metric(r, weight, minibatch_scale)`** computes `γ ω² r` elementwise. Both factors are required. The
  signature is written in the new grammar.
  - **Pairing.** A tuple or list `r` pairs with:
    - a tuple or list weight of exactly `len(r)` arrays; or
    - an array weight with `shape[0] == len(r)`. A mode axis of 1 raises (review finding 1).

    An array `r` pairs with an array weight, elementwise. A packed `r` with a ragged weight raises, naming
    `pack_batch` (ruling 2). A packed weight with a mode axis of 1 broadcasts: this is the documented limit.
  - **Per pair (`_check_pair`).** Equal rank, and each axis of `ω` either 1 or `r`'s size, so a weight never grows
    `r` (finding 2). The checks are static, so they run under jit, and the message names the mode.
  - **`weight=None`** gives `γ r` for any `r`, a stack `K` included.
  - **`γ`.**
    - It must be a scalar (minor 1).
    - It is cast to the result dtype, so a float64 0-d `γ` does not upcast a float32 `r`.
    - With `None` there is no multiplication.
  - **Result.** The dtype is the lower floating precision of `r` and `ω`, and the result is an array even when 0-d.
  - **Docstring** (finding 3, minor 5):
    - `M: Y* -> Y`, and `J^T(data_metric(…))` is the weighted gradient. A doctest checks this against a central
      difference.
    - Taylor coordinates, Q7's recipe, and the contragredient rule for a `Y` element held in derivatives.
    - The structural raises.
    - The packed padding comes back as `ω²` times whatever is there; masking is the engine's.
- **`_omega_squared`** is the private core. It casts and squares exactly as `_squared_weights` does, which stays in
  place for M2.
- **`blocks_to_weight(kind, order, blocks, y)`** reshapes the blocks to the full form, reading only `y`'s rank
  (ragged or packed, with or without `C`). It keeps the blocks' dtype and backend. Structural errors raise: the block
  shape, the rank of `y`, ragged modes of different ranks, and entries with jets.
- **The converters, retyped** (finding 3, 9.2). Their code is unchanged.
  - `jet_to_dual_jet` is `Y* -> Y` turned. With `weight=None` it reads as `Y -> Y` turned.
  - `dual_jet_to_jet` is the inverse.
  - Both say "for `w_tilde` and the held tilde variables, never for a transpose". The division by zero is now the
    only behaviour, and the reference to the frontend function is gone.
  - The seed sentence now reads "the engine's adjoint sweeps are seeded by a dual jet", where it used to say "the
    seed of a transpose".
- **The module text.** Five groups now: the data metric added, the dual jet described as the engine's form. The
  block-form weight paragraph says that `data_metric` takes the full form.

**`t3toolbox/backend/fitting.py`.**
- New: `LeastSquaresBatch.data_metric(self, r)`, which is `jets.data_metric(r, self.weight, self.minibatch_scale)`.
- Its docstring says that a block-weight batch raises by rank until the batch takes the full form, and it has a
  doctest.

**`t3toolbox/fitting.py`.**
- Re-exports `data_metric` and `blocks_to_weight`.
- `jet_to_dual_jet` is no longer imported.
- The frontend `dual_jet_to_jet` (O-20's safe-mode check) is deleted.
- Both converter names are out of `__all__` and out of the namespace.
- The module text is updated.

**`docs/contributor/signature_style.md`, micro-grammar.** The type rule:
- the type first, then `len`, then the shape; the type names the element when `len=` follows, the whole object
  otherwise;
- a function of any kind types the whole object and lists the kinds' shapes after a colon;
- the vocabulary is 9.0's: `*` the algebraic dual, `[G]` presence;
- a role name comes first, with a colon (`# M r: Y, …`);
- a placement or dtype tag comes after the shape (`HOST bool, static`); this was minor 3;
- untyped signatures keep `len` first;
- the exemplar is `data_metric`'s signature.

## 2. Tests

- **`tests/test_data_metric.py`** (new).
  - The oracle is an explicit `np.ndindex` loop. It reads `ω` at index 0 on its size-1 axes and does no
    broadcasting.
  - Covered: probe ragged (tuple and mode-leading array weights), probe packed (full edge, per mode, mode axis of 1),
    apply and entries.
  - Sizes `F`, `W`, `C` distinct, all equal, and with no stack.
  - Weights per slot, per sample, per entry, one number, and a `C` of 1. The scale is `None`, a number, or a 0-d
    array.
  - Structure:
    - another rank raises, `K` included;
    - `None` gives `γ r` for a `K`-stacked `r`;
    - a weight that would grow `r` raises;
    - the exact mode count;
    - a packed `r` with a ragged weight raises;
    - the scale must be a scalar;
    - both factors are required;
    - the result keeps `r`'s structure, and a 0-d result is an array.
  - Precision: the float32/float64 grid, with scales of each kind.
  - `blocks_to_weight`:
    - the placement against loops (ragged and packed, `C` empty or not, three jet shapes);
    - **bitwise equal to the old block weighting through `data_metric(…, None)`** on all three kinds, the bridge for
      M2. The test's docstring says that with a scale the two agree only to rounding (minor 4);
    - wrong blocks raise.
  - The batch method:
    - equals `data_metric` with the batch's weight and scale;
    - **a block-weight batch raises by rank** (apply, probe, entries), pinning ruling 3.
- **`tests/_typing_laws.py`** (new): the law helper.
  - `JetLaw` has `P`, `P_star`, `P_star_turned` and `inputs`, with `lead=` for mode-leading arrays. It refuses
    `ℓ = 1`, repeated `ℓ`, and an order that is all zeros.
  - `InputLaw` has `inputs`, `V_star`, `V`, `tucker`, `tucker_dual` and `entries`.
  - `law_error` and `assert_law` do the comparison.
  - **It is not named `_typing.py` as the plan says.** That name is shadowed by CPython's built-in `_typing` module,
    and `from _typing import …` fails. The ambient law is left for M4, which tests the ambient transposes.
- **`tests/test_type_laws.py`** (new).
  - `data_metric` under the jet law and the input law (probe, full edge; `γ` fixed), and with no weight.
  - `blocks_to_weight` and the batch method under the jet law (minor 2).
  - The converters in both of r4's readings.
  - `rescale_direction` as the jet law: its sample and data, and `t3_probe`/`t3_apply` at the moved sample.
  - `rescale_coordinates` and `t3_unscale_coordinates` as the input law (probe and apply, three orders).
  - The helper's entries and `X*` laws, against `t3_entries` and the apply corewise transpose.
  - **Negative controls** (finding 4 included):
    - no square;
    - no flip;
    - a flip on one axis of two;
    - a dual jet read as a jet;
    - `c` moved as a measurement, for both `data_metric` and the probe corewise transpose: the right law passes and
      the wrong one fails at O(1).
- **`test_jets.py`.**
  - `TestFrontendDualJetToJet` is deleted.
  - The frontend calls in two `TestDualJet` tests are dropped: one subtest each, plus some plain asserts.
  - `TestFrontendReach` is updated: 12 backend names now, including the two new ones; one own function; the two
    converter names asserted absent.
  - The module docstring is updated, and the unused `safety` and jax imports are removed.
  - One accident during the edit: deleting the class first also removed the module-level `interpolated_derivatives`
    that followed it. It was restored verbatim.
- **`test_dispatch.py`.** New `test_jit_data_metric_and_blocks_to_weight`: 98 subtests, with the scale traced
  where it is an argument. Its loop is shared with `test_jit_jets_and_data_preparation` through `_check_jit_cases`.
  That test's count is unchanged.
- **`test_jax_absent.py`.** The script calls `blocks_to_weight` and `data_metric` with no jax. The frontend safe-mode
  check is replaced by the backend's silent division.

**Mutants: 15 of 15 killed.** The mutants are listed in `m1/mutants.py`; each mutant ran against the two new test
files.
- In `data_metric`: no square; `1/ω`; the scale dropped; the scale not cast; no rank check; a weight that grows `r`;
  a mode axis of 1 broadcast; a packed `r` reading a tuple weight; a scale of any shape.
- In `blocks_to_weight`: the blocks on the trailing axes; no block-shape check.
- In the method: the scale forgotten.
- In the helper: no turn; `P` the identity; `V` equal to `V_star`.

**By script.** On jax, a float32 `r` with a float64 weight and a float64 0-d scale gives float32, with x64 off and
on, eager and under jit. A rank mismatch raises under jit.

## 3. The gate (`-n 2`)

| Step | Result |
| --- | --- |
| suite, `-v` (run directly, for the per-node log) | 1,918 passed, 5 skipped, 1 xfailed, 91,249 subtests, exit 0, 35:00; peak RSS of the parent 4.3 GB |
| doctest-modules | PASS, 323 passed (M0: 321; +3 new docstrings, -1 for the deleted frontend `dual_jet_to_jet`) |
| doctest-quickstart | PASS |
| doctest-pages | PASS, 47 files, 0 failures (`signature_style.md` included) |
| examples | PASS, 14, 0 failed |
| sphinx `-W` | PASS, 0 warning/error lines |

The lowest free memory in these steps was 10.0 GB.

**Tally against M0's `-v` log** (`m0/tally.py`; output in `m1/tally_m0_vs_m1.txt`): 37 nodes differ, all of them
expected.
- **New nodes:**
  - 16 in `test_data_metric.py`;
  - 10 in `test_type_laws.py`;
  - `test_dispatch::test_jit_data_metric_and_blocks_to_weight`.
- **Deleted:** the 7 nodes of `TestFrontendDualJetToJet`.
- **Dropped exactly their frontend calls:**
  - `test_an_object_of_the_other_type_than_kind_says_is_not_guarded_against`, from 5 to 4;
  - `test_the_jet_shape_is_a_tuple_and_the_entries_kind_has_none`, from 7 to 6.
- **`test_every_public_function_has_a_doctest`** goes from 11 to 13: the two new public functions of `jets`.
- **`TestFrontendReach::test_the_names_of_t3toolbox_fitting`** keeps its 13 subtests (12 + 1, against 11 + 2 before),
  although its content changed.

Every other node has the same outcome and the same subtest count.

## 4. Departures and choices

- The law helper is `tests/_typing_laws.py`, not `_typing.py`: that name is shadowed by CPython's built-in
  `_typing`.
- The `InputLaw.ambient` method is deferred to M4, which tests the ambient transposes. Built now, it would have
  nothing to test.
- The jit coverage is a new test node, not added to the existing jit bucket, so that no existing node's count moves.
- Role-named returns: `data_metric` and the method return `# M r: Y, …`, and `blocks_to_weight` returns
  `# omega: …`. 9.1 gives the weight no algebraic type of its own.

## 5. Found outside the slice

- `jets.py`'s module paragraph on the block-form weight stays until M2 switches the checkers. It now points to the
  full form of `data_metric`.
