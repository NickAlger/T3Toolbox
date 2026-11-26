NOT YET

# P9b review, round 1: Engine E, the entries kind on both call types

Reviewer, 2026-11-20, on the working tree of `graded-messages` (the implementer's uncommitted changes on `81e1174a`).

The verdict is NOT YET only because the coordinator's ruling on `'w'` (finding 1) is to be built in round 1. I agree
with the ruling, and I suggest a form for it below. Everything else I checked is right, against dense constructions
of my own, with duplicate indices in every mode and a Tucker rank above a mode size.

## My checks (`p9b_check.py`; seed 0 on the current env, seed 1 on the floor)

The problem: `d = 3`, with tails where `L`'s and `R`'s end bonds differ at both ends. Mode 1 has size 2 under a Tucker
rank of 3 (the scatter's mode size must come from `D`). `W = 6`, with an index repeated in every mode (`[0, 2, 2, 1,
2, 0]`, `[1, 1, 1, 0, 1, 0]`, `[3, 0, 3, 3, 1, 2]`). A weight `()`. `K = 2`.

Each check below shares no code with the engine or the reference:
- **`z`** equals `T[index]`, the dense tensor of `(D, L)` indexed directly: no contraction, no one-hot vector.
- **`g` by the transposed dense Jacobian,** every coordinate of every `V_j` and `H_j`: the tangent tensor of that
  coordinate (`L` left, `U` over `E` or `E` itself, `R` right, closed by `R`'s end bond), indexed at the entries and
  paired with `ω² r`.
  - Duplicate indices add up correctly through the scatter: a dropped duplicate would change `V_tilde`'s columns `2`
    (mode 0), `1` (mode 1) and `3` (mode 2).
- **`Jp`** equals the dense tangent tensor indexed at the entries, per `k`.
- **`Kp`** equals the dense second derivative of `Φ(x) = Σ ω² r T(x)[index]`, entry by entry: the `ε¹` coefficient of
  a degree-`2d` polynomial, interpolated exactly.
- **The corewise point `(D, L)`.**
  - `z` and `Kp` equal the manifold call's.
  - The corewise `g` equals the dense Jacobian in the raw cores (every entry of every `D_j`, `L_j`), indexed at the
    entries.
- **`keep_W`, at step level.** `V_scatter_step(..., keep_W=True)` equals my per-sample construction (`out[s, ..., :,
  index[s]] = eta[s]`) exactly. The summed form equals its sum over `W`, with the shape `K + (n, N)` and `N` taken from
  `D`.

Worst relative error over all of it: 6.8e-15, on both environments.

## Findings

**1. Should fix (the coordinator's ruling, to be built in round 1): `'w'` on an entries call is the index, not
`w`.**

*I agree with the ruling.* A name means its definition (`SAMPLE_VARIABLES.md`: `w[i]` is the input vector; for
entries, `e_index[i]`). The reference module returns the one-hot vectors. P5's "the sample as given" was a shorthand
that coincides with the definition only where the sample *is* the input vectors (probe and apply). My run confirms
that today it returns the integer index, `ndarray int64 (3, 6)`.

*A suggestion for the form*, so that it costs nothing when not wanted and follows the house rules without a special
case:
- **Make entries' `w` a row of entries' graph**, a whole step `w_one_hot_step(index, D)` that reads the index and `D`
  (for each mode's `N_i`, and for the dtype).
- **Then the existing machinery does the rest:**
  - it is built only when wanted (the closure);
  - `D` is counted for the dtype (O-11), so the vectors are in the result dtype without any array made in a default
    dtype (row 5.6), even when `'w'` is the only thing wanted;
  - probe's and apply's `'w'` stay the sample as given (no row; the engine's existing branch).
- **The one-hot vectors can be `jets.index_to_one_hot`** (the engine already imports `jets`), cast once. That gives
  them its index semantics: a numpy index outside `[-N, N)` raises an `IndexError` naming the mode, as the gather
  itself does on numpy. Building them by hand would risk a second, different out-of-range rule.
- **Test:** `'w'` on entries equals `index_to_one_hot(index, shape)` in the result dtype, float32 with a float32 frame,
  on both call types. It is the same object across calls only in the sense of equal values.

**2. Minor. The docstring's `want` entry and the zero-names message for entries** will need the new meaning of `'w'`
once 1 is built (the handover's section 1 says "`'w'` on an entries call is the sample as given"). Part of 1.

## The rest, confirmed

- **The derivation** `_entries(_APPLY_ROWS)` by `_ENTRIES_STEPS` is one table, tested row for row against the
  docstring's. Every read of `w` becomes `index`; the scatter also reads `D`; the short nodes go at `order = ()` and
  their readers read the full trunks.
  - Its two invariants raise `RuntimeError` (a row reading the input with no step in the table; a row depending on
    `w` or a short node).
  - The twin path through the full `eta_tilde` is right: at `order = ()` the short node *is* the full one.
- **The gather** `B[..., index]` moved to `W + [K] + C + (n,)` is a slice, as the old `_entry_xis`. It serves `xi`,
  `xi_U_dot` (of `V`) and the raw `xi_dot` (of `D_dot`).
- **The scatter.**
  - `np.add.at` and `.at[index].add` both add duplicates. The implementer's mutant of a dropping form (`+=` on numpy,
    `.at[].set` on jax) is killed on both backends.
  - The `W x n` memory is as the design review asked.
  - `N` comes from `D`, and the implementer's `max(N, n)` mutant is now killed by the rank-above-size test. My problem
    has the same case.
- **The index.** It is out of the dtype count and never cast, and `W` is read from it. A float32 frame with an int64
  index gives float32.
- **The zero names** are off entries' graphs with apply's reason, pending the maintainer, as ruled for P9a.
- **The bridge.**
  - The order-0 old functions and `EntriesKind`.
  - Entry 120's migration: `t3_entries_derivatives` and `EntriesDerivativesKind` against the **apply** kind at
    `affine_jet(index_to_one_hot(index, shape), pp)`, with the `t!` conversions.

  Both pass on both environments.
- **The mutant note.** The equivalent mutant (the index left in the floating set) is equivalent for the reason given:
  the next line puts the uncast index among the values. Its non-equivalent form is killed.

## What I ran (single process)

- **Current env:** `test_sampling_entries`, `test_sampling_apply`, `test_sampling_graph`, `test_sampling_variables`, the
  corewise file, the bridge, `test_import_graph`, the entries dispatch test, and the module doctests of
  `sampling_variables` and `sampling_steps`: 92 passed, 10,088 subtests.
- **Floor env (`tttt`):** `test_sampling_entries`, `test_sampling_graph`, the bridge and the entries dispatch test, 46
  passed.
- `p9b_check.py` on both environments: OK.

The working tree is as I found it, apart from this file. Nothing of mine is left running.
