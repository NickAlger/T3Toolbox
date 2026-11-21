SIGN OFF

# P4d review, round 1: `jets` and `data_preparation` in the jet shape; their frontend reach

Reviewer, 2026-11-19, on the working tree of `graded-messages` (the implementer's uncommitted changes on `50346a1d`).
No blocking and no should-fix finding; two minor ones. I checked the mathematics independently of the
implementer's tests, against an oracle of my own (roots of unity plus one plain einsum per node, from P4b's review)
and against P4a's reviewed `jet_contract`. Every check agrees. My three mutants, aimed at the order of operations,
are all caught.

## The mathematics (my own checks, `p4d_check.py`)

- **The pairing.** `sum_k omega_k^2 a_k b_k` equals the corner of `a * jet_to_dual_jet(b)`, the corner taken by
  `jet_contract('JWi,JWi->', ...)`. Checked at `(3,)`, `(1, 2)`, `(2, 0)` and `(1, 1, 2)`, so at m = 1, 2 and 3,
  with a nonuniform weight: equal to 1e-12.
- **Weigh, then flip.** Slot `k` of the dual jet is `weight[n - k]**2 jet[n - k]`, checked directly at every
  shape, and for the probe kind mode by mode. Two mutants confirm the order matters and is tested (below).
- **The round trip.** `dual_jet_to_jet(jet_to_dual_jet(x))` gives `x` back, on a packed probe array `(d,) + F + W +
  (N,)` too (it comes back as a tuple, mode by mode, as entry 115 and the kind rule say).
- **`k_1! ... k_m!`.**
  - *Data:* `f(s) = exp(a_1 s_1 + a_2 s_2)(1 + c s_1 s_2)` at the jet shape `(2, 3)`, with its Taylor
    coefficients and its mixed partial derivatives at 0 each written out by hand (the product rule).
    `taylor_to_derivatives` and `derivatives_to_taylor` map one to the other exactly.
  - *Weights:* `derivative_weight_to_taylor_weight` multiplies by `k_1! k_2!`, and `prepare_derivative_data` with
    no weight gives `k_1! k_2!` per slot, with the probe mode axis leading.
  - *Why the weight is multiplied:* `½ Σ ω_d,k² (k! r_k)² = ½ Σ (k! ω_d,k)² r_k²`, so the Taylor weight is
    `k! ω_d`.
- **`rescale_direction`.** It gives the data of the same tensor at the new sample. For a dense `T`, the data `y` at
  input jets (by my oracle, not by `dense_probe`), and the rescaled `(ww', y')`: `y'` equals my oracle at `ww'`, to
  1e-12, for both kinds. The cases:
  - `(2,)` with mode jet sizes `(2,)`, `(3,)`, `(1,)` and one length;
  - `(1, 2)` with mixed sizes `(2, 2)`, `(1, 3)`, `(3, 2)` and a tuple of two lengths;
  - `(1, 2)` with planes and a numpy array of two lengths;
  - `(2, 1)` with one length.

  The reason it holds: the input jets and the outputs are polynomials in `s`, and `s_j = L_j s'_j` multiplies the
  coefficient of `s^k` by `L^k` on both sides.
- **`rescale_coordinates`.** The rescaled data are the data of `T / (s_0 ⊗ s_1 ⊗ s_2)` at the rescaled input jets.
  For probe, mode `i` is left open, so its data are divided by `s_i`; for apply they are unchanged. Checked by the
  oracle at `(2,)` and `(1, 2)`, both kinds, to 1e-12.

## The checkers

- **Spot checks.** `check_sample` raises when `W` disagrees and on a jet axis of size 0, and passes a good line at
  `(1,)`. `check_weight` raises on a bare vector `(d,)`, on `(d, n + 1)` at m = 2, on a mode dimension of 1, and on
  the jet axes exchanged; it passes `(d,) + F`.
- **The tests' lists** match the plan's: too few axes, `W` disagreeing once the jet axes are read, a size-0 jet
  axis, data not on the full box, and the old weight shapes.
- **The trap** of a forgotten jet shape is stated honestly in the docstrings of all three checkers, with a doctest
  that passes.

**Minor finding 1. The docstring of `check_sample` states the trap in one direction only.**
- It says that input jets given with `order = ()` read as one more axis of `W` and pass.
- The converse also passes: a jet shape with one variable too many reads a `W` axis as a jet axis. Measured:
  `check_sample('probe', (1, 1), affine_jet(ww, pp), shape)` passes, with `W = (5,)` read as a second jet axis of
  5 slots.
- `check_data` would then usually catch it (the data must be on the full box), but `check_sample` alone does not.
- *Resolve:* one clause in the paragraph "What a shape cannot say": "or a jet shape with an extra variable can read
  an axis of `W` as a jet axis". Optional.

## The frontend

- **The razor.** `t3toolbox/fitting.py` wraps and the backend computes:
  - its `rescale_coordinates` checks the coordinate scales (the frontend's job) and calls the backend;
  - its `dual_jet_to_jet` reads the kind and the jet shape, makes the safe-mode check on the divisor as the backend
    will compute it, and calls the backend.
  - It reaches two private backend helpers (`_kind_and_order`, `_squared_weights`). The second was already there in
    P3, so this is not new in kind.
- **`dual_jet_to_jet`**, measured:
  - safe mode raises on a zero weight at m = 2;
  - an integer `order` with a zero weight is a `TypeError`, not the zero-divisor error, so the jet shape is read
    first;
  - `safety.unsafe()` divides, and gives exactly the backend's result (`inf` and `nan` in the same places, equal
    elsewhere);
  - under `jax.jit` it divides, with no check (the trace skips it), and the non-finite places agree;
  - with a nonzero weight the round trip is exact.
- **Reach.** `t3toolbox.affine_jet`, `t3toolbox.fitting.affine_jet` and the same for `index_to_one_hot` are the
  backend function objects (`is`).

**Minor finding 2. The frontend `rescale_coordinates` checks the scales before it reads the jet shape.** With an
integer `order` and wrong scales, it raises the scales' `ValueError` where `dual_jet_to_jet`, by the rule the
handover states (section 4, item 5), would raise the jet shape's `TypeError` first. Harmless (both are errors); one
line, `backend_jets._kind_and_order(...)` first, would make the two frontend functions agree. Optional.

## The departures

All four are sound:
1. `tests/test_import_graph.py`: `jets` imports `contractions` for the one reader of the jet shape. The pin that
   `contractions` does not import `jets` keeps the graph acyclic.
2. The doctest of `TuckerTensorTrain.unscale_coordinates`: a caller the plan's search missed. Only the order
   changes, and the doctest passes.
3. `rescale_direction` raises for entries: with no jets there is no direction, and the plan's signature names
   probe and apply.
4. A 1-d length of size 1 at m = 1: the same as one number, unambiguous.

## The search for `pp` and integer orders (handover section 5)

I repeated it on the modules P1 to P4 wrote (`jets`, `data_preparation`, `sampling_definitions`, `contractions`):
- no parameter `pp` and no integer `order` remain;
- the hits are a direction in a docstring example, `order + 1` as the box of a tuple jet shape in comments, and the
  doctest that shows an integer raises.

In `t3toolbox/fitting.py` the integer orders and `(ww, pp)` pairs are the old model layer (`_canonical_weight`,
`_check_ragged_sample` and the `*_derivatives_model` factories), which the switch-over replaces. Each remaining hit
stays for the reason the handover gives.

## `subTest` labels

I ran the four slice test files and the dispatch tests that P4d touches (133 passed, 1,309 subtests) with
`TestCase.subTest` patched to record every label that is not a builtin (str, int, float, bool, None, or tuples and
lists of them). None.

## Mutations (`mutate_p4d.py`, a scratch copy, `python -B`, deleted after)

Run against `tests/test_jets.py`, `tests/test_data_preparation.py` and the doctests of `jets` and `fitting`. All
three are caught, and they complement the implementer's 20:
- **flip, then weigh**, with the weight unflipped: 83 failures;
- **the inverse without its flip:** 69 failures;
- **the frontend `dual_jet_to_jet` without reading the jet shape first:** 1 failure, the test that pins it.

## What I ran (single process)

- `p4d_check.py`, my own checks above.
- The slice's tests with the label checker: 133 passed, 1,309 subtests, 573 s.
- The module doctests of `jets`, `data_preparation`, `fitting`, `t3toolbox/__init__` and `tucker_tensor_train`:
  102 passed, from real runs.
- `mutate_p4d.py`.

The scripts are in `/tmp/claude-1000/-home-nick-repos-T3Toolbox/352ad37b-5ac8-40a1-a326-a7fc3a369cb2/scratchpad/reviewer/`.
The working tree is as I found it, apart from this file. The scratch copy is deleted, and nothing of mine is left
running.
