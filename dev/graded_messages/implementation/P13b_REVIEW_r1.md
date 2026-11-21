SIGN OFF: the removals, the helper's conversion, the examples' ground truth and the two carried items are right. The example differences are rounding amplified; I measured the operator at about 4e-16 old against new.

# P13b review, round 1: the derivative methods removed

Reviewer, 2026-11-21. This reviews the uncommitted P13b on top of `986247bb`, against P13a's carry list, the plan's
P13 and the rulings. Everything was run single-process. The old tree was extracted read-only with `git archive` into
my scratch directory, and the working tree is untouched apart from this file.

## The helper `tests/_derivative_data.py`: right, derived and checked independently

**The forward.**
- `z(s)` is the contraction along `w + s p`.
- The merged method at `affine_jet(w, p)` and `(n,)` gives the Taylor coefficients `T_t` with `z(s) = sum_t T_t s^t`.
- The old derivatives are `D_t = z^(t)(0) = t! T_t`, so `taylor_to_derivatives` (slot `t` times `t!`) is exact.
- Entries is apply at the one-hot jet of the index, along `e_index + s p`. Right.

**The transposes.**
- The old ones were the adjoints of `v -> (D_t v)_t` under the plain pairing `sum_t <D_t v, c_t>`, so they returned
  `sum_t D_t^T c_t = sum_t T_t^T (t! c_t)`.
- The merged transposes are `J^T` under the same plain pairing over every slot. I verified this pairing in P12a,
  against `<J v, c>` summed over all slots.
- So the old transpose of `c` is the merged transpose of `t! c_t`, which is `taylor_to_derivatives(c)`. That is what
  `_scale` does.
- A packed probe vector is scaled on axis 1, after `d`. Right.

**Checked** (`p13b_check.py`). Against my own `D_t`, an exact polynomial fit in `s` of the dense contraction:
- `dd.forward` for probe, apply and entries, at `n` = 1 to 3;
- `dd.corewise_transpose`, by `<grad, x_dot> = d/de sum_t <D_t(x + e x_dot), c_t>`, interpolated exactly in `e`.

The worst error is 8.4e-15. The implementer also checked the helper against the old methods while they existed
(4.7e-16, all four classes).

## The removals and the callers

- **The 24 methods are gone,** with their banners, imports and `See Also` entries. No frontend module, example or
  page doctest calls one: my search finds only prose, in `docs/entries_apply_probe.md:182` and
  `naming_conventions.md:14`, which is P19's.
- **The 162 moved calls.** The rewriter routes every one through the helper. The tests pass: the gate, and my run of
  the frontend, probe-derivative, manifold and uniform-probing files, 121 passed.

## The examples

**The dense ground truth.** `taylor_to_derivatives(dense_apply(A, affine_jet(ww, pp), (order,)))` is exact by the
forward derivation above. Its independent anchor is the conversion pin below.

**The operators.** The forward is `taylor_to_derivatives` of `Z.apply(jet)`, and the transpose is `apply_transpose` of
`taylor_to_derivatives(r / s)`. Both are right by the transpose derivation above.

**Old against new, sampled on `fit_hilbert_from_apply_derivatives.py`.**
- *The printed output.* Level 1 is identical. From level 2 the errors differ in the 3rd or 4th digit, one iteration
  count differs, and the conclusion (best level 5) is the same.
- *The operator itself.* I imported the example's `apply_derivative_operator` from each tree, with the same seeded
  inputs: a `TuckerTensorTrain`, a frame and a residual. The forward agrees to 4.4e-16, and every block of the
  transpose to at most 6.0e-16.
- So the operator's meaning is unchanged, and the drift is MC-SGD amplifying rounding over minibatches and rank
  levels.

**The fit time** (5.8 s to 11.2 s in my run). This is correctly flagged by the implementer for the performance work,
not P13.

## `chunking.md`

The file is gone, with its `design_notes.rst` entry, the markdown links and the `:doc:` links, and Sphinx passed with
`-W`. What remains are plain-text mentions in code comments and in the `chunk_size` glossary entry, which are P19's.

## The carried items: done

1. **Nested lists.** The test now covers all 13 entries methods. The implementer's mutants on the five missing ones
   are killed.
2. **The derivative-conversion pin.** `TestTheDerivativeConversion` checks the `t`-th derivative in `s`, from an exact
   degree-`d` polynomial fit of the dense contraction times `t!`. It uses neither `affine_jet` nor
   `taylor_to_derivatives`. It compares against both the merged method and `dense_probe` / `dense_apply`, at `n` up to
   4 (slots beyond `d` vanish), and it outlives P20.

## The deviation: agree

`tests/test_probe_derivatives.py` tests the old `sampling_derivatives` kernels, which stay until P20, so it goes with
them. Its frontend calls moved to the helper. The merged methods' dense anchor is the new pin.
