# P13 design review: the implementer's note on the frontend sampling methods

Reviewer, 2026-11-21, on `P13_DESIGN_NOTE.md`, read only. I read it against the plan's P13 and 2.5, O-44 and O-49,
`jets.check_sample`, and today's frontend signatures.

**Overall:** the design is right. Each method is one backend call, `check_sample` is the frontend's structural check,
the engine keeps the rest, and no safe-mode check is needed.

There is one finding the note does not raise, and it is the most important. **Two methods take an old positional call
silently and give wrong numbers.**

## A. The silent argument swap on probe (new finding; for the maintainer)

Today's forms are:
- `TuckerTensorTrain.probe_corewise_transpose(ztildes, ww, sum_over_probes=False)`;
- the static `probe_ambient_transpose(ztildes, ww, sum_over_probes=False)`;
- and the uniform twin of the first.

The merged forms are `(ww, residual, ...)`. For probe, a residual and an input sample are both a list of `d` arrays
`W + (N_i,)` (at `C = ()` and `order = ()`). So an old positional call `x.probe_corewise_transpose(ztildes, ww)`:
- passes `check_sample` (the residual reads as a sample);
- passes the engine's held-`Jp` shape check (the sample reads as a residual of the type of `z`);
- returns `grad <ww, z(x; ztildes)>` in place of `grad <ztildes, z(x; ww)>`.

The two differ for `d >= 3`; for `d = 2` they coincide by symmetry.
- **The corewise transpose:** the default flips too (per-sample becomes summed), so the shape may betray it downstream,
  or may not.
- **The ambient transpose:** the default stays off, so even the shapes are unchanged. It is entirely silent.

The other changed calls fail loudly:
- the apply and entries transposes: the residual is not a list, or not an integer;
- `T3Tangent.*_transpose`: the frame comes first now;
- the backend functions: `x` comes first now.

The library's own call sites are covered by the tests. The hazard is a user's code on upgrade. This is a silent wrong
answer, the library's top bug class.

*Options, the maintainer's call (it touches N16):*
1. **Recommended:** make `residual` (and what follows the sample) **keyword-only** on the frontend transposes, e.g.
   `x.probe_corewise_transpose(ww, *, residual, order=(), sum_over_samples=True)`. The old positional call then raises
   a `TypeError`. N16 fixes the *order* of the arguments; keyword-only keeps the order and makes the swap impossible.
   For one rule across the frontend, apply it to every transpose, the corewise Hessian (`residual`,
   `core_directions`) and the corewise tangent (`core_directions`), and leave the backend positional.
2. A note in the CHANGELOG's upgrading section naming exactly these two methods (and the uniform twin), with the
   symptom. This is the minimum, and needed with option 1 too.

## 1. The frontend checks: agree; nothing lost, the razor kept

- **`check_sample` looks at shapes only.** Its docstring says so: "the values are not looked at"; it is static under
  `jit`. So "always, under `jit` too" is correct, and it has no numerical side, so safe mode adds nothing. The bare
  `J`, `J^T` and the corewise trio have no numerical precondition (the catalog).
- **The checks that live in the engine stay there:** the residual's shape, the stacks and the core sizes, from P11 and
  P12a round 1. A raw-`.data` user gets them too, which is the razor.
- **The frontend's own checks are inseparable from the classes:** `check_sample`'s call, `validate()`, and the
  `TypeError` of point 3. That is the razor's exception, rightly used.
- **What a frontend user had before.** Today's plain methods checked nothing of their own beyond `validate()`; the
  `*_derivatives` methods called the perturbation checkers, and `check_sample` replaces them. So nothing is lost, and
  `check_sample` *adds* the integer-`order` and form checks.
- **The packed sample on the uniform classes** is checked by the engine (`d`, `N`). It loses little: one `W` is
  inherent to a packed array, and a non-tuple `order` raises in `len(order)`. Agree.

## 2. The split: agree

- **P13a:** the signatures, the six new methods per tensor-train class, the transpose call sites and `check_sample`.
  The `*_derivatives` methods stay.
- **P13b:** the 24 removals, every caller, the examples, the pages, and `chunking.md` with its links.

This gives two reviews with one subject each. Two conditions:
- **Point A's decision lands in P13a,** because it is the signature commit.
- **P13b's search for the removed names covers the `:doc:` links** the note lists, so `sphinx -W` stays green. The
  plain-text mentions wait for P19, as planned.

## 3. The `TypeError` for a `TuckerTensorTrain` as `core_directions`: agree, generalized

Guard the *form*, not one class: accept only a pair (tuple or list) of two core sequences, and raise a `TypeError`
otherwise. The message names the trap and the fix ("a direction is the pair of core tuples; for a `TuckerTensorTrain`
`y` holding one, pass `y.data`").

This covers, with one check:
- `TuckerTensorTrain`;
- `UniformTuckerTensorTrain` (pass its supercore pair);
- `T3Tangent`;
- anything else.

Test each.

## 4. `x.entries(<nested list>)`: agree

`np.asarray` before `check_sample` keeps today's behaviour. A float list then fails `check_sample`'s integer check, a
clearer error than today's. A jax array is left as is. Apply the same conversion in `entries_corewise_*` and the
static `entries_ambient_transpose`, so the five entries methods agree.

## 5. The examples' dense ground truth: agree for the examples, with one test kept independent

`dense_apply` or `dense_probe` at `affine_jet(ww, pp)`, then `taylor_to_derivatives`, is right for the *examples*,
whose dense number illustrates a fit's error.

But that route shares `affine_jet` and `taylor_to_derivatives` with the merged method it is compared against. The
library's independent check of the derivative conversion today is `dense_*_derivatives`, which P20 deletes.

*Keep one test that survives P20:* `taylor_to_derivatives` of `dense_probe` at `affine_jet(ww, pp)` equals an
independent derivative. For example, the `n`-th derivative in `t` of `<T, ⊗_j (w_j + t p_j)>`, by exact polynomial
interpolation in `t` (degree at most `d`). If P4's dual-jet tests already pin this against something that outlives
P20, name that test in P13b and that is enough.

## 6. The tests: sufficient, with point A's

The note lists:
- the doctests at `()`, `(2,)` and `(1, 1)`, with the trap;
- each method against its backend function;
- the `check_sample` errors;
- the corewise methods' symmetry and polynomial-exact quotient;
- the mirror and garbage;
- the sum flag;
- `test_dispatch`.

That is right. Add:
- **if option 1 is adopted:** an old positional call of each probe transpose raises a `TypeError`;
- **in any case:** a test that `probe_corewise_transpose` and `probe_ambient_transpose` at `d = 3` with swapped `ww`
  and `residual` give different numbers. This is a guard on the test design itself: the call-site tests must not be
  symmetric problems.

## Summary

| # | Request |
| --- | --- |
| A | The silent positional swap on the probe transposes: keyword-only `residual` (recommended, the maintainer's call), plus the CHANGELOG note |
| 1 | Agreed |
| 2 | Agreed, with A in P13a |
| 3 | Agreed, generalized to "a pair, else `TypeError`" |
| 4 | Agreed, for all five entries methods |
| 5 | Agreed for the examples; keep one conversion test independent of `affine_jet` and `taylor_to_derivatives` that outlives P20 |
