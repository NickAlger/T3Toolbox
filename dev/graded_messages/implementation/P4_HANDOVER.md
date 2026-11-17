# P4: handover

Slice P4 of `PART1_PLAN.md`: the checkers of a frame, without dense tensors, and the frames' cached
`consistency_residual`. Written 2026-11-17 on `graded-messages` at `78b6295` (P3 and the note of the planned
pause). Nothing is committed. No enforcement is added anywhere: that is P16's.

**Read section 4 first if short of time**: the route the plan names loses accuracy in proportion to the norm of
the tensor, and what was built is that route with one step added (finding 4.1).

## 1. What changed

| File | Change |
| --- | --- |
| `t3toolbox/backend/fv_operations.py` | `fv_frame_consistency_residual` rewritten through the cores (same signature, same formula); `fv_centers_consistency_residual` **new**; five private helpers; `__all__`; the import of `t3_conversions` (used only by the dense route) replaced by `t3_orthogonalization`; one clause in the module docstring |
| `t3toolbox/backend/ufv_operations.py` | `ufv_frame_consistency_residual`, `ufv_centers_consistency_residual` **new**; one private helper; `__all__` |
| `t3toolbox/frame_variations_format.py` | `T3Frame.consistency_residual` **new** (cached property); `T3Frame.is_consistent` through it; its docstring; one sentence of `to_t3`'s docstring that said a check would have to densify |
| `t3toolbox/uniform_frame_variations_format.py` | `UT3Frame.consistency_residual` **new** (cached property); `UT3Frame.is_consistent` through it, no longer through `UniformTuckerTensorTrain` arithmetic |
| `tests/backend/test_frame_consistency.py` | **new**, 30 tests |
| `tests/test_frame_variations_format.py` | 5 tests (the `is_consistent` cases) |
| `tests/test_uniform_frame_variations_format.py` | 4 tests (the `is_consistent` cases) |
| `tests/test_dispatch.py` | rows in `test_jit_backend` and `test_jit_uniform_frame_helpers` |
| `tests/test_jax_absent.py` | one test and its script |

Nothing is deleted. `tests/test_import_graph.py` needed no change (its acyclicity test passes with the new import).

### The public surface

Each backend function returns a residual of shape `C` (the frame's stack; 0-d unstacked) and never raises on a
numerical condition, reads no mode, and imports nothing from `safety`.

```python
# backend/fv_operations.py
def fv_frame_consistency_residual(frame) -> NDArray              # frame = (D, U, L, R), ragged.  shape=C
    # ||T(D, L) - T(D, R)|| / max(1, ||T(D, R)||), per stack element.  U is not read.
def fv_centers_consistency_residual(frame, centers) -> NDArray   # centers = (Vc, Hc); either may be None.  shape=C
    # max over the sites i and the families given of
    #   ||T_V[i] - T(D, L)|| / max(1, ||T(D, L)||)   and   ||T_H[i] - T(D, L)|| / max(1, ||T(D, L)||)

# backend/ufv_operations.py
def ufv_frame_consistency_residual(data) -> NDArray              # data = uniform frame .data.  shape=C
def ufv_centers_consistency_residual(data, centers) -> NDArray   # centers = (Vc_sc, Hc_sc), bare supercores of the
    # shapes of ufv_variation_shapes(data), (d,)+C+(nU, N) and (d,)+C+(rL, nD, rR); either may be None.  shape=C

# frontend
T3Frame.consistency_residual      # @ft.cached_property -> NDArray, shape=stack_shape: fv_frame_consistency_residual(self.data)
T3Frame.is_consistent(rtol=1e-9)  # -> bool NDArray, shape=stack_shape: self.consistency_residual <= rtol
UT3Frame.consistency_residual     # @ft.cached_property -> NDArray, shape=stack_shape: ufv_frame_consistency_residual(self.data)
UT3Frame.is_consistent(rtol=1e-9) # -> bool NDArray, shape=stack_shape: self.consistency_residual <= rtol
```

The cached properties are cached exactly as `orthogonality_residual` is: `functools.cached_property` on the frozen
dataclass.

### What consistency of the centers means (the statement implemented)

From `SAMPLE_VARIABLES.md` section 1 ("The center cores `(Vc, Hc)` are the base point written into the frame's
hole. ... The network with `Vc[i]` under `U[i]` at one site, or with `Hc[i]` over `D[i]` at one site, and `L` cores
to the left and `R` cores to the right, is the base point"), the note to row 4.2 ("centers that are given and do
not agree with `(D, L)`"), and the plan's section 2.4 ("at each site, the tensor with `Vc[i]` under `U[i]` (and
with `Hc[i]` over `D[i]`), `L` cores to the left and `R` cores to the right, against `T(D, L)`"):

> For every site `i`: let `T_V[i]` be the tensor of the T3 with Tucker cores `D[j]` for `j != i` and `Vc[i]` at
> `i`, and TT cores `L[j]` for `j < i`, `U[i]` at `i`, `R[j]` for `j > i`; let `T_H[i]` be the tensor of the T3
> with Tucker cores `D` and TT cores `L[j]` for `j < i`, `Hc[i]` at `i`, `R[j]` for `j > i`. The centers are
> consistent with the frame when `T_V[i] = T(D, L)` and `T_H[i] = T(D, L)` for every `i`, where `T(D, L)` is the
> tensor of the T3 `(D, L)`. A tail (an end bond above one) is summed over its index in each tensor.

Verified before coding on the library's own frames and centers (`t3_orthogonal_representations`, with
`squash_tails=True` and `False`, stacks `()`, `(3,)`, `(2, 2)`, six structures): dense `T_V[i]`, `T_H[i]` and
`T(D, L)` agree to `1e-12`. This is the same statement `scripts/review/sample_variables_check.py` part E checks
through the samples (`z` with `Vc`, with `Hc`, through `D`).

What the decisions do not fix, and what was chosen (section 4.2): the normalization and the reduction over the
sites.

### How the code is built

- `_t3_left_orthogonal_form(tucker_cores, tt_cores)`: sums the two tails, then `t3_orthogonalization
  .t3_left_orthogonalize` (the route of `geometry.point_norm_sq`). `_t3_left_orthogonal_norm`: the norm of the last
  TT core. `_t3_difference_norm(x, y)`: the difference of two T3s in left-orthogonal form as one T3 (Tucker cores
  stacked, TT cores block-diagonal by `_t3_core_direct_sum`, the first core of `y` negated), brought to
  left-orthogonal form, its norm read off.
- `fv_frame_consistency_residual`: `(D, L)` and `(D, R)` each to left-orthogonal form, the norm of their
  difference, over `max(1, ||T(D, R)||)`. Three orthogonalizations.
- `fv_centers_consistency_residual`: `(D, L)` to left-orthogonal form once; for each site and each family given,
  the T3 with the center to left-orthogonal form and the norm of its difference from the base point. At most
  `4 d + 1` orthogonalizations.
- `_zero_stack_elements_that_are_not_finite`: see 4.3.
- The uniform twins mask the frame with its own masks (`ufv_apply_frame_masks`) and the centers with the frame's
  variation masks (`ufv_apply_variations_masks` with `ufv_variation_masks(frame_masks)`), once, split the masked
  supercores along the leading `d` axis into `d` zero-padded cores, and call the ragged function (section 4.2).
  Masks stay host numpy; the supercores go through `xnp`.
- Dispatch is inferred from the arrays in every function. Nothing in the backend reads a mode or imports `safety`.
  `t3_linalg` is not imported (the cycle the plan names).

## 2. The tests, and what each group establishes

`tests/backend/test_frame_consistency.py` (30 tests). The file computes each definition itself from dense
tensors (`t3_to_dense` of each network, a norm over the mode axes); structures with and without tails, `d = 1`,
a Tucker rank above its mode size; stacks `()`, `(3,)`, `(2, 2)`.

| Group | Establishes |
| --- | --- |
| `TestRaggedFrameResidual` (10) | The residual is the dense definition on hand-made frames (random cores: nothing orthogonal, nothing consistent), per stack element, with tails; a library-built frame passes at `1e-12`, at every norm from `4e-6` to `4e14`; a relative perturbation of `1e-9` shows as `1e-9`; float32 in, float32 out at float32 rounding; a perturbed `L` or `R` fails at every site; `U` is not read; one wrong stack element fails alone; a zero frame gives 0 |
| `TestRaggedCentersResidual` (9) | The residual is the dense definition (frame and centers hand-made; each family alone; the library's frame with random centers); the library's centers pass at `1e-12`, at every norm; a center taken from another point fails at every site in either family, and the other family alone still passes; a relative error of `1e-8` shows as `1e-8`; the library's centers on a frame with a wrong `R` or `U` fail (`Hc` alone does not read `U`); per stack element; `(None, None)` raises |
| `TestNoDenseTensor` (2) | `d = 12`, `N = 100` (a dense tensor of `1e24` entries): each of the four checkers returns the right answer in under 0.5 s (measured: 2 ms and 33 ms ragged) |
| `TestUniformTwins` (5) | Each uniform residual equals the ragged function's on each stack element's real cores, sliced by the test's own indexing, on hand-made uniform frames whose ranks are drawn per stack element, with garbage in the padding, with and without tails, and equals the dense definition; garbage of size `1e3` in every padded slot of the frame and of the centers changes no bit of any residual (the fixture asserts it changed the padding and only the padding); library-built uniform frames pass over the stack matrix (unstacked, stacked, two stack axes, forced-larger padding, ranks varying over the stack), clean and with garbage; a wrong real slot fails only its stack element; `(None, None)` raises |
| `TestCheckersNotGuarantees` (4) | A checker returns a number on wildly wrong input; a `nan` or `inf` in a core gives `nan` for that stack element and leaves the others, with no error (ragged and uniform); cores whose products overflow give no error; an AST search of `t3toolbox/backend/`: no backend function uses any of the four names, except the two uniform twins calling their ragged functions |

`tests/test_frame_variations_format.py` (5): `T3Frame.consistency_residual` is the dense definition on hand-made
frames with tails and stacks; `is_consistent` passes a library-built frame, fails a perturbed `L` or `R` at every
site, and `rtol` is the threshold; per stack element; **computed once** (the backend checker is wrapped by a mock:
one call for three verdicts and two reads on one frame, the argument is the frame's `.data`, a second frame makes
its own call); no dense tensor (`d = 12`, `N = 100`, under 0.5 s).

`tests/test_uniform_frame_variations_format.py` (4): `UT3Frame.consistency_residual` equals the ragged frame's
residual of each element (via `to_t3frame`), numbers compared; a mixed stack; computed once, through the uniform
twin, and `UniformTuckerTensorTrain.norm` is never called; no dense tensor. The file's existing cases
(`test_is_consistent_matches_oracle`, `test_frame_methods_garbage_robust`) pass unchanged.

`tests/test_dispatch.py`: the three new backend functions and both cached properties under `jax.jit` (each family
of centers alone too); jax in gives jax out, eager and compiled, in the dtype of the cores; the compiled value
equals the eager one. `tests/test_jax_absent.py`: all four backend checkers and the frontend properties run with
jax hidden, with the jax-absent warning turned into an error.

**Testing the tests.** 30 mutations of the code in a scratch copy (`python -B`, `PYTHONDONTWRITEBYTECODE=1`), each
run against the slice's tests: all 30 caught. Among them: the difference taken across the two gauges; a tail not
summed (each end); `R` to the left of the site, `L` to the right; the first or the last site skipped; the minimum
over the sites; a family ignored; the frame, `Vc` or `Hc` unmasked on the uniform layout; the centers masked with
the bond masks unshifted; either property uncached; `rtol` ignored; `UT3Frame` through `UniformTuckerTensorTrain`
arithmetic. Removing the zeroing of a stack element that is not finite makes numpy's SVD run forever on an `inf`
(the run was stopped at its time limit, which is why the guard exists).

## 3. The gate

GATE_PLACEHOLDER

## 4. Departures from the plan, findings, and the choices it left open

### 4.1 The route the plan names is not accurate as written; one step was added (please look)

The plan: "The route is the one `geometry.point_norm_sq` takes today: `t3_operations` (the difference) and
`t3_orthogonalization` (a left orthogonalization, then the last core's norm)". Built literally (the difference
`T(D, L) - T(D, R)` as one T3 from the stores as they stand, then one left orthogonalization), a **consistent**
frame shows a residual that grows with the norm of its tensor:

| `||T||` | residual of a consistent frame, the literal route | as built |
| --- | --- | --- |
| `4e-2` | `1e-14` | `5e-16` |
| `4e3` | `1.6e-12` | `1.4e-14` |
| `4e6` | `2.2e-9` | `1.5e-14` |
| `4e9` | `2.0e-6` | `1.4e-14` |
| `4e14` | `0.57` | `1.3e-14` |

(float64; `d = 5`, mode sizes 20 to 24, ranks 4 to 7; the largest of 20 seeds. The centers' residual behaves the
same way: at most `2.1e-14` as built.) A random T3 with unit Gaussian cores has a large norm (`4e6` here; the
`d = 12`, `N = 100` case gave a residual of `2e4`), so the check would have failed on ordinary frames at `1e-12`
and at the default `rtol = 1e-9`.

The cause: `L` is left-orthogonal (unit-scale cores, the norm in the last one) and `R` is right-orthogonal (the
norm in the first one). An orthogonalization step is backward stable in the norm of the matrix it factors, and
that matrix holds a block of each term; with one block of scale 1 and the other of scale `||T||`, the small block
is perturbed by `eps ||T||` relative to itself, and the difference by `eps ||T||^2`.

What was built: each term is first brought to left-orthogonal form by itself (one orthogonalization each, the same
library routine), and then the plan's route is applied to the two. Both blocks then have unit scale until the last
core, where both have scale `||T||`. It is the plan's route with one step before it, not another method, and it
does not assume the frame is orthogonal (a hand-made frame is orthogonalized like any other). I did not see this
as a departure that needs a decision, but it is the main thing to check.

**Accuracy measured, as built.**

| Case | Frame residual | Centers residual |
| --- | --- | --- |
| float64 numpy, library-built, norms `4e-2` to `4e14` | at most `1.5e-14` | at most `2.1e-14` |
| float64 numpy, the 72 small cases of the tests (tails, stacks) | at most `1.8e-14` | (same bound) |
| float32, jax (its default), norms `4e3` to `4e9`, eager and under `jit` | `1.0e-6` to `1.7e-6` | `1.3e-6` to `1.9e-6` |
| float32 numpy (a float64 frame cast to float32) | `3e-7` to `5e-7` | `3e-7` to `4e-7` |
| against the dense definition on hand-made frames | relative difference at most `6e-15` | |
| an inconsistency of relative size `1e-9` (frame), `1e-8` (center) | shows as that number to 3 digits | |

So the default `rtol = 1e-9` of `is_consistent` cannot pass a float32 frame (`1e-6`); P16's enforcement goes
through `safety.effective_rtol` (`rtol_jax = 1e-5`), which does. Where `||T|| < 1` the formula's `max(1, .)`
makes the residual absolute, as today.

### 4.2 Choices the plan left open

1. **The centers' residual: normalization and reduction.** `max(1, ||T(D, L)||)` in the denominator, by analogy
   with the frame's formula (the reference tensor there is `T(D, R)`; here it is the base point `(D, L)`, which the
   decisions name). One number per stack element: the **maximum** over the sites and the families.
2. **Either family of centers may be `None`** and is then not checked (the engine takes `Vc` alone, row 4.1, so a
   raw-data user may hold only `Vc`). `(None, None)` raises `ValueError`: a structural misuse, not a numerical
   condition.
3. **The uniform centers are two bare supercores** `(Vc_sc, Hc_sc)`, the first two members of the variations
   `.data` that `ut3_orthogonal_representations` returns, masked with the frame's variation masks, as the plan's
   section 2.3 has the engine do with `Vc`. P15 decides what the uniform geometry returns as centers; if it returns
   the 4-tuple `.data`, its caller passes the first two members.
4. **The uniform twins delegate to the ragged functions** on the masked supercores, split along `d` into
   zero-padded cores (a zero slot contributes nothing, so they stand for the same tensors, stack element by stack
   element, at ranks that vary over the stack). The alternative, a sweep through `xscan` on supercores, would need
   the T3 with the center at one site to have a different width at that site. Under `jit` the `d` steps are
   unrolled; these are checkers, not inner-loop code. Equality with ragged, robustness to garbage and tails are
   tested; there are no output masks to assert (the output is one number per stack element).
5. **The difference is formed as a general T3 sum** (Tucker cores stacked, TT cores block-diagonal over all three
   legs), not by using that the terms share Tucker cores. Simpler, one helper for all three uses, at the cost of
   doubled Tucker ranks in the difference.
6. **`UT3Frame.is_consistent` now has the ragged formula**, `<= rtol * max(1, ||T(D, R)||)`; before it was
   `<= rtol * ||T(D, R)||`. They differ only where `||T|| < 1`. The plan says it "moves to the twin", and the twin
   has the ragged formula.
7. **Tests of timing** assert under 0.5 s where 2 ms and 33 ms were measured ("well under a second").
8. **Where the tests live.** The backend checkers in a new file, `tests/backend/test_frame_consistency.py`; the
   `is_consistent` cases in the two files the plan names.

### 4.3 A checker that does not raise: what it took

"Each returns a residual and never raises." numpy's SVD raises `LinAlgError` on a matrix with a `nan`, and on a
Tucker core with an `inf` it did not return at all (a run was stopped after 20 s; today's dense function returned
`nan` in both cases). So each ragged checker first finds the stack elements with a core that is not finite, among
the cores it reads; those elements are checked as zeros and get the residual `nan`, which fails every threshold;
the other elements keep their residual. The same on jax and under `jit` (`xnp.where`). Finite cores whose products
overflow (`1e200` in every core) still make numpy's SVD raise inside the sweep; that `LinAlgError` is caught and
the whole stack gets `nan`. This is the only `try` in the slice.

On the uniform layout a `nan` in the **padding** gives `nan`, because the library's masking multiplies
(`nan * 0 = nan`): existing behaviour of `ufv_apply_frame_masks`, not changed here.

### 4.4 New names

Public: none beyond the plan's (`fv_centers_consistency_residual`, `ufv_frame_consistency_residual`,
`ufv_centers_consistency_residual`, `T3Frame.consistency_residual`, `UT3Frame.consistency_residual`, section
11.3). Private, in `fv_operations`: `_t3_left_orthogonal_form`, `_t3_left_orthogonal_norm`, `_t3_core_direct_sum`,
`_t3_difference_norm`, `_zero_stack_elements_that_are_not_finite`; in `ufv_operations`:
`_masked_frame_as_ragged`.

### 4.5 Not done, on purpose

- No enforcement (P16). No geometry member (O-31). The frontend frame classes carry no centers (O-4); no frontend
  function calls the centers' residual.
- No documentation page was changed (`docs/numerical_contracts.md` and `docs/contributor/precompute_and_caching.md`
  name the cached residuals; P19).

## 5. Outside the slice, to be looked at

1. **`(x - y).norm()` of the public T3 arithmetic has the accuracy problem of 4.1.** For the same tensor in two
   gauges (`(D, L)` and `(D, R)` of one frame, norm `4e6`): `(a - b).norm() / b.norm() = 3.7e-10`, where the new
   residual gives `4e-15`. It is why the old `UT3Frame.is_consistent` needed its loose `rtol`. Anyone who measures
   a small difference of two T3s in different gauges is affected. The private `_t3_difference_norm` is the remedy;
   by the razor it may deserve a public name (in `t3_linalg`, which `fv_operations` cannot import), which is a
   naming decision I did not take.
2. **`tt_squash_tails` (ragged) promotes float32 numpy cores to float64** (`xnp.ones((r, 1))` in an einsum), so a
   frame the library builds from a float32 numpy point has float64 cores in places. Seen while measuring; the
   checkers sum their tails themselves and keep the dtype.
3. **numpy's SVD does not return on a Tucker core with an `inf`** (4.3): `t3_orthogonal_representations` and every
   other orthogonalization of the library would hang the same way on such input.
4. `backend/fv_operations.py` no longer imports `t3_conversions` (it was used only by the dense route). Nothing in
   the tree reached `t3_conversions` through `fv_operations`.
