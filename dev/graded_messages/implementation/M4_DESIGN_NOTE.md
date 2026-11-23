# M4 design note: the transposes typed

Implementer, 2026-11-23, on HEAD (`80a374ce` and the CI-fix commit). Read only. Implements plan §5, the M4 row of §7,
Q2 (`c`), Q11, and the type grammar of `docs/contributor/signature_style.md` (M1).

## 1. The 42 functions, and the rename's mechanics

An AST scan finds exactly the plan's 42 functions with a parameter `residual`:
- **Backend, 21:** `tv_*_transpose`, `t3_*_corewise_transpose`, `t3_*_corewise_hessian`, `t3_*_ambient_transpose`
  (`apply`, `entries`, `probing`; 12); `ut3_*_corewise_{transpose,hessian}` (6); `utv_*_transpose` (3).
- **Frontend, 21:** `TuckerTensorTrain` 9, `UniformTuckerTensorTrain` 6, `T3Tangent` 3, `UT3Tangent` 3.

Left alone, as the plan says (deleted in P17/P20): the old kinds' `SamplingKind.transpose`/`corewise_hessian`, and
the four `utv_*_transpose*_from_sweep` helpers. Inside the library, every call to the 42 passes `c` by position.

**Step A, scripted:** `m4/rename.py`, modelled on M0's, with an edit log of file, line, column, old and new.
1. **The parameter and its uses.** In each of the 42 `def`s (found by AST), the argument `residual` and every
   `Name` node `residual` in its body become `c`. The script checks that no nested function rebinds the name.
2. **Keyword calls**, `residual=` → `c=`, located by AST:
   - in code: the 123 test calls;
   - in docstrings: the 17 doctest lines, matched by a regex on `residual=` inside the 42 functions' docstrings and
     their modules' doctests.
   - A `residual=` keyword in a call to anything else is reported and left alone; the scan expects none.
3. **Inline literals.** ` ``residual`` ` → ` ``c`` ` inside the 42 docstrings.

**The collision check.** `t3_apply_ambient_transpose` already has a local `c = xnp.asarray(residual)`. After the
rename it reads `c = xnp.asarray(c)`: correct, but rule-based inversion could not undo it. So:
- the inverse replays the edit log backwards instead of reversing the rules;
- the forward run lists every pre-existing `c` in the 42 scopes (one, this local) for review by eye.

**The round trip.** HEAD → forward → inverse must give HEAD back, byte for byte, under `t3toolbox/` and `tests/`.

The two curvature test files have no `residual=` call, so the CI fix and M4 do not touch the same lines.

**Step B, by hand:** the prose of §2, kept as a separate diff so that the scripted part stays mechanical. The
reviewer can rerun the script on HEAD and diff against it.

## 2. The exact text

**The parameter's docstring entry** (probe shown; apply `P*`, entries `ℝ`):

> **c**: an element of `Y` (here `P*⊗V_i` per mode), paired with `z` slot by slot: `<c, z> = Σ c·z` over the modes,
> the slots, the samples and the mode edge. It carries no metric, and a transpose applies none. For the gradient
> of a weighted least-squares fit, pass `data_metric(r, weight, minibatch_scale)`, or `batch.data_metric(r)`. With
> no weight and no scale this is `r` itself.

**Prose.**
- "the Gauss-Newton `J^T r`" and "the summed gradient `J^T r`" become `J^T c`, with "(the gradient: `J^T(M r)`)"
  where a gradient is meant. This covers the plan's sites (`manifold.py:768, 916`; `uniform_manifold.py:643`;
  `tucker_tensor_train.py:3837, 4003`; `uniform_tucker_tensor_train.py:498`), the residual wording around them
  (`manifold.py:760`; `sampling_variables.py:121-122, 872`), and the 42 docstrings' own `J^T r`.
- The "turned inside (a dual jet)" prose of the 42 and their modules becomes "`c` in slot coordinates; the engine
  turns it". `sampling_derivatives.py` (old layer) is not touched.
- **The two examples** keep their code and gain one comment citing the contragredient rule (§1):
  `examples/fit_hilbert_from_apply_derivatives.py:142-144` and `..._flat.py:137-139`.
- **P13's right/wrong adjoint doctests** stay, renamed. They are the only guard against a swapped sample and `c`
  (§9.3).

## 3. Signature comments: what M4 types, what waits for M5

**Recommendation: M4 types the `c` line only. M5 does the rest.**
- Today the `c` line says "of the type of z", which is wrong: `c` is in `Y`, not `Y*`. M4 must rewrite that line in
  any case.
- M1's grammar is settled, so writing the type there now costs nothing:
  - `# P*⊗V_i, len=d, elm_shape=F+W+[K]+C+(Ni,)` (probe);
  - `# P*, shape=F+W+[K]+C` (apply);
  - `# ℝ, shape=W+[K]+C` (entries);
  - the ambient ones keep their `W+C` shapes.
- Every other line of the 42 signatures, and the function-level types (`inputs × Y -> X*`, ...), wait for M5, as
  the plan says.
- The alternative is the plan's literal reading: no type in M4, and "shaped like `z`" on the `c` line. That would
  leave a known-wrong word in place for one slice.

## 4. Tests

1. **The rename changes no number.** The suite at `-n 2 -v`, with the node tally against M3's log: identical
   outcomes and subtest counts, plus M4's new nodes.
2. **A static guard.** Every one of the 42 has a parameter `c` and none called `residual` (`inspect.signature`), so a
   later re-addition shows. The four helpers left alone are listed in the test as deliberate exceptions.
3. **The laws on all 42** (§9.3; `tests/_typing_laws.py`), each function through an adapter that builds its
   arguments, ragged and uniform, at orders `(2,)` and `(1, 1)`:
   - **Jet law:** `c` moves by `P*` (`ℓ^-k`) and the inputs by `P`; `X*` stays fixed.
   - **Input law:**
     - arguments: the frame's `D_i · Δ_i^-1` (or the point's `B_i`); the inputs `Δ_i w_i`; `c` by `V` for probe, not
       at all for apply, by `∏Δ_i[n_i]` for entries; the Hessians' direction (`D_dot` / `B_dot`) by `Δ^-1`;
     - output: the mode-axis blocks of `X*` times `Δ_i`, the TT blocks fixed.
   - **The ambient part, left from M1:** `InputLaw.ambient` takes the CP output summed to a dense tensor and
     multiplies it by `⊗_i Δ_i`.
   - Uniform: `Δ` is padded with ones; the masks zero the padding.
   - The frontend's moved frames are not orthonormal, so those calls run under `safety.unsafe()` (the transposes
     are metric-free).
4. **The plan's oracle.** On a drawn batch (`take`, with a per-entry weight), `<J^T(batch.data_metric(r)), ẋ>` is
   compared with a central difference of the dense misfit `½ γ Σ ω² r²` (`oracle.misfit_full`).
   - The direction moves one core at a time. `z` is then affine in `t`, the misfit quadratic, and the central
     difference exact to rounding.
   - Coverage: every kind, and order `(2,)` for probe and apply.
5. **The law tests' teeth.** Mutants: `c` taken as `V_i*` (`Δ^-1` instead of `Δ`), a missing flip, the ambient
   factor on one mode only.

## 5. What I would change, and the decisions

- **D1 (§3):** type the `c` line now. Recommended.
- **D2:** for the upgrade note in P19's list: "`residual=` → `c=` in the 42 transposes and Hessians; positional calls
  are unaffected." There is no shim, which follows the maintainer's clean-break line; an old keyword raises
  `TypeError`.
- **Unchanged:** the Hessians' names (decision 4), the old layer, and `t3_inner_product_corewise_*`.
