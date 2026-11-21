NOT YET: the update is consistent and confined to the sections it names, but the external-optimizer bridge (P17, N47) leaves out the geometry, which carries sharing, the regularizer's form and (with the batch) the minibatch scale. Three smaller points should be settled in the plan text before the maintainer is asked.

# Review of the plan update for the corewise surface (2026-11-20)

Reviewer, on `implementation/PLAN_UPDATE_corewise.md` and `git diff dev/graded_messages/PART1_PLAN.md`. I read
`docs/naming_conventions.md` (the `_hessian` and `_tangent` entries), the plan's sections 2.4, 2.5, 5.2, P12 to P20,
7.4, 8, 10 and 11, and `point_objective` (N20). Nothing was run.

## 1. Coverage and consistency: good

Every section the account lists is changed, and the diff touches nothing else.

**The sections changed:**
- the head paragraph;
- 2.4: a row, the paragraph, the sketch, and "New in these modules";
- 2.5: the class table and the today-to-after table;
- 5.2: the open names, and the P12, P13, P17, P19 and P20 rows;
- the bullets of P12, P13, P17 and P19;
- the 5.4 row;
- sections 8 and 10;
- 11.2, and the new 11.5.

**The counts are right.**
- `TuckerTensorTrain`: 9 + 3 + 3 = 15.
- `UniformTuckerTensorTrain`: 6 + 6 = 12.
- The removed columns and the removal table (166 / 30 / 5 / 131) are rightly unchanged: the six new names are additions,
  and that is said explicitly.

**The arguments follow N16's rule:**
- the object first;
- then the sample;
- then what is applied (`core_directions` third for the jvp; `residual`, then `core_directions` for the Hessian);
- then `order`.

`API_IMPACT.md` left alone is right: it is the design record, and its row on `t3_corewise_frame` holds until P20.

**Two small gaps:**
- **5.2 still opens with "The names are settled"**, then says some are open. Reword it to "settled except section 11.5,
  open since 2026-11-20".
- **Section 7.4 (shared Tucker factors) is not touched**, but the new recipe bears on it (finding A). Add a line there.

## 2. Findings

**A. The external-optimizer bridge must go through the geometry (should fix before the maintainer is asked).**

As written, P17's recipe is "one `oracle.query(x.data, batch, ('f', 'g'))` at the point, flatten with
`t3_to_vector`". N47 deliberately takes no geometry ("on a corewise fit the frame is the point"). That holds only for
plain `COREWISE` on a full batch with no regularizer. Three things the writer missed all live in the geometry or the
batch:
1. **Shared Tucker factors.** On `shared(COREWISE, sharing)` the parameters are one factor per group.
   - The raw gradient has one block per mode, and must be summed over each group (the tied embedding's transpose).
   - Flattening the raw cores with `t3_to_vector` and handing them to L-BFGS treats the copies as independent, so the
     first step unties them silently.
   - The recipe must use the geometry's frame and gradient map (`geom.frame(x)`, P15; the tie), or state that it is
     for unshared fits only and check it (`has_shared_tucker_factors`).
2. **The regularizer depends on the geometry.**
   - `IdentityRegularizer(lambda)` is weight decay on `COREWISE` and the HS ridge on `MANIFOLD`, so "the regularizer's
     gradient" is not one formula.
   - `point_objective` (N20), which N47 says it extends, takes `geom` first for exactly this reason.
3. **The minibatch scale.**
   - The oracle multiplies `f` and `g` by `minibatch_scale` on a drawn batch (O-21), which the recipe inherits for free.
   - But the regularizer is scaled by the library's own minibatch rule (`batch/n`, as in the optimizers today), so a
     caller adding `rho` by hand on an optax minibatch loop must apply the same rule. Otherwise the objective differs
     from the one `mc_sgd`/`adam` minimize.

*Resolve:*
- N47 takes the geometry first, as N20 does: `point_value_and_gradient(geom, oracle, batch, x, regularizer=None)`.
- It returns `f` (plus `rho`, scaled as the library's minibatch steps scale it) and the gradient in the geometry's
  coordinates (tied on a shared geometry).
- P17's tests add three cases:
  - a shared corewise fit, where the flat gradient equals the tied one and a step keeps the tie;
  - a regularized fit, where the helper equals `point_objective` plus a gradient check;
  - a drawn batch, where it equals the step objective `mc_sgd` uses.
- The recipe text says the same.

This strengthens the case for adopting the helper (O-50): three of its pieces are the razor's "easy to get wrong".

**B. O-49 against N47: one form for a direction.**
- I agree with O-49 (the pair of core tuples only). It is what the corewise transposes return today (the raw
  `(tucker_grads, tt_grads)`).
- A `TuckerTensorTrain` holding a direction also has a trap: its `+` is the tensor sum (rank concatenation), not the
  sum of directions. The library's arithmetic on core pairs is `corewise.*`.
- But N47 then returns `g` as a `TuckerTensorTrain`, so its gradient could not be passed to `*_corewise_jvp` as a
  direction.
- *Resolve:* N47 returns the pair too (flatten with `t3_to_vector(g)`, which takes the pair as it is), or the plan
  says why it differs.

**C. `sum_over_samples` on the corewise Hessian (decide, and say so).**
- The plan's own rule (2.4) is "`sum_over_samples=True` where there is a sum". The corewise Hessian sums over `W` as
  the transpose does, and the engine's `keep_W` already serves `Kp` (P10).
- Yet the Hessian's row and sketch, and N46, have no flag. The omission predates this update, but the new frontend
  methods make it public twice over.
- *Recommend:* add `sum_over_samples=True` to the six Hessians (backend and frontend). It costs nothing, and makes them
  parallel to the transposes.
- The jvp has no sum (its output is per sample, of the type of `z`), so it rightly has no flag; say so in 2.4 so the
  asymmetry reads as deliberate.

**D. The names (O-48): `_jacobian` fits the trio better than `_jvp`. The maintainer's call.**
- The two existing tails name operators. `naming_conventions.md` defines `_hessian` as "the derivative of the same-stem
  corewise transpose along raw core directions": the operator `∇²<c, z>` applied to `core_directions`. `_transpose` is
  `Jᵀ` applied to `residual`.
- The operator of the new function is `J`, so `_jacobian` / `_transpose` / `_hessian` reads as one family: `J`, `Jᵀ`,
  the derivative of `Jᵀ`. `_transpose` even reads naturally as "the Jacobian's transpose".
- The writer rejects `_jacobian` because it "suggests the whole matrix". That objection applies equally to `_hessian`,
  which also names a matrix, takes a direction, and returns a product. The signature (`core_directions`) disambiguates
  both.
- `_jvp` is an automatic-differentiation acronym of the *product* form. Its natural siblings are `_vjp` and `_hvp`, not
  `_transpose` and `_hessian`, so it mixes two naming schemes in one trio.
- `_jvp` is clearer to a jax user, and the maintainer uses jax, so it is defensible.
- *Recommend:* propose `_jacobian` (N43 to N45), with `_jvp` as the first alternative. Section 8's `naming_conventions.md`
  entry records whichever is chosen, beside `_hessian`, in the same "applied operator" wording.
- N46 (`_hessian` on the frontend, not `_hvp`) is right: one stem per operation across the tiers.
- On `_tangent`: the writer's reason is right (it would read as a manifold tangent). Note also that the `_tangent` stem
  belongs to the kernel pieces that P20 deletes.

## 3. The tests planned: mostly sufficient; three additions

P12 has:
- `jax.jvp`;
- the old substitution while it stands;
- the adjoint identity per element of `K`;
- a step count;
- the uniform masks, garbage and mirroring;
- dispatch rows.

That is the right set. What to add or sharpen:
1. **A numpy dense oracle as the primary check.** The house rule is that numerical correctness is checked numpy-only;
   `jax.jvp` is skipped where jax is absent, and the substitution is deleted at P20. A dense ground truth is exact and
   cheap here:
   - `z` is linear in the dense tensor `T(x)`, and `T` is multilinear in the `2d` cores;
   - so `J x_dot = dense_probe(T_dot, ww, order)`, with `T_dot` the sum of the `2d` dense tensors that each have one
     core replaced by its direction.

   This works at every `order`, input jets included, and survives P20. Keep `jax.jvp` as the second, independent
   check.
2. **The step count, made exact and structural.** Count the rows the engine runs (`_closure` for `want=('Jp',)` on the
   corewise call, or a counting wrapper on the sweep bodies) and assert each sweep row once. State the substitution's
   larger count once, as the record. Not timing.
3. **The Hessian methods (P13):**
   - add the symmetry `<y, H x> = <x, H y>` (cheap, independent);
   - the doctest's "difference quotient" should be a polynomial-exact one (`J^T c` is polynomial in `x`, of degree at
     most `2d - 1`), or jax where present.

   The P12 entries need nothing new for the uniform layout: the engine's exit masks cover `Jp`. The garbage case should
   put garbage in the padding of `core_directions` too.

## 4. The rest

- **O-50:** adopt the helper, reshaped by A. The recipe stays documented for backend users, through the helper's
  backend twin.
- **The P20 check-in (section 10):** right as written. The frontend `COREWISE` `T3Frame` (O-4) is the likely surviving
  use.

## Requested before the maintainer sees it

1. **(A)** Reshape N47 and P17's recipe around the geometry (sharing, the regularizer, the minibatch scale), with the
   three tests. Add a line to 7.4.
2. **(B)** Make N47's return form consistent with O-49.
3. **(C)** State the `sum_over_samples` decision for the Hessians (recommend adding it), and that the jvp has none on
   purpose.
4. **(D)** Put `_jacobian` beside `_jvp` in 11.5 with the trio argument, and let the maintainer choose.
5. **Tests:** the numpy dense oracle, the structural step count, and the Hessian's symmetry and polynomial-exact
   quotient.
6. **Nit:** 5.2's "The names are settled".
