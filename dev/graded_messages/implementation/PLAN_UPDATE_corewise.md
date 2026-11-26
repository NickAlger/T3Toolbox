# Plan update: the corewise surface (2026-11-20)

**Revised after the critic's review** (`PLAN_UPDATE_corewise_REVIEW.md`; the answers in
`PLAN_UPDATE_corewise_RESPONSE_r1.md`): the bridge goes through the geometry (N47 takes `geom` first; three tests);
one form for a direction (the pair); `sum_over_samples=True` on the corewise Hessians; `_jacobian` (recommended) or
`_jvp` for the maintainer, the plan's text writing `_jacobian`; the numpy dense oracle as the primary test; 5.2 and 7.4.
The tables below are brought up to date. **Amended after the critic's second review** (`PLAN_UPDATE_corewise_REVIEW_r2.md`,
agreed): on a shared geometry N47's gradient is the mean over each group, replicated, in raw coordinates (the projection
onto the tied subspace, so an external optimizer's line search sees the true slope; the sum would overstate it by `k`),
stated in P17 and N47, with a gradient check along a tied random direction in the shared test (it fails for the sum).

Writer: the implementer, at the maintainer's second planned pause (after P11, `e34f7ca7`). For the reviewer's critique,
then the coordinator, then the maintainer. Only `PART1_PLAN.md` and this file were edited.

## The change

Corewise optimization began as a substitution into the manifold code (`t3_corewise_frame(x) = (B, G, G, G)`). The
engine already treats it as first class: its own derived graph, each sweep once, `core_directions`. The public surface
now does too. The maintainer approved three additions and one check:

1. **Gap 1, P12:** the corewise forward derivative `J x_dot`, backend and uniform twins.
2. **Gap 2, P13:** frontend methods for it and for the corewise Hessian. This reopens "the corewise Hessian has no
   frontend method", on the maintainer's rule that each user can do everything in their own layer.
3. **Gap 3, P17 and P19:** a tested recipe for external optimizers on a corewise fit, with a small helper as an option.
4. **Gap 4, P20:** `t3_corewise_frame` checked at P20, with the maintainer consulted between P19 and P20.

## What changed, where

| Section | Change |
| --- | --- |
| Head | A status paragraph, "Changed on 2026-11-20 for the corewise surface", stating the four points and that nothing else changed |
| 2.4, the function table | One row: today none (the substitution `tv_probe(ww, (B_dot, G_dot), t3_corewise_frame(x))`); new `t3_probe_corewise_jvp(x, ww, core_directions, order=())`, its twin, apply alike, entries without `order` |
| 2.4, the paragraph on the transposes | "The corewise Hessian has no frontend method and stays" becomes: it stays in the backend and also gets a frontend method (the maintainer's ruling, his rule). A paragraph on the forward derivative: the trio at a point (`J x_dot`, `J^T c`, `nabla^2 <c, z> x_dot`); one engine call wanting `'Jp'` (P8's raw route); why the substitution duplicates the sweeps through `G`; **the forward derivative has no sum flag (no sum); the corewise Hessian gains `sum_over_samples=True` (row and sketch)** |
| 2.4, the sketches | The signature of `t3_probe_corewise_jvp`; `ut3_probe_corewise_jvp` added to the list of uniform twins |
| 2.4, "New in these modules" | The six new names. The counts of the removal table (166 / 30 / 5 / 131) are unchanged, said explicitly |
| 2.5, the class table | `TuckerTensorTrain` after: 9 becomes 15 (plus three `*_corewise_jvp`, three `*_corewise_hessian`); `UniformTuckerTensorTrain` after: 6 becomes 12. The removed columns are unchanged. Two rows in the today-to-after table; a paragraph "New (2026-11-20)" on the methods and their `core_directions` |
| 5.2 | "The names are settled except those of section 11.5, open since 2026-11-20"; the rows of P12, P13, P17, P19 and P20 name the new content |
| 7.4 | A bullet: the bridge goes through the geometry so a step keeps shared factors tied; P17 tests it |
| P12 | A bullet: the six functions; nothing breaks (the Hessians' new keyword has today's default). Tests: **a numpy dense oracle first** (`J x_dot = dense_probe(T_dot)`, `T_dot` the sum of the `2d` one-core-replaced tensors, exact at every order), `jax.jvp` second, today's substitution while it stands, the adjoint identity, **a structural step count**, the uniform twins' masks, garbage (in `core_directions` too), mirroring, the Hessians' unsummed blocks, dispatch rows |
| P13 | A bullet: the six methods on each of the two classes; nothing breaks. Tests: each equals its backend function; doctests; **the Hessian's symmetry and a polynomial-exact difference quotient**; mirroring; `check_sample` |
| P17 | A bullet: **the bridge through the geometry**, the helper N47 recommended; the lbfgs and optax examples take it; tests in the suite against the library's own optimizers' objective: plain, **shared factors** (the tie kept), **regularized** (equal to `point_objective`), **a drawn batch** (equal to `mc_sgd`'s step objective); L-BFGS-B and optax skipped where absent |
| P19 | Pages steer corewise users to the corewise functions. P19 ends with a note to the maintainer listing every remaining use of `t3_corewise_frame` and `ut3_corewise_frame` |
| 5.4 | A row for the substitution path: it stops at P16 and P19, and is deleted at P20 if unused, after the consultation |
| 8 | A "Corewise" paragraph: `entries_apply_probe.md`, `transposes.md` (the trio), `fitting_and_optimization.md` (the external-optimizer recipe as a doctest), `naming_conventions.md` (the `_jvp` stem); the substitution replaced where shown |
| 10 | A third "before the deletion" item: the check-in on `t3_corewise_frame`; delete if nothing needs it, else keep with the reason |
| 11.2 | O-48 (names: `_jacobian` or `_jvp`), O-49 (one form for a direction, the pair), O-50 (adopt the helper through the geometry), each with a proposal |
| 11.5 (new) | The proposed names N43 to N47, each with its alternatives |

**Not edited:**
- `API_IMPACT.md`. It is the design-phase record, and its row "`t3_corewise_frame` is unchanged" still holds until P20.
  The coordinator asked for edits to two files only.
- P16. The lbfgs example moves to `fitting.local_model` there as planned, and P17 then moves it to the recipe.

## The names proposed (section 11.5)

| # | Name | Alternatives |
| --- | --- | --- |
| N43 | **The maintainer's choice between two:** `t3_{probe,apply}_corewise_jacobian(x, ww, core_directions, order=())`, `t3_entries_corewise_jacobian(x, index, core_directions)` (recommended), or the same with `_jvp` | `_forward`, `_tangent`, `_derivative` |
| N44 | `ut3_*_corewise_jacobian` (or `_jvp`) | as N43 |
| N45 | `{probe,apply}_corewise_jacobian(ww, core_directions, order=())`, `entries_corewise_jacobian(index, core_directions)` (or `_jvp`) | as N43 |
| N46 | `{probe,apply}_corewise_hessian(ww, residual, core_directions, order=(), sum_over_samples=True)`, `entries_corewise_hessian(index, residual, core_directions, sum_over_samples=True)` | `_hvp` (rejected: one stem per operation across the tiers) |
| N47 (recommended, O-50) | `point_value_and_gradient(geom, oracle, batch, x_cores, regularizer=None) -> (f, g)` beside `point_objective` (N20); `fitting.point_value_and_gradient(geometry, oracle, batch, x, regularizer=None)`; `g` the pair of core tuples in the geometry's coordinates | `point_objective_and_gradient`; `corewise_value_and_gradient`; `flat_objective(...)`, a closure for scipy |

**`_jacobian` or `_jvp`.** The plan's text writes `_jacobian`, the recommendation; if the maintainer chooses `_jvp`,
every `*_corewise_jacobian` reads `*_corewise_jvp`.
- *For `_jacobian` (the critic's argument, which I adopt):* the existing tails name the operator applied (`_transpose`
  is `J^T` on a residual; `_hessian` is defined as the derivative of `_transpose` along directions). So `_jacobian`,
  `_transpose`, `_hessian` is one family. My objection, that it suggests the whole matrix, applies to `_hessian`
  equally, and the signature settles both.
- *The library's own precedent:* `tv_*_jacobian_from_sweep` names the applied product `J v`. Those are deleted at P20;
  no current name collides.
- *For `_jvp` (my first proposal):* it is the term a jax user knows at once. But its siblings are `_vjp` and `_hvp`,
  so beside `_transpose` and `_hessian` it mixes two schemes.

## The open choices

- **O-48:** `_jacobian` (recommended) or `_jvp`; `_hessian` kept on the frontend.
- **O-49:** one form for a direction, the pair of core tuples (proposed firmly). A `TuckerTensorTrain` holding a
  direction is a trap: its `+` concatenates ranks. N47 returns its gradient as the pair too, so it can be passed back
  as a direction.
- **O-50:** adopt the helper N47, through the geometry. A raw query flattened by hand is right only for plain
  `COREWISE`, on a full batch, with no regularizer. Otherwise it gets three things wrong, silently:
  - the tie of shared factors;
  - the regularizer's form;
  - the minibatch rule for the regularizer.

  The decision is the maintainer's.
- **The check-in before P20.** The frontend `COREWISE` geometry's `T3Frame` (O-4) likely still uses
  `t3_corewise_frame`, so P19's note will list at least that use.
