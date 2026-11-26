# Plan update for the corewise surface: response, round 1

Writer: the implementer, 2026-11-20, to `PLAN_UPDATE_corewise_REVIEW.md` and the coordinator's rulings. Only
`PART1_PLAN.md` and `PLAN_UPDATE_corewise.md` were edited. Every point is taken.

## Finding A: the bridge goes through the geometry. Done.

- **P17.** The bullet is rewritten as "The external-optimizer bridge for a corewise fit (through the geometry)". It
  says why a raw `oracle.query` flattened by hand is right only for plain `COREWISE` on a full batch with no
  regularizer:
  - **the tie:** on `shared(COREWISE, sharing)` a flat raw gradient unties the copies on the first step, silently;
  - **the regularizer:** `IdentityRegularizer(lam)` is weight decay on `COREWISE` and the HS ridge on `MANIFOLD`;
  - **the minibatch rule:** the oracle scales `f` and `g` by `minibatch_scale` (O-21); the regularizer is scaled as the
    library's minibatch steps scale it.
- **N47.** It takes the geometry first, as `point_objective` (N20) does:
  `point_value_and_gradient(geom, oracle, batch, x_cores, regularizer=None) -> (f, g)`, with its frontend
  `fitting.point_value_and_gradient(geometry, oracle, batch, x, regularizer=None)`.
  - `f` is the objective with the regularizer's term.
  - `g` is the gradient in the geometry's coordinates (tied on a shared geometry), as the pair of core tuples.
- **The tests, in the suite, each against the library's own optimizers' objective:**
  - plain `COREWISE`, with L-BFGS-B and optax (skipped where absent);
  - **shared factors:** the flat gradient equals the tied one, and a step keeps the tie;
  - **regularized:** the value equals `point_objective`, and the gradient passes the gradient check;
  - **a drawn batch:** the value and gradient equal the step objective `mc_sgd` minimizes.
- **If O-50 declines the helper,** the recipe is written out through the same geometry calls, and states what it
  covers.
- **Section 7.4** gains a bullet: the bridge goes through the geometry so that a step keeps shared factors tied.
- **Section 8.** The external-optimizer section of `fitting_and_optimization.md` explains why the raw form is wrong in
  the three cases.
- **O-50 now proposes adopting the helper.** The tie, the regularizer's form and the minibatch rule are three pieces
  easy to get wrong silently, which is the razor's criterion. The decision is the maintainer's.

## Finding B: one form for a direction. Done.

- **O-49 is proposed firmly:** the pair of core tuples only. It is what the corewise transposes return, and a
  `TuckerTensorTrain` holding a direction is a trap, because its `+` concatenates ranks.
- **Section 2.5** says so in the paragraph on the new methods.
- **N47 returns its gradient as the pair too,** so it can be passed back as a direction (`t3_to_vector(g)` flattens
  it).

## Finding C: `sum_over_samples` on the corewise Hessian. Done.

- **The flag is added to the six Hessians,** backend and frontend:
  - `t3_*_corewise_hessian(x, ww, residual, core_directions, order=(), sum_over_samples=True)` in 2.4's row and sketch
    (the blocks `[W] + K + C + core`);
  - the frontend methods in 2.5 and N46.
- **2.4 says the forward derivative has no flag, on purpose.** Its output is per sample, of the type of `z`, with no
  sum.
- **P12 notes the new keyword** (its default is today's behaviour, so nothing breaks) and tests its unsummed blocks.

## Finding D: the names (O-48). Done.

- **Two candidates are presented in 11.5's N43, for the maintainer:**
  - `_jacobian`, with the critic's argument. The existing tails name the operator applied: `_hessian` is defined as
    the derivative of `_transpose`. So `_jacobian`, `_transpose`, `_hessian` is one family, and the "whole matrix"
    objection applies to `_hessian` too.
  - `_jvp`, with mine: the product form, which a jax user knows; but its siblings are `_vjp` and `_hvp`.
- **On collisions:** the library's `tv_*_jacobian_from_sweep` already names the applied product `J v`, and is deleted
  at P20. No current name collides with `*_corewise_jacobian`.
- **I recommend `_jacobian`,** and the plan's text writes it. Section 11.5 says that if `_jvp` is chosen, every
  `*_corewise_jacobian` reads `*_corewise_jvp`.
- **`_hessian` is kept** on the frontend, not `_hvp`.
- **Section 8's naming entry** records whichever stem is chosen, in the "applied operator" wording.

## Point 5: the tests. Done.

- **P12, the primary check:** a numpy dense oracle, exact at every order and independent of jax. It uses
  `J x_dot = dense_probe(T_dot, ww, order)`, with `T_dot` the sum of the `2d` one-core-replaced dense tensors (apply
  and entries alike), at `()`, `(2,)`, `(1, 1)` and mixed input jet sizes. It survives P20.
- **P12, the other checks:**
  - `jax.jvp` is the second check, where jax is present;
  - the substitution while it stands;
  - the adjoint identity per element of `K`;
  - **the step count is structural:** the rows the engine runs, each sweep once, with the substitution's count stated
    once as the record; not timing;
  - garbage also in the padding of `core_directions`.
- **P13, for the Hessian:** the symmetry `<y, H x> = <x, H y>`, and a polynomial-exact difference quotient (`J^T c`
  has degree at most `2d - 1` in `x`), or jax where present.

## Point 6: the small gaps. Done.

- **5.2** reads "The names are settled except those of section 11.5, open since 2026-11-20".
- **7.4** has the bullet of finding A.
- **The head paragraph** says the plan was revised the same day after the critic's review, and what changed.

## The plan's diff against `HEAD`

173 lines inserted, 19 changed, all in the sections the account lists. The counts are unchanged:
- `TuckerTensorTrain` 15;
- `UniformTuckerTensorTrain` 12;
- the removal table 166 / 30 / 5 / 131.
