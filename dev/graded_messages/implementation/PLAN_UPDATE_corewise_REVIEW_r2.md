AGREED: all six points of round 1 are taken. One sentence and one test clause on the shared case should be folded in before the maintainer sees it; no further review is needed for that.

# Review of the plan update for the corewise surface, round 2

Reviewer, 2026-11-20, on `PLAN_UPDATE_corewise_RESPONSE_r1.md` and `git diff dev/graded_messages/PART1_PLAN.md`
against `HEAD`. Read only.

## Round 1, point by point

| Point | Asked | Now | Verdict |
| --- | --- | --- | --- |
| A | The bridge through the geometry: sharing, the regularizer, the minibatch scale; three tests; 7.4 | See below | taken |
| B | One form for a direction | O-49 is the pair only (with the `+` trap stated in 2.5), and N47 returns the pair, flattened by `t3_to_vector(g)` | taken |
| C | `sum_over_samples` on the Hessian | See below | taken |
| D | `_jacobian` beside `_jvp` | See below | taken |
| Tests | The dense oracle, a structural step count, the Hessian's symmetry and quotient | See below | taken |
| Nit | 5.2's "the names are settled" | "settled except those of section 11.5" | taken |

**A, the bridge through the geometry.** N47 is now
`point_value_and_gradient(geom, oracle, batch, x_cores, regularizer=None)`, beside `point_objective`, with a frontend
twin. P17 says why the raw form is wrong, in three cases:
- the tie;
- the regularizer's form;
- the minibatch rule.

It has four suite tests, each against the library's own objective: plain, shared, regularized, and a drawn batch
against `mc_sgd`'s step objective. Section 7.4 has a bullet, and so does section 8.

**C, `sum_over_samples`.** The flag is added to the six Hessians: the 2.4 row and sketch (`[W] + K + C`), 2.5, N46,
and a P12 note with a test of the unsummed blocks. The forward derivative has no flag, and 2.4 now says that is on
purpose.

**D, the names.** N43 presents both stems, with both arguments, and the plan's text writes `_jacobian` throughout.
Section 11.5 says the substitution if `_jvp` is chosen. The only `_jvp` occurrences left are the alternatives, which
is right. The fact that `tv_*_jacobian_from_sweep` already names an applied `J v` supports the stem.

**Tests.**
- P12's primary check is the numpy dense oracle (`J x_dot = dense_probe(T_dot, ...)`, with apply and entries alike)
  at `()`, `(2,)`, `(1, 1)` and mixed jet sizes. `jax.jvp` comes second.
- The step count is structural: the rows the engine runs.
- Garbage goes into the padding of `core_directions` too.
- P13 has the Hessian's symmetry and the polynomial-exact quotient.

**Scope.** The diff is still confined to the sections the account lists. The counts are unchanged: 15, 12, and
166 / 30 / 5 / 131.

## To fold in: what "the tied gradient" is, in coordinates

N47 returns `g` "in the geometry's coordinates … as the pair of core tuples", so `t3_to_vector(g)` is "the flat gradient
in the point's own coordinates". On a shared geometry those are the raw coordinates: one factor per mode, replicated
within each group. In those coordinates there are two candidates for the "tied" gradient, and they differ by the group
size `|G|`:
- the sum over the group, replicated, which is the gradient with respect to the *shared* parameter;
- the mean over the group, replicated, which is the orthogonal projection of the raw gradient onto the tied subspace.

Only the second is the gradient of the function an external optimizer actually sees on the flat raw vector, restricted
to tied directions: `<g, d> = <grad f, d>` for every tied `d`. With the sum, L-BFGS's directional derivatives, Wolfe
checks and curvature pairs are off by `|G|` in the group blocks. The zero set is the same, so it is a slowed or
misbehaving line search, not a different minimizer. The other consistent choice is to flatten in the shared parameters
(one factor per group) with the sum.

P17's planned shared tests do not tell these apart. "The flat gradient equals the tied one" and "a step keeps the tie"
hold for both.

*Fold in:*
- One sentence in P17 and N47 naming the convention. I recommend raw replicated coordinates with the projection (the
  mean), since that matches "`t3_to_vector(g)` in the point's own coordinates".
- In the shared test, the gradient check along a **tied** random direction: the polynomial-exact quotient of `f` equals
  `<g, d>`. This fixes the factor `|G|`.

## For the maintainer (unchanged by this round)

- O-48: `_jacobian` (recommended) or `_jvp`.
- O-49: the pair only.
- O-50: adopt the helper.
- The check-in before P20.
