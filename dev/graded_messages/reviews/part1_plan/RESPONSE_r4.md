# Response to review r4 of `PART1_PLAN.md`

Writer's response, 2026-11-16. The three minor items are fixed in the plan, which is now the final draft
(status line), awaiting the maintainer.

| # | Fix |
| --- | --- |
| 33 | A new row N30 in the table of names. It proposes `regularizer=None` right after the positional arguments of `gradient_descent`, `mc_sgd` and `adam`, by analogy with 8.3's `newton_cg`, and notes that in `mc_sgd` and `adam` it then comes before `draw`. Sections 2.4 and 2.5 now call it N30's proposal. |
| 34 | Section 10's table has a row for `tests/test_jax_absent.py`: the string `bfit.probe_derivatives_kind(order)` in its subprocess script (line 71), removed in P16. |
| 35 | P13 says that if O-27 is answered as proposed, it extends `backend/sampling_definitions.py` (P2's module) so that `z` accepts a dense tensor, with its tests in `tests/test_sampling_definitions.py` and its doctest. P13 already names O-27 in "Needs"; P2 does not depend on it. |
