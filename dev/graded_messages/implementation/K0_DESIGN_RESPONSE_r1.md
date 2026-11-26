# Response to `K0_DESIGN_REVIEW_r1.md`

Implementer, 2026-11-25. The note is revised in place (revision 2, its header says what changed); the rewritten prose
is in `k0/prose_edits.md`; new evidence in `k0/experiments/` (`e1c.py`, `mutate_step.py`). The orchestrator's
rulings on the four open points and on M2 are applied as given. Every finding is accepted.

**S1 (mutation (i) cannot fail). Accepted.** Right: the identity holds for any diagonal weight, so the test's own
square changes both sides. Replaced by library mutations, a step per kind, run on prototypes of the re-bases in a
scratch copy (`mutate_step.py` scales one step's output by `1 + eps`):
- the re-based `test_residual_weights`: `H_tilde_dot_step` ×(1+1e-3) fails both probe subtests; `H_tilde_step`
  ×(1+1e-3) fails all four (apply and entries through their `Kp`, probe through its `g`); `V_top_step`, shared by both
  sides of apply's identity, passes, as an identity test must, and that block is pinned by
  `test_sampling_apply.TestByDefinition`;
- the jvp rows (your point in §6): `H_tilde_dot_step` ×(1+1e-6) fails probe and uniform probe (4e-7 to 8e-7 against
  1e-12), `H_tilde_step` ×(1+1e-6) fails apply and one-hot apply (4e-7 to 6.5e-7) and probe;
- (iii) measured too: the dense forward without `t!` gives 0.92 and 1.0 against 1e-5.
Each mutation now states its expected outcome (note §6).

**S2 (the rewritten prose left to the build). Accepted.** `k0/prose_edits.md` gives before and after for every
docstring and comment K0 rewrites: the seven module docstrings that change (`fitting` describes the oracle and the
batch as they are at K0; `uniform_fitting` loses its second and third paragraphs), `ut3_sampling.py:63-66` (deleted:
it describes the pre-engine wrappers), `fitting.py:154-160` and `:745-749`, `entries.py:113-116`, `common.py:524-526`,
`utv_sampling.py:449-453`, the comments inside `probing.__all__`, and the test prose. It also lists what I reviewed
and left (true after K0). On `probing`'s "Reference module for the signature style": it stays, since the kept
functions carry the house form and the page names the module; the pages' `compute_mu` quotations stay with P19.
The deletions whole stay as line ranges in the note.

**S3 (the added test pins export lists). Accepted.** One assertion kept: `t3toolbox.backend.sampling_derivatives` is
not among the module files and `find_spec` finds nothing. It goes in `tests/test_import_graph.py`
(`test_the_old_sampling_kernels_are_gone`), beside that file's other statements about which modules exist. No
export lists, no name patterns. The count stays 1,882.

**M1 (`taylor_to_derivatives` in the re-based forwards). Accepted.** The factorials are inline; the oracle names no
library function but `dense_*`, and K6 does not touch it.

**M2 (`api_reference.rst:75-80`). Applied as ruled.** A scan of every `:py:` role of the page finds nine missing
targets (the six factories, and `least_squares_problem`, `Problem`, `LeastSquaresSetup` at `:87-90`), plus
`sampling_derivatives` after K0. Both bullets go whole and `:130` loses the module; the text around them stays
correct, including "the one exception to the rule above". Nothing needs new documentation, so nothing is left to P19
here except adding the new names. Before and after are in `prose_edits.md` §3.

**M3 (one of three history comments corrected). Accepted.** All three (`:237`, `:241`, `:299`) are left: they
annotate frozen `HISTORICAL` rows.

**M4 (the negative control should fail (c)). Accepted, and it needed one more patch.** With the engine's entry and
exit masks removed, the corewise faces' padding fills (8e31 to 5e32), but the tangent faces' stays 0.0: they re-mask
again in `_remasked_transpose_output`. With that re-mask also removed (`ufv_masking.ufv_apply_variations_masks`
patched to return the supercores), all six rows fill the padding (E1c, `e1c.py`). The control asserts
`assertRaisesRegex(AssertionError, 'padding not zero')` on clause (c) (note §3.6).

**§6 (order and gate).** Both added: a recorded run of the edited test files on the untouched library, log kept in
`k0/`, and the slow tier of the edited `test_uniform_curvature.py` before the deletion (note §6, Order step 1).

**Open point 1 (the floors).** Applied as ruled: 60, with the count at K0 in the comment. I kept the count rather than
named literals: the named step strings would pin K5's code.

**Open point 4.** Noted that the parametrization does not run as is (your `ttslack_kinds.py`); it goes to K5 as ruled.
