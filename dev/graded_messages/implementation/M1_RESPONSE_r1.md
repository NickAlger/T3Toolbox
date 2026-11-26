# M1 response to review round 1

Implementer, 2026-11-22, on `M1_REVIEW_r1.md` and Claude's rulings. Every finding is accepted.

## Should fix

1. **Mixed precision.** `test_through_the_metric_it_is_the_old_block_weighting_bitwise` now runs every pairing of
   float32 and float64 (`r` and weight both ways round, and equal). It compares with what the engine uses today,
   `jet_to_dual_jet`'s seed with its turn undone (an exact flip). It asserts the dtype and `np.array_equal`: 40
   subtests, up from 10.
   - The review's mutant (square in the input precision, cast after) now **fails 8 subtests**.
   - With `_omega_squared` as built, everything passes.
2. **The deleted comment block.** The six-line header "Layer dispatch for the model…" is restored before
   `_is_uniform_frame` in `t3toolbox/fitting.py`, verbatim from `cb3c0fdc`. Checked by `diff` against
   `git show cb3c0fdc:t3toolbox/fitting.py`, lines 282 to 290.

## Minor

- **Unicode** (ruled). `signature_style.md` now says that the algebraic types are written in Unicode (`⊗`, `⊕`, `ℝ`,
  `*`), as in the glossary, while identifiers stay ASCII (`omega`, `gamma`), in code and in the comments that name
  them. The page's doctest passes.
- **The handover's count** now reads 10 new nodes in `test_type_laws.py`. So the 37 differing nodes are:
  - 27 new;
  - 7 deleted;
  - 2 that drop exactly their frontend call;
  - the doctest-coverage node (11 to 13).
- **`test_jax_absent`'s unused `safety` import** is dropped, and the handover's note about it is removed.

## Reruns (`-n 2`)

- The touched test files: `test_data_metric`, `test_type_laws`, `test_jets`, `test_jax_absent`, and
  `test_dispatch`'s non-`TestDispatch` classes plus the two jit tests of jets and data preparation. Result: **99 passed, 1,171 subtests, 0 failed** (7:27).
- The module doctests of `fitting.py`, `backend/jets.py` and `backend/fitting.py`: **20 passed**.

No full gate was run: the changes are a test, a comment and a page sentence.
