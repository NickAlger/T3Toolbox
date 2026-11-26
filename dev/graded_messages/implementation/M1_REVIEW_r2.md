SIGN OFF

# M1 review, round 2

Reviewer, 2026-11-22, on `M1_RESPONSE_r1.md` and the uncommitted tree. Both should-fix findings of round 1 are
resolved, and the minor points are taken.

1. **Mixed precision (finding 1): resolved.**
   - `test_through_the_metric_it_is_the_old_block_weighting_bitwise` now runs the float32/float64 pairings and
     asserts the dtype and `np.array_equal`.
   - I reran my mutant L4 (`ω²` squared in the higher precision, then cast) on a fresh scratch copy. It now fails 8
     subtests, every one of them a float32 `r` with a float64 weight: probe ragged, probe packed and apply. The
     unmutated copy passes: 26 passed, 275 subtests in `test_data_metric` and `test_type_laws`.
2. **The deleted comment block (finding 2): resolved.**
   - I extracted the block from `git show cb3c0fdc:t3toolbox/fitting.py` and from the working tree, from
     `# Layer dispatch for the model` to `def _is_uniform_frame`, together with the lines before it. `diff` finds them
     identical.
   - `git diff cb3c0fdc -- t3toolbox/fitting.py` now shows only the planned changes: the module text, the re-exports,
     `__all__`, the comment on `rescale_coordinates`, and the removal of the frontend `dual_jet_to_jet`.
3. **Minor points: taken.**
   - `signature_style.md` states the convention: algebraic types in Unicode (`⊗`, `⊕`, `ℝ`, `*`), as in the
     glossary; identifiers in ASCII (`omega`, `gamma`), in code and in the comments that name them.
   - The handover's count reads 10 new nodes in `test_type_laws.py`.
   - `test_jax_absent` no longer imports `safety`.

The changes since round 1 are a test, a restored comment and a sentence on a page. The implementer's reruns of the
touched files and module doctests cover them, so my round-1 conclusions stand: the 37 tally nodes, the bitwise
agreement with my construction, and no existing number changed. Nothing is left running, and the scratch copies are
deleted apart from `scratchpad/m1rev/new`.
