# S0c handoff: uniform transpose output hygiene (base b6295d0, uncommitted)
## Files changed
- backend/uniform_fitting.py: `_UniformKind._remask` = `ufv_variation_masks_over_stack(frame_data[5], C, K)` + `ufv_apply_variations_masks`. All six kinds' `transpose` now run the bare seam and then `_remask`.
- backend/utv_sampling.py: `_remasked_transpose_output` attaches AND applies `_gauge_masks_over_Knew`. Used by the six public `utv_*transpose`. The seam comments and docstrings now say the seams are "bare by contract".
- backend/ut3_sampling.py: `_remasked_core_gradient` = `ut3_apply_masks` with the point's masks. Used by the six `ut3_*_corewise_*transpose`. Rewrote the comment that said the padding is "NOT guaranteed clean".
- tests/test_uniform_probing.py: `TestTransposeOutputHygiene` covers utv x6 and ut3 x6 (528 subtests).
- tests/backend/test_uniform_fitting.py: `TestKindTransposeOutputHygiene` covers 6 kinds x 2 frame types (144 subtests).
- CHANGELOG.md: `## [2026.3.0] — unreleased` / `### Changed — breaking` entry. docs/contributor/uniform_internals.md: one masking-schedule bullet.
## c39 (python -I -B), before -> after
- Padded output: 4.1e7 / 3.2e5 (small COREWISE / MANIFOLD) and 3.0e12 / 4.4e8 (large) -> 0 everywhere. Real block bit-identical: True.
- c39's re-mask cost (the multiply only): 0.55 / 0.38 / 0.12 / 0.09 % -> 0.50 / 0.42 / 0.11 / 0.06 %. After the change this is a redundant second pass.
- Full kind re-mask including the host-side mask build: ~22 us, which is 1.1 % (small) and 0.15-0.18 % (large). About 12 us of that is `ufv_variation_masks_over_stack`. Under jit the masks are folded constants; a jax x64 jit run is also bitwise.
- CG state: before S0c, the UNIFORM_COREWISE padding was g 3e10-2.5e12 and hvp(g) 4.5e21-1.5e25. MANIFOLD was already 0. Now 0 on both.
## Tests (T27)
- Per wrapper, four checks: (a) exact masks from the RAGGED frame/point core shapes; (b) real block bitwise equal to the bare seam (the pre-S0c body, frozen in the test); (c) padding == 0, while the bare padding is > 0, so the test is not vacuous; (d) 1e6*(1-mask) garbage in the frame/point -> identical output.
- Cases: default and forced pad; C in {(), (2,), (2,3)}; K in {(), (3,)}; both sum_over_probes; manifold and corewise frames; varying-C; TT-slack.
- Mutations (no re-mask, all-true masks, unshifted gauge masks): every subtest fails for utv, ut3 and kinds.
- Counts: touched files 64 passed / 872 subtests. doctest-modules 194 passed. Docs page doctests OK. Full suite `-n 2`: 901 passed, 1 skipped, 43,211 subtests, EXIT 0 (one later docstring-only edit to ut3_sampling.py; its doctests were re-run).
## Conflicts and judgment calls
1. The CHANGELOG entry sits under "Changed — breaking", following §G. The text says it breaks only code that read the padding.
2. C.0 says the seam docstrings "already say" that project attaches the masks. Only utv_apply_transpose_from_sweep said so, and it described the pre-S0c behaviour. I reworded it.
3. I added the uniform_internals.md bullet (§G does not list it for S0c) so the documented masking schedule stays complete.
4. Check (d) says "direction", but transposes have no direction operand. The garbage goes into the frame or point padding instead. The packed residual's mode padding follows the prefix-zero convention rather than being masked, so it is not corrupted.
5. Swapping the frame masks for the point masks in `_remask` survives the tests. At TT slack, the bare output is structurally 0 on slots real by the point but padded by the frame; the garbage sits only in boundary slots, where both mask sets agree. The code uses the frame argument, as the plan says. T23's check of the same rule belongs to S8.
6. The 1.1 % at the small size is above the plan's 0.08-0.61 % band, because the band timed the multiply alone. I added no K=() fast path to the shared helper, to keep the change small.
7. Found, not fixed: the uniform kinds' `forward` rejects a K-stacked variation (`_variations` raises a reshape ValueError). This matters for the later K+C hvp work (§D).
8. No new test_dispatch test: test_jit_uniform_{probing, corewise_transpose, derivatives, optimizer_wholestep, weighted_gn_model} already jit every re-masked path.
## Draft commit message
uniform: re-mask every uniform transpose output (S0c) -- kinds' transpose, utv_*_transpose and ut3_*_corewise_*transpose zero their padding (was up to 3e12 garbage); real block bit-identical; T27 tests; CHANGELOG [2026.3.0]
Body: the file bullets above, then the two attribution lines from the session reminder.
