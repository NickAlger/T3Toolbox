# S3 review (reviewer) -- probe_derivatives two-hole, base main @ 930686a, tree UNCOMMITTED

Status: FINAL. **VERDICT: REVISE** -- one small must-fix (A1, an API-type fix in the estimator); everything else
is approvable as is, and a re-review need only cover the A1 delta. Reviewer scripts and logs:
`/tmp/claude-0/-home-claude/29728d3c-c433-5028-9de9-1af7ead3c123/scratchpad/impl/review_s3/` (`r1`..`r9`, run from
the repo root as `python -I -B <script>`). Nothing in the repo was edited, nothing committed.

## 1. Correctness, independently -- PASS

`review_s3/r1_correctness.py` (log `r1.log`). 88 cases: d in {4, 3, 1} at a frame base point `(U, L)` x orders
0-3 x K in {(), (2,), (2,3)} x C in {(), (2,)}, plus raw *unsquashed* cores (d = 3 boundary ranks (2, 3); d = 1
(2, 3)). Residual jets carry a random `omega[mode, order]` (z~ = omega^2 r). Per K + C element where the oracle is
unstacked; every oracle norm asserted > 1e-6 (non-vacuous).

| comparison | worst rel. err |
|---|---|
| shipped kernel vs `c22.two_hole_v3(..., 'resweep')` (the F13 prototype) | 4.9e-16 |
| shipped kernel vs the `_trs` twins (`probe_derivatives_two_hole_trs`) | 5.3e-16 |
| shipped kernel vs the block-sum oracle (one-hole at the doubled-rank Y, F5; d = 1 via the doubled embedding) | 6.1e-16 |
| shipped kernel vs **my jax-autodiff oracle** `jax.jvp(jax.grad(theta -> <omega^2 r, S(Phi(theta))>))`, dense Phi (ones-squashed tails), float64; S by nested `jacfwd` in s, independent of the subset expansion | 9.3e-16 |
| shipped kernel at `z~ = omega^2 (S(Phi(theta)) - y)` vs jax's **full HVP of 1/2 sum omega^2 (S - y)^2 minus jax's GN term** `J^T omega^2 J theta_hat` | 8.5e-16 |
| one-shot == `_from_sweep(model probe sweep)` | bitwise, 0 failures / 66 frame cases |

The twins are genuinely the dense-`trs` prototype: built from the existing `*_trs` kernels plus one dense
6-/9-term `contract` per product-rule term (c22's strings). Only `tests/backend/test_curvature.py` uses them;
nothing under `t3toolbox/` imports `_curvature_oracles` (one comment names the file; precedent at
`sampling_derivatives.py:348`).

## 2. The production-form rule -- PASS (read line by line)

- `_adj_tilde_tangent_step` is exactly the product rule of `_adj_tilde_step` in its form: **prop** = the same
  reverse-shift `pair()` (`[jet^(s), (s+1) jet^(s+1)]`, identical to GN's `stacked`) fused into two GEMMs
  (`pair(carry_dot) . Pxi` and `pair(carry) . d(Pxi)` with `d(Pxi) = P_dot xi + P xi_dot`), the order-0 branch as
  GN's; **src** = the inner order scan over the edge order r, `d(edge P) . deta + (edge P) . deta_dot`, live slice
  `W*K*r^2`. Dispatch is on all eleven operands (scan_body principle 3); the inline `_src_step` closure mirrors GN's
  (same justification). `n_frame` explicit, read off the C-only P by both module-level scan bodies.
- `_adj_tangent_sweep_scanned` = `_adj_sweep_scanned` with the undotted carry riding as scan data; sigma by
  `tt_reverse` (works on supercores). Same init/dtype conventions as GN.
- Dotted forward: `build_input_jets`/`compute_sigma_jets`/`compute_tau_jets` (banded) and `compute_deta_jets`
  (order scan) -- the production kernels, unmodified. Undotted state: production `compute_nu_jets` (recurrence),
  `compute_deta/tau/sigma_tilde_jets` (order scans). No corewise eta or dx~i is formed.
- dx~i: six dense-`trs` 4-operand terms, the strings of `_dxi_tilde_jets_step` with K moved -- the GN form;
  module-level map body; d-prefixed uniform twin.
- Assemblies: Tucker = GN's two strings with K moved; TT factored; both W-chunked via `_wchunked_sum` with
  module-level `_tucker_tangent_chunk_summed` / `_tt_tangent_chunk_summed` (`trs` as the `extra` argument, not a
  capture). `_flatten_w` reused. No dense-`trs` prototype code in the package.

## 3. The factored assembly -- PASS

- Algebra: `d(tau~ group) + d(deta~ group) = [d(N xi) + d(M deta~)] . nu + [N xi + M deta~] . nu_dot` -- plain
  distributivity over the shared `nu^(s)` (both pairs are indexed by nu's order s), so 4 finals are exact.
  `M[t,s] = sum_r trs[t,r,s] mu^(r)` (shared by sigma~ and deta~), `N[r,s] = sum_t trs[t,r<2,s] tau~^(t)` -- checked
  against the GN strings `f_sig/f_tau/f_det` index by index.
- Agreement with the 9-term form (random jets, `r3_cost.py`): 1.4e-15 (small), 3.0e-15 (d4 r16 o3 W2000) -- the
  same as the plan's 6-final form (1.4e-15 / 3.1e-15); the twin test at test sizes: <= 4.4e-16 (`r9_margin.py`).
- The binomial-weights-first fix is the right one for numpy: the greedy `_pairwise_path` pairs the two operands
  sharing the most letters, and `rWCa` / `sWCi` share all of W and C, so the plan's 3-operand string runs a
  `(o+1) x 2 x W x r x n` outer product first. Measured (numpy, TT assembly alone): shipped 0.018 / 0.57 s vs the
  plan's 3-operand 6-final form 0.087 / 4.0 s vs the 9-term 0.036 / 3.3 s. On jax (one einsum, XLA picks) the
  2-step costs ~25 % on that term on CPU (0.214 vs 0.172 s at d4 r16 o3 W2000; `r8_jax_forms.py`) and saves ~9 % of
  the per-row scratch (167,936 vs 184,320 B). Results equal to roundoff on both paths.

## 4. Cost and memory -- cost PASS; memory remedy in place (see m1 and A1 below)

- **exact/GN (numpy CPU, `r3_cost.py`, two runs):** d5 r8 W300 o2: **2.81x / 3.09x**; d4 r16 W2000 o3:
  **2.21x / 2.15x**. Gate <= 3.5x PASS at both sizes (implementer 2.34-2.67 / 2.26; plan band 2.4-3.0). The small
  size is noisy on this 2-core box (GN = 0.04 s).
- **nu re-sweep:** 1.9-2.9 % of the two-hole (plan F33: 1-3 %).
- **Memory (uniform + jax f32, `memory_analysis().temp_size_in_bytes`, `r4_memory.py`):** per-W-row scratch,
  two-hole/GN = 2.50x (d5 r8 o2), 3.31x (d4 r16 o3), 3.36x (d4 r32 o3), 3.48x (K = 3) -- reproduces the
  implementer's 2.0-3.5x, over the 1.5x gate, so the plan's remedy (`hessian=`) applies, as C.4 words it.
- **Defaults bit-identical:** `estimate_chunk_size` / `max_chunk_size_within` with `hessian` omitted == HEAD's
  module (loaded from `git show HEAD:`) on 512 configurations (K, C, shards, dtype, W, order, ranks): 0 differ.
- **The `hessian=True` answer, chunked peak vs the GN assembly at its default chunk** (`r4b_memory_rem.py`, W a
  multiple of the chunk): **1.48x, 1.56x, 1.61x, 1.25x** (vs 2.50/3.31/2.02/3.65x at the unchanged chunk).
  See m1.
- The estimator test is non-vacuous: it asserts the two-hole per-row > 1.5x GN's, that `hessian=True` lowers both
  answers, and both bounds (and near-maximality of the budget answer).

## 5. Batching -- PASS

- Signatures match D: jets `(o+1)+W+C`, dotted `(o+1)+W+K+C`, input jets `(2,)+W[+K]+C`, z~ `(o+1)+W+C+(N,)`,
  output `K+C`, W summed.
- HISTORICAL: the new section uses 65 distinct strings: **51 new rows** (all present) + 14 pre-existing GN rows
  (`r2_vocab.py`). Every new row is used. (The handoff/commit message says 56 -- N1.)
- `'W'` vs the plan's `'C'`: for each `'W'` row either supplement pins the split (e.g. `tsWKCa,sWCi`: |W| gives |C|
  from op 2 then |K| from op 1; |C| works symmetrically). Recording the one the call sites pass is what
  `TestCallSiteConsistency` checks, and matches the GN precedent (`tWCa,tWKCo->KCao: 'W'`).
  `TestSupplementContract`, `TestCallSiteConsistency`, the oracle sweep, and the sharding sweep
  (`test_contractions_sharding.py`, run standalone: 1 passed, 1002 subtests, so not skipped) are green.
- Multi-axis: the test covers W=(2,3), K=(2,2), C=(2,3) and W=(). I added multi-axis W through the **uniform
  chunked** path (`_flatten_w` engaged), numpy and jax-x64: W=(2,4)/K=(2,)/C=(2,3) and W=(3,3)/K=(2,2)/C=(2,) at
  chunk 3, real block == ragged <= 4.9e-16 (`r7_multiW.py`).
- Chunking on uniform + jax with >= 3 chunks: the dispatch test runs W = 7 at chunk 2 (3 full chunks through the
  `lax.scan`, plus a remainder) and compares to dense (x64, 1e-12).

## 6. Uniform-readiness claim -- CONFIRMED

`r5_uniform.py`: **garbage** (50x randn) in every padded slot of theta, theta_hat, z~, ww and pp; a C = (2,)
stack whose elements have **different ranks** (the variety); theta masked with its masks, theta_hat masked with
the **same** masks broadcast over K, nothing else masked. Real block == ragged per element: <= 3.1e-16 (d = 3, 4,
1; K = (2,), (2,3); dense and numpy-chunked). Negative control: unmasked theta_hat -> 4.4e2 .. 1.4e3 (F15).
`ut3_apply_masks` broadcasts C masks over a K + C direction as C.10 step 2 assumes. So S8's wrapper is mask theta
+ theta_hat, pack, call, re-mask -- the claim holds and does not mislead S8.

## 7. Tests -- PASS

- T5/T6/T11 present, seeded (`setUp` seeds the global RNG for `TuckerTensorTrain.randn`; per-test
  `default_rng`), `subTest` throughout, numpy-only.
- Mutation sensitivity, three of my own (not among the implementer's 64), each in a scratch copy
  (`r6_mutate.py`): reverse-shift weight `s+1 -> s` in the dotted prop: RED; Tucker tangent chunk W axis of ww/pp
  `1 -> 2`: RED (only the uniform-chunked test catches it -- good coverage of that path); `xi_dot` perturbation leg
  reading `ww` for `pp`: RED.
- Dispatch: jit one-shot ragged; from-sweep compiles once across two points (`traces == 1`); uniform chunked jit.
- No `use_jax` anywhere in the new code (inferred at the lowest level). House style and signature comments are
  followed (see the NITs). CHANGELOG bullet present under the unreleased header.

## Findings

### BLOCKER
None.

### MAJOR

**A1. `estimate_chunk_size(hessian=...)` / `max_chunk_size_within(hessian=...)` is a `bool` under the name of the
settled string mode argument (I-2: `hessian='gauss_newton' | 'exact'`, case-insensitive).**
`sampling_derivatives.py:2136` / `:2191`; used as `if hessian:` (`:2172`, `:2210`). The same keyword will mean a
string on `newton_cg` / `Problem.local_model` / the factories from S4-S6, and the handoff's own next step (item 6)
is that S4/S6 pass it when resolving `chunk_size='auto'` (`optimizers.py:_resolve_chunk_size`). The natural
pass-through `estimate_chunk_size(..., hessian=hessian)` then gives `'gauss_newton'` -> truthy -> the two-hole
chunk on every GN fit, silently (a smaller chunk: slower, not wrong -- but a misconfiguration nothing reports).
This is a new public surface in a breaking release, so fix the type now rather than in S6.
**Fix (either):** (a) preferred -- take the I-2 vocabulary, `hessian: str = 'gauss_newton'`, lower-cased and
validated (`ValueError` naming the two values otherwise), `'exact'` sizing both assemblies; the S6 resolver then
passes the solver's own value through; or (b) keep a bool under a non-colliding name (`exact_hessian=False`) and
raise `TypeError` on a non-bool. Update the docstrings, the CHANGELOG bullet and
`test_hessian_also_bounds_the_two_hole` (add the rejection case). Defaults stay bit-identical either way.

### MINOR

**m1. The memory gate after the remedy is met in the "balanced" sense, not strictly at 1.5x -- record it.** With
`hessian=True` the two-hole's chunked assembly peaks at **1.48x / 1.56x / 1.61x / 1.25x** the GN assembly at its
default chunk (d5 r8 o2; d4 r16 o3; d4 r32 o3; K = 3 -- W a multiple of the chunk, `r4b_memory_rem.py`). The
policy (min of the two balanced chunks) holds each assembly at its **own** jet floor, and the two-hole's resident
jets are themselves 1.1-1.6x the GN's (the dotted companions), so this is principled and matches the docstring
("both assemblies stay within their floors"). But the handoff should not read as "gate met": state the
post-remedy ratio (1.25-1.61x) for `verification_records.md` (S13) and let Nick decide whether a strict 1.5x is
wanted (it would mean balancing the two-hole against the GN floor instead, a one-line change). No code change
required for S3.

**m2. The twin test's 1e-15 threshold has a ~2x margin.** `test_production_equals_the_trs_twins` asserts
`err < 1e-15`; over the test's draws and 20 extra seeds the worst piece is 4.45e-16 (`r9_margin.py`). The plan
says the kernels are "pinned to the twins at 1e-15", so this follows the plan, but CI runs two numpy
generations (different `c_einsum` / BLAS summation orders) and the precedent for kernel-vs-twin pins is 1e-12
(`tests/test_jet_recurrence.py`). The mutations that matter are term-level (a dropped or rescaled
term is an O(1e-2..1) change), so 1e-14 (or 1e-13) loses no sensitivity. Suggest loosening, or keep the plan's number knowingly.

**m3. jax-side cost of the binomial-weights-first fix -- record it.** The 2-step pairs are the right fix for numpy
(5.5-7x on the TT assembly vs the plan's 3-operand strings), but on jax they bind XLA's order: 0.214 s vs 0.172 s
(+25 %) on the TT assembly at d4 r16 o3 W2000, CPU, with ~9 % *less* per-row scratch (`r8_jax_forms.py`). Not
worth changing (memory is the jax constraint, and it improves); worth a line in the handoff so S8's
compile/run measurements are read correctly. (A backend-neutral alternative -- teaching `_pairwise_path` to avoid
outer products -- has a library-wide blast radius and is out of scope.)

### NIT

**N1.** The handoff/draft commit message says "56 HISTORICAL rows"; the diff adds **51** rows (143 -> 194; the
new section uses 65 distinct strings, 14 of them pre-existing). Fix the count.

**N2.** Docstrings of `assemble_{tucker,tt}_tangent_variation_jets` and the kernel say chunking is "uniform +
jax"; it is the uniform path (a `lax.scan` on jax, an eager loop on numpy -- `r7_multiW.py` ran numpy chunked).
The GN `assemble_tt_variation_jets` says "the uniform path". Align the wording.

**N3.** The `t3_probe_corewise_derivatives_hessian` doctest seeds `default_rng(0)` but `t3_corewise_randn` draws
from the unseeded global RNG, so theta is not reproducible (`doctest_style.md`: seed or fixed inputs). The
asserted booleans have wide margins, so it is not flaky; add `np.random.seed(0)`.

### Out of scope (observation for the HANDOFF backlog, not S3's)

**O1.** `_wchunked_sum` computes chunk 0 and the remainder outside the `lax.scan`; XLA apparently keeps their
scratch live together, so a W that leaves a near-full remainder peaks at ~1.8-2x the balanced estimate -- GN too (d4 r16 o3:
9.5 GB at W = 8c vs 17.5 GB at W = 9c - 1, `r4b.log`). It affects the GN assembly and the two-hole equally, so
it does not change S3's ratios, but it undercuts the "balanced" promise of `estimate_chunk_size` for unlucky W.

## 8. Test runs (all on the uncommitted tree, numpy 2.5.3 / jax 0.12.0.1, 2 CPUs)

| run | result |
|---|---|
| `tests/backend/test_curvature.py tests/test_jet_recurrence.py tests/test_probe_derivatives.py tests/test_contractions_interpreter.py tests/test_chunk_size_estimator.py` | 82 passed, 3160 subtests |
| `tests/test_contractions_sharding.py` (standalone, 4 virtual devices) | 1 passed, 1002 subtests |
| `test_dispatch.py::test_jit_probe_derivatives_two_hole` + `test_chunk_size_estimator.py` | 12 passed |
| `pytest --doctest-modules t3toolbox` | 207 passed |
| `docs/getting_started.rst` + every `docs/*.md`, `docs/contributor/*.md` page (bar `doctest_style.md`) | OK |
| full suite, half A (`tests/backend`, estimator, contractions x2, corewise, dispatch, equality, fitting) | 317 passed, 4701 subtests |
| full suite, half B (the rest) | 646 passed, 1 skipped, 40002 subtests |
| total | **963 passed, 1 skipped, 44,703 subtests** (= the implementer's count) |
| slow tier `T3TOOLBOX_SLOW_TESTS=1 tests/test_oracle_sweep.py` (once) | 5 passed, 12,790 subtests (green first time) |

Not run: `sphinx -W` (not installed here, as for the implementer).

## Must-fix

1. **A1** -- the estimator's `hessian=` type: take the I-2 vocabulary (`'gauss_newton' | 'exact'`, validated), or
   a non-colliding bool name that rejects non-bools; docstrings, CHANGELOG bullet, a rejection case in the test.

## Should-fix / record (not blocking)

- m1: record the post-remedy memory ratio (1.25-1.61x the GN assembly; "balanced against its own floor") in the
  handoff for `verification_records.md`; strict 1.5x is Nick's call.
- m2: loosen the twin pin to 1e-14 (or keep the plan's 1e-15 knowingly; ~2x margin today).
- m3: note the jax +25 % / -9 % memory trade of the 2-step pairs in the handoff.
- N1 (51 rows, not 56), N2 (chunking wording), N3 (seed the doctest's global RNG).
- O1 for the HANDOFF backlog (`_wchunked_sum` remainder doubles the peak; pre-existing, GN too).

VERDICT: REVISE

---

## Re-review round (A1 + nits applied; tree still uncommitted)

- **A1 -- RESOLVED.** `_check_hessian_mode` (`sampling_derivatives.py:2016-2025`): rejects anything that is not a
  `str` whose lower-case is in `('gauss_newton', 'exact')`, with a `ValueError` naming both values; both estimators
  call it first and branch on `== 'exact'`. Behaviour (`review_s3/r10_rereview.py`): `'gauss_newton'` /
  `'GAUSS_NEWTON'` == the default (6303 balanced, 1323 budget at d4 r16 o3 W32000); `'exact'` / `'EXACT'` /
  `'Exact'` agree (2975 / 399, the old `True` answers); `True`, `False`, `None`, `0`, `'gn'`, `' exact'` all raise
  `ValueError` in both. **Defaults bit-identical to HEAD** on the 512-configuration grid (768 comparisons incl. an
  explicit `hessian='gauss_newton'` column): 0 differ. New `test_hessian_mode_is_the_solver_vocabulary` covers the
  rejections (bools, None, int, typos, empty) in both estimators; the bounds test uses the string vocabulary and
  checks case-insensitivity. Docstrings and the CHANGELOG bullet use the vocabulary.
- **m2** twin pin now 1e-14 with the reason in the docstring (observed <= 4.5e-16). **N1** handoff says 51 rows.
  **N2** "the uniform path (a `lax.scan` on jax ...)" wording at `:2956`, `:3096`, `:3210`. **N3** doctest seeds
  the global RNG. **m1 / m3** recorded in the handoff (post-remedy 1.48/1.56/1.61/1.25x; jax +25 % / -9 %).
- Runs: `test_chunk_size_estimator.py` + `backend/test_curvature.py` + `test_contractions_interpreter.py`: 57 passed,
  2219 subtests; `pytest --doctest-modules t3toolbox/backend/sampling_derivatives.py`: 1 passed (the new doctest);
  `python -m doctest docs/chunking.md`: OK. Full suite, same two halves: **318 + 646 = 964 passed, 1 skipped,
  44,710 subtests** (+1 test / +7 subtests: the new rejection test).

VERDICT: APPROVE
