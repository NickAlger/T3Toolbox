# S7b handoff (implementer) -- DONE, all gates green; base main @ b6ad78b; tree UNCOMMITTED (13 files). Scratch: impl/s7b/
## Design (I-38 "scale by operands")
- Backend `tv_gauge_residual(frame, variations, scale=None)` / `utv_gauge_residual(..., scale=None)`. None = the untouched
  self-relative code path (bitwise). Explicit `scale` (>= 0, per K+C element, trailing-axis broadcast: scalar / C / K+C;
  any other shape = structural ValueError via shared private `tv_operations._check_gauge_scale_shape`) is a FLOOR:
  grams / max(own norm, scale, tiny). Uniform: masked norms, masks untouched (host np), supercores xnp.
- Frontend: a private `_gauge_scale` `functools.cached_property` (returns None = leaf). Arithmetic writes the RESULT's
  slot via `vars(out)['_gauge_scale'] = ...` (the same __dict__ slot cached_property fills; frozen dataclass permits it):
  `a +- b` -> max(scale(a), scale(b)); `c*a` (rmul / neg / truediv route through it) -> |c|*scale(a) ONLY if `a` carries
  one (a scaled leaf is still a leaf; its own norm is exactly |c|||a||); `stack_tangents` -> the leaves' scales stacked
  row-major, only if some leaf carries one. scale(leaf) = corewise_norm() (masked for uniform). Recorded only when
  `safety.checks_active(...)` (safe + eager): unsafe()/a trace pay nothing and store nothing (no tracer in the cache).
  `gauge_residual` = backend with `scale=self._gauge_scale`. Helpers in manifold.py (`_operand_gauge_scale`,
  `_record_{sum,scaled,stacked}_gauge_scale`), reused by uniform_manifold.
- Pytree: flatten stays `(frame, variations), None` -> a round trip rebuilds with empty caches -> self-relative
  (documented: docstrings, both registration comments, numerical_contracts.md, CHANGELOG). Not a leaf, not aux; ==/hash
  unchanged (tested). Every other op (project/retract/transport/project_ambient, normalized, sum_tangents,
  unstack_*, copy, to_jax, reverse, absorb_weights) yields a leaf -- conservative.
- Alternatives rejected (recorded in docs/contributor/precompute_and_caching.md "The one seeded cache"): a field
  (breaks cores-only, enters pytree); pre-seeding cached `gauge_residual` (a gauge contraction per arithmetic op, eager);
  a lazy thunk over operands (retains a whole iterate chain); the sum rule ||a||+||b|| (honest bound, not the ruling).
## Before/after (impl/s7b/repro.py: small ragged probe_derivatives, hessian='exact', steepest + random, h 1e-3..1e-6)
- before (git-archive b6ad78b): gauge(H_fd-Hp) 8.1e-5..1.5e-1 -> MANIFOLD.norm RAISES at all 10 (h,dir); gauge(H_fd)
  alone 6.5e-9 at h=1e-6 (would raise too). after: gauge(H_fd-Hp) 6.3e-17..2.4e-16 -> PASS at all 10;
  |H_fd-Hp|/|Hp| unchanged (2e-9..2.6e-7). Uniform synthetic (impl/s7b/uni_quick.py): base raises 3/4 configs, now 0.
## Bit-identity: impl/s7b/bitident.py vs `git archive b6ad78b` (separate processes; tv/utv scale=None + leaf
  .gauge_residual, K x C x {gauged, raw, x1e10, zero}, 3 seeds): 252 arrays, 0 differ. Permanent: frozen-body tests.
## Tests (numpy-only + dispatch)
- backend/test_tv_operations +TestGaugeResidualScale (4): frozen-body bitwise; floor (scale 0 / 0.5||v|| bitwise ==
  default, large = /scale); broadcast scalar/C/K+C + per-element slices; bad shapes raise.
- test_manifold +TestGaugeOperandScale (6), test_uniform_manifold +TestUniformGaugeOperandScale (7; _CONFIGS incl.
  forced pad + varying-rank C stack; garbage padding; per-element scale == ragged twin), test_fitting
  +TestFiniteDifferenceHessianCheckInSafeMode (real S0b pattern, discriminating: self-relative > 1e-9 asserted).
- test_dispatch: jit tv/utv residual with traced scale; jit `(2.5*(a-b)).gauge_residual` ragged+uniform; eager jax
  `_gauge_scale` / residual / stack_tangents -> jax out.
- Mutants (impl/s7b/mutants.py) 10/10 RED: seeding off, literal max(scale,tiny), sum rule, signed c, stack order,
  recorded in unsafe, None-path summation order, uniform neg drop, uniform backend ignores scale, no shape check.
## Gates
targeted 8 files 322 passed / 1564 sub; module doctests 213 (S7 211, +2); pages + quickstart OK; sphinx -W exit 0;
full -n 2: dispatch+backend 318 / 2096 sub (S7 314 / 2042); rest 744 passed, 1 skipped / 42619 sub (S7 730 / 42509;
1 warning = pre-existing adam-on-manifold). Slow tier SERIAL: 5 passed / 12,790 sub (= S7; 3 pre-existing warnings).
## Judgment calls
1. FLOOR semantics (max(own, scale)) instead of the brief's literal "divide by max(scale, tiny)": never stricter than
   self-relative (the literal max rule makes `(v+v)` up to 2x stricter than `v`/`2*v`); scale=0 == default bitwise;
   identical wherever it matters (cancellation: own << operands). Raw users get the same contract.
2. Recording gated on `checks_active` (no cost/tracers under unsafe()/trace); a tangent built in unsafe() is a leaf.
3. Scaled LEAF stays a leaf (no norm per scalar multiply; mathematically identical under the floor).
4. Scope = the ruling's ops (+, -, scalar *, stack_tangents; neg/rmul/truediv route through *). normalized /
   sum_tangents / unstack_tangents not propagated (leaf = conservative) -- easy follow-up if wanted.
5. Non-scalar `c` in `*` records nothing (corewise_scale broadcasts it over core axes; not a scalar multiple).
6. Docs beyond the brief: precompute_and_caching.md (the seeded cache is mechanism 3 in form; its 3 admissibility
   properties + alternatives) and a catalog implementation note. Heading anchor avoids apostrophes (MyST slug).
## Draft commit message
backend+frontend: the safe-mode gauge check scaled by the operands (S7b, I-38) -- tv/utv_gauge_residual take
scale= (a floor under the own norm; None = unchanged, bitwise); T3Tangent/UT3Tangent +, -, scalar *, stack_tangents
record the operands' scale in the result's cached slot, so MANIFOLD.norm(H_fd - hessian(p)) passes in safe mode
  Body: Slice S7b (I-38; S0b report section 5). The near-cancelling difference of gauged tangents was gauge-checked
  against its own tiny norm and raised on each operand's 1e-16 roundoff. A leaf keeps the self-relative check; a
  genuinely ungauged difference still fails. Cores-only dataclasses; the scale is not pytree state (a jit round trip
  reverts to self-relative). Bit-identical to b6ad78b at scale=None; 10/10 mutants red.
