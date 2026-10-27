# Deferred items from the slice reviews

| from | item | where it lands |
|---|---|---|
| S0c review m1 | The uniform kinds' `forward` raises on a K-stacked variation (`uniform_fitting.py` `_variations`, ≈ :90). The plan's §D needs K+C support for the hvp. Fix: mirror `_remask`'s stack handling in `_variations` + a test | S8 (uniform exact mode), at the latest |
| S0c review m2 | Record the real eager re-mask overhead (~1.1 % on a small transpose including the host mask build; ~0.13 % large; 0 under jit) beside F34 in `verification_records.md` | S13 |
| S0d review | `examples/fit_hilbert_uniform_newton_cg.py` prints a "matches the ragged run to numerical tolerance" claim that was already false at HEAD before S0d | S13 (examples/docs pass) |
| S0d handoff | `docs/contributor/curvature_internals.md`, the release notes and the CLAUDE.md current-state pointer carry the zero-reference material | S13 |
| S0e handoff | `dev/paper_scope.md` still says `GaussNewtonModel` (dev/ is not edited by slice agents) | S13 (dev archive / HANDOFF pass) |
| S0e review | Plan C.16's comment "reduced to minimal ranks on manifold geometries" is wrong: today's `_setup` reduces a uniform `x0` on BOTH geometries and a ragged `x0` only on manifold geometries; the `least_squares_problem` docstring states the truth | plan text amended at S13's archive pass |
| S0e review | A variant of the `docs/conf.py` xref transform would also fix dead `:py:func:` links on ~90 pages while staying `-W` clean | optional docs commit, S13 |
| S1S2 review | The slow-tier O2 oracle sweep is flaky (draws from the unseeded global RNG; one run failed `grad_fd` at 1.05e-6 / 4.9e-6 against a `< 1e-6` bar, on base and final alike). Seed it | before the S13 release gate |
| S1S2 handoff | Plan M5 names only `AXIS_SIZE`; `tests/test_contractions_sharding.py` has its own size table and needed l/n/x/z too (done in S1) | plan text amended at S13 |
| S3 review (**for Nick**) | Memory gate: with `estimate_chunk_size(hessian='exact')` the two-hole's TT-assembly peak is **1.25–1.61× the GN assembly** (balanced against the exact sweep's own larger jet floor) — near, not strictly under, the plan's 1.5× gate. The plan's remedy is in; whether 1.5× must be strict is Nick's call. Also: on jax CPU the two-step pair intermediates cost +25 % time for −9 % memory | Nick, before S13 |
| S3 review | `_wchunked_sum` roughly doubles the peak when W leaves a near-full remainder chunk — pre-existing, affects GN too | backlog |
| S3 implementer | S4/S6 must pass `hessian='exact'` when resolving `chunk_size='auto'` in exact mode (the estimators now take the solver's string vocabulary) | S4 / S6 |
