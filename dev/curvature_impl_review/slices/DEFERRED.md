# Deferred items from the slice reviews

| from | item | where it lands |
|---|---|---|
| S0c review m1 | The uniform kinds' `forward` raises on a K-stacked variation (`uniform_fitting.py` `_variations`, ≈ :90). The plan's §D needs K+C support for the hvp. Fix: mirror `_remask`'s stack handling in `_variations` + a test | S8 (uniform exact mode), at the latest |
| S0c review m2 | Record the real eager re-mask overhead (~1.1 % on a small transpose including the host mask build; ~0.13 % large; 0 under jit) beside F34 in `verification_records.md` | S13 |
| S0d review | `examples/fit_hilbert_uniform_newton_cg.py` prints a "matches the ragged run to numerical tolerance" claim that was already false at HEAD before S0d | S13 (examples/docs pass) |
| S0d handoff | `docs/contributor/curvature_internals.md`, the release notes and the CLAUDE.md current-state pointer carry the zero-reference material | S13 |
