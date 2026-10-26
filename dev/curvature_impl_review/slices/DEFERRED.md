# Deferred items from the slice reviews

| from | item | where it lands |
|---|---|---|
| S0c review m1 | The uniform kinds' `forward` raises on a K-stacked variation (`uniform_fitting.py` `_variations`, ≈ :90). The plan's §D needs K+C support for the hvp. Fix: mirror `_remask`'s stack handling in `_variations` + a test | S8 (uniform exact mode), at the latest |
| S0c review m2 | Record the real eager re-mask overhead (~1.1 % on a small transpose including the host mask build; ~0.13 % large; 0 under jit) beside F34 in `verification_records.md` | S13 |
