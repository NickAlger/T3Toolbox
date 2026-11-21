# Old derivative kernels against the engine at the jet (measured 2026-11-21, at the third planned pause)

Run from this folder: `PYTHONPATH=<repo>:. python bench_apply_jets.py time <configs>` (or `count`), likewise
`bench_probe_jets.py`; `bench_order3.py` / `bench_order3_probe.py <configs>` add the engine with each three-operand
explicit-path einsum given its cheapest first pair (a monkeypatch of `np.einsum`, for the experiment only). The old
kernels (`sampling_derivatives`) are deleted in P20, so these scripts run only before it.

Findings (numpy, 8-core laptop, quiet machine; "fwd" = the tangent forward Jp; ratios new/old):

| (d, N, r, W, order) | apply fwd / transpose | fixed order | probe fwd / transpose | fixed order |
|---|---|---|---|---|
| (4, 12, 5, 60, 4) | 2.4 / 1.8 | 2.2 / 1.8 | 2.3 / 1.6 | -- |
| (4, 50, 40, 2000, 4) | 4.1 / 1.0 | 0.33 / 0.34 | 2.4 / 1.4 | 0.34 / 0.41 |
| (4, 30, 20, 2000, 8) | 6.3 / 1.4 | 0.42 / 0.34 | 2.9 / 1.6 | 0.47 / 0.37 |
| (6, 100, 30, 4000, 4) | 3.9 / 1.1 | 0.36 / 0.41 | -- | -- |

- No compilation (eager numpy). At the example's size the gap is per-call Python overhead (numpy's Python path
  machinery for multi-operand einsums, `jet_contract`'s bookkeeping); constant per call.
- At scale the forwards' gap is the order inside a jet step, `cda,aib,di->cdb`: `_pairwise_path` contracts the carried
  jet with the core first (the rank x mode x rank work once per jet slot). Contracting the cheapest pair first makes
  the engine 2-3x FASTER than the old kernels, transposes included, with the same values (<= 2e-15).
- numpy's own `optimize='greedy'` made it worse (5-8x): the order must be chosen explicitly.
- jax not measured.
