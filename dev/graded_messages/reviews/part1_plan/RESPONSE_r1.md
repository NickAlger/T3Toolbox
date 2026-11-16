# Response to review r1 of `PART1_PLAN.md`

Writer's response, 2026-11-16. The plan is now draft r2. Every finding was checked against the code (or against
T3Polynomial, read only) before acting. All 20 are accepted; none is rejected. Slice numbers below are r2's.

| # | Verdict | Where it is resolved in r2 |
| --- | --- | --- |
| 1 | Accepted. Verified: `t3polynomial/fitting.py:384-385` and `:597-598` call `X.probe_derivatives`. | Section 9: two run-time replacements (`_newton_fit`, and `probe_derivatives` given back through `taylor_to_derivatives(x.probe(...))`), the `stats` keys checked; section 10 item 3 runs the benchmark with both. |
| 2 | Accepted. Verified: `t3toolbox/fitting.py:339` (`_blm` builds the old backend model), `:591`; `tests/test_jax_absent.py:71,80`; `tests/test_aligned_padded_frame_frontend.py:152`. | The frontend local model moves in the switch (P16, part 4). A private copy of the old model does not work, because the corewise frame changes in the same slice and the old kinds read four stores (P16 preamble). Its test list is completed. |
| 3 | Accepted. Verified: `tests/test_contractions_sharding.py:133-211` imports `sampling_derivatives`. | P11 pins the property on the engine (`g` and `Kp`, orders 0 and 3, ragged and uniform, `W` sharded under `shard_map`); P19 deletes the old class. Also in sections 6 and 7.6 and the table of 5.4. |
| 4 | Accepted. Verified: `backend/probing.py:1745-1755`. | Section 2.4: same convention as the transposes (a residual, no weight), today's argument order kept. The question goes to the maintainer as O-22. |
| 5 | Accepted. Verified by a search for `*_derivatives` callers; `docs/chunking.md` doctests call the frontend method. | P13 lists every caller. Its `chunk_size` cases in `tests/test_chunk_size_estimator.py` are removed there, and `docs/chunking.md` is removed there. P16 lists its tests and pages. Section 6 table corrected. |
| 6 | Accepted in part. Verified that `frame(x)` returning centers can land alone. | The geometry change is now its own slice (P15). The corewise-frame change stays in the switch. P16 says why the backend optimizers cannot land before the frontend: the names are the same and the frontend wrappers call them. A further split with transitional code is O-24. |
| 7 | Accepted. Verified against `docs/contributor/testing_strategy.md:84-99`. | Section 2.3 step 2 states the masks (the frame's `D` and `L` masks for `core_directions` and `Kp`). P10 and P11 use the full `_CONFIGS` matrix. P11 adds the TT-slack test and the mutation check; also section 7.2. |
| 8 | Accepted. Verified: `_level_newton_kwargs` returns `{}`, and `_newton_fit` passes no `chunk_size`. | Section 1 corrected. O-1 notes that the gate compares chunking-off runs, and that "The cost" in the decisions is not borne out. |
| 9 | Accepted. Verified: `common.get_backend` gives `jax_map` for uniform with jax; `sampling_derivatives.py:504`, `1259`, `1440`, `1504`, `1620` use `xmap`. | O-19 states the facts and the cost. Section 1 names the memory cost; the scope table, section 2.3 step 6 and section 7.2 point to O-19. |
| 10 | Accepted. | N6 and N7 follow `a_to_b` (`derivatives_to_taylor`, `taylor_to_derivatives`, `derivative_weight_to_taylor_weight`). N25 to N29 added (frontend keyword, the modules, the diagram script). The history removed from N12. |
| 11 | Accepted. | Section 2.4: house-style sketches for `t3_probe`, `tv_probe_transpose`, `t3_probe_corewise_hessian`, `dual_jet_to_jet`, the four derivative-data functions, the five preparation functions, `point_objective`. Section 2.5: `probe`, `probe_transpose`, the frontend `newton_cg`. |
| 12 | Accepted. | O-20; the sketch no longer says "raises"; P3 tests the round trip for nonzero weights only. |
| 13 | Accepted. Verified the cited pages name old kernels or the old fitting API. | Section 8 lists the seven pages. The `__init__.py` re-exports go in P16. The diagrams are O-23 with the proposal stated, and the arrow rule (9.5) is in section 8. |
| 14 | Accepted. Verified: `optimizer_display.py` stores `train_err`/`val_err` per row. | P16 part 3: the stored record in full shape, and the `eps |f|` fallback, each with a test; section 6, "Fitting" row. |
| 15 | Accepted. | Section 5.1: the bridge compares only with code never rerouted (`sampling_derivatives`, the old kernels, the old kinds). The reading of "used once" is O-25. |
| 16 | Accepted. Verified the cycle `fv_operations -> t3_linalg -> ranks -> sharing -> fv_operations`, and `max(1, ...)` at `fv_operations.py:241-253`. | P4: the route through `t3_operations` and `t3_orthogonalization`, the formula kept, and a test on a shape whose dense tensor cannot be allocated. |
| 17 | Accepted. Verified `contractions._solve_group_lengths` raises on an unknown `len_<G>`. | Section 2.4 (after `jet_contract`) and P1. |
| 18 | Accepted. | Section 10: the grep covers `t3toolbox/`, `tests/`, `docs/`, `examples/`, and excludes `dev/`, `CHANGELOG.md` and the release notes. |
| 19 | Accepted. Verified the default `squash_tails=True` and that `geometry.frame` calls it. | Section 4, "Tails"; section 2.3 step 6. |
| 20 | Accepted, all items. | P8 (at the frame's base point); P14 (exact enumeration of subsets); section 7.5 and O-18 agree; section 5.1 (P4 named); section 2.2 (layout first, then the count); P16 part 2 (a draw-less oracle raises); P2 (tilde returned turned); O-21 (the minibatch scale); O-8 (`held` in the protocol); P5 split into P5 and P6; P0 paths. |

On the open points: O-4 now says that only the form is open; O-18 says that only the placement is open. Added:
O-19 (finding 9), O-20 (12), O-21 (20), O-22 (4), O-23 (13), O-24 (6), O-25 (15). The slices are now P0 to P19
(20, of which 19 change the library or the documentation).
