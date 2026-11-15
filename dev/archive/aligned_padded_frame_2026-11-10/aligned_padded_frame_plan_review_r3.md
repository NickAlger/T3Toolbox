# Critic review r3 — `dev/aligned_padded_frame_impl_plan_2026-11-09.md` (round 2, CHANGES rows 2.1–2.10)

Scope: rows 2.1–2.10 and every section they touch (§2.3, §3.4–§3.7, §5.1, §5.5–§5.7, T6, T12, T14, D-18, D-21, D-25,
D-33–D-35, D-37), checked against my r2 checklist. **(checked)** = run at `a2e26ca8`.

## The r2 checklist

| r2 item | status |
|---|---|
| 1 MAJOR, D-37 | **Resolved.** `require_frame_structure(x_cores, frame)` is the third optional member, using integers and held masks only and "never orthogonalizes" (§3.5, §5.5, D-37). Ragged: core shapes vs `compute_orthogonal_representation_ranks` of `x`'s ranks, plus `C`, `N`. Uniform: `var_masks`, both boundary entries, `shape`, widths. `local_model(frame=)` and the frontend `start_frame` use it; D-25, §3.7 (`geometry → ranks`), T14(a) (per-family isolation plus a per-family mutation check) follow |
| 2 rank-0 `padded_from` | Resolved in principle (§3.5, §3.6, T14(h)). One inconsistency remains: finding 1 |
| 3 uniform no-op compacts | Resolved (§2.3, §3.6, T6, T12) |
| 4 kernel host data | Resolved. Bool masks precomputed; group tables as nested tuples of Python ints (static under `partition_static`); np index arrays at trace time; the module's own `lru_cache`d `_jitted_align_kernel()` |
| 5 T14(g) | Resolved: (g1) full rank with a gauge-rotated frame, FD plus matched quadratics; (g2) the padded `x0`: finite and decreasing |
| 6 `cone_*` with `g0norm_newton` | Resolved (§3.6, §5.6, T14(d), D-33); `'cone'` itself stays legal, as `'zero'` |
| 7 §5.7 | Resolved, in full. Checked against the code: the frontend `newton_cg` ends at `tr_radius0`, and `gradient_descent` ends in `**kwargs`, so "appended after `tr_radius0`" and "through `**kwargs`" are right. NIT 2 |
| 8 `padded_start` public | Resolved (`__all__`; doctest of the `least_squares_problem` route; D-35 corrected and flagged to Nick) |
| 9 T14(d) tolerance | Resolved (≤ 1e-12 relative) |
| NITs 10–15 | Resolved: expanded returns, `_frame_theta`, `ALIGN_*` named once (`padded_start` forwards only set options), T18 on module files, the ignored-`oversampled_passes` sentence, D-18's gloss |

## What remains

### 1. MINOR — §3.5 vs §3.6: two different "zero old point" rules

- The frontend detects the zero tensor by "**any** Tucker rank or interior TT rank 0" (§3.6). It then skips the
  reduction, and for `newton_cg` passes the data through unchanged (§3.6: "reaches the backend as rank-0 data").
- The uniform member recognizes it only when "**every** Tucker mask empty and every interior bond mask empty" (§3.5).
  The ragged member relies on `self.frame(x_old)`.
- A partially-zero-rank `x_old`, such as Tucker `(0, 2, 2)` with TT `(1, 0, 2, 1)` (the zero tensor, structurally),
  falls between the rules. **(checked)** Its `t3_orthogonal_representations` has non-empty stores: up `(2, 5)`,
  `(2, 6)`, left `(0, 2, 0)`, …. So the ragged member would pin arbitrary orthonormal rows as the "old block". The
  uniform member would not detect it and would call `ut3_orthogonal_representations` on unreduced zero-rank data.

**Fix (one rule in one place).** The frontend canonicalizes any structurally-zero `x_old` to the canonical rank-0
tensor (`zeros(shape, (0,)*d, (1,)+(0,)*(d−1)+(1,), stack_shape)`, same backend) before passing it on. Both members
then recognize the canonical form by the same "any rank 0" test and use the empty frame. The ragged member must not
rely on `self.frame` for this, because only the canonical rank-0 form gives empty stores. Add a partially-zero case
to T14(h).

### 2. NIT — §5.7: `-> typ.Tuple[Point, dict]`

`Point` is not a name in `optimizers.py`. The current signature returns
`Tuple[Union[TuckerTensorTrain, UniformTuckerTensorTrain], dict]`; write that, as the unchanged parameters do.

Nothing else remains. Once finding 1 is fixed (and the NIT if convenient), I expect to approve.

VERDICT: REVISE
