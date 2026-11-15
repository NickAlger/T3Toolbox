# Critic review r4 — `dev/aligned_padded_frame_impl_plan_2026-11-09.md` (round 3, CHANGES rows 3.1–3.2)

Scope: rows 3.1–3.2 and the sections they touch (§3.5 members, §3.6, T14(h), T15, D-21, §5.7). Both items of r3 are
resolved.

| r3 item | status |
|---|---|
| 1 MINOR: one zero rule | **Resolved.** The rule appears with the same words in all three places: "structurally zero when ANY Tucker rank or ANY interior TT rank is 0". It is read off core shapes (ragged) or mask sums (uniform), per stack element, in both members (§3.5). The frontend converts such an `x_old` to the canonical rank-0 tensor first, keeping shape, stack, backend and uniform `N` (§3.6, D-21). Neither member uses the orthogonal representation for a zero `x_old`; the reason (non-empty stores at partially-zero ranks, and the uniform raise) is recorded. A stack mixing zero and nonzero elements is a structural error in the members, which serve the unstacked optimizers. T14(h) covers the canonical and the partial case on both layers, through the frontend and directly through the members, against the canonical result (≤ 1e-14), plus the mixed-stack rejection. T15 covers the frontend conversion |
| 2 NIT: §5.7 return type | **Resolved:** `typ.Tuple[typ.Union[t3.TuckerTensorTrain, ut3.UniformTuckerTensorTrain], dict]` (the module's `Point` alias) |

Nothing the changes touch introduces a new inconsistency:
- D-21, §3.5, §3.6 and T14(h) state the same rule;
- `None` stays only for the cone in the T3-free layer;
- `require_frame_structure`, the kernel and the stats records are unchanged.

The plan is ready for Nick's review. The items flagged to him (D-35's dropped frontend `padded_start`, D-33's
fourteen keywords, D-8 realized per layer) are decisions for him, not defects.

VERDICT: APPROVE
