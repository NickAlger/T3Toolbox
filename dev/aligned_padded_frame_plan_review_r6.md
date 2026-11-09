# Critic review r6 — `dev/aligned_padded_frame_impl_plan_2026-11-09.md` (round 5, CHANGES rows 5.1–5.3)

Scope: rows 5.1–5.3 and the text they touch (§0, §2.3, §11, T14(d), D-38). Everything r5 raised is resolved.

| r5 item | status |
|---|---|
| 1 MINOR: the level-2 reading | **Resolved.** "Hint-free and aligned with oversampling: 49 → 21–33" matches rows (b), (c), (d), (e) and (g) (21–33). Row (f), without oversampling, is stated separately (42, 29) in §0, §2.3 and §11. The noise is sourced (`zero_reference_cone_2026_11_02.md`, "Noise calibration", which exists). The seed spread (up to 13) is the stated yardstick. §0's basin bullet carries the 4-pass exception |
| 2 MINOR: zero passes and unread options | **Resolved.** The loops (literal `ALIGN_*` defaults) list the three places where a value is not read. The optimizer level (`None` defaults) raises on zero passes plus an explicit `*_oversampling` / `*_oversampled_passes` / `*_tol` / `*_alternate`, in `newton_cg`, `gradient_descent` and `padded_start`. `*_seed` is allowed, because it is read. The message names the fix. D-38 states it, and T14(d) tests it one option at a time |
| NITs 3–5 | **Resolved.** One "T3Polynomial adopts" bullet remains, and the baseline gains the no-oversampling row. The uniform iteration-0 frame is compared with the compacted start. The cone is "the design's second deliverable", with its evidence cited |

Optional wording, not blocking: in §2.3, "`max_passes = 0` is never an error" could read "is never an error in the
loops", to sit cleanly beside the optimizer-level rule two bullets above.

VERDICT: APPROVE
