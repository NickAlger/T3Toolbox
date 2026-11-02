# Judgment calls during the padded-frame / cone-reference implementation

*Kept by the orchestrating session (Nick's instruction, 2026-11-02): when an unforeseen issue arises during
implementation, the session decides and records here — what the issue was, what was decided, and why. Nick reviews
these at the review stops (after S3, S7, S10, S12) or whenever he likes. Plan:
`dev/padded_frame_cone_impl_plan_2026-11-02.md`; slice records: `dev/padded_frame_slices/S*_handoff.md`,
`S*_review.md`.*

| # | slice | issue | decision | why |
|---|---|---|---|---|
| J-1 | S1 | Commit attribution: the session (Fable 5.1) commits code written by an Opus 5.5 implementer agent; Nick's rule is "name the actual model in use". | Implementation commits carry BOTH trailers: `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` (orchestration, verification, commit) and `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` (implementation). | Honest attribution of who wrote what; neither line alone would be accurate. |
| J-2 | S1 | The plan (§2.3, §4.1) and the first docstring described a rank-deficient hint's trailing directions as "the Householder completion"; they are the SVD's noise-level singular vectors re-orthogonalized against the pins, so their orientation is roundoff- and backend-dependent (reviewer's probe: projector distance 1.4 when content inside span(P) changes; output still valid at ≤ 1.5e-15). | Keep the algorithm (no threshold, always a valid completion); correct the wording in the plan and the docstring; require hints with ≥ k directions per slot in any projector-equality test (S3–S5). | The behaviour is exactly the plan's R5/E1 exception; only the description was wrong. A content-independent completion would need a value-based choice, which the plan forbids. |
