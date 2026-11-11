# The store rename plan — critic review, round 26 (draft 25, confirmation)

Reviewed: the draft-25 diff against `098c61bd`, which records R2's three approved rows and its seven hand edits.

**The three new rows, checked against `v2026.4.0`.**
- `tv_oblique_gauge_projection`'s `dG1 = tt_vars[ii]` is a TT variation, so `H1`.
- `_assemble_tt_variations_core` returns `(dG_tilde,)` of shape `(rLi, nUi, rR(i+1))` in old letters.
- `assemble_tt_variations` returns `dG_tildes` of shape `(…, rLi, nUi, rR(i+1))` in old letters.
- `_assemble_tt_variation_jets_dense` returns `dG_tildes` of shape `[W+]K+C+(rLi, nUi, rRi)` in old letters.

Each is `(rL, nD, rR)` in the new widths, so `H…` is right by the shape rule. Each image is free in its function
(`ast` on the tagged source).

**§4.6.** The record of R2's seven hand edits (four message/regex pairs, plus three Tucker-store words in
`uniform_padded_frames.py` with their tests) is a statement of what was done, with no new rule.

No MAJOR or MINOR finding.

VERDICT: AGREE
