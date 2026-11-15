# The store rename plan — critic review, round 23 (draft 23, confirmation)

Reviewed: the draft-23 diff against `a178cb63`, which adds the session's rows V1–V5 under the V/H rule.

- **V1–V5 match the code.**
  - **V2.** `test_jet_recurrence._problem`'s `dict(…, dU=dU, dG=dG, …)` gives `V=V, H=H` by the NAME map. B7's
    `p['dG']`→`p['H']` covers the eight reads at `:104–114`, and `p['dU']` is never read. The label at `:153` is a
    transpose output, so `'dD'` is right.
  - **V3.** `_var` returns `(dU, dG)`, a variation pair built at the frame's widths (`_ND`, `_NU`), so `V, H` is right.
    **I confirm the `dG→H` completion.** It is a TT variation in the tangent-coordinate sense, which the V/H rule
    names `H`. The "`G`-family derivative names are untouched" clause covers derivatives such as `dG_tildes`, not
    variation cores.
  - **V5.** `H_i = <D_i, core_env_i>`, `V_i = <U_i, core_env_i>` at `tv_operations.py:400–401` matches what that
    algorithm reads off the environment.
- §3 now states the tangent-coordinate (`V`/`H`) versus derivative (`dD…`) distinction once, and the `U`-family row
  points to it.
- The freeness of `V`, `H`, `V_v`, `H_v` is recorded, and the four point-lettered locals stay held for Nick.

No MAJOR or MINOR finding.

VERDICT: AGREE
