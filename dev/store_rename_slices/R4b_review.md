# R4b review — the sampling kernels' prose and the shift symbols (plan draft 26, §7 row R4b; §3.2, §3.3, §4.4, §4.10, §4.12, §4A, §1.2a, §5)

Reviewer, 2026-11-13. Reviewed: the working tree on `ebf563a9` (HEAD; its code tree is R4a's `e3085085`), which holds
R4b uncommitted (11 library/test files, all prose; `dev/store_rename_check.py`; `diagrams_inventory.md`; a snapshot of
the diff, 3,697 lines, was taken before any run), with `R4b_handoff.md`, `R4b_name_hunks.tsv` (header only) and
`R4b_realignment.diff`. Every check below is my own, run independently; scripts and outputs are in the session
scratchpad under `rev4b/`, not in the repo. Heavy jobs ran one after another.

## Findings

### MINOR 1 — `probing._assemble_tangent_z_step` (`:535, 541–542`): the prose still says `B`, `dB` where the code says `D`, `V`

```
        x: typ.Tuple[NDArray, NDArray, NDArray, NDArray],   # (B, dB, eta, deta)
    D, V, eta, deta = x
    # passively over the C-only frame core B, so no split is needed); eta is W+C and dB is the
    # variation core K+C, so term2 needs len(C) -- recovered here from the C-only core B.
```

R2 renamed the NAMEs by §1.2a (`B, dB → D, V`: `assemble_tangent_z` takes the frame's down store `(nDi,Ni)` and the
Tucker variation `(nUi,Ni)`; both library callers, `probing.py:623, 758`, pass `down_tucker_cores`), but the three
comment lines were not converted; the twin `sampling_derivatives._tangent_z_jets_step` was (`(D, V, eta_jet,
deta_jet)`, "the C-only D", "V is the variation core"). No gate sees a bare `B`/`dB`. Fix: `(D, V, eta, deta)`, "the
C-only frame core D", "V is the variation core K+C", "the C-only core D" — as the twin reads.

### MINOR 2 — the signature labels pair `V.` with `dG.` (§3.3: "Pairs stay symmetric … `(V, H)`")

`sampling_derivatives.py:899–900, 932–933, 1076–1077, 1106–1107, 1160–1161, 1190–1191` now read
`var_tucker_cores V.` over `var_tt_cores dG.`, and `:672` "the variation cores V, dG"; `:733, 763, 829, 1052` label the
`var_tt_cores` parameter `# dG.` beside `elm_shape=K+C+(rLi,nDi,rR(i+1))`, a TT variation's shape. None of these
functions has a local `dG` to mirror. It resolves with ruling R7 below (`dG → H` in the forward kernels, the prose
following); listed separately because the pair breaks §3.3 even where no NAME is involved.

## The session's rulings, verified site by site (the implementer applies them in its revision)

### R1 — the 104 `P`/`Q` → `L`/`R` prose lines: all stand

My scan pairs every changed line HEAD → tree and lists each word token that went `P…`/`Q…` → `L…`/`R…`: **169 lines**
in R4b's files (the tier-D tuples and pqu rows plus Appendix C), every Appendix C line among them (104 of 104). I read
all 169 with their enclosing function. Every one names a frame's left or right core, by the function's own code:

- the tuple/carry comments mirror locals that tier C made `L`/`R` (`_mu_step (L, xi): C+(rLi,nDi,rL(i+1))`,
  `_sigma_hat_step (R, xi): C+(rRi,nDi,rR(i+1))`, `_tau_tilde_step`, `_deta_step`, `_deta_jets_order_step`,
  `_adj_src_order_step`, `_adj_tangent_src_order_step`, the scanned-step tuples, `_sigma_hat_tangent_jets_step`);
- the signature labels sit on `left_tt_cores[_dot]` / `right_tt_cores[_dot]` (A1-free keywords that name the store);
- the formulas (`tau~ L xi + mu L deta~`, `sigma R nu + mu dG nu + mu L tau`, `prop(carry; L xi)`, `d(edge L)`,
  `d(R xi) = R_dot xi + R xi_dot`, `src^(s) = … (edge^(r) L) …`) are the products the bodies contract with those
  locals (`contract('WCa,Caib->WCib', ew, L)` "edge^(r) L", `contract('WKCa,Caib->WKCib', w_r * sig_r, R)` "sigma R");
- "via L / via R" are `compute_mu(left_tt_cores, …)` / the `right_tt_cores` sweeps; `_curvature_oracles.py:272–273`
  describe `adjoint_sweep_tangent_trs(L, Ld, …)`.

**None** is `P⁺`, the P-block, the perturbation `P`/`pp` ("perturbation vectors P", "contracting ``P`` (order 1)",
"X + s P" all kept — I grepped every remaining bare `P`/`Q`/`O` in the eleven files: only the perturbation `P`, author
initials, `O(…)`, and `_curvature_oracles`' factored products `Q`, `Qd` (§1.5 item 14) remain), nor QR's `Q`. Nothing
to revert. (Register the 104 as §3.4 text rows, as R4a's MINOR 1 was.)

### R2 — item 12's generic `n`: `:473` takes `n`, and two T3-only step comments with it

`compute_eta_jets(tt_cores, …)` (`sampling_derivatives.py:472–476`) holds **either**: `t3_probe_derivatives:200` passes a
T3's own `tt_cores`, `_probe_frame_sweep_jets_from_xi:597` the frame's `up_tt_cores`. By item 12's own rule ("a T3's
own cores, or either, takes the generic `n`") and "one size, one label", it is `n`, as its siblings
`compute_mu_jets`/`compute_nu_jets` (`:388, 416`, also "either") already are. Apply:

- `:473` `tt_cores … C+(rLi,ni,rR(i+1))`; `:476` `eta_jets … (order+1,)+W+C+(ni,)`;
- the map bodies: `_eta_jets_order_step` carry (`:434`) `(order+1,)+W+C+(ni,) ; … ; C+(rLi,ni,rR(i+1))`,
  `_eta_jets_core` return (`:455`) `(order+1,)+W+C+(ni,)`.

The probe kernel `probing.compute_eta` keeps `nU`: its keyword is A1's `up_tt_cores` (lettered by its keyword, K13).
The same ruling covers the two T3-only step comments the handoff reported (§6): `apply._t3_apply_step` (`:39`) and
`entries._t3_entries_step` (`:44`) hold `(v|ind, B, G)`, a T3's own cores (callers `t3_apply`, `t3_entries` only), so
`C+(ni,Ni) ; C+(rLi,ni,rL(i+1))` (the `rL(i+1)` stays, D-3). My scan of every `t3_*`/`_t3_*`/`ut3_*`/corewise function
in R4b's files finds no other `nD`/`nU` on a T3's own core. The other item-12 lines check out: `:388, 416, 518, 995,
3230–3240` `n` (applied); `probing.py:550` and `sampling_derivatives.py:872` `nD` (both callers pass
`down_tucker_cores`); the probe two-hole assemblies' `n` (`assemble_tucker_tangent_variations`,
`assemble_tt_tangent_variations`, their steps and jet twins) — core-shaped corewise Hessian blocks, so `n` is right and
K17-shape-point needed it.

### R3 — prose and code carry one letter: two NAME rows `D → B`, one prose fix

The object decides:

- `probing._assemble_z_step` (`:309–314`): `eta, D = x` / `# (eta, D)`. `assemble_z`'s `tucker_cores` is `C+(ni,Ni)`
  (generic) and its only caller is `t3_probe:179` with the point's own `tucker_cores` → **`B`** (NAME row, 2 tokens;
  comment `(eta, B)`).
- `sampling_derivatives._z_jets_step` (`:506–512`): `eta_jet, D = data` / `# (eta_jet, D)`. `assemble_z_jets`
  (`C+(ni,Ni)`), only caller `t3_probe_derivatives:202` with the point's own cores; its docstring already says
  `z_i^(t) = B_i eta_i^(t)` → **`B`** (2 tokens; comment `(eta_jet, B)`).
- `t3_probe_derivatives:191–192` "`B_i x_i`", "`B_i p_i`": the call passes the point's `tucker_cores` → `B` stands.
- Kept `D`, by their A1 keywords (§1.2a "kernels lettered by their keywords"): `_xi_step` (`compute_xi(down_tucker_cores…)`),
  `_deta_tilde_step`, `_deta_tilde_jets_step`, `_deta_tilde_tangent[_jets]_step`, `_tangent_z_jets_step`,
  `_assemble_tangent_z_step` (whose prose is MINOR 1's).

`B` is free in both functions (ast). My per-function scan of prose letters against code locals in the eleven files finds
no other mismatch beyond MINOR 1 and R7 (`Ddot` prose beside `Dd` code is the house `_dot`/`d` spelling, as before).

### R4 — `H-tilde_k` / `V-tilde_k`: fine

A-0465 spells `dxi-hat_k`, `mu-hat_{k-1}`, `sigma-hat_k`, `xi-hat_k`; `-tilde_k` is that style, keeps the `=` column
(20, 1-based) and matches `entries.py:152`'s ``V-tilde``. (NIT 1: the inventory entry's R1 roles line still says
"→ dD-tilde_k".)

### R5 — verified against the code

- **`DF → dF`** (`_curvature_oracles.py:10`): the directional derivative of `F` along `theta_hat`, the same object as
  the next line's `dPhi(theta)[theta_hat]` (D-11). Right; register it as a tier-D/E row (the plan's 15 lines miss it).
- **`delta_U_tilde → V_tilde`, `delta_G_tilde → H_tilde`** (`probing.assemble_tucker_variations` /
  `assemble_tt_variations`): T4S's `δŨ`, `δG̃`; the functions return `V_tildes` `(...,nUi,Ni)` / `H_tildes`
  `(...,rLi,nDi,rR(i+1))` (R2's names) — the shape rule's `V`/`H`, as §4.8 converts the notes. Right.
- **`'apply transpose H (…)'`** (`test_jet_recurrence.py:154`): it labels `got[1]`, the TT block, whose dict key B7 made
  `'H'`; its sibling label is B7's `'apply transpose V'`. Right (a string, no NAME).
- **P-0640/P-1215's bond letters** `L.shape = (rL, nD, rL)`, `R.shape = (rR, nD, rR)` (tag: `(rL, nU, rR)` for both):
  the frame's own field comments wrote `(rLi, nUi, rL(i+1))` / `(rRi, nUi, rR(i+1))` already at the tag
  (`frame_variations_format.py:158–159`, `uniform_frame_variations_format.py:112–113`), as do the kernels'
  (`compute_mu`'s `C+(rLi,nDi,rL(i+1))`, `compute_nu`'s `C+(rRi,nDi,rR(i+1))`). `L_iax L_iay = I_xy` sums the left bond
  and the leg and leaves the right bond free (`einsum('...iaj,...iak->...jk', L, L)`, `fv_operations.py:222`);
  `R_xaj R_yaj = I_xy` sums the leg and the right bond (`'...iaj,...kaj->...ik'`, `:225`). The correction aligns the
  backend block with the frame's own contract; D-3 keeps the symbols, not a wrong bond. Right; Nick sees it at the stop.

### R6 — K8 ignoring the NAME `Ud` in `sampling_derivatives.py`: justified

Every remaining `Ud` NAME (`:2703–2729`, `:3172–3193`) is the dotted up core: `(dxi~_dot, U, U_dot): … ; C+(...) ;
K+C+(...)`, contracted as `'WCa,KCaib->WKCib'` / `'KCaib,WCb->WKCai'` — TT-shaped, `(rL, nU, rR)` with `K`, the same
shape as `U`. Tier C maps old `Ud → Dd` and `Od → Ud` simultaneously in this file, so any `Ud` NAME left is an image of
`Od`, never a U-family leftover, and K15 pins them. The carve is code-lines-only, one file; prose `Ud` still hits
(there is none). Justified narrowing.

### R7 — the reported-not-renamed NAMEs, by the shape rule

**`dG` → `H`: the forward kernels' TT variation.** Each holds an element of `var_tt_cores`, whose signature says
`(rLi,nDi,rR(i+1))` (`probing.compute_sigma:401`, `compute_tau:438`, `compute_deta:489`; `compute_sigma_jets:733`,
`compute_tau_jets:763`, `compute_deta_jets:829`, `_apply_derivatives_from_jets:1052`) — an `(rL, nD, rR)` variation:

| site | tokens | deciding line |
|---|---|---|
| `probing._sigma_step` (param + use) | `dG` ×2 | `t2 = contract('WCa,KCaib,WCi->WKCb', mu, dG, xi, …)`, from `compute_sigma`'s `var_tt_cores` |
| `probing._sigma_sweep_step` | `dG` ×2 | `R, U, dG, xi, dxi, mu = x` (xs of `compute_sigma`) |
| `probing._deta_step` | `dG` ×2 | `L, R, dG, … = x` (xs of `compute_deta`) |
| `apply._apply_from_xis_step` | `dG` ×2 | xs of `_apply_from_xis(…, var_tt_cores)` |
| `sampling_derivatives._sigma_banded_step` | `dG` ×2, `dGxi → Hxi` ×3 | beside `Rxi`, `Udxi` |
| `sampling_derivatives._sigma_jets_step` | `dG` ×2 | xs of `compute_sigma_jets` |
| `sampling_derivatives._deta_jets_order_step`, `_deta_jets_core` | `dG` ×3, ×3 | xs of `compute_deta_jets` |
| `tests/_jet_trs_twins.py` `_sigma_jet_step`, `_sigma_sweep_trs`, `compute_deta_jets_trs` | `dG` ×2 each | twins of the above (one letter across twins) |

`H` and `Hxi` are free in each function (ast); every call is positional (no `dG=` anywhere). At the corewise frame these
kernels receive `Gdot` in that slot; the slot is still the variation's (the kernels are lettered by their keyword
`var_tt_cores`, §1.2a). **The prose follows** (MINOR 2 and R3's one-letter rule): the tuple comments `(R, U, H, …)`
(`probing:392, 474`; `sd:723, 782, 803, 1038`; `apply:436`), the labels `# H.` / `var_tt_cores H.` (`sd:733, 763, 829,
900, 933, 1052, 1077, 1107, 1161, 1191`), the formulas `mu H xi`, `sigma R nu + mu H nu + mu L tau`, `H_i(xi_i)`, "the
variation core H", "(sigma / H / tau)", "U and H reversed", `# mu H` (`probing:380, 480, 1504, 1578`; `sd:672, 697, 699,
743, 771, 785, 794, 839, 843`; twins `:79, 99, 112`).

**`dG` stays** where it is the corewise Hessian's TT block or a raw direction of the point's own `G`, core-shaped
`(rLi,ni,rR(i+1))` (§3.1): `probing.assemble_tt_tangent_variations` and its step, `t3_probe_corewise_hessian_from_sweep`;
`sd._assemble_tt_tangent_variation_jets_dense`, `_tt_tangent_variation_jets_step`, `assemble_tt_tangent_variation_jets`,
`t3_probe_corewise_derivatives_hessian_from_sweep`, `_apply_tangent_assemble_step`, `_apply_derivatives_hessian_from_jets`;
`apply._apply_corewise_hessian_adjoint`; `_curvature_oracles.vertical_direction`, `apply_derivatives_tangent_assembly_trs`;
the `st['dG']` keys; `ut3_sampling`'s `(dB, dG)`.

**T3-own `D` → `B`:** R3's two rows above.

**`rsh`** (`_curvature_oracles.sigma_hat_tangent_jets_trs:483–490`): `rsh = sigma_hats[::-1]`, beside `rG = rv(G)`,
`rxi = xi[::-1]` — "reversed σ̂", with `sh` the σ̂ abbreviation that tier C's row removed everywhere else (`sh →
sig_hat`). No shift operator appears in that function or its docstring, so nothing collides in place, and §1.5 item
14 keeps the compounds `psh`/`pshd`. But it is now the only `sh`-means-σ̂ left in the tree, in a jet-oracle file whose
twinned kernel uses `sh` for the shift ("the reverse shift" `shᵀ` is exactly what `rsh` could be misread as).
**Recommend `rsh → rsig_hat`** (3 tokens, free); it is outside every row, so the session decides (or Nick at the stop).
Keeping it is harmless.

## For the session (items outside the rows)

- **S1 — `entries.py:344`** "Up-index edge variables by slicing Tucker-core fibers" (`_entry_xis`): a travel word
  §3.3 targets; R1's grep was case-sensitive, so the row list misses it (a K9 keep today). Its siblings in the same file
  read "upward" (`:124, 151, 314`). Recommend a row: "Upward edge variables …".
- **S2 — `test_curvature.py:8–12`** (module docstring): the left store `L` (`<L_i, H_i>`, `t3_corewise_frame(D, L)`)
  and the lift `L` (`L v`, `<L v, y> == <v, L^T y>`) share one docstring, line 10 in one expression — §3.1's collision
  ("an existing `(U, L)`"), pre-existing at the tag and not on §3.1's site list. Recommend §3.1's template on line 10:
  "``tv_to_dense(t3_corewise_frame(D, L), theta_hat) == tv_to_dense(frame, v)`` for ``theta_hat`` the lift of ``v``";
  line 12 may stay (no store letter on it).
- **S3 — `apply.py:168–169`** `xi-hat_i = D_i^T w_i`, `delta-xi_i = V_i^T w_i` (tag: `U_i^T`, `dU_i^T`): T4S's layout.
  In the code's, `compute_xi` is `contract('Cio,Wo->WCi', D, w)`: `D_i w_i`, `V_i w_i` (`(nD,N)(N)`), as
  `sampling_derivatives.py:613` (`D_i x_i`) and `:191` (`B_i x_i`) write it; R4a's S3 made the same correction
  (`D V^T`). The only `^T w` left in the code tree. Two comment lines.
- **Plan text (writer):** R1's 104 rows; `DF → dF`; item 12's `:473` → `n` and the two T3-only step lines; R3's and R7's
  NAME rows; the K8 narrowing; §1.2a's `V~_k`/`H~_k` sketch → `-tilde_k`; §3.3's `-ward` list drops `manifold.py:949,
  1172`, `test_contractions_interpreter.py:227, 335` (R4a's).

## NITs

- **N1.** `diagrams_inventory.md`, A-0465's roles line (R1's) still reads "dU-tilde_k (§1.2a: … D forms) →
  dD-tilde_k over (nU, N)"; the new text says `V-tilde_k`. A record only; worth one edit so the entry agrees with itself.
- **N2.** `probing.py:1566–1567` are 121/123 characters, the tag's own lengths restored by the type-column fix; within
  reason.

## (1) The diagrams and equations

**All 7 R4b blocks read against the tag and the tree** (Appendix A verbatim = the file: K12).

- **A-0465**, against the ragged loop below it: `dxi-hat_k = mu-hat_{k-1} . U_k . sigma-hat_k` is
  `contract('WCa,Caib,WKCb->WKCi', mu, U, sig_hat)`: `mu` `(rL_k)`, `U_k` `(rL_k, nU_k, rR_{k+1})`, `sig_hat`
  `(rR_{k+1})`, both bonds summed → `(nU_k)`, "over the up mode nU". `H-tilde_k = mu-hat (x) xi-hat_k (x) sigma-hat_k`
  is `'WCa,WCi,WKCb->KCaib'`: `xi-hat_k = D_k w_k` is `(nD_k)` → `(rL, nD, rR)`, a TT variation. `V-tilde_k =
  dxi-hat_k (x) w_k` is `'Wo,WKCa->KCao'` → `(nU, N)`, a Tucker variation. Sizes agree across the three lines (`nU` in
  through `U_k`, out through `V-tilde`; `nD` in through `xi-hat`, out through `H-tilde`). The `=` (col 20) and `#`
  (col 71) columns now agree on all three lines.
- **P-0640 / P-1215** (identical): `D_xo D_yo = I_xy`, `D` `(nD, N)`, `o = N` summed → `(nD, nD)`, down-orthogonal
  (`einsum('...io,...jo->...ij', D, D)`, `fv_operations.py:218`); `U_ixj U_iyj`, `U` `(rL, nU, rR)`, both bonds summed
  → `(nU, nU)`, up-orthogonal (`'...iaj,...ibj->...ab'`, `:219`); `L_iax L_iay`, `R_xaj R_yaj` as R5 above.
- **P-0646 / P-1221**: `down_tucker_supercore (d, nD, N)` down-orthogonal, `up_tt_supercore (d, rL, nU, rR)` up, the
  left/right supercores `(d, rL, nD, rL)` / `(d, rR, nD, rR)` — UFVF-0110's shapes; the tag's backwards adjectives
  (§4.3, `probing.py:646–647, 1221–1222`) now agree with their store words.
- **P-0651 / P-1226**: `(down, up, left, right) = (D, U, L, R)`, the physical order in the new words.

**The shift notation, re-derived against `_shift_down`/`_shift_up`** (`sampling_derivatives.py:315–335`): `_shift_down`
returns `t * [0, x^(0), …, x^(order-1)]`, so `(sh x)^(t) = t x^(t-1)`, zero at `t = 0`; `_shift_up` returns
`(s+1) * [x^(1), …, x^(order), 0]`, so `(shᵀ x)^(s) = (s+1) x^(s+1)`, zero at `s = order`. Adjoint, truncation
included: `Σ_t t x^(t-1) y^(t) = Σ_s x^(s) (s+1) y^(s+1)` over `s ≤ order-1`, and `(shᵀ y)^(order) = 0`. Iterated:
`(sh^r x)^(t) = t(t-1)…(t-r+1) x^(t-r) = r! C(t,r) x^(t-r)`; `((shᵀ)^r x)^(s) = (s+1)…(s+r) x^(s+r) = r! C(s+r,r)
x^(s+r)`. The pairs: forward `[x^(t), t x^(t-1)]·[b0, b1] = Σ_{s<2} C(t,t-s) x^(t-s) b^(s)`; reverse `[x^(s), (s+1)
x^(s+1)]·[b0, b1] = Σ_{u<2} C(s+u,u) x^(s+u) b^(u)`. The scans: `eta = Σ_r (mu^(r)/r! . G) . sh^r nu` gives `Σ_r
C(t,r) mu^(r) G nu^(t-r)`, the docstring's convolution; `src^(s) = Σ_r C(r+s,r) (edge^(r) L) . deta~^(r+s)` from the
carried `(shᵀ)^r deta~` and the `1/r!` xs; `dxi~^(u) = Σ_j (mu^(j) U) . [(shᵀ)^u sigma~]^(j) + [(shᵀ)^u tau~]^(j) .
(U nu^(j))` with `[(shᵀ)^1 σ̃]^(j) = (j+1) σ̃^(j+1)` (the `C(j+1, j) = j+1` the docstring names); the TT assemblies'
`[(shᵀ)^k sigma]^(r)`, `k ∈ {0,1}`. All agree with sample edit 7 (`Λ` → `sh`, draft 26). `_shift_up`, `_shift_down`,
`_shift_up_pair`, `_shift_down_pair` keep their names; no `U^`/`D^` shift, `DPhi`, `DF(` is left anywhere in
`t3toolbox/`, `tests/`, `examples/` (K7 pattern grep: 0), and no code variable holds the operator.

## (2) The prose

- **Frame letters**: every `O`/`P`/`Q`/frame `U` in the eleven files is converted (my grep of the remaining bare
  letters: only the perturbation `P`, author initials, `O(…)`, the factored products). The corewise sentences
  (`apply.py:320–322`, `probing.py:1426–1429`, `sd.py:2384–2389, 2408–2411`) follow Nick's template ("the Section 6.3
  substitution `B -> D`, `G -> U, L, R`: … at the frame `(B, G, G, G)`; the manifold operation requires an orthogonal
  frame, the corewise one works for any `B`, `G`"); the TT-slot-only ones write `(U,L,R)->G` (`ut3_sampling.py:111,
  229`); `U = L = R = G`, `R = U = G`, `U = G`; `theta = (B, G)` beside a point given by stores `theta = (D, L)` with
  corewise frame `(D, L, L, L)` (§1.2a "where the two letters meet"); `(Bdot, Gdot)`, `xidot = Bdot w`.
- **Store words in the new meaning, checked against the code**: `_TUCKER_SLACK`'s "frame up ranks (2, 4, 2) < down
  (4, 4, 4)" — I built the frame: `down_ranks (4, 4, 4)`, `up_ranks (2, 4, 2)`; `_SH_*`'s "up rank 2 < 3/4" (mode 0's
  up width `min(n, rL·rR) = 2`); `blocksum`'s slot comments against `tv_to_t3`'s concatenations (last core `[[R ; H],
  [0 ; U]]`, middle `R` top-first-first, `L` top-second-second, `U` bottom-second-first); `(down, left)` masks;
  `_mask_once`/`_gauge_masks_over_Knew`'s `(down, up, …)` = `ufv_variation_masks`'s order; the chunk estimators'
  `nU = n  # up-core mode`.
- **`-ward`**: every R1 row applied; a case-insensitive grep of `(var|adjoint)-(up|down)`, `(up|down) (jets|eta|sweep|
  lift)`, `up-index` over `t3toolbox/`, `tests/`, `examples/` leaves only `entries.py:344` (S1). The "up-shift(ed)" keeps
  name `shᵀ`, as `_shift_up` does.

## (3) The realignment

`R4b_realignment.diff`: 32 −/+ pairs, every one `split()`-equal with the same indentation. My adjacent-pair scan (pairs
of consecutive lines that shared a trailing-`#` or `name:  type` column, matched tag → tree by difflib) finds **0**
ragged pairs in the eleven files against both the tag and HEAD — and **49** when run on HEAD's own text, so it sees the
drift R2 left. Lines grown past 120: only `probing.py:1566–1567` (the tag's lengths, N2) and three pre-existing long
signature comments that the shift letters lengthen by 1–8 (`sd:434, 1283, 1379`; sample edit 7's approved form).

## (4) Runs (one after another, `t3toolbox` env unless stated)

First pass (the tree as the handoff left it; `git diff` checksum identical before and after the suite):

| run | result |
|---|---|
| battery (`run --b9 battery_ref/b9`, then `compare battery_ref <mine>`) | **PASS (bitwise)**, 2,190 arrays, 856 texts, 0 errors, `NEW=True`, `LETTERS_NEW=True` |
| gates (`--stage R4b`) | K0, K2, K3, K6, K8b, K9 (+converse, registered), K13, K14, K15 (0 hunks), K15-cross, K17 (+ the three shape checks) **PASS**; K1 2, K4 16, K5 7, K7 49, K8 395 — **every hit in `docs/`, the notes or `CLAUDE.md`** (my per-file count: 0 in `t3toolbox/`, `tests/`, `examples/`), R5's; K13-generic INFO 3 (R2 above); K16 REPORT 20 (R5) |
| diagram check | K10/K10b PASS (33 in place, 20 re-laid out), K11 PASS (252 fences), K12 54/54 code-tree blocks filled |
| doctest outputs (static, HEAD → tree, every changed `.py`) | 297 examples, **0 expected outputs changed**, 1 source changed (`probing.py:1294`'s trailing comment) |
| suite `-n 3` | **1,347 passed, 4 skipped, 48,724 subtests passed, 0 failed** (21 min 17 s; the one warning is the intended adam one) |
| `pytest --doctest-modules t3toolbox` | **241 passed** |
| floor env (`tttt`: numpy 1.22.4, jax 0.4.30), the three touched test modules together | **1 failed**, 158 passed, 2 skipped — twice: `test_jet_recurrence.py::TestForwardJets` 'apply (Euclidean)' core 0, rel err **1.22e-14 > TOL 1e-14**. Not R4b's: the NAME stream and every number are unchanged (K15, battery); the same three modules **pass** with `OPENBLAS_NUM_THREADS=1`, and `test_jet_recurrence` + `test_curvature` alone pass twice on the tree and twice on HEAD (HEAD's three-module run passed once). A standalone floor-env script gives that check's max rel err as 4.7e-16 … 1.3e-15 depending only on what was imported first, on either tree. A pre-existing roundoff-level tolerance that multithreaded BLAS can cross (CLAUDE.md: "1e-14 cross-path tolerances are suspect"); for the session, since CI's floor leg runs it |

Not yet run on purpose: `sphinx -W` and the docs-page doctests — they go on the revision (the rulings change docstrings
again), one heavy job at a time.

## The session's rulings on the items outside the rows (2026-11-13, relayed to the implementer)

All four as recommended, each a listed row: `rsh → rsig_hat`; `entries.py:344` "Up-index" → "Upward";
`test_curvature.py:10` the lift in words (§3.1 template); `apply.py:168–169` `D_i w_i`, `V_i w_i`. And a listed
test-only change outside the token map: `test_jet_recurrence.py`'s `TOL` 1e-14 → 1e-13 (the arc's standing rule on
1e-14 cross-path tolerances), with the floor-env observation as its reason.

## Verdict (first pass)

R4b's prose, shift notation, `-ward` words, blocks and realignment are right, and every gate and run passes except the
floor-env tolerance above (not R4b's). The revision applies the session's rulings as verified: R2 (`compute_eta_jets`
and the two T3-only steps take `n`), R3 (`D → B` in `_assemble_z_step`, `_z_jets_step`), R7 (`dG → H` in the forward
kernels, the prose with it), MINOR 1, and N1; S1–S3 and `rsh` wait for the session.

VERDICT: REVISE

## Revision 1 (the rulings applied by the implementer; reviewed 2026-11-13)

Read as a diff against my first-pass snapshot (the first pass re-applied to HEAD, then `diff -u` against the tree), and
against the handoff's §0 and Appendix B. Every item is applied as ruled, and nothing else changed:

- **R2.** `compute_eta_jets` (`:473, 476`), `_eta_jets_order_step`'s carry (`:434`), `_eta_jets_core`'s return (`:450`):
  `ni`; `apply._t3_apply_step:39`, `entries._t3_entries_step:44`: `C+(ni,Ni) ; C+(rLi,ni,rL(i+1))`. K13-generic is
  down to 2 (`probing.py:550`, `sd:872`, both `nD` by their callers).
- **R3.** `_assemble_z_step` and `_z_jets_step`: `eta, B = x` / `eta_jet, B = data`, the contraction on `B`, comments
  `(eta, B)`, `(eta_jet, B)`.
- **R7.** `dG → H` in the eleven functions (`dGxi → Hxi` beside `Rxi`, `Udxi`), and `test_jet_recurrence.py:173, 179`
  (the test local fed to `compute_deta_jets`' `var_tt_cores`, the object `_problem` calls `H`): **36 NAME hunks, all
  listed** (`R4b_name_hunks.tsv`; K15 36/36, K15-cross PASS with `dGxi, Hxi, rsh, rsig_hat`). The prose follows at
  every site I listed; `H.` takes two spaces so `len=d` keeps its column, as `R.`/`V.` do. My grep of `dG` in the
  eleven files now finds only the corewise Hessian functions (`apply._apply_corewise_hessian_adjoint`,
  `probing.assemble_tt_tangent_variations` and the corewise two-hole, `sd`'s corewise assemblies and Hessians,
  `vertical_direction`, `apply_derivatives_tangent_assembly_trs`) and the `st['dG']` keys: right.
- **MINOR 1** `(D, V, eta, deta)`, "the C-only frame core D", "V is the variation core", "the C-only core D"; **N1**
  the inventory's roles line. Closed.
- **The session's four rows**: `rsig_hat` (3 tokens; the `cd,` operand padded so the `contract` argument columns hold);
  `entries.py:344` "Upward edge variables", re-registered as EXEMPT like its siblings `:124, 151, 314` (K9, converse and
  registered PASS) with a `R1_s33_ward_sites.tsv` row; `test_curvature.py:10–12` "``tv_to_dense(t3_corewise_frame(D,
  L), theta_hat) == tv_to_dense(frame, v)`` for ``theta_hat``, the lift of ``v``" (§3.1's template; the next bullet's
  `<L v, y>` has no store letter and stays); `apply.py:168–169` `D_i w_i`, `V_i w_i` (`(nD,N)(N,) = (nD,)`, the code's
  `'Cio,Wo->WCi'`).
- **`TOL` 1e-14 → 1e-13** (`test_jet_recurrence.py:32`, trailing comment kept), the listed test-only change, with the
  floor-env reason in §0 item 11.
- **Realignment**: whitespace only on the renamed lines (`sd:706, 712, 794`, the twin's continuation, the oracle's
  `cd,`); my adjacent-pair scan: 0 ragged pairs in the eleven files against the tag.

**Runs on Revision 1** (mine, one after another; the `git diff` checksum identical before the battery and after the
last run):

| run | result |
|---|---|
| gates (`--stage R4b --hunks R4b_name_hunks.tsv --names dGxi,Hxi,rsh,rsig_hat`) | K0, K2, K3, K6, K8b, K9 (+converse, registered), K10/K10b, K11, K13, K14 (0 lost; the listed `H`s licensed), **K15 36/36 listed**, K15-cross, K17 (+ shape) **PASS**; K1/K4/K5/K7/K8 2/16/7/49/395, all in the docs, notes and `CLAUDE.md` (R5's); K12 54/54 code-tree blocks |
| battery | **PASS (bitwise)**, 2,190 arrays, 856 texts, 0 errors |
| doctest outputs (static, HEAD → tree) | 297 examples, 0 expected outputs changed |
| raggedness scan | 0 ragged pairs |
| suite `-n 3` | **1,347 passed, 4 skipped, 48,724 subtests passed, 0 failed** (16 min 44 s) |
| `pytest --doctest-modules t3toolbox` | **241 passed** |
| `python -m doctest docs/getting_started.rst` and every `docs/*.md`, `docs/contributor/*.md` but `doctest_style.md` | **pass** (0 failures) |
| `sphinx -W --keep-going -b html` | **build succeeded**, exit 0, no warning (14 min 53 s) |
| floor env (`tttt`), the three touched test modules together | **159 passed, 2 skipped** (TOL 1e-13); `--doctest-modules t3toolbox` **241 passed** |

No MAJOR or MINOR remains: every ruling is applied as verified, MINOR 1 and 2 and N1 are closed, the session's four rows
and the `TOL` change are in, and every gate and run passes.

VERDICT: APPROVE
