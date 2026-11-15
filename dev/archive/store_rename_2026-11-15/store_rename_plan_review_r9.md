# The store rename plan — critic review, round 9 (over-reach)

Reviewed: draft 9 (1,000 lines, 01:58), read as a literal-minded implementer. This round withdraws "no letters in the
API" and adds §0's exhaustive change list, §1.5 DO NOT TOUCH with K14, the bounded §3, the reopened D-2 and the
implementer's brief.

How I checked: I enumerated the public API's letter and Greek names (signatures, dataclass fields and properties of
every `__all__` member); every `U`-letter NAME token outside tier C's mapped files; the SVD/QR letter locals in mapped
files; and every match of tier D's shift/derivative patterns.

**Sound.**
- §0's nine-item list is exhaustive, with "any rename not on it … forbidden without asking".
- §3's bounded instruction ("not listed → not renamed, ask") and §3.3's "a `U`-family name not on that list is not
  renamed".
- §1.5 items 1, 3, 4, 6–10 cover what the public API exposes: the Greek stems and derived names (`xis`, `etas`, `mus`,
  `nus`, `sigma_tildes`, `theta`, `rho`, `alpha`, `kappa_guard`, `tau`, …), `svd_U`/`svd_Vt`, `eps`, the operand
  letters `A`, `M`, `S`, `T`, `X`, `Y`, `Z`, and everything not in the map.
- K3 per form; the brief's trace rule.

## MAJOR

**O1. The token map is not closed.** K14 compares only the names on `dnt.txt`. Nothing mechanical fails on an
identifier that is renamed but is in neither the map nor the list; the brief's trace rule is a reviewer's reading.
Proposal, mechanical, per file, on NAME tokens:
- **K15 after R2.** R2 is the script slice, and its edits are in-place token replacements, so the old and new NAME
  sequences have equal length. At every position, new must equal old unless R1's hit table lists that (file, line,
  column) with exactly that image. The hit table enumerates the tiers and §3.4's dispositions occurrence by occurrence.
- **K15 after R3, R4a, R4b and R5.** `difflib` the NAME sequences against the previous slice's commit. Every unequal
  hunk must lie in a hunk the handoff lists (R3: the `update` helper, steps, loops and tests; R4: §4.11's loop
  variables and §3.4's R4 dispositions). A prose-only slice leaves the NAME stream identical.
- **Optional cross-check: the set difference.** Old − new ⊆ map sources, and new − old ⊆ map images ∪ the handoffs'
  new names (`_update_stores`, `_STORE_LETTERS`, `_COLOURS`, `n_cores`, `start_dense`, the `c`-suffixed locals, …).
  It is cheap, but blind to an occurrence renamed into a name that already exists (a T3 `U` → `D` in a file where `D`
  is a legitimate image), so it cannot replace the positional check.

K14 then becomes a corollary; keep it as the readable list.

**O2. §1.5 misses two meaning classes of `U`, and §3.1 licenses one of them.**
- **`U` as a T3's own Tucker core in non-frame code.** These are not decomposition factors:
  - `t3_linalg.py` outside 660–760: `U`, `Ux`, `Uy`, `Uxi`, `Uyi`, `Uw`, `UL`, `UR`, `Ua`, `Ub`, `Utilde`, `Uy_rev`.
    §1.5 files these under "decomposition factors", the wrong meaning.
  - `geometry.py:395–418`, `t3_operations.py:89`.
  - `t3_orthogonalization.py`: `Uio`, `Uox`, `U_i_o`. §3.4's `Uio`→`Bio` is the only listed change.
  - Tests: `oracle_sweeps/sweep_models.py:49–53`, `test_weighted.py:401` (`Ux`), and `test_sharing.py:245, 598`
    (`U_rows`).

  §3.1's bullet "A T3's own Tucker cores, which T4S writes `U` … These become `D`" reads as licence for all of them.
  It contradicts tier C's "outside the range `U` is a T3's Tucker core and stays" and §3.3's `B` on the T3 pages.
  Bound it: only tier D's corewise tuples, tier C's `U`-family row and §3.4's classified hits become `D`; every other
  T3 `U` stays.
- **`U`/`u` meaning *uniform*:**
  - `UT3*`; `UB`, `UV`, `UW`, `UF`, `UFb`, `Usc` (`test_dispatch.py:1705–1708`,
    `test_aligned_padded_frame_frontend.py:72`, `test_uniform_frame_variations_format.py:398`);
  - `o1_uniform.py:193`'s `U = ut3.UniformTuckerTensorTrain.stack(us)`, a uniform T3, in a file tier C's `U`-family
    row touches;
  - `utv_sampling`'s "U3", and the `u`/`ut3_`/`ufv_`/`utv_`/`_u_` prefixes.

Add both classes to §1.5 and to `dnt.txt`.

## MINOR

1. **The catch-all tier-C row is still a rule, not a list.** It reads "the rule map `U→D, O→U, P→L, Q→R` (plus
   letter-suffixed locals), scoped by the enclosing function". "Plus letter-suffixed locals" is open-ended, and "every
   `U` in the function" can take in other meanings (`test_sharing.py:576–598`: `UU, OO = frame_d[0], frame_d[1]` beside
   `U_rows`). Replace the row with R1's enumerated (file, function, token → image) list, reviewed in R1. R2 applies
   that list and nothing else.
2. **Rule-like sentences to bound to their sites:**
   - §3.3 "A point given by a frame's stores is written in store letters, e.g. `θ = (D, L)`": list the sites
     (`fitting.py:380`, `ut3_sampling.py:374`, `shared_geometry.py:45`, the curvature page, …).
   - §3.3 "A misnamed variation takes the house letter instead": say "the one site `tv_operations.py:292–331`".
   - §4.11 "Loop variables named for the old store are renamed by hand": say "these, and no others".
   - §4.1 "Docstrings state the rule in one clause": say "the listed convention-statement docstrings".
   - Tier D's shift/derivative patterns cite "about 13 docstrings and comments". Enumerate them. Outside
     `sampling_derivatives.py` I find 15 code lines (`sharing.py:757`; `t3_linalg.py:887, 889`; `optimizers.py:220`;
     `tv_operations.py:510`; `fv_operations.py:560, 579`; `_curvature_oracles.py:11, 64, 178, 238`;
     `test_uniform_curvature.py:1617, 1704`; `test_curvature.py:1577, 2852`) plus `deferred_and_rejected.md:134` and
     `curvature_internals.md:310`. All are the intended meaning, so this is a matter of enumeration.
   - §1.5 item 2's "Sites:" reads as exhaustive but misses factor sites in files the maps touch:
     - `test_sharing.py:372`, QR `Q`;
     - `test_padded_frame.py:99`, `Q, R = qr`;
     - `test_padded_start.py:582` and `_padded_frame_oracles.py:164`, an SVD's `Q`, inside functions tier C maps;
     - `test_dispatch.py:163, 2095`, `P` from a QR (a completion).

     Add them, or write "include" and make `dnt.txt` the authority.
3. **D-2: form (a)'s silent case is the common case.** `'UL'` is one of the two colours, and `'U'` is the obvious
   single-store call. Both are accepted with the other meaning; only a string containing `'O'` raises. Lay out a third
   form for Nick:
   - (c) letters as a collection of single-letter strings, `('D', 'L')`, with a `str` refused.
   - It keeps his letters, yet every 2026.4.0 call fails loudly, and `'O'` can still get its pointed message.

   The choice is Nick's; the plan should show (c) beside (a) and (b).

## NIT

- §0's "Everything else stays byte-identical": say "every other identifier and code token". Prose still changes by
  tier E.

VERDICT: REVISE (2 MAJOR, 3 MINOR)
