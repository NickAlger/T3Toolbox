# The store rename plan — critic review, round 1

Reviewed: `dev/store_rename_plan_2026-11-10.md` draft 1 as on disk 2026-11-11 00:11 (463 lines), against the HANDOFF
decision, the three inventories, BR, the code at `79b2bd6e` (= `v2026.4.0` for `t3toolbox/`, `tests/`, `examples/`) and
`t4s.pdf`. Nick's rulings of 2026-11-11 (no T4S exemption; corewise frame `(D, G, G, G)`; `P`/`Q` → `L`/`R`
everywhere; sentinels as an option) arrived mid-review. Findings they make moot are marked **MOOT**; what they require
of draft 2 is §R. Measurements were run in the `t3toolbox` env; scripts in the session scratchpad.

**Spot checks (all matched the code):** inv 2 §4 rows `padded_frames.py:13, 554, 560, 561, 581, 624, 636, 646, 655,
687, 750, 755, 760, 765`, `probing.py:646, 647, 773, 824, 1221, 1222`, `t3_orthogonalization.py:147, 171`,
`ut3_orthogonalization.py:189`, `geometry.py:348`, `t3_svd.py:312, 345`, `fv_conversions.py:58, 60, 157, 159, 181,
182, 184, 185, 219, 251`; inv 2 §9 `uniform_padded_frames.py:541-609` (34 rows) and `tv_operations.py:136-401` (15);
inv 1 §7 `test_padded_frame.py:72-216, 619-629` (35). The token-map invariants hold: every A1 image is new (only
`fv_conversions.py:58, 157, 219` carry a target spelling today, all handled); `'U'`→`'down'`, `'O'`→`'up'` in the
adapter, `_COLOURS` and §4.5; old `nU`→`n_down`, `nD`/`nO`→`n_up`; B2's in-place key swap keeps the `_SEED_TAGS`
integers with the physical store. Fixture claims measured: `xS` stores `(3,6,2)/(2,4,2)/(1,2,5,1)/(1,2,2,1)`,
`UT3Frame.from_t3frame(F, 8, 7, 5, 6, 3)` gives widths `7, 5, 6, 3`. D-1's factual claim is true: `t4s.pdf` has no
shift operator; `\Dsh`/`\Ush` are defined in `exact_hessian_two_hole.tex:44-45`.

## MAJOR

**M1. Bare store words in code prose have no tier, no slice and no check.** §1.1 says prose words are never scripted;
tier E takes only the flagged rows (HAND, AMBIG, ...) plus inv 2 §3/§4; K1–K6 cannot see a bare word. That leaves
the unflagged STORE-WORD rows of kind `cmt`/`doc`/`str` in inv 2 §9 — 175 rows with a bare `up→down`/`down→up`, of
which 50 are tier-D listings and about 125 are free prose (e.g. `uniform_padded_frames.py:563` "step 2, up edges",
`:592` "down edges"; `fv_operations.py:178, 239, 298-299, 313, 317`; the comments `padded_frames.py:143, 167` and
`uniform_padded_frames.py:89`, `# 'down' | 'left' | 'right'`, which B2 does not reach because they are comments) —
plus inv 1 §7's equivalent SW rows (`test_padded_frame.py:166, 184, 402, 561-562, 679`; the subTest label
`'in-block down'` at `:296`, a string B2 does not match). After R4 all of these read backwards and every gate is
green. Fix: (a) tier E takes every SW row of inv 1 §7 and inv 2 §9 that no A/B/D tier rewrites, with a per-file count;
(b) a residual gate after R4 (and R5 for docs): a tokenizer report of every `\b(up|down)\b` in comments, docstrings
and strings of the code tree, each hit matched to an inventory keep row (ORTH/MSG/UNREL) or failing — BR §2.A's
"afterwards `grep -rwE 'up|down'` lists only known non-store hits", made a gate. This is the only check that can see
the class.

**M2. Tier C is incomplete, so code letters fall through.** Tier C names files; a code (NAME) letter row in any other
file is neither tier C, nor tier E (CL is not a flag in §1.1), nor App. A. Example: inv 1 §7 proposes
`D0, U0, L0, R0 = old`, `D, U, L, R = self.start` for `test_padded_frame.py:111, 135-139, 205-216, 619-629`, but the
file is in no tier, while its module `padded_frames.py` and its sibling `_uniform_padded_frame_testing.py` are mapped
— so after R2 `U` means opposite stores in a module and its test. Also missed: `wU, wD, wL, wR`
(`_uniform_padded_frame_testing.py:43-85`, the per-store widths) are not in A3-scoped, and K2's `[kmpiW][UDLR]`
cannot see lowercase `w`. Under Nick's ruling 1 every frame-letter local converts, so tier C must be derived
exhaustively: every CL row of inv 1 §7 and App. A and every CORE-LETTER entry of inv 2 §8, each assigned a file (and
line range where a letter has another meaning), with the R1 hit table proving no CL row is unassigned.

**M3. The battery is blind to width swaps on the padded-frame paths, and misses sharing and `K`.** §5.2 says "the
battery uses unequal stores and widths throughout"; it does not. Measured: `x_old`'s frame is `(2,3,2)/(2,3,2)`, the
B7 target `T3Frame.padded(x_old, (3,5,3), (1,3,3,1))` is `(3,5,3)/(3,5,3)`, `x0`'s frame equal too. These are the
paths tier C and A3-scoped rewrite most (`nU0`/`nD0`, `kU`/`kD`, `pinU`/`pinO`, `WU`/`WD`, the message-function
swaps); a swapped pair of equal width ints is bitwise-invisible there. Add a slack target (measured:
`T3Frame.padded(x_old, (4,5,4), (1,2,3,1))` gives `(4,5,4)/(2,5,3)`) and a slack old frame (`test_padded_frame.py:743`
uses `((4,3,3,2), (1,2,3,2,1))`), in both layers, through the step and both loops. Also uncovered: the shared
uniform/tied paths (`ufv_shared_frame_data`, `utv_orthogonal_gauge_projection(shared_data=)`, `utv_operations.py:230-254`,
whose `up_mask_frame`, `up_mask_v`, `down_slices` A2 renames) and the shared geometries (`shared_manifold`) in B8; a
`K`-stacked tangent (`stack_tangents`, the `n_K` broadcasts) in B5; the stacked uniform twin of `xC` (name it).

## MINOR

1. **The adapter's tree test is always False.** `hasattr(bvf.T3Frame, 'up_tt_cores')`: a dataclass field without a
   default is not a class attribute (measured: `hasattr(T3Frame, 'up_tucker_cores')` is `False` on today's tree). Use
   `'up_tt_cores' in {f.name for f in dataclasses.fields(bvf.T3Frame)}`. The reverse map must also cover A3 (B2 reads
   the width properties: `n_down`→`nU`, ...), not only A1/A2.
2. **`wall_time` breaks the pass criterion.** `stats['history']` rows carry the host clock
   (`backend/optimizers.py:1207, 1657`); "every numeric stats entry ... `np.array_equal`" cannot pass. Exclude it by name.
3. **`update` subsets.** B7 runs 7 of the 15 non-empty subsets; all 15 in both layers is cheap, plus one permuted word
   tuple against the same letter set (the canonicalization).
4. **A doctest `up` that is not a store.** `fitting.py:290-291`: `>>> up = ut3m.UNIFORM_MANIFOLD.randn(...)` is a
   uniform tangent. The script now tokenizes doctest examples, so B1 renames it to `down`; add it to KEEP_BARE. BR's
   eight exceptions were counted on code tokens only, so re-derive the list over doctest lines.
5. **`down_tt_ranks`** (`uniform_frame_variations_format.py:203, 665`, return-shape comments; inv 1 rows 203/665) is in
   neither A1 labels nor K1. A2 turns the line above into `down_ranks` and this stale label survives every check.
6. **"…" in the token map.** `pU…`→`p_down…`, `iU…`→`i_down…`: list every token and image (`pU/pD/pL/pR`,
   `iU/iD/iL/iR`, `mU0`…`mR1` explicitly), and add `wU/wD/wL/wR` (M2).
7. **Docstrings must be partitioned, not processed twice.** `rename_python` applies the A tiers to STRING tokens and
   also tokenizes the `>>>`/`...` lines as code. If both paths touch an example line, every A2 swap on it applies twice
   and cancels — silently at the minimal-frame doctests (D-4). State the contract: example lines by the code path
   only, prose and expected output by the text path only.
8. **f-strings depend on the interpreter.** On Python ≥ 3.12 tokenize splits f-strings into FSTRING_MIDDLE + NAME, so
   the NAME pass renames `{self.up_ranks}` and the label text stays: `up_ranks={self.down_ranks}`
   (`frame_variations_format.py:288`, `manifold.py:211`, `uniform_manifold.py:184`). Pin the script to the 3.11 env or
   handle FSTRING_MIDDLE; B2's repr comparison is the net.
9. **The migration snippet** (inv 3 §1(b), adopted in §6) omits `up_tucker_cores_dot`/`down_tt_cores_dot`; `\b` does
   not match inside the `_dot` forms, though the table lists them.
10. **`nU` as a plain Tucker rank.** `optimizers.py:176-180`, `tests/test_chunk_size_estimator.py:274-276` and the
    `sampling_derivatives` chunk estimators (inv 2 C5 tail) use `nU` for a UT3 point's Tucker rank; A3 makes it
    `n_down`, while the house UT3 symbol is `n` (`UniformTuckerTensorTrain.from_t3(N, n, r)`, `ut3_operations`
    `n`/`r`) and inv 1 C7 proposed `n`. D-3 covers only `rL`/`rR`. Decide and list the sites.
11. **`dU` beside the new `U` outside the exempt files** (D-6 keeps the identifiers): `tv_operations.py:292-331`
    `dU = tt_variations[i]` — a TT variation — is assembled next to `U = up_tt_cores[i]` after tier C; `:401`
    `dU_i = <U_i, core_env_i>` names the Tucker variation by the up store's letter; `utv_operations.py:213-215, 315`
    (`Udot` "has the DOWN width", `[D | dU] = [down | up]`). The house letters exist: `V`/`H` (CLAUDE.md, δU = V,
    δG = H). Folded into §R-2.

## NIT

- §1.3 annotates `update: typ.Tuple[str, ...]` while D-2 accepts any non-`str` collection; annotate
  `typ.Collection[str]`. A generator reaches `len()` and raises `TypeError`, not `ValueError`.
- After B1, `fv_operations.py:654`'s loop variable `u` iterates the down weights (§4.10 fixes only `dn` on :653);
  likewise `u`/`o` in `test_padded_frame.py:491` and `_uniform_padded_frame_testing.py:41`.
- D-5's "`Ψ` is free" holds for the code, note and page; the dense oracles use `psi` for the right interfaces
  (`_padded_frame_oracles.py:26-48`, `test_padded_frame.py:172`). Say so in their letter note.
- Header: `docs/` differs from `v2026.4.0` by `glossary.md` and `index.rst`; "the code equals the tag" holds for
  `t3toolbox/`, `tests/`, `examples/` only.

## MOOT under Nick's rulings

D-1 as recommended (exempt list exactness, `_curvature_oracles.py`/`test_curvature.py` as unlisted jet twins, the
keeps "frame down jets"/"up-index"); D-6's `B`; D-7's exact-pattern scoping of `P`/`Q`; §3.3's "cited T4S formulas
keep T4S's letters" and tier D's `(P, Q, O) → G` keep.

## §R. What the rulings require of draft 2

- **R-1, no exemption.** Tier C rows for `probing`, `sampling_derivatives`, `apply`, `entries`, `_jet_trs_twins.py`,
  `test_jet_recurrence.py`, `_curvature_oracles.py`, `test_curvature.py`, `test_uniform_curvature.py` (inv 2 §8 has the
  proposals: `Ud→Dd` with `Od→Ud`, `mO`, `On`, `dmO`, `dOn`, `Odxi`, `nO`). **String letters:** `test_jet_recurrence.py`
  builds `dict(U=U, O=O, P=P, Q=Q, dU=dU, ...)` (`:67`, NAME, mapped) and reads `p['O']`, `p['P']`, `p['Q']` (34 STRING
  keys, unmapped by any tier; outside K3's `{2,4}`) — a scoped STRING-letter map plus a check. **The shifts:** new
  symbols for `D`/`U` in the code prose (~41 lines), `\Dsh`/`\Ush` (and their uses, `exact_hessian_two_hole.tex:573-583`),
  `deferred_and_rejected.md:134`, the glossary; checked against the symbols table and a grep of the touched files
  (`E`, `M`, `S`, `T`, `Ψ` are taken; `entries.py:361` already has a local `D` = mode count). Say whether
  `_shift_down`/`_shift_up` (words, not stores) keep their names and are carved out of the checks. The
  "frame down jets"/"up-index" keeps now read wrong beside the new words; reword them ("the η jets", "the ξ index").
  The four notes and the §6.3 substitution convert, each with one T4S correspondence line.
- **R-2, `(D, G, G, G)`.** State the letter of a raw T3's Tucker cores outside frames: `D` (consistent with the
  corewise frame) or `B` (README, user guide, `TuckerTensorTrain` doctests, glossary) — two letters for one object
  otherwise. Map the T3-variation identifiers that now sit beside the new `U` in converted modules (`Udot`, `Gdot`,
  `dU`, `dU_t`, `dU_tilde(s)`, `Ud`, `U_g`, `Udot_g`; ~200 tokens) and MINOR 11's `dU`; recommend spellings.
- **R-3, `P`/`Q` → `L`/`R` everywhere.** Still NAME-scoped: `P` is the completion/P-block (`padded_frames`,
  `uniform_padded_frames.py:117` parameter, `:566`), `Q` the QR factor (`fv_operations.py:315`), and `R` is an RNG
  helper in `test_jet_recurrence.py:166-176` and `test_probe_derivatives.py:488, 497` — a file-wide `Q→R` corrupts them.
  List the derived names (`P_cores`, `P_cores_dot`, `P_dot`, `Pxi`, `dPxi`, `Pd`, `Qxi`, `Q_dot`).
- **R-4, sentinels — my judgment.** In the Python pass the one-lookup-per-token map is already simultaneous by
  construction: chains (`up_tucker_ranks`→`down_ranks`, itself an A2 source), swaps and `Od→Ud` beside `Ud→Dd` cannot
  collide, and a sentinel phase adds a second rewrite whose intermediate is not runnable. Where sentinels earn their
  place is the text: tier D, the docs, the tex notes and the shift/letter rewrites in prose, which are regexes applied
  in several passes; there a two-phase map makes sequential edits safe, and "no `ZQX` survives" is a cheap extra gate.
  It proves every matched token finished, not that every token that needed a match was matched (M1, M2 and the
  bijection check cover that). Recommendation: tokenizer map for Python, sentinel two-phase for text, the zero-sentinel
  grep in the R2/R4/R5 gates. The risk is in the scoping, not the tool.
- **Slices.** The converted sampling/jet modules (~900 rows) plus the shift rename make R4 the largest slice; split it
  (frame/padded/uniform prose vs sampling + jets + shifts) with a battery run and a review stop after the jets.

VERDICT: REVISE (3 MAJOR, 11 MINOR)
