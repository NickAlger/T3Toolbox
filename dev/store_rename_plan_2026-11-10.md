# The store rename — plan (draft 0, session, 2026-11-10)

> **Status:** first draft by the orchestrating session, written from the three inventories
> (`dev/store_rename_inventory_1_frontend_tests.md`, `_2_backend.md`, `_3_docs.md`) and the blast-radius analysis
> (`dev/up_down_naming_blast_radius.md`). Next: writer/critic rounds to agreement, Nick's review, then
> implementer/reviewer slices R1–R5. A **breaking** change; it is the first item of release 2026.5.0 and precedes the
> glossary update and the 601-row docs-consistency pass (`dev/glossary_naming_decisions.md`), which must run on the
> renamed code.

## 0. The decision and the principle

Nick, 2026-11-10. A T3 is a TT chain of cores with a Tucker factor hanging off each core. Its frame-variations format
holds one store of cores per directed edge. **A store is named for the side of its edge that its cores occupy**, in the
standard picture (TT chain on top, Tucker factors hanging below): the TT–TT edges have a **left** and a **right** store,
the Tucker–TT edges an **up** store (the TT-shaped cores above the edge) and a **down** store (the Tucker-factor-shaped
cores below it). With that rule the orthogonality adjective follows from the name: left cores are left-orthogonal, right
cores right-orthogonal, up cores up-orthogonal (`up_orthogonalize_tt_cores`), down cores down-orthogonal
(`down_orthogonalize_tucker_cores`). Every orthogonalization method in the library already carries the right adjective.
What is wrong today is the store vocabulary, an accident of an early version: the Tucker-factor store is called "up"
(after the SVD letter U) and the TT-shaped store "down" (letter O). **The words flip.** Down→up and left→right then both
increase, like x and y.

Consequences decided with it:

- **No letters in the API.** `update` takes words, not `'ULRO'`; the four widths are `n_up, n_down, r_left, r_right`,
  never `nU, nD, rL, rR` (today's `nU` is the Tucker-store width = the new `n_down`; today's `nD` = the new `n_up`).
- **Letters stay in diagrams and equations: U = up (TT-shaped), D = down (Tucker factor), L, R** — the words' initials,
  like L and R today — unless a severe conflict is found (§3 reports one and proposes a bounded exemption).
- **Physical tuple order unchanged.** `T3Frame.data` keeps its order, which in the new words reads
  `(down, up, left, right)`; save/load is positional (`'f%d_%d'` keys), so files are unaffected.
- **Same-spelling names swap meaning** (`up_ranks`, `down_ranks`, `up_mask`, `up_weights`, …). Accepted: the users today
  are Nick and known colleagues; after the T4S and T3Polynomial papers there will be more, so fix now and say it in the
  first line of the release notes.

## 1. The mapping (one simultaneous substitution)

The old and new vocabularies overlap (old "up" = new "down"; old diagram letter D already marks the TT store in some
figures), so the rename is **one simultaneous mapping applied in a single pass per file**, never a sequence of
find-and-replace steps (a sequential pass corrupts every token it touches twice). The implementer builds the token map
below into one script and applies it once, then does the hand edits of §4.

| today | meaning (unchanged) | new |
|---|---|---|
| store word `up` (Tucker-factor-shaped cores) | the down side of a Tucker–TT edge | `down` |
| store word `down` (TT-shaped cores) | the up side of a Tucker–TT edge | `up` |
| `left`, `right` | unchanged | unchanged |
| `up_tucker_cores`, `up_tucker_supercore`, `up_weights`, `up_weight_supercore`, `up_mask`, `variations_up_mask`, `up_ranks`, `up_tucker_ranks` | the Tucker store's … | `down_…` |
| `down_tt_cores`, `down_tt_supercore`, `down_weights`, `down_weight_supercore`, `down_mask`, `variations_down_mask`, `down_ranks`, `down_tucker_ranks` | the TT store's … | `up_…` |
| width `nU` (and `nU°`, `nUi`, `WU`) | the Tucker store's width | `n_down` (`n_down°`, `n_down_i`, `W_down`) |
| width `nD` (`nD°`, `nDi`, `WD`) | the TT store's width | `n_up` |
| widths `rL`, `rR` (`WL`, `WR`) | unchanged meaning | `r_left`, `r_right` (`W_left`, `W_right`) |
| `update='ULRO'`, letters `'U' 'L' 'R' 'O'` | which stores a pass updates | `update=('down', 'up', 'left', 'right')` (a tuple of words; `'U'`→`'down'`, `'O'`→`'up'`) |
| the two-colour passes `'UL'` / `'RO'` | | `('down', 'left')` / `('right', 'up')` |
| core letter `U` (Tucker factor) in equations, diagrams, docstrings, variable names | | `D` |
| core letter `O` (TT-shaped core) | | `U` |
| letters `P`, `Q` (left, right cores in T4S notation) | | `L`, `R` (outside the §3 exemption) |
| `_SEED_TAGS = {'up': 0, 'down': 1, …}` | the seed streams' tags | `{'down': 0, 'up': 1, …}` — **the integers stay with the physical store** (the Gaussian seeds of 2026.4.0 are unchanged bitwise) |
| adjectives `up-orthogonal`, `down-orthogonal`, `left-`, `right-`; all `*_orthogonalize_*`, `*_svd_*`, `is_left/right_orthogonal` | | **unchanged**; the 22 sites that use an adjective backwards today (inventory 1 §1, inventory 2 §7, inventory 3) are fixed by hand |
| index letters inside contraction strings; `_shift_up`, `_shift_down`; "warm up"; sampling directions | not stores | **unchanged** (inventory 2 §6: no named-contraction helper or HISTORICAL string changes; only the table's comments flip) |

Pre-renames that must precede the mapping so the new D is free (inventory 2 §5 C2): the padded-frame coordinate
products `D, D_L, DU, …` (51 rows) become `E, E_L, EU, …`; the old down-core locals `D`/`DD` in `fv_operations` take the
store word; `D = shape[0]` in `entries` becomes `d0`. The forward jet shift `D` in the T4S-algorithm modules is covered
by §3.

## 2. The compatibility surface (what breaks, how)

Exhaustive tables: inventory 1 §2 (frontend), inventory 2 §2 (backend). The shape of it:

- **Loud** (AttributeError / TypeError on stale code): the renamed dataclass fields `T3Frame.up_tucker_cores →
  down_tucker_cores`, `down_tt_cores → up_tt_cores`, the uniform supercore fields, the `nU/nD/rL/rR` properties and
  keywords (`UT3Frame.from_t3frame(n_down=, n_up=, r_left=, r_right=)` and relatives), the 22 backend store-word keywords
  (`probing.compute_*`, `sampling_derivatives.compute_*_jets`: `up_tucker_cores → down_tucker_cores`,
  `down_tt_cores → up_tt_cores`), the 16 width keywords, `update` (a bare string is refused with a message naming the
  words).
- **Silent** (same spelling, swapped meaning): the properties `up_ranks` / `down_ranks` on `T3Frame`, `T3FrameWeights`,
  `UT3Frame`, `UT3Variations`, `UT3FrameWeights`; the fields `up_weights`/`down_weights`, `up_mask`/`down_mask`,
  `variations_up_mask`/`variations_down_mask`, `up_weight_supercore`/`down_weight_supercore` (and their constructor
  keywords); the backend keywords `up_ranks`/`down_ranks` of `ranks.frame_has_minimal_ranks` and
  `ufv_masking.ufv_make_frame_masks` (same positions, swapped names). These are listed by name in the release notes'
  first paragraph; the shapes differ (Tucker-shaped vs TT-shaped), so most misuse fails on a shape check.
- **Unchanged**: positional order of every frame tuple and uniform 6-tuple, `save`/`load`, every orthogonalization and
  SVD method, every optimizer keyword (`padded_*`, `cone_*` carry no store word), `alternate=`, all HISTORICAL strings.
- **Downstream**: T3Polynomial has one local unpack (`scripts/g01_g0_common.py:121`); `implicit_probing` does not use
  T3Toolbox; the research repo makes no library calls with these names. T3Polynomial's fix is a one-line follow-up after
  the release.

## 3. The letter conflict and the exemption (decision D-1)

Inventory 2 §5 C3, confirmed by inventory 1 §1.1 (3, 4) and inventory 3 §2: in `sampling_derivatives` (and the jet
kernels that transcribe the T4S algorithm) **`U` is the reverse jet shift inside the same equations that name the frame
cores** (e.g. `(μU)·[Uᵘσ̃]`), and the T4S letters `(U, O, P, Q)` with the substitution `(P, Q, O) → G` appear in cited
formulas there and in the two math notes written in T4S notation (`exact_hessian_two_hole.tex`, `shared_t3_math.tex`).
Making U the up-store letter there is a genuine ambiguity within single equations — the severe conflict Nick's rule
anticipates.

**Proposal:** adopt U/D/L/R everywhere **except** the T4S-transcription modules (`sampling_derivatives.py`, the jet
kernels it owns — the exact file list is the implementer's first deliverable in R1) and the two T4S-notation math notes,
which keep the paper's letters (U, O, P, Q), each with a one-line correspondence note at the top: *"T4S letters: U is
this library's down store (the Tucker factor), O its up store, P/Q the left/right stores."* Code that implements a cited
paper's formulas keeping that paper's notation is standard; the exemption is small and bounded; it also removes
inventory 2's C1 (old and new U side by side) and C4 (the `(P, Q, O) → G` substitution). The alternative — renaming the
jet shift inside the transcription — makes those modules diverge from the paper they cite, which is worse than one
scoped exemption. Within the exempted modules the **store WORDS still flip** (they are API keywords: `down_tt_cores →
up_tt_cores` etc.); only the equation letters keep the paper's convention. The glossary's T4S-correspondence entry
states both.

Other letter facts (no conflict): no `D` means a diagonal anywhere; `D` as the derivative `DΦ` in the exact-Hessian
note and as a message `D_{a→b}` in the padded-frame note are not in the same equations as store letters, but the
padded-frame note's `D_{a→b}` and `U_{x→t}` are renamed (`M_{a→b}`, `F_{x→t}`; D-5) so that note reads in the new
letters throughout.

## 4. Hand edits (where a token swap is wrong)

Lists with proposed text: inventory 2 §4 (69 backend edits), inventory 1 §3 (frontend convention sentences), inventory
3 §1(c) (15 doc pages needing a rewrite, not swaps) and §5. Classes of hand edit:

1. **Sentences that use both senses** ("the down core is up-orthogonal", "stores pointing toward the root", "down
   edges"): rewritten from the new convention, not swapped.
2. **The convention statements** (9 in docs, ~20 in docstrings): replaced by the one paragraph of inventory 3 §1(a),
   verbatim or by reference to `docs/frame_variations.md`, which becomes its home.
3. **Backwards adjectives** (22 sites across the layers' frame classes, `tucker_tensor_train.py`, two tests,
   `user_guide.rst:142`): corrected.
4. **The corewise frame** `(U, G, G, G)` (~22 sites): the point's own Tucker cores sit in the down slot and are written
   `D`: `(D, G, G, G)`.
5. **Error messages** naming stores (`padded_frames`, `uniform_padded_frames`, `geometry`, `fv_operations`): new words;
   the 12 tests that assert on that text updated with them.
6. **`_SEED_TAGS`** and its test copy (`test_padded_frame.py:91-92`): words swapped, integers kept — a test asserts the
   2026.4.0 seed rows bitwise.
7. **Two ASCII diagrams** (`frame_variations.md:13-17`, `user_guide.rst:124-133`) redrawn with U above and D below.
8. **The glossary** (`docs/glossary.md`): items N4 and N17 and the 51 store-related rows of
   `dev/glossary_naming_decisions.md` re-expressed; the "orthogonal core" entry loses its clash paragraph; a
   T4S-correspondence entry added.

## 5. Verification (the rename changes no number)

- **The bitwise battery (the main check).** A script `dev/store_rename_battery.py` runs, at v2026.4.0 and at the renamed
  tree, the same seeded calls — both layers, unshared and shared, stacked and not, equal and UNEQUAL up/down ranks —
  through: frame construction and conversion (`from_t3frame` and back), `t3svd`/retraction, the four orthogonalizations,
  `compute_orthogonal_representation_ranks`, the padded start and the aligned loop (all four `update` subsets incl. the
  two colours), `newton_cg` with and without `padded_from` / `'cone'`, probe derivatives and jets. Every array must be
  **bitwise identical** between the two trees (keys translated by the map). This is what proves the swap is a rename.
- **Unequal ranks everywhere.** Inventory 1 §1.2: most ragged fixtures and four doctests have equal up and down ranks,
  so a missed or doubled swap would pass. Every test module that builds frames gains at least one case with different
  Tucker-store and TT-store widths; the battery uses unequal widths throughout.
- **Completeness greps** (fail the suite if any hit): `\bnU\b|\bnD\b|\brL\b|\brR\b` in `t3toolbox/`, `tests/`, `docs/`
  (outside the exempted modules' equations); `'ULRO'|'UL'|'RO'` as `update` values; the old field names
  `up_tucker_cores|down_tt_cores|up_tucker_supercore|down_tt_supercore`; `update: str`.
- **The suite**: full suite at `-n 3` (t3toolbox env), the floor env (tttt) single-process, the numpy-1.26 scratch env
  for the padded-frame files; all 2,047 doctest examples (no output changes expected: inventory 1 §4); the dispatch
  rows (`update` as a hashable tuple of words is a jit static argument); the import-graph test; `sphinx -W`; the
  markdown and quickstart doctests on both envs.
- **PDFs rebuilt**: `padded_frame_cone.pdf` (Table 1), `pad_safe_svd.pdf` (two words), `exact_hessian_two_hole.pdf`
  and `shared_t3_math.pdf` (the correspondence line; their T4S letters stay under D-1).
- **Downstream**: T3Polynomial's `g01_g0_common.py:121` updated and its suite run, after the release.

## 6. Documentation and release notes

- The convention paragraph and picture (inventory 3 §1(a)) in `docs/frame_variations.md`, referenced everywhere else.
- `CHANGELOG.md` `[2026.5.0]` **Changed — breaking** entry and `docs/release_notes.md` "Upgrading from 2026.4.0"
  (drafts in inventory 3 §1(b), house format): the meaning swap stated first with the list of same-spelling names, the
  old → new table, the letter removal from the API, the width symbols, the T4S correspondence, the one-line migration
  recipe (search the listed spellings; for `up_ranks`/`down_ranks` swap, for the rest rename).
- The glossary updated (§4.8). The docs-consistency pass (601 rows) runs AFTER this plan's slices, on the renamed text.

## 7. Slices

| slice | content | check |
|---|---|---|
| **R1** | the exempted-module list (D-1) and the correspondence notes; the pre-renames (§1: `D…→E…` products, `DD`, `D = shape[0]`); the battery script run at v2026.4.0 to record the reference arrays (kept in the scratchpad or `dev/`, never committed as npz) | suite unchanged; battery reference written |
| **R2** | the simultaneous mapping over `t3toolbox/` and `tests/` (one script, one pass), incl. `update` as a tuple of words, the width symbols, `_SEED_TAGS` | battery bitwise vs R1's reference; full suite; completeness greps |
| **R3** | the hand edits of §4.1–4.6 in code and docstrings; the unequal-rank fixtures; the 12 message-text tests | suite; doctests on both envs; dispatch rows |
| **R4** | docs: the convention page, every user and contributor page (inventory 3 §5), the two diagrams, the math notes and PDFs, the glossary and its work list, CHANGELOG and release notes | `sphinx -W`; markdown doctests; the three greps over `docs/` |
| **R5** | acceptance: the §5 list end to end on three envs; the T3Polynomial follow-up prepared; handoff | — |

Each slice: implementer builds and writes `dev/store_rename_slices/R<k>_handoff.md`; reviewer verifies with its own
checks (for R2: an independent re-derivation of the token map from the inventories, and its own battery run) and writes
`R<k>_review.md`; review stops with Nick after R2 and R4.

## 8. Decisions needed from Nick

- **D-1** The T4S exemption of §3 (recommended) — or rename the jet shift and convert the two notes.
- **D-2** `update` as a tuple of words `('down', 'up', 'left', 'right')` (hashable, jit-static; order irrelevant;
  recommended) — or a frozenset, or a single string of space-separated words.
- **D-3** The optional rename of the plain-TT bond widths (inventory 2 "WIDTH-GENERIC", 86 lines): leave them (they
  are not store widths; recommended) or fold them into the `r_left`/`r_right` vocabulary.
- **D-4** The `padded_start` doctest prints `F.up_ranks` and passes with either name at its structure; change it to
  print both ranks at an unequal-rank structure (recommended) so a stale name shows.
- **D-5** The padded-frame note's `U_{x→t}` and `D_{a→b}`: rename to `F_{x→t}` and `M_{a→b}` (recommended) so the
  note is free of the old letters.
- **D-6** The corewise frame letter `(D, G, G, G)` (recommended by the rule) — or keep `U` there as the SVD letter
  with a note (it would be the only non-exempt U left).

## 9. Risks

- A sequential replace corrupting double-touched tokens — prevented by the one-pass script and the bitwise battery.
- A silent swap passing on equal-rank fixtures — prevented by §5's unequal-rank requirement.
- Doc sentences that read correctly after a swap by accident but say the wrong thing — the hand-edit lists are
  exhaustive by construction (inventories), and the reviewer reads every convention sentence.
- Readers of the T4S paper meeting the library's U as the TT-shaped core — the glossary and the two exempted notes carry
  the correspondence; the release notes say it.
