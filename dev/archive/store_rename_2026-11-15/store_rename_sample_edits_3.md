# Store rename: ten more sample edits (batch 3), and the least-determined one

Purpose: a third test of `dev/store_rename_plan_2026-11-10.md` (draft 18, final), in the areas batches 1 and 2
(`dev/store_rename_sample_edits.md`, `_2.md`) did not touch: a user page, the glossary and its work list, the
CHANGELOG and release notes, a test module, a uniform kernel outside `ufv_masking.py`, the dispatch tests,
`CLAUDE.md`, a user-guide diagram, a both-senses identifier, and a LaTeX width subscript. Nothing in the tree was
modified.

How to read it (as batches 1 and 2):

- **BEFORE** is verbatim from the working tree. `t3toolbox/`, `tests/` and `examples/` equal `v2026.4.0`
  (`git diff --stat v2026.4.0` touches only `docs/glossary.md`, `docs/index.rst`, `README.md`).
- **AFTER** is the final text once every slice touching the lines has run; the slice is named. Where the literal plan
  and the right answer differ, both are shown and labelled. Script-made code AFTERs were generated with one
  simultaneous token map per block (the tier rows quoted under "Governs") and realigned whitespace-only (§4 item 10);
  prose AFTERs are by hand.
- **Measured** means run on today's tree in the `t3toolbox` env (scratch scripts, not kept), reading the old names.
- **What could go wrong** names the slip, what in the plan prevents it, and says so when nothing does.

| # | site | mechanism | does the plan determine it? |
|---|---|---|---|
| 1 | `frame_variations.md` § Four rank stores, and the convention paragraph | §6, §3, tier E, A2 | yes, through §6's corrections to inv 3 §1(a) |
| 2 | glossary **frame cores**; work-list N4, N17 and store rows | §4.9, tier E, K1/K8 carve-outs | the entry yes; the work list's re-expressions are ungated and carry stale forms |
| 3 | CHANGELOG first breaking bullet; release-notes recipe | §6, §2, §1.3 | the silent list yes (44, re-scanned); two promises are false |
| 4 | `test_uniform_frame_variations_format.py` fixture and rank asserts; §5.2's new test | A2, A3, B1, §5.2 | yes |
| 5 | `utv_to_ut3`'s tied branch: masks, `pad`, widths | A2, A3, tier C, tier E | names yes; `Udot`'s letter no; its width comments contradict §4A(f) |
| 6 | `test_dispatch.py` alignment cases and recorded schedule | §4.5 (R3), tier E | code yes; the prose `update` forms no |
| 7 | `CLAUDE.md` canonical-orderings gotcha and core-letter note | tiers A, D, E (inv 3 App. A) | no: tier D turns inv 3's T4S tuple into a false sentence |
| 8 | `user_guide.rst:125–142` diagram and core list | §4A, K10, K10b | yes |
| 9 | `_u_up_messages` / `_u_down_messages`; "up edges" / "down edges" comments | A2, tier C, tier E, §4.2 | identifiers yes; the comments get the word swap §4.2 forbids |
| 10 | `shared_t3_math.tex` `n_{D,i}` → `n_{U,i}`, with `\dot D`, `D_g`, `Z` | §4.8, §4A(f), K16 | yes; the contraction-size check is written out |

---

## 1. A user page that states the convention, and the convention paragraph it receives

**Where:** `docs/frame_variations.md:68–75`, the first paragraph of § "Four rank stores — and why excess rank is
inert", which §6 makes the home of the convention paragraph, its picture and the T4S note.

**Governs:** §6 ("gets the convention paragraph, the picture and the T4S note, from inv 3 §1(a) **corrected**": widths
`nD`, `nU`, `rL`, `rR`; no "notes keep T4S's letters" sentence; `d` both the mode count and the upright differential
(D-11); a T3's own Tucker cores are `B` (D-6); the letter ↔ word mapping stated once, here and in the glossary's symbol
table); §3 (the T4S line in Nick's words, and this section is one of its places); A2 in R2 (`up_ranks`↔`down_ranks` in
`docs/*.md`); tier E, inv 3 §5.1's row `:70–71`, in R5; K2, K8 (T4S-line carve-out), K9.

**BEFORE**

*`docs/frame_variations.md:68–75`*
~~~md
## Four rank stores — and why excess rank is inert

A `T3Frame` does not carry *one* rank tuple, it carries **four**: `up_ranks` (the rows of each Tucker
factor `Uᵢ`), `down_ranks` (the mode leg of each down/outer core `Oᵢ`), and `left_ranks` / `right_ranks`
(the TT bonds, in each sweep direction). That redundancy is not bookkeeping sloppiness — it is what makes
non-minimal ranks harmless, and it is worth understanding before reading
[`numerical_contracts.md`](numerical_contracts.md)'s verdict that minimal rank is a precondition for
nothing.
~~~

**After R2 (interim).** A2 swaps the two names in place and leaves the letters, so the page reads wrong until R5:
"`down_ranks` (the rows of each Tucker factor `Uᵢ`), `up_ranks` (the mode leg of each down/outer core `Oᵢ`)". No
doctest is on these lines, so nothing fails in between.

**AFTER (R5)**
~~~md
## Four rank stores — and why excess rank is inert

A `T3Frame` does not carry *one* rank tuple, it carries **four**: `down_ranks` (the rows of each Tucker
factor `Dᵢ`), `up_ranks` (the mode leg of each up core `Uᵢ`), and `left_ranks` / `right_ranks`
(the TT bonds, in each sweep direction). That redundancy is not bookkeeping sloppiness — it is what makes
non-minimal ranks harmless, and it is worth understanding before reading
[`numerical_contracts.md`](numerical_contracts.md)'s verdict that minimal rank is a precondition for
nothing.

**Store names.** Draw a T3 with its TT chain on top and its Tucker factors hanging below, one under each TT core:

```
   1 --- c0 --- c1 --- c2 --- 1        TT chain: bonds r0 ... r3, each with a left and a right side
         |      |      |
         l0     l1     l2              Tucker factors: edges n0 ... n2, each with an up and a down side

   one TT bond:      c(i) --|-- c(i+1)        left core L(i) on the left side, right core R(i+1) on the right
   one Tucker edge:     c(i)                  up core U(i): TT-shaped, above the edge
                        -+-
                        l(i)                  down core D(i): the Tucker factor, below the edge
```

Every internal edge carries two frame cores, one on each side, and each is orthonormal over the indices on its
own side. **A store is named for the side its cores occupy**: the left store holds the cores on the left of
every TT bond, the right store those on the right, the up store the TT-shaped cores above every Tucker edge,
and the down store the Tucker factors below it. The orthogonality adjectives say the same thing: a left core is
left-orthogonal, an up core up-orthogonal (`up_orthogonalize_tt_cores`), a down core down-orthogonal
(`down_orthogonalize_tucker_cores`). Name a store by its side, never by where its edge points: as a directed
edge, a core's edge points away from its side (a left core's edge points right, an up core's edge points down
to the leaf). A frame keeps the stores in the order `(down, up, left, right)`, Tucker family first as in
`TuckerTensorTrain.data`.

**Letters.** Diagrams, equations, shape comments and the `update` argument write the stores `D`, `U`, `L`, `R`
(`Dᵢ` the down core, the Tucker factor; `Uᵢ` the up core). Their widths are `nD` (the down store's: the Tucker
factor's rows), `nU` (the up store's: the up core's mode leg), `rL` and `rR` (the left and right stores'
bonds); at a minimal frame the down and up ranks agree. A T3's own cores, outside a frame, are `Bᵢ` (Tucker) and
`Gᵢ` (TT), as in `TuckerTensorTrain`. `d` is both the number of modes and, upright, the differential (`dΦ`);
context tells them apart.

**T4S notation.** T4S writes `U`, `O`, `P`, `Q` for our `D`, `U`, `L`, `R`; its `δU` is our `δD` (the library's
`tucker_variations`, `V`), its `δG` our `δG` (`tt_variations`, `H`). The paper ↔ code maps
([`entries_apply_probe.md`](entries_apply_probe.md) §8, [`batching_and_stacking.md`](batching_and_stacking.md) §8)
keep the paper's letters.
~~~

**What changed against inv 3 §1(a), and what forces each change:**

- The widths sentence: inv 3 wrote `n_down`, `n_up`, `r_left`, `r_right` (D-8 withdrew them; K2 fails them).
- inv 3's T4S note ended "Pages that keep the paper's letters (the paper ↔ code maps, and the math notes written in its
  notation) say so". D-1 converted the notes, so only the maps keep the paper's letters (§3), and the note's body is
  §3's line verbatim.
- "Diagrams and equations use the initials": under D-2(c) and D-8 the letters are also in `update` and the shape
  comments, so the sentence names all four places.
- The `d` and `B` sentences are §6's additions.

**What could go wrong.**

- **Copying inv 3 §1(a) verbatim.** K2 catches `n_down`. Nothing catches "the math notes written in its notation" keep
  T4S's letters: no store word for K9, no `O` for K8. §6's sentence is the only guard.
- **Adding the corewise "the down slot holds B" sentence** from draft 17's CHANGES or batch 2 §11. Draft 18 dropped it
  (§1.2a: "no sentence says what the down slot holds: the tuple says it"), and CHANGES are history only. Determined.
- **The T4S note spans four physical lines** (its letters on the first two), and K8 matches its backticked `` `O` ``
  and `δU`. The carve-out is "the T4S correspondence lines (§3)". If R1 lists them by line, every physical line of the
  note must be listed (§12 item 11).
- **K9 on the inserted prose.** Its "up"/"down" have no aligned old token, so they pass only by clause (c), "part of a
  tier-E proposed text". The corrected paragraph is the implementer's text, not inv 3's, so R5's handoff must register
  it as the row's text, or K9 fails a correct paragraph.
- The rest of the section follows by tier E and A2 without judgement: `:82–83`'s table rows, `:94` (A2 gives
  "`down_ranks=(4,4,4)` alongside `up_ranks=(2,4,2)`", which is right: the Tucker store is (4,4,4)), `:101`
  "`down == up`", `:104` "the operative (up) ranks".

---

## 2. The glossary's **frame cores** entry, and the work-list rows N4, N17 and three store rows

**Where:** `docs/glossary.md:266–269`; `dev/glossary_naming_decisions.md:30` (the N17 note), `:81` (N4), `:93` (N17),
and the store rows `:198`, `:752`, `:877`.

**Governs:** §4.9 ("Glossary (inv 3 §5.2) and the 51 work-list rows (inv 3 §7; N4 and N17 resolve). The retired
spellings are listed under "also: (2026.4.0 …)" for one release"); A1 and A3 by script in R2 (the glossary is
`docs/*.md`); tier E (inv 3 §5.2 `:266–269`, §7) with §1.4's layering, R5; §3 (T4S letters only in the T4S entry);
K1, K8 and the carve-out "lines of `docs/glossary.md` containing "(2026.4.0""; §5.3's scope, which does not include
`dev/`.

**BEFORE**

*`docs/glossary.md:266–269`*
~~~md
- **frame cores** (up `Ui`, down `Oi`, left `Li`, right `Ri`; `up_tucker_cores`, `down_tt_cores`, `left_tt_cores`,
  `right_tt_cores`; also: `P`/`Q` for left/right, `D` or "outer" for down). Up: orthonormal Tucker cores. Down:
  TT-shaped `(rL, nD, rR)`, the TT core of the Tucker-variation terms (`Vi` varies in the complement of `Ui`'s rows).
  Left/right: the orthogonal TT environments. [Frame-variations](frame_variations.md#a-tangent-is-a-sum-of-d-terms)
~~~

**After R2 (interim).** A1 and A3 rewrite the field names and `nD` in place: "(up `Ui`, down `Oi`, …;
`down_tucker_cores`, `up_tt_cores`, …) … Down: TT-shaped `(rL, nU, rR)`". Wrong until R5; the glossary has no doctests.

**AFTER (R5)**
~~~md
- **frame cores** (down `Di`, up `Ui`, left `Li`, right `Ri`; `down_tucker_cores`, `up_tt_cores`, `left_tt_cores`,
  `right_tt_cores`; also: "outer" for up, T4S's letters (see **T4S**),
  (2026.4.0, names reversed: up `Ui` (Tucker), down `Oi` or `Di`, `up_tucker_cores`, `down_tt_cores`)).
  Down: the orthonormal Tucker factors `(nD, N)`. Up: TT-shaped `(rL, nU, rR)`, the TT core of the Tucker-variation
  terms (`Vi` varies in the complement of `Di`'s rows). Left/right: the orthogonal TT environments. Each store is
  named for the side of its edge its cores occupy. [Frame-variations](frame_variations.md#a-tangent-is-a-sum-of-d-terms)
~~~

The whole "(2026.4.0 …)" group sits on one physical line, so the carve-out covers every retired spelling. The **T4S**
entry gains §3's line ("T4S writes `U`, `O`, `P`, `Q` for our `D`, `U`, `L`, `R`; its `δU` is our `δD` …").

**BEFORE → AFTER, the work list** (`dev/glossary_naming_decisions.md`, R5)

*`:30`*
~~~md
- N17: once Nick decides (the adjectives stay; the store names may change), release or replace the HELD rows and
  update the glossary's **orthogonal core** and **frame cores** entries.
~~~
~~~md
- N17 (resolved 2026-11-10): the adjectives stayed and the store names flipped (the 2026.5.0 store rename,
  `dev/store_rename_plan_2026-11-10.md`). The HELD rows below are released in the new store words; the glossary's
  **orthogonal core** and **frame cores** entries were rewritten by the rename (R5).
~~~

*`:81`* (N4)
~~~md
| N4 | NAMING | the down store and its cores are named **down, `O`** (not `D`, not "outer" as a name). The orthogonality *adjective* is N17, not N4 | `align_padded(update=...)` letters `'U'`, `'O'`, `'L'`, `'R'`; `T3Frame.data = (up, down, left, right)`. Not "outer TT bonds" (`tucker_tensor_train.py:962`), which are the boundary bonds and stay |
~~~
~~~md
| N4 | NAMING (**RESOLVED** 2026-11-10: the store rename, 2026.5.0) | a store is named for the side of its edge its cores occupy: the TT-shaped store and its cores are **up, `U`**, the Tucker store and its cores **down, `D`**; not "outer" as a name (T4S's word; T4S's `O` is our `U`, its `U` our `D`). The orthogonality *adjective* is N17, not N4 | `align_padded(update=...)` takes a tuple of the letters `'D'`, `'U'`, `'L'`, `'R'` (a `str` raises); `T3Frame.data = (down, up, left, right)`; widths `nD`, `nU`, `rL`, `rR`. Not "outer TT bonds" (`tucker_tensor_train.py:962`), which are the boundary bonds and stay |
~~~

*`:93`* (N17): the status cell becomes "NAMING (**RESOLVED** 2026-11-10: the adjectives stay, the stores flip)"; the
text keeps the method convention and Nick's reasoning, with its last two sentences replaced by "The 2026.5.0 store
rename swapped the store names to agree. The former HELD rows below are released in the new store words; rows the rename
itself applied are marked done"; the evidence cell adds "(the rename's one-way A1 aliases)" after `fv_conversions.py:184`
and "(the rename's §4.3 rows)" after `:3108`.

*`:198`, `:752`, `:877`* (store rows)
~~~md
| 71 | § Four rank stores — and why excess rank is inert | N4 | "down/outer" | "down" |
| 164 | `tv_oblique_gauge_projection` | N4 | "down/outer" | "down" |
| 101 | `T3Frame` | N4 | rank diagram core "D1" (`T3Frame`; the next line already says `nO1`) | "O1" |
~~~
~~~md
| 71 | § Four rank stores — and why excess rank is inert | N4 | done by the store rename (R5): "`up_ranks` (the mode leg of each up core `Uᵢ`)" | — |
| 164 | `tv_oblique_gauge_projection` | N4 | "up/outer" (the rename's tier-E row swapped the word only) | "up" |
| 101 | `T3Frame` | N4 | done by the store rename (§4A block 98–108): "U1", next line `nU1` | — |
~~~

**What could go wrong.**

- **inv 3 §5.2's proposal for this entry** ("also: T4S's `Ui` for down, `Oi` and "outer" for up, `Pi`/`Qi` for
  left/right") puts T4S letters outside the T4S entry. K8 (`\bO…i\b`) fails `Oi`, as it should: §3 keeps T4S letters
  to the listed places. Determined and caught.
- **The carve-out is per line, and the glossary hard-wraps at 120.** Wrapped mid-list, `down_tt_cores` lands on a line
  without "(2026.4.0" and K1 fails it; the easy "fix" is to drop the retired name, which the house rule (aliases for one
  release) forbids. The plan does not say to keep the group on one line.
- **The neighbouring entry `:102`**, "**orthogonal core** (…; also: outer-orthogonal)", fails K8's `outer[- ]orthogonal`.
  It is N17's own alias, the house rule keeps aliases, and no carve-out covers it. Not determined (§12 item 11).
- **The work list's re-expressions are stale and ungated.** inv 3 §7 proposes, for these and other rows, "Verification:
  `update=` takes store words" (`:81`), "(next line `n_up_1`)" (`:877`), "`n_down_i`" (`:872–874`), "`r_left[d]` and
  `r_right[0]`" (`:101`, `:201`; D-3 keeps `rLd`, `rR0`), "`(r_left_i, n_down_i, r_right_(i+1))`" (`:110`), "``n_up``"
  (`:580`), "``n_up != n_down``" (`:903`). §1.4's L5 re-expresses word widths as letters, but nothing re-expresses the
  `update` form, `dev/` is outside §5.3's scope, and the 601-row pass that applies these rows to the docs runs after
  R6, when no K-check runs. Not prevented.
- **The "current" column.** `:752`'s current text is "down/outer" at `v2026.4.0` and "up/outer" after the rename (inv 2's
  row swaps the word). inv 3 §7 re-expresses only the target. The docs pass asserts each row's current text, so this
  fails loudly, which is good, but the re-expression should update both columns.

---

## 3. The CHANGELOG's first breaking bullet and the release-notes migration recipe, as the plan drafts them

**Where:** new `CHANGELOG.md` `## [2026.5.0]` § "Changed — breaking", first bullet; new `docs/release_notes.md`
§ "Upgrading from 2026.4.0"; the root `release_notes_2026.5.0.md` is the CHANGELOG section.

**Governs:** §6 (house format; the verdict first, then the same-spelling list "stated as silent, full stop", widths
included, "these do not raise", no shape-check claim; the table with "raises if left?"; the SWAP map with the `_dot`
forms, `nU`↔`nD`, `rL`/`rR` dropped; what does not change; letters and T4S; the `update` form; no "No store letters in
the API" bullet); §2 (the 44 silent sites); §1.3 (`_update_stores` and its message); inv 1 §2, inv 2 §2. Both files are
K-check carve-outs, so no gate reads either section.

**The silent list, re-derived.** An `ast` scan of every public function's parameters and every class's fields and
properties in `t3toolbox/` for A2 and A3 tokens gives exactly §2's sites: `T3Frame` (2 properties), `T3FrameWeights` (2
fields, 2 properties), `UT3FrameMasks` (2 fields), `UT3Frame` (2 rank and 2 width properties), `UT3VariationsMasks` (2
fields), `UT3Variations` (2 + 2), `UT3FrameWeights` (2 fields, 2 + 2 properties); `nU`/`nD` keywords on `from_t3frame`,
`from_t3variations`, `from_t3frameweights`, `from_t3tangent` and the backend `t3frame_to_ut3frame`,
`t3variations_to_ut3variations`, `t3frameweights_to_ut3frameweights`, `ufv_make_frame_masks`; `up_ranks`/`down_ranks`
keywords on `frame_has_minimal_ranks` and `ufv_make_frame_masks`. 18 + 4 + 16 + 6 = 44. Nothing else.

**BEFORE:** none in the tree. The plan's source is inv 3 §1(b) (`dev/store_rename_inventory_3_docs.md:105–207`), whose
second bullet ("**No store letters in the API.** … `'UL'` → `('down', 'left')` … become `n_down`, `n_up`, `r_left`,
`r_right` …") and SWAP line `'nU': 'n_down', 'nD': 'n_up', 'rL': 'r_left', 'rR': 'r_right'}` are withdrawn by §6.

**AFTER, CHANGELOG (R5)**
~~~md
### Changed — breaking

- **The frame stores `up` and `down` have swapped names: a name that kept its spelling now means the other store.**
  Same spelling, other store, and these do not raise: `up_ranks` / `down_ranks` (`T3Frame`, `T3FrameWeights`,
  `UT3Frame`, `UT3Variations`, `UT3FrameWeights`), `T3FrameWeights.up_weights` / `down_weights`,
  `UT3FrameWeights.up_weight_supercore` / `down_weight_supercore`, `UT3FrameMasks.up_mask` / `down_mask`,
  `UT3VariationsMasks.variations_up_mask` / `variations_down_mask` (the last four pairs also as constructor keywords),
  the `up_ranks=` / `down_ranks=` keywords of `ranks.frame_has_minimal_ranks` and `ufv_masking.ufv_make_frame_masks`,
  the width properties `nU` / `nD` (`UT3Frame`, `UT3Variations`, `UT3FrameWeights`), and the width keywords `nU=` /
  `nD=` (`UT3Frame.from_t3frame`, `UT3Variations.from_t3variations`, `UT3FrameWeights.from_t3frameweights`,
  `UT3Tangent.from_t3tangent`, `ufv_conversions.t3frame_to_ut3frame`, `t3variations_to_ut3variations`,
  `t3frameweights_to_ut3frameweights`, `ufv_masking.ufv_make_frame_masks`). `nD` is now the down store's width (the
  Tucker factors' rows) and `nU` the up store's (the TT-shaped cores' mode leg). At a minimal frame both stores have
  the same ranks, so stale code returns plausible numbers and is wrong only at slack. A store is now named for the
  side of its edge its cores occupy, as left and right always were: the **up** store is the TT-shaped cores
  (up-orthogonal, `up_orthogonalize_tt_cores`), the **down** store the orthonormal Tucker factors (down-orthogonal,
  `down_orthogonalize_tucker_cores`). In 2026.4.0 the two words named the direction the cores' edges point, the other
  way round. Not aliased, as for `GaussNewtonModel` in 2026.3.0: a meaning swap cannot be aliased.
~~~

**AFTER, release notes (R5)**
~~~~md
## Upgrading from 2026.4.0

**The frame stores `up` and `down` have swapped names: a name that kept its spelling now means the other store,
and does not raise.** `frame.up_ranks` was the Tucker factors' ranks and is now the up cores' mode legs;
`UT3Frame.nU` was the padded Tucker width and is now the up store's. At a minimal frame the two stores agree, so
such code even returns the right numbers there: search for the names below rather than waiting for an error.

| 2026.4.0 | 2026.5.0 | raises if left? |
|---|---|---|
| `up_ranks` / `down_ranks` (`T3Frame`, `T3FrameWeights`, `UT3Frame`, `UT3Variations`, `UT3FrameWeights`) | `down_ranks` / `up_ranks` | **no** |
| `T3FrameWeights.up_weights` / `down_weights` (fields and keywords) | `down_weights` / `up_weights` | **no** |
| `UT3FrameWeights.up_weight_supercore` / `down_weight_supercore` (fields and keywords) | swapped | **no** |
| `UT3FrameMasks.up_mask` / `down_mask` (fields and keywords) | swapped | **no** |
| `UT3VariationsMasks.variations_up_mask` / `variations_down_mask` (fields and keywords) | swapped | **no** |
| keywords `up_ranks=` / `down_ranks=` (`ranks.frame_has_minimal_ranks`, `ufv_masking.ufv_make_frame_masks`) | swapped | **no** |
| properties `nU` / `nD` (`UT3Frame`, `UT3Variations`, `UT3FrameWeights`) | `nD` / `nU` | **no** |
| keywords `nU=` / `nD=` (the four `from_*` converters, `ufv_conversions.*_to_ut3*`, `ufv_make_frame_masks`) | `nD=` / `nU=` | **no** |
| `T3Frame.up_tucker_cores` / `down_tt_cores` (fields and keywords) | `down_tucker_cores` / `up_tt_cores` | yes |
| `UT3Frame.up_tucker_supercore` / `down_tt_supercore` | `down_tucker_supercore` / `up_tt_supercore` | yes |
| keywords `up_tucker_cores[_dot]=` / `down_tt_cores[_dot]=` (`probing.compute_*`, `sampling_derivatives.compute_*_jets`) | `down_tucker_cores[_dot]=` / `up_tt_cores[_dot]=` | yes |
| `update='ULRO'` / `'UL'` / `'RO'` (`T3Tangent.align_padded`, `UT3Tangent.align_padded`, `tv_align_padded_frame`, `utv_align_padded_frame`) | `('D', 'U', 'L', 'R')` / `('D', 'L')` / `('U', 'R')` | yes for a `str`; see below |

**Migrate in one pass.** Two `sed` passes undo each other. Swap every name at once, and only these (never
`*_orthogonalize_*` or `*_svd_*`, whose words are unchanged):

```python
import re
SWAP = {'up_tucker_cores': 'down_tucker_cores', 'down_tt_cores': 'up_tt_cores',
        'up_tucker_cores_dot': 'down_tucker_cores_dot', 'down_tt_cores_dot': 'up_tt_cores_dot',
        'up_tucker_supercore': 'down_tucker_supercore', 'down_tt_supercore': 'up_tt_supercore',
        'up_ranks': 'down_ranks', 'down_ranks': 'up_ranks',
        'up_weights': 'down_weights', 'down_weights': 'up_weights',
        'up_weight_supercore': 'down_weight_supercore', 'down_weight_supercore': 'up_weight_supercore',
        'up_mask': 'down_mask', 'down_mask': 'up_mask',
        'variations_up_mask': 'variations_down_mask', 'variations_down_mask': 'variations_up_mask',
        'nU': 'nD', 'nD': 'nU'}
pattern = re.compile(r'\b(' + '|'.join(sorted(SWAP, key=len, reverse=True)) + r')\b')
source = pattern.sub(lambda m: SWAP[m.group(1)], source)    # then review the nU/nD hits: they may be your own names
```

`\b` treats `_` as a word character, so `up_mask` does not match inside `variations_up_mask`, nor `up_tucker_cores`
inside `up_tucker_cores_dot` (hence its own entry). `rL` and `rR` are unchanged. Your own `nU`/`nD` locals are swapped
along with the keywords, which keeps them consistent.

**`update` needs a hand edit.** It takes a tuple of the new store letters, any order: `'D'` down, `'U'` up, `'L'` left,
`'R'` right. Swap each old letter (`'U'` → `'D'`, `'O'` → `'U'`; `'L'` and `'R'` stay) and split: `'ULRO'` →
`('D', 'U', 'L', 'R')` (the default), `'UL'` → `('D', 'L')`, not `('U', 'L')`, which now updates the up store; `'RO'` →
`('U', 'R')`. A `str` raises:

```
ValueError: update must be a non-empty tuple of the store letters 'D', 'U', 'L', 'R' without repeats (D down, U up,
L left, R right), e.g. ('D', 'L'); got 'ULRO'; 'O' was the 2026.4.0 letter for the TT-shaped store, now 'U' (and
2026.4.0's 'U', the Tucker store, is now 'D')
```

A tuple or list of the old letters, which 2026.4.0 also accepted, raises only if it holds `'O'`: convert it the same way.

**What does not change.** Positions: `T3Frame.data` and every rank, mask, weight and uniform tuple keep their order,
which now reads `(down, up, left, right)`, so `up, down, left, right = frame.data` keeps working (rename the locals to
read right), and so do positional width calls (`UT3Frame.from_t3frame(F, N, 7, 5, ...)`). Saved files load unchanged.
Every number is bit-identical. The orthogonalization and SVD method names are unchanged.

**Letters.** Diagrams, equations and shape comments write `D`, `U`, `L`, `R` and the widths `nD`, `nU`, `rL`, `rR`;
a T3's own Tucker cores are `B`. Docs before 2026.5.0 wrote `U` for the Tucker factor and `O` for the TT-shaped core.
T4S writes `U`, `O`, `P`, `Q` for our `D`, `U`, `L`, `R`; its `δU` is our `δD` (`tucker_variations`, `V`), its `δG`
our `δG` (`tt_variations`, `H`). See [Four rank stores](frame_variations.md#four-rank-stores--and-why-excess-rank-is-inert).
~~~~

**Checked against the tables, and what could go wrong.**

- **Every silent site is named,** in both texts (the scan above). The width keywords and properties are in the silent
  list, as §6 requires.
- **"Every 2026.4.0 call fails loudly" (§1.3, §2) is false for tuple callers.** Measured: on today's tree,
  `tv_align_padded_frame(start, v, (2, 2, 2), (1, 2, 2, 1), update=X)` runs for X = `'UL'`, `('U', 'L')`,
  `['U', 'L']` and `('U',)`; 2026.4.0's check is `set(update) <= set('ULRO')`, which any collection passes. Form (c)
  accepts `('U', 'L')` and updates up and left, where 2026.4.0 updated Tucker and left. The annotation was `str`, so this
  is undocumented use, but the table's "raises if left?" must not say a plain "yes"; hence the last row and the sentence
  under the error. The plan does not prevent it.
- **The pointed message is only for a `str` containing `'O'`.** `'UL'` gets the bare message, and its obvious fix,
  `tuple('UL')`, is accepted and silently means up and left. The recipe says "not `('U', 'L')`"; the helper should add its
  sentence for every `str`, since every `str` is a 2026.4.0 call (§12 item 2).
- **"These do not raise" overclaims for the width keywords.** Measured: today's `from_t3frame(F, N=8, nU=5, nD=7, rL=6,
  rR=3)` at `xS` (what a stale `nU=7, nD=5` call means after the rename: a Tucker width of 5 below the Tucker rank 6)
  raises `ValueError: index can't contain negative values` from inside numpy; when both swapped widths fit both stores it
  pads silently. Recommended wording: "these need not raise, and when one does the message does not name the width". I
  kept §6's "these do not raise" above, as drafted.
- **Copying inv 3 §1(b).** Both files are carve-outs, so its `n_down` widths, its word-tuple `update` and its "No store
  letters in the API" bullet would pass every gate. §6's text is the only guard.
- **The SWAP itself.** Longest-first alternation is what makes `up_tucker_cores_dot` and `variations_up_mask` match
  whole; dropping the sort would leave `up_tucker_cores_dot` unrenamed (`\b` fails before `_dot`) and is loud only for
  backend callers.

---

## 4. A test fixture with unequal widths, its rank assertions, and §5.2's new assertion

**Where:** `tests/test_uniform_frame_variations_format.py:30–47` (`_make_frame` and the module constants) and `:63–75`
(`test_construct_and_structure`); §5.2's new test goes in the same module.

**Governs:** A2 (`_UP_R`↔`_DOWN_R`, `up_r`↔`down_r`, `up_ranks`↔`down_ranks`); A3 (`nU`↔`nD`; explicit `_NU`↔`_ND`); B1
(the locals `up`/`down`); §0 (physical order); §4 item 10 (body realignment, R4a); §5.2 ("one assertion per silent site
… at `xS`'s structure, the widths at `UT3Frame.from_t3frame(F, 8, 7, 5, 6, 3)` (`F.nD == 7`, `F.nU == 5`) … against the
physical core shapes … in each class's test module"); K15, the bijection.

**BEFORE**

*`tests/test_uniform_frame_variations_format.py:30–47`*
~~~python
def _make_frame(d, N, nU, nD, rL, rR, shape, up_r, down_r, left_r, right_r, ss=()):
    """Build a structurally-valid UT3Frame with prefix masks (ranks may vary per stack element)."""
    up    = np.random.randn(*((d,) + ss + (nU, N)))
    down  = np.random.randn(*((d,) + ss + (rL, nD, rR)))
    left  = np.random.randn(*((d,) + ss + (rL, nU, rL)))
    right = np.random.randn(*((d,) + ss + (rR, nU, rR)))
    masks = ubv.UT3FrameMasks(
        _prefix_mask(up_r, nU), _prefix_mask(down_r, nD),
        _prefix_mask(left_r, rL), _prefix_mask(right_r, rR),
    )
    return ubv.UT3Frame(up, down, left, right, tuple(shape), masks)


# concrete padded structure shared across tests
_D, _N, _NU, _ND, _RL, _RR = 3, 6, 4, 5, 3, 2
_SHAPE = (4, 5, 6)
_UP_R, _DOWN_R = [2, 3, 4], [3, 4, 5]            # (d,)
_LEFT_R, _RIGHT_R = [1, 2, 3, 1], [1, 2, 2, 1]   # (d+1,)
~~~
*`:63–75`*
~~~python
    def test_construct_and_structure(self):
        for ss in ((), (2,)):
            with self.subTest(stack=ss):
                B = self._frame(ss)
                self.assertEqual(B.shape, _SHAPE)
                self.assertEqual(B.stack_shape, ss)
                self.assertEqual(B.uniform_structure, (_D, _N, _NU, _ND, _RL, _RR, ss))
                # first stack element's ranks recover the chosen ranks
                idx = (slice(None),) + (0,) * len(ss)
                self.assertEqual(np.asarray(B.up_ranks)[idx].tolist(), _UP_R)
                self.assertEqual(np.asarray(B.down_ranks)[idx].tolist(), _DOWN_R)
                self.assertEqual(np.asarray(B.left_ranks)[idx].tolist(), _LEFT_R)
                self.assertEqual(np.asarray(B.right_ranks)[idx].tolist(), _RIGHT_R)
~~~

**AFTER (R2; R4a realigns `down  =` and `up    =`)**
~~~python
def _make_frame(d, N, nD, nU, rL, rR, shape, down_r, up_r, left_r, right_r, ss=()):
    """Build a structurally-valid UT3Frame with prefix masks (ranks may vary per stack element)."""
    down  = np.random.randn(*((d,) + ss + (nD, N)))
    up    = np.random.randn(*((d,) + ss + (rL, nU, rR)))
    left  = np.random.randn(*((d,) + ss + (rL, nD, rL)))
    right = np.random.randn(*((d,) + ss + (rR, nD, rR)))
    masks = ubv.UT3FrameMasks(
        _prefix_mask(down_r, nD), _prefix_mask(up_r, nU),
        _prefix_mask(left_r, rL), _prefix_mask(right_r, rR),
    )
    return ubv.UT3Frame(down, up, left, right, tuple(shape), masks)


# concrete padded structure shared across tests
_D, _N, _ND, _NU, _RL, _RR = 3, 6, 4, 5, 3, 2
_SHAPE = (4, 5, 6)
_DOWN_R, _UP_R = [2, 3, 4], [3, 4, 5]            # (d,)
_LEFT_R, _RIGHT_R = [1, 2, 3, 1], [1, 2, 2, 1]   # (d+1,)
~~~
~~~python
    def test_construct_and_structure(self):
        for ss in ((), (2,)):
            with self.subTest(stack=ss):
                B = self._frame(ss)
                self.assertEqual(B.shape, _SHAPE)
                self.assertEqual(B.stack_shape, ss)
                self.assertEqual(B.uniform_structure, (_D, _N, _ND, _NU, _RL, _RR, ss))
                # first stack element's ranks recover the chosen ranks
                idx = (slice(None),) + (0,) * len(ss)
                self.assertEqual(np.asarray(B.down_ranks)[idx].tolist(), _DOWN_R)
                self.assertEqual(np.asarray(B.up_ranks)[idx].tolist(), _UP_R)
                self.assertEqual(np.asarray(B.left_ranks)[idx].tolist(), _LEFT_R)
                self.assertEqual(np.asarray(B.right_ranks)[idx].tolist(), _RIGHT_R)
~~~

The fixture keeps every value and position: the Tucker supercore is still built first, at width 4, with ranks
`[2, 3, 4]`; only its names change (`down`, `nD`, `_ND`, `_DOWN_R`). The printed `uniform_structure` is the same
`(3, 6, 4, 5, 3, 2, ss)` and now reads `(d, N, nD, nU, rL, rR, C)`. The widths (4, 5) and the ranks differ, so this
test already sees a half-done swap.

**New (R3, §5.2), in `TestUT3Frame`**
~~~python
    def test_stores_are_named_by_side(self):
        """The silent names of UT3Frame, UT3FrameMasks and from_t3frame, read against the physical layout at a frame
        whose four stores and four padded widths all differ (the 2026.5.0 store rename)."""
        xS = t3.TuckerTensorTrain.randn((5, 6, 7), (3, 6, 2), (1, 2, 5, 1))
        F = bvf.T3Frame.from_t3(xS)                                     # down (3, 6, 2), up (2, 4, 2)
        for U in (ubv.UT3Frame.from_t3frame(F, 8, 7, 5, 6, 3),         # positional: N, nD, nU, rL, rR
                  ubv.UT3Frame.from_t3frame(F, N=8, nD=7, nU=5, rL=6, rR=3)):
            self.assertEqual((U.nD, U.nU), (7, 5))
            self.assertEqual((U.down_tucker_supercore.shape[-2], U.up_tt_supercore.shape[-2]), (U.nD, U.nU))
            self.assertEqual(U.down_ranks.tolist(), [c.shape[-2] for c in F.down_tucker_cores])   # [3, 6, 2]
            self.assertEqual(U.up_ranks.tolist(), [c.shape[-2] for c in F.up_tt_cores])          # [2, 4, 2]
            self.assertEqual((U.masks.down_mask.shape[-1], U.masks.up_mask.shape[-1]), (U.nD, U.nU))
~~~

Measured on today's tree, by the old names: `T3Frame.from_t3(xS)` has Tucker ranks (3, 6, 2) and TT-shaped ranks
(2, 4, 2); `UT3Frame.from_t3frame(F, 8, 7, 5, 6, 3)` has supercores `(3, 7, 8)` and `(3, 6, 5, 3)`, so position 2 (7) is
the Tucker width before and after. The module already imports `t3`, `bvf` and `ubv`. The `UT3Variations` and
`UT3VariationsMasks` sites get the same test in `TestUT3Variations`.

**What could go wrong.**

- **Writing the expected values from the old names** (`U.nU == 7`, or the keyword call `nU=7, nD=5`): that is the
  silent swap itself. §5.2 spells out `F.nD == 7`, `F.nU == 5`, and the assertions compare against the supercore shapes
  and the ragged frame's core shapes, not against remembered numbers. Determined.
- **A tautology.** `U.down_ranks` is computed from `U.masks.down_mask`, so asserting one against the other passes whatever
  the names mean. §5.2's example (`F.down_ranks == tuple(c.shape[-2] for c in F.down_tucker_cores)`) is the ragged one;
  for the uniform classes the physical cores are padded, so the check goes against the ragged source frame and the
  supercore widths, as above. The plan does not say this for the uniform classes; mild.
- **"Restoring" `_NU, _ND = 4, 5`** after the swap, so the constants keep their old values: the TT ranks `[3, 4, 5]` then
  sit in a width-4 mask and the rank assertion fails. Caught.
- The test's local `B` (a `UT3Frame`) is not §1.5 item 5's `B`; it stays.

---

## 5. A uniform kernel beyond the masking site: `utv_to_ut3`'s tied branch

**Where:** `t3toolbox/backend/utv_operations.py:209–258`, the SF-T3 branch of `utv_to_ut3` (the tied doubled-rank
embedding) and the width reads after it.

**Governs:** A2 (`up_sc`↔`down_sc`, `var_up_mask`↔`var_down_mask`, `up_mask_frame`, `up_mask_v`, `down_slices`↔
`up_slices`); A3 (`nU_pad`↔`nD_pad`, `nU`↔`nD` in code and comments); tier C `utv_operations.py` (`U→D, O→U`); tier E
(inv 2 rows `:213–215`: `O_i→U_i; UP→DOWN`); §0 (physical order); §1.2a, §3.3 ("A `U`-family name not on that list is
not renamed without asking"); §4A(f), D-8 (adjacency); K8, K9, K15; battery B5.

**BEFORE**

*`t3toolbox/backend/utv_operations.py:209–258`*
~~~python
    With ``shared_data`` (the frame's uniform SF-T3 companion), the embedding is built TIED, mirroring
    the ragged branch: per nontrivial group the common gauged ambient direction ``Udot``
    (:py:func:`~t3toolbox.backend.sharing.fv_tied_ambient_directions` -- garbage-immune through the
    companion's masked ``U_M``) takes the variation slot at EVERY group mode, and the companion's
    center core ``H_i`` replaces ``O_i`` (an exact rewrite of each Tucker term). ``Udot`` has the UP
    width, so the variation block is rebuilt at height ``nU``: singleton slices zero-pad from ``nD``
    (block masks pad with False); group block masks are the frame's UP mask."""
    up_sc, down_sc, left_sc, right_sc, shape, _base_masks = frame_data
    tkv, ttv, _shape_v, var_masks = variations_data
    var_up_mask, var_down_mask, var_left_mask, var_right_mask = var_masks

    use_jax = tree_contains_jax((up_sc, down_sc, left_sc, right_sc, tkv, ttv))
    xnp, _, _ = get_backend(True, use_jax)

    if shared_data is not None and sharing_module.nontrivial_groups(shared_data.groups):
        udots = sharing_module.fv_tied_ambient_directions((tkv, ttv), shared_data)
        nU_pad, nD_pad = up_sc.shape[-2], down_sc.shape[-2]
        pad = nU_pad - nD_pad
        d0 = up_sc.shape[0]
        ss_v = tkv.shape[1:-2]                              # K + C
        n_K0 = len(ss_v) - (up_sc.ndim - 3)
        up_mask_frame = frame_data[5][0]                    # (d,)+C+(nU,), HOST
        up_mask_v = np.broadcast_to(                        # (d,)+K+C+(nU,): the frame mask over K
            up_mask_frame.reshape(up_mask_frame.shape[:1] + (1,) * n_K0 + up_mask_frame.shape[1:]),
            up_mask_frame.shape[:1] + tuple(ss_v) + up_mask_frame.shape[-1:])
        mode_slot = {ii: (gi, jj)
                     for gi, group in enumerate(sharing_module.nontrivial_groups(shared_data.groups))
                     for jj, ii in enumerate(group)}
        tkv_slices, down_slices, block_masks = [], [], []
        for ii in range(d0):
            if ii in mode_slot:
                gi, jj = mode_slot[ii]
                tkv_slices.append(udots[gi])                             # ONE array per group, (K+C)+(nU, N)
                down_slices.append(shared_data.centers[gi][jj])          # H_i replaces O_i, (C)+(rL, nU, rR)
                block_masks.append(up_mask_v[ii])
            else:
                v = tkv[ii]
                tkv_slices.append(xnp.concatenate(
                    [v, xnp.zeros(v.shape[:-2] + (pad, v.shape[-1]))], axis=-2))
                O = down_sc[ii]
                down_slices.append(xnp.concatenate(
                    [O, xnp.zeros(O.shape[:-2] + (pad, O.shape[-1]))], axis=-2))
                block_masks.append(np.concatenate(
                    [var_down_mask[ii], np.zeros(var_down_mask[ii].shape[:-1] + (pad,), bool)], axis=-1))
        tkv = xnp.stack(tkv_slices, axis=0)                 # (d,)+K+C+(nU, N)
        down_sc = xnp.stack(down_slices, axis=0)            # (d,)+C+(rL, nU, rR)
        var_down_mask = np.stack(block_masks, axis=0)       # (d,)+K+C+(nU,), HOST

    d  = up_sc.shape[0]
    nU = up_sc.shape[-2]; N = up_sc.shape[-1]; nD = down_sc.shape[-2]
~~~

**AFTER (R2 for NAMEs and widths; R4a for the docstring, the comment letter `U_i` and the realignment)**
~~~python
    With ``shared_data`` (the frame's uniform SF-T3 companion), the embedding is built TIED, mirroring
    the ragged branch: per nontrivial group the common gauged ambient direction ``Bdot``
    (:py:func:`~t3toolbox.backend.sharing.fv_tied_ambient_directions` -- garbage-immune through the
    companion's masked ``U_M``) takes the variation slot at EVERY group mode, and the companion's
    center core ``H_i`` replaces ``U_i`` (an exact rewrite of each Tucker term). ``Bdot`` has the DOWN
    width, so the variation block is rebuilt at height ``nD``: singleton slices zero-pad from ``nU``
    (block masks pad with False); group block masks are the frame's DOWN mask."""
    down_sc, up_sc, left_sc, right_sc, shape, _base_masks = frame_data
    tkv, ttv, _shape_v, var_masks = variations_data
    var_down_mask, var_up_mask, var_left_mask, var_right_mask = var_masks

    use_jax = tree_contains_jax((down_sc, up_sc, left_sc, right_sc, tkv, ttv))
    xnp, _, _ = get_backend(True, use_jax)

    if shared_data is not None and sharing_module.nontrivial_groups(shared_data.groups):
        udots = sharing_module.fv_tied_ambient_directions((tkv, ttv), shared_data)
        nD_pad, nU_pad = down_sc.shape[-2], up_sc.shape[-2]
        pad = nD_pad - nU_pad
        d0 = down_sc.shape[0]
        ss_v = tkv.shape[1:-2]                              # K + C
        n_K0 = len(ss_v) - (down_sc.ndim - 3)
        down_mask_frame = frame_data[5][0]                  # (d,)+C+(nD,), HOST
        down_mask_v = np.broadcast_to(                      # (d,)+K+C+(nD,): the frame mask over K
            down_mask_frame.reshape(down_mask_frame.shape[:1] + (1,) * n_K0 + down_mask_frame.shape[1:]),
            down_mask_frame.shape[:1] + tuple(ss_v) + down_mask_frame.shape[-1:])
        mode_slot = {ii: (gi, jj)
                     for gi, group in enumerate(sharing_module.nontrivial_groups(shared_data.groups))
                     for jj, ii in enumerate(group)}
        tkv_slices, up_slices, block_masks = [], [], []
        for ii in range(d0):
            if ii in mode_slot:
                gi, jj = mode_slot[ii]
                tkv_slices.append(udots[gi])                             # ONE array per group, (K+C)+(nD, N)
                up_slices.append(shared_data.centers[gi][jj])            # H_i replaces U_i, (C)+(rL, nD, rR)
                block_masks.append(down_mask_v[ii])
            else:
                v = tkv[ii]
                tkv_slices.append(xnp.concatenate(
                    [v, xnp.zeros(v.shape[:-2] + (pad, v.shape[-1]))], axis=-2))
                U = up_sc[ii]
                up_slices.append(xnp.concatenate(
                    [U, xnp.zeros(U.shape[:-2] + (pad, U.shape[-1]))], axis=-2))
                block_masks.append(np.concatenate(
                    [var_up_mask[ii], np.zeros(var_up_mask[ii].shape[:-1] + (pad,), bool)], axis=-1))
        tkv = xnp.stack(tkv_slices, axis=0)                 # (d,)+K+C+(nD, N)
        up_sc = xnp.stack(up_slices, axis=0)                # (d,)+C+(rL, nD, rR)
        var_up_mask = np.stack(block_masks, axis=0)         # (d,)+K+C+(nD,), HOST

    d  = down_sc.shape[0]
    nD = down_sc.shape[-2]; N = down_sc.shape[-1]; nU = up_sc.shape[-2]
~~~

Also in this function, by tier E: `:196` "``[U ; V]``" → "``[D ; V]``" and `:201` "``tucker_mask = concat([up,
down])``" → "``concat([down, up])``" (the frame's down mask, then the variations' up mask).

The physical order holds by construction: `frame_data[5][0]` keeps index 0, the Tucker mask, and its name now says
"down"; `pad = nD_pad - nU_pad` is the Tucker height minus the up height, the same number as before. `udots` stays: it is
lowercase, so not a `U`-family name for K8, and no list has it.

**What could go wrong.**

1. **Re-indexing the mask** (`frame_data[5][1]`, "up is index 1 now"). The index is a NUMBER token, so K15 and the
   bijection are blind. Only the battery would see it, and B5 does not reach this code (item 5 below).
2. **"Fixing" the width comments by adjacency.** `(rL, nD, rR)` on `up_slices` and `up_sc`, and `(nD,)` on `var_up_mask`,
   are right: the tied branch rebuilds the up slot at the down (Tucker) height, zero-padding `nU → nD`, because the
   companion's centers `H_i` are at the group factor's width. §4A(f) and D-8 say "an edge touching an up core is `nU`",
   so a reviewer applying the rule turns three correct comments wrong. K13 does not parse assignments. The script's swap
   gets them right; the plan's review rule does not. Not flagged by the plan (§12 item 7).
3. **`Udot`'s letter.** K8 matches `Udot` (`:210`, `:213`), so it must change; `utv_to_ut3` is on no §1.2a list, so §3.3
   forbids choosing. I wrote `Bdot`, following the enumerated code row for the same object
   (`fv_share_tucker_variations`). This is §11.
4. **`UP` in capitals.** K9 checks `up`, `down`, `Up`, `Down`. `:213` and `:215` are the only capitalized store words in
   the tree, and K9 cannot see them. inv 2 has their rows, so tier E applies them; no gate backs them.
5. **The battery does not reach this code.** B5 lists "`utv_operations.utv_orthogonal_gauge_projection(…, shared_data=…)`
   (the `up_mask_frame`, `up_mask_v`, `down_slices` code, `utv_operations.py:230–254`)", but that code is in
   `utv_to_ut3`, reached by `utv_retract(…, shared_data=…)` (`:351`); `utv_orthogonal_gauge_projection`'s shared path
   returns `ufv_share_tucker_variations` (`:473–474`). And `xG`'s default padded widths are equal (measured: Tucker 6,
   TT-shaped 6, though the stores are (6, 6, 2) and (3, 6, 2)), so `pad = 0` and a slip in `pad` is invisible even when
   called.

---

## 6. A dispatch-test row that passes `update=`, and the recorded schedule

**Where:** `tests/test_dispatch.py:2004–2026` (`test_align_padded_frame_dispatch`: its docstring and cases), `:2058`
(a docstring), `:2416` (`test_the_step_is_chosen_by_use_jit`: the recorded schedule).

**Governs:** §4.5 (R3: LETTER-API rows swapped letter by letter and split; "The recorded schedules `('ULRO', t)` become
`(('D', 'U', 'L', 'R'), t)`"); §1.3 (`_update_stores`, `_STORE_LETTERS`, `_COLOURS`; the loops pass `_STORE_LETTERS` and
`_COLOURS[0]`); tier E (inv 1 rows `:2008`, `:2020`, `:2058`, `:2345`, `:2395`, `:2424`); K3, K15; §5.4 (the dispatch rows;
`_STEP_STATICS`).

**BEFORE**

*`tests/test_dispatch.py:2004–2008, 2017–2026`*
~~~python
    def test_align_padded_frame_dispatch(self):
        """One alignment step: numpy in -> numpy out, and jax in jits (ranks, sharing and update static) to pure jax
        that agrees with numpy (x64 on in this module): the movement directly, the frames gauge-invariantly
        through the dense tangent projection P_{T_F} Y of a fixed Y (a new column's sign follows its SVD's, which is
        backend-dependent). Generic hints, so the content determines the subspaces. A truncating step, a shared
        two-colour half (R and O transported), the zero tensor's empty frame, and a stack C = (2,)."""
        ...
        cases = {   # old frame, its ranks, working ranks, output ranks, sharing, update
            'truncating': (bvf.t3_orthogonal_representations(x_old)[0].data, ((2, 2, 2, 2), (1, 2, 2, 2, 1)),
                           ((3, 4, 4, 3), (1, 3, 4, 3, 1)), ((3, 3, 3, 3), (1, 3, 3, 3, 1)), None, 'ULRO'),
            'shared, UL': (bvf.t3_orthogonal_representations(x_tied)[0].data, ((2, 2, 2, 2), (1, 2, 2, 2, 1)),
                           ((3, 3, 3, 3), (1, 3, 4, 3, 1)), (None, None), (0, 1, 0, 1), 'UL'),
            'empty':      (padded_frames.fv_empty_frame((4, 5, 6, 3)), ((0, 0, 0, 0), (1, 0, 0, 0, 1)),
                           ((2, 3, 3, 2), (1, 2, 3, 2, 1)), (None, None), None, 'ULRO'),
            'C = (2,)':   (bvf.t3_orthogonal_representations(x_stacked)[0].data, ((2, 2, 2, 2), (1, 2, 2, 2, 1)),
                           ((3, 4, 4, 3), (1, 3, 4, 3, 1)), ((3, 3, 3, 3), (1, 3, 3, 3, 1)), None, 'ULRO'),
        }
~~~
*`:2058`* and *`:2416`*
~~~python
        every leaf -- two oversampled passes, the truncating pass and one 'UL' / 'RO' pair (tol 0, max_passes 5), at
                self.assertEqual(seen, [(u, on_jax) for u in ('ULRO', 'ULRO', 'ULRO', 'UL')])
~~~

**AFTER (R3 for the values; R4a for the docstrings)**
~~~python
    def test_align_padded_frame_dispatch(self):
        """One alignment step: numpy in -> numpy out, and jax in jits (ranks, sharing and update static) to pure jax
        that agrees with numpy (x64 on in this module): the movement directly, the frames gauge-invariantly
        through the dense tangent projection P_{T_F} Y of a fixed Y (a new column's sign follows its SVD's, which is
        backend-dependent). Generic hints, so the content determines the subspaces. A truncating step, a shared
        two-colour half (right and up transported), the zero tensor's empty frame, and a stack C = (2,)."""
        ...
        cases = {   # old frame, its ranks, working ranks, output ranks, sharing, update
            'truncating': (bvf.t3_orthogonal_representations(x_old)[0].data, ((2, 2, 2, 2), (1, 2, 2, 2, 1)),
                           ((3, 4, 4, 3), (1, 3, 4, 3, 1)), ((3, 3, 3, 3), (1, 3, 3, 3, 1)), None, ('D', 'U', 'L', 'R')),
            'shared, DL': (bvf.t3_orthogonal_representations(x_tied)[0].data, ((2, 2, 2, 2), (1, 2, 2, 2, 1)),
                           ((3, 3, 3, 3), (1, 3, 4, 3, 1)), (None, None), (0, 1, 0, 1), ('D', 'L')),
            'empty':      (padded_frames.fv_empty_frame((4, 5, 6, 3)), ((0, 0, 0, 0), (1, 0, 0, 0, 1)),
                           ((2, 3, 3, 2), (1, 2, 3, 2, 1)), (None, None), None, ('D', 'U', 'L', 'R')),
            'C = (2,)':   (bvf.t3_orthogonal_representations(x_stacked)[0].data, ((2, 2, 2, 2), (1, 2, 2, 2, 1)),
                           ((3, 4, 4, 3), (1, 3, 4, 3, 1)), ((3, 3, 3, 3), (1, 3, 3, 3, 1)), None, ('D', 'U', 'L', 'R')),
        }
~~~
~~~python
        every leaf -- two oversampled passes, the truncating pass and one ('D', 'L') / ('U', 'R') pair (tol 0,
        max_passes 5), at
                self.assertEqual(seen, [(u, on_jax) for u in (('D', 'U', 'L', 'R'),) * 3 + (('D', 'L'),)])
~~~
(`:2058` rewrapped with the docstring's next line. The `'truncating'` and `'C = (2,)'` value lines grow to 121
characters; the file already has 22 lines over 120.) `update` stays a hashable jit static as a tuple, and the step
canonicalizes it, so the `jax.jit(…, static_argnames=(…, 'update'))` call is unchanged.

**What could go wrong.**

- **Split order vs canonical order.** `'ULRO'` split letter by letter is `('D', 'L', 'R', 'U')`. In the cases it works
  (the step canonicalizes; each order would be its own jit static, harmless here). At `:2416` it fails, because `seen`
  records what the loop passes, `_STORE_LETTERS`, canonical. Caught by the suite; §4.5's `(('D', 'U', 'L', 'R'), t)`
  determines the canonical form.
- **Asserting against the module constants** (`padded_frames._STORE_LETTERS`, `_COLOURS[0]`) is a tautology: a wrong
  constant passes. inv 1's row proposed constants `ALL`, `C1`, new NAMEs K15 would flag unless R3's handoff listed them.
  Literal tuples, as above.
- **The prose rows get word tuples.** inv 1's rows give `:2058` "one ('down', 'left') / ('right', 'up') pair", and the
  same for `:2345`, `:2395`, `:2424`. §1.4's layering has no step for D-2(c), K3 matches in code files only strings
  holding `'O'`, and no other gate reads them, so the literal plan writes word tuples that name no value the code
  accepts. The plan also does not say whether R3 (which owns LETTER-API rows; inv 1 marks these `LA`) or R4a (which owns
  tier-E prose) converts them. If R3 does, R4a's locator fails loudly on the changed lines, which is the safe outcome.
  Not determined (§12 item 3).
- **The case label.** inv 1's row gives `'shared, down+left'`, which breaks the dict's value column by eight characters
  (§4 item 10 would then realign the whole dict). `'shared, DL'` keeps the column and matches `('D', 'L')` below it, but
  it is not the row's text. Left as `'shared, UL'` it now reads "up and left", and K3 cannot see it (no `'O'`). A small
  call for the reviewer.
- **`:2008`, "(R and O transported)"**: no K8 context matches "R and O", so a missed rewrite passes every gate; inv 1's
  row ("right and up") is the only guard.

---

## 7. `CLAUDE.md`: the canonical-orderings gotcha and the core-letter note

**Where:** `CLAUDE.md:212–214` (the first "gotcha", which agents read before writing any frame code) and `:275–276`.

**Governs:** tier A (A1 `down_tt_cores`→`up_tt_cores` in `CLAUDE.md`, R2); tier D (`(up, down, left, right)`,
`(U, O, P, Q)`); tier E (inv 3 App. A rows `:213–214`, `:275`), with §1.4 (tier E first, then tier D over every unfenced
line); §3 (T4S letters only in "`CLAUDE.md`'s notation map", `:68–69`); §6; K4, K5, K8.

**BEFORE**

*`CLAUDE.md:212–214`*
~~~md
- **Canonical core-tuple orderings (frontend takes precedence).** `TuckerTensorTrain.data =
  (tucker_cores, tt_cores)`; `T3Frame.data = (up, down, left, right) = (U, O, P, Q)` (the `O`/down core
  is called **`down_tt_cores`**, not "outer"); `T3Variations.data = (tucker_variations, tt_variations)`;
~~~
*`CLAUDE.md:275–276`*
~~~md
- Uppercase single-letter core names (`U V G P Q O …`) are intentional — ignore "should be
  lowercase" linter notes.
~~~

**After R2 (interim):** "(the `O`/down core is called **`up_tt_cores`**, not "outer")". Wrong until R5.

**Literal plan (R5).** Tier E writes inv 3 App. A's text, "`T3Frame.data = (down, up, left, right)`, our `(D, U, L,
R)`, T4S's `(U, O, P, Q)` (the up core is **`up_tt_cores`**; T4S says "outer")"; then tier D runs over the line and
turns `(U, O, P, Q)` into `(D, U, L, R)`:
~~~md
  (tucker_cores, tt_cores)`; `T3Frame.data = (down, up, left, right)`, our `(D, U, L, R)`, T4S's `(D, U, L, R)` (the
  up core is **`up_tt_cores`**; T4S says "outer"); `T3Variations.data = (tucker_variations, tt_variations)`;
~~~
K4 then passes, and the line misstates T4S. Without tier D, K4 fails it, because `:213` is not one of §3's T4S places.

**AFTER (recommended, R5)**
~~~md
- **Canonical core-tuple orderings (frontend takes precedence).** `TuckerTensorTrain.data =
  (tucker_cores, tt_cores)`; `T3Frame.data = (down, up, left, right) = (D, U, L, R)` (the `U`/up core
  is called **`up_tt_cores`**, not "outer"); `T3Variations.data = (tucker_variations, tt_variations)`;
~~~
~~~md
- Uppercase single-letter core names (`D U L R B G V H …`) are intentional — ignore "should be
  lowercase" linter notes.
~~~

The T4S tuple stays in the notation map (`:68–69`, which gets §3's line). Line lengths are unchanged to within two
characters, so nothing below rewraps.

**What could go wrong.**

- **Tier D over tier E** (above): tier D's patterns are not idempotent on text that cites T4S's tuple on purpose. Not
  prevented; §12 item 5.
- **`:275`'s letter list.** inv 3 proposes "(`D U L R V H G P Q O …`)". After the rename `P`, `Q` and `O` name no core in
  code (`P` the completion and the direction, `Q` QR's, `O` the oracle module alias), and `B` does (tier C's B rows).
  An agent would read `P Q O` as live core letters. K8 does not see a letter inside one backtick span. Not determined.
- **`:182`** ("disjoint from the core/variation symbols `U`/`P`/`Q`/`O`/`G`/`H`/`V`", a history sentence) holds a
  backticked `` `O` ``, which K8 fails; inv 3's row ("add `D`, `L`, `R`") keeps it. It needs the current set
  (`D`/`U`/`L`/`R`/`B`/`G`/`V`/`H`) or a carve-out.
- **No line tells an agent that `nU` and `nD` swapped meaning.** §6 states the mapping once (the convention paragraph and
  the glossary). Agents write shape comments from memory of the old tree, and K13 checks only store-named entries. A
  pointer here ("widths: `nD` the down store's, `nU` the up store's; see `docs/frame_variations.md` § Four rank stores")
  would cost one clause. Not in the plan; Nick's call.

---

## 8. A user-guide ASCII diagram and its core list, rewritten whole

**Where:** `docs/user_guide.rst:125–133` (the tangent diagram; every line starts with a TAB) and `:139–142` (the
frame-core list).

**Governs:** §4A (seed "`user_guide.rst:124–142`"; one inventory block per diagram, so R1 splits it into `125–133` and
`139–142`); K10 (raw text, tabs not expanded) for the diagram, K10b for the list; K12; tier E (inv 3 §5.3 `:127`,
`:130`, `:132`, `:139`, `:142`); §4.3 (the backwards adjective at `:142`); K8.

**BEFORE**
~~~text
	      1--H0--R1--R2--1   1--L0--H1--R2--1   1--L0--L1--H2--1
	         |   |   |          |   |   |          |   |   |
	v  =     U0  U1  U2    +    U0  U1  U2    +    U0  U1  U2
	         |   |   |          |   |   |          |   |   |

	      1--O0--R1--R2--1   1--L0--O1--R2--1   1--L0--L1--O2--1
	         |   |   |          |   |   |          |   |   |
	   +     V0  U1  U2    +    U0  V1  U2    +    U0  U1  V2
	         |   |   |          |   |   |          |   |   |

  - up_tucker_cores   = (U0,...,U(d-1)), orthogonal
  - left_tt_cores     = (L0,...,L(d-1)), left-orthogonal
  - right_tt_cores    = (R0,...,R(d-1)), right-orthogonal
  - down_tt_cores     = (O0,...,O(d-1)), down-orthogonal
~~~

**AFTER (R5)**
~~~text
	      1--H0--R1--R2--1   1--L0--H1--R2--1   1--L0--L1--H2--1
	         |   |   |          |   |   |          |   |   |
	v  =     D0  D1  D2    +    D0  D1  D2    +    D0  D1  D2
	         |   |   |          |   |   |          |   |   |

	      1--U0--R1--R2--1   1--L0--U1--R2--1   1--L0--L1--U2--1
	         |   |   |          |   |   |          |   |   |
	   +     V0  D1  D2    +    D0  V1  D2    +    D0  D1  V2
	         |   |   |          |   |   |          |   |   |

  - down_tucker_cores = (D0,...,D(d-1)), down-orthogonal
  - up_tt_cores       = (U0,...,U(d-1)), up-orthogonal
  - left_tt_cores     = (L0,...,L(d-1)), left-orthogonal
  - right_tt_cores    = (R0,...,R(d-1)), right-orthogonal
~~~

**The diagram passes K10 in place**, checked mechanically on raw text: 9 lines, equal lengths, and every differing
column holds a label character on both sides (`U→D` on the Tucker rows, `O→U` on the chain). The TAB is kept.

**The list is a K10b re-layout.** It is reordered to the tuple order (down, up, left, right), as inv 3's `:142` row asks,
and "orthogonal" grows to "down-orthogonal" (the form batch 1's approved `T3Frame` docstring uses). The `=` column (22)
and the `(` column (24) map to themselves. Derivations (§4A d): `Dᵢ` is `(nᵢ, Nᵢ)` with orthonormal rows, summed over
the leaf index below the edge, so down-orthogonal; `Uᵢ` is `(rᵢ, nᵢ, rᵢ₊₁)`, summed over both bonds above the edge, so
up-orthogonal. `:142`'s "down-orthogonal" on the TT-shaped core was the backwards adjective.

**What could go wrong.**

- **Sequential replacement.** `U→D` then `O→U` is right; `O→U` then `U→D` turns the chain's new `U0` into `D0`. Hand
  only; K8 catches a leftover `O0`, but nothing catches a leftover `U0` on the Tucker row (a valid letter). The
  reviewer's read is the guard.
- **Tabs expanded by an editor**: K10 compares raw text, so the lengths differ and it fails. Caught.
- **Treating `124–142` as one block**, as the seed list writes it: rewriting only the list satisfies K12 and leaves the
  diagram backwards. §4A(a)'s one-block-per-diagram rule prevents it, if R1 splits the seed.
- `:97`'s "Outer TT core unfoldings are full rank" is a separate tier-E row (work-list `:300–301`), not part of either
  block.

---

## 9. A travel word and a store word in one identifier: `_u_down_messages`, and "up edges"

**Where:** `t3toolbox/backend/uniform_padded_frames.py:368–393` (the two message functions), their call sites `:550`,
`:561`, the step comments `:563` and `:592`, and the letters beside them (`:564–566`, `:599–602`).

**Governs:** A2 (`_u_up_messages`↔`_u_down_messages`; §1.5 item 12: `_u_` names change only by the token map); tier C
`uniform_padded_frames.py` (`U→D, O→U, PhiO→PhiU, PhiU→PhiD, XiO→XiU, XiU→XiD, pinO→pinU, rowO→rowU, selO→selU,
mvO→mvU`; D-5 `DL→PsiL, DR→PsiR`); R3 (`'U' in update` → `'D'`, `'O'` → `'U'`); tier E (inv 2 rows `:563` `up→down` and
`:592` `down→up`, plain STORE-WORD); §4.2 (both senses: name the store **and** its arrow; four listed sites; "R1's
inventory adds any other C19 site"); §1.1 (cites `uniform_padded_frames.py:563` "up edges" as its example of a free-prose
store word); K9.

**BEFORE**
~~~python
def _u_down_messages(
        VP:  NDArray,   # (d,)+C+(Ka, N): the variations' completion rows
        U:   NDArray,   # (d,)+C+(nT, N)
) -> typ.Tuple[
    NDArray,   # PhiO. (d,)+C+(Ka, nT)
    NDArray,   # XiO.  (d,)+C+(Ka, min(N, Ka))
]:
    """The messages of the edges core -> leaf, all modes at once."""
    PhiO = contractions.contract('dCkn,dCun->dCku', VP, U)
    return PhiO, _root(VP.swapaxes(-2, -1) - contractions.contract('dCun,dCku->dCnk', U, PhiO))


def _u_up_messages(
        H_P:  NDArray,   # (d,)+C+(rT, Ka, rT): the tangent cores at the Tucker completion slots
        bL:   NDArray,   # (d,)+C+(rT, Ka, Kb)
        bR:   NDArray,   # (d,)+C+(Kb, Ka, rT)
        O:    NDArray,   # (d,)+C+(rT, nT, rT)
        PhiL: NDArray,   # (d,)+C+(Kb, rT): edge c_i -> c_(i+1); zero at d-1
        XiL:  NDArray,   # (d,)+C+(Kb, Kb)
        PhiR: NDArray,   # (d,)+C+(Kb, rT): edge c_i -> c_(i-1); zero at 0
        XiR:  NDArray,   # (d,)+C+(Kb, Kb)
) -> typ.Tuple[
    NDArray,   # PhiU. (d,)+C+(Ka, nT)
    NDArray,   # XiU.  (d,)+C+(Ka, z)
]:
    """The messages of the edges leaf -> core, all modes at once."""
    ...
    PhiO, XiO = _u_down_messages(V[..., Na:, :], U)
    ...
    PhiU, XiU = _u_up_messages(H[..., Na:, :], L[..., Na:, Rb:], R[..., Rb:, Na:, :], O, PhiL, XiL, PhiR, XiR)

    # -- step 2, up edges: one completion per sharing group, the members' contents side by side
    Abar = U[..., :Na, :].swapaxes(-2, -1)                                       # (d,)+C+(N, Na)
    g_old, g_new = U[..., Na:, :].swapaxes(-2, -1), V[..., Na:, :].swapaxes(-2, -1)
    Y = _u_content_d(g_old, g_new, PhiU[..., Na:], XiU) if 'U' in update else g_old
    ...
    # -- down edges, all modes at once, with the incoming coordinate maps (zero past the ends)
    ...
    g_old = _unfold(_new_coordinates_d(_new_coordinates_d(O[..., Na:, :], oL, -3), oR, -1), -2)
    Abar = _unfold(_pad_trailing(O[..., :Rb, :Na, :Rb], (rT1, Na, rT1)), -2)
    Y = _u_content_d(g_old, g_new, PhiO[..., Na:], XiO) if 'O' in update else g_old
    P, _, mvO = _u_edge_d(Abar, pinO, g_old, Y, rowO, selO)
~~~

**AFTER (R2 for the names, R3 for the `update` letters; the two comments as R2 leaves them)**
~~~python
def _u_up_messages(
        VP:  NDArray,   # (d,)+C+(Ka, N): the variations' completion rows
        D:   NDArray,   # (d,)+C+(nT, N)
) -> typ.Tuple[
    NDArray,   # PhiU. (d,)+C+(Ka, nT)
    NDArray,   # XiU.  (d,)+C+(Ka, min(N, Ka))
]:
    """The messages of the edges core -> leaf, all modes at once."""
    PhiU = contractions.contract('dCkn,dCun->dCku', VP, D)
    return PhiU, _root(VP.swapaxes(-2, -1) - contractions.contract('dCun,dCku->dCnk', D, PhiU))


def _u_down_messages(
        H_P:  NDArray,   # (d,)+C+(rT, Ka, rT): the tangent cores at the Tucker completion slots
        bL:   NDArray,   # (d,)+C+(rT, Ka, Kb)
        bR:   NDArray,   # (d,)+C+(Kb, Ka, rT)
        U:    NDArray,   # (d,)+C+(rT, nT, rT)
        PhiL: NDArray,   # (d,)+C+(Kb, rT): edge c_i -> c_(i+1); zero at d-1
        XiL:  NDArray,   # (d,)+C+(Kb, Kb)
        PhiR: NDArray,   # (d,)+C+(Kb, rT): edge c_i -> c_(i-1); zero at 0
        XiR:  NDArray,   # (d,)+C+(Kb, Kb)
) -> typ.Tuple[
    NDArray,   # PhiD. (d,)+C+(Ka, nT)
    NDArray,   # XiD.  (d,)+C+(Ka, z)
]:
    """The messages of the edges leaf -> core, all modes at once."""
    ...
    PhiU, XiU = _u_up_messages(V[..., Na:, :], D)
    ...
    PhiD, XiD = _u_down_messages(H[..., Na:, :], L[..., Na:, Rb:], R[..., Rb:, Na:, :], U, PhiL, XiL, PhiR, XiR)

    # -- step 2, up edges: one completion per sharing group, the members' contents side by side
    Abar = D[..., :Na, :].swapaxes(-2, -1)                                       # (d,)+C+(N, Na)
    g_old, g_new = D[..., Na:, :].swapaxes(-2, -1), V[..., Na:, :].swapaxes(-2, -1)
    Y = _u_content_d(g_old, g_new, PhiD[..., Na:], XiD) if 'D' in update else g_old
    ...
    # -- down edges, all modes at once, with the incoming coordinate maps (zero past the ends)
    ...
    g_old = _unfold(_new_coordinates_d(_new_coordinates_d(U[..., Na:, :], oL, -3), oR, -1), -2)
    Abar = _unfold(_pad_trailing(U[..., :Rb, :Na, :Rb], (rT1, Na, rT1)), -2)
    Y = _u_content_d(g_old, g_new, PhiU[..., Na:], XiU) if 'U' in update else g_old
    P, _, mvU = _u_edge_d(Abar, pinU, g_old, Y, rowU, selU)
~~~

**The comments `:563` and `:592` (R4a).** Literal plan (inv 2's rows):
~~~python
    # -- step 2, down edges: one completion per sharing group, the members' contents side by side
    # -- up edges, all modes at once, with the incoming coordinate maps (zero past the ends)
~~~
Recommended (§4.2's form):
~~~python
    # -- step 2, the down store's edges (l_i -> c_i): one completion per sharing group, members side by side
    # -- the up store's edges (c_i -> l_i), all modes at once, with the incoming coordinate maps (zero past the ends)
~~~

**The identifiers are determined.** In 2026.4.0 the "down" of `_u_down_messages` was both the travel direction (the
messages run core → leaf) and the store, because stores were named for where their edges point. After the rename the
store sense wins: the messages on the up store's edges are `_u_up_messages`, though they travel down. The docstrings
keep their arrows ("core -> leaf", inv 2's `:376` NOTE), and the arrows are what disambiguate. `_u_up_messages` takes
`D` and returns `PhiU`: messages are named for their edge's store and computed through the far side's frame, here the
down core. That reads like a mismatch and is right.

**What could go wrong.**

- **The comments get the form §4.2 exists to prevent.** The word swap matches `_down_edge` literally and reads backwards
  as a direction ("down edges" are edges pointing down, which are the up store's). inv 2's rows are plain swaps, and
  §1.1 holds up `:563` as the model free-prose row, so the literal plan writes "step 2, down edges". K9 passes it
  (clause (b)). Batch 2 fixed one such site (`test_contractions_interpreter.py:363`, now on §4.2's list) but not its
  siblings: `:563`, `:592` here, and `tests/backend/test_padded_frame.py:562` ("… a nonzero old point's down frame spans
  the whole far side and the up edge's content vanishes", inv 1: "… the down edge's content"). Not prevented (§12
  item 6). The docs twins are covered: inv 3's rows for `padded_frame_cone.tex:310, 314–315` already name store and
  arrow.
- **Flipping the docstring arrows** to "match" the new names ("leaf -> core" on `_u_up_messages`). The arrow is the fact;
  inv 2 marks the docstrings keep, and K9 sees no store word. Reviewer only.
- **"Fixing" the far-frame argument** (`_u_up_messages(V[..., Na:, :], U)` "because up messages use the up core").
  K15 catches it (an unlisted NAME change), and the battery's B7 would differ bitwise.

---

## 10. The eleventh of batch 1 revisited: `n_{D,i}` → `n_{U,i}` in `shared_t3_math.tex`, by contraction sizes

**Where:** `docs/shared_t3_math.tex:344–351` (`prop:tilt`), `:356–370` (the "No complement basis" remark), `:403–414`
(`rem:svdsolve`), `:482–492` (`rem:embedding`): the note's four `n_{D,i}` (K16's whole list for this note) and the letters
on the same lines.

**Governs:** §4.8's row for this note (in the tangent sections `:318–420`, `486–491` the frame's down store and its
directions → `D_i`, `D_g`, `\dot D`; the tied coordinates `B` → `Z`; `n_{D,i}` → `n_{U,i}`; `O_g` kept); §4A(a) (every
`.tex` width form inventoried, one block per equation); §4A(f) (sizes agree across every contraction); K16 (sign-off);
K2, K8 (`\dot U`, `U_g`; `O_g` carved out); R5; Nick's PDF review.

**BEFORE**

*`docs/shared_t3_math.tex:347–350, 356–370`*
~~~tex
a single \emph{gauged ambient direction} $\dot U \in \R^{N_g \times n_g}$
with $U_g^\top \dot U = 0$: in the ($S$-absorbed) frame coordinates,
\begin{equation}\label{eq:tied}
C_i = \dot U\, S_i \quad \text{for a common gauged } \dot U, \qquad i \in g .
...
\begin{remark}[No complement basis; rectangular $S_i$]
An earlier draft stated \eqref{eq:tied} through an orthonormal complement
$O_g$ of $U_g$ ($\dot U = O_g B$, coordinates $C_i = B S_i$), suggesting a
complement must be constructed once per group and tied across it. The
subspace depends only on $U_g$ (tied by hypothesis) and the $S_i$; no
complement basis is ever materialized (nor is one materialized anywhere in
T3Toolbox), and the per-mode rotational freedom in the factorization
$\Gamma_i = S_i S_i^\top$ (replacing $S_i$ by $S_i Q_i$ together with the
correspondingly rotated orthonormal core) cancels identically in
\eqref{eq:tied} and in the projection \eqref{eq:gramsolve}. Note also that in
implementation $S_i \in \R^{n_g \times n_{D,i}}$ with
$n_{D,i} = \min(n_g,\ r_{i-1} r_i)$ --- rectangular in general, so an
individual Gram $\Gamma_i$ may be singular at a perfectly smooth shared
point; only the group sum is invertible there
(Proposition~\ref{prop:kappa}(iii)).
~~~
*`:405–406, 412–414`*
~~~tex
M_g = \begin{bmatrix} S_{i_1}^\top \\ \vdots \\ S_{i_k}^\top \end{bmatrix}
\in \R^{(\sum_{i\in g} n_{D,i}) \times n_g},
...
$\dot U^\top = M_g^{+}\, C^{\mathrm{stack}}$ (then
$C_i^\top \leftarrow S_i^\top \dot U^\top$, the corresponding row block of
$M_g \dot U^\top$), computed from one thin SVD of $M_g$ with a relative
~~~
*`:489–492`*
~~~tex
per-mode factor blocks $[\,U_g \,|\, \dot U S_i\,]$ (the $S$-absorbed
coordinates) is \emph{not} tied: the blocks differ across the group in value
and, when the $n_{D,i}$ differ, even in shape, and no per-mode averaging
repairs this. $\dot U$ is recovered from tied coordinates by the
~~~

**AFTER (R5)**
~~~tex
a single \emph{gauged ambient direction} $\dot D \in \R^{N_g \times n_g}$
with $D_g^\top \dot D = 0$: in the ($S$-absorbed) frame coordinates,
\begin{equation}\label{eq:tied}
C_i = \dot D\, S_i \quad \text{for a common gauged } \dot D, \qquad i \in g .
...
\begin{remark}[No complement basis; rectangular $S_i$]
An earlier draft stated \eqref{eq:tied} through an orthonormal complement
$O_g$ of $D_g$ ($\dot D = O_g Z$, coordinates $C_i = Z S_i$), suggesting a
complement must be constructed once per group and tied across it. The
subspace depends only on $D_g$ (tied by hypothesis) and the $S_i$; no
complement basis is ever materialized (nor is one materialized anywhere in
T3Toolbox), and the per-mode rotational freedom in the factorization
$\Gamma_i = S_i S_i^\top$ (replacing $S_i$ by $S_i Q_i$ together with the
correspondingly rotated orthonormal core) cancels identically in
\eqref{eq:tied} and in the projection \eqref{eq:gramsolve}. Note also that in
implementation $S_i \in \R^{n_g \times n_{U,i}}$ with
$n_{U,i} = \min(n_g,\ r_{i-1} r_i)$ --- rectangular in general, so an
individual Gram $\Gamma_i$ may be singular at a perfectly smooth shared
point; only the group sum is invertible there
(Proposition~\ref{prop:kappa}(iii)).
~~~
~~~tex
M_g = \begin{bmatrix} S_{i_1}^\top \\ \vdots \\ S_{i_k}^\top \end{bmatrix}
\in \R^{(\sum_{i\in g} n_{U,i}) \times n_g},
...
$\dot D^\top = M_g^{+}\, C^{\mathrm{stack}}$ (then
$C_i^\top \leftarrow S_i^\top \dot D^\top$, the corresponding row block of
$M_g \dot D^\top$), computed from one thin SVD of $M_g$ with a relative
~~~
~~~tex
per-mode factor blocks $[\,D_g \,|\, \dot D S_i\,]$ (the $S$-absorbed
coordinates) is \emph{not} tied: the blocks differ across the group in value
and, when the $n_{U,i}$ differ, even in shape, and no per-mode averaging
repairs this. $\dot D$ is recovered from tied coordinates by the
~~~

`D` and `Z` are otherwise unused in the note (its only `D` today is in `n_{D,i}`, plus an author initial), `Q_i` at
`:363` is a rotation and stays (§1.5 item 2), and `O_g` stays.

**The contraction-size check (the reviewer's, §4A f), written out.** The note writes factors `N × n` (columns
orthonormal). Sizes at a group mode `i ∈ g`: `D_g, \dot D ∈ R^{N_g × n_g}`; the TT core's up-matricization `W_i ∈
R^{n_g × r_{i-1} r_i}` (`:79`); the up core's matricization `Ū_i ∈ R^{n_{U,i} × r_{i-1} r_i}` with orthonormal rows.

1. `S_i = W_i Ū_iᵀ` (the code's `S_iᵀ = ⟨U_i, H_i⟩`): `(n_g × r_{i-1}r_i)(r_{i-1}r_i × n_{U,i})`, contracted over
   `r_{i-1} r_i` on both sides, gives `n_g × n_{U,i}`. Matches `:366`.
2. `n_{U,i}` = the rank of `W_i` ≤ `min(n_g, r_{i-1} r_i)`, equal at a generic point, and `Ū_i` can have that many
   orthonormal rows only because `n_{U,i} ≤ r_{i-1} r_i`. Matches `:367`.
3. `Γ_i = S_i S_iᵀ`: contracted over `n_{U,i}`, gives `n_g × n_g`, and equals `W_i W_iᵀ` (`:93`) since `Ū_i Ū_iᵀ = I`.
4. `C_i = \dot D S_i` (eq:tied): `(N_g × n_g)(n_g × n_{U,i})`, contracted over `n_g`, gives `N_g × n_{U,i}`; transposed,
   the code's Tucker variation `(nUᵢ, N)`, whose rows are `nU` by D-8.
5. eq:gramsolve: `Σ_i C_i S_iᵀ` is `(N_g × n_{U,i})(n_{U,i} × n_g)`, contracted over each `n_{U,i}`, so `N_g × n_g`;
   times `(Σ Γ_i)^{-1}`, `n_g × n_g`, gives `\dot D`, `N_g × n_g`. Consistent.
6. `M_g` stacks the `S_iᵀ`, each `n_{U,i} × n_g`, so it is `(Σ n_{U,i}) × n_g` (`:406`); `C^stack` stacks the `C_iᵀ`, so
   `(Σ n_{U,i}) × N_g`; `M_g^+ C^stack` is `(n_g × Σ n_{U,i})(Σ n_{U,i} × N_g)`, contracted over `Σ n_{U,i}`, so
   `n_g × N_g = \dot Dᵀ`. Consistent.
7. rem:embedding: the block `\dot D S_i` is `N_g × n_{U,i}`, so the blocks "differ … when the `n_{U,i}` differ, even in
   shape" (`:491`). Consistent.

With the label left as `n_{D,i}` (now the down store's width), every one of these breaks visibly: the down width at a
group mode is `n_g` itself (`D_g` has `n_g` columns), so `S_i` would be square, contradicting "rectangular in general";
`n_{D,i} = min(n_g, r_{i-1}r_i)` would be false whenever `r_{i-1} r_i < n_g`; `Ū_i` could not have `n_g` orthonormal rows
in `r_{i-1} r_i < n_g` columns; and "when the `n_{D,i}` differ" would be vacuous inside a group. The code agrees after R2:
`sharing.py:557`'s `# S_i^T, C+(nDi, nU)` becomes `C+(nUi, nD)`, `:510`'s `C + (sum_i nD_i, n_g)` becomes
`(sum_i nU_i, n_g)`, and `:704`'s `(K+C)+(sum nD, N)` becomes `(sum nU, N)`.

**What could go wrong.**

- **A partial edit** (`:366–367` done, `:406` or `:491` missed) reads plausibly either way. K16 lists all four forms for
  sign-off, and the check above fails on any one left. Determined and checkable, which batch 1's §11 asked for.
- **Ordering of the letter conversions.** A note-wide `U→B` regex for the point's own factors (§4.8, right in §2's point
  algorithms) followed by `B→Z` turns `U_g` here into `Z_g`. Hand edits with sentinels; the tangent sections take `D`.
- **`Q_i` at `:363` against K8.** K8's `\b[PQ]_(?:i|last|\{)` matches it. It is a rotation, so §3.4's inventory must
  classify it as kept, or K8 fails a correct line.
- **`:472` and `:484` vs `:489`.** §4.8's tangent ranges are `:318–420` and `486–491`; `:472` and `:484` are "classified
  symbol by symbol". Read literally, `:484`'s doubled factor `[U_g | \dot U]` takes the point's own `B_g` and `:489`'s
  `[U_g | \dot U S_i]` takes the frame's `D_g`: one factor, two letters, five lines apart in one remark. I would write
  `D_g` throughout `:469–495`, leaning on the notation paragraph's "at a point in orthonormal representation `D_i` is
  `B_i`". Not determined (§12 item 13).
- **The note and the code name the same direction differently**: the note's `\dot D` here is the code's `Udot` in
  `sharing.py`, which §1.2a renames `Bdot`. That is §11.

---

## 11. Least sure: the letter of the tied ambient direction (edit 5's `Udot`)

The object: per nontrivial sharing group, the common gauged ambient direction of the group factor,
`Udot = M_g^+ [V_i1; …; V_ik]` (`sharing._tied_filtered_solve`), with `V_i = S_iᵀ Udot` on tied input. The plan gives it
different letters, or none, at its five homes:

| site | what the plan says | letter |
|---|---|---|
| `sharing.py`, `fv_share_tucker_variations` (NAME at `:635`, docstring `:586–592`) and `_tied_filtered_solve_transpose` (`:693–702`) | §1.2a, B forms: "`Udot→Bdot`, the direction on a group's factor" | `B` |
| `sharing.py`, `_tied_filtered_solve` (`:672–687`, comments) and `fv_tied_ambient_directions` (`:715–722`, docstring) | on no list; K8 matches `Udot` | none |
| `utv_operations.py`, `utv_to_ut3` docstring (`:210`, `:213`) | on no list; K8 matches; inv 2 kept `Udot` as "the T3 letter" | none |
| `shared_t3_math.tex`, `prop:tilt`, `prop:proj`, `rem:svdsolve`, `rem:embedding` (`\dot U`) | §4.8: the tangent sections' down-store directions → `\dot D` | `D` |
| `sharing.py`, `fv_tied_lift_factors` docstring (`:744–747`, `Udot_g = M_g^{+eps}[…]`, the same solve, damped) | on no list; it is the corewise lift's raw direction, which §1.2a makes `B` everywhere else | none |

Both letters follow from D-12 on different readings. **B:** the code computes this direction exactly as the corewise lift
computes the raw group direction (same solve, damping aside), and the lift is `B` (`tv_corewise_lift`,
`fv_share_tucker_variations`). **D:** it is gauged (`D_gᵀ \dot D = 0`), it lives in the tangent's coordinates
(`C_i = \dot D S_i` is the S-absorbed Tucker variation), and in the tied embedding it takes the variation slot beside
`D_g`, which is D-12's "a tangent's Tucker variation `V` … takes `D` forms".

**What I wrote in edit 5:** `Bdot`, following the one enumerated code row for this object, so the code is at least
self-consistent. **What the plan determines:** only that `utv_to_ut3`'s `Udot` must change (K8) and that the implementer
must not pick the letter (§3.3). The result as planned: one solve is `\dot D` in the note and `Bdot` in the code, and
three code sites are unassigned.

**Options for Nick:**

- (a) `B` in code and note: the note's tangent sections write `\dot B` for the tied direction, beside `D_g`.
- (b) `D` for the gauged tied direction everywhere (`fv_share_tucker_variations`, `_tied_filtered_solve*`,
  `fv_tied_ambient_directions`, `utv_to_ut3` → `Ddot`), and `B` only for the corewise lift's `Bdot_g`
  (`tv_corewise_lift`, `fv_tied_lift_factors`).
- (c) As planned, with one sentence in the shared note's notation paragraph that the code writes this direction `Bdot`.

**Recommendation: (b).** D-12's object test reads it as a tangent coordinate (gauged, in the variation slot, beside
`D_g`), and it keeps the note and the code in step. It changes one enumerated row (`fv_share_tucker_variations`), so it is
a ruling, not a reading. Whatever the choice, the three unlisted `sharing.py` sites and `utv_to_ut3` join §1.2a.

---

## 12. Plan issues surfaced

1. **Tuple callers of `update` stay silent under D-2(c)** (edit 3). 2026.4.0 accepted any collection of letters
   (measured: `('U', 'L')`, `['U', 'L']`, `('U',)` all run); form (c) accepts `('U', 'L')` with the other store. §1.3's
   "Every 2026.4.0 call fails loudly" and §2's "loud" are false for them. Fix the claim, and say it in the release notes'
   table (as drafted in edit 3).
2. **The pointed `update` message fires only for a `str` holding `'O'`** (edit 3). `'UL'` gets the bare message, and its
   obvious fix `('U', 'L')` is accepted and wrong. Add the sentence for every `str`.
3. **No layering step for D-2(c) in prose** (edits 2, 6). inv 1 and inv 3 propose word tuples (`('down', 'left')`,
   `('right', 'up')`) for prose rows: `test_dispatch.py:2058, 2345, 2395, 2424`, `glossary.md:220–222, 366–367`,
   `aligned_padded_frame.md:60–66, 138`, `padded_frame_internals.md:46, 116, 122`, `padded_frame_cone.tex:384, 387, 412`,
   and the work list's N4 row. K3 bans only `'ULRO'|'UL'|'RO'` in docs and `'O'`-holding strings in code. Add L7 (word
   tuples → letter tuples; `('right', 'up')` → `('U', 'R')`), extend K3 to `\('(?:down|up|left|right)'` in docs and
   docstrings, and say whether R3 or R4a converts prose LA rows.
4. **The work list is ungated, and its consumer runs after R6** (edit 2). inv 3 §7's re-expressions carry `n_up_1`,
   `n_down_i`, `r_left[d]`, `update=` "store words"; `dev/` is outside §5.3; the 601-row pass applies the rows to the docs
   after the last K-check. Run K2/K3 over the work list's target column at R5, re-run K2–K9 after the docs pass, and
   re-express the "current" column too.
5. **Tier D is not idempotent on text that cites T4S's tuple** (edit 7). It runs after tier E, so inv 3 App. A's
   `CLAUDE.md:213` text "T4S's `(U, O, P, Q)`" becomes "T4S's `(D, U, L, R)`". Drop the T4S tuple there (the notation map
   carries the line), or exempt T4S citations from tier D.
6. **Both-senses sites in code are not on §4.2's list** (edit 9): `uniform_padded_frames.py:563, 592` and
   `tests/backend/test_padded_frame.py:562`. Their rows are plain word swaps, and §1.1 cites `:563` as the model
   free-prose row. Add them to §4.2 and fix §1.1's example.
7. **`utv_to_ut3`'s width comments contradict the adjacency rule** (edit 5). The tied branch pads the up slot to the
   down height, so `(rL, nD, rR)` and `(nD,)` on up-store names are right. Note the exception in §4A(f), or reword the
   comments ("padded to the down width `nD`").
8. **Battery B5 names the wrong function and a pad-free fixture** (edit 5). The cited code is in `utv_to_ut3`, reached by
   `utv_retract(…, shared_data=…)`; and `xG`'s padded widths are equal, so `pad = 0`. Call `utv_to_ut3(…, shared_data=…)`
   on a uniform frame of `xG` with `nD ≠ nU` (e.g. `from_t3frame(F, None, 7, 6)`, `pad = 1`).
9. **The tied direction's letter** (§11): `fv_tied_ambient_directions`, `_tied_filtered_solve`, `fv_tied_lift_factors`
   (text) and `utv_to_ut3` are on no list though K8 forces them, and the code (`Bdot`) and the note (`\dot D`) disagree.
   Needs Nick's ruling.
10. **K9 misses capitals.** It checks `up`, `down`, `Up`, `Down`; `utv_operations.py:213, 215` write `UP`. Add `UP`/`DOWN`.
11. **Line-based carve-outs meet hard-wrapped prose** (edits 1, 2). The glossary's "(2026.4.0" lines and the T4S
    correspondence lines are skipped per physical line; a wrapped alias list or T4S note exposes its continuation to K1
    and K8. State that each carved-out group stays on one line, or carve out by block. Separately, `glossary.md:102`'s
    alias "outer-orthogonal" (N17's own) fails K8's `outer[- ]orthogonal` with no carve-out.
12. **"These do not raise" overclaims for the width keywords** (edit 3): a swapped width below the other store's ranks
    raises numpy's `index can't contain negative values` (measured). Say "need not raise".
13. **`shared_t3_math.tex`'s tangent ranges split `rem:embedding`** (edit 10): `:484` is outside `486–491` and "classified
    symbol by symbol", so one factor can be `B_g` at `:484` and `D_g` at `:489`. Extend the range to `:469–495` with
    `D_g`.
14. **§6's `naming_conventions.md` bullet "frame stores are words"** is false under D-2(c) and D-8: `update` takes store
    letters and the widths are letters. inv 3's text for it ("Frame stores are words, never letters: … widths `n_down` …
    `update` takes store words") fails twice. Suggested text: "Frame store names are words (`down_*`, `up_*`, `left_*`,
    `right_*`); their letters `D`, `U`, `L`, `R` name them in `update`, diagrams and equations, and their widths are `nD`,
    `nU`, `rL`, `rR`."
15. **`CLAUDE.md`** (edit 7): inv 3's `:275` keeps `P Q O` and omits `B`; `:182`'s history list fails K8; and no line
    warns an agent that `nU`/`nD` swapped meaning.
16. **Small:** §5.4 says "A word tuple stays a jit static" (form (b)'s wording; it is a letter tuple). R3's gate omits
    the module doctests although R3 converts doctest LA rows (`frame_variations_format.py:1850` iterates a letter
    string, each a one-character `str` that form (c) refuses); a mis-conversion surfaces only at R4a. §2 says "the last
    three pairs are also constructor keywords"; it is four (`T3FrameWeights.up_weights`/`down_weights` are dataclass
    fields too), as edit 3's CHANGELOG text says.

---

## 13. The session's review (2026-11-11)

Reviewed edit by edit against the code, the plan (draft 18) and Nick's rulings. **Verdict on the agent's decisions:
good throughout.** Every recommended AFTER is the form I would have written; where the literal plan would produce a
wrong or ambiguous text the agent said so and gave the right form. The plan is sound in principle; §12's sixteen items
are enumeration gaps, two wrong claims and one ruling, all sent to the writer for draft 20.

| edit | verdict | notes |
|---|---|---|
| 1 user page | correct | the convention paragraph as corrected (letters `nD/nU/rL/rR`, no "notes keep T4S letters", `B` for a T3's own cores, `d` dual) is the one to ship; inv 3 §1(a) must not be copied verbatim |
| 2 glossary + work list | correct | plan fixes: carve-outs per block not per line; a carve-out for N17's own alias "outer-orthogonal"; the work list's rows (both columns) re-expressed at R5 and gated (K2/K3 over the target column), since the 601-row docs pass runs after the last K-check |
| 3 CHANGELOG + recipe | correct, two claims corrected | (i) 2026.4.0 accepted any collection of letters, so tuple callers of `update` stay silent under form (c) — the release notes say so (as drafted), the plan's "every 2026.4.0 call fails loudly" is amended; the pointed sentence goes on EVERY `str`; (ii) "these do not raise" → "need not raise; when one does the message does not name the width" (measured: a swapped width can raise numpy's negative-index error) |
| 4 test fixture + §5.2 test | correct | for the uniform classes the §5.2 assertion compares against the ragged source frame and the supercore widths (not mask-vs-ranks, a tautology) |
| 5 `utv_to_ut3` tied branch | correct | the width comments `(rL, nD, rR)` on up-store names are RIGHT there (the tied embedding rebuilds the up slot at the down height) — §4A(f)'s adjacency rule gets the exception "an embedded or rebuilt block states its actual height; verify by the code"; battery B5 must call `utv_to_ut3(…, shared_data=…)` via `utv_retract` on a fixture with `nD ≠ nU` (`pad ≠ 0`) |
| 6 dispatch rows | correct | a layering step for D-2(c) in prose (word tuples → letter tuples), K3 extended, R3 vs R4a ownership of prose LA rows stated; label `'shared, DL'` |
| 7 `CLAUDE.md` | correct | drop the T4S tuple from `:213` (the notation map carries the correspondence); tier D exempts T4S citation lines; the letter list `(D U L R B G V H …)`; `:182`'s history list updated; ADD the one-clause pointer that `nD`/`nU` swapped meaning (session decision: agents write shape comments from memory of the old tree) |
| 8 user-guide diagram | correct | K10 in place for the diagram, K10b for the reordered list; one block per diagram |
| 9 `_u_up/_u_down_messages` | correct | identifiers determined (store sense wins; the docstring arrows are the facts); the two step comments and `test_padded_frame.py:562` join §4.2's store-and-arrow list |
| 10 `shared_t3_math.tex` widths | correct | the contraction-size check as written is the reviewer's method for every `.tex` width; `Q_i` at `:363` classified kept; `rem:embedding` range extended to `:469–495` with `D_g` |
| 11 the tied direction's letter | **session ruling: option (b)** | the gauged tied ambient direction is a frame-context object (gauged against `D_g`, in the variation slot beside `D_g`) → `Ddot` / `\dot D` in `fv_share_tucker_variations`, `_tied_filtered_solve*`, `fv_tied_ambient_directions`, `utv_to_ut3`; `B` only for the corewise lift's raw direction (`tv_corewise_lift`, `fv_tied_lift_factors`' `Bdot_g`); one enumerated row changes; Nick may object |

Also adopted from §12: K9 checks `UP`/`DOWN`; `naming_conventions.md`'s bullet reads "Frame store names are words
(`down_*`, `up_*`, `left_*`, `right_*`); their letters `D`, `U`, `L`, `R` name them in `update`, diagrams and equations,
and their widths are `nD`, `nU`, `rL`, `rR`"; §5.4's "word tuple" → "letter tuple"; R3's gate runs the module doctests;
§2's "last three pairs" → four.

**Recommendation.** Three batches, thirty edits: the first found six plan defects, the second fifteen enumeration gaps,
the third sixteen, none of them a misspecified principle — every one is a missing row, a stale inventory text, a
wrong count or a wording. R1 is itself the exhaustive enumeration step (hit table, freeness check, `letter_sites`,
the diagram inventory, the battery reference) and will catch the same class mechanically. After draft 20, start R1.
