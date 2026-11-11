# Store rename inventory 3: documentation (2026-11-10)

Inventory for the breaking store rename decided in `dev/HANDOFF.md` ("DECISION (Nick 2026-11-10): THE STORE RENAME"),
partition 3 of 3: all documentation. Scope: `docs/*.md|rst` (user pages), `docs/contributor/*`, the math notes
`docs/*.tex` (and their PDFs), figures and diagrams, `README.md`, `CHANGELOG.md`, `docs/release_notes.md`, the
glossary `docs/glossary.md`, the work list `dev/glossary_naming_decisions.md`; `CLAUDE.md` as an appendix (it may
also sit in inventory 1 or 2). Working tree of 2026-11-10 at `b272e5de`. Line numbers are of that tree.

**The decision, as applied here.** A store is named for the side of its edge its cores occupy. Old → new:

| 2026.4.0 | 2026.5.0 | cores |
|---|---|---|
| up store, letter `U` | **down** store, letter `D` | the Tucker-factor-shaped cores, down-orthogonal |
| down store, letter `O` (`D` in the `T3Frame` docstring) | **up** store, letter `U` | the TT-shaped cores, up-orthogonal |
| left / right, `L` / `R` | unchanged | |
| widths `nU`, `nD` (also `nO`), `rL`, `rR` | `n_down`, `n_up`, `r_left`, `r_right` | note `nU` → `n_down`, `nD` → `n_up` |
| `update` letters `'U'`, `'O'`, `'L'`, `'R'` | words `'down'`, `'up'`, `'left'`, `'right'` | spelling of a set of words is the plan's call; written here as tuples |
| tuple `(up, down, left, right)` | `(down, up, left, right)` | positions unchanged |

In this file "old `U`" means the 2026.4.0 frame Tucker letter and "old `O`" the 2026.4.0 TT-shaped frame letter.

## 0. Method and counts

A scanner (session-local, not kept: `/tmp/claude-1000/-home-nick-repos-T3Polynomial/3eb9e6ad-d789-494e-ac17-23291348eda0/scratchpad/scan.py`) printed every line in scope that matched a
store word (`up`/`down`/`upward`/`downward` as a token or an identifier part), a width symbol (`nU nD nO rL rR`,
`n_{U,i}`, `n^D_i`, ...), a core letter (`U`, `O`, `D` as standalone symbols, `\mathbf U`, ...), an `update` string
(`'UL'`, `'ULRO'`, ...), an orthogonality adjective (`up-orthogonal`, `down_svd`, `outer`, `upward`, ...), or a T4S
citation. That is 202 / 61 / 483 / 28 / 61 / 78 candidate lines in 55 files. Every hit was read and either put in a
table below or counted as unrelated in §9 (big-O, "up to", "warm-up", jet-order shifts, decision codes `D-41`,
"Method D", the `O2` oracle, author initials, ...).

**Counts** (rows of the §5, §6 and Appendix A tables, 196 in all; a row may carry several categories; a row can
cover a line range):

| category | rows | meaning |
|---|---|---|
| STORE-WORD | 73 | a store word or store-named identifier/string that flips (incl. 10 `update`-string rows and ~8 private names that follow the code) |
| WIDTH | 30 | a width symbol (`nU`, `nD`, `nO`, `rL`, `rR`, `n_{U,i}`, `n^D_i`, ...) |
| CORE-LETTER | 43 | `U`/`O`/`D` naming a frame core in an equation, list or diagram (old `U` → `D`, old `O` → `U`) |
| ADJECTIVE | 14 | an orthogonality adjective; all stay. **Backwards in docs: `user_guide.rst:142`** (`down_tt_cores ... down-orthogonal`); glossary `:107` quotes the clash; one more in a record (`CHANGELOG.md:714`, "Tucker up-SVD") |
| FIGURE | 6 figures (8 rows) | §3: 2 ASCII diagrams to redraw, 1 PDF table to regenerate, 2 tikz figures only if their note leaves T4S letters, 1 comment that already shows the new picture |
| CONVENTION-SENTENCE | 9 | a sentence stating the old convention, quoted with a proposed rewrite (plus the §1(a) insert) |
| T4S-CITE | 21 places (§4) | a place that uses or cites T4S's letters (`U`, `O`, `P`, `Q`, `δU`, `δG`) |
| CONFLICT | 19 (C1–C19; 45 rows refer to one) | a `D` or `U` that already means something else in the same context (§2) |
| work-list rows (§7) | 51 | rows of `dev/glossary_naming_decisions.md` that mention a store word or letter |

Records left alone: 17 historical lines in `CHANGELOG.md` and 0 in `docs/release_notes.md` (§8).

## 1. Drafts for the docs to reuse

### (a) The convention, one paragraph and a picture

Proposed home: `docs/frame_variations.md` § "Four rank stores" (the section the glossary, `aligned_padded_frame.md`
and the padded-frame internals link to), right after its first paragraph. The glossary's **store** entry quotes its
first two sentences.

> **Store names.** Draw a T3 with its TT chain on top and its Tucker factors hanging below, one under each TT core:
>
> ```
>    1 --- c0 --- c1 --- c2 --- 1        TT chain: bonds r0 ... r3, each with a left and a right side
>          |      |      |
>          l0     l1     l2              Tucker factors: edges n0 ... n2, each with an up and a down side
>
>    one TT bond:      c(i) --|-- c(i+1)        left core L(i) on the left side, right core R(i+1) on the right
>    one Tucker edge:     c(i)                  up core U(i): TT-shaped, above the edge
>                         -+-
>                         l(i)                  down core D(i): the Tucker factor, below the edge
> ```
>
> Every internal edge carries two frame cores, one on each side, and each is orthonormal over the indices on its own
> side. **A store is named for the side its cores occupy**: the left store holds the cores on the left of every TT
> bond, the right store those on the right, the up store the TT-shaped cores above every Tucker edge, and the down
> store the Tucker factors below it. The orthogonality adjectives say the same thing: a left core is left-orthogonal,
> an up core up-orthogonal (`up_orthogonalize_tt_cores`), a down core down-orthogonal
> (`down_orthogonalize_tucker_cores`). Name a store by its side, never by where its edge points: as a directed edge,
> a core's edge points away from its side (a left core's edge points right, an up core's edge points down to the
> leaf). A frame keeps the stores in the order `(down, up, left, right)`, Tucker family first as in
> `TuckerTensorTrain.data`; the widths are `n_down` (the Tucker rows), `n_up`, `r_left` and `r_right`. Diagrams and
> equations use the initials `D`, `U`, `L`, `R`.

T4S correspondence note (same section, and the glossary's **T4S** entry):

> **T4S notation.** The T4S paper (Appendix A) writes a frame as `Uᵢ`, `Pᵢ`, `Qᵢ`, `Oᵢ`: the orthonormal Tucker basis,
> the left- and right-orthogonal TT cores, and the "outer orthogonal" TT-shaped core. In this library these are the
> down core `Dᵢ` (stored transposed, `(n, N)`), the left core `Lᵢ`, the right core `Rᵢ` and the up core `Uᵢ`: **the
> paper's `U` is our `D`, and our `U` is the paper's `O`**; "outer orthogonal" is up-orthogonal. The paper's
> variations `δUᵢ`, `δGᵢ` are our `Vᵢ`, `Hᵢ`, and its modes are numbered from 1. Pages that keep the paper's letters
> (the paper ↔ code maps, and the math notes written in its notation) say so where the notation is set.

### (b) CHANGELOG entry and release-notes section (house format)

House format, from 2026.2.0 / 2026.3.0: a headline paragraph under `## [version] — date` that ends with
"Upgrade notes: `docs/release_notes.md`, "Upgrading from X"."; then `### Changed — breaking` with bold-lead bullets,
"not aliased" stated with the 2026.2.0 precedent ("so a breaking release breaks loudly once rather than carrying two
names"); the release notes open the section `## Upgrading from X` with a bold one-line verdict, then bold-lead
paragraphs ordered "roughly by how likely you are to meet it", with code blocks for the migration. The root
`release_notes_2026.N.0.md` (the GitHub release body) is the CHANGELOG section without its version heading.

**CHANGELOG draft** (the headline paragraph's sentence; the release's other items are not this inventory's):

```markdown
... **The frame stores `up` and `down` swap names** (breaking; below), so that a store is named for the side its
cores occupy, as `left` and `right` already were. Upgrade notes: `docs/release_notes.md`, "Upgrading from 2026.4.0".

### Changed — breaking

- **The frame's up and down stores have swapped names: every name that kept its spelling now means the other
  store.** A store is named for the side of its edge its cores occupy, as left and right always were: the **up**
  store is the TT-shaped cores (up-orthogonal, built by `up_orthogonalize_tt_cores`), the **down** store the
  orthonormal Tucker factors (down-orthogonal, `down_orthogonalize_tucker_cores`). In 2026.4.0 the two words named
  the direction the cores' edges point, which is the other way round, and contradicted the orthogonality adjectives
  of the method names. Same spelling, other store: `up_ranks` / `down_ranks` (`T3Frame`, `T3FrameWeights`,
  `UT3Frame`, `UT3Variations`, `UT3FrameWeights`), `up_weights` / `down_weights`, `up_weight_supercore` /
  `down_weight_supercore`, `UT3FrameMasks.up_mask` / `down_mask`, `UT3VariationsMasks.variations_up_mask` /
  `variations_down_mask`, and the `up_ranks=` / `down_ranks=` keywords of `ranks.frame_has_minimal_ranks` and
  `ufv_masking.ufv_make_frame_masks`. **These do not raise**: at a minimal frame both stores have the same ranks, so
  stale code returns plausible numbers and is wrong only at slack. Changed spelling, so stale code raises:
  `T3Frame.up_tucker_cores` → `down_tucker_cores` and `down_tt_cores` → `up_tt_cores` (dataclass fields, so also
  constructor and `dataclasses.replace` keywords), `UT3Frame.up_tucker_supercore` → `down_tucker_supercore` and
  `down_tt_supercore` → `up_tt_supercore`, and the backend keywords `up_tucker_cores[_dot]=` → `down_tucker_cores[_dot]=`
  (`probing.compute_xi`, `compute_deta_tilde`, `compute_deta_tilde_tangent` and their `sampling_derivatives` jet
  twins) and `down_tt_cores[_dot]=` → `up_tt_cores[_dot]=` (`compute_eta`, `compute_sigma`, `compute_tau`,
  `compute_dxi_tilde`, `compute_dxi_tilde_tangent` and twins). Positions do not change: `T3Frame.data` and every rank,
  mask and weight tuple keep their order, which now reads `(down, up, left, right)`, so positional code and saved
  files are unaffected, and every result is bit-identical (seeded `padded_seed` / `cone_seed` streams included).
  The method names (`down_orthogonalize_tucker_cores`, `up_orthogonalize_tt_cores`, `up_svd`, ...) were already
  right and are unchanged. Not aliased, as for `GaussNewtonModel` in 2026.3.0: a meaning swap cannot be aliased.
  Old changelog entries, old docs and the T4S paper use the old words or other letters; see the release notes.
- **No store letters in the API.** `T3Tangent.align_padded`, `UT3Tangent.align_padded` and
  `tv_align_padded_frame` / `utv_align_padded_frame` take store words in `update` instead of the letters `'U'`, `'O'`,
  `'L'`, `'R'`: `'ULRO'` → <all four>, `'UL'` → `('down', 'left')`, `'RO'` → `('right', 'up')` (the two colours of
  `alternate=True` are unchanged as sets of stores). The uniform widths `nU`, `nD`, `rL`, `rR` (properties of
  `UT3Frame`, `UT3Variations`, `UT3FrameWeights`) become `n_down`, `n_up`, `r_left`, `r_right` -- `nU` was the
  Tucker rows, so it is `n_down` -- and the `nD=` keyword of `UT3Frame.from_t3frame`,
  `UT3Variations.from_t3variations`, `UT3FrameWeights.from_t3frameweights`, `UT3Tangent.from_t3tangent`,
  `ufv_conversions.t3frame_to_ut3frame` / `t3variations_to_ut3variations` / `t3frameweights_to_ut3frameweights` and
  `ufv_masking.ufv_make_frame_masks` becomes `n_up=`. A letter string raises `ValueError`, an old keyword
  `TypeError`, an old property `AttributeError`.
- `T3Frame.__repr__` and every error message that names a store use the new words.
```

**Release-notes draft** (`docs/release_notes.md`, new first section; the timing and glossary items of 2026.5.0 go
after it):

````markdown
## Upgrading from 2026.4.0

**The frame stores `up` and `down` have swapped names: a name that kept its spelling now means the other store.**
`frame.up_ranks` used to be the Tucker rows and is now the TT-shaped cores' mode legs. Such code does not raise,
and at a minimal frame it even returns the right numbers, so search for the names below rather than waiting for an
error.

**Why.** A store is now named for the side of its edge its cores occupy, as `left` and `right` always were. Draw the
TT chain on top and the Tucker factors below it: the **up** store is the TT-shaped cores above each Tucker edge
(up-orthogonal), the **down** store the Tucker factors below it (down-orthogonal). The 2026.4.0 words named where
the cores' edges point, the opposite way, and disagreed with the method names `up_orthogonalize_tt_cores` and
`down_orthogonalize_tucker_cores`, which are unchanged. See [Four rank stores](frame_variations.md#...).

| 2026.4.0 | 2026.5.0 | raises if left? |
|---|---|---|
| `T3Frame.up_tucker_cores` (field, keyword) | `T3Frame.down_tucker_cores` | yes |
| `T3Frame.down_tt_cores` (field, keyword) | `T3Frame.up_tt_cores` | yes |
| `UT3Frame.up_tucker_supercore` | `UT3Frame.down_tucker_supercore` | yes |
| `UT3Frame.down_tt_supercore` | `UT3Frame.up_tt_supercore` | yes |
| `up_ranks` (`T3Frame`, `T3FrameWeights`, `UT3Frame`, `UT3Variations`, `UT3FrameWeights`) | `down_ranks` | **no: swaps meaning** |
| `down_ranks` (same classes) | `up_ranks` | **no** |
| `T3FrameWeights.up_weights` / `down_weights` | `down_weights` / `up_weights` | **no** |
| `UT3FrameWeights.up_weight_supercore` / `down_weight_supercore` | `down_weight_supercore` / `up_weight_supercore` | **no** |
| `UT3FrameMasks.up_mask` / `down_mask` | `down_mask` / `up_mask` | **no** |
| `UT3VariationsMasks.variations_up_mask` / `variations_down_mask` | `variations_down_mask` / `variations_up_mask` | **no** |
| keywords `up_ranks=` / `down_ranks=` (`ranks.frame_has_minimal_ranks`, `ufv_masking.ufv_make_frame_masks`) | `down_ranks=` / `up_ranks=` | **no** (positional calls unchanged) |
| keyword `up_tucker_cores[_dot]=` (`probing`, `sampling_derivatives`) | `down_tucker_cores[_dot]=` | yes |
| keyword `down_tt_cores[_dot]=` (same modules) | `up_tt_cores[_dot]=` | yes |
| widths `nU`, `nD`, `rL`, `rR` (`UT3Frame`, `UT3Variations`, `UT3FrameWeights`) | `n_down`, `n_up`, `r_left`, `r_right` | yes |
| keyword `nD=` (the `from_*` converters, `ufv_make_frame_masks`) | `n_up=` | yes |
| `update='ULRO'` / `'UL'` / `'RO'`, letters `'U'`, `'O'`, `'L'`, `'R'` | words: all four / `('down', 'left')` / `('right', 'up')`; `'down'`, `'up'`, `'left'`, `'right'` | yes |

**Do not swap with two `sed` passes**: the second pass undoes the first. Swap in one pass, and only these names
(never `*_orthogonalize_*` or `*_svd*`, whose words are unchanged):

```python
import re
SWAP = {'up_tucker_cores': 'down_tucker_cores', 'down_tt_cores': 'up_tt_cores',
        'up_tucker_supercore': 'down_tucker_supercore', 'down_tt_supercore': 'up_tt_supercore',
        'up_ranks': 'down_ranks', 'down_ranks': 'up_ranks',
        'up_weights': 'down_weights', 'down_weights': 'up_weights',
        'up_weight_supercore': 'down_weight_supercore', 'down_weight_supercore': 'up_weight_supercore',
        'up_mask': 'down_mask', 'down_mask': 'up_mask',
        'variations_up_mask': 'variations_down_mask', 'variations_down_mask': 'variations_up_mask',
        'nU': 'n_down', 'nD': 'n_up', 'rL': 'r_left', 'rR': 'r_right'}
pattern = re.compile(r'\b(' + '|'.join(sorted(SWAP, key=len, reverse=True)) + r')\b')
source = pattern.sub(lambda m: SWAP[m.group(1)], source)    # then review the nU/nD/rL/rR hits: they may be your own names
```

`\b` treats `_` as a word character, so `up_mask` does not match inside `variations_up_mask`. The `update` strings
need a hand edit. Code that unpacks positionally, `up, down, left, right = frame.data`, keeps working; rename the
locals to `down, up, left, right` so they read right.

**What does not change.** Tuple and field order, the save format (keyed by position), every number (seeded starts
included), and the orthogonalization and SVD method names.

**Letters.** The API has none. Diagrams and equations in the docs use the initials: `D` (down, the Tucker factors),
`U` (up), `L`, `R`; widths `n_down`, `n_up`, `r_left`, `r_right`. Docs before 2026.5.0 used `U` for the Tucker
factor and `O` for the TT-shaped core. **T4S** writes the frame as `Uᵢ, Pᵢ, Qᵢ, Oᵢ`: the paper's `U` is our `D`,
and our `U` is the paper's `O` (its "outer orthogonal" is our up-orthogonal); its `P`/`Q` are our `L`/`R`, its
`δU`/`δG` our `V`/`H`.
````

### (c) Pages that need a careful hand rewrite, not word swaps

Both senses of up/down (edge direction vs store side) in one sentence, a convention stated, or a letter conflict:

1. `docs/frame_variations.md` — the convention's home: two tables (symbol, rank columns), § Four rank stores, the
   weights list, the tangent diagram; insert §1(a). Interacts with work-list items E8, E9, E10.
2. `docs/glossary.md` — entries **orthogonal core** (its "Clash" sentence resolves), **frame cores**, **store**,
   **corewise frame**, **two-colour pass**, **alignment pass**, **padded widths**, **gauge conditions**, **Tucker
   core**, **tangent metric**, **variation masks**, **`.data`**, **T4S**, and the Symbols rows `P`, `O, D` (plus a
   new `U` row).
3. `docs/user_guide.rst` § T3 manifold (diagram + core list) and § Minimal rank conditions.
4. `docs/contributor/padded_frame_internals.md` § The step as message passing: "the down edges ... then the up
   edges" mixes edge direction and store; message `D`; `U′`; the seed-tag list must keep its integers.
5. `docs/padded_frame_cone.tex` §2 table, §3 messages and Algorithm 1 (same two-sense problem; `D_{a→b}`,
   `U_{x→t}`), §4 seed tags and representation dependence, §5 "the stores pointing toward the root (up, left)".
6. `docs/aligned_padded_frame.md` § The padded frame (store definition), § The schedule (colours), § Jit or eager.
7. `docs/weighting.md` § `T3FrameWeights` (table, absorption sentence, a doctest comment) and
   `docs/contributor/weighted_internals.md` (four rank families, the abandoned design's letters, the act-on table).
8. `docs/contributor/curvature_internals.md` — frame `(U, O, L, R)` beside the lift `L`, `DΦ`, raw `θ = (U, L)`,
   `U̇`, `U_g`, a doctest unpacking `U, _, L, _ = frame`.
9. `docs/contributor/sharing_internals.md` and `docs/sharing.md` — `O_i` "down core", `U̇`, `U_g`, `nD_i`, the "up
   width `nU` (the `U̇` slot)", "new up-slots".
10. `docs/symmetric_probe_derivatives.tex` — line 373 store words in T4S letters; lines 491 / 500 describe the
    Tucker edge from the old upside-down picture ("up-edge to its Tucker core").
11. `docs/shared_t3_math.tex` — "up-leg", "new up-slots", `n_{D,i}`, in a note whose `U` is the Tucker factor.
12. `docs/exact_hessian_two_hole.tex` line 117 (T4S letters equated with the old word tuple) and `n^D_i`, in a note
    whose `D` is the derivative.
13. `docs/batching_and_stacking.md` §2 table and §8 paper ↔ code map; `docs/entries_apply_probe.md` §8 (add the
    correspondence; decide the T3-level `Uᵢ` in §1/§3).
14. `docs/numerical_contracts.md` (`U/O/L/R`, `Uᵀ δU`, and a doctest variable `D` on the same page).
15. `CHANGELOG.md` / `docs/release_notes.md` (new sections, §1(b)) and `dev/glossary_naming_decisions.md` (§7).

Everything else is a mechanical swap: `chunking.md`, `transposes.md:117`, `fitting_and_optimization.md:216`,
`parameters_not_closures.md`, `precompute_and_caching.md:112`, `numerical_contract_catalog.md`,
`reading_signatures.md` / `signature_style.md` (follow the code), `pad_safe_svd.tex` (two words), `naming_conventions.md`
(one new bullet).

## 2. Conflicts (reported, not resolved)

**No `D`-as-diagonal conflict exists.** Every diagonal in scope is spelled `diag(...)`, `\diag`, `\mathrm{diag}`
or "diagonal"; no page writes a diagonal matrix `D`. The `D` conflicts are other meanings:

| # | where | the other `D` / `U` | severity |
|---|---|---|---|
| C1 | `exact_hessian_two_hole.tex:110-111, 210-232, 240-262, 283, 322, 394, 551, 1014-1026` | `DΦ`, `D²Φ`, `D_ζP`: the derivative, ~25 uses | **severe** if the note took `D` for the Tucker core |
| C2 | `exact_hessian_two_hole.tex:44, 573` | `\Dsh = \mathsf{D}`, the jet down-shift | severe (same note) |
| C3 | `symmetric_probe_derivatives.tex:111` | `D^t z_i(W) P^t`, the derivative tensor | moderate (one use; the note keeps T4S letters anyway) |
| C4 | `padded_frame_cone.tex:286-292, 315` | `D_{a→b}`, a message (R×k), in the section after the store table | **severe**: same note, same algorithm as the stores |
| C5 | `contributor/padded_frame_internals.md:36` | `D = [Ā | P⁺]ᵀ g_new`, the same message | severe: same section as the store table |
| C6 | `contributor/curvature_internals.md:310` | `DΦ[Lv] = Ev` (F14 row); the page's frame becomes `(D, U, L, R)` at line 13 | mild (different section) |
| C7 | `contributor/deferred_and_rejected.md:134` | `D^r x`, a jet shift operator; line 197's `U_k` would become `D_k` | mild (different bullets) |
| C8 | `numerical_contracts.md:104-107` | doctest variable `D = H_fd - Hp`; line 51 would read "D/U/L/R conditions" | mild; rename the variable (e.g. `diff`) |
| C9 | T4S-notation math notes: `exact_hessian_two_hole.tex` (~56 `U` uses, 2 tikz figures), `symmetric_probe_derivatives.tex` (~40), `shared_t3_math.tex` (~45, incl. `U_g`, `U̇`), `ttm_t3m_ht_note.tex` (~15, HT "leaf frames") | **load-bearing old `U`** = the Tucker factor (T4S, Tucker/HT literature) | **severe**: converting these notes to `D`/`U` would collide with C1–C3 and with the literature; recommend they keep the paper's letters and state the correspondence |
| C10 | `padded_frame_cone.tex:96-97, 152-153, 171, 284-286` | `U_{x→t}`: the generic frame on a directed edge (every store is a "`U`"), and `U′` | **severe**: `U_{ℓ_i→c_i}` would be `D_i` and `U_{c_i→ℓ_i}` would be `U_i` |
| C11 | `padded_frame_cone.tex:140-141, 174, 195, 206-211` | `A = UΣVᵀ`, the matrix case's SVD `U` (with completions `P`, `Q`) | moderate (self-contained proof; but `P`/`Q` also = T4S left/right) |
| C12 | `contributor/padded_frame_internals.md:28` | `U′`, the completed far frame (generic) | moderate |
| C13 | `entries_apply_probe.md:65, 85, 120-121`; `t3m_methods.md:24, 27` | `Uᵢ` = a T3's Tucker core (not a frame), T4S letter; the user guide and README call these `Bᵢ` | moderate: recommend `Bᵢ` |
| C14 | `sharing.md:134, 146, 156`; `contributor/sharing_internals.md:17, 59-61, 90-92, 175-176`; `contributor/curvature_internals.md:29, 47, 65-67, 102, 181, 228`; `CHANGELOG` (record) | `U̇`, `δU_i`, `U_g`, `θ = (U, L)`: raw Tucker-factor directions and the group factor, T4S-derived | **severe** for a mechanical swap (`Ḋ`, `D_g` collide with C6 on the curvature page); a choice: keep T4S's `U` for raw T3 cores (and say so), or use `B` (`Ḃ`, `B_g`) as the user guide does |
| C15 | `contributor/parameters_not_closures.md:224` | `U → UQ, G → QᵀG`: raw T3 gauge example | mild: `B → BQ` |
| C16 | `contributor/curvature_internals.md:13, 17` (pre-existing) | `L` = left cores and `L` = the lift in the same formula block | not new; the rename does not touch it |
| C17 | `shared_t3_math.tex:358` | `O_g`, an orthogonal complement basis | none after the rename (`O` leaves our notation) |
| C18 | code shape comments quoted in `reading_signatures.md:12-15`, `contributor/signature_style.md:56-59` (`C+(rLi,nUi,rL(i+1))`) | if code keeps letters in shape comments with the new meanings, `nUi` must become `nDi` and `nDi` `nUi` — a silent swap | cross-partition decision (inventories 1/2): words or swapped letters in shape comments |
| C19 | edge-direction vs store words: `padded_frame_internals.md:29-31`, `padded_frame_cone.tex:310-315, 403-404`, `symmetric_probe_derivatives.tex:491, 500` | "down edges", "the stores pointing toward the root (up, left)", "up-edge to its Tucker core" | severe for a mechanical swap: a store's edges point away from its name; rewrite by hand (§1(c)) |

Decisions this leaves for Nick: (1) do the T4S-notation notes (C9) keep the paper's letters with a correspondence
line (recommended), or convert? (2) raw T3 Tucker cores and directions outside the frame (C13–C15): keep `U`/`U̇`
as T4S does, or `B`/`Ḃ` as the user guide does? (3) `padded_frame_cone.tex`'s generic edge frame `U_{x→t}` and
message `D_{a→b}` (C4, C10): which letters replace them?

## 3. Figures and diagrams

| figure | shows | needs | how |
|---|---|---|---|
| `docs/frame_variations.md:13-17` (ASCII) | three tangent terms, `U0 U1 U2` under the TT chain | redraw: `U` → `D` | by hand in the markdown |
| `docs/user_guide.rst:124-133` (ASCII) | six tangent terms: `U` row under `H`-terms, `O0..O2` in the chain of `V`-terms | redraw: `U` → `D`, `O` → `U` | by hand |
| `docs/padded_frame_cone.tex` Table 1 (`:104-115`) | directed edge / store / core / slot, `U_i`, `O_i`, `n_{U,i}`, `n_{D,i}` | regenerate the PDF | edit, `latexmk -pdf padded_frame_cone.tex` (`docs/padded_frame_cone.pdf`, committed) |
| `docs/exact_hessian_two_hole.tex` fig. holes (`:150-192`) | raw T3: `G_1..G_4` on top, `U_1..U_4` below (T4S letters, raw Tucker factors) | none if the note keeps T4S letters (C9); else relabel | tikz inline; `latexmk -pdf` |
| `docs/exact_hessian_two_hole.tex` lift figure (`:338-375`) | frame slots `P_{i-1}`, `O_i`, `Q_{i+1}`, `δU_i`, `U̇_i` | as above | as above |
| `docs/exact_hessian_two_hole.tex` comment `:64` | "TT cores on top (yellow), Tucker factors below (blue), the T4S convention" | none: it is the new picture | — |

No other figure carries a store letter or word. `docs/t3_dots.png` (unreferenced icon: yellow/red TT circles on top,
blue Tucker circles below, no letters; added in fd3d93fd, no generating script) already shows the new picture.
`docs/favicon.ico` has no letters. The hole and ladder tikz figures (`:716-740`, `:800-822`) carry no store letters
(the `\hole` macro's "down label" is spatial). No SVG and no figure-generating scripts exist in the repo. The
T3 diagrams in `README.md:15-20` and `user_guide.rst:14-20, 178-196` use `B` and `G` and are unaffected.

**PDFs.** Four are committed and built by hand (`latexmk -pdf`; `latexmk` and `pdflatex` are installed here):
`padded_frame_cone.pdf` — **must rebuild** (store table, store words, seed tags, colours; linked from the user docs);
`exact_hessian_two_hole.pdf` — **must rebuild** (line 117 equates T4S letters with the old word tuple; `n^D_i`);
`shared_t3_math.pdf` — **must rebuild** if its words change (up-leg ×2, new up-slots, `n_{D,i}` ×4);
`pad_safe_svd.pdf` — two occurrences of "the down step"; rebuild, or defer per the work list's rule for `.tex`
rows. `symmetric_probe_derivatives.tex` and `ttm_t3m_ht_note.tex` have no PDF in the repo (linked as `.tex` on
GitHub); edit the source only.

## 4. T4S-CITE: every place the docs use T4S's letters

The correspondence note (§1(a)) should be linked or quoted from each. "keep" = keep the paper's letters there.

| loc | current | proposed |
|---|---|---|
| `batching_and_stacking.md:390-393` | "**Paper ↔ code** (Appendix A of the T4S paper): `U`=up_tucker, `P`=left_tt, `Q`=right_tt, `O`=down_tt (called `down_tt_cores`, not "outer"); `δU`=tucker_variations (`V`), `δG`=tt_variations (`H`)." | "**Paper ↔ code** (Appendix A of the T4S paper): `U` = `down_tucker_cores` (our `D`), `P` = `left_tt_cores` (`L`), `Q` = `right_tt_cores` (`R`), `O` (outer-orthogonal) = `up_tt_cores` (our `U`); `δU` = `tucker_variations` (`V`), `δG` = `tt_variations` (`H`). The paper's `U` is our `D`." Keep the disjointness sentence. |
| `batching_and_stacking.md:43-44` | "frame TT core `Pᵢ`", "variation core `δUᵢ`" | "left frame core `Lᵢ`", "Tucker variation `Vᵢ`" (or keep with "(T4S letters)") |
| `batching_and_stacking.md:90`, `contributor/batching_internals.md:49` | "(`U`,`P`,`Q`,`O`,`G`,`H`,`B`)" | "(`D`,`U`,`L`,`R`,`V`,`H`,`G`,`B`, and T4S's `P`,`Q`,`O`)" |
| `entries_apply_probe.md:228-271` (§8, 12 table rows + 252 "cores `P` (left), `Q` (right), `O` (central)" + 271 "`(Pᵢ, Qᵢ, Oᵢ)`") | the paper's letters | keep; add after line 233: "Letters are the paper's: its `U` is our down core `D`, its `O` our up core `U`, its `P`/`Q` our `L`/`R` ([T4S notation](frame_variations.md#...))." |
| `transposes.md:37` | "the Section 6.3 substitution `P,Q,O → G`" | keep (a citation) |
| `frame_variations.md:27-28` | "`L`/`P`", "`R`/`Q`" | keep; add "`U` (T4S: `O`)" and "`D` (T4S: `U`)" to rows 25-26 |
| `frame_variations.md:58` | "Appendix A.3 of the T4S paper" | keep; the correspondence note follows in § Four rank stores |
| `glossary.md:713` (`P` row) | "`LocalModel.frame` as `(U, O, P, Q)`" | keep (it is a T4S-letter docstring); the code docstring is partition 1/2 |
| `glossary.md:717` (`O`, `D` row) | see §5.2 | rewrite |
| `glossary.md:373` | "variation cores, `δU`/`δG`" | "`δU`/`δG` in T4S" |
| `glossary.md:267` | "also: `P`/`Q` for left/right" | "also: T4S's `P`/`Q` for left/right, `O` for up, `U` for down" |
| `contributor/testing_strategy.md:41, 52` | "Appendix A.3.1, eqs 49–52"; "`[Q | P]` honest-boundary TT mask (the paper rule)" | keep |
| `contributor/weighted_internals.md:27` | "`left→P, right→Q`" | see §6 |
| `contributor/curvature_internals.md:23` | "T4S ... App. A for the frame and gauge" | keep |
| `padded_frame_cone.tex:83, 95, 165-166` | "those of T4S"; "T4S App. A.1, A.3, Algorithm 11" | keep; add the correspondence after the store table |
| `exact_hessian_two_hole.tex:115-135` | "$(U_i, O_i, P_i, Q_i)$ = \code{T3Frame.data} = (up, down, left, right)" | see §6; the note keeps T4S letters |
| `exact_hessian_two_hole.tex:478-480, 539, 545, 640` | "`P,Q,O→G`", "$Q=O=P=G$", "the manifold frame $(U,O,P,Q)$ and the corewise frame $(U,P,P,P)$" | keep |
| `symmetric_probe_derivatives.tex:373-375, 384, 428, 559` | the frame in T4S letters | keep; fix the word list at 373 (§6) |
| `shared_t3_math.tex` (whole note) | T3 in T4S style, `U_i` Tucker factors with orthonormal columns | keep (C9); add a notation line |
| `CLAUDE.md:68-69, 115, 213-214` | notation map | Appendix A |
| code: `LocalModel.frame` docstring `(U, O, P, Q)` | — | partition 1/2 |

## 5. User pages: exhaustive table

Columns: location, current text, category, proposed text. Categories: SW = STORE-WORD, WID = WIDTH,
CL = CORE-LETTER, ADJ, FIG, CONV = CONVENTION-SENTENCE, T4S, CONF = conflict (§2). "follow code" = a private name
quoted from the code: use whatever the code rename produces.

### 5.1 `docs/frame_variations.md` (hand rewrite; home of §1(a))

| loc | current | cat | proposed |
|---|---|---|---|
| :16 | `    U0   U1   U2            U0   U1   U2            U0   U1   U2` | FIG, CL | `    D0   D1   D2            D0   D1   D2            D0   D1   D2` |
| :19 | "(plus a matching set with the Tucker variation `(Vᵢ)` in place of `Uᵢ`.)" | CL | "(plus a matching set with the Tucker variation `(Vᵢ)` in place of `Dᵢ` and the up core `Uᵢ` in the chain.)" |
| :25 | `| up-Tucker `Uᵢ` | `U` | the Tucker factors (retained directions) | `nU` |` | SW, CL, WID, CONV | `| down (Tucker) `Dᵢ` | `D` (T4S: `U`) | the Tucker factors (retained directions) | `n_down` |` |
| :26 | `| down-TT `Oᵢ` | `O` | the Tucker *complement* (where `V` varies) | `nD` |` | SW, CL, WID, CONV | `| up (TT-shaped) `Uᵢ` | `U` (T4S: `O`) | the Tucker *complement* (where `V` varies) | `n_up` |`; put the two rows in tuple order (down, up, left, right) |
| :27 | `| left-TT `Lᵢ` | `L`/`P` | ... | `rL` |` | WID, T4S | `... | `r_left` |` |
| :28 | `| right-TT `Rᵢ` | `R`/`Q` | ... | `rR` |` | WID, T4S | `... | `r_right` |` |
| :29 | "Tucker-factor direction (in the `nD` complement) | `nD`" | WID | "(in the `n_up` complement) | `n_up`" |
| :30 | "TT-core direction | `nU`, `rL`, `rR`" | WID | "`n_down`, `r_left`, `r_right`" |
| :34 | "(`U0…U(d-1)`, `L0…L(d-1)`, …), but the TT **bonds** run `rL0 … rLd` and" | CL, WID | "(`D0…D(d-1)`, `L0…L(d-1)`, …), but the TT **bonds** run `r_left[0] … r_left[d]` and" |
| :35 | "`rR0 … rRd` — **`d+1`** of them." | WID | "`r_right[0] … r_right[d]` — ..." |
| :57 | "(`Uᵢᵀ Vᵢ = 0`; `Lᵢ` left-orthogonal" | CL (+ E10) | "(`Dᵢ Vᵢᵀ = 0`; `Lᵢ` left-orthogonal" |
| :70-71 | "it carries **four**: `up_ranks` (the rows of each Tucker factor `Uᵢ`), `down_ranks` (the mode leg of each down/outer core `Oᵢ`), and `left_ranks` / `right_ranks` (the TT bonds, in each sweep direction)." | CONV, SW, CL, ADJ ("outer", N4) | "it carries **four**: `down_ranks` (the rows of each Tucker factor `Dᵢ`), `up_ranks` (the mode leg of each up core `Uᵢ`), and `left_ranks` / `right_ranks` (the TT bonds, in each sweep direction)." Then insert §1(a) and the T4S note. |
| :82 | `| `Uᵢ` (up / Tucker) | rows orthonormal | `nᵢ ≤ Nᵢ` |` | SW, CL | `| `Dᵢ` (down / Tucker) | rows orthonormal | `nᵢ ≤ Nᵢ` |` |
| :83 | `| `Oᵢ` (down / outer) | `nᵢ × (rᵢ·rᵢ₊₁)` unfolding, rows orthonormal | ...` | SW, CL, ADJ | `| `Uᵢ` (up) | `nᵢ × (rᵢ·rᵢ₊₁)` matricization, rows orthonormal | ...` (row order: put after `Dᵢ` as now) |
| :94 | "returns `up_ranks=(4,4,4)` alongside `down_ranks=(2,4,2)`." | SW (meaning swap) | "returns `down_ranks=(4,4,4)` alongside `up_ranks=(2,4,2)`." (verify against the renamed `__repr__`) |
| :101 | "`has_minimal_ranks` additionally requires `up == down` and `left == right`" | SW | symmetric: may stay; for consistency "`down == up`" |
| :104 | "The tangent variations are shaped by the operative (down) ranks" | SW | "... by the operative (up) ranks" |
| :121 | "(`d` each: `up`↔`H.nU`, `down`↔`V.nD`, `left`↔`H.rL`, `right`↔`H.rR`)" | SW, WID | "(`d` each: `down`↔`H.n_down`, `up`↔`V.n_up`, `left`↔`H.r_left`, `right`↔`H.r_right`)" |

### 5.2 `docs/glossary.md`

| loc | current | cat | proposed |
|---|---|---|---|
| :102-109 (orthogonal core) | "... *up* (also outer), its `ni × (ri·r(i+1))` matricization has orthonormal rows (`up_orthogonalize_tt_cores`). Tucker core: *down*, `B Bᵀ = I` (`down_orthogonalize_tucker_cores`). The adjective names the side of the cut edge whose indices the orthonormality sums over. **Clash:** the frame pages name these by store instead (`O` "down-orthogonal", `U` "up orthogonal"); a rename of the stores is being assessed." | ADJ (stays), CONV | keep through "...sums over."; replace the Clash sentence with: "The frame stores are named the same way: an up core is up-orthogonal, a down core down-orthogonal (2026.5.0; before, the store names were the other way round)." |
| :119 (padded widths) | "(`n`, `r`, `N`; frames `nU`, `nD`, `rL`, `rR`; ..." | WID | "(`n`, `r`, `N`; frames `n_down`, `n_up`, `r_left`, `r_right`; ..." |
| :195-196 (Tucker core) | "(`tucker_cores`; `Bi`, `Ui` in a frame; also: Tucker factor, Tucker basis, leaf)" | CL | "(`tucker_cores`; `Bi`, `Di` in a frame (`Ui` in T4S); also: ...)" |
| :220-222 (alignment pass) | "One update of the stores named in `update` (letters `U`, `O`, `L`, `R`); the others are transported." | SW (update) | "One update of the stores named in `update` (`'down'`, `'up'`, `'left'`, `'right'`); the others are transported." |
| :252-253 (corewise frame) | "(`(U, G, G, G)`; `t3_corewise_frame`). The non-orthonormal frame whose down, left and right cores are the TT cores" | CL, SW | "(`(D, G, G, G)`; ...). The non-orthonormal frame whose up, left and right cores are the TT cores" (or `(B, G, G, G)`, C14) |
| :266-269 (frame cores) | "(up `Ui`, down `Oi`, left `Li`, right `Ri`; `up_tucker_cores`, `down_tt_cores`, `left_tt_cores`, `right_tt_cores`; also: `P`/`Q` for left/right, `D` or "outer" for down). Up: orthonormal Tucker cores. Down: TT-shaped `(rL, nD, rR)`, the TT core of the Tucker-variation terms (`Vi` varies in the complement of `Ui`'s rows). Left/right: the orthogonal TT environments." | CONV, SW, CL, WID, T4S | "(down `Di`, up `Ui`, left `Li`, right `Ri`; `down_tucker_cores`, `up_tt_cores`, `left_tt_cores`, `right_tt_cores`; also: T4S's `Ui` for down, `Oi` and "outer" for up, `Pi`/`Qi` for left/right). Down: the orthonormal Tucker factors. Up: TT-shaped `(r_left, n_up, r_right)`, the TT core of the Tucker-variation terms (`Vi` varies in the complement of `Di`'s rows). Left/right: the orthogonal TT environments. Each store is named for the side of its edge its cores occupy." |
| :273 (gauge conditions) | "`Ui Viᵀ = 0` for every `i` (code layout)" | CL | "`Di Viᵀ = 0` for every `i` (code layout)" |
| :284 (gauge projection) | "compensating through the down and right cores." | SW | "compensating through the up and right cores." |
| :344-347 (store) | "(`up_ranks`, `down_ranks`, `left_ranks`, `right_ranks`; ...). One of the frame's four core families, each orthonormal in its own matricization or unfolding. Its ranks are set by the orthogonalization sweep (up, left, right over the left chain, then down); the differences are the slack." | CONV, SW | "(`down_ranks`, `up_ranks`, `left_ranks`, `right_ranks`; ...). One of the frame's four core families, named for the side of its edge its cores occupy, each orthonormal over that side. Its ranks are set by the orthogonalization sweep (down, left, right over the left chain, then up); the differences are the slack." |
| :360-361 (tangent metric) | "Diagonal weights on a tangent's coordinates (`up`, `down`, `left`, `right`, each length `d`)" | SW | "(`down`, `up`, `left`, `right`, each length `d`)" |
| :366-367 (two-colour pass) | "(`'UL'`, `'RO'`; `alternate=True`). A pass updating the up and left stores, then one updating right and down" | SW (update) | "(`('down', 'left')`, `('right', 'up')`; ...). A pass updating the down and left stores, then one updating right and up" |
| :370 (variation masks) | "from the frame's as `(up, down, left[:-1], right[1:])`" | SW | "`(down, up, left[:-1], right[1:])`" |
| :373 (variations) | "also: variation cores, `δU`/`δG`" | T4S | "also: variation cores; `δU`/`δG` in T4S" |
| :609 (`.data`) | "a frame's `(up, down, left, right)`" | SW | "a frame's `(down, up, left, right)`" |
| :687-688 (T4S) | entry | T4S | append: "Its frame letters differ: its `U` is our `D`, its `O` our `U` ([T4S notation](frame_variations.md#...))." |
| :713 (`P` row) | "the left frame core in the T4S paper's notation (... `LocalModel.frame` as `(U, O, P, Q)`)" | T4S | keep |
| :717 (`O`, `D` row) | "the down frame core: `O` in the user guide, `frame_variations.md`, the math notes and the `update` letters; `D` in the `T3Frame` docstring; "outer" in the minimal-rank list" | CONV, CL | two rows: "`U` — the up frame core `Ui` (TT-shaped; frame pages, diagrams); in T4S and the T4S-notation math notes, the Tucker factor (our `D`); the SVD factor in `UΣVᵀ`; the generic edge frame `U_{x→t}` (padded-frame note)" and "`D` — the down frame core `Di` (the Tucker factors); the derivative `DΦ` (exact-Hessian note, curvature page); the message `D_{a→b}` (padded-frame note, if kept); `O` — T4S's up core (outer-orthogonal), not used by the library after 2026.5.0" |
| :79, :106 | "every Tucker core down-orthogonal"; "`down_orthogonalize_tucker_cores`" | ADJ | stays (right) |
| :712 | "the up-matricization `Wi` (shared-factor math note)" | ADJ | stays |
| :237 | "the upward single-store rotations" | — | unrelated (energy goes up) |
| :408 | "`ξ` up the Tucker cores" | — | stays: travel direction, matches the new picture |

### 5.3 `docs/user_guide.rst`

| loc | current | cat | proposed |
|---|---|---|---|
| :97 | "- Outer TT core unfoldings are full rank: ni <= (ri*r(i+1))" | ADJ (T4S "outer"), M/U | "- TT core up-matricizations are full rank: ni <= (ri*r(i+1))" (releases work-list rows 300-301) |
| :127 | `v  =     U0  U1  U2    +    U0  U1  U2    +    U0  U1  U2` | FIG, CL | `v  =     D0  D1  D2    +    D0  D1  D2    +    D0  D1  D2` |
| :130 | `1--O0--R1--R2--1   1--L0--O1--R2--1   1--L0--L1--O2--1` | FIG, CL | `1--U0--R1--R2--1   1--L0--U1--R2--1   1--L0--L1--U2--1` |
| :132 | `+     V0  U1  U2    +    U0  V1  U2    +    U0  U1  V2` | FIG, CL | `+     V0  D1  D2    +    D0  V1  D2    +    D0  D1  V2` |
| :139 | "- up_tucker_cores   = (U0,...,U(d-1)), orthogonal" | SW, CL | "- down_tucker_cores = (D0,...,D(d-1)), down-orthogonal" |
| :142 | "- down_tt_cores     = (O0,...,O(d-1)), down-orthogonal" | SW, CL, **ADJ backwards** | "- up_tt_cores       = (U0,...,U(d-1)), up-orthogonal" (list the four in tuple order: down, up, left, right) |
| :215 | "    frame Tucker core  U_i   (T3Frame)        C ..." | CL | "    frame Tucker core  D_i   (T3Frame) ..." |

### 5.4 `docs/aligned_padded_frame.md`

| loc | current | cat | proposed |
|---|---|---|---|
| :20-21 | "A frame has four **stores**: the families of frame cores (up, down, left, right), each one rank axis per mode or bond (see [Four rank stores](...))." | CONV, SW | "A frame has four **stores**: the families of frame cores (down, up, left, right), each named for the side of its edge its cores occupy, each one rank axis per mode or bond (see [...](...))." |
| :60-61 | "a `'UL'` pass (up and left stores), then an `'RO'` pass (right and down), the other colour transported." | SW (update) | "a `('down', 'left')` pass (the down and left stores), then a `('right', 'up')` pass, the other colour transported." |
| :65-66 | "An odd remainder runs one final `'UL'` pass. ... 3 pairs and a final `'UL'`." | SW (update) | "`('down', 'left')`" ×2 |
| :138 | "(the oversampled `'ULRO'`, the truncating `'ULRO'`, `'UL'`, `'RO'`)" | SW (update) | "(the oversampled all-store pass, the truncating all-store pass, `('down', 'left')`, `('right', 'up')`)" |

### 5.5 `docs/weighting.md`

| loc | current | cat | proposed |
|---|---|---|---|
| :71 | `| `up` | `H`'s Tucker (`nU`) leg | `d` |` | SW, WID | `| `down` | `H`'s Tucker (`n_down`) leg | `d` |` |
| :72 | `| `down` | `V`'s complement (`nD`) leg | `d` |` | SW, WID | `| `up` | `V`'s complement (`n_up`) leg | `d` |` |
| :73-74 | "`H`'s left bond (`rL`)", "`H`'s right bond (`rR`)" | WID | "(`r_left`)", "(`r_right`)" |
| :100 | "(`down`→`V`, `up`/`left`/`right`→`H`)" | SW | "(`up`→`V`, `down`/`left`/`right`→`H`)" |
| :110 | "`up = down =` the Tucker weights" | SW | symmetric; may stay ("`down = up =`") |
| :117 | `>>> print(gk.up_ranks, gk.down_ranks)              # up = down = the Tucker weights` | SW (doctest) | `>>> print(gk.down_ranks, gk.up_ranks)              # down = up = the Tucker weights` (output unchanged) |
| :124 | "(where the complement rank `nD` equals the Tucker rank" | WID | "`n_up`" |

### 5.6 `docs/batching_and_stacking.md`

| loc | current | cat | proposed |
|---|---|---|---|
| :42 | "| frame Tucker core `Uᵢ` (`T3Frame`) |" | CL | "| frame Tucker core `Dᵢ` (`T3Frame`) |" |
| :43-44 | "frame TT core `Pᵢ`", "variation core `δUᵢ`" | T4S | §4 |
| :90 | "(`U`,`P`,`Q`,`O`,`G`,`H`,`B`)" | CL, T4S | §4 |
| :162, :164 | "`C + (rL, n, rR)`" (T3 core, bond symbols) | WID | optional: `C + (r_left, n, r_right)`; the bond meaning is unchanged |
| :281 | "`[U_i ; V_i]` is one array" | CL | "`[D_i ; V_i]`" |
| :390-393 | paper ↔ code map | T4S, SW, CL | §4 |
| :415 | "`T3Frame.data = (up, down, left, right)`" | SW | "`(down, up, left, right)`" |

### 5.7 `docs/entries_apply_probe.md`

| loc | current | cat | proposed |
|---|---|---|---|
| :65 | "Tucker-core fibers `Uᵢ[…, idxᵢ]`" | CL, CONF C13 | "`Bᵢ[…, idxᵢ]`" (a T3's Tucker core; user guide letter) or keep with a T4S note |
| :85 | "(`zᵢ = Uᵢ ηᵢ`)" | CONF C13 | "(`zᵢ = Bᵢᵀ ηᵢ`)" or keep |
| :120-121 | "through `Uᵢ` (`N·n`)", "assemblies `Uᵢηᵢ`" | CONF C13 | `Bᵢ` or keep |
| :228-271 | §8 paper ↔ code map | T4S | keep; add the correspondence sentence (§4) |
| :241, :245 | "(contract up)", "(contract down)" | — | stay: travel direction, the paper's and the new picture's |

### 5.8 `docs/numerical_contracts.md`

| loc | current | cat | proposed |
|---|---|---|---|
| :51 | "the frame is orthonormal (U/O/L/R conditions)" | CL | "(D/U/L/R conditions)" |
| :65 | "the gauge grams `Uᵀ δU`, `Lᵀ δG` over a scale" | CL, T4S | "the gauge grams `D Vᵀ`, `Lᵀ H` over a scale" |
| :104-107 | `>>> D = H_fd - Hp` (and two uses) | CONF C8 | rename the doctest variable, e.g. `diff` |
| :246 | "four rank stores — `up`/`down` and `left`/`right`" | SW | symmetric; may stay, or "`down`/`up`" |
| :249 | "returns `up_ranks=(4,4,4)` with `down_ranks=(2,4,2)`" | SW (meaning swap) | "returns `down_ranks=(4,4,4)` with `up_ranks=(2,4,2)`" |

### 5.9 Other user pages

| loc | current | cat | proposed |
|---|---|---|---|
| `chunking.md:80` | "`(order+1)·nU·N` (Tucker gradient" | WID | "`(order+1)·n_up·N`" (the Tucker variation's rows are the up width; `nU` was imprecise, `:126` says `nO`) |
| `chunking.md:126` | "Tucker (`nO`, `N` legs)" | WID | "Tucker (`n_up`, `N` legs)" |
| `fitting_and_optimization.md:216` | "the raw cores `(U,G,G,G)`, Euclidean metric" | CL, CONF C14 | "`(D,G,G,G)`" or "`(B,G,G,G)`" |
| `transposes.md:117` | "used in place of the orthogonal left/right/down frames" | SW | "the orthogonal left/right/up frames" |
| `transposes.md:37` | "`P,Q,O → G`" | T4S | keep |
| `sharing.md:268` | "the `down` family lives on the `nD` legs" | SW, WID | "the `up` family lives on the `n_up` legs" |
| `sharing.md:134, :146, :156` | "`U̇`", "`V_i = S_iᵀ U̇`", "the `δU_i` are equal", "`U̇ = M_g⁺ [...]`" | CONF C14 | decision (C14) |
| `t3m_methods.md:24, :27` | "`T[x] = Σ_n (∏_i U_i[n_i, x_i]) · C[n]`", "`W_i = U_i^x ⊙ U_i^y`" | CONF C13 | keep (T3 Tucker factors, Khatri–Rao literature) or `B_i` |
| `reading_signatures.md:12-15` | "`elm_shape=C+(rLi,nUi,rL(i+1))`", "`W+C+(nUi,)`", "`W+C+(rLi,)`" | WID, CONF C18 | follow the code's new shape comments of `probing.compute_mu` |
| `t3svd_minimal_ranks.md:16-18, 21, 32, 35, 46, 48-49, 52-53, 62-63` | "`rL_i <= n_i * rR_i`" ... (`rL_i = r_i`, `rR_i = r_{i+1}`, defined at :21) | WID | optional: T3 bond symbols, not frame stores; meaning unchanged |
| `naming_conventions.md` § Parameter names (after :119) | — | CONV (new) | add: "- Frame stores are words, never letters: `down_*`, `up_*`, `left_*`, `right_*`, widths `n_down`, `n_up`, `r_left`, `r_right`, `update` takes store words. Diagrams and docs use the initials `D`, `U`, `L`, `R`." |
| `naming_conventions.md:206` | "`up_orthogonalize_tt_supercores`" | ADJ | stays |
| `uniform_masks_vs_ranks.md`, `uniform_supercore_layout.md`, `uniform_equivalence_contract.md`, `rank_continuation.md`, `getting_started.rst`, `index.rst`, `api_reference.rst`, `README.md` | — | — | no store word, width or frame letter |

## 6. Contributor pages and math notes: exhaustive table

### 6.1 `docs/contributor/padded_frame_internals.md` (hand rewrite)

| loc | current | cat | proposed |
|---|---|---|---|
| :8-9 | "Each **directed** edge has one frame core, and the four directions are the library's four stores:" | CONV | "Each **directed** edge has one frame core, orthonormal over the side the edge comes from; its store is named for that side, so a store's edges point away from its name (the up store's edges point down, from chain to leaf):" |
| :14 | "| `l_i → c_i` | `up[i]` | the Tucker index |" | SW | "| `l_i → c_i` | `down[i]` | the Tucker index |" |
| :15 | "| `c_i → l_i` | `down[i]` | the down index |" | SW | "| `c_i → l_i` | `up[i]` | the up core's mode index (`n_up`) |" |
| :27-28 | "(`_down_message`, `_left_message_step`, `_right_message_step`, `_up_message`). On each directed edge `a → b`, `Φ = Pᵀ Z_far U′`" | SW (follow code), CONF C12 | names follow code; `U′` per decision (C10) |
| :29-30 | "The order is the down edges, the left messages from the far end, the right messages from the near end, then the up edges." | SW, CONF C19 | "The order is the up store's edges (`c_i → l_i`), the left messages from the far end, the right messages from the near end, then the down store's edges (`l_i → c_i`)." |
| :31 | "(`_up_edge`, `_left_edge_step`, `_right_edge_step`, `_down_edge`)" | SW (follow code) | follow code (expected `_down_edge`, ..., `_up_edge`) |
| :36 | "`D = [Ā | P⁺]ᵀ g_new`" | CONF C5 | rename per decision |
| :42-43 | "Its up edge stacks the members' contents side by side in one `Y` (`_up_edge`)" | SW | "Its down edge (`l_i → c_i`) stacks ... (`_down_edge`, follow code)" |
| :46 | "Two-colour passes (`'UL'`, then `'RO'`)" | SW (update) | "(`('down', 'left')`, then `('right', 'up')`)" |
| :86 | "(`_seed_rows`; tags up 0, down 1, left 2, right 3; ..." | SW, **hazard** | "(tags: down (the Tucker store) 0, up 1, left 2, right 3 — the integers are those of 2026.4.0, so seeded streams are unchanged; ...)" |
| :116 | "the oversampled `'ULRO'`, the truncating `'ULRO'`, `'UL'` and `'RO'`" | SW (update) | word forms |
| :122 | "a `while_loop` of `(UL, RO)` pairs, a `cond` for the final `'UL'`" | SW (update) | word forms |
| :140 | "The uniform one takes the frame's up and left masks as its variation masks." | SW | "... the frame's down and left masks ..." |

### 6.2 `docs/contributor/weighted_internals.md`

| loc | current | cat | proposed |
|---|---|---|---|
| :18-21 | "left-orthogonal (`rL`) ... right-orthogonal (`rR`). Those four rank families (`nU`/`nD`/`rL`/`rR`) ... (`rLₖ ≠ rRₖ` in general" | WID | "(`r_left`) ... (`r_right`). Those four rank families (`n_down`/`n_up`/`r_left`/`r_right`) ... (`r_left[k] ≠ r_right[k]` in general" |
| :23 | "Forcing one class would need `down=None, left=right` for the plain case" | SW | "... `up=None, left=right` ..." |
| :27 | "(`up→U, down→O, left→P, right→Q`, variations untouched)" | SW, CL, T4S | "(`down`→`D`, `up`→`U`, `left`→`L`, `right`→`R`, variations untouched)" |
| :47 | "the **variations** (`down`→V, `up`/`left`/`right`→H)" | SW | "(`up`→V, `down`/`left`/`right`→H)" |

### 6.3 `docs/contributor/curvature_internals.md` (hand rewrite; C6, C14, C16)

| loc | current | cat | proposed |
|---|---|---|---|
| :13 | "with orthogonal frame `(U, O, L, R)` (`T3Frame.data`)" | CL | "`(D, U, L, R)`" |
| :28 | "`S_iᵀ = ⟨O_i, H_i⟩`" | CL (+ E6 row 319) | "`S_iᵀ = ⟨U_i, H_i⟩`" |
| :29 | "`U̇_i = (S_iᵀ)^{+ε} V_i` ... raw core directions of the base point `θ = (U, L)`" | CL, CONF C14 | per C14: `Ḋ_i`/`(D, L)` or `Ḃ_i`/`(B, L)` |
| :47 | "`tv_to_dense(t3_corewise_frame(U, L), L v)`" | CL, CONF C16 | "`t3_corewise_frame(D, L)`" (the lift `L` beside the cores `L` is pre-existing) |
| :65-67 | `# raw directions of the base point (U, L)`, `>>> U, _, L, _ = frame`, `t3_corewise_frame((U, L))` | CL (doctest) | `(D, L)`, `>>> D, _, L, _ = frame`, `t3_corewise_frame((D, L))` |
| :102 | "`compute_xi(U̇, ww)`" | CONF C14 | per C14 |
| :158 | "(`frame_data[5]`'s up / left masks)" | SW | "down / left masks" |
| :181 | "`U_g = (Q * sign(diag(R))).T`" | CONF C14 | per C14 (`D_g`) |
| :212 | "**Why Haar, and only for `U`.**" | CL | "**... only for `D`.**" |
| :228 | "`U̇_g = M_g^{+ε}[V_{i₁}; …]`" | CONF C14 | per C14 |
| :232 | "own ceiling `rL·rR`" | WID | optional `r_left·r_right` |
| :244 | "multiples of the padded `nD`" | WID | "the padded `n_up`" |
| :246 | "(`eps·max(Σ nD_i, nU)` ×" | WID | "(`eps·max(Σ n_up[i], n_down)` ×" |
| :310 | "`DΦ[Lv] = Ev`" | CONF C6 | report only |
| :318 | "the uniform frame's up / left masks ⊆ the point's" | SW | "down / left masks" |

### 6.4 `docs/contributor/sharing_internals.md`

| loc | current | cat | proposed |
|---|---|---|---|
| :17-19 | "`V_i = S_iᵀ U̇` ... `S_i = W2_i O2_iᵀ` ... `W2_i = S_i O2_i`" | CONF C14, CL | `O2_i` is a matricized old-`O` core (code local): follow code; `U̇` per C14 |
| :28 | "with the stored `O_i`/`R_i`" | CL | "`U_i`/`R_i`" |
| :59, :61 | "`U̇ = M_g⁺ [...]`", "`{δU_i all equal}`" | CONF C14 | per C14 |
| :65 | "when the `nD_i` differ" | WID | "`n_up[i]`" |
| :90-91 | "`[U_g; V_i]` ... (when the `nD_i` differ)" | CONF C14, WID | per C14; `n_up[i]` |
| :92 | "recover `U̇` per group" | CONF C14 | per C14 |
| :94-95 | "replace the down core `O_i` by the companion's center `H_i` (the identity `S_i`-absorbed-`O_i = H_i` ..." | SW, CL | "replace the up core `U_i` by ... (`S_i`-absorbed-`U_i = H_i`" |
| :104 | "first-order mass in the new up-slots" | SW | "first-order mass in the new Tucker slots" (old up = Tucker; avoid both words) |
| :156 | "roundoff in `U_M`'s padded rows" | — | SVD factor; unrelated |
| :162 | "`eps·max(Σ_i nD_i, n_g)`" | WID | "`eps·max(Σ_i n_up[i], n_g)`" |
| :175-176 | "rebuilds the variation block at the **up** width `nU` (the `U̇` slot)" | SW, WID, CONF C14 | "at the **down** width `n_down` (the Tucker-direction slot)" |

### 6.5 Other contributor pages

| loc | current | cat | proposed |
|---|---|---|---|
| `numerical_contract_catalog.md:36` | "(`up`/`down`, `left`/`right`)" | SW | symmetric; may stay |
| `numerical_contract_catalog.md:56` | "(U/O/L/R conditions)" | CL | "(D/U/L/R conditions)" |
| `parameters_not_closures.md:92` | "`(up, down, left[:-1], right[1:])`" | SW | "`(down, up, left[:-1], right[1:])`" |
| `parameters_not_closures.md:93` | "the `(U,G,G,G)` corewise frame" | CL, CONF C14 | "`(D,G,G,G)`" |
| `parameters_not_closures.md:219` | "`nD == nU`; the first shared geometry where a group's rank differed from the down rank broke it" | WID, SW | "`n_up == n_down`; ... differed from the up rank ..." |
| `parameters_not_closures.md:224` | "(`U → UQ`, `G → Qᵀ G` ...)" | CONF C15 | `B → BQ` or keep |
| `precompute_and_caching.md:112` | "the `⟨O, H⟩` pairing exact" | CL | "the `⟨U, H⟩` pairing" |
| `batching_internals.md:49` | "(`U`,`P`,`Q`,`O`,`G`,`H`,`B`)" | CL, T4S | §4 |
| `testing_strategy.md:40` | "claiming `rR+rL` rank" | WID | optional `r_right + r_left` |
| `testing_strategy.md:41, :52` | T4S eqs; "`[Q | P]`" | T4S | keep |
| `signature_style.md:56-59` | "`elm_shape=C+(rLi,nUi,rL(i+1))`" ... | WID, CONF C18 | follow code |
| `deferred_and_rejected.md:197` | "`E_S = ⊗_k (U_k^⊥ if k ∈ S else U_k)`" | CL, CONF C7 | "`(D_k^⊥ if k ∈ S else D_k)`" |
| `deferred_and_rejected.md:186` | "`ut3_down_orthogonalize_tucker_cores(x)`" | ADJ | stays |
| `uniform_svd_prefix_orthogonalization.md:79` | "named `down_svd_*` / `left_svd_*` / `right_svd_*`" | ADJ | stays |
| `uniform_svd_prefix_orthogonalization.md:17, 21, 42`; `uniform_polymorphism.md:56` | `B = U·(S Vᵀ)`, `Q = U` | — | SVD factor; unrelated |

### 6.6 `docs/padded_frame_cone.tex` (hand rewrite; rebuild the PDF)

| loc | current | cat | proposed |
|---|---|---|---|
| :74-75 | "orthonormal frame cores in four \emph{stores} (up, down, left, right)" | SW | "(down, up, left, right)" |
| :96-97 | "the frame $U_{x\to t}$ ... $U_{x\to t}$ lies in $H_x\otimes\bigotimes U_{y\to x}$" | CONF C10 | per decision |
| :98-99 | "So one \emph{frame core} per directed edge stores the frame, and the four directions are exactly the library's four stores (Table~\ref{tab:stores})." | CONV | "... and the library's four stores are the four kinds of directed edge, each named for the side its core spans, which is the side the edge points away from (Table~\ref{tab:stores})." |
| :108 | "$\ell_i\to c_i$ & \code{up[i]} & $U_i\in\R^{n_{U,i}\times N_i}$ & the Tucker index" | FIG, SW, CL, WID, CONF C4 | "$\ell_i\to c_i$ & \code{down[i]} & $D_i\in\R^{n_{D,i}\times N_i}$ & the Tucker index" |
| :109 | "$c_i\to \ell_i$ & \code{down[i]} & $O_i\in\R^{r_{L,i}\times n_{D,i}\times r_{R,i+1}}$ & the down index" | FIG, SW, CL, WID | "$c_i\to \ell_i$ & \code{up[i]} & $U_i\in\R^{r_{L,i}\times n_{U,i}\times r_{R,i+1}}$ & the up core's mode index" |
| :110-111 | "$L_i\in\R^{r_{L,i}\times n_{U,i}\times r_{L,i+1}}$", "$R_i\in\R^{r_{R,i}\times n_{U,i}\times r_{R,i+1}}$" | FIG, WID | "$n_{D,i}$" in both |
| :113-114 | caption | — | add: "Our $D_i$ and $U_i$ are T4S's $U_i$ and $O_i$." (T4S-CITE) |
| :140-141, 174, 195, 206-211 | matrix case $A=U\Sigma V^\T$, $P\perp U$ | CONF C11 | report |
| :152-153, 171 | "$C'_{x\to t}\ominus U_{x\to t}$", "$U_{t\to x}$" | CONF C10 | per decision |
| :284-292 | messages $\Phi$, $\Xi$, $D_{a\to b}$, $o$ with $U'$ | CONF C4, C10 | per decision |
| :310 | "Stage 1 (old frames): down messages; left messages from the far end; right messages from the near end; up messages" | SW, CONF C19 | "Stage 1 (old frames): up-store messages ($c_i\to\ell_i$); left ...; right ...; down-store messages ($\ell_i\to c_i$)" |
| :312, :318 | `\code{_down_message}`, `\code{_up_message}`, `\code{_up_edge}`, `\code{_down_edge}` | SW (follow code) | follow code |
| :314-315 | "Stage 2 (message order): up edges (one per sharing group, ...); the left sweep ...; the right sweep ...; down edges. At each: ..., $D$, movement." | SW, CONF C19, C4 | "Stage 2 (message order): down-store edges $\ell_i\to c_i$ (one per sharing group, ...); the left sweep; the right sweep; up-store edges $c_i\to\ell_i$. ..." |
| :339 | "(tags up 0, down 1, left 2, right 3; ..." | SW, hazard | "(tags: down 0, up 1, left 2, right 3, the 2026.4.0 integers; ..." |
| :362 | "A TT or down seed row is read in the slot coordinates of the neighbours' frames" | SW | "A seed row of a TT-shaped store (left, right or up) is read ..." |
| :384 | "step updating up and left (\code{'UL'}); $F\gets$ step updating right and down (\code{'RO'})" | SW (update) | "step updating down and left (\code{('down','left')}); ... right and up (\code{('right','up')})" |
| :387, :412 | "\code{'UL'}" | SW (update) | word form |
| :403-404 | "Updating the stores pointing toward the root $c_{d-1}$ (up, left) and then those pointing away (right, down)" | SW, CONF C19 | "Updating the stores whose edges point toward the root $c_{d-1}$ (down, left) and then those whose edges point away (right, up)" |
| :83, :95, :165-166, :487-488 | T4S citations | T4S | keep |

### 6.7 `docs/exact_hessian_two_hole.tex` (T4S notation; rebuild the PDF)

| loc | current | cat | proposed |
|---|---|---|---|
| :115-117 | "At a point $X$ the orthogonal frame (T4S App. A.1, A.3, Algorithm 11; ...) is $(U_i, O_i, P_i, Q_i)$ = \code{T3Frame.data} = (up, down, left, right):" | SW, T4S, CONF C1 | "... is $(U_i, O_i, P_i, Q_i)$ in T4S's letters, \code{T3Frame.data} = (down, up, left, right) in the code's words (our $D_i, U_i, L_i, R_i$):" |
| :119-120, :122, :306 | "$n^D_i$" (old down width: $O_i$'s middle leg; $\delta U_i\in\R^{n^D_i\times N_i}$; $S_i^\T\in\R^{n^D_i\times n_i}$) | WID, CONF C1 | "$n^O_i$" (the note's letters) or "$n^{\mathrm{up}}_i$" |
| :893 | "$\max(\sum_{i\in g}n^D_i,n_g)$" | WID | same as above |
| :848 | "\code{frame_data[5]}'s up and left masks" | SW | "down and left masks" |
| :103-105, 122-132, 316-329, 359-377, 399-419, 478-545, 575-576, 600, 623-651, 670, 748, 781-794, 829-832, 860-895, 920-922, 934 | T4S letters $U$, $O$, $P$, $Q$, $\delta U$, $\dot U$, $U_g$ | T4S, CONF C9 | keep |
| :110-111, 210-262, 283, 322, 394, 551, 1014-1026; :44, :573 | $D\Phi$, $D^2\Phi$, $D_\zeta P$; $\Dsh$ | CONF C1, C2 | report |
| :150-192, :338-375 | two tikz figures with $U_i$, $O_i$ | FIG (conditional) | §3 |

### 6.8 `docs/shared_t3_math.tex` (rebuild the PDF if edited)

| loc | current | cat | proposed |
|---|---|---|---|
| :196 | "apply $Y^\top$ to the up-leg of every core in $g$" | SW, hand | "... to the Tucker leg (mode leg) of every TT core in $g$" |
| :286 | "$(U_g^\top U_i)$ to the up-leg of core $i$" | SW, hand | "... to the Tucker leg of TT core $i$" |
| :625 | "content in the new up-slots" | SW | "content in the new Tucker slots" |
| :366-367, :406, :491 | "$n_{D,i}$" (old down width) | WID, CONF C9 | "$n_{\mathrm{up},i}$" (this note's $U$ is the Tucker factor) |
| :80 | "\emph{up-matricization} of core $G_i$ (Tucker leg to rows)" | ADJ | stays (method convention) |
| :96 | "mode-$i$ down-orthogonalization" | ADJ | stays |
| :67-492 (~45 lines) | $U_i$, $U_g$, $\dot U$ (T3 Tucker factors) | CONF C9 | keep; add a notation line |
| :358 | "$O_g$ of $U_g$" | CONF C17 | none |

### 6.9 `docs/symmetric_probe_derivatives.tex` (no PDF in repo)

| loc | current | cat | proposed |
|---|---|---|---|
| :373-374 | "$((\mathbf U_i),(\mathbf O_i),(\mathbf P_i),(\mathbf Q_i))$ (up-Tucker, down-, left-, and right-TT cores)" | SW, T4S, hand | "... (in T4S's letters: the code's down (Tucker), up, left and right cores)" |
| :491 | "(its left and right bonds and the up-edge to its Tucker core)" | SW (picture), CONF C19 | "(its left and right bonds and the edge down to its Tucker core)" |
| :500 | "the Tucker core $\delta\mathbf U_i$ has one internal edge (the down-edge to its TT core)" | SW (picture), CONF C19 | "(the edge up to its TT core)" |
| :66-301, 379-631 | T4S letters | T4S, CONF C9 | keep |
| :111 | "$D^t \mathbf z_i(\mathbf W)\mathbf P^t$" | CONF C3 | report |
| :596-597 | "shifts \emph{up}", "shifts \emph{down}" | — | unrelated (jet order) |

### 6.10 `docs/pad_safe_svd.tex`, `docs/ttm_t3m_ht_note.tex`

| loc | current | cat | proposed |
|---|---|---|---|
| `pad_safe_svd.tex:76` | "the Tucker down-orthogonalization, both TT sweeps, the down step, and the T3-SVD scan" | SW, ADJ | "the Tucker down-orthogonalization, both TT sweeps, the TT up-orthogonalization, and the T3-SVD scan" |
| `pad_safe_svd.tex:301-302` | "the Tucker down-orthogonalization, both TT sweeps (...), the down step, and the T3-SVD scan" | SW, ADJ | same |
| `pad_safe_svd.tex` (U throughout, "Method~D") | SVD factor; design name | — | unrelated |
| `ttm_t3m_ht_note.tex:163, :250` | "down-orthogonalize the Tucker factors" | ADJ | stays (right) |
| `ttm_t3m_ht_note.tex:74-238` (~15 lines) | $U_i$ Tucker factors / HT leaf frames | CONF C9 | keep |

## 7. `dev/glossary_naming_decisions.md`: rows that mention a store word or letter

Each must be re-expressed in the new convention; the decision supersedes the HELD state. "→" gives the new form.

| wl line | item / target | current | re-expressed |
|---|---|---|---|
| :30 | glossary-edit note N17 | "once Nick decides (the adjectives stay; the store names may change), release or replace the HELD rows ..." | "Decided 2026-11-10: the adjectives stay and the stores flip (2026.5.0 store rename). Release the HELD rows as re-expressed below; the glossary's **orthogonal core** and **frame cores** entries change with the rename slice." |
| :40 | decision 1 (Tucker core) | "`tucker_cores` / `up_tucker_cores` / `tucker_variations`" | "`tucker_cores` / `down_tucker_cores` / `tucker_variations`" |
| :81 | **N4** | "the down store and its cores are named **down, `O`** (not `D`, not "outer" as a name)... `align_padded(update=...)` letters `'U'`, `'O'`, `'L'`, `'R'`; `T3Frame.data = (up, down, left, right)`" | "the TT-shaped store and its cores are named **up, `U`**, and the Tucker store **down, `D`** (a store is named for the side its cores occupy; Nick 2026-11-10). Not "outer" as a name (T4S's word, an alias of up-orthogonal); T4S's `Oᵢ` is our `Uᵢ` and its `Uᵢ` our `Dᵢ`. Verification: `update=` takes store words; `T3Frame.data = (down, up, left, right)`. Not "outer TT bonds" (`tucker_tensor_train.py:962`), which stay" |
| :93 | **N17** | "(**HELD**: Nick is deciding) the orthogonality adjectives follow the **method convention** ... a store rename is being assessed. The HELD rows below assume the adjectives stay and the frame pages change; a store rename would supersede them" | "(**RESOLVED** 2026-11-10) the adjectives follow the method convention and stay; the store names were changed to agree (2026.5.0). Every former HELD row is released below with the new store words; `fv_conversions.py:184`'s alias pair and the `ranks.py:644` comment are rename-slice edits" |
| :101 | E8 | "the extra *bonds* are `rLd` and `rR0`" | "`r_left[d]` and `r_right[0]`" (or keep the symbols) |
| :103 | E10 | "the Tucker gauge condition is written `Uᵢᵀ Vᵢ = 0` | `Ui`, `Vi` are stored `(n, N)`; ... `U_i V_i^T = 0`" | "... written `Uᵢᵀ Vᵢ = 0` (old letters) | `Di`, `Vi` are stored `(n, N)`; the code states `D_i V_i^T = 0` after the rename" |
| :110 | E17 | "`fv_variation_shapes` gives a TT variation the shape `(rL_i, nU_i, rR_(i+1))` ..., and the left/right frame cores carry `nU_i`" | "`(r_left_i, n_down_i, r_right_(i+1))` ... carry `n_down_i`" (spelling per C18) |
| :123 | M/U note | "called with the middle slot on the down cores `O` (`P_O`, the down edge's `g_new`, ...)" | "on the up cores `U` (`P_O` until the code renames it, the up edge's `g_new`, ...)" |
| :127 | M/U note | "`_up_matricization`" | stays (ORTH) |
| :197 | `frame_variations.md:57` E10 | "`Uᵢᵀ Vᵢ = 0`" → "`Uᵢ Vᵢᵀ = 0`" | → "`Dᵢ Vᵢᵀ = 0`" |
| :198 | `frame_variations.md:71` N4 | "down/outer" → "down" | superseded: §5.1 row :70-71 ("`up_ranks` (the mode leg of each up core `Uᵢ`)") |
| :200 | `frame_variations.md:83` M/U | "`nᵢ × (rᵢ·rᵢ₊₁)` unfolding" → "matricization" | keep; the same line also takes `Oᵢ (down / outer)` → `Uᵢ (up)` |
| :201 | `frame_variations.md:125` E8 | "padding `Ld`/`R0` bonds" → "carrier bonds `rLd` / `rR0`" | "carrier bonds `r_left[d]` / `r_right[0]`" (optional) |
| :300 | `user_guide.rst:97` M/U | "Outer TT core unfoldings" → "TT core matricizations" | merge with :301 |
| :301 | `user_guide.rst:97` N17 | HELD: "Up (outer) TT core unfoldings are full rank" | "TT core up-matricizations are full rank" |
| :305 | `user_guide.rst:142` N17 | HELD: "..., up-orthogonal" | "up_tt_cores = (U0,...,U(d-1)), up-orthogonal" (§5.3) |
| :319 | `curvature_internals.md:28` E6 | quoted text contains "`S_iᵀ = ⟨O_i, H_i⟩`" | the replacement text takes "`⟨U_i, H_i⟩`"; whichever slice runs second re-greps |
| :580 | `fv_operations.py:640` E6 | "``down`` -> V's ``nD`` leg; ``up``/``left``/``right`` -> H's" | "``up`` -> V's ``n_up`` leg; ``down``/``left``/``right`` -> H's" |
| :576 | `fv_operations.py:98` E17 | "`(rL_i, nU_i, rR_(i+1))`" | per C18 |
| :596-598 | `linalg.py` `up_svd` M/U | "up matricization of U" | stays (ORTH; `U` is the SVD factor) |
| :629 | `padded_frames.py:148` M/U | "the mode size (up) or the real rows of the core's matricization (down) or unfolding (left, right)" | "the mode size (down) or the real rows of the core's matricization (up) or unfolding (left, right)" |
| :645 | `probing.py:646` N17 | HELD "down-orthogonal elements" | "down_tucker_supercore ... down-orthogonal elements" |
| :646 | `probing.py:647` N17 | HELD "up-orthogonal elements" | "up_tt_supercore ... up-orthogonal elements" |
| :650 | `probing.py:1221` N17 | as :645 | as :645 |
| :651 | `probing.py:1222` N17 | as :646 | as :646 |
| :659 | `ranks.py:651` E6 | "the uniform frame's left/down masks were too small" | "left/up masks" |
| :736 | `t3_orthogonalization.py:106` N17 | HELD "keep, or "up-orthogonal tt core"" | "up-orthogonal tt core" (the local `Oaxb` follows code) |
| :737 | `t3_orthogonalization.py:147` N17 | HELD "down-orthogonal tucker core" | released as is |
| :738-739 | `t3_{down,up}_svd_tt_core` M/U | "outer unfolding"; "contract ... down into the tucker core below" | ADJ "outer" → "up"; "down into the tucker core below" stays (spatial) |
| :752-753 | `tv_operations.py:164, 181` N4 | "down/outer" → "down" | → "up" |
| :771-772 | `ufv_operations.py:277, 294` N4 | "down/outer" → "down" | → "up" |
| :826 | `ut3_orthogonalization.py:189` N17 | HELD "new_tt = up-orthogonal O" | "new_tt = up-orthogonal U" |
| :870 | `frame_variations_format.py:63` N4 | "diagram cores `D0`, `D1`, `D2`" → "`O0`, `O1`, `O2`" | → "`U0`, `U1`, `U2`"; the same diagram's Tucker `Ui` → `Di` |
| :871 | `:72` N4 | "`down_tt_cores = (D0, ..., D(d-1))`" → "(O0, ...)" | → "`up_tt_cores = (U0, ..., U(d-1))`" |
| :872-874 | `:73, 74, 78` E17 | "`elm_shape=(rLi, nUi, rL(i+1))`" etc. | width spelling per C18 (`nUi` → `n_down_i`, or swapped letters `nDi`) |
| :876, :878 | `:93, :105` E17 | "Ud" → "U(d-1)" | → "D(d-1)" |
| :877 | `:101` N4 | rank diagram core "D1" (next line `nO1`) → "O1" | → "U1" (next line `n_up_1`) |
| :879 | `:118` N17 | "down_tt_cores = (O0,...,O(d-1)), outer-orthogonal  O_aib O_ajb = delta_ij" HELD | "up_tt_cores = (U0,...,U(d-1)), up-orthogonal  U_aib U_ajb = delta_ij" |
| :883 | `:355` N4 | "down/outer D_i (all i): ``einsum(..., D, D) = I``" → "down O_i ..." | → "up U_i (all i): ``einsum('...iaj,...ibj->...ab', U, U) = I``" |
| :884 | `:365` N17 | "(left/right/outer/Tucker)" HELD | "(left/right/up/down)" |
| :885 | `:1271` N4 | `fv_to_t3` diagram core "D2" → "O2" | → "U2" |
| :886 | `:1371` N17 | "tucker_cores = (U0,U1, U2, U3), up orthogonal" HELD | "down_tucker_cores = (D0, D1, D2, D3), down-orthogonal" |
| :887 | `:1372` N17 | "down_tt_cores = (O0, ..., O3), down orthogonal" HELD | "up_tt_cores = (U0, ..., U3), up-orthogonal" |
| :902 | `manifold.py:1343` N17 | "(the left/right/outer-orthogonal representation)" HELD | "(the left/right/up-orthogonal representation)" or "(the orthogonal representation)" |
| :903 | `manifold.py:1638` E6 | "a slack frame with ``nD != nU``" | "``n_up != n_down``" |
| :940 | `tucker_tensor_train.py:615` M/U | "Down TT core unfoldings are full rank" | "TT core up-matricizations are full rank" (store word "Down" was the old one) |
| :945 | `:2956` M/U | "contract non-orthogonal factor down into the tucker core below" | stays (spatial) |
| :946-947 | `:2978, :3002` N17 | HELD "up-orthogonal" | released |
| :948 | `:3108` N17 | HELD "down-orthogonal" | released |
| :968 | `uniform_frame_variations_format.py:1122` N17 | HELD | "down_tucker_cores = (D0, ..., D3), down-orthogonal" |
| :969 | `:1125` N17 | HELD "down_tt_cores = (O0, O1, O2, O3), up-orthogonal" | "up_tt_cores = (U0, U1, U2, U3), up-orthogonal" |

Not store rows despite a match (false positives of a letter scan): every E6 row whose text holds a decision code
(`D-39` … `D-49`, `D1`–`D3`, "Phase D") or the oracle name `O2`/`O-2`, and the M/U rows that only say "M/U".

## 8. Records left alone

`CHANGELOG.md` historical entries stay as written (the work list's rule; the new entry says old entries use the old
words): `:49`, `:51` (`'UL'`/`'RO'`), `:183` (`(2,)+W+K+C+(nOi,)`), `:267`, `:298` (`U_M`, `U̇_g`), `:575`
(`max|UᵀV|`), **`:714` ("Tucker up-SVD", a backwards adjective)**, `:717` (`up_tucker_cores`), `:798-799`
(`rL_i*rR_i`), `:832-833` ("replacing the down cores", "the up width"), `:871` (`U_i + V_i`), `:900`
(`up`/`down`/...), plus three T4S citations (`:636`, `:638`, `:789`). `docs/release_notes.md` historical sections:
no store word, width or frame letter. The root `release_notes_2026.{3,4}.0.md` are untracked GitHub release bodies
(records; the 2026.4.0 one repeats CHANGELOG `:49`, `:51`).

## 9. Unrelated hits (accounted, no row)

Big-O `O(...)` (README, index, user guide, t3m_methods, transposes, numerical_contracts, chunking, deferred,
signature_style, curvature, catalog, pad_safe_svd, shared_t3_math, exact_hessian, release notes, CHANGELOG): ~60
lines. English "up"/"down" ("up to", "up front", "warm-up", "blow up", "picked up", "down to", "downstream", "write
down", "down-weight"): ~95 lines. Jet-order shifts (`_shift_up`, "down-shift", `\Dsh`): 6. Decision/plan codes
(`D-41`, `D1`, "Phase D", `U1–U6`, `O2`, `O-2`): ~25. "Method D" (pad_safe_svd): 4. Author initials and "Ph.D.":
~12. "OO frontend": 6. SVD `U` with no frame letter on the page (pad_safe_svd, uniform pages): ~25. "outer"
meaning outer product/loop/most: ~20.

## Appendix A. `CLAUDE.md` (may also be in inventory 1 or 2)

| loc | current | cat | proposed |
|---|---|---|---|
| :68-69 | "Notation map (paper ↔ code): `U`=up_tucker, `P`=left_tt, `Q`=right_tt, `O`=down_tt (outer); `δU`=tucker_variations (V), `δG`=tt_variations (H)." | T4S, SW | "Notation map (paper ↔ code): `U` = down_tucker (our `D`), `P` = left_tt (`L`), `Q` = right_tt (`R`), `O` = up_tt (outer = up-orthogonal; our `U`); `δU` = tucker_variations (`V`), `δG` = tt_variations (`H`)." |
| :115 | "(`P,Q,O → G`, paper §6.3)" | T4S | keep |
| :158 | "**`T3FrameWeights`** (`up/down/left/right`, each len d)" | SW | "(`down/up/left/right`, each len d)" |
| :182 | "disjoint from the core/variation symbols `U`/`P`/`Q`/`O`/`G`/`H`/`V`" | CL | add `D`, `L`, `R` |
| :213-214 | "`T3Frame.data = (up, down, left, right) = (U, O, P, Q)` (the `O`/down core is called **`down_tt_cores`**, not "outer")" | SW, CL, T4S | "`T3Frame.data = (down, up, left, right)`, our `(D, U, L, R)`, T4S's `(U, O, P, Q)` (the up core is **`up_tt_cores`**; T4S says "outer")" |
| :275 | "Uppercase single-letter core names (`U V G P Q O …`)" | CL | "(`D U L R V H G P Q O …`)" |
