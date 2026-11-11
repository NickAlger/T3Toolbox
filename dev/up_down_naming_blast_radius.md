# The up/down store names: blast radius and resolutions (2026-11-10)

**Question.** The frame stores `(up, down, left, right)` call the `U` cores "up" and the `O` cores "down", while the
orthogonality adjectives (`down_orthogonalize_tucker_cores`, `up_orthogonalize_tt_cores`) say the `U` cores are
down-orthogonal and the `O` cores up-orthogonal. How large is a rename, and which one should we do?

**Answer.** The store names reach **2,011 occurrences on ~1,430 lines in 78 files** (plus 425 lines for the
`nD` letter). The public surface that would change is **16 spellings at 44 sites**: 22 attributes on 7 classes and 22
backend keywords. Nothing else in the public API carries the store words. Method names, `update='ULRO'`, tuple order
and the save format are untouched. **Recommendation: (A) rename the stores by their letters, "the U store" and "the
O store"**, with no aliases, in one breaking release. A pure flip (C) costs the same and leaves 22 of the 44 public
sites still spelled the same but meaning the other store.

Scope: `t3toolbox/`, `tests/`, `examples/`, `docs/*.md|rst|tex`, `docs/contributor/`, `README.md`, `CHANGELOG.md`,
`CLAUDE.md`: 188 files, 101k lines, working tree of 2026-11-10 (including the uncommitted `docs/glossary.md`).
Out of scope: `dev/` records and generated `docs/autoapi`/`_build`.

## 0. Why the names clash: two conventions in one tuple

Every frame core sits on a directed edge of the T3 tree and is orthonormal over the side the edge comes **from**.

| store | core | directed edge | orthonormal over | name follows |
|---|---|---|---|---|
| left | `Lᵢ` | `cᵢ → cᵢ₊₁` (points right) | `(rᵢ, nᵢ)`, the left side | the side it comes from (= its adjective) |
| right | `Rᵢ` | `cᵢ → cᵢ₋₁` (points left) | `(nᵢ, rᵢ₊₁)`, the right side | the side it comes from |
| **up** | `Uᵢ` | `lᵢ → cᵢ` (points up) | `Nᵢ`, the leaf below | the direction it **points** |
| **down** | `Oᵢ` | `cᵢ → lᵢ` (points down) | `(rᵢ, rᵢ₊₁)`, the chain above | the direction it **points** |

So taken alone, each pair is consistent: left/right name the side a core comes from, and up/down name where it
points. The code even relies on the up/down reading: `padded_frames._up_edge` is "the edges leaf → core", and
`padded_frame_cone.tex:403` groups "the stores pointing toward the root (up, left)". Mixing the two conventions in
one tuple is the accident. The paper uses neither word: it writes `Uᵢ, Pᵢ, Qᵢ, Oᵢ` and "outer orthogonal" (T4S App.
A.3, Alg. 11).

## 1. Inventory

**Method.** Python files were split with `tokenize` into identifiers, string literals, comments and docstrings;
`.md/.rst/.tex` were scanned as text. Every word with an `up`/`down` component (`(^|_|-)(up|down|upward|downward)`)
was classified. Identifiers were classified from an exact table. Bare words were classified by ordered context rules,
then every residual hit (~140) was read and assigned by hand, and false positives were spot-checked (~60 overrides).
The scripts lived in the session scratchpad. The family counts below can be reproduced with grep, and a few are
quoted.

Categories:
- **STORE**: the U/O store and anything named after it. Examples: `up_tucker_cores`, `down_ranks`, `up_mask`,
  `'up'`, `_up_edge`, `_u_down_messages`, `(up, down, left, right)`, "the down cores".
- **ORTH**: an orthogonality adjective or SVD/unfolding direction, method convention. Examples:
  `down_orthogonalize_tucker_cores`, `up_svd_pair`, `t3_down_svd_tt_core`, `_up_matricization`, "down-orthogonal".
  These stay.
- **MSG**: travel direction of sampling messages and tree-spatial words. Examples: "contract up" (T4S Alg. 5),
  "up-index ξ", "adjoint-up lift", "upward edge variables", "pushing the remainder down into the Tucker core". These
  stay in every option.
- **UNREL**: unrelated. Examples: "up to", "up front", "warm-up", "one level up", "down-weights", `_shift_up*` (jet
  order shifts).

`_up_matricization` is filed under ORTH, not under STORE as the brief guessed. It is the `n × (rL·rR)` unfolding, the
same "up" as `up_svd` and `up_orthogonalize_tt_cores`.

### 1.1 Counts by area

| area | STORE | ORTH | MSG | UNREL |
|---|---|---|---|---|
| library frontend `t3toolbox/*.py` | 368 | 55 | 6 | 22 |
| library backend `t3toolbox/backend/` | 1,074 | 107 | 49 | 95 |
| tests | 431 (26 files) | 44 | 7 | 33 |
| examples | **0** | 0 | 0 | 6 |
| user docs (`docs/*.md`, `*.rst`) | 64 | 13 | 4 | 32 |
| contributor docs | 29 | 2 | 0 | 35 |
| math notes (`*.tex`) | 32 (5 files) | 6 | 2 | 14 |
| CHANGELOG / CLAUDE.md / README | 5 / 8 / 0 | 1 / 0 / 0 | 0 | 11 / 4 / 0 |
| **total** | **2,011** | **228** | **68** | **252** |

STORE occurrences by kind:

| area | code identifiers | string literals | comments | docstrings | docs/tex |
|---|---|---|---|---|---|
| frontend | 163 | 32 | 42 | 131 | |
| backend | 628 | 38 | 286 | 122 | |
| tests | 319 | 28 | 44 | 40 | |
| docs, tex, CHANGELOG, CLAUDE.md | | | | | 138 |

- **Identifiers:** 63 distinct STORE names in code, 1,110 tokens. Of these, 244 tokens are the bare locals `up` and
  `down`, and 5 are `_down`. In library code, 483 tokens use a public spelling and 308 tokens are 38 private names
  (`up_sc`, `up_widths`, `_up_edge`, `_down_message`, `_u_up_messages`, `variations_down`, …).
- **Identifier mentions in prose:** 264, such as `up_ranks` written in a docstring.
- **Bare words in prose:** 539 (242 in listings like `(up, down, left, right)` or `up/down`, 70 quoted, 227 in free
  prose). Of the 1,038 exact `up`/`down` tokens across everything, 830 are store names and 208 are not (159 English,
  32 adjective, 17 direction). **A sed on bare words is therefore unsafe.**
- **String literals:** 70 in the library and 28 in tests (6 test files). They are store labels in error messages,
  `_STORE_NAMES`, and the `_SEED_TAGS` keys. Some of the test literals assert on message text
  (`test_padded_frame.py:405`, `'old_frame down\[1\]'`, …).
- **Doctests:** 76 STORE hits on `>>>` lines in 8 files. These are CI-checked, so a missed identifier fails loudly.
  Prose has no such net.
- **Family counts** (`grep -rhoE` over the scope):

  | pattern | hits | files |
  |---|---|---|
  | `\b(up_tucker_cores\|down_tt_cores)(_dot)?\b` | 440 | 28 |
  | `\b(up\|down)_ranks\b` | 159 | 23 |
  | `\b(variations_)?(up\|down)_mask\b` | 124 | 9 |
  | `\b(up_tucker\|down_tt)_supercore\b` | 62 | 9 |
  | `\b(up\|down)_(weights\|weight_supercore)\b` | 37 | 6 |
  | `\b(t3_\|ut3_\|_)?(up\|down)_(orthogonalize\|svd)\w*` (ORTH, stays) | 142 | 18 |
  | `\b_?shift_(up\|down)\w*` (UNREL) | 48 | 2 |
  | `\b(up\|down)\b` (mixed, all categories) | 1,167 | 122 |

### 1.2 The letter `D` for the O store

The O store has two letters today: `nO` (108 tokens, mostly the sampling code) and `nD` (418 tokens). Related forms:
`kD`/`kDi` (26), `mD0` (37), `dn` (67), and `D0…D(d-1)`/`Dᵢ` as core names (about 46). Together these sit on **425 lines in
46 files**. `nU` (758) reads the same whether the U means "up" or "U". `nD` reads only as "down". It is public in
two forms: the property `nD` on `UT3Frame`, `UT3Variations` and `UT3FrameWeights`, and the keyword `nD=` on
`UT3Frame.from_t3frame`, `UT3Variations.from_t3variations`, `UT3FrameWeights.from_t3frameweights`,
`UT3Tangent.from_t3tangent`, `ufv_conversions.t3frame_to_ut3frame`, `t3variations_to_ut3variations`,
`t3frameweights_to_ut3frameweights` and `ufv_masking.ufv_make_frame_masks`.

### 1.3 Public surface that a store rename changes

Found with an `ast` walk of every public class and function.

- **Attributes: 14 spellings at 22 sites on 7 classes.**
  - `T3Frame.up_tucker_cores`, `.down_tt_cores`, `.up_ranks`, `.down_ranks`. The first two are dataclass fields,
    so they are also constructor keywords and `dataclasses.replace` keys.
  - `T3FrameWeights.up_weights`, `.down_weights`, `.up_ranks`, `.down_ranks`.
  - `UT3Frame.up_tucker_supercore`, `.down_tt_supercore`, `.up_ranks`, `.down_ranks`.
  - `UT3FrameMasks.up_mask`, `.down_mask`.
  - `UT3VariationsMasks.variations_up_mask`, `.variations_down_mask`.
  - `UT3Variations.up_ranks`, `.down_ranks`.
  - `UT3FrameWeights.up_weight_supercore`, `.down_weight_supercore`, `.up_ranks`, `.down_ranks`.
- **Backend keywords: 6 spellings at 22 parameters in 16 functions.**
  - `probing.compute_{xi, deta_tilde, deta_tilde_tangent}(up_tucker_cores[, up_tucker_cores_dot])`.
  - `probing.compute_{eta, sigma, tau, dxi_tilde, dxi_tilde_tangent}(down_tt_cores[, down_tt_cores_dot])`.
  - The six `sampling_derivatives.compute_*_jets` twins.
  - `ranks.frame_has_minimal_ranks(up_ranks, down_ranks)` and `ufv_masking.ufv_make_frame_masks(up_ranks, down_ranks)`.
- **Optional:** `nD` (3 properties, 8 keywords; see §1.2).
- **User-visible text, not API:** `T3Frame.__repr__` prints `up_ranks=` (`frame_variations_format.py:288`), and
  the library's error messages name the stores (part of the 70 library string literals).

**Unchanged in every option:**
- The 18 public ORTH names: the `TuckerTensorTrain` and `UniformTuckerTensorTrain` `*_orthogonalize_*` and `*_svd_*`
  methods, `linalg.up_svd`/`up_svd_pair`/`down_svd_pair`, and the `t3_`/`ut3_orthogonalization` functions.
- The `update` letters `'U'`/`'O'`.
- Every tuple's **order**: `.data`, `.structure`, the `compute_orthogonal_representation_ranks` output, and the
  mask tuples.
- The save format: `save_core_families` and `ufv_save` key arrays by position (`'f%d_%d'`), never by name.
- `nU`.
- One hazard: `_SEED_TAGS = {'up': 0, 'down': 1, …}` must keep its **integers**, or `padded_seed` streams change.
  The test oracle `test_padded_frame.py:91` keeps its own copy of the table, so it would not catch a slip.

### 1.4 Adjectives that use the store convention (wrong in every option)

These are 26 lines in 13 files, and every option has to fix them. They are the glossary's N17 rows.
`dev/glossary_naming_decisions.md` lists 14 of them:

- `frame_variations_format.py:1371-1372`
- `uniform_frame_variations_format.py:1122,1125`
- `probing.py:646-647,1221-1222`
- `tucker_tensor_train.py:2978,3002,3108`
- `ut3_orthogonalization.py:189`
- `user_guide.rst:142`
- `t3_orthogonalization.py:147`

**Not yet in the record (12):**
- `t3_orthogonalization.py:171`: `t3_down_orthogonalize_tucker_cores` says "Orthogonalize Tucker cores
  **upwards**".
- `geometry.py:348`: "Tucker up-orth".
- `fv_conversions.py:181-185`: local aliases `up_orthogonalize_tucker_cores = t3_down_orthogonalize_tucker_cores`
  and `down_orthogonalize_tt_cores = t3_up_orthogonalize_tt_cores`.
- `fv_conversions.py:219,251`: "Orthogonalize Tucker cores upward…", "…TT cores downward…".
- "Tucker up-SVD" for `t3_down_svd_tucker_core`: `t3_svd.py:312,345` and `test_tucker_tensor_train.py:3202`.
  `CHANGELOG.md:714` has it too, but it is a record and stays.
- `test_frame_variations_format.py:447`: "U: up-orthogonal".

### 1.5 Downstream and precedent

- **Downstream:**
  - T3Polynomial uses **none** of the 16 public spellings. It has one local unpack,
    `up, down, left, right = frame` (`scripts/g01_g0_common.py:121`).
  - The research repo has only local `nU`/`nD` names, in 15 prototype files.
  - PyPI users of 2026.4.0 are unknown.
- **House precedent** (2026.2.0 and 2026.3.0): renames are **not aliased**, "so a breaking release breaks loudly once
  rather than carrying two names" (`GaussNewtonModel` → `LocalModel`; `UniformGaussNewtonModel` and
  `stats['newton']` removed). The pattern is a CHANGELOG "Changed — breaking" bullet plus an "Upgrading from X"
  paragraph in `docs/release_notes.md` that names old and new. The glossary keeps retired names under "also:" for one
  release.

## 2. Resolutions

### A. Rename the U/O stores by their letters (recommended)

Prose says "the U store" and "the O store"; identifiers carry the letter; `ULRO` is unchanged.

Name candidates, checked for collisions:

| candidate | collisions / problems | verdict |
|---|---|---|
| **letters**: `U_ranks`, `O_ranks`, `U_tucker_cores`, `O_tt_cores`, `U_mask`, `variations_O_mask`, `nO` | Capital letters in attribute names break PEP 8, with precedent: `UT3Frame.nU/nD/rL/rR/N` and the `C/W/K` blocks (`naming_conventions.md` would add one line to its exceptions). `U` also names the SVD factor in `linalg`. `O` looks like `0` and like big-O in prose. Lower-case `u_*` is out because `u` already means *uniform* (`ut3_`, `ufv_`, `_u_down_messages`). | **best**. These letters are already the paper's, `update`'s, the diagrams', the glossary's and the sampling code's (`nO`). |
| **role**: `tucker_cores` / `outer_tt_cores`, `tucker_ranks` / `outer_ranks` | `tucker_ranks` is `TuckerTensorTrain.tucker_ranks`. `tucker_weights` is a `T3Weights` field that `from_t3weights` copies into **both** stores. `tucker_edge_mask` exists. "outer" has about 60 other uses (outer product 29, outer loop/steps ~30, outermost 20, outer tangent stack 7), and "outer cores/ranks" reads as the boundary cores/bonds (`tucker_tensor_train.py:962`, "outer TT bonds"). Glossary N4 already rejected "outer" as a name. | possible, but with more collisions |
| leaf / chain | "leaf" is the pytree word: `leaf_structure`, `ufv_leaf_structure` and `apply_func_to_leaf_subtrees` are public, with about 300 uses. | no |
| factor / complement, basis / environment | "factor" has about 1,500 uses (Tucker-factor alias, SVD factors). "complement" already means the gauge complement where `V` lives, which is not `O`. | no |

**Cost:**
- **Size:** 1,426 lines in 78 files (852 of them code or strings), or **~1,735 lines in 81 files with `nD → nO`**.
  This is about 1.7 % of the scanned lines.
- **Public API:** the 16 spellings at 44 sites, plus `nD` if chosen.
- **Tests:** 26 files, about 380 lines, including the message-text assertions.
- **Docs:** 20 files (15 pages and 5 tex notes), about 94 lines. The tex notes (32 hits) need their PDFs rebuilt;
  the decision record's rule of changing them "when the note is next rebuilt" also works. CLAUDE.md is affected.
  Historical CHANGELOG entries stay as they are.

**Mechanics:**
- The 60 compound STORE identifiers (the 63 minus bare `up`, `down`, `_down`) form an exact token map, which is
  **sed-safe with `\b`**. No STORE name is
  a bounded token inside an ORTH or UNREL name; for example, `\bup_tucker_supercore\b` does not match
  `masked_up_tucker_supercore`, and `up_sc` and `up_svd` are distinct. The same map covers the 264 prose mentions.
- The 244 bare `up`/`down` code tokens can be renamed at the tokenizer level, with 8 exceptions to exclude:
  `test_fitting.py:1148-1152` (a tangent named `up`) and `_curvature_oracles.py:529-536` (`up` = the ξ lambda).
- The 539 bare prose words have to be edited by hand. The 242 listings can be done with a regex and a review.
- Keep the `_SEED_TAGS` integers.
- Afterwards, `grep -rwE 'up|down'` should list only known non-store hits: the 208 bare tokens of §1.1 plus
  hyphenated forms such as `down-orthogonal` and `up-index`.
- The suite, doctests and `-W` docs build check the code half.

**Deprecation:** follow the house rule of no aliases. Old attributes raise `AttributeError` and old keywords raise
`TypeError`. Add a rename table in CHANGELOG and in "Upgrading from 2026.4.0", where a `\b`-bounded sed with the
16-spelling map is safe for user code. Aliases are not cheap here anyway. A warning property can alias any of the 22
attributes, but the 12 dataclass fields are also constructor keywords (each needs a custom `__init__`), and the 22
backend keywords would need `**kwargs` shims.

**Residual confusion:** low. left/right keep a directional name and U/O carry letters, which is asymmetric but not
contradictory. The adjectives then stand alone. Old records say up/down, and the glossary keeps them as retired
aliases.

### B. Rename only in prose and docstrings

- **What changes:** the 227 free-prose words, on about 189 lines in 49 files (86 backend, 57 tests, 31 frontend, 22
  tex, 29 docs; 2 more in CHANGELOG stay as records). A note in `frame_variations.md` and the glossary would say "U store (API: `up_*`)". The §1.4 fixes
  are also needed.
- **What it costs:** no tests, no API, no deprecation. The 242 listings and 70 quoted names cannot move, because they
  mirror code that still says `up`.
- **Residual confusion:** high and permanent. Every `frame.up_ranks` sits beside "U store". The "up store" alias
  never retires. New code keeps adding to the old names; 2026.4.0 alone added `_up_edge`, `_up_message`,
  `_u_up_messages` and `'up'` seed tags.
- **Fit with the other options:** B's prose is exactly A's prose, so B is a valid first step toward A. It is not
  compatible with C.

### C. Flip the directional words (U → "down", O → "up")

- **What it fixes:** all four stores then follow the come-from convention (§0), and the store names match the
  adjectives.
- **Footprint:** the same files and lines as A, with these differences:
  - **Silent meaning flip:** 22 of the 44 public sites keep their spelling and swap meaning. These are every
    `*_ranks`, mask and weights name, and the keywords `up_ranks=`/`down_ranks=`. Only the 4 core/supercore fields
    and the 18 core keywords vanish loudly (`up_tucker_cores` → `down_tucker_cores`). At a minimal frame `nU = nO`,
    so a stale `frame.up_ranks` returns plausible numbers. It is wrong only at slack, which the library deliberately
    allows (`frame_variations.md`, "Four rank stores").
  - **No deprecation path:** a meaning flip cannot be deprecated. Breaking loudly needs an interim release where
    neither name exists, which is A, followed by a second breaking release.
  - **Not sed-able:** the rename is a three-way swap through a placeholder, restricted to STORE tokens. A single
    stray hit among the 228 ORTH uses of the same words silently inverts a correct adjective.
  - **`nD` still has to become `nO`,** because D would now mean the U store.
  - **Old records keep the old meaning forever:** CHANGELOG, the published 2026.x docs, `dev/archive`, T3Polynomial
    notes and the tex PDFs.

**Rejected outright: (D) flip the adjectives instead.** That would mean calling U up-orthogonal and renaming 18
public methods. It contradicts what "left-orthogonal" means (orthonormal on the named side), and Nick has ruled the
adjectives right.

## 3. Recommendation and the decisions for Nick

**Do A with letters, unaliased, in one breaking release.** The words of the bad pair are removed rather than
re-pointed. The vocabulary that replaces them is already in use in the paper, `update='ULRO'` (the newest API), the
diagrams and the glossary. The cost is a token-map rename plus about 540 hand edits to prose. Unlike C, every stale
use fails loudly. The decisions:

1. **Letters or role names** (`tucker`/`outer`)? Recommend letters (collision table above).
2. **Spelling of the letter identifiers.** Recommend capitals (`U_ranks`, `O_tt_cores`, `variations_U_mask`), with
   the exception catalogued next to `nU`/`rL`. Lower-case `u_` clashes with *uniform*. The local spelling for the
   bare `up`/`down` tuples is the implementer's choice.
3. **`nD → nO`** (3 properties, 8 keywords, about 425 lines)? Recommend yes. Otherwise "D for down" survives in the
   public API.
4. **Release.** Recommend bundling into 2026.5.0 with the glossary, so the glossary defines the U/O stores once.
   That makes 2026.5.0 breaking. If it must stay additive, ship B's prose now and the identifiers in the next
   breaking release.
5. **Glossary record.** This reverses N4 ("named down, `O`") to "O store". N17 resolves as "method convention, no
   store name to clash with". Add the 12 sites in §1.4 to its work list.
