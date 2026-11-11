# Glossary naming decisions and the docs-consistency work list (2026.5.0)

Two things in one file: the record behind `docs/glossary.md` (each canonical name, its alternatives, the rule that
chose it), and the **work list** for the implementer who makes the docs and docstrings agree with the glossary in the
next step of this release. Built from the three scout reports (`dev/glossary_candidates_{A_user_docs,B_backend,
C_frontend}.md`); every ERROR below was checked against the code on 2026-11-10. The glossary task edited only
`docs/glossary.md`, `docs/index.rst` and the README's documentation line; everything else is listed here.

**Naming rule.** Canonical name by (a) the public API's spelling (identifiers, keywords, `update` letters, data-field
names), then (b) the most-used spelling in the rendered docs (`docs/*.md`, `docs/*.rst`, `docs/contributor/*.md`),
then (c) the math notes (`docs/*.tex`). A name that is wrong for the code skips to the next rule.

**Kinds.** NAMING: pick the canonical form. ERROR: a wrong claim (M/U: matricization vs unfolding). CODE: a behaviour fix for the implementer. Rows
marked **HELD** wait for Nick's decision on their item.

**How to use the work list.** The items are defined once in [Items](#items) (kind, decision or claim, and a one-line
verification note: what in the code decides it). The [work list by file](#work-list-by-file) then gives every edit:
line, enclosing heading (docs) or function/class (code), the item, the current wording and the replacement. Bulk
renames show the matched phrase; adapt grammar around it. Line numbers are as of commit `daa1e23d` plus this task's
uncommitted files; re-grep if the file has moved. `CHANGELOG.md` and the historical sections of `release_notes.md`
are records and are left alone, except item E3. Rows in `docs/*.tex` (the math notes) are lower priority: change them
when the note is next rebuilt.

**Glossary edits that go with this list** (same commit as the item):

- E8 and E9 rename two `frame_variations.md` headings that `docs/glossary.md` links to
  (`#d-cores-d1-bonds--and-the-extra-ld-r0`, `#a-tangent-is-a-sum-of-d-terms`); update those links and any other page
  linking to them (`grep -rn "frame_variations.md#" docs`). The `-W` docs build catches a missed one.
- N3: the glossary's **same-frame** entry says "SF in the contract tables"; change it to the new label.
- N17: once Nick decides (the adjectives stay; the store names may change), release or replace the HELD rows and
  update the glossary's **orthogonal core** and **frame cores** entries.
- N2: the four `fitting_and_optimization.md` rows (lines 52, 55, 76, 78) are one doctest session; apply them
  together and re-run that page's doctest.
- Aliases that the renames retire (`Ld`, "sample stack", "padded restart", ...) stay in the glossary's "also:" lists
  for one release, so readers of older docs still land on the entry.

## The three hardest decisions

1. **Tucker core** (the `(ni, Ni)` matrices `Bi`), over **Tucker factor**. Rule (a): the data field and attribute are
   `tucker_cores` / `up_tucker_cores` / `tucker_variations`. Against it: the docs use "Tucker factor" more (25 vs 14 in
   the user pages), the literature calls these matrices factors and the small tensor the core, and the sharing API
   itself says factors (`has_shared_tucker_factors`, `t3_tie_tucker_factors`, `ut3_tucker_factors_shared`). The
   collision with the Tucker decomposition's core tensor is handled by calling that the **central core** (item N15).
   **No bulk rename**: the glossary lists "Tucker factor" as an alias, and it stays natural in the sharing context.
   Nick's call: the API is split, and (a) wins only because the data layout is the more basic API.
2. **real slot / padding** (uniform slots whose mask is `False`), over **junk slot**, padded slot, garbage,
   don't-care. Rule (a) leans this way (`pad_safe_svd`, `pad_safe_pinned_completion`; no identifier says "junk") and
   (b) settles it: "padding" is by far the most used (23 user, 45 contributor). Against it: "padding" also names
   three other things (a zero-padded warm start, the padded frame, the carriers once called "frame-internal
   padding"), while "junk slot" (Nick's working term) collides with nothing. The glossary keeps "padding" for the
   uniform sense only and gives the other senses their own names (**warm start**, **padded frame**, **carrier**;
   items N14, E8). "Garbage" stays as the word for the *values* in padding. **Ruled "padding" (Nick, 2026-11-10)**;
   no bulk rename of "junk" (item N12).
3. **edge mask** (`tucker_edge_mask`, `tt_edge_mask`), over **rank mask**. Rule (a): the `UT3Masks` fields and the
   user guide's "Tucker edge masks" / "TT edge masks". Against it: "rank mask" is more used (11 vs 6 user, 12 vs 2
   contributor, 78 vs 29 code), and the rationale page is titled "Rank masks". **No bulk rename**: the glossary lists
   "rank mask" as an alias; prefer "edge mask" in new text.

## Other decisions (no edit needed)

| canonical | alternatives kept as aliases | rule |
|---|---|---|
| **frame** | orthogonal frame, orthonormal frame; API "orthogonal" means orthonormal (`is_orthogonal`) | (a) `T3Frame` |
| **frame-variations format** | frame-variations representation, (frame, variations) pair | (a) module `frame_variations_format` |
| **perturbation vectors** (`pp`) | directions `P`, perturbation directions | (a) API docstrings, naming_conventions.md registry |
| **g0 reference** (`g0_reference`) | Newton reference, reference gradient norm, `‖g0‖` | (a) |
| **store** | rank store ("store vs recompute" in the contributor docs is a different word use) | (b) |
| **sharing group** | group, partition ("group" is also a `contract` axis group; the glossary qualifies both) | (b) |
| **`contract`** (grouped contraction) | grouped einsum; "named contraction functions" = the retired ~104 functions only | (a) |
| **mode** | index, physical index, leg | (b) |
| **completion** | new block, P-block; also an SVD's completion at zero singular values (one entry, two senses) | (b) |
| **Gauss-Newton** (hyphen), **Hilbert–Schmidt** (en dash) | Gauss–Newton, Hilbert-Schmidt | (b) per term; typography only, not listed below |

## Items

| item | kind | decision or claim | verification note |
|---|---|---|---|
| N1 | NAMING | `W` is the **probe stack** (not "sample stack", "probe / sample stack", "vec/index stack") | API split (`sum_over_probes` vs `sample=`); docs 20:4 for "probe stack"; naming_conventions.md registers `W` (probe stack) |
| N2 | NAMING | the derivative order is spelled **`order`**, never `K` | every `*_derivatives` op and model takes `order=`; naming_conventions.md §Index letters reserves `K` for the tangent stack (2026-08-24 ruling). The curvature operators `K_{Aᵀω²r}`, `K_{grad f}` are left as math notation (the glossary's symbols table warns) |
| N3 | NAMING | the same-frame property label is **`FRAMES`** (proposed; any one word beside `ORTH`/`GAUGE`/`TIED` works), not `SF` (which is SF-T3 on the same page) | the check is `safety.frames_equal`; numerical_contracts.md has both `SF` (same-frame) and "The shared (SF-T3) surface" |
| N4 | NAMING | the down store and its cores are named **down, `O`** (not `D`, not "outer" as a name). The orthogonality *adjective* is N17, not N4 | `align_padded(update=...)` letters `'U'`, `'O'`, `'L'`, `'R'`; `T3Frame.data = (up, down, left, right)`. Not "outer TT bonds" (`tucker_tensor_train.py:962`), which are the boundary bonds and stay |
| N5 | NAMING | **old block / new block** (not A-block / P-block) | no API name; rendered docs 10:6 |
| N6 | NAMING | **bare Jacobian `𝒥`** for the tangent sampling map; the Riemannian Jacobian is `J = 𝒥∘Π` | `T3Tangent.probe/apply/entries` apply no gauge projection (`fitting.py` calls them "bare"); entries_apply_probe.md §4 notes |
| N7 | NAMING | **warm start** (not "padded restart", "continuation restart") | the most-used term; the point is `resize`'s zero-padded `x_old` (rank_continuation.md "How to use it") |
| N8 | NAMING | **hint-free start** / hint-free padded frame (not "seeded start", "seeded padded frame") | 17:2 in user docs; `T3Frame.padded` docstring says "hint-free" |
| N9 | NAMING | **probe vectors** (not "test vectors", "measurement vectors") | naming_conventions.md registers `ww` as probe/apply vectors; API docstrings say "probe vectors" |
| N10 | NAMING | the data symbol is **`y`** (not `b`) in prose formulas | `fitting_and_optimization.md` writes `y`; the keyword is `data=`. Code variables named `b` in examples/README may stay |
| N11 | NAMING | the forward operator is **`S(x)`** (not `F(X)`) | every user page writes `S(x)`; `F` occurs only in `fitting.py` |
| N12 | NAMING (**ruled 2026-11-10: "padding"**; no rows) | **padding** for uniform slots whose mask is `False` | Rules (a) `pad_safe_*` and (b) 23+45 doc uses. No bulk rename of the 35 "junk" sites: they are all padded-frame and pinned-completion text, where "padded" already means the zero-padded rank increase and its *real* new slots, so the rename would recreate the collision N14 removes. The glossary lists "junk slot" as the padded-frame code's word |
| N13 | NAMING | **store transport** (not bare "transport") for a store not in `update` | `MANIFOLD.transport` is vector transport, a different operation |
| N14 | NAMING | the new slots of a zero-padded point are **new slots**, not "the padding" | they are real slots (masks `True`), so "padding" would mean junk |
| N15 | NAMING | **central core** for the Tucker core tensor (never "central Tucker core") | "Tucker core" is the API's name for `tucker_cores` |
| N17 | NAMING (**HELD**: Nick is deciding) | the orthogonality adjectives follow the **method convention** (rule a): a Tucker core with `B Bᵀ = I` is *down-orthogonal* (`down_orthogonalize_tucker_cores`), a TT core orthonormal in its `ni × (ri·r(i+1))` matricization is *up-orthogonal* (`up_orthogonalize_tt_cores`; alias outer-orthogonal). Nick's reasoning: left cores are left-orthogonal and right cores right-orthogonal, and the `O` cores' orthonormal index group lies upward of the Tucker–TT edge, so the method adjectives are right and the **store names** "up"/"down" are the problem; a store rename is being assessed. The HELD rows below assume the adjectives stay and the frame pages change; a store rename would supersede them | `fv_conversions.py:184` binds `up_orthogonalize_tucker_cores = t3_down_orthogonalize_tucker_cores`; `ranks.py:644` "Tucker down-orthogonalization: up_i = ..."; `tucker_tensor_train.py:3159` (`B Bᵀ = I`, "down orthogonal") vs `:3207` (TT, "up orthogonal"); `up_svd_tt_core` (`:2978`) and `:3108` contradict even the method convention |
| E1 | ERROR | docstrings cite helpers that do not exist: a `numerically_equal` method and `common._jax_or_warn` | no `numerically_equal` in `t3toolbox` (grep); the method is `allclose`. The helper is `common.jax_or_warn` (no underscore) |
| E2 | ERROR | `g0_reference` value lists omit `'cone'` | `_G0_REFERENCES = ('zero', 'warm_start', 'cone')` (backend/optimizers.py:630); history records `'cone'` (l.1561) |
| E3 | ERROR | release notes say `use_jit=True` without jax raises | `common.jax_or_warn` warns and runs on numpy (since 2026.2.0, CHANGELOG `[2026.2.0]` Changed) |
| E4 | ERROR | masks described as prefix without qualification, or as going gappy under a T3 product | on the uniform layer `+` concatenates rank masks (`ut3_add`; ut3_operations.py:405 "The output mask can go gappy"); there is no uniform T3×T3 product (`UniformTuckerTensorTrain.__mul__` is scalar-only, `:253-262`); only a weights' `kronecker` Kroneckers masks. Prefix is the canonical form only |
| E5 | ERROR | the import-graph check is credited to the wrong test file | `tests/test_import_graph.py::test_the_backend_graph_is_acyclic`; `tests/test_contractions_interpreter.py` has no such check |
| E6 | ERROR (**ruled 2026-11-10: drop**) | internal review / plan / fact / test-plan codes outside `dev/` | the codes (`R1-12`, `H2-8`, `S1b`, `I-38`, `F19`, `T5`, ...) point into `dev/archive/` and cannot be resolved by a user. Ruling: drop them from docstrings, comments, runtime messages and the contributor pages, with no key (they stay in `dev/`). **Eight** runtime messages print one: `shared_geometry.py:186, 192, 207, 210`, `manifold.py:1249`, `backend/stacking.py:526`, `backend/linalg.py:225`, `backend/common.py:628`; the tests use bare `assertRaises`, so rewording is safe. The math notes' rows (`.tex`) are included for consistency, lowest priority. Two ID-keyed tables (`curvature_internals.md` facts, the `exact_hessian_two_hole.tex` checks) each get one structural row |
| E7 | ERROR | an example cites a doc that does not exist | `docs/derivative_order_information_and_conditioning.md` is absent |
| E8 | ERROR | the carriers are called `Ld`, `R0`; there is no core `Ld` | frame cores are indexed `L0…L(d-1)`; `fv_zero_point_frame` sets `left[d-1] = right[0] = 0`. The carriers are `L(d-1)` (`left[d-1]`) and `R0` (`right[0]`); the extra *bonds* are `rLd` and `rR0`. Only `docs/glossary.md` links the renamed anchor |
| E9 | ERROR | "a tangent is a sum of `d` terms" | `T3Tangent` docstring: "the sum of 2d single-core variation terms" (T4S eq. 46); the page's own diagram adds the Tucker set |
| E10 | ERROR | the Tucker gauge condition is written `Uᵢᵀ Vᵢ = 0` | `Ui`, `Vi` are stored `(n, N)`; `tv_orthogonal_gauge_projection` states `U_i V_i^T = 0` |
| E11 | ERROR | "`get_backend` dispatches ... `xwhile` to `jax.lax.while_loop`" | `get_backend` returns `(xnp, xmap, xscan)`; `common.xwhile` is separate and runs a Python loop unless `use_jit=True` with a jax state |
| E12 | ERROR | "the Gauss-Newton models" for the local models | `GaussNewtonModel` became `LocalModel` in 2026.3.0 and carries the exact Hessian too |
| E13 | ERROR | "the Taylor-series machinery the library is named for (T4S)" | the library is T3Toolbox; T4S is the paper |
| E14 | ERROR | `require_concrete_masks` docstring says `aux_data` is "identity-hashed, never-inspected" | mask holders hash by value (`common.ValueHashedMasks`) |
| E15 | ERROR | regularizers "receive a `GeometryOps`" | `geometry.py` calls `GeometryOps` the retired record-of-lambdas; the protocol is `optimizers.Geometry` |
| E16 | ERROR | comments cite the removed function `tWCi_Cio_to_tWCo` | the named contraction functions were removed (2026.1.0); `contract('tWCi,Cio->tWCo', ...)` replaced them |
| E17 | ERROR | docstring typos in index ranges and shapes | `tucker_ranks = (n0, …, n(d-1))`; modes, cores and diagrams are 0-based; `fv_variation_shapes` gives a TT variation the shape `(rL_i, nU_i, rR_(i+1))` (it reads `right[i].shape[-1]`), and the left/right frame cores carry `nU_i` |
| E18 | ERROR | examples call a T3-SVD truncation "the best rank-r approximation" | T3-SVD truncation is quasi-optimal, not optimal (contributor/t3svd_verification.md §1, the Oseledets-type bound) |
| E19 | ERROR | the user guide and `sharing.md` mix 1-based and 0-based mode indices, and one unfolding is garbled | modes are `N0…N(d-1)`; bond `ri` separates cores `0..i-1` from `i..d-1`, so mode `i`'s Tucker ceiling is `ri·r(i+1)` |
| E20 | ERROR | a user page points readers at `CLAUDE.md` | `CLAUDE.md` is not part of the published docs |
| E21 | ERROR | "frame-inner" used for the stack-leading layout | frame-inner is the `W + K + C` order of the batch blocks (batching_and_stacking.md §3) |
| E22 | ERROR (docstring clarity) | edge-variable docstrings use "leftward"/"rightward" without saying whether they mean the side or the travel direction | `compute_mu` and `compute_tau_tilde` are both forward (left-to-right) `xscan`s, yet `_mu_step` is "the leftward sweep" and `_tau_tilde_step` "the adjoint rightward sweep"; the words name the side of the variable. The T4S paper does not use them |
| M/U | ERROR | "matricization" and "unfolding" used for each other | Nick's definitions (consistent with T4S): a **matricization** cuts a Tucker–TT edge (one mode's index as rows, all other modes as columns, Kolda & Bader); an **unfolding** cuts a TT–TT edge (the ordered modes split, left as rows, right as columns). Each row was classified by the edge its matrix cuts; for a single core, the left/right reshapes cut a TT–TT edge (unfoldings) and the mode-leg reshape `ni × (ri·r(i+1))` cuts the Tucker–TT edge (a matricization). Where a sentence covers every edge (T3-SVD spectra, condition numbers), the replacement names both |
| C1 | CODE (**label ruled 2026-11-10**) | a `'cone'` run's verbose display names no reference | `optimizer_display._REFERENCE_NAMES` (`:121`) lacks `'cone'`: the converged line ends "…‖g₀‖, g₀ = zero tensor)" for `'zero'` but only "…‖g₀‖)" for `'cone'` (confirmed by a run). Fix: `'cone': 'cone (aligned zero tensor)'` plus a display test |

**Backlog, not docs:** `TuckerTensorTrain.__mul__` carries an unused `use_jax=None` (`tucker_tensor_train.py:1967`).

**Code identifiers that use the wrong word (M/U; reported, no rename proposed).** Two, both non-public:
`backend/padded_frames._unfold` / `_fold` (imported by `backend/uniform_padded_frames.py`) reshape any core slot to
the columns, and are called with the middle slot on the down cores `O` (`P_O`, the down edge's `g_new`, `g_old`,
`Abar`), which makes those matricizations; and the test
`tests/test_tucker_tensor_train.py::test_compute_minimal_ranks_matches_matricization` (cited in
`docs/t3svd_minimal_ranks.md:27, :135` and `tests/test_sharing.py:1382`), which checks Tucker matricizations *and* TT
unfoldings. Every public identifier is right: `_up_matricization`, `singular_values_of_matricizations` (Tucker),
`singular_values_of_unfoldings` (TT), and the tests' `all_matricization_ss` / `all_unfolding_ss`.

## Work list by file

User pages first, then contributor pages, the package, the examples, and the math notes.

### `docs/aligned_padded_frame.md`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 169 | § What the measurements showed | N8 | "seeded start" | "hint-free start" |

### `docs/batching_and_stacking.md`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 386 | § 8. Decoding the names (so you can read the code) | N1 | "vec/index stack" | "probe stack" |
| 413 | § 9. Gotchas (the ones that have actually bitten) | N1 | "vec/index stack" | "probe stack" |

### `docs/chunking.md`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 193 | § Sharding over `W` | N1 | "sample stack" | "probe stack" |

### `docs/entries_apply_probe.md`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 8 | § Entries, applies, and probes — the three sampling operations | E20 | "... the terse version ... `CLAUDE.md`" | point at a published page (e.g. `user_guide.rst` "The three sampling operations") |
| 19 | § TL;DR | N9 | "test vectors" | "probe vectors" |
| 20 | § TL;DR | N9 | "test vectors" | "probe vectors" |
| 24 | § TL;DR | N9 | "test vectors" | "probe vectors" |
| 25 | § TL;DR | N9 | "test vectors" | "probe vectors" |
| 47 | § Notation | N1 | "probe / sample stack" | "probe stack" |
| 47 | § Notation | N9 | "test-vector" | "probe vectors" |
| 48 | § Notation | N9 | "test vectors" | "probe vectors" |
| 102 | § 2. How they relate (the containment hierarchy) | N9 | "test vectors" | "probe vectors" |
| 105 | § 2. How they relate (the containment hierarchy) | N9 | "test vectors" | "probe vectors" |
| 135 | § 4. The Riemannian / tangent versions (and their transposes) | N6 | "`T3Tangent` the Riemannian Jacobian `𝒥`" | "`T3Tangent` the bare Jacobian `𝒥`" |
| 158 | § 4. The Riemannian / tangent versions (and their transposes) | N1 | "probe/sample stack" | "probe stack" |
| 164 | § All three are Riemannian-compatible (verified) | N10 | `f(X) = ½‖S(X) − b‖²` | `f(X) = ½‖S(X) − y‖²` |
| 192 | § 5. Measurement spaces (for fitting) | N10 | `r = S(X) − b` | `r = S(X) − y` |

### `docs/fitting_and_optimization.md`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 48 | § 2.1 Drive a library optimizer (the common case) | N9 | "measurement vectors" | "probe vectors" |
| 52 | § 2.1 Drive a library optimizer (the common case) | N2 | `>>> K, rng = 2, ...` | `>>> order, rng = 2, ...` |
| 55 | § 2.1 Drive a library optimizer (the common case) | N2 | `apply_derivatives(ww, pp, K)` ... `shape (K+1,)+W` | `apply_derivatives(ww, pp, order)` ... `shape (order+1,)+W` |
| 76 | § 2.1 Drive a library optimizer (the common case) | N2 | `shape (K+1,)` | `shape (order+1,)` |
| 78 | § 2.1 Drive a library optimizer (the common case) | N2 | `order=K` | `order=order` |
| 205 | § 2.3 Custom minibatching — the `draw` | N9 | "measurement vectors" | "probe vectors" |
| 208 | § 2.3 Custom minibatching — the `draw` | N1 | "sample stack" | "probe stack" |
| 483 | § 4.9 Regularization: an optional objective term | M/U | "An inverse-unfolding-singular-value prior" | "An inverse-singular-value prior (on the matricizations and unfoldings)" |
| 543 | § 4.12 The exact Hessian and the trust region | N7 | "continuation restart" | "warm start (zero-padded)" |
| 548 | § 4.12 The exact Hessian and the trust region | N7 | "padded restart" | "warm start (zero-padded)" |
| 550 | § 4.12 The exact Hessian and the trust region | N7 | "padded restart" | "warm start (zero-padded)" |
| 607 | § 4.12 The exact Hessian and the trust region | N7 | "continuation restart" | "warm start (zero-padded)" |

### `docs/frame_variations.md`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 8 | § A tangent is a sum of `d` terms | E9 | heading "A tangent is a sum of `d` terms" | "A tangent is a sum of `2d` terms" (update links to this anchor: docs/glossary.md) |
| 32 | § `d` cores, `d+1` bonds — and the "extra" `Ld`, `R0` | E8 | heading "`d` cores, `d+1` bonds — and the "extra" `Ld`, `R0`" | "... — and the carriers `L(d-1)`, `R0`" (update the link to this anchor in docs/glossary.md) |
| 36 | § `d` cores, `d+1` bonds — and the "extra" `Ld`, `R0` | E8 | "**`Ld` and `R0` are never used**" | "**`L(d-1)` and `R0` are never used**" |
| 42 | § `d` cores, `d+1` bonds — and the "extra" `Ld`, `R0` | E8 | "the `Ld`-adjacent slot ... the padding `Ld` / `R0` cores" | "the `L(d-1)`-adjacent slot ... the carriers `L(d-1)` / `R0`" |
| 57 | § Gauge conditions make the variations *coordinates* | E10 | "`Uᵢᵀ Vᵢ = 0`" | "`Uᵢ Vᵢᵀ = 0`" |
| 71 | § Four rank stores — and why excess rank is inert | N4 | "down/outer" | "down" |
| 77 | § Four rank stores — and why excess rank is inert | M/U | "Each core must be orthonormal *in its own unfolding*" | "Each core must be orthonormal *in its own matricization or unfolding*" |
| 83 | § Four rank stores — and why excess rank is inert | M/U | "`nᵢ × (rᵢ·rᵢ₊₁)` unfolding, rows orthonormal" | "`nᵢ × (rᵢ·rᵢ₊₁)` matricization, rows orthonormal" |
| 125 | § Weights are a diagonal metric on those coordinates | E8 | "padding `Ld`/`R0` bonds" | "carrier bonds `rLd` / `rR0`" |

### `docs/numerical_contracts.md`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 50 | § The numerical properties and their checkers | N3 | "SF" | "`FRAMES` (same-frame)" |
| 130 | § `T3Tangent` | N3 | "SF" | "`FRAMES` (same-frame)" |
| 132 | § `T3Tangent` | N3 | "SF" | "`FRAMES` (same-frame)" |
| 133 | § `T3Tangent` | N3 | "SF" | "`FRAMES` (same-frame)" |
| 138 | § `T3Tangent` | N3 | "SF" | "`FRAMES` (same-frame)" |
| 146 | § The geometries (the semantic `inner`/`norm` live here) | N3 | "SF" | "`FRAMES` (same-frame)" |
| 147 | § The geometries (the semantic `inner`/`norm` live here) | N3 | "SF" | "`FRAMES` (same-frame)" |
| 148 | § The geometries (the semantic `inner`/`norm` live here) | N3 | "SF" | "`FRAMES` (same-frame)" |
| 174 | § The shared (SF-T3) surface | N7 | "continuation restart" | "warm start (zero-padded)" |
| 180 | § `fitting.LocalModel` | N3 | "SF" | "`FRAMES` (same-frame)" |
| 181 | § `fitting.LocalModel` | N3 | "SF" | "`FRAMES` (same-frame)" |
| 192 | § The exact Hessian (`hessian='exact'`) | N7 | "padded restart" | "warm start (zero-padded)" |
| 204 | § The optimizers' relative stop | N8 | "seeded start" | "hint-free start" |
| 235 | § Minimal ranks: the settled verdict | N7 | "continuation restarts" | "warm start (zero-padded)" |

### `docs/rank_continuation.md`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 7 | § Rank continuation — choosing which ranks to grow | M/U | "the singular values of the current iterate's unfoldings" | "the singular values of the current iterate's matricizations and unfoldings" |
| 29 | § The idea (Section 5.4.1) | M/U | "the singular values of the current iterate's matrix unfoldings" | "the singular values of the current iterate's matricizations and unfoldings" |
| 126 | § How to use it | N7 | "padded restart" | "warm start (zero-padded)" |
| 130 | § How to use it | N7 | "padded restart" | "warm start (zero-padded)" |

### `docs/release_notes.md`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 221 | § Upgrading from 2026.0.0 | E3 | "... and raise if jax is not installed." | "... and, since 2026.2.0, warn and run eager if jax is not installed." |

### `docs/sharing.md`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 70 | § What the group spectrum is | M/U | "the singular values of a matrix unfolding across that edge" | "the singular values of the matricization (Tucker edge) or unfolding (TT bond) across that edge" |
| 86 | § What the group spectrum is | M/U | "the cut separating the shared leg from everything else has unfolding exactly the concatenation of face 1" | "... has matricization exactly the concatenation of face 1" (one leg against the rest) |
| 160 | § Optimizing over shared factors | N7 | "padded restart" | "warm start (zero-padded)" |
| 165 | § Optimizing over shared factors | N7 | "continuation restarts" | "warm start (zero-padded)" |
| 173 | § Rank machinery: minimal ranks, dimension, continuation | E19 | "the per-mode Tucker ceiling `n_i ≤ min(N_i, r_{i-1}·r_i)`" | "`n_i ≤ min(N_i, r_i·r_{i+1})`" |
| 176 | § Rank machinery: minimal ranks, dimension, continuation | E19 | `n_g ≤ min(N_g, Σ_{i∈g} min(N_g, r_{i-1}·r_i))` | `n_g ≤ min(N_g, Σ_{i∈g} min(N_g, r_i·r_{i+1}))` |
| 209 | § Rank machinery: minimal ranks, dimension, continuation | N7 | "padded restart" | "warm start (zero-padded)" |

### `docs/t3m_methods.md`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 24 | § What there is to truncate | N15 | "the central Tucker core `C`" | "the central core `C`" |

### `docs/t3svd_minimal_ranks.md`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 21 | § What "minimal ranks" means | M/U | "These are exactly the **matricization ranks** of a generic tensor" | "These are exactly the **matricization and unfolding ranks** of a generic tensor" |
| 23 | § What "minimal ranks" means | M/U | "the minimal dimension of that edge is the rank of the corresponding matricization" | "... the rank of the corresponding matricization (a Tucker edge) or unfolding (a TT bond)" |
| 123 | § This is not specific to the Tucker extension | M/U | "sees the *true* matricization singular values" | "sees the *true* matricization and unfolding singular values" |

### `docs/transposes.md`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 138 | § Tangent — `T3Tangent.apply_transpose(c, ww, frame, sum_over_probes=…)`  *(static, takes a frame)* | N6 | "The Riemannian Jacobian transpose:" | "The bare Jacobian transpose `𝒥ᵀ` (compose with `MANIFOLD.project` for `Jᵀ`):" |

### `docs/uniform_equivalence_contract.md`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 79 | § The full-rank qualifier (measured 2026-08-27) | N7 | "padded restart" | "warm start (zero-padded)" |
| 82 | § The full-rank qualifier (measured 2026-08-27) | E6 | "it could not survive the S1b fix" | "it could not survive the pad-safe SVD fix (2026.2.0)" |

### `docs/uniform_masks_vs_ranks.md`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 44 | § Masks change under arithmetic — and can go non-prefix | E4 | "under `+` they **concatenate**, under `×` they **Kronecker**" | "under `+` they **concatenate**, under `×` they **Kronecker** (on the uniform layer `×` exists only as a weights' `kronecker`; there is no uniform T3×T3 product)" |
| 53 | § Masks change under arithmetic — and can go non-prefix | E4 | "So after `+`/`×` your object's `.masks` may be **gappy**." | "So after `+` (or a weights' `kronecker`) the `.masks` may be **gappy**." |

### `docs/uniform_ranks_and_varieties.md`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 38 | § The real question: a batch on a manifold, or a batch in a variety? | E13 | "the Taylor-series machinery the library is named for (T4S)" | "the Taylor-series machinery of the T4S paper" |

### `docs/uniform_supercore_layout.md`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 21 | § The layout | E21 | "(the "frame-inner" convention in `docs/batching_and_stacking.md`)" | "(the stack-leading layout; `docs/batching_and_stacking.md`)" -- frame-inner is the `W + K + C` order of the blocks |

### `docs/user_guide.rst`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 9 | § Tucker tensor trains | N15 | "representation of the central Tucker core" | "representation of the central core (the Tucker decomposition's core tensor)" |
| 97 | § Minimal rank conditions | M/U | "Outer TT core unfoldings are full rank" | "... TT core matricizations are full rank" (the adjective is N17, HELD) |
| 97 | § Minimal rank conditions | N17 | "Outer TT core unfoldings are full rank" | HELD (N17): "Up (outer) TT core unfoldings are full rank" |
| 103 | § Minimal rank conditions | E19 | "Ni x (N1*...*N(i-1)*N(i+1)*...*N(d-1))" | "Ni x (N0*...*N(i-1)*N(i+1)*...*N(d-1))" |
| 104 | § Minimal rank conditions | E19 | "(N*...*Ni) x (N(i+1)*...*N(d-1))" | "(N0*...*N(i-1)) x (Ni*...*N(d-1))" (bond ri sits between cores i-1 and i) |
| 122 | § T3 manifold | E19 | "R^(N1 x ... x Nd)" | "R^(N0 x ... x N(d-1))" |
| 142 | § T3 manifold | N17 | "down_tt_cores = (O0,...,O(d-1)), down-orthogonal" | HELD (N17): "..., up-orthogonal" |
| 166 | § The three sampling operations: entries, apply, probe | N9 | "test vectors" | "probe vectors" |
| 173 | § The three sampling operations: entries, apply, probe | N6 | "(the Riemannian Jacobian 𝒥 and its transpose 𝒥ᵀ)" | "(the bare Jacobian 𝒥 and its transpose 𝒥ᵀ; the Riemannian Jacobian is J = 𝒥∘Π)" |
| 177 | § The three sampling operations: entries, apply, probe | E19 | "a N1 x N2 x N3 Tucker tensor with vectors u1, u2, u3 (lengths N1, N2, N3" | "an N0 x N1 x N2 ... u0, u1, u2 (lengths N0, N1, N2" (and the diagram labels) |
| 255 | § Uniform Tucker tensor trains | E4 | "but ``+`` and the Hadamard product produce *gappy* working-form masks" | "but ``+`` (and, for edge weights, ``kronecker``) produces *gappy* working-form masks" |
| 260 | § Uniform Tucker tensor trains | E4 | "the first ni entries are 1, and the remaining entries are 0" | "in canonical form the first ni entries are 1 and the rest 0" |
| 262 | § Uniform Tucker tensor trains | E4 | "the first ri entries are 1, and the remaining entries are 0" | "in canonical form the first ri entries are 1 and the rest 0" |
| 268 | § Uniform Tucker tensor trains | E4 | "The masks are boolean prefix vectors rather than integer ranks" | "The masks are boolean vectors rather than integer ranks" |
| 301 | § NumPy and JAX | E12 | "the Gauss-Newton models" | "the local models (`LocalModel`)" |

### `docs/contributor/curvature_internals.md`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 28 | § The formula, and where each piece lives | E6 | ``` \| frame factors `T_i = ⟨L_i, H_i⟩`, `S_iᵀ = ⟨O_i, H_i⟩`, centers `H_i = L_i T_i` \| `fv_operations.fv_center_factors` \| `T_i` by zipper on demand, never stored on the frame (I-1); `σ(S_i)` is the mode-`i` Tucker spectrum, `σ(T_i)` the bond spectrum; `‖X‖² = Σσ(S_0)²` \| ``` | ``` \| frame factors `T_i = ⟨L_i, H_i⟩`, `S_iᵀ = ⟨O_i, H_i⟩`, centers `H_i = L_i T_i` \| `fv_operations.fv_center_factors` \| `T_i` by zipper on demand, never stored on the frame; `σ(S_i)` is the mode-`i` Tucker spectrum, `σ(T_i)` the bond spectrum; `‖X‖² = Σσ(S_0)²` \| ``` |
| 50 | § The lift and its one-sided exactness | E6 | ``` there (F2, F3: 6e-15 / 9e-16); only a hand-built frame with a *wide* `T_i` would give a least-squares lift ``` | ``` there (6e-15 / 9e-16); only a hand-built frame with a *wide* `T_i` would give a least-squares lift ``` |
| 74 | § The lift and its one-sided exactness | E6 | ``` **The damping is a solver parameter, not a geometry field** (I-22; the "options are not parameters" rule ``` | ``` **The damping is a solver parameter, not a geometry field** (the "options are not parameters" rule ``` |
| 77 | § The lift and its one-sided exactness | N7 | "padded restart" | "warm start (zero-padded)" |
| 79 | § The lift and its one-sided exactness | E6 | ``` special case (E2, F19; the trust region bounds the rest). ``` | ``` special case (the trust region bounds the rest). ``` |
| 89 | § The two-hole sweep | E6 | ``` **The block-sum oracle** (F5) is the reference every two-hole is tested against, and lives only in ``` | ``` **The block-sum oracle** is the reference every two-hole is tested against, and lives only in ``` |
| 90 | § The two-hole sweep | E6 | ``` `tests/backend/_curvature_oracles.py` (I-15): embed `(θ, θ̂)` into a doubled-rank train `Y` whose ``` | ``` `tests/backend/_curvature_oracles.py`: embed `(θ, θ̂)` into a doubled-rank train `Y` whose ``` |
| 103 | § The two-hole sweep | E6 | ``` affine in `t` like `xi`. **Layout: companion arrays** (I-10) — undotted jets are `(order+1,)+W+C+(…)`, ``` | ``` affine in `t` like `xi`. **Layout: companion arrays** — undotted jets are `(order+1,)+W+C+(…)`, ``` |
| 107 | § The two-hole sweep | E6 | ``` **The dotted kernels are recurrences** (I-33, Nick 2026-10-27, superseding the earlier "hold the ``` | ``` **The dotted kernels are recurrences** (Nick 2026-10-27, superseding the earlier "hold the ``` |
| 113 | § The two-hole sweep | E6 | ``` dotted pushthroughs, each a two-term reverse recurrence in one GEMM; shared since S12 as ``` | ``` dotted pushthroughs, each a two-term reverse recurrence in one GEMM; shared as ``` |
| 118 | § The two-hole sweep | E6 | ``` **The factored assembly** (F33). The TT gradient assembly is trilinear; its derivative has three terms ``` | ``` **The factored assembly**. The TT gradient assembly is trilinear; its derivative has three terms ``` |
| 121 | § The two-hole sweep | E6 | ``` assembly (1.4–2.0× faster on that term, equal to 3e-15), probe 9 → 4, apply / entries 3 → 2 (S10/S11). The ``` | ``` assembly (1.4–2.0× faster on that term, equal to 3e-15), probe 9 → 4, apply / entries 3 → 2. The ``` |
| 125 | § The two-hole sweep | E6 | ``` **The T/S transforms** (F16). At the *manifold* frame the corewise sweep and adjoint state are cheap ``` | ``` **The T/S transforms**. At the *manifold* frame the corewise sweep and adjoint state are cheap ``` |
| 128 | § The two-hole sweep | E6 | ``` not use them: the default is **recompute** (I-11, the house low-memory preference) — `xi` / `mu` are ``` | ``` not use them: the default is **recompute** (the house low-memory preference) — `xi` / `mu` are ``` |
| 153 | § Model-level rules | E6 | ``` stored (`is_exact = curvature is not None`, I-3, B1); the term is present iff `geometry_data is not None`. ``` | ``` stored (`is_exact = curvature is not None`); the term is present iff `geometry_data is not None`. ``` |
| 155 | § Model-level rules | E6 | ``` **θ̂ masking from the FRAME** (uniform; m29, F18, F23, F25). On a uniform supercore a direction's padding ``` | ``` **θ̂ masking from the FRAME** (uniform). On a uniform supercore a direction's padding ``` |
| 159 | § Model-level rules | E6 | ``` is a strict superset at TT slack — and re-masks the output (the S0c rule: every uniform transpose returns ``` | ``` is a strict superset at TT slack — and re-masks the output (the rule: every uniform transpose returns ``` |
| 164 | § Model-level rules | E6 | ``` **Flat slots** (I-12). The curvature term is the second fundamental form of the representable set. A slot ``` | ``` **Flat slots**. The curvature term is the second fundamental form of the representable set. A slot ``` |
| 166 | § Model-level rules | M/U | "puts no constraint on its unfolding" | "puts no constraint on its matricization (a Tucker slot) or unfolding (a TT slot)" |
| 167 | § Model-level rules | E6 | ``` there; at `d = 1` every slot is maximal at once (the set is open in `ℝ^N`, E4, E5). The formula needs no ``` | ``` there; at `d = 1` every slot is maximal at once (the set is open in `ℝ^N`). The formula needs no ``` |
| 175 | § The canonical zero-point frame (the default Newton reference) | E6 | ``` the choice, so the library fixes one (I-29): `fv_zero_point_frame(frame, groups, seed=ZERO_FRAME_SEED)`. ``` | ``` the choice, so the library fixes one: `fv_zero_point_frame(frame, groups, seed=ZERO_FRAME_SEED)`. ``` |
| 193 | § The canonical zero-point frame (the default Newton reference) | E6 | ``` completions 6 of 16 ragged/uniform pairs drifted, with one reference 0 of 16 (F30), and the 44 O2 pairs ``` | ``` completions 6 of 16 ragged/uniform pairs drifted, with one reference 0 of 16, and the 44 O2 pairs ``` |
| 194 | § The canonical zero-point frame (the default Newton reference) | E6 | ``` agree to 3.5e-12 (F37). ``` | ``` agree to 3.5e-12. ``` |
| 214 | § The canonical zero-point frame (the default Newton reference) | E6 | ``` **exactly 0** whenever the data have no energy on the leading coordinates (F32); a Haar draw never does ``` | ``` **exactly 0** whenever the data have no energy on the leading coordinates; a Haar draw never does ``` |
| 218 | § The canonical zero-point frame (the default Newton reference) | E6 | ``` moved it by up to ~1.7×; P1b, E9), which is harmless for a relative stop. `RandomState`'s legacy stream is ``` | ``` moved it by up to ~1.7×), which is harmless for a relative stop. `RandomState`'s legacy stream is ``` |
| 221 | § The canonical zero-point frame (the default Newton reference) | E6 | ``` reference would be exactly 0 — P1a, E8), so their default is the warm start, and the member is optional ``` | ``` reference would be exactly 0), so their default is the warm start, and the member is optional ``` |
| 222 | § The canonical zero-point frame (the default Newton reference) | E6 | ``` and duck-typed for custom geometries (m34). Alternatives considered: [`deferred_and_rejected.md`](deferred_and_rejected.md). ``` | ``` and duck-typed for custom geometries. Alternatives considered: [`deferred_and_rejected.md`](deferred_and_rejected.md). ``` |
| 239 | § Shared factors | N7 | "Padded restarts" | "warm start (zero-padded)" |
| 243 | § Shared factors | E6 | ``` - **The uniform padded `s_g` mask** (S9). The uniform companion holds padded slices (row splits in ``` | ``` - **The uniform padded `s_g` mask**. The uniform companion holds padded slices (row splits in ``` |
| 251 | § Shared factors | E6 | ``` (`project_mean(K θ̂)`). The shared retraction's second order is **measured** (O(h²) finite differences, ``` | ``` (`project_mean(K θ̂)`). The shared retraction's second order is **measured** (O(h²) finite differences), ``` |
| 252 | § Shared factors | E6 | ``` F14), not proven. ``` | ``` not proven. ``` |
| 254 | § The operand-scaled gauge check (I-38, S7b) | E6 | ``` ## The operand-scaled gauge check (I-38, S7b) ``` | ``` ## The operand-scaled gauge check ``` |
| 267 | § The operand-scaled gauge check (I-38, S7b) | E6 | ``` Recorded limits, both accepted: **F1** — dividing an FD quotient by `2h` divides the recorded scale too, so ``` | ``` Recorded limits, both accepted: dividing an FD quotient by `2h` divides the recorded scale too, so ``` |
| 270 | § The operand-scaled gauge check (I-38, S7b) | E6 | ``` [`../numerical_contracts.md`](../numerical_contracts.md)). **F2** — a safe-mode leaf `±` now computes both ``` | ``` [`../numerical_contracts.md`](../numerical_contracts.md)). And a safe-mode leaf `±` now computes both ``` |
| 277 | § The trust region (Steihaug–Toint) | E6 | ``` One CG body serves both step controls (I-5): the state carries `radius` and `boundary`; with ``` | ``` One CG body serves both step controls: the state carries `radius` and `boundary`; with ``` |
| 282 | § The trust region (Steihaug–Toint) | E6 | ``` module constants (O-2). A descent-guard step with scale < 1 counts as a boundary step for the 2Δ rule. ``` | ``` module constants. A descent-guard step with scale < 1 counts as a boundary step for the 2Δ rule. ``` |
| 285 | § The trust region (Steihaug–Toint) | E6 | ``` `\|pred\| ≤ 10·ε·(‖ω⊙y‖·‖ω⊙r‖ + \|f\|)` with `ε = np.finfo(residual dtype).eps` (host-side, dtype-only — I-25, ``` | ``` `\|pred\| ≤ 10·ε·(‖ω⊙y‖·‖ω⊙r‖ + \|f\|)` with `ε = np.finfo(residual dtype).eps` (host-side, dtype-only), ``` |
| 286 | § The trust region (Steihaug–Toint) | E6 | ``` m30), then `trust_region_floor` (`Δ < 10⁻¹²·Δ0`). A rejected trial keeps `x` and its local model, so the ``` | ``` then `trust_region_floor` (`Δ < 10⁻¹²·Δ0`). A rejected trial keeps `x` and its local model, so the ``` |
| 298 | § Facts the design rests on | E6 | the ID-keyed table, lines 298-329 | Drop the `#` (fact-ID) column of the "Facts the design rests on" table, header included, and the plan codes inside its cells (`T15`, `T16`, `T20`, `T26`, `S0c`, `S3`, `S3b`, `S12`, `m29`, `m30`, `n15`, `F19`/`E2` cross-references): keep each row's claim and measurement. Rows that differed only by ID stay separate rows |
| 329 | § Facts the design rests on | E5 | "the import DAG is acyclic" credited to `tests/test_contractions_interpreter.py` | split the row: the acyclic import graph is `tests/test_import_graph.py` |
| 331 | § Facts the design rests on | E6 | ``` Memory and compile time (S3, S8, S12): the two-hole's TT-assembly kernel takes 2.2–2.8× the Gauss-Newton ``` | ``` Memory and compile time: the two-hole's TT-assembly kernel takes 2.2–2.8× the Gauss-Newton ``` |
| 332 | § Facts the design rests on | E6 | ``` assembly's per-row scratch after S3b (the running chunked program, which also holds the chunk's jet slices, ``` | ``` assembly's per-row scratch after the recurrence conversion (the running chunked program, which also holds the chunk's jet slices, ``` |

### `docs/contributor/deferred_and_rejected.md`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 11 | § Rejected | M/U | "the t3svd-then-project path's *first* unfolding SVD" | "... *first* matricization SVD" (`dense_t3svd` starts with the mode-0 Tucker SVD) |
| 42 | § Rejected | E6 | ``` - **A capability flag for the exact Hessian (`has_corewise_hessian`)**. Rejected (I-14): `corewise_hessian` ``` | ``` - **A capability flag for the exact Hessian (`has_corewise_hessian`)**. Rejected: `corewise_hessian` ``` |
| 51 | § Rejected | E6 | ``` - **The damping as a geometry field.** Rejected (I-22): it changes how the solver approaches the problem, ``` | ``` - **The damping as a geometry field.** Rejected: it changes how the solver approaches the problem, ``` |
| 54 | § Rejected | E6 | ``` - **Other completions for the zero-reference frame** (I-29, adopted the seeded, keyed, nested Haar recipe — ``` | ``` - **Other completions for the zero-reference frame** (adopted the seeded, keyed, nested Haar recipe — ``` |
| 66 | § Deferred (would be built if the need materializes) | E6 | ``` - **Per-axis uniform `sum_stack(axis=)`** (review H2-7; ruled defer + document, 2026-08-24). Ragged ``` | ``` - **Per-axis uniform `sum_stack(axis=)`** (ruled defer + document, 2026-08-24). Ragged ``` |
| 178 | § Deferred (would be built if the need materializes) | E6 | ``` - ~~**Backend `ut3_norm` / `ut3_inner` twins**~~ **Built 2026-08-22** (with the review's S11 fix: their jax ``` | ``` - ~~**Backend `ut3_norm` / `ut3_inner` twins**~~ **Built 2026-08-22** (with the review's fix: their jax ``` |
| 191 | § Deferred (would be built if the need materializes) | E6 | ``` (uniform weighting S1), which had to reproduce the chain; it is parked in that module's private ``` | ``` (the uniform weighting build), which had to reproduce the chain; it is parked in that module's private ``` |
| 194 | § Deferred (would be built if the need materializes) | E6 | ``` `weighted_internals.md` (S1), `backend/ut3_linalg.py`. ``` | ``` `weighted_internals.md`, `backend/ut3_linalg.py`. ``` |
| 196 | § Deferred (would be built if the need materializes) | E6 | ``` - **Normal-space sector norms — deferred** (I-13, Nick 2026-10-26). The curvature term's normal-space operand Z ``` | ``` - **Normal-space sector norms — deferred** (Nick 2026-10-26). The curvature term's normal-space operand Z ``` |
| 203 | § Deferred (would be built if the need materializes) | M/U | "the t3svd unfolding condition numbers" | "the t3svd matricization and unfolding condition numbers" |
| 204 | § Deferred (would be built if the need materializes) | E6 | ``` - **The Newton schedule — an open, empirical question** (I-36, Nick 2026-10-27). Gauss-Newton vs the exact ``` | ``` - **The Newton schedule — an open, empirical question** (Nick 2026-10-27). Gauss-Newton vs the exact ``` |
| 207 | § Deferred (would be built if the need materializes) | E6 | ``` warm-started level starts much closer to the quadratic basin than the cold starts S0b measured); still ``` | ``` warm-started level starts much closer to the quadratic basin than the cold starts measured earlier); still ``` |
| 211 | § Deferred (would be built if the need materializes) | E6 | ``` user option (I-37). The inputs gathered so far (the experiment records, archived under `dev/archive/`; numbers ``` | ``` user option. The inputs gathered so far (the experiment records, archived under `dev/archive/`; numbers ``` |
| 218 | § Deferred (would be built if the need materializes) | E6 | ``` residual-dependent curvature state per model instead of recomputing it per Hessian-apply (I-11: ~+60 % of the ``` | ``` residual-dependent curvature state per model instead of recomputing it per Hessian-apply (~+60 % of the ``` |
| 223 | § Deferred (would be built if the need materializes) | E6 | ``` and the lift (I-21: not reused, recomputed by zipper); a **curvature-carrying regularizer hook** (m19: the ``` | ``` and the lift (not reused, recomputed by zipper); a **curvature-carrying regularizer hook** (the ``` |
| 226 | § Deferred (would be built if the need materializes) | E6 | ``` (J-12: the trust region's `objective_floor` covers the resolution limit, not a slow plateau); `tr_radius0` ``` | ``` (the trust region's `objective_floor` covers the resolution limit, not a slow plateau); `tr_radius0` ``` |
| 228 | § Deferred (would be built if the need materializes) | E6 | ``` zero-point frame too (J.13 / I-31: it would make zero-start trajectories layer-independent, but changes ``` | ``` zero-point frame too (it would make zero-start trajectories layer-independent, but changes ``` |
| 240 | § Deferred (would be built if the need materializes) | E6 | ``` - **The uniform tied projection's clip** (S9, a follow-up slice). `ufv_share_tucker_variations` — the shared ``` | ``` - **The uniform tied projection's clip** (a follow-up slice). `ufv_share_tucker_variations` — the shared ``` |
| 248 | § Deferred (would be built if the need materializes) | E6 | ``` - **The operand-scaled gauge check's follow-ups** (S7b; [`curvature_internals.md`](curvature_internals.md)). ``` | ``` - **The operand-scaled gauge check's follow-ups** ([`curvature_internals.md`](curvature_internals.md)). ``` |

### `docs/contributor/doctest_style.md`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 22 | § The principle (this is the part that matters) | M/U | "`t3svd`'s singular values are the dense unfoldings' spectra" | "... are the dense matricizations' and unfoldings' spectra" |

### `docs/contributor/fitting_internals.md`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 57 | § Regularization | E6 | ``` **Why an objective term and not Newton-Hessian-only damping** (decision D2): damping the Hessian alone is ``` | ``` **Why an objective term and not Newton-Hessian-only damping**: damping the Hessian alone is ``` |
| 62 | § Regularization | E6 | ``` **Identity reg is `ρ = ½λ‖x‖²` in the geometry's own tangent metric** (decision D1) → it contributes ``` | ``` **Identity reg is `ρ = ½λ‖x‖²` in the geometry's own tangent metric** → it contributes ``` |
| 119 | § What's deferred (not built) | M/U | "built from the cores' unfolding singular values" | "built from the matricization and unfolding singular values" |
| 143 | § Plans and research pointers | E6 | ``` `dev/archive/regularization_design.md` (the full regularization build — the D1/D2/D3 decisions, the ``` | ``` `dev/archive/regularization_design.md` (the full regularization build — its design decisions, the ``` |
| 144 | § Plans and research pointers | E6 | ``` `v_X`/`value` derivations verified this section summarizes, the uniform mask-safety spec, and the S4 ``` | ``` `v_X`/`value` derivations verified this section summarizes, the uniform mask-safety spec, and the ``` |

### `docs/contributor/numerical_contract_catalog.md`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 1 | § S2 — Numerical-contract catalog (precondition vs caveat sweep) | E6 | ``` # S2 — Numerical-contract catalog (precondition vs caveat sweep) ``` | ``` # Numerical-contract catalog (precondition vs caveat sweep) ``` |
| 14 | § S2 — Numerical-contract catalog (precondition vs caveat sweep) | N3 | "SF" | "`FRAMES` (same-frame)" |
| 68 | § Master table (by surface) | N3 | "SF" | "`FRAMES` (same-frame)" |
| 76 | § `T3Tangent` (after the §S3 rename) | E6 | ``` ### `T3Tangent` (after the §S3 rename) ``` | ``` ### `T3Tangent` (after the rename) ``` |
| 79 | § `T3Tangent` (after the §S3 rename) | N3 | "SF" | "`FRAMES` (same-frame)" |
| 81 | § `T3Tangent` (after the §S3 rename) | N3 | "SF" | "`FRAMES` (same-frame)" |
| 83 | § `T3Tangent` (after the §S3 rename) | N3 | "SF" | "`FRAMES` (same-frame)" |
| 87 | § `T3Tangent` (after the §S3 rename) | N3 | "SF" | "`FRAMES` (same-frame)" |
| 94 | § Geometries (the semantic `inner`/`norm` live here — Nick's refinement) | N3 | "SF" | "`FRAMES` (same-frame)" |
| 95 | § Geometries (the semantic `inner`/`norm` live here — Nick's refinement) | N3 | "SF" | "`FRAMES` (same-frame)" |
| 96 | § Geometries (the semantic `inner`/`norm` live here — Nick's refinement) | N3 | "SF" | "`FRAMES` (same-frame)" |
| 108 | § `LocalModel` (fitting) | N3 | "SF" | "`FRAMES` (same-frame)" |
| 109 | § `LocalModel` (fitting) | N3 | "SF" | "`FRAMES` (same-frame)" |
| 126 | § The blurs this fixes (current code conflations) | N3 | "SF" | "`FRAMES` (same-frame)" |
| 127 | § The blurs this fixes (current code conflations) | N3 | "SF" | "`FRAMES` (same-frame)" |
| 162 | § Implementation notes (for S3–S5, not decisions here) | E6 | ``` ## Implementation notes (for S3–S5, not decisions here) ``` | ``` ## Implementation notes (not decisions here) ``` |
| 168 | § Implementation notes (for S3–S5, not decisions here) | E6 | ``` `t3toolbox/safety.py` (S1, done). ``` | ``` `t3toolbox/safety.py` (done). ``` |
| 174 | § Implementation notes (for S3–S5, not decisions here) | E6 | ``` - **Cost mitigation (done in S5):** cache the atol-*independent* **residual**, not the atol-bool — ``` | ``` - **Cost mitigation (done):** cache the atol-*independent* **residual**, not the atol-bool — ``` |
| 178 | § Implementation notes (for S3–S5, not decisions here) | E6 | ``` - **Operand-scaled GAUGE (2026-10-27, ruling I-38):** the GAUGE residual of a tangent built by ``` | ``` - **Operand-scaled GAUGE (2026-10-27):** the GAUGE residual of a tangent built by ``` |
| 185 | § Implementation notes (for S3–S5, not decisions here) | E6 | ``` - **Trace detection (done in S5):** the eager-only skip must fire even when the checked operand is a ``` | ``` - **Trace detection (done):** the eager-only skip must fire even when the checked operand is a ``` |

### `docs/contributor/padded_frame_internals.md`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 20 | § The step as message passing | N5 | "A-block" | "old block" |
| 21 | § The step as message passing | N5 | "P-block" | "new block" |
| 24 | § The step as message passing | N5 | "A-block" | "old block" |
| 45 | § The step as message passing | N13 | "**Transport.** A store not in `update` ..." | "**Store transport.** A store not in `update` ... (not vector transport, `MANIFOLD.transport`)" |
| 67 | § The uniform layer: block layout | N5 | "A-block" | "old block" |
| 67 | § The uniform layer: block layout | N5 | "P-block" | "new block" |
| 68 | § The uniform layer: block layout | N5 | "A-block" | "old block" |
| 85 | § The start: the pad-safe pinned completion, and why a QR | N5 | "A-block" | "old block" |
| 87 | § The start: the pad-safe pinned completion, and why a QR | M/U | "drawn on the host at the ragged unfolding's length" | "drawn on the host at the length of the store's ragged matricization or unfolding" |
| 101 | § The start: the pad-safe pinned completion, and why a QR | E6 | ``` 1.0 in span). This was accepted (D-42, D-49): the aligned output is representation-independent to `tol`. ``` | ``` 1.0 in span). This was accepted: the aligned output is representation-independent to `tol`. ``` |
| 103 | § The start: the pad-safe pinned completion, and why a QR | E6 | ``` against its own old frame (`tests/backend/test_padded_frame.py`, T5; `TestStartChains`). ``` | ``` against its own old frame (`tests/backend/test_padded_frame.py`; `TestStartChains`). ``` |
| 107 | § Jit: kernels, getters, and compile accounting | E6 | ``` Nothing jits unless asked (D-39). ``` | ``` Nothing jits unless asked. ``` |
| 125 | § Jit: kernels, getters, and compile accounting | E6 | ``` and sets no precedent (D-47; [`scan_body_principles.md`](scan_body_principles.md)). ``` | ``` and sets no precedent ([`scan_body_principles.md`](scan_body_principles.md)). ``` |
| 131 | § Jit: kernels, getters, and compile accounting | E6 | ``` The loops do not warn (D-40); the user doc recommends `use_jit=True` for jax. ``` | ``` The loops do not warn; the user doc recommends `use_jit=True` for jax. ``` |
| 160 | § Where the guarantees live | E6 | ``` - `tests/backend/test_padded_frame.py`: the ragged start (T5 convention reference), step (T1–T4 against dense ``` | ``` - `tests/backend/test_padded_frame.py`: the ragged start (the convention reference), step (against dense ``` |
| 161 | § Where the guarantees live | E6 | ``` oracles) and loop (T6, the pair rule). ``` | ``` oracles) and loop (the pair rule). ``` |
| 162 | § Where the guarantees live | E6 | ``` - `tests/backend/test_uniform_padded_frame.py`: the uniform twins against the ragged ones per element (T8–T12, T17), ``` | ``` - `tests/backend/test_uniform_padded_frame.py`: the uniform twins against the ragged ones per element, ``` |
| 163 | § Where the guarantees live | E6 | ``` the chains (T20). ``` | ``` the chains. ``` |
| 164 | § Where the guarantees live | E6 | ``` - `tests/backend/test_padded_start.py`: the seam (T14, T19). ``` | ``` - `tests/backend/test_padded_start.py`: the seam. ``` |
| 165 | § Where the guarantees live | E6 | ``` - `tests/test_aligned_padded_frame_frontend.py`: the frontend twins (T15). ``` | ``` - `tests/test_aligned_padded_frame_frontend.py`: the frontend twins. ``` |

### `docs/contributor/parameters_not_closures.md`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 107 | § Options are not parameters | E6 | ``` The exact Hessian's lift damping is the worked case (2026.3.0, I-22). It is not a property of the ``` | ``` The exact Hessian's lift damping is the worked case (2026.3.0). It is not a property of the ``` |

### `docs/contributor/precompute_and_caching.md`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 37 | § Three mechanisms, in increasing order of danger | E6 | ``` **The one seeded cache (2026-10-27, ruling I-38).** `T3Tangent._gauge_scale` / `UT3Tangent._gauge_scale` ``` | ``` **The one seeded cache (2026-10-27).** `T3Tangent._gauge_scale` / `UT3Tangent._gauge_scale` ``` |

### `docs/contributor/scan_body_principles.md`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 10 | § The problem | E11 | "`get_backend` dispatches `xscan` / `xmap` ... and `xwhile` to `jax.lax.while_loop`." | "`get_backend` dispatches `xscan` / `xmap` ...; `common.xwhile` (not from `get_backend`) uses `jax.lax.while_loop` only with `use_jit=True` and a jax state, a Python loop otherwise." |

### `docs/contributor/sharing_internals.md`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 29 | § The `S_i` machinery: recompute by zipper, never retain, never re-SVD | E6 | ``` hazards (symmetric tensors have repeated `Γ` eigenvalues). **History (2026-08-22 review, S9):** the ``` | ``` hazards (symmetric tensors have repeated `Γ` eigenvalues). **History (2026-08-22 review):** the ``` |
| 41 | § The `S_i` machinery: recompute by zipper, never retain, never re-SVD | N7 | "continuation restarts" | "warm start (zero-padded)" |
| 98 | § Padded restarts: which tangent channels are alive | N7 | "Padded restarts" | "warm start (zero-padded)" |
| 117 | § Padded restarts: which tangent channels are alive | N7 | "padded restart" | "warm start (zero-padded)" |
| 164 | § The uniform mirror: delegation, and two hard-won lessons | N7 | "padded restart" | "warm start (zero-padded)" |

### `docs/contributor/testing_strategy.md`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 125 | § Exercising a mask check is not testing it — isolate it | E6 | ``` - **S1.** `test_mask_mismatch_is_rejected` perturbed only the **tucker** mask. Deleting the tt-mask ``` | ``` - `test_mask_mismatch_is_rejected` perturbed only the **tucker** mask. Deleting the tt-mask ``` |
| 127 | § Exercising a mask check is not testing it — isolate it | E6 | ``` - **S3.** The frame-weight version *widened the supercores* as well as perturbing the mask. So the ``` | ``` - The frame-weight version *widened the supercores* as well as perturbing the mask. So the ``` |

### `docs/contributor/uniform_internals.md`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 4 | § Uniform layer — implementation internals | E6 | ``` (Started during the docs user/dev split, S1; the S2/S3 extractions and the P9 polymorphism-lenses ``` | ``` (Started during the docs user/dev split; later extractions and the polymorphism-lenses ``` |

### `docs/contributor/uniform_svd_prefix_orthogonalization.md`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 3 | § Why uniform orthogonalization must be SVD-based (the prefix-mask contract) | E4 | "The uniform layer represents per-stack ranks as **prefix** masks `[0, rank)`" | "Orthogonalization and `ut3svd` produce **prefix** masks `[0, rank)` (the canonical form; arithmetic can make masks gappy, `../uniform_masks_vs_ranks.md`)" |
| 68 | § The completion at zero-σ slots (pad-safe since 2026-08) | E6 | ``` the completion there, which the masks then erase — a silently lost tangent direction (review S1b; it hit ``` | ``` the completion there, which the masks then erase — a silently lost tangent direction (it hit ``` |

### `docs/contributor/verification_records.md`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 41 | § Correctness | E6 | ``` \| `<p, H p>` vs `d²/dt² f(R(tp))` (Richardson), resolving the curvature term itself \| within 1e-3 of the curvature term (T7, every kind) \| ``` | ``` \| `<p, H p>` vs `d²/dt² f(R(tp))` (Richardson), resolving the curvature term itself \| within 1e-3 of the curvature term (every kind) \| ``` |
| 50 | § Convergence | E6 | ``` - **E2 — the critic's first restart experiment was ONE unseeded draw.** `check_restart_tr.py` seeded only a ``` | ``` - **The critic's first restart experiment was ONE unseeded draw.** `check_restart_tr.py` seeded only a ``` |
| 53 | § Convergence | E6 | ``` the seeded rows instead (F19, F27). ``` | ``` the seeded rows instead. ``` |
| 54 | § Convergence | E6 | ``` - **F19 (seeded; probe derivatives, order 2, ω = `[1, .5, .25]`, restart rank 2 → 3, target 4, 5 % noise):** ``` | ``` - **Seeded: probe derivatives, order 2, ω = `[1, .5, .25]`, restart rank 2 → 3, target 4, 5 % noise:** ``` |
| 56 | § Convergence | E6 | ``` 0.5, 0 rejected. **F26** (its reference interplay): zero 7.01e4 vs warm start 2.07e4 (ratio 0.30); GN 36 vs 39 ``` | ``` 0.5, 0 rejected. **Its reference interplay:** zero 7.01e4 vs warm start 2.07e4 (ratio 0.30); GN 36 vs 39 ``` |
| 58 | § Convergence | E6 | ``` - **F27 (E2's three apply restarts, seeded):** warm/zero reference ratios 1.9e-2, 1.6e-14, 3.1e-1. The library's ``` | ``` - **The three apply restarts above, seeded:** warm/zero reference ratios 1.9e-2, 1.6e-14, 3.1e-1. The library's ``` |
| 62 | § Convergence | E6 | ``` - **F28 / F38 (no minimal reduction):** Tucker slack 20 outer / 4 rejected / 78 CG iterations → 2.6e-9; TT slack ``` | ``` - **No minimal reduction:** Tucker slack 20 outer / 4 rejected / 78 CG iterations → 2.6e-9; TT slack ``` |
| 66 | § Convergence | E6 | ``` problem — identical from the minimal-rank tensor, f = 1.335520e3, E11 — not of the frame.) ``` | ``` problem — identical from the minimal-rank tensor, f = 1.335520e3 — not of the frame.) ``` |
| 67 | § Convergence | E6 | ``` - **F22 / F29 (plain apply, target `(4,4,4)`, rank-2 fit, W = 200, 1 % noise; the GN warm-up uses ``` | ``` - **Plain apply, target `(4,4,4)`, rank-2 fit, W = 200, 1 % noise; the GN warm-up uses ``` |
| 70 | § Convergence | E6 | ``` 0.941 — the warm-up ends at a different point; the kernels are bit-identical. T16 itself is re-based on ``` | ``` 0.941 — the warm-up ends at a different point; the kernels are bit-identical. The rate test itself is re-based on ``` |
| 74 | § Convergence | E6 | ``` - **T21 / T25 re-measured (S5, S8):** T21 (restart, W = 60) ragged 8 outer / 0 rejected / 27 CG; uniform 8/0/27, ``` | ``` - **The restart and slack tests re-measured:** the restart (W = 60) ragged 8 outer / 0 rejected / 27 CG; uniform 8/0/27, ``` |
| 75 | § Convergence | E6 | ``` the same final objective. T25 Tucker slack 14/1/26, TT slack 22/6/70 — uniform backend and frontend identical ``` | ``` the same final objective. Tucker slack 14/1/26, TT slack 22/6/70 — uniform backend and frontend identical ``` |
| 76 | § Convergence | E6 | ``` counts, losses row by row ≤ 1.9e-13. T22 (shared restart) ragged 10/0/41, uniform 13/0/40 (same final ``` | ``` counts, losses row by row ≤ 1.9e-13. The shared restart: ragged 10/0/41, uniform 13/0/40 (same final ``` |
| 83 | § Cost and memory | E6 | ``` \| exact / GN Hessian-apply, probe derivatives (numpy CPU; d5 r8 o2 W300 and d4 r16 o3 W2000) \| S3: 2.2–2.7×; after the recurrence conversion (S3b) 2.8–3.5× — Gauss-Newton got 2.3× faster, the two-hole's dominant pieces were recurrences already (plan gate ≤ 3.5×) \| ``` | ``` \| exact / GN Hessian-apply, probe derivatives (numpy CPU; d5 r8 o2 W300 and d4 r16 o3 W2000) \| first build 2.2–2.7×; after the recurrence conversion 2.8–3.5× — Gauss-Newton got 2.3× faster, the two-hole's dominant pieces were recurrences already (plan gate ≤ 3.5×) \| ``` |
| 84 | § Cost and memory | E6 | ``` \| the same for the other kinds (S10–S12) \| probe 2.8–2.9×, apply 3.1–3.2×, entries 3.1–3.3×; apply derivatives 3.0–3.1× (d5) / 3.5–3.9× (d4) \| ``` | ``` \| the same for the other kinds \| probe 2.8–2.9×, apply 3.1–3.2×, entries 3.1–3.3×; apply derivatives 3.0–3.1× (d5) / 3.5–3.9× (d4) \| ``` |
| 86 | § Cost and memory | E6 | ``` \| two-hole per-row scratch vs the GN transpose (uniform, jax f32) \| the TT-assembly kernel alone (unchunked `memory_analysis`): 2.2–2.8× after S3b (was 2.5–3.5×); the running chunked program (the estimator's slope, current jax): ~1.3–2.2× (~1× on jaxlib 0.4.30); unchunked peak 1.85–2.1× \| ``` | ``` (as the current row, with "after S3b" → "after the recurrence conversion") ``` |
| 89 | § Cost and memory | E6 | ``` \| S0c uniform-transpose re-mask overhead \| F34: 0.08–0.61 % of the transpose; re-measured ~1.1 % eager on a small transpose including the host mask build, ~0.13 % large, 0 under jit \| ``` | ``` \| uniform-transpose re-mask overhead \| 0.08–0.61 % of the transpose; re-measured ~1.1 % eager on a small transpose including the host mask build, ~0.13 % large, 0 under jit \| ``` |
| 93 | § The zero reference | E6 | ``` - F30: per-layer LAPACK zero references differ ragged vs uniform by 0.56–1.93×, and 6 of 16 seeded `newton_cg` ``` | ``` - Per-layer LAPACK zero references differ ragged vs uniform by 0.56–1.93×, and 6 of 16 seeded `newton_cg` ``` |
| 94 | § The zero reference | E6 | ``` pairs drift; with one reference, 0 of 16. F31 / F37: the canonical frame is valid (orthogonality residual 0, ``` | ``` pairs drift; with one reference, 0 of 16. The canonical frame is valid (orthogonality residual 0, ``` |
| 98 | § The zero reference | E6 | ``` - F32: on data with no energy on the leading coordinates, the LAPACK and identity-prefix completions give a ``` | ``` - On data with no energy on the leading coordinates, the LAPACK and identity-prefix completions give a ``` |
| 100 | § The zero reference | E6 | ``` E9: a mode rotation moves the reference by ≤ 1.6–1.7×; a nonzero start's `‖g‖` is invariant. ``` | ``` A mode rotation moves the reference by ≤ 1.6–1.7×; a nonzero start's `‖g‖` is invariant. ``` |
| 104 | § The downstream measurement (S0b; T3Polynomial against the 2026.3.0 release candidate, small sizes) | E6 | ``` ### The downstream measurement (S0b; T3Polynomial against the 2026.3.0 release candidate, small sizes) ``` | ``` ### The downstream measurement (T3Polynomial against the 2026.3.0 release candidate, small sizes) ``` |
| 116 | § The Newton-schedule experiments (I-36 inputs; the records are archived under `dev/archive/`) | E6 | ``` ### The Newton-schedule experiments (I-36 inputs; the records are archived under `dev/archive/`) ``` | ``` ### The Newton-schedule experiments (the records are archived under `dev/archive/`) ``` |
| 161 | § The implementation (slice records) | E6 | ``` \| the ragged step vs the dense oracle, d ∈ {1, 2, 3, 5} and the prototype list \| T1: principal-angle sines and movements within 1e-9 (the bar); frames orthonormal to 1e-12; the old block bitwise; both base-point reconstructions to 1e-12 \| ``` | ``` \| the ragged step vs the dense oracle, d ∈ {1, 2, 3, 5} and the prototype list \| principal-angle sines and movements within 1e-9 (the bar); frames orthonormal to 1e-12; the old block bitwise; both base-point reconstructions to 1e-12 \| ``` |
| 164 | § The implementation (slice records) | E6 | ``` \| the start's convention (D-41) \| each store equals the seeded pinned QR of its keyed rows (T5); successive starts reuse no seed (T20, both layers) \| ``` | ``` \| the start's convention \| each store equals the seeded pinned QR of its keyed rows; successive starts reuse no seed (both layers) \| ``` |
| 165 | § The implementation (slice records) | E6 | ``` \| representation dependence (D-42, D-49; d = 4, generic, `t3svd` and shared points, three seeds each) \| the layers' starts differ by 0.95–1.00 in span; the aligned frames by 7e-4–3e-2 at the default `tol` (equal pass counts) and 8e-13–6e-11 at `tol=1e-10` (pass counts within 2) \| ``` | ``` \| representation dependence (d = 4, generic, `t3svd` and shared points, three seeds each) \| the layers' starts differ by 0.95–1.00 in span; the aligned frames by 7e-4–3e-2 at the default `tol` (equal pass counts) and 8e-13–6e-11 at `tol=1e-10` (pass counts within 2) \| ``` |

### `docs/contributor/weighted_internals.md`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 31 | § The tangent weight is a METRIC (the design change that caught us out) | E8 | "the frame's `d+1`-th left/right cores are base-point padding" | "the carriers `left[d-1]` / `right[0]` hold the base point" |

### `t3toolbox/__init__.py`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 9 | module | E12 | "the Gauss-Newton fitting models" | "the local models (`LocalModel` and its factories)" |

### `t3toolbox/backend/common.py`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 443 | `tree_contains_jax` | E6 | ``` return is_jax_ndarray(T)                                              # branch (review R2-8) ``` | ``` return is_jax_ndarray(T)                                              # branch ``` |
| 451 | `tree_to_jax` | E6 | ``` if isinstance(T, typ.Sequence) and not isinstance(T, (str, bytes)):   # str = a leaf (review R2-8) ``` | ``` if isinstance(T, typ.Sequence) and not isinstance(T, (str, bytes)):   # str = a leaf ``` |
| 514 | `readonly_mask_copies` | E6 | ``` (review H1-5; the ``_weight_matrix`` precedent).""" ``` | ``` (the ``_weight_matrix`` precedent).""" ``` |
| 526 | `ExplicitEquality` | E6 | ``` The library's equality rule (ruled 2026-08-24, review R1-13 extended): **hash/eq belongs to the ``` | ``` The library's equality rule (ruled 2026-08-24): **hash/eq belongs to the ``` |
| 583 | `_content_key` | E6 | ``` # dtype-blind array_equal eq that also zip-truncated) could disagree (review H1-6 / R2-7). ``` | ``` # dtype-blind array_equal eq that also zip-truncated) could disagree. ``` |
| 628 | `_value_key` | E6 | `_value_key` TypeError: 'a ValueHashedFields field holds a non-numpy array (%s): aux fields must be HOST data -- a device array belongs in the traced leaves, not the jit cache key (review R2-7)' | 'a ValueHashedFields field holds a non-numpy array (%s): aux fields must be HOST data -- a device array belongs in the traced leaves, not the jit cache key' |
| 753 | `require_concrete_masks` | E14 | "the (identity-hashed, never-inspected) output ``aux_data``" | "the (value-hashed) output ``aux_data``" |

### `t3toolbox/backend/fitting.py`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 8 | module | N1 | "sample stack" | "probe stack" |
| 148 | `block_sumsq_over_probes` | N1 | "sample stack" | "probe stack" |
| 154 | `_flat_gather` | N1 | "sample stack" | "probe stack" |
| 313 | `SamplingKind` | N1 | "sample stack" | "probe stack" |
| 351 | `transpose` | N1 | "sample stack" | "probe stack" |
| 380 | `corewise_hessian` | N1 | "sample stack" | "probe stack" |

### `t3toolbox/backend/fv_conversions.py`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 192 | `t3_orthogonal_representations` | E6 | ``` # The pad-safe uniform sweep (review S1b): every SVD below goes through pad_safe_svd, with ``` | ``` # The pad-safe uniform sweep: every SVD below goes through pad_safe_svd, with ``` |

### `t3toolbox/backend/fv_operations.py`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 98 | `fv_variation_shapes` | E17 | "``(rL_i, nU_i, rR_i)`` per TT variation" | "``(rL_i, nU_i, rR_(i+1))`` per TT variation" (the next lines already explain the shift) |
| 491 | `fv_center_factors` | M/U | "# the mode-1 unfolding" | "# the mode-1 matricization" |
| 578 | `fv_lift_factors` | N7 | "padded restart" | "warm start (zero-padded)" |
| 581 | `fv_lift_factors` | N7 | "continuation restart" | "warm start (zero-padded)" |
| 640 | `fv_absorb_weights` | E6 | ``` Approach-1 / metric-on-variations): ``down`` -> V's ``nD`` leg; ``up``/``left``/``right`` -> H's ``` | ``` i.e. on the variations): ``down`` -> V's ``nD`` leg; ``up``/``left``/``right`` -> H's ``` |

### `t3toolbox/backend/geometry.py`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 115 | `t3_left_orthogonal_norm_sq` | E6 | ``` return xnp.sum(last * last, axis=(-3, -2, -1))           # per stack element (review H3-5) ``` | ``` return xnp.sum(last * last, axis=(-3, -2, -1))           # per stack element ``` |
| 342 | `inner` | E6 | ``` matching the uniform twin and the frontend ``MANIFOLD.inner`` (review H3-5).""" ``` | ``` matching the uniform twin and the frontend ``MANIFOLD.inner``.""" ``` |
| 545 | `inner` | E6 | ``` stack, a 0-d scalar unstacked), matching the uniform twin (review H3-5).""" ``` | ``` stack, a 0-d scalar unstacked), matching the uniform twin.""" ``` |

### `t3toolbox/backend/linalg.py`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 8 | module | M/U | "(directions match the core-unfolding conventions)" | "(directions match the core matricization/unfolding conventions)" |
| 225 | `truncated_svd` | E6 | `truncated_svd` ValueError: 'truncated_svd: min_rank=%d > max_rank=%d -- the contract is 1 <= min_rank <= max_rank (min_rank used to silently win; review R3-6)' | 'truncated_svd: min_rank=%d > max_rank=%d -- the contract is 1 <= min_rank <= max_rank' |
| 330 | `up_svd` | M/U | "Compute (truncated) singular value decomposition of 3-tensor up unfolding." | "... of a 3-tensor's up matricization (middle index as rows)." |
| 348 | `up_svd` | M/U | "# up unfolding of U is orthonormal" | "# up matricization of U is orthonormal" |
| 463 | `up_svd_pair` | M/U | "Orthogonalizes ``G`` via its up unfolding" | "Orthogonalizes ``G`` via its up matricization" |
| 628 | `pad_safe_svd` | N7 | "continuation restart" | "warm start (zero-padded)" |
| 748 | `filtered_reciprocal` | N7 | "continuation restart" | "warm start (zero-padded)" |
| 1017 | `pad_safe_pinned_completion` | N8 | "seeded start" | "hint-free start" |

### `t3toolbox/backend/optimizer_display.py`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 121 | `_format_table` | C1 | `_REFERENCE_NAMES = {'zero': ..., 'warm_start': ..., 'override': ...}` (no `'cone'`, so a cone run's display names no reference) | add `'cone': 'cone (aligned zero tensor)'` (Nick's ruling), plus a display test |

### `t3toolbox/backend/optimizers.py`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 340 | `Problem` | N1 | "sample stack" | "probe stack" |
| 478 | `flat_draw` | N1 | "sample stack" | "probe stack" |
| 551 | `_prepare_jit_inputs` | E1 | "jax-absent policy, ``common._jax_or_warn``" | "jax-absent policy, ``common.jax_or_warn``" |
| 947 | `gradient_descent` | E6 | ``` alpha *= 0.5        # halve BETWEEN trials only (review R7-7) ``` | ``` alpha *= 0.5        # halve BETWEEN trials only ``` |
| 1161 | `NewtonInfo` | E2 | `NewtonInfo` docstring: g0_reference is "``'zero'`` ..., ``'warm_start'`` ..., or ``'override'``" | add "``'cone'`` (the gradient norm at the zero tensor on the aligned frame)" |
| 1208 | `NewtonInfo` | E2 | `g0_reference` field comment `'zero' \| 'warm_start' \| 'override'` | `'zero' \| 'warm_start' \| 'cone' \| 'override'` |
| 1328 | `newton_cg` | E2 | signature comment `'zero' \| 'warm_start'` | `'zero' \| 'warm_start' \| 'cone'` |
| 1417 | `newton_cg` | N14 | "whose own frame is arbitrary on the padding, and ... aligns the padding with the misfit gradient" | "whose own frame is arbitrary on the new slots, and ... aligns the new slots with the misfit gradient" |
| 1619 | `newton_cg` | E6 | ``` alpha *= 0.5        # halve BETWEEN trials only: alpha stays the last trial's step (review R7-7) ``` | ``` alpha *= 0.5        # halve BETWEEN trials only: alpha stays the last trial's step ``` |
| 1623 | `newton_cg` | E6 | ``` accepted = not (ls_failed and olsf == 'stop')                # 'stop' rejects: x unchanged (review R7-7) ``` | ``` accepted = not (ls_failed and olsf == 'stop')                # 'stop' rejects: x unchanged ``` |

### `t3toolbox/backend/padded_frames.py`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 7 | module | N8 | "seeded start" | "hint-free start" |
| 148 | `_seed_rows` | M/U | "# the seed length: the mode size (up) or the real rows of the core's unfolding" | "... or the real rows of the core's matricization (down) or unfolding (left, right)" |
| 254 | `fv_padded_frame` | N8 | "The seeded padded frame: the old frame in the leading slots" | "The hint-free padded frame: the old frame in the leading slots" |
| 294 | `_unfold` | M/U | "A core unfolded with one slot as the columns (core axes only)." | "A core reshaped with one slot as the columns: an unfolding for an end slot, a matricization for the middle slot (core axes only)." |
| 301 | `_fold` | M/U | "# C+(rows, cols): an unfolding" | "# C+(rows, cols): a matricization or unfolding" |
| 722 | `fv_aligned_padded_frame` | N8 | "seeded start" | "hint-free start" |

### `t3toolbox/backend/probing.py`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 222 | `_mu_step` | E22 | "One edge of the leftward sweep of compute_mu" | "One edge of the left-to-right sweep of compute_mu" |
| 233 | `compute_mu` | E22 | "Compute leftward edge variables" | "Compute the left edge variables mu (accumulated left to right)" |
| 258 | `compute_nu` | E22 | "Compute rightward edge variables" | "Compute the right edge variables nu (accumulated right to left)" |
| 374 | `_sigma_step` | E22 | "perturbation-leftward (sigma) recursion" | "left-side perturbation (sigma) recursion" |
| 408 | `compute_sigma` | E22 | "var-leftward edge variables sigma" | "left-side variation edge variables sigma" |
| 445 | `compute_tau` | E22 | "var-rightward edge variables tau" | "right-side variation edge variables tau" |
| 646 | `tv_probe` | N17 | "up_tucker_supercore ... up orthogonal elements" | HELD (N17): "down-orthogonal elements" |
| 647 | `tv_probe` | N17 | "down_tt_supercore ... down orthogonal elements" | HELD (N17): "up-orthogonal elements" |
| 899 | `_tau_tilde_step` | E22 | "the adjoint rightward sweep of compute_tau_tilde" | "One edge of the adjoint left-to-right sweep of compute_tau_tilde" (it is a forward `xscan`) |
| 917 | `compute_tau_tilde` | E22 | "Adjoint-var-rightward edge variables tau_tilde" | "Adjoint right-side variation edge variables tau_tilde" |
| 943 | `compute_sigma_tilde` | E22 | "Adjoint-var-leftward edge variables sigma_tilde" | "Adjoint left-side variation edge variables sigma_tilde" |
| 1221 | `tv_probe_transpose` | N17 | "up_tucker_supercore ... up orthogonal elements" | HELD (N17): "down-orthogonal elements" |
| 1222 | `tv_probe_transpose` | N17 | "down_tt_supercore ... down orthogonal elements" | HELD (N17): "up-orthogonal elements" |

### `t3toolbox/backend/ranks.py`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 308 | `edge_condition_numbers` | M/U | "Edge condition numbers ... of the matrix unfoldings." | "... of the matricizations and unfoldings." (the next lines already say so) |
| 379 | `compute_continuation_ranks` | M/U | "from the current iterate's unfolding singular values" | "from the current iterate's matricization and unfolding singular values" |
| 651 | `compute_orthogonal_representation_ranks` | E6 | ``` non-minimal TT bond they do not, and the uniform frame's left/down masks were too small (review S1). ``` | ``` non-minimal TT bond they do not, and the uniform frame's left/down masks were too small. ``` |
| 746 | `frame_has_minimal_ranks` | N7 | "continuation restart" | "warm start (zero-padded)" |

### `t3toolbox/backend/regularization.py`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 12 | module | E15 | "each method receives a ``GeometryOps``" | "each method receives a geometry (an `optimizers.Geometry`, e.g. `ManifoldGeometryOps`)" |
| 14 | module | M/U | "the Grasedyck-Kramer inverse-unfolding-singular-value weighting" | "the Grasedyck-Kramer inverse-singular-value weighting of the matricizations and unfoldings" |

### `t3toolbox/backend/sampling_derivatives.py`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 36 | module | N2 | `t = 0, 1, ..., K` | `t = 0, 1, ..., order` |
| 47 | module | E16 | "via tWCi_Cio_to_tWCo" | "via `contract('tWCi,Cio->tWCo', ...)`" |
| 57 | module | N1 | "sample stack" | "probe stack" |
| 155 | `t3_probe_derivatives` | N1 | "sample stack" | "probe stack" |
| 173 | `t3_probe_derivatives` | N2 | "highest derivative order K." | "highest derivative order." |
| 211 | `check_perturbation_vectors` | N1 | "sample stack" | "probe stack" |
| 217 | `check_perturbation_vectors` | N1 | "sample stack" | "probe stack" |
| 227 | `check_perturbation_index` | N1 | "sample stack" | "probe stack" |
| 235 | `check_perturbation_index` | N1 | "sample stack" | "probe stack" |
| 524 | `assemble_z_jets` | E16 | "(``tWCi_Cio_to_tWCo``)" | "(``contract('tWCi,Cio->tWCo', ...)``)" |
| 740 | `compute_sigma_jets` | E22 | "Variation-leftward edge-variable jets sigma" | "Left-side variation edge-variable jets sigma" |
| 770 | `compute_tau_jets` | E22 | "Variation-rightward edge-variable jets tau" | "Right-side variation edge-variable jets tau" |
| 878 | `assemble_tangent_z_jets` | E16 | "(``tWCi_Cio_to_tWCo``)" | "(``contract('tWCi,Cio->tWCo', ...)``)" |
| 951 | `tv_probe_derivatives` | N1 | "sample stack" | "probe stack" |
| 1028 | `t3_apply_derivatives` | N1 | "sample stack" | "probe stack" |
| 1123 | `tv_apply_derivatives` | N1 | "sample stack" | "probe stack" |
| 1230 | `tv_entries_derivatives` | N1 | "sample stack" | "probe stack" |
| 1359 | `compute_tau_tilde_jets` | E22 | "Adjoint-var-rightward edge-variable jets tau_tilde" | "Adjoint right-side variation edge-variable jets tau_tilde" |
| 1370 | `compute_sigma_tilde_jets` | E22 | "Adjoint-var-leftward edge-variable jets sigma_tilde" | "Adjoint left-side variation edge-variable jets sigma_tilde" |
| 1747 | `assemble_tt_variation_jets` | N1 | "sample stack" | "probe stack" |
| 1834 | `_chunk_slope` | N1 | "sample stack" | "probe stack" |
| 2027 | `tv_probe_transpose_derivatives_from_sweep` | N1 | "sample stack" | "probe stack" |
| 2062 | `tv_probe_derivatives_transpose` | N1 | "sample stack" | "probe stack" |
| 2075 | `tv_probe_derivatives_transpose` | N1 | "sample stack" | "probe stack" |
| 2077 | `tv_probe_derivatives_transpose` | N1 | "sample stack" | "probe stack" |
| 2082 | `tv_probe_derivatives_transpose` | N1 | "sample stack" | "probe stack" |
| 2261 | `tv_apply_transpose_derivatives_from_sweep` | N1 | "sample stack" | "probe stack" |
| 2285 | `tv_apply_derivatives_transpose` | N1 | "sample stack" | "probe stack" |
| 2295 | `tv_apply_derivatives_transpose` | N1 | "sample stack" | "probe stack" |
| 2321 | `tv_entries_transpose_derivatives_from_sweep` | N1 | "sample stack" | "probe stack" |
| 2360 | `tv_entries_derivatives_transpose` | N1 | "sample stack" | "probe stack" |
| 2400 | `t3_probe_corewise_derivatives_transpose` | N1 | "sample stack" | "probe stack" |
| 3458 | `dense_probe_derivatives` | N2 | `elm_shape=(K+1,Ni)` | `elm_shape=(order+1,Ni)` |

### `t3toolbox/backend/sharing.py`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 505 | `fv_shared_frame_data` | E6 | ``` and the tied projection was silently 30% off, review S9.) ``` | ``` and the tied projection was silently 30% off.) ``` |
| 595 | `fv_share_tucker_variations` | N7 | "continuation restarts" | "warm start (zero-padded)" |
| 760 | `fv_tied_lift_factors` | N7 | "padded restart" | "warm start (zero-padded)" |
| 763 | `fv_tied_lift_factors` | N7 | "Padded restarts" | "warm start (zero-padded)" |
| 1010 | `ufv_shared_frame_data` | E6 | ``` the 'frames packed from ragged are not guaranteed' caveat, went with it -- review S9.) The padded ``` | ``` the 'frames packed from ragged are not guaranteed' caveat, went with it.) The padded ``` |
| 1024 | `ufv_shared_frame_data` | E6 | ``` # went with it, review S9.) ``` | ``` # went with it.) ``` |

### `t3toolbox/backend/stacking.py`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 41 | `_is_branch` | E6 | ``` treating them as branches recurses forever (review R2-8); every tree recursion here uses this.""" ``` | ``` treating them as branches recurses forever; every tree recursion here uses this.""" ``` |
| 526 | `tree_zip` | E6 | `tree_zip` ValueError: 'tree_zip: trees differ in structure (a branch of length %d paired with %r) -- a silent zip would truncate to the shorter branch (review R2-12)' | 'tree_zip: trees differ in structure (a branch of length %d paired with %r); zipping them would silently truncate to the shorter branch' |

### `t3toolbox/backend/t3_linalg.py`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 120 | `t3_sum_stack` | E6 | ``` elif isinstance(axis, np.ndarray):                     # ndarray axis (0-d or 1-d) works too (R2-13) ``` | ``` elif isinstance(axis, np.ndarray):                     # ndarray axis (0-d or 1-d) works too ``` |
| 192 | `t3_inner_product` | E6 | ``` use_jax = any([is_jax_ndarray(c) for c in tuple(x[0]) + tuple(x[1]) + tuple(y[0]) + tuple(y[1])])   # list/tuple mix ok (R2-13) ``` | ``` use_jax = any([is_jax_ndarray(c) for c in tuple(x[0]) + tuple(x[1]) + tuple(y[0]) + tuple(y[1])])   # list/tuple mix ok ``` |

### `t3toolbox/backend/t3_orthogonalization.py`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 106 | `_up_orthogonalize_tt_cores_step` | N17 | "# Oaxb. outer-orthogonal tt core" | HELD (N17): keep, or "up-orthogonal tt core" |
| 147 | `_down_orthogonalize_tucker_cores_step` | N17 | "# new_Uxo.  up-orthogonal tucker core" | HELD (N17): "down-orthogonal tucker core" |
| 310 | `t3_down_svd_tt_core` | M/U | "Compute SVD of ith TT-core outer unfolding and keep non-orthogonal factor with this core." | "Compute the SVD of the ith TT core's matricization (mode index as rows) and keep the non-orthogonal factor with this core." |
| 347 | `t3_up_svd_tt_core` | M/U | `t3_up_svd_tt_core`: "Compute SVD of ith TT-core right unfolding and contract non-orthogonal factor down into the tucker core below." | "Compute the SVD of the ith TT core's matricization (mode index as rows) and contract the non-orthogonal factor down into the Tucker core below." (it calls `up_svd_pair`; "right unfolding" is also factually wrong) |

### `t3toolbox/backend/tt_orthogonalization.py`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 40 | `_tt_left_step_pad_safe` | E6 | ``` """Pad-safe twin of :py:func:`_tt_left_step` for the uniform representation (review S1b): the ``` | ``` """Pad-safe twin of :py:func:`_tt_left_step` for the uniform representation: the ``` |
| 76 | `tt_left_orthogonalize` | E6 | ``` completion columns stay off the padded slots (review S1b) and the output chain is bitwise-clean. ``` | ``` completion columns stay off the padded slots and the output chain is bitwise-clean. ``` |

### `t3toolbox/backend/tv_operations.py`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 164 | `tv_oblique_gauge_projection` | N4 | "down/outer" | "down" |
| 181 | `tv_oblique_gauge_projection` | N4 | "down/outer" | "down" |

### `t3toolbox/backend/ufv_conversions.py`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 104 | `ut3_orthogonal_representations` | E6 | ``` # SVD-based; uniform_masks makes every SVD in the sweep PAD-SAFE (review S1b: at a numerically ``` | ``` # SVD-based; uniform_masks makes every SVD in the sweep PAD-SAFE (at a numerically ``` |

### `t3toolbox/backend/ufv_masking.py`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 205 | `ufv_variation_masks_over_stack` | E6 | ``` ``C``-stack mask to ``K+C`` variations and fail in a reshape, review C3). Host numpy: static structure.""" ``` | ``` ``C``-stack mask to ``K+C`` variations and fail in a reshape). Host numpy: static structure.""" ``` |

### `t3toolbox/backend/ufv_operations.py`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 277 | `ufv_frame_orthogonality_residual` | N4 | "down/outer" | "down" |
| 294 | `dev` | N4 | "down/outer" | "down" |
| 537 | `ufv_tied_lift_factors` | N7 | "padded restart" | "warm start (zero-padded)" |

### `t3toolbox/backend/uniform_fitting.py`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 17 | module | E6 | ``` **Protocol note (review H2-9, ruled keep + document):** the kinds' ``forward`` / ``transpose`` keep ``` | ``` **Protocol note (ruled: keep and document):** the kinds' ``forward`` / ``transpose`` keep ``` |

### `t3toolbox/backend/uniform_padded_frames.py`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 7 | module | N5 | "A-block" | "old block" |
| 7 | module | N5 | "P-block" | "new block" |
| 11 | module | N8 | "seeded start" | "hint-free start" |
| 13 | module | M/U | "generated on the host at the ragged unfolding's size" | "... at the size of the store's ragged matricization or unfolding" |
| 91 | `_store_seeds` | M/U | "# ... the unfolding's first row slot, real slots" | "# ... the matricization's or unfolding's first row slot, real slots" |
| 101 | `_store_seeds` | M/U | "placed at the unfolding's real rows" | "placed at the real rows of the store's matricization or unfolding" |
| 117 | `_store_completion` | M/U | "# ... the old cores unfolded at the new widths" | "# ... the old cores matricized or unfolded at the new widths" |
| 120 | `_store_completion` | M/U | "# ... the unfolding's real rows" | "# ... the real rows of the matricization or unfolding" |
| 179 | `ufv_padded_frame` | N5 | "A-block" | "old block" |
| 179 | `ufv_padded_frame` | N5 | "P-block" | "new block" |
| 180 | `ufv_padded_frame` | N5 | "A-block" | "old block" |
| 182 | `ufv_padded_frame` | N5 | "A-block" | "old block" |
| 182 | `ufv_padded_frame` | N8 | "The seeded padded frame in block layout: the old frame in the A-blocks" | "The hint-free padded frame in block layout: the old frame in the old blocks" (also N5) |
| 266 | `ufv_padded_frame` | N5 | "A-block" | "old block" |
| 266 | `ufv_padded_frame` | N5 | "P-block" | "new block" |
| 269 | `ufv_padded_frame` | N5 | "A-block" | "old block" |
| 509 | `_selection` | N5 | "A-block" | "old block" |
| 511 | `_selection` | N5 | "A-block" | "old block" |
| 623 | `utv_align_padded_frame` | N5 | "A-block" | "old block" |
| 623 | `utv_align_padded_frame` | N5 | "P-block" | "new block" |
| 653 | `utv_align_padded_frame` | N5 | "A-block" | "old block" |

### `t3toolbox/backend/ut3_linalg.py`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 202 | `_ut3_left_orthogonalized` | E6 | ``` # gradient came out NaN (review S11). The derivative is instead the exact multilinear rule ``` | ``` # gradient came out NaN. The derivative is instead the exact multilinear rule ``` |

### `t3toolbox/backend/ut3_masking.py`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 10 | module | E4 | "Masks are boolean prefix vectors, static structure, ..." | "Masks are boolean vectors (prefix in canonical form, possibly gappy after arithmetic), static structure, ..." |

### `t3toolbox/backend/ut3_orthogonalization.py`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 9 | module | E6 | ``` mask-aware **pad-safe** SVD the sweeps use (review S1b) is derived in ``docs/pad_safe_svd.tex``. ``` | ``` mask-aware **pad-safe** SVD the sweeps use is derived in ``docs/pad_safe_svd.tex``. ``` |
| 138 | `down_orthogonalize_tucker_supercores` | E6 | `at a numerically rank-deficient point the sigma~0 completion columns stay OFF the padding (review` | `at a numerically rank-deficient point the sigma~0 completion columns stay OFF the padding (a` |
| 139 | `down_orthogonalize_tucker_supercores` | E6 | `S1b -- a black-box SVD may place them in padded slots, which the masks then erase: a lost tangent` | `black-box SVD may place them in padded slots, which the masks then erase: a lost tangent` |
| 189 | `up_orthogonalize_tt_supercores` | N17 | "new_tt = down-orthogonal O" | HELD (N17): "new_tt = up-orthogonal O" |
| 203 | `up_orthogonalize_tt_supercores` | E6 | ``` # pad-safe (review S1b): sigma~0 completions stay off the padded (a,b)/rank slots ``` | ``` # pad-safe: sigma~0 completions stay off the padded (a,b)/rank slots ``` |

### `t3toolbox/backend/ut3_svd.py`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 247 | `_ut3svd_step_pad_safe` | E6 | ``` """Pad-safe twin of :py:func:`_ut3svd_step` (review S1b): both kept-basis SVDs go through ``` | ``` """Pad-safe twin of :py:func:`_ut3svd_step`: both kept-basis SVDs go through ``` |
| 335 | `ut3svd_supercores` | E6 | ``` # kept-basis SVD (pre-orthogonalization + scan) is PAD-SAFE (review S1b). ``` | ``` # kept-basis SVD (pre-orthogonalization + scan) is PAD-SAFE. ``` |
| 453 | `_ut3svd_shared_step_pad_safe` | E6 | ``` """Pad-safe twin of :py:func:`_ut3svd_shared_step` (review S1b): the bond-rounding SVD goes ``` | ``` """Pad-safe twin of :py:func:`_ut3svd_shared_step`: the bond-rounding SVD goes ``` |
| 489 | `_ut3svd_shared_supercores` | E6 | ``` # kept-basis SVD (pre-orthogonalization + all four phases) is PAD-SAFE (review S1b). ``` | ``` # kept-basis SVD (pre-orthogonalization + all four phases) is PAD-SAFE. ``` |
| 521 | `_ut3svd_shared_supercores` | E6 | ``` # -- pad-safe content recurrences (HOST; review S1b), phase by phase -- ``` | ``` # -- pad-safe content recurrences (HOST), phase by phase -- ``` |

### `t3toolbox/backend/utv_operations.py`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 583 | `utv_project_ut3_onto_tangent_space` | E6 | ``` does -- without it a raw-``.data`` user of a shared frame gets an untied projection (review R9-11). ``` | ``` does -- without it a raw-``.data`` user of a shared frame gets an untied projection. ``` |

### `t3toolbox/backend/utv_sampling.py`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 199 | `utv_apply_transpose_from_sweep` | E6 | ``` sum_over_probes=False,   # matches the ragged twin (review H2-4) ``` | ``` sum_over_probes=False,   # matches the ragged twin ``` |
| 211 | `utv_probe_transpose_from_sweep` | E6 | ``` sum_over_probes=False,   # matches the ragged twin (review H2-4) ``` | ``` sum_over_probes=False,   # matches the ragged twin ``` |
| 223 | `utv_entries_transpose_from_sweep` | E6 | ``` sum_over_probes=False,   # matches the ragged twin (review H2-4) ``` | ``` sum_over_probes=False,   # matches the ragged twin ``` |
| 330 | `utv_apply_transpose_derivatives_from_sweep` | E6 | ``` sum_over_probes=False,   # matches the ragged twin (review H2-4) ``` | ``` sum_over_probes=False,   # matches the ragged twin ``` |
| 343 | `utv_probe_transpose_derivatives_from_sweep` | E6 | ``` sum_over_probes=False,   # matches the ragged twin (review H2-4) ``` | ``` sum_over_probes=False,   # matches the ragged twin ``` |
| 359 | `utv_entries_transpose_derivatives_from_sweep` | E6 | ``` sum_over_probes=False,   # matches the ragged twin (review H2-4) ``` | ``` sum_over_probes=False,   # matches the ragged twin ``` |

### `t3toolbox/fitting.py`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 7 | module | N11 | ``½‖F(X) − y‖²`` | ``½‖S(X) − y‖²`` |
| 205 | `LocalModel` | N11 | ``r = F(X) − y`` | ``r = S(X) − y`` |
| 306 | `LocalModel` | N11 | `# r = F(X) − y` | `# r = S(X) − y` |
| 641 | `_check_ragged_sample` | E6 | `errors at construction instead of an einsum / index error deep inside the first gradient` | `errors at construction instead of an einsum / index error deep inside the first gradient.` (the next line drops its parenthesis) |
| 642 | `_check_ragged_sample` | E6 | ``` (the 2026-08-22 review, C13). ``W`` is read off the sample; ``C`` is the point's stack.""" ``` | ``` ``W`` is read off the sample; ``C`` is the point's stack.""" ``` |

### `t3toolbox/frame_variations_format.py`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 63 | `T3Frame` | N4 | diagram cores `D0`, `D1`, `D2` | `O0`, `O1`, `O2` |
| 72 | `T3Frame` | N4 | `down_tt_cores = (D0, ..., D(d-1))` | `down_tt_cores = (O0, ..., O(d-1))` |
| 73 | `T3Frame` | E17 | "left_tt_cores = (L0, ..., L(d-1)), elm_shape=(rLi, ni, rL(i+1))" | "elm_shape=(rLi, nUi, rL(i+1))" |
| 74 | `T3Frame` | E17 | "right_tt_cores = (R0, ..., R(d-1)), elm_shape=(rRi, ni, rR(i+1))" | "elm_shape=(rRi, nUi, rR(i+1))" |
| 78 | `T3Frame` | E17 | "tt_variations = (H0, ..., H(d-1)), elm_shape=(rLi, nUi, rRi)" | "elm_shape=(rLi, nUi, rR(i+1))" (`fv_variation_shapes` reads `right[i].shape[-1]`) |
| 80 | `T3Frame` | E8 | "Note that Ld and R0 are not used in these diagrams." | "Note that L(d-1) and R0 (the carriers) are not used in these diagrams." |
| 93 | `T3Frame` | E17 | diagram label "Ud" | "U(d-1)" |
| 101 | `T3Frame` | N4 | rank diagram core "D1" (`T3Frame`; the next line already says `nO1`) | "O1" |
| 105 | `T3Frame` | E17 | diagram label "Ud" | "U(d-1)" |
| 118 | `T3Frame` | N17 | "down_tt_cores = (O0,...,O(d-1)), outer-orthogonal  O_aib O_ajb = delta_ij" | HELD (N17): keep, or "up-orthogonal" |
| 128 | `T3Frame` | E17 | "the N1 x ... x Nd tangent vectors" | "the N0 x ... x N(d-1) tangent vectors" |
| 309 | `padded` | N8 | "# the seeded padded frame (old frame in the le..." | "# the hint-free padded frame (old frame in the le..." |
| 310 | `padded` | N8 | "The seeded padded frame of ``x_old`` at larger ranks" | "The hint-free padded frame of ``x_old`` at larger ranks" |
| 355 | `is_orthogonal` | N4 | "down/outer D_i (all i):   ``einsum('...iaj,...ibj->...ab', D, D) = I``" | "down O_i (all i):   ``einsum('...iaj,...ibj->...ab', O, O) = I``" |
| 365 | `is_orthogonal` | N17 | "Orthogonal cores (left/right/outer/Tucker)" | HELD (N17): keep, or "(left/right/up/down)" |
| 1271 | `fv_to_t3` | N4 | `fv_to_t3` diagram core "D2" | "O2" |
| 1371 | `t3_orthogonal_representations` | N17 | "tucker_cores = (U0,U1, U2, U3), up orthogonal" | HELD (N17): "down-orthogonal" |
| 1372 | `t3_orthogonal_representations` | N17 | "down_tt_cores = (O0, O1, O2, O3), down orthogonal" | HELD (N17): "up-orthogonal" |
| 1438 | `T3FrameWeights` | E6 | ``` Four families, each ``len=d`` (one per variation core, Approach-1 / metric-on-variations): ``` | ``` Four families, each ``len=d`` (one per variation core; the weights are a metric on the variations): ``` |

### `t3toolbox/manifold.py`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 451 | `__truediv__` | E6 | ``` """Scale by ``1/s`` (scalar only) -- ``v / 2 == v * 0.5`` (review R1-12).""" ``` | ``` """Scale by ``1/s`` (scalar only) -- ``v / 2 == v * 0.5``.""" ``` |
| 976 | `probe_derivatives` | N1 | "sample stack" | "probe stack" |
| 977 | `probe_derivatives` | N1 | "sample stack" | "probe stack" |
| 1018 | `apply_derivatives` | N1 | "sample stack" | "probe stack" |
| 1093 | `probe_derivatives_transpose` | N1 | "sample stack" | "probe stack" |
| 1099 | `probe_derivatives_transpose` | N1 | "sample stack" | "probe stack" |
| 1104 | `probe_derivatives_transpose` | N1 | "sample stack" | "probe stack" |
| 1249 | `stack_tangents` | E6 | `stack_tangents` ValueError: '... or unstack the K-stacked one first (unstack_tangents). (Used to die inside numpy with an inhomogeneous-shape error -- review H3-6.)' | 'stack_tangents: every leaf must carry the same tangent stack K to stack into one outer level; got K shapes %s. Stack the K=() leaves separately, or unstack the K-stacked one first (unstack_tangents).' |
| 1343 | `frame` | N17 | "(the left/right/outer-orthogonal representation)" | HELD (N17): keep, or "(the orthogonal representation)" |
| 1638 | `retract` | E6 | ``` slots hold different cores, e.g. a slack frame with ``nD != nU`` -- review H3-8); when the ``` | ``` slots hold different cores, e.g. a slack frame with ``nD != nU``); when the ``` |
| 1675 | `_flatten_tangents` | E6 | ``` # ---- The operand scale of the safe-mode gauge check (I-38). Shared with the uniform twin (UT3Tangent): both ``` | ``` # ---- The operand scale of the safe-mode gauge check. Shared with the uniform twin (UT3Tangent): both ``` |

### `t3toolbox/optimizers.py`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 518 | `newton_cg` | E2 | signature comment `'zero' \| 'warm_start'` | `'zero' \| 'warm_start' \| 'cone'` |
| 595 | `newton_cg` | N14 | "by aligning the padding with the misfit gradient" | "by aligning the new slots with the misfit gradient" |

### `t3toolbox/safety.py`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 159 | `comparison_rtol` | E1 | "(``numerically_equal`` and friends)" | "(``allclose`` and friends)" |
| 178 | `_inside_jax_trace` | E6 | ``` Deliberately jit-only (review H6-12): under ``grad``/``vmap`` with concrete operands the probe returns ``` | ``` Deliberately jit-only: under ``grad``/``vmap`` with concrete operands the probe returns ``` |

### `t3toolbox/shared_geometry.py`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 133 | `__init__` | E6 | ``` # the backend geometries' canonical_groups keys (review H2-8). ``` | ``` # the backend geometries' canonical_groups keys. ``` |
| 158 | `__eq__` | E6 | ``` and other._partition == self._partition)   # canonical: label spelling is irrelevant (H2-8) ``` | ``` and other._partition == self._partition)   # canonical: label spelling is irrelevant ``` |
| 186 | `frame` | E6 | `frame` TypeError: 'this SharedGeometry wraps the UNIFORM base %s: frame() takes a UniformTuckerTensorTrain, got %s -- build the geometry from the ragged base (shared(MANIFOLD/COREWISE, ...)) for ragged points (review R9-9)' | 'this SharedGeometry wraps the UNIFORM base %s: frame() takes a UniformTuckerTensorTrain, got %s -- build the geometry from the ragged base (shared(MANIFOLD/COREWISE, ...)) for ragged points' |
| 192 | `frame` | E6 | `frame` TypeError: 'this SharedGeometry wraps the RAGGED base %s: frame() takes a TuckerTensorTrain, got %s -- use shared(UNIFORM_MANIFOLD/UNIFORM_COREWISE, ...) for uniform points (review R9-9)' | 'this SharedGeometry wraps the RAGGED base %s: frame() takes a TuckerTensorTrain, got %s -- use shared(UNIFORM_MANIFOLD/UNIFORM_COREWISE, ...) for uniform points' (move the `%` onto the previous literal's line) |
| 207 | `shared_frame_data` | E6 | `shared_frame_data` TypeError: 'this SharedGeometry wraps the UNIFORM base %s: expected a UT3Frame, got %s (review R9-9)' | 'this SharedGeometry wraps the UNIFORM base %s: expected a UT3Frame, got %s' |
| 210 | `shared_frame_data` | E6 | `shared_frame_data` TypeError: 'this SharedGeometry wraps the RAGGED base %s: expected a T3Frame, got %s (review R9-9)' | 'this SharedGeometry wraps the RAGGED base %s: expected a T3Frame, got %s' |
| 428 | `_labels_from_partition` | E6 | ``` # aux carries the TYPE (a subclass survives the round-trip -- review R7-13) and the CANONICAL ``` | ``` # aux carries the TYPE (a subclass survives the round-trip) and the CANONICAL ``` |
| 429 | `_labels_from_partition` | E6 | ``` # partition (label spelling never splits the jit cache -- review H2-8). The reconstructed ``` | ``` # partition (label spelling never splits the jit cache). The reconstructed ``` |

### `t3toolbox/tucker_tensor_train.py`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 66 | `TuckerTensorTrain` | E17 | diagram label "nd" | "n(d-1)" |
| 70 | `TuckerTensorTrain` | E17 | diagram label "Nd" | "N(d-1)" |
| 101 | `TuckerTensorTrain` | E17 | "``tucker_ranks = (n0, r1, ..., n(d-1))``" | "``tucker_ranks = (n0, n1, ..., n(d-1))``" |
| 615 | `get_minimal_ranks` | M/U | "Down TT core unfoldings are full rank" | "... TT core matricizations are full rank" (the adjective is N17, HELD) |
| 624 | `get_minimal_ranks` | E19 | "(N*...*Ni) x (N(i+1)*...*N(d-1)) matrix unfoldings" | "(N0*...*N(i-1)) x (Ni*...*N(d-1)) matrix unfoldings" (bond ri) |
| 625 | `get_minimal_ranks` | E19 | "Ni x (N1*...*N(i-1)*N(i+1)*...*N(d-1)) matricizations" | "Ni x (N0*...*N(i-1)*N(i+1)*...*N(d-1)) matricizations" |
| 2138 | `t3m` | E6 | ``` __rmul__ = __mul__      # scalar * x == x * scalar (elementwise * commutes; review R1-12) ``` | ``` __rmul__ = __mul__      # scalar * x == x * scalar (elementwise * commutes) ``` |
| 2183 | `corewise_equal` | E1 | "equality use :py:meth:`numerically_equal`." | "equality use :py:meth:`allclose`." |
| 2956 | `up_svd_tt_core` | M/U | `up_svd_tt_core`: "Compute SVD of ith TT-core right unfolding and contract non-orthogonal factor down into the tucker core below." | as `t3_up_svd_tt_core` above |
| 2978 | `up_svd_tt_core` | N17 | `up_svd_tt_core` Returns: "with ith TT-core down orthogonal" | HELD (N17): "up-orthogonal" |
| 3002 | `up_svd_tt_core` | N17 | "# TT-core is down orthogonal" | HELD (N17): "# TT-core is up-orthogonal" |
| 3108 | `orthogonalize_relative_to_tt_core` | N17 | "# Core below G1 is up orthogon[al]" (a Tucker core, `B Bᵀ = I`) | HELD (N17): "down-orthogonal" |
| 3920 | `probe_derivatives` | N1 | "sample stack" | "probe stack" |
| 3921 | `probe_derivatives` | N6 | "for the Riemannian Jacobian use" | "for the (bare) Jacobian use" |
| 3957 | `apply_derivatives` | N1 | "sample stack" | "probe stack" |
| 4210 | `t3svd` | M/U | "the singular values of the dense matrix unfoldings" | "the singular values of the dense matricizations and unfoldings" |
| 4225 | `t3svd` | M/U | "a smooth tensor has gradually decaying unfolding spectra" | "... gradually decaying matricization and unfolding spectra" |
| 4433 | `continuation_ranks` | M/U | "feeds the unfolding singular values to" | "feeds the matricization and unfolding singular values to" |
| 4621 | `t3svd_dense` | E17 | "# shape=stack_shape+(N1, N2, .., Nd)" | "# shape=stack_shape+(N0, ..., N(d-1))" |
| 4926 | `_require_t3_weights_consistent` | E6 | ``` a deep einsum broadcast error, or to broadcast silently (review R1-15 / R10-6).""" ``` | ``` a deep einsum broadcast error, or to broadcast silently.""" ``` |
| 4962 | `t3_weighted_norm` | E6 | ``` (the kwarg its backend and uniform twins always had -- review H2-6).""" ``` | ``` (the kwarg its backend and uniform twins always had).""" ``` |

### `t3toolbox/uniform_frame_variations_format.py`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 52 | `UT3FrameMasks` | E4 | "Slot ``j`` of an edge is real iff its mask is ``True`` there (the prefix/canonical form)." (`UT3FrameMasks`) | "Slot ``j`` of an edge is real iff its mask is ``True`` there; canonically a prefix, gappy in a padded frame's block layout." |
| 67 | `__post_init__` | E6 | ``` # writeable caller array mutated in place would leave a stale jit cache key (review H1-5). ``` | ``` # writeable caller array mutated in place would leave a stale jit cache key. ``` |
| 241 | `padded` | N8 | "# the seeded padded frame in block layout" | "# the hint-free padded frame in block layout" |
| 244 | `padded` | N8 | "The seeded padded frame of ``x_old`` at larger ranks, in block layout" | "The hint-free padded frame of ``x_old`` at larger ranks, in block layout" |
| 548 | `__post_init__` | E6 | ``` # writeable caller array mutated in place would leave a stale jit cache key (review H1-5). ``` | ``` # writeable caller array mutated in place would leave a stale jit cache key. ``` |
| 1122 | `ut3_orthogonal_representations` | N17 | "tucker_cores = (U0,U1, U2, U3), up orthogonal" | HELD (N17): "down-orthogonal" |
| 1125 | `ut3_orthogonal_representations` | N17 | "outer_tt_cores = (O0, O1, O2, O3), down orthogonal" | HELD (N17): "down_tt_cores = (O0, O1, O2, O3), up-orthogonal" (the name `outer_tt_cores` is also stale: N4) |

### `t3toolbox/uniform_manifold.py`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 101 | `UT3Tangent` | E6 | ``` stored (the split-agnostic stacking of increment 2c). ``` | ``` stored (split-agnostic stacking). ``` |
| 154 | `align_padded` | N5 | "P-block" | "new block" |
| 261 | `_check_same_tangent_space` | E6 | ``` # first (review H5-5): the tangent space depends only on real content, so frames identical up to ``` | ``` # first: the tangent space depends only on real content, so frames identical up to ``` |
| 555 | `probe` | N6 | "the single-sample Riemannian Jacobian ``𝒥``" | "the single-sample bare Jacobian ``𝒥``" |
| 632 | `probe_derivatives` | N6 | "the forward Riemannian Jacobian derivatives" | "the forward bare-Jacobian derivatives" |
| 634 | `probe_derivatives` | N1 | "sample stack" | "probe stack" |
| 638 | `probe_derivatives` | N1 | "sample stack" | "probe stack" |
| 663 | `probe_derivatives` | N1 | "sample stack" | "probe stack" |
| 819 | `probe_derivatives_transpose` | N1 | "sample stack" | "probe stack" |

### `t3toolbox/uniform_tucker_tensor_train.py`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 55 | `UT3Masks` | E4 | "Slot ``j`` of an edge is real iff its mask is ``True`` there (the prefix/canonical form)." | "Slot ``j`` of an edge is real iff its mask is ``True`` there; canonically a prefix, gappy after ``+``." |
| 67 | `__post_init__` | E6 | ``` # writeable caller array mutated in place would leave a stale jit cache key (review H1-5). ``` | ``` # writeable caller array mutated in place would leave a stale jit cache key. ``` |
| 255 | `__mul__` | E6 | ``` operation -- convert with ``.to_t3()`` and use ragged ``*``; review H3-7.)""" ``` | ``` operation -- convert with ``.to_t3()`` and use ragged ``*``.)""" ``` |
| 267 | `__truediv__` | E6 | ``` """Scale by ``1/s`` (scalar only; the ``__mul__`` guard rejects the rest) -- review R1-12.""" ``` | ``` """Scale by ``1/s`` (scalar only; the ``__mul__`` guard rejects the rest).""" ``` |
| 282 | `allclose` | E1 | ":py:meth:`~t3toolbox.tucker_tensor_train.TuckerTensorTrain.numerically_equal`" | ":py:meth:`~t3toolbox.tucker_tensor_train.TuckerTensorTrain.allclose`" |
| 302 | `corewise_equal` | E1 | "use :py:meth:`numerically_equal`." | "use :py:meth:`allclose`." |
| 446 | `probe_derivatives` | N1 | "sample stack" | "probe stack" |
| 470 | `probe_derivatives` | N1 | "sample stack" | "probe stack" |
| 518 | `entries_derivatives` | N1 | "sample stack" | "probe stack" |
| 716 | `minimal_ranks` | E6 | ``` array would leak tracers into ``aux_data`` -- review R8-9).""" ``` | ``` array would leak tracers into ``aux_data``).""" ``` |
| 864 | `stack` | E6 | ``` with an inhomogeneous-shape error (review R8-6); re-pad first (``from_t3(x, n=, r=)`` / ``` | ``` with an inhomogeneous-shape error; re-pad first (``from_t3(x, n=, r=)`` / ``` |
| 1312 | `from_ut3svd` | E6 | ``` ragged detour (review R10-4). Padding only grows: smaller-than-tight ``n``/``r`` raise. ``` | ``` ragged detour. Padding only grows: smaller-than-tight ``n``/``r`` raise. ``` |
| 1331 | `from_ut3svd` | E4 | "# HOST bool; padded with False (still prefix form)" | "# HOST bool; padded with False (real slots unchanged)" |

### `examples/fit_hilbert_from_apply_derivatives.py`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 22 | module | E7 | "(see ``docs/derivative_order_information_and_conditioning.md``)" | drop the parenthesis (the page does not exist), or point at `docs/fitting_and_optimization.md` §5 |
| 26 | module | N2 | `]_{t=0..K}` | `]_{t=0..order}` |
| 28 | module | N2 | "a vector of ``K+1`` numbers" | "a vector of ``order+1`` numbers" |
| 73 | module | N2 | "# highest derivative order K" | "# highest derivative order" |
| 103 | `unit_pairs_many_p` | N1 | "sample stack" | "probe stack" |
| 119 | `dense_apply_derivatives` | N1 | "sample stack" | "probe stack" |
| 140 | `apply_derivative_operator` | N1 | "sample stack" | "probe stack" |
| 144 | `apply_derivative_operator` | N1 | "sample stack" | "probe stack" |

### `examples/fit_hilbert_from_apply_derivatives_flat.py`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 11 | module | N1 | "sample stack" | "probe stack" |
| 31 | module | N2 | `]_{t=0..K}` | `]_{t=0..order}` |
| 54 | module | N2 | "# highest derivative order K" | "# highest derivative order" |
| 84 | `unit_pairs_many_p` | N1 | "sample stack" | "probe stack" |
| 99 | `flatten_pairs` | N1 | "sample stack" | "probe stack" |
| 112 | `dense_apply_derivatives` | N1 | "sample stack" | "probe stack" |
| 128 | `rms` | N1 | "sample stack" | "probe stack" |
| 134 | `apply_derivative_operator` | N1 | "sample stack" | "probe stack" |
| 138 | `apply_derivative_operator` | N1 | "sample stack" | "probe stack" |
| 254 | `main` | N1 | "sample stack" | "probe stack" |
| 255 | `main` | N1 | "sample stack" | "probe stack" |

### `examples/fit_hilbert_from_apply_derivatives_topt.py`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 16 | module | N1 | "sample stack" | "probe stack" |
| 39 | module | N2 | "# highest derivative order K" | "# highest derivative order" |
| 62 | `unit_pairs_many_p` | N1 | "sample stack" | "probe stack" |
| 75 | `dense_apply_derivatives` | N1 | "sample stack" | "probe stack" |

### `examples/fit_hilbert_from_entries_lbfgs.py`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 49 | module | E18 | "the best rank-``r`` dense T3-SVD" | "the rank-``r`` dense T3-SVD truncation (quasi-optimal)" |

### `examples/fit_hilbert_regularized.py`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 30 | module | M/U | "weighting cores by inverse unfolding singular" | "weighting cores by inverse matricization and unfolding singular" |

### `examples/fit_hilbert_tensor_newton_cg.py`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 243 | `oracle_relerr` | E18 | same | same |

### `examples/fit_hilbert_uniform_newton_cg.py`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 39 | module | N7 | "padded restart" | "warm start (zero-padded)" |
| 104 | `oracle_relerr` | E18 | "Relative error of the best rank-(tucker,tt) approximation" | "Relative error of the quasi-optimal rank-(tucker,tt) T3-SVD truncation" |
| 173 | `main` | N7 | "padded restart" | "warm start (zero-padded)" |

### `examples/fit_hilbert_uniform_probe_derivatives_newton_cg.py`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 23 | module | N2 | `]_{t=0..K}` | `]_{t=0..order}` |
| 25 | module | N2 | "a length-``K+1`` jet" | "a length-``order+1`` jet" |
| 39 | module | N2 | `order=K` | `order=ORDER` (the script's constant) |
| 89 | `dense_probe_derivative_jets` | N1 | "sample stack" | "probe stack" |

### `examples/fit_varied_rank_tensor_newton_cg.py`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 6 | module | M/U | "whose matrix unfoldings all have a similar singular-value decay" | "whose matricizations and unfoldings all have a similar singular-value decay" |
| 8 | module | M/U | "different unfoldings carry very different amounts of" | "different matricizations and unfoldings carry very different amounts of" |
| 12 | module | M/U | "current iterate's unfoldings." | "current iterate's matricizations and unfoldings." |
| 32 | module | M/U | "the T3-SVD singular values of the converged iterate's unfoldings" | "... of the converged iterate's matricizations and unfoldings" |
| 228 | `oracle_relerr` | E18 | same | same |

### `docs/exact_hessian_two_hole.tex`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 148 | § Setup and notation | N1 | "sample stack" | "probe stack" |
| 290 | § The Lagrangian form, and the two levels | M/U | "Eckart--Young for one unfolding" | "Eckart--Young for one matricization or unfolding" |
| 291 | § The Lagrangian form, and the two levels | E6 | ``` the argument but on measurement: $252.85288420$ vs $252.85288182$ (design-plan check F6), the ``` | ``` the argument but on measurement: $252.85288420$ vs $252.85288182$, the ``` |
| 293 | § The Lagrangian form, and the two levels | E6 | ``` retraction's finite-difference ratios $4.0$ and $4.25$ ($O(h^2)$, F14), and $3.7\times10^{-6}$ here (check C8a, ``` | ``` retraction's finite-difference ratios $4.0$ and $4.25$ ($O(h^2)$), and $3.7\times10^{-6}$ here (the remainder of the parenthesis without "check C8a,") ``` |
| 301 | § The lift | E6 | ``` recomputes them by contraction only (\code{fv_center_factors}; I-1: never stored on the frame, never re-SVD'd): ``` | ``` recomputes them by contraction only (\code{fv_center_factors}; never stored on the frame, never re-SVD'd): ``` |
| 335 | § The lift | E6 | ``` an exact one-sided inverse (checks C2a--c, F2/F3). Only a hand-built frame with a wide $T_i$ would give a ``` | ``` an exact one-sided inverse. Only a hand-built frame with a wide $T_i$ would give a ``` |
| 391 | § The lift | E6 | ``` $f_\eps\le1/(2\eps)$ (check C4a), $f_\eps\approx1/\sigma$ for $\sigma\gg\eps$, $\approx\sigma/\eps^2$ below; the ``` | ``` $f_\eps\le1/(2\eps)$, $f_\eps\approx1/\sigma$ for $\sigma\gg\eps$, $\approx\sigma/\eps^2$ below; the ``` |
| 393 | § The lift | N7 | "continuation restart" | "warm start (zero-padded)" |
| 397 | § The lift | E6 | ``` the computation only through the traced leaf \code{LiftFactors.eps}, so a new damping never recompiles (I-22). ``` | ``` the computation only through the traced leaf \code{LiftFactors.eps}, so a new damping never recompiles. ``` |
| 402 | § The lift | E6 | ``` (check C3). The Hessian applies $\Pi$ outside. ``` | ``` The Hessian applies $\Pi$ outside. (drop "(check C3).") ``` |
| 442 | § The Newton system | E6 | ``` (bit-identical to Gauss--Newton's, check C14c) and \code{curvature_precompute}. The per-step overhead is the ``` | ``` (bit-identical to Gauss--Newton's) and \code{curvature_precompute}. The per-step overhead is the ``` |
| 544 | § Two holes: the product rule of the one-hole sweep | E6 | ``` involves neither. The default is \emph{recompute} (I-11): $\xi,\mu$ are reused from the model's sweep (identical at ``` | ``` involves neither. The default is \emph{recompute}: $\xi,\mu$ are reused from the model's sweep (identical at ``` |
| 547 | § Two holes: the product rule of the one-hole sweep | E6 | ``` $T$/$S$ transform of the manifold one, $\nu_c=T\nu$, $\eta_c=S\eta$ --- F16 --- which a deferred ``store'' ``` | ``` $T$/$S$ transform of the manifold one, $\nu_c=T\nu$, $\eta_c=S\eta$, which a deferred ``store'' ``` |
| 558 | § Two holes: the product rule of the one-hole sweep | E6 | ``` $B^\T$ summing the gradient blocks where $\th$'s cores sit. This one-hole-only identity (check C9b, $3.4\times10^{-16}$) ``` | ``` $B^\T$ summing the gradient blocks where $\th$'s cores sit. This one-hole-only identity ($3.4\times10^{-16}$) ``` |
| 569 | § Two-index jets and the recurrence-only rule | E6 | ``` kernels spawns three dotted products with the same order wiring. Layout: \emph{companion arrays} (I-10) --- ``` | ``` kernels spawns three dotted products with the same order wiring. Layout: \emph{companion arrays} --- ``` |
| 587 | § Two-index jets and the recurrence-only rule | E6 | ``` (\code{_inverse_factorials} supplies $1/r!$). \textbf{The recurrence-only rule} (I-33): every production kernel is one ``` | ``` (\code{_inverse_factorials} supplies $1/r!$). \textbf{The recurrence-only rule}: every production kernel is one ``` |
| 601 | § Two-index jets and the recurrence-only rule | E6 | ``` (\code{_adj_tilde_tangent_step}; check C10d): ``` | ``` (\code{_adj_tilde_tangent_step}): ``` |
| 620 | § Two-index jets and the recurrence-only rule | E6 | ``` $d(\mu G)=\dot\mu G+\mu\dot G$, $d(G\nu)=\dot G\nu+G\dot\nu$ (\code{_dxi_tilde_tangent_order_step}; check C10e). ``` | ``` $d(\mu G)=\dot\mu G+\mu\dot G$, $d(G\nu)=\dot G\nu+G\dot\nu$ (\code{_dxi_tilde_tangent_order_step}). ``` |
| 677 | § The assembly, factored by the product rule | E6 | ``` apply / entries (below). (F33 measured the planned six-final factored jet form $1.4$--$2.0\times$ faster on that ``` | ``` apply / entries (below). (A measurement put the planned six-final factored jet form $1.4$--$2.0\times$ faster on that ``` |
| 689 | § The assembly, factored by the product rule | E6 | ``` $\dot E_2=\dot\xi\otimes\nu+\xi\otimes\dot\nu$: four $\mathtt W$-summed finals (check C11c). Any pairing of the three ``` | ``` $\dot E_2=\dot\xi\otimes\nu+\xi\otimes\dot\nu$: four $\mathtt W$-summed finals. Any pairing of the three ``` |
| 690 | § The assembly, factored by the product rule | E6 | ``` groups into two intermediates gives four; slice S3's dense form paired the $\tilde\tau$ and $\ddet$ groups through ``` | ``` groups into two intermediates gives four; an earlier dense form paired the $\tilde\tau$ and $\ddet$ groups through ``` |
| 712 | § The assembly, factored by the product rule | E6 | ``` C11a: equal to the nine-term dense binomial form to $6.3\times10^{-16}$). Apply/entries have one group, so ``` | ``` equal to the nine-term dense binomial form to $6.3\times10^{-16}$). Apply/entries have one group, so ``` |
| 750 | § The assembly, factored by the product rule | E6 | ``` \State undotted: $\nu$ re-swept through $G$; $\ddet=Uc$; $\tilde\tau$, $\tilde\sigma$ \Comment{GN kernels, recompute (I-11)} ``` | ``` \State undotted: $\nu$ re-swept through $G$; $\ddet=Uc$; $\tilde\tau$, $\tilde\sigma$ \Comment{GN kernels, recompute} ``` |
| 771 | § The assembly, factored by the product rule | E6 | ``` order $0$ every jet kernel reduces to the plain kind's (check C9d). ``` | ``` order $0$ every jet kernel reduces to the plain kind's. ``` |
| 846 | § The uniform masking rule | E6 | ``` masks are load-bearing, not hygiene (an unmasked $\thh$ corrupts the real block by $2.6\times10^3$; F15, F18). Every ``` | ``` masks are load-bearing, not hygiene (an unmasked $\thh$ corrupts the real block by $2.6\times10^3$). Every ``` |
| 849 | § The uniform masking rule | E6 | ``` \code{self.masks}, which is a strict superset at TT slack (F23) --- runs the mask-free polymorphic kernel, and ``` | ``` \code{self.masks}, which is a strict superset at TT slack --- runs the mask-free polymorphic kernel, and ``` |
| 880 | § Shared (tied) Tucker factors | N7 | "Padded restarts" | "warm start (zero-padded)" |
| 885 | § Shared (tied) Tucker factors | E6 | ``` \code{project}) of the plain one. The shared retraction's second order is measured (F14), not proven. ``` | ``` \code{project}) of the plain one. The shared retraction's second order is measured, not proven. ``` |
| 911 | § The canonical zero-point frame and the Newton reference | E6 | ``` fixes one (I-29): ``` | ``` fixes one: ``` |
| 933 | § The canonical zero-point frame and the Newton reference | E6 | ``` same real blocks, $\le3.9\times10^{-16}$; the 44 ragged/uniform optimizer pairs agree to $3.5\times10^{-12}$, F37). ``` | ``` same real blocks, $\le3.9\times10^{-16}$; the 44 ragged/uniform optimizer pairs agree to $3.5\times10^{-12}$). ``` |
| 936 | § The canonical zero-point frame and the Newton reference | E6 | ``` (F32; Haar seeds 0--5 gave 9.5--24 there). The reference is a \emph{convention} --- a deterministic data-gradient ``` | ``` (Haar seeds 0--5 gave 9.5--24 there). The reference is a \emph{convention} --- a deterministic data-gradient ``` |
| 950 | § The trust region | E6 | ``` \paragraph{One CG body for both step controls} (\code{_cg_body}, I-5). The state carries the radius and a ``` | ``` \paragraph{One CG body for both step controls} (\code{_cg_body}). The state carries the radius and a ``` |
| 995 | § The trust region | E6 | ``` Gauss--Newton Cauchy length whatever the Hessian. Measured (seeded probe-derivative restart, F19): Gauss--Newton ``` | ``` Gauss--Newton Cauchy length whatever the Hessian. Measured (seeded probe-derivative restart): Gauss--Newton ``` |
| 997 | § The trust region | E6 | ``` 0 rejected; the q-quadratic tail $e_{k+1}\le0.92\,e_k^2$ at power 1.0 (F22). ``` | ``` 0 rejected; the q-quadratic tail $e_{k+1}\le0.92\,e_k^2$ at power 1.0. ``` |
| 1009 | § Verification | E6 | the ID-keyed table, lines 1009-1030 | Drop the check-ID column of the verification table (`l` column and the C1--C14 cells; the table becomes two columns). The in-text "check Cxx" citations are dropped by the rows below |
| 1034 | § Verification | E6 | ``` At C8 the point was far from a minimizer: the exact quadratic form was $-5449.96$ against the Gauss--Newton $30.33$, ``` | ``` At the second-order check the point was far from a minimizer: the exact quadratic form was $-5449.96$ against the Gauss--Newton $30.33$, ``` |
| 1040 | § Verification | E6 | ``` (\code{tv_gauge_residual(frame, variations, scale)}, I-38). Measurements: ``` | ``` (\code{tv_gauge_residual(frame, variations, scale)}). Measurements: ``` |

### `docs/pad_safe_svd.tex`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 29 | (top) | E6 | ``` \code{dev/review\_2026-08-22/repros/S1b/})} ``` | ``` drop the dev-path citation (\code{dev/review\_2026-08-22/repros/S1b/}): shipped text cites no `dev/` paths ``` |
| 77 | § Setting | M/U | "of a zero-padded unfolding, and the row index of an unfolding is often" | "of a zero-padded matricization or unfolding, and its row index is often" |
| 90 | § The failure mode | M/U | "the rows of an unfolding are" | "the rows of a matricization or unfolding are" |
| 106 | § The failure mode | N7 | "padded restart" | "warm start (zero-padded)" |
| 113 | § Prior art | E6 | ``` \code{README}, under \code{repros/S1b/}). Nobody relies on the SVD's null-space completion to grow ``` | ``` drop "under \code{repros/S1b/}" (a dev path) ``` |
| 126 | § Prior art | N7 | "padded restart" | "warm start (zero-padded)" |
| 306 | § Integration into the T3 sweeps | M/U | "The row mask of a mid-sweep unfolding" | "The row mask of a mid-sweep matricization or unfolding" |
| 336 | § Alternatives: designed, measured, rejected | E6 | ``` \code{dev/review\_2026-08-22/repros/S1b/}.) ``` | ``` drop the dev-path citation (\code{dev/review\_2026-08-22/repros/S1b/}) ``` |

### `docs/padded_frame_cone.tex`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 96 | § Frames of a T3, and padded frames | M/U | "the tensor unfolded with $x$'s side as rows" | "the tensor reshaped with $x$'s side as rows (a matricization at a Tucker edge, an unfolding at a TT edge)" |
| 120 | § Frames of a T3, and padded frames | N5 | "A-block" | "old block" |
| 121 | § Frames of a T3, and padded frames | N5 | "A-block" | "old block" |
| 121 | § Frames of a T3, and padded frames | N5 | "P-block" | "new block" |
| 129 | § Frames of a T3, and padded frames | N5 | "A-block" | "old block" |
| 275 | § One step on cores | N5 | "A-block" | "old block" |
| 302 | § One step on cores | N5 | "A-block" | "old block" |
| 330 | § One step on cores | N5 | "A-block" | "old block" |
| 330 | § One step on cores | N5 | "P-block" | "new block" |
| 340 | § The start | M/U | "with the ragged unfolding's row length" | "with the row length of the ragged matricization or unfolding" |
| 341 | § The start | N5 | "A-block" | "old block" |
| 358 | § The start | N5 | "A-block" | "old block" |

### `docs/shared_t3_math.tex`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 615 | § Rank continuation: edge conditioning under sharing | N7 | "padded restarts" | "warm start (zero-padded)" |

### `docs/symmetric_probe_derivatives.tex`

| line | where | item | current | replacement |
|---|---|---|---|---|
| 331 | § The unifying view and the implementation | N1 | "sample stack" | "probe stack" |
| 348 | § The unifying view and the implementation | N1 | "sample stack" | "probe stack" |
| 547 | § Extensions (implemented in the code) | N1 | "sample stack" | "probe stack" |
| 634 | § Recurrence and convolution realization | N1 | "sample stack" | "probe stack" |

## Defined only in the glossary

These terms are used in the docs or docstrings but defined nowhere else; the glossary defines them from the code and
the math notes and links to the nearest page that uses them. The owning page should get its own sentence.

- **sweep** (as a bare term; also used for test parameter grids) -- owner: contributor/scan_body_principles.md or
  entries_apply_probe.md §8.
- **zipper** (`tt_zipper_left_to_right`) -- used in contributor/sharing_internals.md, defined only in the docstring.
- **level**, **ladder** (rank continuation) -- owner: rank_continuation.md.
- **warm start** (as a term) -- owner: rank_continuation.md "How to use it".
- **base point** -- owner: frame_variations.md.
- **pinned completion** (`linalg.pinned_completion`, `pad_safe_pinned_completion`) -- CHANGELOG and docstrings
  only; contributor/padded_frame_internals.md explains why a QR but not the term. Owner: that page.
- **Armijo** (the condition `f(retract(αp)) ≤ f + c_armijo·α·⟨g, p⟩`; 40 halvings in `newton_cg`, 50 in
  `gradient_descent`) -- owner: fitting_and_optimization.md §5.
- **T4S** -- owner: index.rst or user_guide.rst (expand the abbreviation once).
- **cone** (why the cone reference and cone merit are so named) and **tangent cone** -- from padded_frame_cone.tex
  §5.2 ("the pieces `T_A(F)` together fill the tangent cone at `A`") and the maintainer's T3Polynomial note
  `dev/zero_reference_cone_2026_11_02.md` (the cone reference is named for the gradient's projection onto the tangent
  cone at zero). Owner: aligned_padded_frame.md.
- **seam** (ragged/uniform seam, split `precompute` → `*_from_sweep` seam, optimizer seam) -- owner:
  naming_conventions.md or contributor/padded_frame_internals.md.
- **twin** (uniform / backend / frontend / `_trs` / dotted twin) -- owner: naming_conventions.md.
- **Hilbert–Schmidt** -- owner: user_guide.rst (one sentence in "Tucker tensor trains represent dense tensors").
- **stratum** -- owner: uniform_ranks_and_varieties.md.
- **backend razor** -- defined only in the repo `CLAUDE.md` (not a Sphinx page); contributor/fitting_internals.md
  has a one-line gloss. Owner: contributor/fitting_internals.md.
- **carrier** -- one use in aligned_padded_frame.md, defined only in padded_frame_cone.tex Table 1. Owner:
  frame_variations.md (replace `Ld`/`R0`).
- **oblique projection**, **ambient projection**, **vector transport** -- listed in numerical_contracts.md, defined
  only in docstrings. Owner: frame_variations.md or fitting_and_optimization.md §2.4.
- **hint** -- defined in aligned_padded_frame.md; in the backend only as a signature comment (fine).
- **pushthrough / combine** -- defined only in `backend/sampling_derivatives.py` comments and the math note.
- **multilinear extension / grid corner** -- `entries_derivatives` docstrings only.
- **padded widths** ("nominal rank", "the level's widths", "padded `N`") -- owner: uniform_supercore_layout.md.
- **gauge-shifted masks** (variation masks) -- owner: uniform_masks_vs_ranks.md.

## Left out: not T3Toolbox vocabulary

- `forward_summed_reverse`, "val score", "adaptive ladder": T3Polynomial (downstream) terms; none occurs in the
  library (the only near hit is the example function `adaptive_continuation`).
- "points × directions" and "Darcy": downstream vocabulary that appears in aligned_padded_frame.md's measurements
  section and the CHANGELOG; the glossary does not define them. The follow-up pass may want a one-line gloss there
  ("the downstream T3Polynomial project's Darcy problem").
- Review and plan codes (`D-42`, `I-33`, `J-12`, `S7b`, `R1-13`, `F37`, "Phase D", ...): excluded by scope; they
  appear in shipped docstrings and contributor pages without a key.

