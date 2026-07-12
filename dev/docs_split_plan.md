# Docs user/dev split plan — audience tiers, archive promotions, stale-content sweep

_Drafted 2026-07-12; approved by Nick the same day. **S1 (`094762c8`), S2 (`c430bc55`), S3
(committed 2026-07-12 late) EXECUTED — S4 (new content: P0 t3m doc, P9–P11 contributor notes)
and S5 (routing-rule CLAUDE.md update, final sweep, archive this plan) REMAIN.** The S3
rewrite-splits were executed by four parallel forks; V1 settled (use_jit is built+tested)._
_Inputs: the 23-doc section-level audience classification (4 readers) + the 27-file `dev/archive`
durability sweep (4 readers), both 2026-07-11/12. Precedent: `dev/archive/docs_pass_plan.md`._
_Standing instruction (Nick): on any point of confusion, stop and ask — don't guess._

## Settled decisions (Nick, 2026-07-11/12)

1. **Two rendered tiers**: user-facing design docs AND a **Contributor guide** — both on the public
   site, clearly separated. `/dev` is reserved for **ephemeral** working state (plans, handoffs,
   archive) — durable knowledge always lands in `docs/`, tagged by audience.
2. **Nick agreed to the classification + proposal as presented**, including: user docs stay flat in
   `docs/` keeping their current filenames (minimizes inbound-link breakage); dev notes move to a
   subdirectory (**`docs/contributor/`**); split-off dev halves named **`*_internals.md`**; site nav
   becomes *Getting started · User guide · Design notes · Contributor guide · API reference*.
3. **Archive promotions**: Nick asked for the `dev/archive` durability sweep before this plan; its
   verified finds (below) are folded in as first-class work items.
4. **Minimal-rank stale claim**: two independent sources (the catalog's settled verdict; a
   corroborating line in `dev/archive/uniform_fix_plan.md`) say minimal rank is NOT a precondition
   for `inner`/`norm` HS-faithfulness; the stale "+ minimal" claims in `t3svd_minimal_ranks.md` and
   CLAUDE.md's audit bullet get fixed. **Executor must re-verify against
   `docs/numerical_contract_catalog.md`'s verdict table before editing** (it is a math contract).

## A. The 23-doc disposition table

**Tier key:** U = user tier (`docs/`, flat) · C = contributor guide (`docs/contributor/`) ·
A = `dev/archive`.

| # | doc | verdict | disposition |
|---|---|---|---|
| 1 | transposes | USER ~97% | U as-is |
| 2 | rank_continuation | USER ~100% | U as-is (the user-tier tone exemplar) |
| 3 | entries_apply_probe | USER ~90% | U; excise ~6 dev lines (history parenthetical, verification-numbers table → C, 2 archive pointers); absorb the probing paper↔code tables (item 16) |
| 4 | t3svd_minimal_ranks | USER ~95% | U; **fix the stale minimal-rank-precondition sentence** (decision 4) |
| 5 | naming_conventions | USER ~80% | U; extract 4 maintainer imperatives → C conventions note; **add the frame-vs-basis paragraph and the intended-asymmetries registry** (promotions P5, P6) |
| 6 | uniform_ranks_and_varieties | USER ~85% | U; excise future-precondition directive + dated footnote → C; **add the why-no-`to_vector` paragraph** (P8) |
| 7 | uniform_backend_jit_recipe | USER ~75% | U; last section (executed optimizer-design rule) → C |
| 8 | uniform_supercore_layout | MIXED ~70% | U keeps layout/why-d-leads/offset-caution; polymorphic-straddle + rejected-alternative paragraphs → C |
| 9 | testing_strategy | DEV ~98% | C wholesale |
| 10 | doctest_style | DEV ~95% | C wholesale; **fix stale "not yet wired into CI"** (false since R5) |
| 11 | signature_style | DEV ~70% | C wholesale + NEW short U page "How to read T3Toolbox signatures" (principle reader-side + micro-grammar decode table incl. `HOST bool, static` + pointer to the C/W/K legend) |
| 12 | t3svd_design_rationale | DEV ~85% | C wholesale; verify the uniform lossy-misuse warning surfaces user-side (it is in the `rank_adjustment_sweep` docstring — confirmed; add one line to t3svd_minimal_ranks if wanted) |
| 13 | t3svd_verification | DEV ~75% | C wholesale (user-facing bounds already in the t3svd docstring) |
| 14 | ambient_derivative_transpose_note | DEV ~95% | C wholesale (durable deferral record) |
| 15 | uniform_svd_prefix_orthogonalization | DEV ~75% | C wholesale; surface its 2 user nuggets (deterministic masks ⇒ compile-once; arbitrary orthonormal completion at zero-σ slots) as sentences in U uniform material |
| 16 | probing_section6_notes | working note, half-obsolete | salvage the paper↔code algorithm tables → into entries_apply_probe (§7 area); the rest → A as a dated superseded note |
| 17 | batching_and_stacking | MIXED 60/40, sentence-braided | **rewrite-split**: U chapter (Start-here, legend, de-historied glossary, §1 with uniform status FIXED, §2, §3 rule+condensed why, §4 decision rule + name-reading, §5 contract, §6, §7 consequences, §8, user-§9, §10, §11) + `contributor/batching_and_stacking_internals.md` (extension rules incl. delegate-don't-reuse + test policy, rename archaeology, reversed-vmap-plan history, RNG/test guidance, blast-radius note) |
| 18 | fitting_and_optimization | MIXED 55/45 | **rewrite-split**: U doc (intro sans plan-links, §1–§3, user halves of §4.2–4.6 as "properties you can rely on", examples list, the newton_cg-vs-SGD practical advice) + `contributor/fitting_internals.md` (§4.1/4.7/4.8, §5 deferred list, plan/research pointers). **Absorbs promotions P1–P3.** Fix: personal-memory citation OUT; `use_jit` tense (see V1); dangling `mcsgd_apply_derivatives.md` → generic research-repo phrasing (the file was deliberately moved there — verified, commit `055c5744`) |
| 19 | numerical_contract_catalog | MIXED 40/60 | **extract** NEW U page "Numerical contracts: what safe mode checks" (classification rule + master tables + minimal-ranks verdict, present tense, migration arrows deleted); the full APPROVED record → C **unchanged** (decision record; notes are never lost) |
| 20 | uniform_masks_vs_ranks | MIXED 50/50, braided | **rewrite-extraction**: U "how masks behave" treatment (mask/projector reading, gappy-under-⊕⊗, canonical-vs-working form, tangent mask algebra + same-mask precondition, host-numpy fact, no-rank-growth line) ; decision ledgers + rejected-representations table → C |
| 21 | uniform_rank_masks_rationale | MIXED 65/35, braided | light rewrite: U paragraph (why gradients can't grow rank; compile behavior at rank changes) into the U uniform material; the doc, near-as-is → C (the guard against dropping masks) |
| 22 | uniform_equivalence_contract | MIXED 55/45 | section split (light seam rewrite): U keeps the contract + variety paragraph + packedness mirror + rtol/atol narrowing (**+ the finite-fill clause, P7**); the two test-strategy "Consequences" + mid-refactor note + plan link → C |
| 23 | uniform_pytree_composition | MIXED 35/65 | clean extraction: U gets the jit-how-to subsection + `.data` layout + 2-line perf story; the rest → C with its 3 stale-residue spots fixed |

## B. Archive promotions (verified-absent by the sweep; re-verify while writing)

**New user-tier doc:**
- **P0 — `docs/t3m_methods.md`** ("Elementwise multiplication: the three t3m methods"), distilled
  from `dev/archive/t3m_plan.md` (+2 nuggets from `t3m_swap_plan.md`): the two truncatable rank
  families (Khatri–Rao Tucker / Kronecker TT); the three-method cost table + the d≳r vs r≫d
  crossover guidance; the joint-truncation quality guarantee; the truncation-spec semantics +
  short-circuit; the oversample semantics rule; the **`oversample=1` + `rtol` ⇒ ~`d·rtol`
  accumulation caveat** (also one line in the `t3m` docstring). Theory reference: link
  `ttm_t3m_ht_note.tex` on GitHub. Every claim re-verified against current code before asserting.

**Into the fitting rewrite-split (item 18):**
- **P1** — the corewise-vs-autodiff finding (hand-rolled jit corewise gradients outperform
  `jax.grad`, Nick's benchmarks; framed regime-dependent per the house perf philosophy).
- **P2** — the exact-GN insight (sampling forwards are linear in the ambient tensor ⇒ the GN model
  is the exact restriction, dropped term identically zero; currently only in test comments).
- **P3** — the L-BFGS story (deliberately not a library optimizer; scipy-bridge is the intended
  pattern; Riemannian L-BFGS deferred → also listed in the C deferred note). Plus two thin adds if
  absent at rewrite time: the matched-pair gauge-singularity reason; the zero-start saddle reason.

**Into naming_conventions (item 5):**
- **P5** — the frame-vs-basis rationale paragraph ("basis" implies minimality; the representation
  is deliberately overcomplete; "frame" is the manifold-optimization term).
- **P6** — the intended-asymmetries registry (from naming_pass_plan §A3, merged with
  method_porting_plan's not-ported reasons): UT3Tangent without `save`/`to_vector`; the
  ragged-vs-uniform signature gaps (`sum_stack(axis)`, `to_dense(squash_tails=)`,
  `t3svd(rtol/atol)`, `project_ambient(method=)`, …) — each marked deliberate.

**Into uniform user material:**
- **P7** — the finite-fill clause (padding must be FINITE: `0 × NaN = NaN` poisons masked
  reductions; zeros are the robust fill) → equivalence-contract U section + a `pack_vectors`
  docstring line.
- **P8** — the why-no-`to_vector` paragraph (varying-rank stacks have no flat-vector signature;
  jax works on the pytree, scipy interop routes through ragged) → `uniform_ranks_and_varieties`.

**New contributor-guide notes:**
- **P9 — `contributor/uniform_polymorphism.md`** (or a section of the uniform internals note): the
  triage lenses (return-type lens: read-outs are polymorphic via mask-once, T3-like results get
  `ut3_` twins; scan-vs-map lens) + the mask-once mechanism + the orthogonalize-inside-read-out
  exactness argument (`Q·(R·C) = (QR)·C`; garbage provably contracts to zero; exactness requires
  no truncation). From `uniform_fix_plan.md`.
- **P10 — `contributor/refactoring_methodology.md`**: the rename/refactor lessons (inventory both
  import forms; `getattr` string refs invisible to token renames; substring module names need
  lookbehinds; frontend-method collisions need scoped renames; never pipe the gate; never edit
  during a gate; delegation rules). Rescued from the now-ephemeral HANDOFF + naming_pass_plan §D +
  doctest_handoff.
- **P11 — deferred/rejected ledger** (a short C note or a section of fitting_internals): no
  truncation kwargs on dense tangent projection (contraction dominates; the `'t3svd'` method is
  the cross-check oracle); the bidiagonal-`trs` unexploited `O(K)` optimization (+ the measured
  32×/78× derivative-op cost ratios); separate `*_derivatives` methods, not kwargs (return-rank
  polymorphism fights the shape-comment contract); no global/ambient jit toggle (rejected).

**HANDOFF backlog additions (ephemeral tier — correct home):**
- The deferred default-path-doctest pass for undocumented public functions ("Nick wants this").
- The `core_shapes` (property, strips stack) vs `get_core_shapes` (static, includes stack)
  inconsistency — **re-verified still live**; a code decision for Nick, not a docs item.
- `dev/paper_scope.md` notes: the two-spaces geometry picture (M vs P, submersion) and the
  apply/entries sweep-level scatter derivations are paper-grade source material.

## C. Stale/incorrect content — fix inventory (in whichever slice touches the file)

1. `batching_and_stacking.md`: uniform layer called "deferred/broken" ×2 (+ glossary row) — shipped long ago.
2. `doctest_style.md`: "not yet wired into CI" — false since R5.
3. `fitting_and_optimization.md`: `use_jit` described in present tense (see V1); personal-memory
   citation (§4.7); dangling `mcsgd_apply_derivatives.md` (→ research-repo phrasing).
4. `t3svd_minimal_ranks.md`: the stale minimal-rank HS-faithfulness sentence (decision 4).
5. CLAUDE.md: the minimal-rank audit bullet carries the same stale "+ minimal" (decision 4).
6. `uniform_pytree_composition.md`: "current broken UT3 code" / "upcoming UT3VariationsMasks" /
   "implemented later" scope note — all long done.
7. `uniform_backend_jit_recipe.md`: "3b and beyond" slice residue.
8. `uniform_ranks_and_varieties.md`: "verified through the UT3Tangent skeleton" residue.
9. `uniform_equivalence_contract.md`: "mid-refactor in the copied-in code" + "running plan" link.
10. `entries_apply_probe.md`: "probe has no ambient transpose was true only before…" history line.

**V1 (verify, then fix):** `backend/optimizers.py` docstring says the `use_jit` machinery is
"(deferred)" while the code contains ~15 `use_jit` uses and the U-plan records it built+tested.
Determine which statement is true of the current code; fix the docstring AND
`fitting_and_optimization.md` §4.5 to match reality.

## D. Execution slices (each gated: `sphinx -W` zero-warning build; suite when code/docstrings touched)

- **S1 — re-shelving + navigation.** Create `docs/contributor/`; move items 9–15 wholesale;
  `design_notes.rst` → user-tier listing; new `contributor_guide.rst`; salvage+archive item 16;
  the easy excisions on items 1–8 (each excised fragment LANDS in its C destination — nothing
  deleted); full inbound-link sweep (CLAUDE.md, docstrings, cross-docs, HANDOFF — grep-inventoried
  before moving, per the rename methodology). Stale fixes 2, 7, 8 ride along.
- **S2 — clean extractions + naming promotions.** Item 11's reader page; item 5's imperative
  extraction + P5 + P6; items 8, 22, 23 section splits (+ P7); item 6 excisions + P8; item 15
  nuggets. Stale fixes 6, 9, 10 ride along.
- **S3 — the hard rewrite-splits.** Item 17 (batching; stale fix 1); item 18 (fitting; P1–P3;
  stale fix 3 + V1); items 20+21 (masks); item 19 (the safe-mode user page). This is the big
  slice; may commit per-file.
- **S4 — new content.** P0 (`t3m_methods.md`, claims re-verified against code; + the docstring
  caveat line — code-touching, suite-gated); P9; P10; P11.
- **S5 — wrap-up.** Stale fixes 4, 5 (decision 4, re-verified); HANDOFF backlog additions +
  refresh; CLAUDE.md routing-rule update (durable → `docs/` split by audience; `/dev` ephemeral
  only) + repoint doc references; archive this plan. Also: `dev/archive/conf_OLD.py` and the
  stray planning PDF flagged as cleanup candidates (Nick's call).

## E. Acceptance criteria

- Every rendered page is unambiguously one tier; the site nav shows the five sections.
- All 23 original docs dispositioned per the table; excised content lands, never vanishes;
  the full catalog/decision records preserved under `contributor/`.
- All archive promotions P0–P11 landed; the archive files themselves stay in `dev/archive` untouched.
- The stale-fix inventory (C.1–C.10 + V1) cleared; the two CLAUDE.md/t3svd minimal-rank fixes made
  after re-verification against the catalog.
- Zero-warning `-W` build after every slice; suite green after any code/docstring touch; no
  dangling inbound links (grep-verified against the pre-move inventory).
