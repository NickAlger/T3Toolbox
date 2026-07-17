# RESUME: extract the lean-jet contractions into contractions.py (no-context handoff)

> **SUPERSEDED 2026-07-17 — the task this note describes was completed by a different mechanism.**
> Nick replaced the named-contraction architecture with the grouped-einsum interpreter
> (`contractions.contract('WCa,Caib,WCi->WCb', ...)`); the inline einsums inventoried below became
> `contract` call sites (slice 3), not named functions, and the hand-written-oracle instructions no
> longer apply. Kept for the einsum inventory and the K-trap analysis (now mechanical — the
> interpreter's rank solve reproduces it). See `dev/HANDOFF.md` "grouped-einsum interpreter".

_Written 2026-07-16 to resume a mid-flight cleanup in a fresh session with no memory of it. Read this
top-to-bottom, then read the referenced code. The broader project guide is `CLAUDE.md` (repo root)._

## The task (Nick's words, paraphrased)

> The standard pattern for grouped contractions is to pass arrays to named contractions in
> `contractions.py`, which handles axis grouping/ungrouping, numpy-vs-jax dispatch, and the greedy
> pairwise ordering for numpy einsums. During the recent lean-jet work in `sampling_derivatives.py` we
> wrote contractions inline (sometimes in private `_*` helpers, sometimes as raw einsums in the
> functions). Now that they work, **move that contraction + index-grouping logic into named
> contractions in `contractions.py`, mirroring how `probing.py` does it, and add tests for the new
> contraction functions.**

Behavior-preserving refactor. The lean jet functions are the *standard* (call-site-wired) probe
derivative implementations; each has a dense `*_trs` twin kept as reference (do NOT touch the `*_trs`
ones — they already delegate to named `trs_*` contractions).

## Status

- **Phase 1 (mu) — DONE, committed `6e9a55d8` (local, NOT pushed).** The template. Three private mu
  helpers moved into `contractions.py` as `Caib_sWCi_to_sWCab`, `tWCa_WCab_to_tWCb`,
  `stWCa_sWCab_to_tWCb`; `compute_mu_jets` calls them; helpers deleted; oracle tests in
  `tests/test_contractions_lean_jets.py`. All gates green. **Read this commit's diff first — it is the
  exact pattern to copy.**
- **Remaining:** `compute_eta_jets`, `compute_sigma_jets`/`compute_tau_jets` (via `_sigma_banded_step`),
  `compute_deta_jets`, `compute_{tau,sigma}_tilde_jets` (via `_adj_tilde_step`). See the inventory below.

## THE CONVENTION (read `contractions.py` lines ~30–91 — the SHARDING block — in full)

Every named contraction: `use_jax = tree_contains_jax((...))`; `xnp, _, _ = get_backend(True, use_jax)`;
extract block shapes; reshape; `_grouped_einsum(xnp, use_jax, '<subscript>', *ops)`; reshape back. Add
the name to `__all__`. Docstring first line is exactly `"""Computes named contraction. Capital letters
indicate grouped indices, which may be empty.` then a blank line + one sentence.

The one rule that governs every subscript (the shardability contract):

- **SHARED block** — appears on ≥2 operands and must be matched (e.g. the core stack `C`, which the core
  operand `Caib` pins via `Caib.shape[:-3]`): give it **its own letter** and flatten its axes to one
  (`size_C = math.prod(C_shape)`). Only its leading axis stays shardable — accepted.
- **PASSIVE block** — lives on **one** operand and rides to the output untouched (e.g. `W`, and `K` when
  it's on only one operand): **do NOT flatten it — let it ride in `'...'`.** `einsum`'s `...` spells a
  variable number of axes natively, keeping each shardable at zero cost.
- **NEVER give two named blocks the same letter, and never flatten one block into another's letter.**
  That is a FUSION and `tests/test_contractions_sharding.py` (compiles under 4 virtual devices, counts
  collectives) will FAIL on it.

**The K trap (why the remaining phases are not mechanical).** `mu` had only W (passive) + C (shared),
so it was easy. The tangent phases carry `K`, and some einsums put **W and K on DIFFERENT operands** —
e.g. deta's `einsum('WCa,KCaib->WKCib', mu_r, dG)`: `W` is on `mu_r`, `K` is on `dG`. They cannot share
one `...` (different operands, different sizes), so each needs its **own letter** (grouped-block form),
and if no operand pins `len(C)` you must pass `n_frame` (`len(C)`). **Mirror the EXISTING K-stacked
contractions** — e.g. `trs_rWCa_KCaib_sWCb_to_tWKCi`, `tWCi_KCio_to_tWKCo` (takes `n_frame`) in
`contractions.py`; grep `n_frame` and `WKC` there. When W and K DO ride the same operand contiguously
(e.g. `sigma_tilde` is `(order,W,K,C,a)`), they can share one `...`.

## Inventory of remaining inline einsums (current line numbers in sampling_derivatives.py)

The lean functions currently **flatten W/K/C to `sW,sK,sC` at the top of the scan step, then use letters**
(the pre-sharding style). The extraction must (a) remove those flatten-reshapes, (b) pass the
*unflattened* arrays to the new named contractions, (c) each contraction re-derives shared-vs-passive
per the rule above. Names below are suggestions (follow the `inputs_to_output` grammar).

**eta — `compute_eta_jets` `_accumulate` (lines 503–505). W+C, NO K (like mu; easy).**
- `503 '...Ca,Caib->...Cib'` (mu_r·G, contract a): `WCa_Caib_to_WCib` — C shared, W `...`.
- `504 '...Cib,s...Cb->s...Ci'` (·nu, contract b): `WCib_sWCb_to_sWCi` — C shared, W `...`, s passive on nu.
- `505 'ts,s...Ci->t...Ci'` (trs weight, contract s): `ts_sWCi_to_tWCi` — here **C is passive** (only on
  the MGN operand, not on trs), so C rides in `...` with W: subscript `'ts,s...i->t...i'`.

**deta — `compute_deta_jets` `_step` (lines 862–868). Carries K. THE K-TRAP LIVES HERE.**
- `862 'WKCa,Caib->WKCib'` (sigma·Q): `WKCa_Caib_to_WKCib` — C shared; W+K ride ONE `...` (both on sig_r).
- `863 'WKCib,sWCb->sWKCi'`: `WKCib_sWCb_to_sWKCi` — C shared; **W+K on mg1, W on nu** → W shared between
  operands (letter), K passive on mg1. Careful: check which blocks are shared vs passive.
- `864 'WCa,KCaib->WKCib'` (mu·dG): **W on mu_r, K on dG** → both need letters + `n_frame` (no C-pinning
  operand that also excludes K). `WCa_KCaib_to_WKCib` (n_frame). ← the hard one.
- `865 'WKCib,sWCb->sWKCi'`: as 863.
- `866 'WCa,Caib->WCib'` (mu·P): `WCa_Caib_to_WCib` — SAME as eta 503 (reuse).
- `867 'WCib,sWKCb->sWKCi'`: `WCib_sWKCb_to_sWKCi`.
- `868 'ts,sWKCi->tWKCi'` (trs weight): `ts_sWKCi_to_tWKCi` — W,K,C passive → `'ts,s...i->t...i'`.

**sigma/tau banded — `_sigma_banded_step` (lines 902–918). Carries K.**
- `902 'Caib,sWCi->sWCab'`: `Caib_sWCi_to_sWCab` — **already exists (mu phase)**, reuse.
- `903 'KCaib,sWCi->sWKCab'` (dG·xi): `KCaib_sWCi_to_sWKCab` — K on dG (letter), C shared, sW passive.
- `904 'Caib,sWKCi->sWKCab'` (O·dxi): `Caib_sWKCi_to_sWKCab` — K on dxi (rides `...`? it's on one operand → passive), C shared.
- `912 'stWKCa,sWCab->tWKCb'`, `913/914 'stWCa,sWKCab->tWKCb'`: fused two-term (s+a contracted). K on one
  operand each. `stWKCa_sWCab_to_tWKCb`, `stWCa_sWKCab_to_tWKCb`.
- `916 'tWKCa,WCab->tWKCb'`, `917/918 'tWCa,WKCab->tWKCb'`: the order-0 branch versions.

**tilde — `_adj_tilde_step` (lines 1426–1440). Carries K.**
- `1426 'Caib,kWCi->kWCab'`: `Caib_sWCi_to_sWCab` — **already exists (mu phase)**, reuse (k plays the s role).
- `1431 'ksWKCa,kWCab->sWKCb'`, `1433 'sWKCa,WCab->sWKCb'`: prop (two-term reverse recurrence).
- `1438 'WCa,Caib->WCib'`: `WCa_Caib_to_WCib` — reuse (eta 503).
- `1439 'WCib,tWKCi->tWKCb'`: `WCib_tWKCi_to_tWKCb`.
- `1440 'ts,tWKCb->sWKCb'`: `ts_tWKCb_to_sWKCb` — trs weight, W/K/C passive → `'ts,t...b->s...b'`.

Reused across phases: `Caib_sWCi_to_sWCab` (done), `WCa_Caib_to_WCib`, the `ts_*` trs-weight ones.

## Gates (run after each phase; all must stay green)

```
PYTHONPATH=$PWD /home/nick/miniconda3/envs/t3toolbox/bin/python -m pytest \
  tests/test_contractions_lean_jets.py tests/test_jet_recurrence.py \
  tests/test_contractions_sharding.py -q
```
- `test_jet_recurrence` — behavior-preserving (lean == trs to ~1e-12). MUST stay green.
- `test_contractions_sharding` — auto-covers every `__all__` entry; FAILS if you flatten a block that
  should ride `...` (the K-trap detector). MUST stay green.
- `tests/test_contractions_lean_jets.py` — ADD an oracle test per new contraction (full-explicit
  `np.einsum` over an empty/single/multi-axis W,K,C shape matrix; mirror `test_contractions_unfused.py`).
- Before finishing: the **full suite** (`pytest tests/ -q`, ~8 min) + the docs `-W` build.

Also `tests/test_contractions_naming.py` statically checks the name↔subscript rule.

## Env / workflow

- Python: `/home/nick/miniconda3/envs/t3toolbox/bin/python`; always `PYTHONPATH=$PWD`. (Env `t3toolbox`,
  py3.11, numpy 2.x / jax 0.10; ~3.5× slower than the old env. `tttt` is the older env.)
- Commit co-author line: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- Work on `main`, direct commits, one reviewable commit per phase. **`6e9a55d8` is unpushed** — ask Nick
  before pushing (he pushes after review).
- Doctests are CI-enforced; run scripts/tests with the env python.

## Context: everything else done this session (all pushed unless noted)

1. `b54f93d2` **jet rename** — the lean recurrence/scan jets took the canonical names; dense forms became
   the `*_trs` reference twins. (This is why the functions you're editing are named `compute_mu_jets`
   etc. and their references are `compute_mu_jets_trs`.)
2. `d84054a1` / `83dd4c70` / `1412ca36` **chunk_size (slices A/B/C)** — a memory knob for the probe-
   transpose 𝒥ᵀ assembly: threaded through the transpose chain (default 100, `None`=dense); an eager,
   measured, memory-balanced estimator `estimate_chunk_size` / `max_chunk_size_within`; and `'auto'`
   wired through the 4 optimizers + fitting kinds. User doc `docs/chunking.md`.
3. **Sharded fitting (slice D) — POSTPONED** by Nick; full design + open decisions in
   `dev/sharded_fitting_plan.md`.
4. `6e9a55d8` **mu contraction extraction** — phase 1 of THIS task.

Living status: `dev/HANDOFF.md`. The batching/stacking mental model (essential for the K-trap):
`docs/batching_and_stacking.md`. The contractions naming grammar: `docs/naming_conventions.md`.
