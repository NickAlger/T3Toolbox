# Batching and stacking — contributor internals

Extension rules, decision history, and test guidance excised from the user-facing
[`../batching_and_stacking.md`](../batching_and_stacking.md) — read that first for the conventions
themselves. This is the material you need when **changing or extending** the batching machinery.

## The interpreter era (2026-07-17): the string is the spec, and the machine honours it

The named-contraction toolkit this note used to govern — ~104 enumerated
`WCa_Caib_WCi_to_WCb`-style functions, each a hand-written flatten + fixed-subscript einsum — was
**replaced by the grouped-einsum interpreter** `contractions.contract('WCa,Caib,WCi->WCb', ...)`.
The subscripts string plays exactly the role the function *name* used to (the batch-group type
signature at the call site), with one decisive difference: **the machine now derives the behavior
from the string**, instead of a hand-written body that could drift from its name. Group sizes are
solved from the operand ndims (identifiability is a rank condition on the subscripts — a call site
either always needs a `len_<G>=` supplement or never does); groups expand into fresh single-axis
letters; one einsum runs on the operands as given.

**Adding a grouped contraction is now: write the string at the call site.** No new function, no
per-function oracle. Two obligations remain, both enforced by tests:

- Record the new string's expected supplement in the frozen `HISTORICAL` table
  (`tests/test_contractions_interpreter.py`) — decide it **by hand first**, then let the table pin
  the solver against your analysis (`TestVocabularyCompleteness` fails until the entry exists).
- Nothing else. The definitional loop oracle, the call-site consistency check, and the any-axis
  sharding sweep all pick the new string up automatically from an AST scan of the source — there is
  no hand inventory to update, which is the point: four hand-maintained inventories of the old
  module were found wrong.

### The two failure eras this design closes (recorded, because the instincts recur)

**Era 1 — delegation/fusion (the named toolkit).** The old extension rule let a full-group name
*delegate* to a fewer-group twin ("the extra group rides along for free"). That justification was a
claim about arithmetic silently generalised into a claim about structure: the fold was numerically
exact — **so no numerical test could ever see it** — but it froze sharding choices, and it was found
only when a downstream consumer tried to shard. *"A type error, morally speaking"* (Nick,
2026-07-15). Per-site judgement missed sites twice (five `t`-folds found, four `K+C` fusions
*internal* to single einsums missed). Under the interpreter this failure class is **inexpressible**:
there is no body to drift from the string, and no reshape in which to fuse anything.

**Era 2 — the unfusing rule's own limit.** The post-fusion rule ("a shared block flattens to a
letter; a passive block rides as `'...'`") was the best a *fixed* subscript could do, and it carried
a residue: a shared block's own axes still flattened, so only its leading sub-axis was shardable.
Per-call expansion removes the flatten and the residue with it — see the sharding section below.

## Decision history

- **Why the letters `W`/`K`/`C`.** They are deliberately disjoint from the core/variation symbols
  (`U`,`P`,`Q`,`O`,`G`,`H`,`B`). Before the rename they were `F`/`V`/`G`, which clashed with the
  TT-core `G` and the Tucker-variation `V`; `apply`/`entries`/`dense_probe` additionally drifted to
  a private `X`/`V`/`I`/`K`/`Z` scheme (so e.g. `mu_VXa` became `mu_WCa`). Removing that overload
  was the motivation for the rename.
- **The map-over-`K` plan was reversed.** The earlier plan for `K`-stacked forward probing deferred
  the 3-block contractions in favour of `vmap`/map over `K`; it was reversed in favour of genuine
  3-block (`W`,`K`,`C`) contractions — consistency with the `contractions.py` toolkit, no Python
  `K` loop on the numpy path, and low-level einsums fold into XLA at least as well as a `vmap`.
- **Frame-as-aux → frame-as-leaf.** The old design made `T3Tangent`'s frame jax `aux_data` guarded
  by object identity (`self.frame is other.frame`) — a numerical check faked as structural. It
  forced a recompile on every frame change (each Newton step) and false-failed after a jit
  round-trip. Numericalizing the guard (`safety.frames_equal`) let the frame become a pytree
  **leaf**: traced data, compile-once across bases. Full story:
  `dev/archive/safe_unsafe_mode_plan.md`.
- **`K`-stacked residuals for the `apply`/`entries` adjoints shipped** (once deferred as a
  `probe_transpose`-style extension). The adjoint-state transpose takes a residual of shape
  `W + K + C` (`K` optional) and carries `K` into the variation gradient — see
  `_apply_transpose_adjoint` in `backend/apply.py`, reused by `tv_entries_transpose_from_sweep`.
  Build history: `dev/archive/apply_entries_handoff.md`.

## Test-writing guidance (batching-specific)

- **Tests are RNG-order sensitive** (one global seed at import) — a bug class we hit. New numerical
  tests are numpy-only (jax invocation is covered by `test_dispatch`); see `CLAUDE.md`.
- **Stacked arrays blow up fast.** In tests keep stack dims 1–2 and core dims small.
- **Everything follows the modern conventions now** — inferred dispatch (no threaded `use_jax`),
  host-numpy masks; see the `uniform_*` notes. (The old `wt3_*` weighted layer, the last holdout
  that threaded `use_jax`, was deleted; its shipped replacement infers dispatch like the rest.)

## Sharding: from "leftmost member only" to "any axis of any group"

**Current state (the interpreter): every sub-axis of every group is shardable, by construction.**
`contract` never reshapes — each group sub-axis is an honest einsum axis — so there is no flatten to
freeze a layout choice, and fusing two groups is not expressible. The compiled sweep in
`tests/test_contractions_sharding.py` verifies it anyway (shard each sub-axis of each group of every
AST-scanned library subscripts string, 4 virtual devices, zero all-gathers), because compiling was
the only instrument that ever caught this class. The user-facing rule this retired — *"shard the
leading axis of your stack"* — is deleted from `batching_and_stacking.md` §4; any axis now works.

**The history below is kept because it explains the design pressure** (and the measurements are the
evidence that the invariant must stay compiled-checked, not argued).

The named toolkit flattened each index block to one axis and never transposed — a pure
reinterpretation, so numerically exact. But a reshape *reindexes which logical elements live where*,
so it was **sharding-free only if the sharded axis was the MAJOR (leftmost) member of the flattened
group**. With `(t=2, W=4)` sharded on `W`, row-major flat index `t*4 + W`: dev0 holds `W ∈ {0,1}` →
flat `{0,1,4,5}`, dev1 → `{2,3,6,7}`; a contiguous 2-way tiling is `{0..3}`/`{4..7}`, so XLA must
insert a collective. Reversed, `(W=4, K=2)`: dev0 holds `W ∈ {0,1}` → flat `{0..3}` = exactly tile
0. Free.

`W` is the **sample** axis (the data being fitted), so it is what a user shards for data-parallel
multi-GPU — and sharding-friendliness w.r.t. `W` is a library concern (Nick, 2026-07-15). The rule bit
at three levels; all three are **measured** (4 virtual devices, counting `all-gather` in the compiled
HLO), not reasoned. Unfusing killed the first two, and `'...'` killed most of the third — **only a *shared* block's
within-block flatten survives**:

| level | what | free? |
|---|---|---|
| **across blocks** — `t`/`d` vs `W` | five passive-broadcast Tucker lifts delegated to a twin that renamed their leading `t` into the `W` block | **was 3 all-gathers each; now 0** |
| **between blocks** — `W` vs `K`, `K` vs `C` | six delegations fusing `W+K`, four einsums fusing `K+C` into one letter `X` | **was 3 all-gathers for `K`** (minor in the `W+K` fold); **now 0** — no block is fused with another, so `W`, `K` and `C` are each shardable |
| **within a SHARED block** — multi-axis `C` | a shared block must be a letter, so its own axes flatten | leading axis free; **minor axis costs 6** — inherent (a letter requires a flatten), and accepted (below) |
| **within a PASSIVE block** — multi-axis `W`/`K` | rides as `'...'`; **nothing is flattened** | **every axis free** — `W`-minor on `W=(4,4)` was 3 all-gathers, now **0** |

**Why it drifted, and why it needed a test.** The folds were *deliberate* (the docstrings said so) and
**numerically exact** — bit-identical to the explicit form — so no numerical test could ever see them,
and nothing in the library shards. The only instrument that can see them is the compiler: shard a
block, compile under virtual CPU devices, assert **0 all-gathers**. That method survives in the
interpreter's any-axis sweep (`tests/test_contractions_sharding.py`); the per-named-function tests it
originally comprised were deleted with the named functions.

**The shardability contract (built 2026-07-16 for the named toolkit; superseded by the stronger
any-axis sweep).** *Every grouped index must be shardable over its first sub-axis* — equivalent to
the no-fusing rule (fusing `X` with `Y` puts one to the right of the other, whose *first* sub-axis
is then non-major in the flatten). It ran as an automatic sweep over every public contraction,
enumerated from the module's own functions, deliberately **not** `__all__` — which was itself found
wrong (78 listed, 101 defined, and the omissions included the family that hid the fusion bug).
**Four hand-maintained inventories of the old module were found wrong**; that lesson carried into
the interpreter's harness, where every enumeration is an AST scan of the source. The interpreter's
sweep strengthens the contract from "first sub-axis" to **any sub-axis** and needs no exemptions.

**The lesson worth keeping** (it motivated the interpreter): under the named toolkit, group
boundaries were inferred *positionally* ("everything left of `C` is `W`"), so handing a `t`-carrying
array to a callee that named no `t` silently redefined `W` as `t+W` — nothing complained. The block
structure was not data; it existed only in the function's *name*, checkable only in the harness.
The interpreter closes most of the gap: the string *is* executed (no body to contradict it), the
solve refuses underdetermined splits unless supplied, and einsum checks every group axis's extent
across operands. What remains uncheckable at runtime is the same as for any einsum: whether the
string the caller wrote is the contraction they meant — which is what the consumer-level dense
oracles are for. (This resolved the standing architecture question of
`dev/archive/OPEN_QUESTION_contractions_architecture_RESOLVED_2026-07-17.md`.)

### The residue that no longer exists (and the un-rejection that removed it)

Under fixed subscripts, a **shared** block had to be one einsum letter, so its own axes flattened
(`size_C = math.prod(C_shape)`) and only its **leading** axis was shardable. This was accepted
(Nick, 2026-07-15: *"forcing sharding on the leading batch subaxis only is acceptable"*) and stated
as the user rule *"shard the leading axis of your stack"*.

**Generating the einsum string per call — one letter per axis — was explicitly REJECTED then**, "so
it is not re-proposed without new information": it would kill the residue and every reshape, but it
traded away the literal, readable subscripts that made the module auditable, to fix a regime nobody
had asked for. **The new information arrived (2026-07-17), and the option became the design.** The
rejection's premise was that generation *replaces* the readable string; in the interpreter the
grouped string at the call site **is** the readable, greppable spec, and generation happens behind
it. The driver was drift-proofing (the enumerated bodies kept drifting: shared-body helpers, caller
semantics in docstrings — human review kept failing as the guard); the any-axis sharding and the
per-axis extent checking arrived as bonuses. The residue and its user-facing rule are gone.

## Maintenance note (blast radius)

When you change a stacking convention, the user-facing `batching_and_stacking.md`, this note, and
`CLAUDE.md` are part of the blast radius — update all of them. The conventions are deliberate; if
you find yourself wanting to break frame-inner, re-read the user doc's §3 first.
