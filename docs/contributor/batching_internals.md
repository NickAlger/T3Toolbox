# Batching and stacking — contributor internals

Extension rules, decision history, and test guidance excised from the user-facing
[`../batching_and_stacking.md`](../batching_and_stacking.md) — read that first for the conventions
themselves. This is the material you need when **changing or extending** the batching machinery.

## Extension rule: naming as documentation — the name is the type, and the body must honour it

The contraction's *name* is its batch-group type signature: a reader at the call site sees exactly
which blocks (`W`/`C`/`K`/`d`…) are live, the same role the shape-comments play for arrays. The rule
that makes that signature honest rather than aspirational:

> **Never FUSE two named blocks (flatten them together). Flatten only what einsum forces you to.**

Two cases, and the distinction is the whole rule:

- A **shared** block — present on several operands, so einsum needs a letter for it (e.g. `C`, pinned by
  a `C`-only core). It *must* become a letter, so its own axes are flattened into one.
- A **passive** block — present on a single operand and riding unchanged to the output (e.g. the `W+K`
  prefix of `WKCi` when the other operand is `C`-only). It needs **no letter and no flatten**: it rides
  as `'...'`. `'...Ci,Cio->...Co'`.

So when a fewer-group contraction would *silently* handle a case, still add the full-group name — but
implement it by letting the extra group **ride**, never by folding it into an existing flat block. **No
fusing**, in either form: not by delegating to a differently-shaped twin, and not by merging two named
blocks into one letter internally.

> **Why not simply "every named block gets its own letter"?** Because it is **impossible**, and finding
> that out is what produced this rule. The `W|K` and `K|C` splits are **not recoverable from the
> operands**: for `WKCi_Cio_to_WKCo`, the splits `W=(2,3),K=()`, `W=(2,),K=(3,)` and `W=(),K=(2,3)` all
> give operand shape `(2,3,3,4)`, and only `Cio` pins `len(C)`. There is no `W_shape` to compute. A
> letter would require threading the split in as a new parameter on ten public backend functions — and
> would *still* flatten each block internally. `'...'` needs no parameter, and flattens nothing.
>
> The fusion was therefore **forced by the signature, not laziness**. The docstring that read *"no
> operand carries C without K, so they ride as one combined block"* was not an excuse; it was stating
> exactly this.

Every name is then a genuine contraction needing its own oracle test — there are no thin wrappers riding
on a callee's test.

### The error this replaces (recorded, because it will otherwise recur)

The rule used to mandate the opposite: add the full-group name and have it **delegate** — *"reshape
the extra group into the block, call the simpler function, reshape back"* — on the justification that
the extra group *"rides along for free"*.

**That justification was a claim about arithmetic, silently generalised into a claim about
structure.** A group that flattens into an existing block *is* free arithmetically — the fold is
numerically exact, bit-identical, verified. It is **not** free for anything that distinguishes axes:
sharding today (below), and layout control, donation, custom partitioning and per-axis
rematerialisation whenever they arrive. So the rule declared a type system and then licensed casts
through it — *"a type error, morally speaking"* (Nick, 2026-07-15). A function named `tWKCi_…` whose
body hands its array to a `WCi_…` callee has a signature that says one thing and a body that does
another; the gap between them is where the bugs live.

The failure mode is the worst kind. Because the fold is exact, **no numerical test can ever see it**,
and nothing in the library sharded, so nothing else could either; it was found only by a downstream
consumer trying to shard. That is the argument for a **structural** rule over per-site judgement —
per-site judgement missed sites **twice**: the first survey found the five `t`-folds and missed four
`K+C` fusions *internal* to a single einsum (subscript `'Wo,WXa->WXao'`, one letter `X` for `K+C`),
which have no callee to name-check and so are invisible to any delegation-based detector. The guard is
therefore a static check of the rule itself (`tests/test_contractions_naming.py`): parse each public
`def`, take the blocks in its name (before `_to_`), take the blocks in its einsum subscripts, assert
the first is a subset of the second. It found all ten sites in seconds, with no device and no HLO.
(Build record: `dev/contractions_unfusing_plan.md`.)

**Test consequence, paid once.** The old rule let a delegating wrapper get by with a thin smoke test —
its callee carried the dense/loop-oracle test. Unfusing makes all ten of them genuine, so each needs a
real oracle test of its own: real work, and real coverage that did not exist before.

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
- **`K`-stacked residuals for the `apply`/`entries` adjoints are deliberately deferred** (a
  `probe_transpose`-style extension; build history: `dev/archive/apply_entries_handoff.md`).

## Test-writing guidance (batching-specific)

- **Tests are RNG-order sensitive** (one global seed at import) — a bug class we hit. New numerical
  tests are numpy-only (jax invocation is covered by `test_dispatch`); see `CLAUDE.md`.
- **Stacked arrays blow up fast.** In tests keep stack dims 1–2 and core dims small.
- **The parked weighted layer still threads `use_jax`** (the old pattern) — don't take it as a
  model for new code. (The uniform layer follows the modern conventions: inferred dispatch,
  host-numpy masks — see the `uniform_*` notes.)

## Sharding: only the leftmost member of a flattened group is free

`contractions.py` flattens each index block to one axis and never transposes — a pure reinterpretation,
so it is numerically exact. But a reshape *reindexes which logical elements live where*, so it is
**sharding-free only if the sharded axis is the MAJOR (leftmost) member of the flattened group**. With
`(t=2, W=4)` sharded on `W`, row-major flat index `t*4 + W`: dev0 holds `W ∈ {0,1}` → flat `{0,1,4,5}`,
dev1 → `{2,3,6,7}`; a contiguous 2-way tiling is `{0..3}`/`{4..7}`, so XLA must insert a collective.
Reversed, `(W=4, K=2)`: dev0 holds `W ∈ {0,1}` → flat `{0..3}` = exactly tile 0. Free.

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

**Why it drifted, and why it needs a test.** The folds were *deliberate* (the docstrings said so) and
**numerically exact** — bit-identical to the explicit form — so no numerical test could ever see them,
and nothing in the library shards. The only instrument that can see them is the compiler. Hence
`tests/test_contractions_sharding.py`: shard a block, compile under 4 virtual CPU devices, assert **0
all-gathers**. It pins the freed `W`/`K`/`C` cases (so the doc's claims are checked, not asserted) and
includes a deliberately-broken fold to prove the check *can* fail (re-fusing one site by hand fails
three tests).

**The shardability contract (built, 2026-07-16) — the standing guard.** *Every grouped index must be
shardable over its first sub-axis.* It is **equivalent** to the no-fusing rule, not a proxy for it —
fusing `X` with `Y` necessarily puts one of them to the right of the other, and the right-hand one's
*first* sub-axis is then non-major in the flatten. It also encodes exactly the limit accepted below. A
static name-vs-subscript check was considered and is weaker: it can be satisfied by writing the letters
and flattening anyway, and it cannot express `'...'` at all.

It runs as an **automatic sweep** (`TestShardabilityContract`) over **every** public contraction —
enumerated from the module's own functions, deliberately **not** `__all__`, which was itself found wrong
(78 listed, 101 defined, and the 23 omissions included the family that hid the fusion bug). 280
(function, block) pairs, ~37s; it holds everywhere today with nothing exempted. **Adding a contraction
adds its coverage automatically — there is no list to update, which is the entire point:** four
hand-maintained inventories of this module have now been found wrong.

Two things not to re-derive if you touch it. The block under test gets `(n_devices, 2)` while every
*other* block gets size **2, not 1** — a size-1 neighbour still tiles correctly when fused, so the check
would pass vacuously. And it is validated by **mutation**: re-fusing `W+K` by hand in
`WKCi_Cio_to_WKCo` fails on `K` and not on `W`, at 3 all-gathers — the block the equivalence argument
predicts, at the cost this table records.

**Why it could drift at all:** group boundaries are inferred *positionally* ("everything left of `C` is
`W`"), so handing a `t`-carrying array to a callee that names no `t` silently redefines `W` to mean
`t+W` — nothing complains. **This is worth sitting with:** the block structure is not data. It exists
only in the function's *name*. The contract now makes that name a **checked** promise — the sweep parses
the block layout from it, so a name that contradicts its body fails — but the structure is still not data
at *runtime*, only in the harness. See the standing
open question in `dev/OPEN_QUESTION_contractions_architecture.md`.

### The residue: within-block flattening (accepted, not a bug to fix)

Unfusing does **not** eliminate block flattening for a **shared** block — one einsum needs a letter for.
Such a block is still flattened (`size_C = math.prod(C_shape)`), because a *variable number of axes per
block* cannot be expressed in a fixed einsum string — that is
precisely why `contractions.py` exists. So a **multi-axis** `W`/`K`/`C` flattens its own axes and only
its **leading** axis is shardable.

**Accepted (Nick, 2026-07-15): "Forcing sharding on the leading batch subaxis only is acceptable."** So
it is a stated **user-facing rule** — *shard the leading axis of your stack*
([`../batching_and_stacking.md`](../batching_and_stacking.md) §4) — not a silent limitation.

**Rejected: generating the einsum string per call, one letter per axis.** It is the only thing that
would kill the residue too — genuinely bulletproof, and it would remove every reshape — but it trades
the literal, readable subscript strings that make `contractions.py` auditable, to fix a regime nobody
has asked for. Recorded here so it is not re-proposed without new information.

## Maintenance note (blast radius)

When you change a stacking convention, the user-facing `batching_and_stacking.md`, this note, and
`CLAUDE.md` are part of the blast radius — update all of them. The conventions are deliberate; if
you find yourself wanting to break frame-inner, re-read the user doc's §3 first.
