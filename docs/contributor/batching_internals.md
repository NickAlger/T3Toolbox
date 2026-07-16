# Batching and stacking — contributor internals

Extension rules, decision history, and test guidance excised from the user-facing
[`../batching_and_stacking.md`](../batching_and_stacking.md) — read that first for the conventions
themselves. This is the material you need when **changing or extending** the batching machinery.

## Extension rule: naming as documentation — delegate, don't silently reuse

The contraction's *name* is its batch-group type signature: a reader at the call site sees exactly
which blocks (`W`/`C`/`K`/`d`…) are live, the same role the shape-comments play for arrays. So even
when a fewer-group contraction would *silently* handle a case — because the extra group is a
**shared, aligned prefix on all operands** that just flattens into an existing flat block (e.g.
`Xi_Xj_to_Xij` already computes `XYi_XYj_to_XYij`, with `Y` riding along) — add the full-group name
(`XYi_XYj_to_XYij`) and have it **delegate** (reshape the extra group into the block, call the
simpler function, reshape back), then call the full-group name. A group that lives on only *some*
operands (e.g. the mode index `d` on the cores but not the shared probe vectors) **cannot** ride
free → it is a genuine new contraction, not a delegating wrapper. Both get a name; only the
implementation differs. (Delegating wrappers are covered by the frame's oracle test plus a thin
smoke test; genuine ones get their own dense/loop-oracle test.)

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
multi-GPU — and sharding-friendliness w.r.t. `W` is a library concern (Nick, 2026-07-15). The rule bites
at three levels; all three are **measured** (4 virtual devices, counting `all-gather` in the compiled
HLO), not reasoned:

| level | what | free? |
|---|---|---|
| **across blocks** — `t`/`d` vs `W` | five passive-broadcast Tucker lifts delegated to a twin that renamed their leading `t` into the `W` block | **was 3 all-gathers each; now 0** |
| **between blocks** — `W` vs `K` | six surviving `W+K` folds | free for `W` (major); **3 all-gathers for `K`** (minor) |
| **within a block** — multi-axis `W`/`K`/`C` | each block flattens its own axes (`size_C = math.prod(C_shape)`) | leading axis free; **minor axis costs 6 (C) / 3 (W)** |

**Why it drifted, and why it needs a test.** The fold was *deliberate* (the docstrings said so) and
**numerically exact** — bit-identical to the explicit form — so no numerical test could ever see it, and
nothing in the library shards. The only instrument that can see it is the compiler. Hence
`tests/test_contractions_sharding.py`: shard `W`, compile under 4 virtual CPU devices, assert **0
all-gathers**. It also pins the safe `W+K` folds (so nobody "fixes" them), pins the `K` limitation (so
the doc's claim is checked, not asserted), and includes a deliberately-broken fold to prove the check
*can* fail.

**The detector**, for the reuse pattern that caused it — group boundaries are inferred *positionally*
("everything left of `C` is `W`"), so mis-grouping can only arise at a **delegation**:

> If the callee's name **drops the leading `t`** while the caller has it, `t` has been folded into `W`
> and `W` is no longer major. (Dropping `K` is always fine — `K` is to `W`'s right.)

**`K`-sharding is not supported today**, and that is a choice, not an oversight: `W` is the stated
target, and the two sites that keep `K` as its own einsum letter (`tWCi_KCio_to_tWKCo`,
`dtWCi_dKCio_to_dtWKCo`) are already `K`-shardable (measured 0). If it is ever wanted, the recipe is
identical to the `t` fix — stop fusing, give `K` its own letter. Numerically a no-op, hence mechanical;
just not free (one more axis), and nothing needs it yet.

## Maintenance note (blast radius)

When you change a stacking convention, the user-facing `batching_and_stacking.md`, this note, and
`CLAUDE.md` are part of the blast radius — update all of them. The conventions are deliberate; if
you find yourself wanting to break frame-inner, re-read the user doc's §3 first.
