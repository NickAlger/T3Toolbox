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

## Maintenance note (blast radius)

When you change a stacking convention, the user-facing `batching_and_stacking.md`, this note, and
`CLAUDE.md` are part of the blast radius — update all of them. The conventions are deliberate; if
you find yourself wanting to break frame-inner, re-read the user doc's §3 first.
