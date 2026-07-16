> **DONE — 2026-07-16 (`4e3381ad`). Archived; kept as the build record.** The contract is built and
> passing: `TestShardabilityContract` in `tests/test_contractions_sharding.py`, an automatic sweep over
> every public contraction, 280 (function, block) pairs, ~37s, holding everywhere with nothing exempted.
> The rule and the two non-obvious details now live in `docs/contributor/batching_internals.md` — read
> that, not this.
>
> **Two things this plan got wrong or did not know:**
> 1. It proposed parsing the ~60 names in `__all__`. There are **101** public defs — `__all__` was itself
>    wrong (78 listed), and its 23 omissions included the family that hid the fusion bug. The sweep
>    enumerates the module's functions instead. *Fifth* inventory of this module found wrong.
> 2. The feasibility probe came out **78/78** (200/200 against the hand-written shape comments), far past
>    the "50 clean + 10 exceptions" bar — so the automatic check was never in doubt. It also found the
>    `trs`/`tus` naming drift and 259 redundant shape comments, both since fixed (`67cac3f3`, `98d8b161`).
>
> **The watch-list's `_pairwise_path`/`'...'` trap is still open** — the contract did not require a guard.
> Live status: `dev/HANDOFF.md`. The standing architecture question:
> `dev/OPEN_QUESTION_contractions_architecture.md` (**not** archived — unresolved, not superseded).

# The shardability contract — every grouped index shardable over its first sub-axis

_Nick's proposal, 2026-07-15. **This is the live work**; the unfusing that motivated it is DONE
(`dae52839`, `f65b341d`) and its plans are archived. Read `docs/contributor/batching_internals.md` for
the rule as it now stands, and `dev/OPEN_QUESTION_contractions_architecture.md` for the standing unease
this probes._

## Why this, and why now

The fusing bug was found three times over, and **the human-maintained inventory was wrong every time**:

1. the upstream survey found the 5 `t`-folds and missed the 4 internal `K+C` fusions (its detector only
   looked at *delegations* — it structurally could not see them);
2. the detector I then wrote into the module docstring missed them too;
3. my "complete, mechanically derived" inventory (unfusing plan §2) then **exempted `_assemble_dU_dxi`
   and `_assemble_dU_dxi_d` as "already clean"** — they fused `K+C` into `X` and backed four *public*
   functions. The implementing agent found them by grepping for the fused letter instead of trusting
   the list.

Three enumerations, three misses. **That is the argument.** A uniform obligation has no list to be
wrong about.

## The contract

> **Every grouped index in `contractions.py` must be shardable over its first sub-axis.**

Nick's proposal, to be built **after B lands**. It is stronger than the static naming check this plan
originally proposed (§3), for three reasons:

1. **It is equivalent to the no-fusing rule, not a proxy for it.** Fusing `X` with `Y` necessarily puts
   one of them to the right of the other, and the right-hand one's *first* sub-axis is then non-major in
   the flatten. So *every block shardable on its first sub-axis* ⟺ *no block is fused with a preceding
   block* ⟺ *no fusing*. Every case we found confirms it: `t`+`W` fused → `W`'s first axis fails;
   `W`+`K` → `K`'s first fails; `K`+`C` → `C`'s first fails. One uniform obligation, no per-site
   judgement — which is what we have now missed **twice**.
2. **It encodes exactly the limit Nick accepted** ("forcing sharding on the leading batch subaxis only is
   acceptable"): it permits the within-block flatten a *shared* block requires, and forbids cross-block
   fusion.
3. **It checks a property, not a form.** A static name-vs-subscript check can be satisfied by writing the
   letters and flattening anyway. This cannot.

**The interesting part.** The test must know where each block starts — the same unpinnable split — but a
test *constructs* its inputs, so it knows. And the block layout is **parseable from the function's name**
(`WKCi_Cio_to_WKCo` ⇒ operand 1 is `W+K+C+(i,)`, operand 2 is `C+(i,o)`), so the check can be **automatic
over the whole file** rather than a hand table. That turns the name from an unchecked promise into a
**checked** one — directly addressing observation #1 in
`dev/OPEN_QUESTION_contractions_architecture.md`. It does not dissolve the unease (the structure still is
not data at *runtime*, only in the harness), but the names can no longer lie silently.

It is also **cheap information about the blocks-as-data direction**: if parsing the names into a block
spec comes out clean, the spec wants to exist; if it is riddled with exceptions, that is evidence too.

**Feasibility to check FIRST** (do not promise before measuring): how many of the ~60 names parse
cleanly. The regular ones are trivial; the `trs_*` family, the order axes, and the sum-over-`W` ones
(`Wo_WKCa_to_KCao` — `W` is absent from the output) need care. 50 clean + 10 documented exceptions is a
good test; 30/30 is a hand table and much less attractive. **Cost**: ~60 functions × ≤3 blocks × ~250ms
≈ under a minute.

## Watch-list

- **`_pairwise_path` and `'...'` — latent, not triggered** (found while landing option B, verified). It
  builds `set(term)` per operand, so the `.` of an ellipsis lands in the term set and counts as a
  *shared index*, skewing the greedy pairing. Harmless today: every `'...'` rewrite is 2-operand, and
  `_grouped_einsum` bypasses `_pairwise_path` entirely at ≤2 operands. **It would bite the first
  3+-operand `'...'` contraction anyone writes.** Worth a guard when this lands — this is the natural
  place, since the contract is what would make such a contraction appear.
- **Probe feasibility BEFORE promising**: how many of the ~60 names parse into a block spec. 50 clean +
  10 documented exceptions is a good test; 30/30 is a hand table and much less attractive. The `trs_*`
  family, the order axes, and the sum-over-`W` sites (`Wo_WKCa_to_KCao` — `W` absent from the output)
  are the awkward ones.
- **Cost**: ~60 functions × ≤3 blocks × ~250ms ≈ under a minute. Trimmable.
- Divisibility: a sharded axis must be divisible by the device count (4 in the existing harness).
