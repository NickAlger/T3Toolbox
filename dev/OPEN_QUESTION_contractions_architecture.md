# OPEN QUESTION — the `contractions.py` architecture (Nick, think about this properly)

> **DO NOT ARCHIVE.** This note is not a plan and has no slices. It is a standing question, parked
> deliberately, and it stays in `dev/` until Nick has thought it through and decides what (if anything)
> to do. Sweeping it into `dev/archive/` would be exactly the wrong move — it is unresolved, not done.

_Raised 2026-07-15, by Nick, immediately after deciding option B for the unfusing work: **"this issue
does make me feel uneasy about the architecture. I can't really put my finger on why."**_

The unease is the point. This note's job is to **preserve the evidence while it is fresh**, so the
thinking can happen later without re-deriving it — not to diagnose. **The observations below are
material, not conclusions.** Nick has not endorsed any of them as *the* reason, and the reason may be
none of them.

---

## What happened (the trigger, in three lines)

1. Some contractions fused blocks the caller might want to distinguish (`t` into `W`, `W` into `K`,
   `K` into `C`). Numerically exact, so invisible to every test; it blocked GPU sharding.
2. The fix looked mechanical: give every named block its own einsum letter.
3. **It is impossible.** The `W|K` and `K|C` splits are not recoverable from the operands. The fusion
   was never laziness — it was *forced by the signature*.

## The observations (material, not conclusions)

**1. The block structure is not data. It exists only in the function's name.**
This may be the load-bearing one. `WKCi_Cio_to_WKCo` *declares* three blocks. No value carries that
information: `W=(2,3),K=()` and `W=(2,),K=(3,)` and `W=(),K=(2,3)` are **the same array**, shape
`(2,3,3,4)`. Only `Cio` pins `len(C)`, and only because it is a separate operand. So the "type" lives in
the caller's head and in a docstring — it cannot be checked, cannot be inferred, and cannot be wrong in
any way the machine can detect. `docs/contributor/batching_internals.md` says outright *"the
contraction's name is its batch-group type signature"* — a type signature over information that does not
exist at runtime.

**2. Every function re-derives the structure positionally, from `ndim` arithmetic.**
`W_shape = WCi.shape[:len(WCi.shape) - 1 - n_frame]` — "everything left of `C` is `W`". Fifty-odd
functions each re-parse the same convention by counting axes. That inference is what made the fusing
*silent*: it cannot tell "`t` then `W`" from "one big `W`", so a caller handing a `t`-carrying array to a
`W`-shaped callee gets a plausible answer.

**3. The abstraction cannot be parameterised, so it is enumerated. 2812 lines.**
`contractions.py` is ~60 hand-written variants of "which blocks sit on which operand". Every new
combination is a new function with a near-identical body. That is what an un-parameterisable abstraction
looks like from the inside. *(Counterpoint, in fairness: the enumeration is also what makes it auditable
— each subscript is literal and greppable, which is exactly how the fusing bug was eventually found.)*

**4. Option B is a hint that the machinery is over-applied.**
`'...'` handled the un-splittable blocks natively — no letter, no flatten, no split, and it *fixed* a
limitation (`W`-minor sharding) that the plan had accepted as inherent. The module's stated reason for
existing is "TWO independent batch blocks on different operand subsets, which a single `'...'` cannot
express". True — but B shows one of those two blocks often needs no letter at all. **How many of the ~60
functions actually need machinery-2?** Not asked yet. Worth asking before anything is redesigned.

**5. The failure mode is structural, not incidental.**
Fusing is numerically *exact*. So the entire numerical suite — the dense oracles, the ragged-vs-uniform
equivalence contract, 40k subtests — is **constitutionally blind** to it. It took a downstream consumer
(T3Polynomial) trying to shard to find it. Whatever the architecture is, it has a class of property that
its test strategy cannot see. That is worth sitting with independently of any redesign.

**6. The module is public API — and probably should not have been.**
_Nick, 2026-07-16: "Probably, we shouldn't have made contractions.py part of the public-facing API
surface since it is really internal plumbing. But too late now."_ Concretely: 101 public defs, every one
rendered into the published API reference (`docs/conf.py`'s `verbatim_signature` pulls each signature
out of the source verbatim). That is why the `trs` canonicalization landed as a **breaking change**
rather than a tidy-up. It also silently raises the price of every direction below: *"make blocks data"*
and *"generate the subscript from a block spec"* both rewrite a surface users are invited to call, so
each is a migration rather than a free choice. **Worth asking when this is taken up: is the public
surface itself part of what should change?** If the module were internal plumbing, most of the cost in
this note evaporates.

## Candidate directions (unexplored — do NOT treat as a shortlist)

Listed so they are not re-invented from scratch. None costed, none recommended.

- **Make blocks data.** A `BlockedArray`-ish carrier of `(array, block_spec)`, so the split is a value:
  checkable, inferable, impossible to mis-group. Cost: a new type through the whole backend, and the
  raw-`.data` razor says users must be able to bypass the frontend — would this become a second frontend?
- **Generate the subscript from a block spec** per call, one letter per axis. Kills every flatten. Costs
  the literal, readable subscripts that make the file auditable (and that found this bug). Recorded as
  **rejected** in `dev/contractions_unfusing_plan.md` §4, on those grounds — but rejected against the
  *old* premise, before B; worth re-examining if the premise moved.
- **`vmap` the batch blocks** instead of hand-flattening them. jax-native, and it is *exactly* the
  "batch this over an axis" operation being hand-rolled. Blocker: numpy has no `vmap`, and the backend is
  dual-path by design (`xnp`/`xmap`/`xscan`). Does `xmap` already reach far enough?
- **Do nothing.** Also a real answer. It works, it is well-tested, and B fixed the concrete problem. The
  unease may be about a cost already paid rather than one still coming.

## Revisit: the `trs_*` contractions are sparse convolutions faked as dense operands

_Nick, 2026-07-16 — a direction, not just an observation. Unlike the material above, this one has an
intended shape; it is parked on cost, not on doubt._

**The idea.** A `trs` binomial tensor **is a sparse convolution tensor**. Passing it to einsum as a
dense operand is the wrong handling. The right form is a **zippering sum of non-`trs` contractions**
— i.e. unroll the convolution over the order axes and sum ordinary contractions — instead of one big
`trs` contraction. **This is a big job**, and it is *not* to be started before there is a correct and
tested `trs` baseline: that baseline is what a rewrite would be validated against. So the current
contractions earn their keep as the reference implementation even if they are eventually replaced.

**Independent measured support (2026-07-16, while probing the shardability contract).** The `trs_*`
family is exactly where `_pairwise_path` degenerates, and the mechanism is the same "dense operand"
mistake:

- `_pairwise_path` is greedy on **shared** indices and is **size-blind by design** (its docstring:
  *"Keyed only on the subscript string (index sharing is size-independent)"*). But a shared index that
  is in the output is a **batch** index and contracts nothing.
- The binomial tensor shares only ONE index with each operand, while the operand pairs share three
  (`d`,`W`,`C` — all batch). So `trs` — **the only operand that can contract anything** — is sorted
  **last**. For `tus,dtWKCa,duWCi,dsWCb->dWKCaib` the first two of three steps contract *nothing*, and
  the intermediate grows to the union of all ten indices before anything is summed.
- Peak intermediate is then `s*t*u` × the output: measured **27×** at order-axis size 3, and **8–64×**
  across the realistic order range 2–4. **8 of 47** multi-operand contractions improve under a
  contracted-index heuristic (aggregate peak 42.3M → 5.8M elements); **2 get worse**.
- Scope: **numpy path only** — `_grouped_einsum` hands jax the raw einsum and lets XLA choose. And this
  is a **peak-memory** measurement, *not* a runtime claim: fewer FLOPs already misled us here once
  (`optimize=True` minimised FLOPs and ran **50×** slower by breaking BLAS-friendliness), so no
  path-heuristic change should be inferred from these numbers alone.

The connection worth keeping: the structural instinct (*a sparse convolution tensor should not be a
dense einsum operand*) and the measured pathology (*the convolution operand is the one that gets
deferred, so nothing contracts until the last step*) are **the same fact seen from two sides**. That
is a reason to believe the zippering direction is right, and a reason the numpy-path oddity is a
symptom rather than a bug to patch.

## Where the evidence lives

- `dev/contractions_sharding_plan.md` — the first fix (order-axis folding); §2 has the all-gather
  measurements; §8 has the separate numpy `optimize=` regime finding (also unresolved).
- `dev/contractions_unfusing_plan.md` — the unfuse-everything plan. **Its §2 and §4 are now known
  wrong** (the splits are unpinnable; the leading-axis limit does not apply to passive blocks); see the
  option-B decision. Read it with that caveat.
- `docs/contributor/batching_internals.md` — the rule and its revision history.
- Commits: `345dad57` (order-axis fix), and the option-B unfusing that follows this note.
