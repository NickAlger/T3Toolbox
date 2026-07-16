# `contractions.py`: unfuse everything — the name is the type, and the body must honour it

> **ARCHIVED 2026-07-15 — DONE (`dae52839`, `f65b341d`), and WRONG in several places. Do not use as a
> reference.** The rule as it actually stands is `docs/contributor/batching_internals.md`; the live
> follow-up is `dev/contractions_shardability_contract_plan.md`. Known-wrong sections, beyond the ⚠️ box
> below (which covers only §2/§4):
> - **§0** says the inventory missed sites "twice". It was **three** times — §2 itself then exempted
>   `_assemble_dU_dxi`/`_assemble_dU_dxi_d` as "already clean"; they fused `K+C` into `X` and backed
>   four public functions.
> - **§2** calls itself "complete, mechanically derived". **It was not** (see above), and its "should be"
>   column is impossible anyway (the splits are unpinnable — ⚠️ box).
> - **§3** proposes a static name-vs-subscript guard. **Superseded** by the shardability contract: a
>   static check can be satisfied by writing the letters and flattening anyway, and cannot express
>   `'...'` at all.
> - **§6** references §3.
> Kept for the history — the reasoning, the measurements, and the roads not taken.

> **⚠️ REVISED 2026-07-15 mid-build — §2 and §4 below are WRONG as written; read this box first.**
> The plan assumed every named block could be given its own einsum letter. **It cannot.** The `W|K` and
> `K|C` splits are **not recoverable from the operands**: for `WKCi_Cio_to_WKCo`, the splits
> `W=(2,3),K=()`, `W=(2,),K=(3,)` and `W=(),K=(2,3)` all give operand shape `(2,3,3,4)`, and only `Cio`
> pins `len(C)`. **The fusion was forced by the signature, not laziness** — the docstring §2 quotes as a
> bad excuse (*"no operand carries C without K"*) was stating exactly this.
>
> **Decision (Nick): option B.** A **passive** block — one that lives on a single operand and rides
> unchanged to the output — needs no letter and no flatten: it rides as `'...'`
> (`'...Ci,Cio->...Co'`). No signature changes. Verified bit-identical, and **0 all-gathers where the
> letter-based option gives 3**.
>
> **The rule is therefore not "every block gets its own letter".** It is: **flatten only what einsum
> forces you to.** A *shared* block (on several operands, e.g. `C` pinned by a C-only core) must become a
> letter — so its own axes flatten and only its **leading** axis shards. A *passive* block needs neither
> — it rides, and **all** its axes shard. This also means **§4's "only a block's leading axis is
> shardable" is half wrong**: true for shared blocks, false for passive ones (measured: `W`-minor on
> `W=(4,4)` costs 3 today, **0** under B).
>
> Nick's "type error" framing survives intact, and is sharper: the error was that the body **destroyed**
> a distinction the name declares — flattening *merges* axes, and the merge is what kills sharding. B
> destroys nothing. A letter would make the body *know* the split; `'...'` makes it *not need* it.

_Started 2026-07-15, immediately after the order-axis sharding fix
(`dev/contractions_sharding_plan.md`, S1–S4 shipped in `345dad57`). **Decision (Nick): unfuse ALL
contractions.** This note is the plan and the reasoning. Scope: 10 functions, one rule revision, one
static guard. Numerically a provable no-op._

## 0. The decision, and why it is the right one

> *"It always struck me as a 'type error', morally speaking."* — Nick

That is exactly what it is, and the existing rule says so itself before violating it.
`docs/contributor/batching_internals.md` opens:

> *"The contraction's **name** is its batch-group type signature: a reader at the call site sees exactly
> which blocks (`W`/`C`/`K`/`d`…) are live."*

and then, two sentences later, mandates the cast that breaks it:

> *"…add the full-group name (`XYi_XYj_to_XYij`) and have it **delegate** (reshape the extra group into
> the block, call the simpler function, reshape back)."*

So the rule **declares a type system and licenses casts through it.** A function named `tWKCi_…` whose
body hands its array to a `WCi_…` callee has a signature that says one thing and a body that does
another; the gap between them is where the bugs live.

**The root error, stated once:** the rule's justification — the extra group *"rides along for free"* —
is a claim about **arithmetic**, silently generalised into a claim about **structure**. It rides free
through the numbers. It does not ride free through anything that cares about *axis identity*. Sharding
is the first such thing we hit. It will not be the last: layout control, donation, custom partitioning,
and per-axis rematerialisation all care about which axis is which.

**And the failure mode is the worst kind.** The fold is numerically **exact** — bit-identical, verified
— so no numerical test can ever see it, and nothing in the library sharded, so nothing else could
either. It stayed wrong silently and was found only by a downstream consumer trying to shard. The class
of bug is *"invisible to the entire test suite by construction"*. That is the argument for a structural
rule over per-site judgement: we have now missed sites **twice** (the survey found the 5 `t`-folds and
missed the 4 internal ones; my first module-docstring detector missed them too).

## 1. The rule

> **Every block named in a function's name must appear as its own letter in its einsum subscript.**

No fusing, in either form — not by delegating to a differently-shaped twin, and not by merging blocks
into one letter internally. If a caller needs a contraction, they call the properly-named one; if it
does not exist, they write it. The subscript string becomes the single source of truth for grouping, and
the name is an honest type signature rather than an aspiration.

This is a **strict strengthening** of "naming as documentation": the name was always the type; now the
body is checked against it.

## 2. The inventory (complete, mechanically derived — §3)

**10 sites, in two kinds.** The second kind is the one that matters for the decision: it is invisible to
the delegation detector the survey proposed, because there is no callee to name-check.

**Delegation-fusing (6)** — fuse `W+K`; safe for `W` (major), **cost 3 all-gathers for `K`** (measured):

| function | delegates to |
|---|---|
| `WKCi_Cio_to_WKCo` | `WCi_Cio_to_WCo` |
| `tWKCi_Cio_to_tWKCo` | `tWCi_Cio_to_tWCo` |
| `tWKCo_Cio_to_tWKCi` | `tWCo_Cio_to_tWCi` |
| `dWKCi_dCio_to_dWKCo` | `dWCi_dCio_to_dWCo` |
| `dtWKCi_dCio_to_dtWKCo` | `dtWCi_dCio_to_dtWCo` |
| `dtWKCo_dCio_to_dtWKCi` | `dtWCo_dCio_to_dtWCi` |

**Internal fusing (4)** — fuse `K+C` into one letter `X`; safe for `W` and `K`, would break **`C`**:

| function | current subscript | should be |
|---|---|---|
| `Wo_WKCa_to_WKCao` | `'Wo,WXa->WXao'` | `'Wo,WKCa->WKCao'` |
| `Wo_WKCa_to_KCao` | `'Wo,WXa->Xao'` | `'Wo,WKCa->KCao'` |
| `dWo_dWKCa_to_dWKCao` | `'dWo,dWXa->dWXao'` | `'dWo,dWKCa->dWKCao'` |
| `dWo_dWKCa_to_dKCao` | `'dWo,dWXa->dXao'` | `'dWo,dWKCa->dKCao'` |

These carry their own justification in the docstring — *"K and C never need separating here (no operand
carries C without K), so they ride as one combined block"* — which is the same numerics-for-structure
error, one block over. **"Never need separating" was true of the arithmetic and false of the type.**

*(Already clean and needing no work: the `_assemble_*` helpers, which dispatch on a `keep_W` flag to a
shared private impl that keeps `t`/`W`/`K`/`C` as separate letters; and the five `t`-folds fixed in
`345dad57`.)*

## 3. The detector — static, complete, cheap (the real guard)

The survey's detector (grep for a delegation whose callee drops the leading `t`) is **structurally
incapable** of seeing the internal fusions, and needs a human to read each hit. Replace it with a check
of the rule itself:

> Parse each public `def`; take the blocks in its **name** (before `_to_`); take the blocks in its
> **einsum subscripts** (or, for a delegation, in the **callee's** name); assert the first is a subset
> of the second.

It found all 10 in seconds, with no HLO compile and no device. This should be a **test**
(`tests/test_contractions_naming.py`), not a grep: it is the guard that makes the rule enforceable
rather than aspirational, it is O(ms), and it catches every future site — including ones nobody thought
to list in the multi-device test.

The multi-device HLO test (`tests/test_contractions_sharding.py`) stays: it checks the *property*
(0 all-gathers) where the static check only checks the *form*. Belt and braces, and they fail for
different reasons — which is the point.

## 4. What this does NOT fix — accepted, and to be documented not believed away

Unfusing does not eliminate **block flattening**: every function still does
`size_W = math.prod(W_shape)`, because a *variable number of axes per block* cannot be expressed in a
fixed einsum string — that is precisely why this file exists. So a **multi-axis** `W`/`K`/`C` flattens
its own axes, and only its **leading** axis is shardable. Measured: sharding `C`'s minor axis costs
**6** all-gathers, `W`'s minor axis **3**.

**Accepted (Nick): "Forcing sharding on the leading batch subaxis only is acceptable."** So this becomes
a **user-facing rule** — *shard the leading axis of your stack* — stated in
`docs/batching_and_stacking.md`, not a silent limitation.

*(The only thing that would kill it too is generating the einsum string per call, one letter per axis.
Genuinely bulletproof, and it would remove every reshape — but it trades the literal, readable subscript
strings that make this file auditable, to fix a regime nobody has asked for. **Rejected**, recorded here
so it is not re-proposed without new information.)*

## 5. The rule revision (the durable half)

`docs/contributor/batching_internals.md`'s extension rule must be revised **deliberately**, not quietly:

- **Keep**: "the name is its batch-group type signature" — this was always right, and is now enforced.
- **Drop**: "have it delegate (reshape the extra group into the block…)". Replace with: write the
  contraction explicitly, every named block as its own letter.
- **Record the error**, because it is instructive and will otherwise recur: *"rides along for free" was
  a numerical claim generalised to a structural one.* A group that flattens into an existing block is
  free **arithmetically** and **not** free for anything that distinguishes axes.
- **Note the test consequence.** The rule currently says delegating wrappers get *"a thin smoke test"*
  while genuine contractions get a full dense/loop-oracle test. Unfusing makes all 10 **genuine** — so
  they need real oracle tests. That is real work, and real coverage we do not have today.

## 6. Testing

1. **Bit-equality per site, against the current implementation, before deleting it** — the delegation/
   fused form is the oracle, and the change is a provable no-op. Over the stack matrix incl. **empty**
   blocks (`= 1 when empty`) and **multi-axis** blocks. This is what makes 10 rewrites safe.
2. **The static naming guard** (§3) — new.
3. **The sharding test** gains the newly-freed cases: `K` on the 6 unfused delegations (currently pinned
   as *expected to reshard* — that characterization test **will flip and must be updated**, which is
   exactly what it was built to do), and `C` on the 4 unfused internal sites.
4. Oracle tests for the 10 now-genuine contractions (§5).
5. Full suite + docs `-W`.

## 7. Slices

1. **S1 — the 6 delegation-fusions.** Explicit einsums, bit-equality tests.
2. **S2 — the 4 internal `K+C` fusions.** Same. (Separate slice: different pattern, different letter
   budget, and the `X`-letter docstrings need rewriting.)
3. **S3 — the static naming guard** (`tests/test_contractions_naming.py`) + extend the sharding test to
   `K`/`C`; flip the `K`-is-unsupported characterization.
4. **S4 — the rule revision** in `batching_internals.md` (§5) + the user-facing leading-axis rule in
   `batching_and_stacking.md` (§4) + module docstring.
5. **S5 — oracle tests** for the now-genuine contractions (§5), if not folded into S1/S2.

## 7a. NEXT (Nick, 2026-07-15) — the shardability contract, as a uniform obligation

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

## 8. Watch-list

- **The `X` letter.** The 4 internal fusions use `X` for the fused `K+C`. Unfused they need `K` and `C`
  as separate letters — check no subscript collision with existing single-axis letters (`a`,`b`,`i`,
  `o`…). Capitals are blocks, lowercase are single axes; the convention has room.
- **`Wo_WKCa_to_KCao` sums over `W`.** Unfusing must not disturb the sum-over-`W` semantics (the output
  drops `W`); the bit-equality test is the check.
- **Perf.** The `t`-split measured ~5–8% slower on numpy 2-operand, a wash (~4% faster) on jax. Expect
  the same order. Nick: jax is the performance path; numpy's bar is "not doing something dumb".
- **Do not also "fix" the within-block flattening** (§4) — it is inherent and accepted.
