# Claude Code session — t3m-swap implementation

*2026-06-14 18:28 – 00:41  ·  T3Toolbox  ·  36 prompts, 298 tool calls*

> Rendered from the Claude Code transcript at the **readable** level: human + assistant messages verbatim; each tool call collapsed to a one-line summary; internal reasoning and raw tool outputs omitted.

---


## 🧑 Nick

Hi Claude! Today I would like to start by implementing the t3m-swap method we planned previously.

## 🤖 Claude

I'll start by reading the planning docs to refresh on the t3m-swap design we worked out.

- 🔧 **Read** `docs/t3m_plan.md`
- 🔧 **Read** `docs/t3m_handoff.md`

Let me study the existing implementations — especially method (b) which (c) reuses, plus the supporting backend functions and the test harness.

- 🔧 **Read** `t3toolbox/backend/t3_linalg.py`
- 🔧 **Read** `tests/test_t3m.py`

Now let me look at the frontend dispatcher and the supporting backend functions I'll be reusing.

- 🔧 **Bash** — Find t3m frontend in tucker_tensor_train.py
- 🔧 **Read** `t3toolbox/backend/linalg.py`

Let me read the frontend `t3m`/`__mul__` and the supporting orthogonalization/operations modules.

- 🔧 **Read** `t3toolbox/tucker_tensor_train.py`
- 🔧 **Read** `t3toolbox/backend/t3_svd.py`

Let me read the operations and orthogonalization modules to see `reverse_tt` and the per-core SVD helpers.

- 🔧 **Read** `t3toolbox/backend/t3_operations.py`

Now let me read the orthogonalization modules to understand the per-core joint truncation helpers.

- 🔧 **Read** `t3toolbox/backend/t3_orthogonalization.py`

Let me look at the plain TT orthogonalization module and the ranks helper.

- 🔧 **Read** `t3toolbox/backend/orthogonalization.py`
- 🔧 **Read** `t3toolbox/backend/ranks.py`

This is a substantial, subtle algorithm. Let me prototype it in `/tmp` and validate against the dense oracle before touching the backend, per the project's "verify the math empirically first" rule. Let me also check the `__init__`/imports quickly.

- 🔧 **Bash** — Confirm helper availability and frontend imports

Now let me write the prototype.

- 🔧 **Write** `/tmp/proto_swap.py`
- 🔧 **Bash** — Run swap prototype against dense oracle

Excellent — the algorithm is **correct**: the exact path (generous ranks) and stacked+max-rank both hit machine precision (~1e-15), and rtol truncation respects its tolerance. The cap-2 case passes the ≤2× bar (0.764 ≤ 2·0.503=1.006) but is ~1.5× the optimal. Let me check whether capping transient swap bonds is what's degrading quality.

- 🔧 **Bash** — Compare transient-capped vs uncapped swap quality

Capping transients makes no difference here — the ~1.5× gap is inherent per-step accumulation, not the cap. Let me compare against method (b) on the same case to see if (b) does better (which would flag a suboptimality in my swap):

- 🔧 **Bash** — Compare existing methods' truncation quality on the same case

Found it: my swap (0.764) is notably worse than `inplace_fused` (0.522) and the optimal (0.503). The structural cause: in the swap method the **bond truncations during bubbling see the un-Tucker-truncated physical leg of the incoming `b` core** (its Tucker compression only happens at the contract, the last step) — an ordering tension that pure-TT TTM doesn't have. Let me test whether a final cheap t3svd cleanup (bonds already ≤ R_tt, so memory preserved) recovers optimal quality:

- 🔧 **Bash** — Test final t3svd cleanup recovers quality

Clear diagnosis. Truncating during the swaps loses information that a final cleanup can't recover (0.764 → still 0.764), while swap-with-generous-ranks + cleanup hits optimal exactly (0.503). The root cause is structural: when mode 0's bubble sets the final bonds, the incoming `b_0` core still carries its **full, un-Tucker-compressed** physical leg (its KR compression only happens at the final contract). Pure-TT TTM doesn't have this because there's no Tucker leg.

Let me check the rtol case against Decision 1 (for a given rtol, (c) must not keep larger ranks than (a)):

- 🔧 **Bash** — Compare (a) vs (c) ranks for fixed rtol (Decision 1 guard)

The exact and tight-tolerance behavior is solid; the quality gap shows up only under aggressive truncation. Let me test one more idea — pre-down-orthogonalizing the Tucker factors (t3svd's first step, folding Tucker weight into the central TT) before swapping — to see if it reduces the "full leg" pollution:

- 🔧 **Bash** — Test pre-down-orthogonalization and oversample+cleanup

Two clear findings:
1. **Pre-down-orthogonalizing** the inputs (t3svd's standard first step — folds Tucker weight into the central TT) is a **free** quality win: 0.764 → 0.611 (1.52× → 1.21×), no memory/asymptotic cost, exact path preserved.
2. Oversampling internal bonds + cleanup is a tunable quality/memory knob (x3 → optimal) but sacrifices the memory win — a future option, not for now.

Let me fold pre-down-orth into the prototype and re-verify the full matrix:

- 🔧 **Edit** `/tmp/proto_swap.py`
- 🔧 **Bash** — Re-run full prototype with pre-down-orth

The prototype is **correct and complete** — let me summarize where it landed before I commit to the final backend code, because one genuine design decision surfaced.

## Status: `t3m_swap` prototype works

Verified against the dense oracle (`/tmp/proto_swap.py`):
- **Exact path** (no truncation / generous ranks): machine precision (~1e-15), all structures, **stacked included**.
- **rtol truncation**: respects tolerance (rtol=1e-2 → 7.7e-3 error).
- **max-rank truncation** (cap 2,2): passes the agreed ≤2× bar.

The algorithm matches the handoff: concatenate `[a_0..a_{d-1}]` + reversed `[b_{d-1}..b_0]`, then for each mode (d-1→0) bubble `b_i` to `a_i` via gauge-managed truncating swaps and contract (COPY-merge `W_i` + joint weighted-Tucker truncation). I track the orthogonality center explicitly (shuttle → bubble-with-center → contract) so every swap/contract SVD is locally optimal — this is what makes the exact path hit machine precision and the gauge "hard part" correct.

## The finding worth your call

Under **aggressive** truncation, method (c) is **~1.2× the optimal error** (cap 2,2: swap 0.611 vs optimal 0.503), whereas (a)=optimal and (b)=1.04×. This is **structural, not a bug**: when the last mode's bubble sets all the internal bonds, the incoming `b` core still carries its **full, un-Tucker-compressed** physical leg (its KR-compression only happens at the final contract). Pure-TT TTM has no Tucker leg, so it doesn't hit this. I already folded in one **free** fix — pre-down-orthogonalizing the Tucker factors (t3svd's standard first step), which improved 1.52×→1.21× at zero memory/asymptotic cost.

I could close the remaining gap with an **oversample-internal-bonds + final cleanup** knob (cap swaps at ~3× target → near-optimal), but that inflates the internal bonds and **erodes (c)'s `O(r̃²)` memory advantage — which is its entire reason to exist** (the `r≫d` regime).

- 🔧 **AskUserQuestion**

What would you like to clarify? Happy to walk through any part of this — e.g. the quality-gap mechanism, how the oversample knob would actually trade memory for accuracy, the per-position `max_tt_ranks` issue, or anything in the prototype. Let me know what's unclear or what extra context you have, and I'll reframe the decisions accordingly.


## 🧑 Nick

I like the extra oversampling knob. However, I want to talk about the algorithm first, to make sure that what you olan to implement is what is in my mind as well. One thing in particular is orthogonalization. My thinking is that a core which is being truncated should first have everything surrounding that core be orthogonal relative to it. Is that what you have in mind, and do you think that is a good decision?

## 🤖 Claude

Yes — that's exactly what I implemented, and yes, I think it's the right decision. Let me state it precisely so we're aligned, because the swap method has two distinct kinds of truncation and the principle has to hold for both.

## The principle = mixed-canonical gauge centered at the active core

For TT/MPS, truncating a core (or the bond at a core) is **globally optimal in Frobenius norm only if that core is the orthogonality center**: everything to its left is left-orthogonal (the left-bond+physical unfolding is an isometry) and everything to its right is right-orthogonal. Then the SVD's discarded singular values *are* the error norm (Eckart–Young lifted through the isometric environment). Without it, you're truncating singular values of a non-orthonormalized unfolding, which don't correspond to the actual error — so you'd silently lose more than the singular values suggest. So this isn't just nice-to-have; it's what makes "drop the small singular values" mean what we want.

This is precisely why I track the orthogonality center **explicitly** through the whole sweep rather than re-orthogonalizing blindly. Concretely, in my prototype:

- **Swap truncations (during bubbling).** Before bubbling `b_i` through the merged block, I *shuttle* the center to the right end of the block, then bubble with the singular values pushed onto the moving `b_i` so the **center travels with it**. Every swap therefore has the center *at the swap pair* — left side left-orth, right side right-orth — so each bond truncation is optimal for the tensor as it stands.
- **Contract truncation (Tucker rank).** After bubbling, the center sits at the `(a_i, b_i)` pair, so the merged core is the center with orthogonal surroundings. The weighted-Tucker SVD reshapes to `(k, αL·γ)` and truncates the Tucker leg weighted by **both** bond environments at once — which is only the correct weighting because both `αL`'s left and `γ`'s right are orthogonal. (This is the same pattern as `t3svd`'s `down_svd_tt_core`.)

That gauge discipline is exactly what makes the no-truncation path reproduce the product to machine precision (~1e-15) — a wrong gauge would show up there immediately.

## The one honest caveat

The gauge makes **each step** optimal, but a *sequence* of locally-optimal SVD truncations is only **quasi-optimal**, not globally optimal — the standard greedy-sweep phenomenon. The swap method does `O(d²)` truncation steps (vs `O(d)` for a plain rounding sweep), and the intermediate representations carry the traveling `b` core's not-yet-compressed Tucker leg, so the per-step-optimal sequence still lands at ~1.2× the true optimum under aggressive truncation. **That residual is not a gauge failure** — it's why the oversample+cleanup knob helps: it defers the final, decisive truncation to a cleanup pass where the Tucker legs are already compressed and there are far fewer steps.

One cost note so it's on the table: maintaining the center requires re-shuttling it across the merged block before each bubble (`O(d)` per mode, `O(d²)` total of non-truncating QR-grade SVD steps) — that stays comfortably within the method's `O(d²r³)` budget.

So: same mental model as yours, and I'd keep the explicit center-tracking (it's cleaner and cheaper than re-orthogonalizing sub-chains from scratch before each op). Does that match what you had in mind — and do you want me to go ahead on that basis, folding in the oversample+cleanup knob you liked?


## 🧑 Nick

Ok, excellent. I do want the oversample+cleanup knob. However first, I want you to take a step back and assess the algorithm. Is this the "correct" generalization of TTM to Tucker tensor trains? Are there any improvements you would suggest? Imagine if the researchers who wrote the TTM paper decided to generalize it to T3s. Is this the algorithm they would come up with?

## 🤖 Claude

Good question to force. Let me reason about it from the math, separating what's a faithful port of TTM from what's genuinely new in T3.

## What TTM actually is, and what ports cleanly

TTM computes a Hadamard product by viewing it as the **diagonal of the outer product**: concatenate the two TTs into a length-`2d` chain (= the outer product `A ⊗ B`), reverse the second so matching modes meet in `d(d-1)/2` swaps, then at each meeting apply a **copy/δ tensor** that collapses the two physical legs `x_i, x_i'` onto their diagonal `x_i = x_i'`. Truncate throughout, in mixed-canonical gauge, so you only ever SVD single-`r` bonds → `O(d²r³)`, `O(r̃²)`.

Three pieces of that port to T3 essentially unchanged, and I'm confident the TTM authors would write them the same way:
- **concatenate + reverse + the `d(d-1)/2` swap schedule** — identical (it's a property of the permutation, nothing tensor-specific).
- **gauge-centered truncating swaps** — identical; this is just TT/MPS truncation done right.
- **the "copy tensor" at the merge becomes the Khatri–Rao** `W_i = U^A_i ⊙ U^B_i`. This is the clean conceptual mapping: TTM's diagonal-in-`x_i` *is* "evaluate both Tucker factors at the same `x_i`," which is exactly the row-wise KR. So the merge is the faithful T3 analog of the δ-tensor.

So the **skeleton is the correct generalization** — the one they'd arrive at.

## The one genuinely new thing in T3 — and why it bites

In pure TT, the physical leg `N_i` is **invariant**: the diagonal merge keeps it at `N_i`, forever, no compression. So the only truncation is the bonds, and centered swaps make that optimal. **No tension.**

In T3 the KR introduces a *second* truncation that pure TT doesn't have: the merged Tucker rank `n^A_i·n^B_i` compresses to `ñ_i ≤ N_i`. And here's the crux — **that compression couples the two cores** (`M_i` maps the *pair* `(n^A_i, n^B_i)`, and a KR's left singular vectors don't factor), so it can only happen when the two cores meet at the merge. Meanwhile the bonds want to be truncated *after* all merges: while `a_0` and `b_0` sit split across the chain, the a–b correlation for mode 0 is carried by the *bonds*; once they merge it becomes internal to `m_0` and stops burdening the bonds. Truncating bonds while the last mode is still split = truncating a representation with **artificially inflated** bond ranks. That's the ~1.2× I measured, and it is **inherent to T3**, not a gauge bug (the gauge is provably right — the exact path is machine-precise).

This is the wall the TTM authors would hit too. And the principled way off it is forced:

> Use the swaps only to bring cores together with **loosely-bounded** intermediate ranks (memory control), and defer the **decisive** truncation to a final rounding pass — run *after* all merges, when every Tucker leg is already compressed and there are `O(d)` steps instead of `O(d²)`.

That is **exactly the oversample + cleanup knob**, which reframes it: it isn't quality polish bolted on — it's the **essential T3-specific completion** of the algorithm, exposing the genuine memory↔quality Pareto frontier that pure-TT-TTM doesn't have. The naive "truncate hard in-place during swaps" version (what the plan literally says today) is the `oversample = 1` corner; it's correct but it's the *worst* point on that frontier.

## Concrete refinements I'd make to the plan

1. **Make oversample+cleanup intrinsic, not optional.** Swap process bounds *both* Tucker and TT ranks at `oversample × target` purely for memory; a final **`t3svd` round at the exact target** does all decisive truncation. Confirmed empirically: `oversample≈3` → 0.504 vs optimal 0.503.
2. **The cleanup is `t3svd` — which we already have** (the handoff's designated rounder). This *unifies three problems into one*: it fixes the quality gap, it's the memory↔quality knob, **and** it cleanly honors a per-position `max_tt_ranks` **sequence** (the cleanup sweep truncates per-bond optimally) — so the second question I asked earlier dissolves; no separate enforcement sweep needed.
3. **Memory advantage survives.** In the `r≫d` regime `r̃ ≪ r`, so `O((k·r̃)²)` with `k≈2–3` is still vastly below the `O(r²)` full product. A modest default (I'd suggest `oversample=2`) keeps near-optimal quality *and* `O(r̃²)` memory; `oversample=1` is available as the max-memory-savings / lowest-quality extreme.
4. **Keep the centered gauge in-process anyway** — even though cleanup does the decisive cut, centered in-process truncation is what guarantees the oversampled bonds don't discard info the cleanup needs.

**Bottom line:** the skeleton is the right algorithm; the honest "T3-correct" version is *swap-with-loose-bounds → KR-merge → `t3svd` cleanup*, with `oversample` as the principled control. I'd update `docs/t3m_plan.md`'s method-(c) description to say this rather than "truncate per-site during swaps."

Two things I'd want your call on before I build it: the **default `oversample`** value (I lean `2`), and the **rtol/atol** path — for max-rank, oversample is just `k × cap`, but for tolerances "oversample" has to mean *loosen the in-process tol and apply the real one at cleanup* (e.g. in-process `rtol/oversample`, exact `rtol` at cleanup). Does that framing match your intent, or were you picturing the knob as max-rank-only?


## 🧑 Nick

This makes sense to me. I would like the default to be no oversampling. The user can choose to use it if they like. We can add a note in the documentation suggesting a modest oversampling value of 2 as a good default the user might consider. I agree with your idea about using tolerance loosening as the oversampling mechanism for rtol/atol. Also, we need to have a coherent plan with both max rank and tols are used. Now, taking a slight detour before implementing. Perhaps one way to think about generalizing TTM is as a special case of the generalization to general Hierarchical Tucker tensor networks. Does the proposed plan make sense in this lens? I don't intend to generalize to Hierarchical tucker, I just want to think about the connections as a thought experiment.

## 🤖 Claude

Great lens — it actually sharpens the whole picture and (reassuringly) confirms the plan is the faithful instance rather than an ad-hoc construction. Let me reason through it.

## Where T3 sits inside HT

A Hierarchical Tucker tensor = a dimension tree with **leaf frames** `U_i[n_i, N_i]` and **transfer tensors** `B_t[n_{t1}, n_{t2}, n_t]` at internal nodes. The familiar formats are points in this space:

- **TT** = HT on a *linear/caterpillar tree* with **trivial leaf frames** (`U_i = I`, `n_i = N_i`).
- **Tucker** = HT on a *star* (one root transfer tensor = the d-way core, d leaf frames).
- **T3** = HT on a *linear tree* with **nontrivial leaf frames**. The leaf frames are the Tucker factors; the internal linear chain of transfer tensors is the core-TT.

So T3 is exactly "TT, but with the leaf frames switched on." Hold that thought — it turns out to be the whole story.

## Hadamard in HT, and TTM as its linear instance

The Hadamard product of two same-tree HT tensors is, universally: **leaf frames → Khatri–Rao** (`W_i = U^A_i ⊙ U^B_i`), **transfer tensors → Kronecker** (`B^A_t ⊗ B^B_t`). Every rank blows up multiplicatively; you then round back down. That's the general fact our merge instantiates exactly.

TTM's swap machinery is then one specific way to do this round for the *linear* tree. Conceptually: `A ⊗ B` is itself a valid HT tensor on the **two trees joined at a new root**; the Hadamard is recovered by **merging each pair of matching leaves** (the copy-tensor/KR). To keep intermediate ranks small you want matching leaves *close* in the joined tree. For a linear tree "close = adjacent," so you reorganize by **swaps** (and the reversal minimizes total swap distance). For a *general* tree, "swaps" generalize to SVD-based subtree/gauge moves — but the principle ("merge matching leaves of `A⊗B` by local restructuring in mixed-canonical gauge, truncating throughout") is tree-agnostic. T3 being linear, the chain-and-swap machinery is the correct instance, with no freedom to do otherwise.

## The payoff: the tension *is* "nontrivial leaf frames"

Here's where the lens earns its keep. Our memory↔quality tension came from the **leaf (KR) compression coupling the two copies** of a mode, so it can only happen at the merge, while the internal-bond truncations want all leaves merged first. In the HT lens that's not a T3 quirk — it's a property of **having nontrivial leaf frames at all**:

- **TT has trivial leaves**, so there's *no leaf compression* → no coupling → no tension → truncate-hard-in-place is already optimal. That's *why* the original TTM gets away with per-swap truncation.
- **T3 (and any leaf-bearing HT)** has the leaf-KR compression → the coupling → the tension → you must bound intermediate ranks loosely and defer the decisive truncation to a final native round.

So **oversample + cleanup is not a patch — it's precisely what nontrivial leaf frames demand**, and it's the same resolution for general HT. The original TTM is the degenerate (trivial-leaf) corner where `oversample` doesn't matter. That's a satisfying place to land: our refinement is the *generic* algorithm and TTM is the special case, not the other way around.

## The lens also retro-justifies several choices

- **Down-orthogonalizing the Tucker factors first** = the standard HT step "orthogonalize leaves toward the root." It wasn't a quality hack; it's the canonical HT preprocessing.
- **Center-at-the-truncated-node gauge** = HT mixed-canonical orthogonalization, verbatim.
- **Cleanup via `t3svd`** = "round with the format's native rounder," which is what HT would do at every level.
- **The `O(d²)` swap count** is revealed as a *linear-tree artifact* (dragging merged leaves past each other). A balanced HT tree would merge in `O(d log d)`-ish — but T3 is linear *by definition*, so `O(d²)` is intrinsic, which is consistent with positioning (c) for small `d`, large `r`.

**Verdict:** yes, the plan is fully coherent in the HT lens. It's the faithful "linear-tree-with-leaf-frames" instance of HT-Hadamard-by-leaf-merging, and the oversample+cleanup refinement is exactly the ingredient that the leaf frames (the "Tucker" in Tucker tensor train) require — the thing pure TT lets you skip.

When we come back from the detour I'll bring the concrete, coherent oversample semantics for **max-rank and tolerances used together** (in-process `k×max_rank` *and* `rtol/k, atol/k`, exact targets at cleanup; cleanup applied only when it can help — i.e. `oversample>1` or a non-uniform per-position `max_tt_ranks` that the swaps can't honor bond-by-bond), with default `oversample=1` and a doc note suggesting `2`. Want me to write that up as the updated plan next?


## 🧑 Nick

Oh, this is interesting! So the balanced tree version actually requires way less bubbles. One could imagine a hypothetical approach where one converts a TT (of T3) into HT, then multiplies, then converts back, thereby reducing bubbling substantially! Does this make sense; please check me if this is wrong? Ok, I don't want to do this now, but I do want to save this idea. Can you please write a mathematical note in .tex format that briefly explains what we have talked about here: The TTM algorithm, our T3M generalization, the hypothetical HT method, the connections between these methods, and the concept of converting between formats then converting back in order to reduce bubbling (if that makes sense to you). The note should contain the mathematics, algorithm boxes, and prose explanation of the ideas.

## 🤖 Claude

Let me check your idea carefully, since you asked — and I think it's a genuinely good thought experiment, but for **TT/T3 inputs specifically it probably doesn't pay off**, for a concrete reason worth pinning down:

**The bubble-count reduction is real**, but **TT→balanced-HT conversion inflates ranks.** A contiguous *middle* block `[a,b]` of a TT connects to the rest through **two** bonds, so its HT separation rank is `≤ r_a·r_{b+1} ≈ r²` (prefix/suffix blocks stay `≈ r`; it's the middle nodes that blow up). Consequently the Hadamard product, which is `≈ r²`-compact as a **TT**, is `≈ r⁴`-compact as a **balanced HT**. So converting moves you into a representation where the very object you're computing is *quadratically larger*.

Netting it out by regime (working bond `r̃`):

- **Linear/TT swap:** `O(d²)` bubbles × `O(r̃³)` ⇒ `O(d² r̃³)`.
- **Balanced HT:** `O(d log d)` merges × `O((r̃²)³)=O(r̃⁶)` ⇒ `O(d log d · r̃⁶)`.

Ratio balanced/linear `≈ (log d / d)·r̃³`. In the **`r̃ ≫ d`** regime — exactly where bubbling matters / where (c) lives — `r̃³ ≫ d`, so balanced is **worse**. In the **`d ≫ r̃`** regime method (b) already avoids bubbling entirely, so you wouldn't swap anyway. So there's no regime where it clearly wins for TT/T3 inputs.

**But the meta-principle is sound** ("compute in whatever topology makes the operation cheapest, then convert back") — it just needs the alternative topology to be genuinely more compact *for this data + operation*, which balanced HT isn't for TT-Hadamard. Partial/optimal rebalancing is a legitimate open direction. I'll record all of this honestly in the note.

Let me write it:

- 🔧 **Write** `docs/ttm_t3m_ht_note.tex`

Let me verify it compiles (if a TeX toolchain is available):

- 🔧 **Bash** — Compile the LaTeX note if pdflatex is present

A stray literal `r̃` Unicode char slipped in — let me fix it to use the macro:

- 🔧 **Edit** `docs/ttm_t3m_ht_note.tex`
- 🔧 **Bash** — Recompile note twice to resolve references

Compiles cleanly — 6 pages. Let me clean up the build artifacts (keep only the `.tex` source, since the repo avoids committing generated files):

- 🔧 **Bash** — Remove LaTeX build artifacts, keep only the source

Done. The note is written and verified.

## What I saved

`docs/ttm_t3m_ht_note.tex` — a 6-page math note (compiles cleanly with `pdflatex`; standard packages only). It captures everything from this discussion:

1. **Setup & notation** — TT, T3, and the two combination rules (Khatri–Rao "COPY" at leaves, Kronecker inside), with the exact product structure of two T3s.
2. **TTM** (Algorithm box) — concatenate + reverse + gauge-centered truncating swaps + the copy-tensor (diagonal) merge; and *why* per-swap truncation is optimal for pure TT (invariant physical leg).
3. **T3M** (Algorithm box) — the same skeleton with the merge changed to Khatri–Rao; the **leaf-frame tension** (KR compression couples the operands → can only merge late, but bonds want to truncate after); and the **oversample-then-round** resolution with default `k=1`, doc-suggested `k≈2`, and the rtol/atol-loosening + per-position-cap handling.
4. **The HT view** — TT/Tucker/T3 as instances; Hadamard-on-trees; the lens *retro-justifies* down-orth (= orthogonalize leaves toward root), the centered gauge (= HT mixed-canonical), and the cleanup (= native round); and frames `O(d²)` bubbling as a linear-tree artifact.
5. **Convert–multiply–convert-back** — your idea, recorded with the honest analysis: bubble count `O(d²)→O(d log d)` but rank inflation `r²→r⁴` (middle blocks), net `(log d/d)·r̃³` → worse in the `r̃≫d` regime; the meta-principle preserved, partial rebalancing flagged as open.
6. **Summary table** of (a)/(b)/(c) + the hypothetical balanced-HT row, plus take-aways.

The note is honest about the convert-back idea not paying off for TT/T3 inputs, with the reasoning, so future-you has the full picture rather than just the appealing intuition.

When you're ready to resume implementation, the plan is: backend `t3m_swap` with the gauge-centered swap/merge I prototyped (verified machine-precise on the exact path, stacked-safe), `oversample` parameter (default 1, no cleanup; >1 ⇒ `t3svd` cleanup at exact targets), coherent max-rank + tol semantics (`k×max_rank` and `tol/k` in-process), then wire up the frontend + tests + `docs/t3m_plan.md` update. Want me to write up that concrete implementation plan next, or pick it up directly?


## 🧑 Nick

Write up the concrete implementation plan next

## 🤖 Claude

- 🔧 **Write** `docs/t3m_swap_plan.md`

Written to `docs/t3m_swap_plan.md` — a self-contained, transcribe-ready build plan. Highlights:

**The verified algorithm, backend-adapted.** The `_swap` / `_contract` / `_move_center` helpers and the driver are the xnp/`'...'` form of the prototype that already hit machine precision on the exact path (stacked included). Including the actual code preserves the verified math now that `/tmp` will vanish.

**The coherent oversample rule (the open item).** One uniform rule that handles max-rank and tolerances together:
- in-process: Tucker cap `⌈k·mtr[i]⌉` (per-mode), TT bond cap `⌈k·max(mrr)⌉` (uniform — swaps can't carry per-position identity), tolerances `rtol/k`, `atol/k` (smaller ⇒ keep more);
- one `t3svd` cleanup at the **exact** `(mtr, mrr, rtol, atol)`.

**Cleanup runs iff `oversample>1` OR `max_tt_ranks` is an explicit sequence.** This elegantly makes the cleanup do triple duty: it's the quality recovery for oversampling, *and* it's how a per-position TT-cap sequence is honored exactly (dissolving the earlier "max_tt_ranks sequence" question — no separate enforcement sweep). The fast path (`oversample=1`, scalar/None caps, rtol) does no cleanup.

**Everything else nailed down:** frontend signature/validation/dispatch changes (incl. `oversample<1` and `oversample≠1 with non-swap` → ValueError), the `test_validation` update (drop the `NotImplementedError` for swap), the new tests (`test_swap`, oversample, per-position TT, stacked+oversample, cross-method joint-quality, jit dispatch), edge cases (`d=1`, boundary bonds, stacked rtol⊥), doc updates, and a 4-slice commit ordering with acceptance criteria.

Two small judgment calls baked in (flagging so you can veto): `oversample` is a **multiplicative** factor (your "value of 2" reading), numeric ≥1, ranks via `⌈k·cap⌉`; and `oversample≠1` with a non-swap method **raises** rather than silently no-ops (matches the structural-error house rule).

Want me to start implementing from slice 1, or adjust anything in the plan first?


## 🧑 Nick

commit and push the plan, then start implementing

## 🤖 Claude

- 🔧 **Bash** — Commit and push the plan + math note

Pushed (`bc929058`). Now slice 1 — the backend. Let me add `t3m_swap` + helpers to `t3_linalg.py`:

- 🔧 **Edit** `t3toolbox/backend/t3_linalg.py`
- 🔧 **Edit** `t3toolbox/backend/t3_linalg.py`

Now the frontend `t3m`:

- 🔧 **Read** `t3toolbox/tucker_tensor_train.py`
- 🔧 **Edit** `t3toolbox/tucker_tensor_train.py`
- 🔧 **Edit** `t3toolbox/tucker_tensor_train.py`
- 🔧 **Edit** `t3toolbox/tucker_tensor_train.py`

Now the tests — add `test_swap` and update `test_validation`:

- 🔧 **Edit** `tests/test_t3m.py`
- 🔧 **Edit** `tests/test_t3m.py`

Now let me run the t3m tests and the adjacent suites:

- 🔧 **Bash** — Run test_t3m suite

t3m suite green. Now the adjacent suites that the frontend change touches:

- 🔧 **Bash** — Run tucker_tensor_train and dispatch suites

Let me confirm both passed cleanly (the trailing lines are debug noise):

- 🔧 **Bash** — Check pass/fail summary lines

Slice 1 complete and green (62 tests OK). Let me commit it:

- 🔧 **Bash** — Commit and push slice 1

Slice 1 pushed (`022216a5`). Now slice 2 — oversample + per-position + stacked + cross-method tests. Let me first measure the actual numbers so my assertions are correctly calibrated:

- 🔧 **Bash** — Measure numbers for slice-2 test calibration

Let me check per-position quality with oversample=2 (the oversample=1 per-position case lands right at the 2× edge):

- 🔧 **Bash** — Per-position quality vs oversample

Per-position caps are honored for all `k`; oversample=2 gives 1.04× quality. Calibration confirmed. Let me add the slice-2 tests:

- 🔧 **Edit** `tests/test_t3m.py`
- 🔧 **Bash** — Run test_t3m with slice-2 tests

Slice 2 green (9 tests). Commit:

- 🔧 **Bash** — Commit and push slice 2

Slice 3 — `test_dispatch` jit cases. Let me look at the existing dispatch test patterns:

- 🔧 **Read** `tests/test_dispatch.py`

Let me check `TuckerTensorTrain` has a `to_numpy` for the smoke test:

- 🔧 **Bash** — Check to_numpy/to_jax on TuckerTensorTrain

`to_numpy` exists. Let me add the dispatch jit cases and the smoke check:

- 🔧 **Edit** `tests/test_dispatch.py`
- 🔧 **Edit** `tests/test_dispatch.py`
- 🔧 **Bash** — Run dispatch suite with t3m jit cases
- 🔧 **Bash** — Show the jit error traceback

The new dispatch test surfaced a **latent pre-existing bug**: `t3_mult` (and `t3m_inplace_fused`) infer `use_jax = is_jax_ndarray(x)` where `x` is the `(tucker, tt)` **tuple** — always `False`, so they never dispatch to jax. My `t3m_swap` works (its helpers infer from real core arrays), but the form/fused paths route through buggy `t3_mult`. Let me check `tree_contains_jax` and fix:

- 🔧 **Bash** — Inspect tree_contains_jax

`tree_contains_jax` recurses correctly. Let me fix both buggy inferences and clean up `t3m_swap`'s unused lines:

- 🔧 **Edit** `t3toolbox/backend/t3_linalg.py`
- 🔧 **Edit** `t3toolbox/backend/t3_linalg.py`
- 🔧 **Edit** `t3toolbox/backend/t3_linalg.py`
- 🔧 **Bash** — Verify no stray xnp in t3m_swap and rerun tests

No stray references; dispatch + t3m green (14 tests). Since I changed `t3_mult`'s dispatch (used by `__mul__` and method (a)), let me run the broader suite to be safe:

- 🔧 **Bash** — Run full relevant suite

All 131 green. Commit slice 3 (includes the latent dispatch-bug fix surfaced by the new jit test):

- 🔧 **Bash** — Commit and push slice 3

Slice 4 — docs. Let me update `t3m_plan.md`, `CLAUDE.md`, and `t3m_handoff.md`:

- 🔧 **Edit** `docs/t3m_plan.md`
- 🔧 **Edit** `docs/t3m_plan.md`

Now `CLAUDE.md` — update the T3M TODO bullet:

- 🔧 **Read** `CLAUDE.md`
- 🔧 **Bash** — Find the T3M TODO bullet in CLAUDE.md
- 🔧 **Read** `CLAUDE.md`
- 🔧 **Edit** `CLAUDE.md`

Now mark `t3m_handoff.md` as completed/superseded:

- 🔧 **Edit** `docs/t3m_handoff.md`

Slice 4 — commit the doc updates:

- 🔧 **Bash** — Commit and push slice 4 (docs)

All four slices done and pushed. Let me run the complete test suite as the final acceptance check:

- 🔧 **Bash** — Run the full test suite

Let me also run the backend contractions suite (referenced in CLAUDE.md's full-suite list):

- 🔧 **Bash** — Locate and run contractions test suite

All green — 131 tests + 29 contractions tests pass. **Method (c) `t3m_swap` is complete, tested, and pushed.**

## Summary of what shipped (5 commits on `main`)

| Commit | Slice |
|---|---|
| `bc929058` | Plan + math note (`t3m_swap_plan.md`, `ttm_t3m_ht_note.tex`) |
| `022216a5` | Slice 1: backend `t3m_swap` + helpers, frontend dispatch/validation/docstring, `test_swap` |
| (slice 2) | oversample / per-position / stacked / joint-quality tests |
| (slice 3) | `test_dispatch` jit cases **+ latent jax-dispatch bug fix** |
| (slice 4) | docs marked done (`t3m_plan.md`, `CLAUDE.md`, `t3m_handoff.md`) |

## What it does
- **Algorithm**: down-orthogonalize Tucker factors → concatenate + reverse into a length-`2d` chain → for each mode bubble the matching pair adjacent via **explicit-center gauge-managed truncating swaps** → KR-merge + joint weighted-Tucker truncation. `O(d²r³)` / `O(r̃²)`.
- **`oversample` knob** (default 1): relaxes intermediate ranks/tols by `k`, then one exact `t3svd` cleanup — the principled fix for the Tucker leaf-frame tension. Also what makes a per-position `max_tt_ranks` sequence honored exactly.
- Verified against the dense oracle: machine precision on exact + generous-rank paths (stacked included), within the ≤2× bar truncated (≈1.0× with `oversample=3`), jit-dispatches under jax with max-rank.

## One thing to flag
Slice 3's jit test surfaced a **pre-existing latent bug**: `t3_mult` and `t3m_inplace_fused` inferred `use_jax = is_jax_ndarray(x)` on the `(tucker, tt)` **tuple** — always `False` — so a jax input silently fell back to numpy (and would raise under jit). Fixed to `tree_contains_jax((x, y))`. This means `*` and methods (a)/(b) now actually dispatch to jax for jax inputs, which they didn't before. Purely a correctness improvement (numbers unchanged for numpy), but worth knowing since it touches the existing `__mul__` path.

T3M is now complete end-to-end — all three methods live and tested. Want me to do anything else (e.g. the deferred least-squares fitting tutorial, or move on to another module)?


## 🧑 Nick

Excellent! The bug you found may be present elsewhere. is_jax_ndarray() was an older pattern which is superceeded by tree_contains_jax(). Can you search through the validated modules for other instances of this bug?

## 🤖 Claude

Good instinct — let me sweep all `is_jax_ndarray(...)` calls in the validated modules and classify each by whether its argument is a single array (correct) or a collection/tuple (the bug — always returns `False`).

- 🔧 **Bash** — List all is_jax_ndarray calls outside deferred layers

Most are the correct iterate-over-cores pattern. Let me read context for the suspicious ones — `t3_linalg.py:38` (`t3_add`), `:558` (`t3_plus_scalar`), and the `common.py` ones:

- 🔧 **Bash** — Read context for suspicious is_jax_ndarray calls

Clear classification. The `common.py` calls (272/295/316) are all guarded by `if is_ndarray(...)` so the arg is a real array there — correct. The iterate-over-cores and single-array (`cores[0]`, dense `T`) calls are all fine. 

**Two genuine bugs remain** (collection arg → always `False`), both in `t3_linalg.py`:
- `t3_add` (line 38) — `is_jax_ndarray(x)` on the `(tucker, tt)` tuple
- `t3_plus_scalar` (line 558) — same, **and** `use_jax` is then dead (never passed to `t3_ones`, which always builds numpy)

Let me confirm the frontend paths that reach these, then fix:

- 🔧 **Bash** — Find frontend callers of t3_add / t3_plus_scalar

`__add__` (both T3+T3 and T3+scalar) hit the buggy dispatch — silently downcasting jax→numpy eagerly, raising under jit. Let me fix both:

- 🔧 **Edit** `t3toolbox/backend/t3_linalg.py`
- 🔧 **Edit** `t3toolbox/backend/t3_linalg.py`

Now add dispatch coverage so these don't regress — `__add__` (T3+T3 and scalar) and `__sub__`:

- 🔧 **Edit** `tests/test_dispatch.py`
- 🔧 **Bash** — Run dispatch + tucker tests with add/sub fixes

62 tests OK — the jit cases prove `+`/`-` now dispatch to jax (they'd have raised on a tracer otherwise). Let me commit the fix:

- 🔧 **Bash** — Commit and push the dispatch bug fixes

Swept and fixed. Here's the full audit of `is_jax_ndarray(...)` across the validated modules:

## Bugs (collection arg → always `False` → silent numpy fallback)

| Location | Status |
|---|---|
| `t3_linalg.t3_mult` | fixed earlier (slice 3) |
| `t3_linalg.t3m_inplace_fused` | fixed earlier (slice 3) |
| **`t3_linalg.t3_add`** (T3+T3 `+`) | **fixed now** |
| **`t3_linalg.t3_plus_scalar`** (T3+scalar `+`) | **fixed now** (+ `use_jax` was dead — `t3_ones` always built numpy; now threaded) |

That's the complete set of the collection-arg bug in the validated layer. The impact was the same each time: a jax input was silently downcast to numpy (eager) or raised under jit. Now `*`, `+`, `-`, and all three `t3m` methods dispatch correctly — with `test_dispatch` jit guards on each.

## Everything else — correct, left as-is

- **iterate-over-cores** (`any([is_jax_ndarray(c) for c in x[0]+x[1]])`): `t3_inner_product_t3`, `t3_norm`, all of `t3_orthogonalization`, most of `t3_operations`, `orthogonalization`. This is the right per-leaf pattern.
- **single-array checks**: `linalg.*` (operate on actual arrays), `dense_t3svd` (dense `T`), my `t3m_swap` helpers (core arrays `GL`/`GR`).
- **guarded by `is_ndarray()` first**: `common.xcat/xappend/xprepend` — the arg is provably an array at that point.
- `tree_contains_jax`'s own base case (`common.py:340`).

## One thing to flag, not a bug

`t3_operations.change_tucker_core_shapes` / `change_tt_core_shapes` use `is_jax_ndarray(cores[0])` — a **single-leaf** check. It's *correct* under the library's homogeneous-cores invariant (all cores same backend), so I left it. If you'd prefer the more robust `tree_contains_jax(cores)` there too (defends against a hypothetical mixed-backend tuple), say the word — it's a trivial, no-risk migration, but it's correct as written so I didn't change it unprompted.

The deferred uniform/weighted layers (`ut3_*`, `ubv_*`, `wt3_*`, `OLD_*`) were **not** swept — they still thread `use_jax` the old way and will be migrated when those layers are repaired.


## 🧑 Nick

migrate change_tucker_core_shapes and change_tt_core_shapes too

## 🤖 Claude

I'll migrate both to `tree_contains_jax`. Let me make the change:

- 🔧 **Edit** `t3toolbox/backend/t3_operations.py`

Let me re-read — I need to replace the existing `use_jax` line, not add a duplicate:

- 🔧 **Read** `t3toolbox/backend/t3_operations.py`

I duplicated the lines. Let me remove the old pair:

- 🔧 **Edit** `t3toolbox/backend/t3_operations.py`

Now `change_tt_core_shapes`:

- 🔧 **Edit** `t3toolbox/backend/t3_operations.py`

Let me find the callers to verify the change and consider dispatch coverage:

- 🔧 **Bash** — Find callers of the changed functions
- 🔧 **Read** `t3toolbox/tucker_tensor_train.py`

`resize` is the validated caller (static shapes → jit-able). Let me add a dispatch test for it:

- 🔧 **Edit** `tests/test_dispatch.py`
- 🔧 **Bash** — Run dispatch + tucker tests after resize migration

62 tests OK. Commit the migration:

- 🔧 **Bash** — Commit and push the migration

Done. Both `change_tucker_core_shapes` and `change_tt_core_shapes` now use `tree_contains_jax(cores)` instead of the single-leaf `is_jax_ndarray(cores[0])`, with a jit dispatch test on `resize` (their validated caller) as a regression guard. Full dispatch + tucker suites green (62 tests), pushed.

That completes the `is_jax_ndarray` sweep of the validated layer: every collection-arg bug fixed (`t3_mult`, `t3m_inplace_fused`, `t3_add`, `t3_plus_scalar`) and the two single-leaf idioms migrated. The remaining `is_jax_ndarray` calls are all genuinely single-array checks (in `linalg.*`, `dense_t3svd`, the `t3m_swap` helpers, and the `is_ndarray`-guarded `common.x*` helpers) or the per-leaf `any([... for c in cores])` pattern — all correct. The deferred uniform/weighted layers remain on the old `use_jax`-threading pattern, to be migrated when those layers are repaired.


## 🧑 Nick

Now I would like to do some stylistic cleanup on the validated modules. The house style for function arguments is to have the function name on the first line, then the arguments on the next lines, one line per argument, then the final paranthesis and return type on yet another line. For each argument, a comment explaining its structural properties (shape, length, ...), or if those are too complicated a note on what it is, is placed after the comma. The types and comments are all vertically aligned. For a reference on this style, you can look at how it is done in probing.py. The idea is that the comments after the comma serve as a kind of "type information". The user who jumps to the function definition immediately understands what shapes are needed, withoug having to read through the documentation. I want you to try to understand the principles here, and talk it over with you. I am open to pushback or feedback on this style

## 🤖 Claude

Let me study the reference carefully before forming an opinion.

- 🔧 **Bash** — Read probing.py header and first functions
- 🔧 **Bash** — Read more probing.py signatures and bodies

Let me see a denser internal function with a return tuple and body locals:

- 🔧 **Bash** — List function defs and return lines in probing.py
- 🔧 **Bash** — Read compute_xis through compute_dxis signatures+bodies

I've internalized it. Let me play back the principle, then give you honest feedback — including where I'd push.

## The principle, as I read it

**In array code, the nominal type is nearly useless — the shape *is* the type.** Every argument is `NDArray` (or `Union[Sequence[NDArray], NDArray]`), so Python's type system tells you almost nothing. The trailing comment recovers the real contract: `len=d`, `elm_shape=W+C+(rLi,)`, etc. So the comment isn't a comment in the usual sense — it's the type annotation that the language can't express. That reframing is the whole thing, and it's correct.

The mechanical rules I extracted (tell me if I've missed any):
1. `def name(` on its own line; each arg on its own line as `name: type, # contract`; closing `) -> ...:` with the **return shape(s)** on their own line(s), return tuples expanded one-element-per-line even when they'd fit.
2. Three columns — **arg name**, **type annotation**, **`#` shape-contract** — all **vertically aligned**, so the signature reads as a table.
3. The contract uses the shared shape vocabulary (`C`/`W`/`K` blocks, lowercase single axes, `elm_shape`, `len=`) — the same vocabulary as the body-local suffixes (`mu_WCa`) and the contraction names (`WCa_Caib_WCi_to_WCb`). One vocabulary across signature, body, and helper names.
4. Payoff: jump-to-definition answers "what shapes does this need?" instantly, which is where people actually look.

I think this is genuinely the right call for this codebase — not a quirk to tolerate. I wouldn't change the core of it.

## Where I'd push (honest costs, none fatal)

- **Unchecked drift.** These "types" aren't verified by anything — a comment can silently lie after a refactor (you've already been bitten by stale doctest outputs from an earlier draft). The mitigation is tests + review, but it's worth being clear-eyed that this is *documentation styled as types*, not types. I'm **not** proposing runtime shape-asserts (against the minimal ethos) — just flagging that the convention's power and its weakness are the same thing.
- **Alignment churn.** Hand-maintained vertical alignment means one longer arg name re-flows the whole block (bigger diffs), and it's why you can't run an autoformatter. You've already opted out of Black for the uppercase names, so this is a consistent stance — just a real, ongoing tax. Accepted, in my view.

Both are inherent and I'd accept them. I don't have a fundamental objection.

## Two things worth settling *before* I sweep

1. **Micro-grammar consistency.** Even the reference drifts on separators/order: `# len=d, elm_shape=W+(Ni,)` (comma) vs `# len=d. elm_shape=C+(nUi,Ni)` (period). If the goal is a scannable table, the cell format should be uniform. I'd standardize on one form — my suggestion: **`# len=d, elm_shape=...`** (comma; `len` first, then `elm_shape`; `# <note>` for non-arrays). Cheap to apply as part of the cleanup; want me to?

2. **Division of labor with the numpydoc docstring.** `probe_t3` carries shapes in *both* the trailing comments *and* the `Parameters`/`Returns` docstring — two sources that can diverge. They do serve different surfaces (jump-to-def vs Sphinx). My lean: scope **this** cleanup to **signatures only** (don't touch docstrings), and separately decide later whether docstrings should stop repeating shapes. Agree?

## Proposed approach

Rather than sweep all validated modules at once, I'd do **one module first** — I'd pick `manifold.py` or `t3_linalg.py` (smaller, mixed conformance) — bring its signatures fully to style, you review the result against `probing.py`, and once you're happy I roll the exact same treatment across the rest. Which module do you want as the pilot, and do you want me to also fix the body-local-suffix naming where it's missing, or strictly signatures-and-returns for now?


## 🧑 Nick

Excellent, I agree. Now that you understand, I want you to write a note about this, so that a future version of yourself will understand the principle without having to have this conversation again. Also, this rule should only be understood to apply "within reason". I'm sure there are circumstances where it doesn't make sense, which can be handled differently on a case by case basis. What is important is the principles. We add comments after the functions to address the deficiency in Python's type system, and make it easy for people using the code to quickly understand what the arguments are.

## 🤖 Claude

I'll capture this as a canonical doc, then point to it from `CLAUDE.md` and the existing memory so future-me finds it without re-deriving. First the doc:

- 🔧 **Write** `docs/signature_style.md`

Now point to it from `CLAUDE.md`'s code-style section:

- 🔧 **Read** `CLAUDE.md`
- 🔧 **Read** `CLAUDE.md`
- 🔧 **Edit** `CLAUDE.md`

Now update the existing memory to point at the canonical doc:

- 🔧 **Read** `/home/nick/.claude/projects/-home-nick-repos-T3Toolbox/memory/signature-args-one-per-line.md`
- 🔧 **Edit** `/home/nick/.claude/projects/-home-nick-repos-T3Toolbox/memory/signature-args-one-per-line.md`

Update the memory index hook:

- 🔧 **Edit** `/home/nick/.claude/projects/-home-nick-repos-T3Toolbox/memory/MEMORY.md`
- 🔧 **Read** `/home/nick/.claude/projects/-home-nick-repos-T3Toolbox/memory/MEMORY.md`
- 🔧 **Edit** `/home/nick/.claude/projects/-home-nick-repos-T3Toolbox/memory/MEMORY.md`

Now commit the doc + CLAUDE.md pointer (the memory files live outside the repo):

- 🔧 **Bash** — Commit and push the signature-style note

Done. The principle is now captured in three places so future-me can't miss it:

- **`docs/signature_style.md`** — the canonical write-up: the principle (the trailing comment *is* the type, because `NDArray` says nothing and shape is the real contract; the two reader-facing goals), the mechanics, the `# len=d, elm_shape=...` micro-grammar, and an explicit **"within reason"** section (trivial scalars, pathological `Union`s, throwaway closures handled case-by-case; the unverified-comments and hand-alignment costs accepted on purpose).
- **`CLAUDE.md`** code-style bullet — rewritten to lead with the principle and link the doc (so it's in the auto-loaded context every session).
- The existing **memory** (`signature-args-one-per-line.md` + the `MEMORY.md` index hook) — updated to "the comment IS the type" and to point at the canonical doc.

The note deliberately foregrounds what you emphasized: it's not a rigid rule but a principle — fill the deficiency in Python's type system and make arguments quick to understand — with deviations expected where the mechanics would fight those goals.

Whenever you'd like, we can pick the pilot module (`manifold.py` or `t3_linalg.py`) and do the actual signature cleanup against this reference.


## 🧑 Nick

Perfect. Your document captures the idea as I see it. Let's start on manifold.py. You can go ahead and make the changes, tell me the lines/functions where you made the changes. Then I will review them and we can talk about it.

## 🤖 Claude

I'll review every signature in `manifold.py` against the style and fix the ones that need it. Let me read the file.

- 🔧 **Read** `t3toolbox/manifold.py`

This is the calibration pilot, so I'll apply the style to the clearly non-conforming (multi-arg, crammed) signatures and add missing shape/role comments, while leaving self-documenting single-arg methods inline ("within reason"). Let me make the edits.

- 🔧 **Edit** `t3toolbox/manifold.py`
- 🔧 **Edit** `t3toolbox/manifold.py`
- 🔧 **Edit** `t3toolbox/manifold.py`
- 🔧 **Edit** `t3toolbox/manifold.py`
- 🔧 **Edit** `t3toolbox/manifold.py`
- 🔧 **Edit** `t3toolbox/manifold.py`

Let me read back the edited regions to verify alignment and that nothing's malformed:

- 🔧 **Bash** — Locate edited signatures
- 🔧 **Bash** — Show edited signatures with whitespace markers

Let me verify the `#` columns are actually aligned (hand-alignment is error-prone):

- 🔧 **Bash** — Check # comment column positions in edited signatures

`#` columns align cleanly. Let me check `unit`, `randn_like`, and `manifold_dim`:

- 🔧 **Bash** — Check alignment in unit/randn_like/manifold_dim

All aligned. Let me confirm the file is syntactically valid and the manifold tests still pass (signature-only edits, but let me be sure):

- 🔧 **Bash** — Import-check and run manifold tests

Done — manifold.py signatures cleaned up. **Not committed yet**; import works and `test_manifold` is green (37 tests). Here's exactly what I changed, for your review:

## Changed (broken out to one-arg-per-line + shape/role comments)

| Function | Line | What |
|---|---|---|
| `manifold_dim` | 28 | added a structural comment on `s` (`(shape, tucker_ranks, tt_ranks)`) and a comment on the `int` return |
| `T3Tangent.from_vector` | 236 | was one crammed line → 3 args; typed `flat: NDArray` (+ "1D variation DOFs" note), commented `tangent_stack_shape` |
| `T3Tangent.random_orthogonal` | 330 | 7 args across 2 lines → broken out; shape comments on `shape`/`tucker_ranks`/`tt_ranks`, role comments on the two stacks |
| `T3Tangent.unit` | 351 | 2 args → broken out; commented `index` with its `(use_tt_coordinate, i, within_index)` structure |
| `T3Tangent.randn_like` | 371 | awkwardly wrapped → 2 clean lines; commented `tangent` |
| `T3Tangent.allclose` | 485 | 4 args → broken out; commented `other` |

All `#` columns verified aligned within each function.

## Two judgment calls I want your eye on

1. **I added missing type annotations** (`flat: NDArray`; `shape/tucker_ranks/tt_ranks: typ.Sequence[int]`; `tangent_stack_shape: typ.Tuple[int,...]`) where args were previously untyped. That's slightly beyond "just comments." Tell me if you'd rather I keep edits to layout+comments only and leave annotations alone.

2. **I left these inline ("within reason")** because each is self-documenting (self + one fully-typed/self-explanatory arg, no shape gap): `zeros_like(tangent)`, `load(file, use_jax=False)`, `is_orthogonal(self, atol)`, `is_gauged(self, atol)`, `transport(self, new_basis)`, `sum_tangents(self, axis)`. My operative rule was: **break out when there's ≥2 real args or a genuine shape/structure contract to state; leave self-documenting single-arg methods inline.** If you'd rather the file be *uniformly* one-arg-per-line regardless, I'll break these out too.

## Left untouched (already conforming)
`probe`, `probe_transpose`, `apply`, `entries`, `apply_transpose`, `entries_transpose`, `to_dense`, `to_t3`, `retract`, `zeros`, `randn`, `project`, `project_dense_onto_tangent`. One nit I noticed but didn't change: in the transpose methods `sum_over_probes: bool = False,` lacks a trailing comment while its sibling args have them — I can add a one-line note to fill the column if you want.

How does this calibrate against what you had in mind?


## 🧑 Nick

Excellent. I have reviewed manifold.py and agree with your edits in all instances. For the questions, 1. Good, please clean the annotations up as you go, if it is clear what they should be. 2. Agree with how you did it. For the nit: yes, you can add a one line note for sum_over_probes. Ok, now let's move on to t3_linalg.py

## 🤖 Claude

Nick approved. First let me finish the manifold.py nit (the `sum_over_probes` notes), then commit manifold.py, then move to t3_linalg.py. Let me see the three transpose signatures:

- 🔧 **Bash** — Locate sum_over_probes lines and their comment columns
- 🔧 **Bash** — Read the three transpose signatures

Nick wants the `sum_over_probes` note added. I'll do that for all three transpose methods — and while there, fix a pre-existing alignment slip in `apply_transpose`/`entries_transpose` (their `sum_over_probes:` name is wider than the `c`/`ww` column, so types didn't share a column) and correct `probe_transpose`'s `ztildes` shape to `W+K+C` (matches its own docstring). Let me edit the three blocks:

- 🔧 **Edit** `t3toolbox/manifold.py`
- 🔧 **Edit** `t3toolbox/manifold.py`
- 🔧 **Edit** `t3toolbox/manifold.py`

Let me verify the alignment of the three transpose blocks:

- 🔧 **Bash** — Locate transpose method line numbers
- 🔧 **Bash** — Verify comment column alignment in the three transpose blocks

All aligned at col 57, consistent across the three. Let me run manifold tests and commit:

- 🔧 **Bash** — Run manifold tests, commit and push manifold.py cleanup

manifold.py done and pushed. Now **t3_linalg.py**. Let me read the current state in full:

- 🔧 **Read** `t3toolbox/backend/t3_linalg.py`

This file's `t3m_*` functions and `_t3m_*` helpers already conform (I wrote them); the cleanup targets are the older functions. Let me make the edits — expanding single-line return tuples, adding missing comments/annotations/return types, fixing `typ.Tuple[NDArray]` → `typ.Tuple[NDArray, ...]`, and giving `t3_plus_scalar` the docstring it's missing.

- 🔧 **Edit** `t3toolbox/backend/t3_linalg.py`
- 🔧 **Edit** `t3toolbox/backend/t3_linalg.py`
- 🔧 **Edit** `t3toolbox/backend/t3_linalg.py`
- 🔧 **Edit** `t3toolbox/backend/t3_linalg.py`
- 🔧 **Edit** `t3toolbox/backend/t3_linalg.py`
- 🔧 **Edit** `t3toolbox/backend/t3_linalg.py`
- 🔧 **Edit** `t3toolbox/backend/t3_linalg.py`

Let me verify syntax/import, alignment, and run the tests that exercise these functions:

- 🔧 **Bash** — Import-check and locate edited signatures
- 🔧 **Bash** — Check alignment of x/axis and x/y comment columns

Alignment is consistent (x/y data-args aligned; param args at their own column — matching the two-group pattern from t3m and `t3_inner`). Let me run the tests that exercise these functions:

- 🔧 **Bash** — Run suites exercising t3_linalg functions

All 71 tests pass. **Not committed** — here's the t3_linalg.py changes for your review:

## Changed (older functions; the `t3m_*`/`_t3m_*` ones already conform, left untouched)

| Function | Line | What |
|---|---|---|
| `t3_add` | 32 | expanded the one-line return tuple to per-element; `typ.Tuple[NDArray]` → `typ.Tuple[NDArray, ...]` |
| `t3_sum_stack` | 68 | added `axis` annotation (`typ.Union[int, typ.Sequence[int], None]`); removed the redundant `]: # (...)` trailing comment (the per-element comments already say it); `,...`→`, ...` |
| `t3_scale` | 139 | added `# (tucker_cores, tt_cores)` to `x`; expanded the return |
| `t3_inner_product_t3` | 161 | added `# (...)` to `x`/`y`; **added the missing return annotation** `-> NDArray:` + shape comment |
| `t3_norm` | 205 | added `# (...)` to `x`; **added the missing return annotation** `-> NDArray:` + shape comment |
| `t3_mult` | 226 | expanded the return; tuple-type fix |
| `t3_plus_scalar` | 560 | added `x`/`s` comments; expanded return; **added a one-line docstring** (it was the only function in the file with none) |

## Judgment calls to flag

1. **Return annotations added** to `t3_inner_product_t3` and `t3_norm` (both previously ended `):` with no return type). Per your "clean up annotations" guidance — they return `NDArray` (a `stack_shape`-shaped array, scalar if unstacked).
2. **Docstring added to `t3_plus_scalar`** — slightly beyond signature style, but it was conspicuously the only undocumented function here. Easy to drop if you'd rather I keep edits strictly to signatures.
3. In **`t3_sum_stack`**, `x` (data tuple, long type) and `axis` (param, shorter type) have their **types/colons aligned but comments at natural positions** (cols 75 vs 66), treating them as two groups — same treatment I gave `t3_inner_product_t3`'s data-args vs `use_orthogonalization`. If you'd prefer one forced comment column per signature regardless of type length, I'll switch to that (it'd add a ~9-space gap on `axis`).
4. **Did NOT annotate the backend `max_*_ranks=None` / `rtol=None` params** in the `t3m_*` functions — left them comment-only (`# int | Sequence[int] | None`), since the comment is the type and that's the established backend pattern (the *frontend* `t3m` annotates them). Flag in case you want backend params annotated too.
5. Left the private `_t3m_*` helpers' `max_rank, rtol, atol` bare (internal, obvious).

How does this look? Once you're happy I'll commit t3_linalg.py.


## 🧑 Nick

1. Agree, good. 2. Agree. Good idea to add missing docstrings. 3. Yes, perfect. This is exactly the sort of "within reason" case I was talking about. 4. Let's talk more about this. Ignoring any convention, do you think it is better to annotate these, or use the comment? 5. Ok, good.

## 🤖 Claude

Genuinely, setting convention aside: **I'd annotate them** — and keep a short comment for only the part the annotation can't carry. Here's the reasoning.

The comments-as-types technique exists to patch a *specific* deficiency: Python can't express array **shapes**, so `NDArray` is nearly content-free and the comment has to carry the real contract. That justification is the whole basis of the house style. But `max_tucker_ranks` isn't an array — its type is an ordinary `int | Sequence[int] | None`, which Python expresses **perfectly**. There's no gap to patch. So using a comment here is reaching for the weaker tool (uncheckable, invisible to IDEs/type-checkers) when the stronger one is fully available — and this is meant to be a general-purpose library, where external users' tooling will actually consume real annotations.

Two more things push me the same way:

- **It keeps the comment's *presence* meaningful.** If comments are used only for the inexpressible (shapes, length constraints, semantics), then "this arg has a `#`" signals "there's a contract here the type system can't capture." If comments also carry ordinary types, that signal blurs — you can't tell load-bearing shape comments from "a type written as a comment."
- **Consistency of the annotation column.** Right now these are the *only* args with an empty annotation slot — every array arg has `typ.Union[Sequence[NDArray], NDArray]`, every flag has `bool`. The mental model is: annotation column = the Python type (weak for arrays), comment column = the real contract. Under that model, `max_*_ranks` should have its (strong) Python type in the annotation column like everyone else.

**But the comment doesn't fully go away** — because part of the contract *is* inexpressible: a `Sequence[int]` annotation can't say it must have **length d** (Tucker) or **d+1** (TT), nor that `rtol` requires unstacked input. That's exactly the array-shape gap reappearing at the length level. So the best form is the same annotation+comment split the array args use:

```python
max_tucker_ranks:   typ.Union[int, typ.Sequence[int], None] = None,  # scalar caps all, or len=d
max_tt_ranks:       typ.Union[int, typ.Sequence[int], None] = None,  # scalar caps all, or len=d+1
rtol:               typ.Optional[float] = None,  # requires unstacked (enforced by the frontend)
atol:               typ.Optional[float] = None,
```

i.e. **express in the type whatever it can express; comment only the rest.** That's strictly more informative than either alone, and it's the deepest version of your own principle rather than a contradiction of it.

So my recommendation: annotate the backend `t3m_*` params (and `rtol`/`atol`), trimming each comment to just the length/semantic part. Two small sub-questions for you:

1. **Apply to the frontend `t3m` too?** It currently annotates `max_*` but as `typ.Union[int, Sequence[int]] = None` (no `None` in the union) — I'd make both ends match (include `None`, or use `typ.Optional`).
2. **Strictness**: include `None` in the union / use `typ.Optional[float]` for rtol/atol? It's technically correct, but the codebase elsewhere writes the looser `int = None` / `float = None` (e.g. `linalg.truncated_svd`, `min_rank: int = None`). I lean correct-and-explicit here, but it'd diverge from those until they're cleaned up too.

Want me to go ahead on that basis?


## 🧑 Nick

Yes, go ahead, frontend too, and go for correct-and-explicit here. Also, patch any notes to reflect this updated understanding. Again, the idea is the principle; rules should be applied within reason.

## 🤖 Claude

Nick approved — annotate (correct-and-explicit, include `None`/`Optional`), frontend too, comment only the inexpressible (length/semantics), and patch the notes. Let me do the backend params first (the 4-line block is identical across all three `t3m_*` functions, so `replace_all`):

- 🔧 **Edit** `t3toolbox/backend/t3_linalg.py`
- 🔧 **Edit** `t3toolbox/backend/t3_linalg.py`

Now the frontend `t3m`. Let me read its current signature exactly:

- 🔧 **Bash** — Find frontend t3m
- 🔧 **Read** `t3toolbox/tucker_tensor_train.py`
- 🔧 **Edit** `t3toolbox/tucker_tensor_train.py`

Let me verify alignment of both, and import-check:

- 🔧 **Bash** — Verify frontend + backend alignment and imports

Frontend `other`/`method` comments are at col 79 but the rest at 78 — off by one. Let me fix (remove one space before `#` on those two lines):

- 🔧 **Edit** `t3toolbox/tucker_tensor_train.py`
- 🔧 **Bash** — Re-verify alignment after fix

Frontend is a clean unified block (all col 78); backend uses natural positions (matching its `x`/`y` style and the `t3_sum_stack` precedent you approved). Imports pass. Let me run the tests, then update the notes:

- 🔧 **Bash** — Run tests after annotation changes

All green. Now let me patch `docs/signature_style.md` to capture the refined understanding (express what the type system can; comment only the inexpressible). Let me read the relevant part:

- 🔧 **Read** `docs/signature_style.md`
- 🔧 **Edit** `docs/signature_style.md`

Now the CLAUDE.md bullet and the memory, for the same nuance:

- 🔧 **Edit** `CLAUDE.md`
- 🔧 **Edit** `/home/nick/.claude/projects/-home-nick-repos-T3Toolbox/memory/signature-args-one-per-line.md`

All updated. Let me run a final check and commit the whole t3_linalg.py cleanup (incl. the annotation refinement + doc patch):

- 🔧 **Bash** — Run test_t3m, commit and push t3_linalg cleanup + refinement

Done and pushed. Summary of this turn:

**Applied the annotation refinement** (point 4):
- Backend `t3m_form_then_round` / `t3m_inplace_fused` / `t3m_swap`: `max_*_ranks → typ.Union[int, typ.Sequence[int], None]`, `rtol`/`atol → typ.Optional[float]`, `oversample → float`. Comments trimmed to only the inexpressible (`len=d` / `len=d+1`, `requires unstacked`, `>= 1`).
- Frontend `t3m`: same, plus added `None` to the unions / `Optional` to rtol/atol (was loosely `= None` without it), and realigned its one-block comment column.

**Patched the notes** to capture the refined understanding — annotate what the type system *can* express, comment only what it *can't* — so a `#`'s presence stays a meaningful signal of an inexpressible contract:
- `docs/signature_style.md`: new "Annotations vs comments" section.
- `CLAUDE.md` bullet + the auto-memory.

**One judgment call worth noting** (didn't want to over-decide): the frontend `t3m` keeps its pre-existing *one-aligned-block* (all comments at col 78, with some wide gaps), while the backend `t3m_*` params use *natural positions* (comments right after each type — matching the backend's `x`/`y` style and the `t3_sum_stack` treatment you approved). So the two files differ slightly: frontend = unified table, backend = per-line natural. Each is internally consistent with its file's precedent. If you'd rather they match (force the backend to a unified column too, or relax the frontend to natural), say which and I'll align them.

All tests green (test_t3m + test_tucker_tensor_train + test_dispatch). 

t3_linalg.py is complete. Want me to move to the next validated module, or address the frontend/backend block-style question first?


## 🧑 Nick

Ok, looks good. On the judgement call, let's talk through this and come to a principled plan. Here is how I currently think about it. One-aligned comment blocks are usually more readable to me. However, sometimes they make the lines very long, which is undesirable, and can make them less readable on smaller screens. So there is a tradeoff here. Another point is that aligning natural groupings can convey information about these groupings to the reader. When I look at the frontend t3m, the way it is writen now (one-aligned) reads easier to my eyes. However, I can see another person appreciating it grouped. Please think through this style convention. What are the pros and cons? Are there any principles that can be used to decide when to use each one? The bigger principle here is to make it easier for an average human coder to understand the arguments when looking at the code.

## 🤖 Claude

Good one to nail down — it's been the recurring judgment call. Let me think it through from what the reader actually does.

## The reader's two tasks

1. **Row-reading** (the common case): "I'm passing `max_tt_ranks` — what does it want?" Find the name, read its comment. This wants the comment **adjacent** to the type.
2. **Column-survey** (rarer): "what are all the contracts here?" Scan down the comment column. This wants a **consistent column**.

One-aligned optimizes (2); natural optimizes (1); grouped is the middle. Most of the time a coder is doing (1) for a single arg, which is why big horizontal gaps hurt — the eye crosses a void (or the comment is off-screen), and that's precisely the small-screen problem you named.

## The key quantitative insight

**The gap one-alignment forces on an argument equals `(longest type+default in the block) − (that arg's type+default)`.** So the *worst* gap in a one-aligned block is the **spread** of type-lengths.

- When type-lengths are **similar** (small spread), one-aligned costs almost nothing — small gaps, short lines, plus the clean column. Pure win. (This is why `probing.py`'s `compute_*` look great one-aligned: every arg is the same `Union[Sequence[NDArray], NDArray]`, spread ≈ 0.)
- When type-lengths **vary a lot** (large spread), one-aligned punishes the short args with big gaps and blows up line length. `t3m` is the bad case: `typ.Union[int, Sequence[int], None]` (≈43) vs `float` (≈10) → spread ≈ 33, so `oversample`'s comment sits ~35 cols from its type and the line hits ~128 chars.

This also resolves your "grouping conveys information" point, and it's not a coincidence: **args with different type-lengths are usually different *kinds* of args** (data tuples vs scalar flags). So high spread ⇒ there's a real grouping ⇒ grouping simultaneously (a) shrinks gaps (align within same-kind groups, which have low internal spread), (b) shortens lines, and (c) signals the structure. The three benefits arrive together exactly when you need them.

## Pros / cons

| | one-aligned | grouped | natural (2-space) |
|---|---|---|---|
| column survey | best | good (per group) | poor (staircase) |
| row adjacency | poor if spread big | good | best |
| line length | worst | medium | best |
| conveys grouping | no | yes | accidental |
| simplicity | simplest | needs grouping judgment | simple |

## Proposed principle (spread + line-budget decide it)

1. **Terse comments first** — the cheapest way to keep lines short, independent of alignment.
2. **One-aligned when the type-length spread is small** *and* the longest line stays within budget (I'd set ~100–110 chars). Default for cohesive signatures; it's the tidiest and most reach for it.
3. **Group-aligned when spread is large but the args partition into role-groups** (operands / options / flags), each with small *internal* spread. Align within each group; groups get their own columns. Use this when one-aligned would breach the budget or open a gap bigger than ~a tab. Best of both: column-scan + grouping signal + short lines.
4. **Natural (2 spaces)** for ≤2 commented args, or when even within a group types vary wildly.
5. **Hard floor:** never let a comment drift so far from its type that the row connection is lost; if alignment would force that, regroup or go natural.

In one line: **the cost of one-aligning is the type-length spread — pay it when it's small, group when it's large and structured, go natural when neither.**

## What this says about `t3m`

By the rule, `t3m`'s spread is large and role-structured, so **grouped/natural is the principled choice** — meaning the **backend** version (max_* aligned, rtol/atol aligned, oversample natural; lines ~90) is actually the more defensible one, and the **frontend** one-aligned (128-char line, 35-col gap) is the outlier. My recommendation: regroup the frontend to match the backend.

But your eye prefers the frontend as-is, and that's legitimate — if you're usually on a wide screen and value the single column, "one-aligned unless it exceeds the line budget" is also a coherent rule (it'd just make `t3m` a budget-exceeding exception you accept).

So the one thing to decide: **do we make the line-budget a hard trigger (⇒ regroup the frontend `t3m`), or a soft preference (⇒ keep `t3m` one-aligned as an accepted wide exception)?** I lean hard-trigger — it's objective, keeps small screens readable, and makes the convention self-deciding rather than per-case taste. Either way I'll write the chosen rule into `docs/signature_style.md`. Which do you want?


## 🧑 Nick

Ok, I think this makes sense, but I want to spot check it with some examples. Can you generate hypothetical examples of functions that illustrate these different cases, and show them to me. I will look at it and see how my eyes parse it. One thing that is not mentioned here is that when things are not aligned, it can look like a big jumble, and row-scanning can be harder for a human. Like, my eyes don't know where to jump to, so they have to do an "O(n)" scan to get to the type annotation or comment. On the other hand, when things are aligned my eyes can immediately jump to the relevant part (arg, annotation, or comment, "O(3)").

## 🤖 Claude

Good — let me build accurate examples so you can eyeball them, and your O(1)-jump-vs-O(n)-scan point is exactly the right lens (it's the strongest argument *for* alignment, and I'll make sure the examples isolate it). Let me write them to a scratch file and machine-verify the columns before showing you.

- 🔧 **Write** `/tmp/style_examples.py`

Let me machine-check the comment columns and line lengths per block:

- 🔧 **Bash** — Report comment columns and line lengths per arg line

Two off-by-one slips to fix (so the "aligned" blocks are genuinely aligned): `t3m_oneAligned`'s first row, and one flag in `assemble_grouped`.

- 🔧 **Edit** `/tmp/style_examples.py`
- 🔧 **Edit** `/tmp/style_examples.py`
- 🔧 **Bash** — Confirm comment-column alignment per block after fixes

All columns verified. Here are the examples for your eyes — names and types are aligned in **every** variant (always O(1)); the only thing that varies is the **comment** column, which is exactly the axis of the decision.

```python
# CASE 1 — LOW spread → one-aligned is a free win
# (identical types ⇒ comment column falls out automatically; small gaps)
def compute_mus(
        left_tt_cores:  typ.Union[typ.Sequence[NDArray], NDArray],  # len=d-1, elm_shape=C+(rLi,nUi,rL(i+1))
        xis:            typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=W+C+(nUi,)
) -> typ.Union[typ.Sequence[NDArray], NDArray]:  # mus. len=d, elm_shape=W+C+(rLi,)
```

```python
# CASE 2 — HIGH spread, but args fall into low-spread ROLE GROUPS (cores long, flags short)

# 2a) ONE-ALIGNED — flags' comments dragged ~37 cols right; comment col O(1) but gappy
def assemble_oneAligned(
        tucker_cores:   typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=C+(ni,Ni)
        tt_cores:       typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=C+(rLi,ni,rR(i+1))
        etas:           typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=W+C+(ni,)
        max_rank:       int = 0,                                    # 0 = no cap
        normalize:      bool = False,                               # rescale to unit norm
        copy:           bool = True,                                # copy inputs before mutating
) -> NDArray:

# 2b) GROUPED — align within each group; blank line separates. Two tight columns, short lines.
def assemble_grouped(
        tucker_cores:   typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=C+(ni,Ni)
        tt_cores:       typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=C+(rLi,ni,rR(i+1))
        etas:           typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=W+C+(ni,)

        max_rank:       int = 0,       # 0 = no cap
        normalize:      bool = False,  # rescale to unit norm
        copy:           bool = True,   # copy inputs before mutating
) -> NDArray:
```

```python
# CASE 3 — HIGH spread that does NOT group cleanly (the real t3m): "options" are one role
#          but their TYPES differ wildly (Union vs Optional vs float). No layout is free.

# 3a) ONE-ALIGNED — tidy column, but `oversample` comment ~37 cols out; longest line ~106
def t3m_oneAligned(
        other:              'TuckerTensorTrain',                             # same shape & stack_shape
        method:             str = 'inplace_fused',                           # algorithm selector
        max_tucker_ranks:   typ.Union[int, typ.Sequence[int], None] = None,  # scalar caps all, or len=d
        max_tt_ranks:       typ.Union[int, typ.Sequence[int], None] = None,  # scalar caps all, or len=d+1
        rtol:               typ.Optional[float] = None,                      # requires unstacked
        atol:               typ.Optional[float] = None,                      # per-step absolute tol
        oversample:         float = 1,                                       # >= 1; swap-only
) -> 'TuckerTensorTrain':

# 3b) NATURAL (2 spaces) — shortest lines, comment always adjacent; but column staircases (O(n))
def t3m_natural(
        other:              'TuckerTensorTrain',  # same shape & stack_shape
        method:             str = 'inplace_fused',  # algorithm selector
        max_tucker_ranks:   typ.Union[int, typ.Sequence[int], None] = None,  # scalar caps all, or len=d
        max_tt_ranks:       typ.Union[int, typ.Sequence[int], None] = None,  # scalar caps all, or len=d+1
        rtol:               typ.Optional[float] = None,  # requires unstacked
        atol:               typ.Optional[float] = None,  # per-step absolute tol
        oversample:         float = 1,  # >= 1; swap-only
) -> 'TuckerTensorTrain':

# 3c) GROUPED-BY-TYPE — align within same-type runs, blank line between. Each column O(1) again.
def t3m_groupedByType(
        other:              'TuckerTensorTrain',    # same shape & stack_shape
        method:             str = 'inplace_fused',  # algorithm selector

        max_tucker_ranks:   typ.Union[int, typ.Sequence[int], None] = None,  # scalar caps all, or len=d
        max_tt_ranks:       typ.Union[int, typ.Sequence[int], None] = None,  # scalar caps all, or len=d+1

        rtol:               typ.Optional[float] = None,  # requires unstacked
        atol:               typ.Optional[float] = None,  # per-step absolute tol

        oversample:         float = 1,  # >= 1; swap-only
) -> 'TuckerTensorTrain':
```

Your O(1)-vs-O(n) point maps cleanly onto these: **3b (natural)** is the one where the comment column staircases (51→53→78→58→41) — that's the "where do I jump?" jumble you described. **3a, 3c** (and all of CASE 1/2) keep comments in predictable columns; the difference is 3a uses *one* column (with big gaps) while 3c uses *one column per group* (small gaps, blank-line-delimited, shorter lines).

So the real contrast for the hard case is **3a vs 3c**: 3a = single predictable column + big gaps + long lines; 3c = several predictable columns (each O(1) once you're in a group) + short lines + visible "kinds," at the cost of vertical space (blank lines) and your eye tracking a few columns instead of one.

Take a look — which of 3a / 3b / 3c does your eye prefer, and does 2b confirm grouping is the clear win when the groups *are* low-spread? Once you tell me, I'll lock the rule into `docs/signature_style.md`.


## 🧑 Nick

2b and 3c are clear winners from the eyeball test.

## 🤖 Claude

Decided: **grouping wins.** The rule crystallizes to — names/types always aligned; when type-length spread is large, split into blank-line-delimited groups of similar types and align the comment column *within* each group (3c); reserve one-column for low-spread signatures (1, 2-no... 2b is grouped). Let me apply 3c to the live `t3m` (frontend + backend) and lock the rule in the docs.

Starting with the frontend `t3m`:

- 🔧 **Edit** `t3toolbox/tucker_tensor_train.py`

Now the backend `t3m_*` — they already align within type-runs, so I just insert blank lines to delimit the groups (and bump `x`/`y` to the 2-space micro-grammar). The 6-line `x…atol` block is identical across all three, so `replace_all`:

- 🔧 **Edit** `t3toolbox/backend/t3_linalg.py`
- 🔧 **Edit** `t3toolbox/backend/t3_linalg.py`

Let me verify alignment of both and import-check:

- 🔧 **Bash** — Show frontend t3m alignment and locate swap

Frontend is clean 3c (groups at 57 / 78 / 62 / 45, blank-delimited). Let me check the backend `t3m_swap` and `t3m_form_then_round`:

- 🔧 **Bash** — Check backend swap signature alignment

Backend is clean 3c too. Let me import-check and run the tests:

- 🔧 **Bash** — Import-check and run t3m + dispatch tests

All green. Now lock the rule into `docs/signature_style.md`. Let me read the section to slot it in:

- 🔧 **Bash** — Find doc sections and the superseded note
- 🔧 **Read** `docs/signature_style.md`
- 🔧 **Edit** `docs/signature_style.md`

Now update the CLAUDE.md bullet to reference the grouping rule:

- 🔧 **Read** `CLAUDE.md`
- 🔧 **Edit** `CLAUDE.md`

Let me add the grouping nuance to the memory too:

- 🔧 **Read** `/home/nick/.claude/projects/-home-nick-repos-T3Toolbox/memory/signature-args-one-per-line.md`
- 🔧 **Edit** `/home/nick/.claude/projects/-home-nick-repos-T3Toolbox/memory/signature-args-one-per-line.md`

Let me run the frontend-exercising suite and commit:

- 🔧 **Bash** — Run tests, commit and push the grouping rule + t3m regroup

Done and pushed. The style question is settled and applied.

## The rule (now locked in `docs/signature_style.md` + `CLAUDE.md` + memory)

Your eyeball test (2b and 3c win) gave the principle:

- **Names and types always align** — both are O(1) jumps. The only question is the **comment** column.
- **The cost of one-aligning = the type-length spread** (the gap each arg gets = `longest type − its type`).
- **Small spread → one comment column** (cheap, tidy — `probing.py`).
- **Large spread → blank-line-delimited groups** of similar-length types (operands / rank caps / tolerances / flags), comments aligned **within** each group. Predictable column per group (your O(1)-within-group), short lines, and the grouping is made visible.
- **Never the staircase** (2-space natural) — it's the O(n) "where do I jump?" jumble you flagged.

## Applied

`t3m` — frontend and all three backend `t3m_*` — regrouped to the 3c layout (operands / `max_*` / `rtol,atol` / `oversample`, blank-line-separated, aligned within each group). That **resolves the frontend-vs-backend discrepancy**: the frontend dropped from a 128-char one-aligned line to grouped, and the backend gained the blank-line group delimiters — both are now identical 3c style.

All tests green (66 across tucker + t3m, plus dispatch). 

t3_linalg.py and manifold.py are both fully cleaned now. Want me to move to the next validated module (e.g. `basis_variations_format.py`, or a backend module like `tangent_operations.py` / `corewise.py`), or pause here?


## 🧑 Nick

Ok, great. At this point, I am convinced that I can trust you to make good decisions for function signature style. However, I am not certain whether a future version of yourself, or perhaps the AI of another future collaborator, will make the right decisions. Let me propose an idea: you create a sub-agent which does not have your context but does have access to the project docs. You create example cases of code to format, and ask the subagent to do the formatting. You check to see if the sub-agent did the formatting the way you would have. If not, you update the project docs to clarify the differences

## 🤖 Claude

Love this — it's a doc-quality test: if the docs alone (no conversation context) let a fresh agent reproduce the style, they're good; where it diverges, the docs have a gap. Let me set it up.

I'll give a fresh general-purpose agent (no access to our discussion) the docs and four unformatted functions chosen to probe the key decisions — low-spread, high-spread-groupable, the hard t3m-like case, and the annotation-vs-comment + within-reason case.

- 🔧 **Agent** (general-purpose) — Doc-only signature formatting test

The fresh agent is running (it has the docs but none of our conversation). I'll prepare my "answer key" while it works, so I can compare its decisions against what I'd do — and use any divergence to find doc gaps.

Here's what I expect for each case:

1. **`combine_edges`** (low spread, all identical types) → **single aligned comment column**; return tuple expanded to one element with a shape comment.
2. **`assemble_and_cap`** (high spread, role-groupable) → **two blank-line-delimited groups** (3 core args | 3 flags), comments aligned within each; cores annotated `typ.Union[...]`, flags `int`/`bool`.
3. **`hadamard`** (the hard t3m-like case) → **grouped-by-type with blank lines** (`x,other` | `max_*` | `rtol,atol` | `oversample`); params **annotated** (`typ.Union[int, typ.Sequence[int], None]`, `typ.Optional[float]`, `float`) with comments trimmed to the inexpressible (`len=d` / `len=d+1`, `requires unstacked`, `>= 1`); return tuple expanded.
4. **`t3_norm`** (within reason, 2 args + annotation) → simple aligned (no blank-line groups needed); `x` tuple-annotated + comment, `use_orthogonalization: bool` + comment; `-> NDArray:` with shape comment.

The most likely doc-gaps it'll expose: (a) does it know to *annotate* expressible types rather than comment them (the refinement we just added)? (b) does it group case 3 by type with blank lines vs one-aligning it? (c) does it avoid over-grouping case 4? I'll report the comparison and patch the docs for any miss when it returns.

The agent nailed it almost completely — working only from the docs, with zero conversation context:

| Case | Decision | Verdict |
|---|---|---|
| 1 `combine_edges` | single column, return expanded | ✅ exactly right |
| 2 `assemble_and_cap` | grouped (cores \| flags), blank line, real annotations | ✅ right |
| 3 `hadamard` | grouped-by-type + blanks; **annotated** `Union`/`Optional` w/ trimmed comments; return expanded | ✅ on the hard parts (annotation refinement included!) |
| 4 `t3_norm` | 2 args → not grouped; annotated; return comment | ✅ right |

That's strong evidence the docs are largely self-sufficient. **One genuine gap surfaced**, exactly the kind of thing this test is for:

In case 3 it grouped `method` (a short `str`) *with* the long `Union` caps and aligned them — which opens a ~20-col gap on `method`, the very thing grouping exists to remove. It grouped by **role** when role and **type-length** disagreed. The doc says "runs of similar-length types (which usually coincide with roles)" but never says what wins when they *don't*. Let me patch that:

- 🔧 **Edit** `docs/signature_style.md`

Patched. Now to honor the loop properly, let me re-run a fresh agent (no context) on the same cases and confirm the doc fix steers `method` correctly:

- 🔧 **Agent** (general-purpose) — Re-verify doc fix (signature formatting)

Re-verification agent launched (fresh context, updated docs). While it runs:

The first pass validated your idea works — the docs alone carried a no-context agent through low-spread, high-spread-groupable, the hard t3m-like case (**including** the annotate-the-expressible refinement), and the within-reason 2-arg case. Only one gap leaked: the **role-vs-type-length conflict** when grouping (`method` force-aligned into the long-`Union` group, opening the gap grouping is meant to remove). I patched `docs/signature_style.md` with the tiebreaker: **group by type-length, not role; an outlier whose length matches no group gets its own blank-delimited group; split out if aligning would open a gap wider than ~a tab.**

The re-run tests whether that patch closes the gap. I'll report whether case 3 now places `method` correctly, then commit the doc fix (and note this validation loop is a reusable tool for future style/convention docs).

Converged. The re-run (fresh agent, patched docs) now handles case 3 correctly — `method` is its **own blank-delimited group** between operands and the `max_*` caps, no forced gap. Its stated reasoning quotes the new tiebreaker almost verbatim: *"grouped by type-length into operands / `str` selector / `max_*` `Union` caps / `Optional[float]` tolerances / bare-`float` oversample."* The other three cases held steady. The loop did exactly what you designed it to: found a real doc gap → patch → re-verify → convergence.

Let me commit the doc fix:

- 🔧 **Bash** — Commit and push the grouping tiebreaker doc fix

Pushed. The validation loop you proposed worked end-to-end:

1. **Built probe cases** targeting each decision (low-spread, high-spread-groupable, the hard t3m-like case, within-reason 2-arg).
2. **Fresh agent, docs only** (no conversation context) → got ~everything right, including the subtle *annotate-the-expressible* refinement we'd just added.
3. **Found one real gap**: role-vs-type-length conflict in grouping.
4. **Patched** the doc with the tiebreaker.
5. **Re-ran** the same fresh-agent test → converged.

Two takeaways worth noting:
- The docs are now empirically self-sufficient for the signature style — a future you, or another collaborator's AI, can reproduce it from `docs/signature_style.md` + `CLAUDE.md` alone.
- **This is a reusable doc-quality pattern**, not just for this convention: any time we write or change a convention doc, a no-context sub-agent + a few probe cases is a cheap way to test whether the doc actually transfers. If you'd like, I can save these four probe cases (and the harness prompt) as something like `docs/_style_selfcheck.md` so the regression is trivially re-runnable after future doc edits — or leave it ad-hoc. Your call.

Where to next? Options: (a) continue the signature cleanup on the next validated module (`basis_variations_format.py`, or a backend module like `tangent_operations.py` / `corewise.py`); (b) set up that reusable self-check artifact; (c) something else.


## 🧑 Nick

Ok, before moving on, there is one concern I want you to think about: "overfitting" the docs. What we are doing is, in some sense, are optimizing to find the "best doc". That is, the doc that an AI agent without context will perform best with, over the set of all function signatures it could possibly encounter. Right now, we have optimized on a few examples. However, we don't want the doc to "overfit" to these examples. We want the doc to "generalize". So, if we make the doc overly dogmatic in order to force the AI to match on these examples, it may cause the AI to do worse on future examples.

## 🤖 Claude

This is a sharp concern and exactly the right one to raise before we make this a habit. Let me think it through honestly, including whether I already overfit a little.

## The framing is right, with one crucial twist

We're doing empirical risk minimization: probe cases = training set, the doc = model, "matches my judgment" = loss, the space of all signatures = the true distribution. So yes — patching the doc to nail a few examples risks **high bias toward those shapes** and brittleness elsewhere. And there's a Goodhart trap underneath: "match Claude's output" is only a *proxy* for the true objective, "**a human understands the args quickly**." Optimize the proxy too hard and you can drift from the real goal.

The twist that changes the strategy: **the "model" isn't just the doc — it's the doc *plus* a strong general reasoner.** The AI isn't a blank parametric function; it arrives with priors and can derive. That means the highest-generalization doc doesn't *enumerate cases* — it conveys the **goal + the WHY**, and lets the reader derive the rest. Rules-without-reasons are what overfit; principles-with-reasons are what generalize. (This is why the doc leads with "the shape *is* the type" and "the cost of aligning = the type-length spread" — a reader who internalizes the cost model can format a signature shape neither of us has ever seen.)

## Did I just overfit? Partly — let me be honest

My tiebreaker patch had two pieces:
- **"Group by type-length, not role"** — this is *not* overfitting. It's a **derivation from the already-stated cost model** (gap = spread). It explains *why* and applies to every signature. Adding a generalizing principle that unifies cases is the opposite of overfitting.
- **"split if the gap exceeds ~a tab (≈8 cols)"** — this is the overfit smell: a **magic number** tuned to make the example come out right. Real readability isn't 8 columns; the number is a context-dependent artifact. I'd soften it to the *principle* ("split when the gap is large enough that the eye loses the row") and keep any number as a hedged "e.g.", not a law.

So one of two pieces leaned toward overfit. Worth fixing.

## Safeguards (the ML analogues, made concrete)

1. **Held-out test set.** The biggest one, and it directly answers your concern: keep a set of probe cases we **never patch against** — only measure on. Find gaps on "dev" cases; report generalization on untouched "test" cases. If the doc only does well on the patched ones, it's overfit. I evaluated on the *same* 4 both times, which can't detect overfitting — that's a real methodological hole.
2. **Diverse, broad probes.** Narrow probes invite memorization; wide ones (varied arity, type vocabularies, nesting, genuinely novel shapes) *force* the doc toward general principle.
3. **Parsimony as regularization.** Only add a clause if it pays off across *many* cases, not one. Prefer strengthening the *why* over adding a *what*. A shorter principle-doc relies more on the AI's reasoning (generalizes); a longer rule-doc replaces reasoning with lookup (overfits). Watch doc length as a regularization signal.
4. **Justify every patch by the true objective, not the proxy.** "This helps the reader because X" generalizes; "this makes case 3 match how I did it" overfits. If I can't tie a patch to reader comprehension, don't make it.
5. **Don't legislate taste.** Many formattings are equi-optimal. When a divergence is a *defensible alternative* rather than a clear error, leave it to judgment — don't write a rule to force my preference. The "within reason / the goal wins over the rule" framing is the built-in regularizer; keep it prominent. (It's also why this isn't "always minimize the doc" — too vague → high *variance* across agents. The target is the bias–variance sweet spot: principle-first, judgment-fallback, specifics only where they cut variance without adding brittleness.)

## Proposed next actions

- **Soften the magic number** in the tiebreaker to a principle (keep a hedged "e.g."), so the one overfit-smell is removed.
- **Adopt a held-out set** going forward: when we self-check a doc, split probes into dev (patch against) and test (measure only), and bias the probes toward variety.
- Keep the **patch test = "does it strengthen a generalizing *why*?"** If a divergence can only be fixed by a shape-specific rule, that's a signal to either generalize it to a principle or accept it as judgment, not legislate it.

Want me to (a) de-overfit that threshold line now, and (b) re-run the self-check with a *fresh, more varied, held-out* batch (that I won't patch against) purely to estimate generalization — treating any new divergence as data about the doc's bias rather than something to immediately "fix"?


## 🧑 Nick

Yes, please do both (a) and (b)

## 🤖 Claude

On it — (a) de-overfit the threshold, then (b) a held-out generalization probe I won't patch against.

## (a) Soften the magic number to a principle

- 🔧 **Edit** `docs/signature_style.md`
- 🔧 **Bash** — Commit the de-overfit doc softening

## (b) Held-out generalization probe

These are **new, deliberately varied** cases (callables, dicts, all-scalar, single-arg, non-T3 domains) that I have **not** used to write or patch the docs. I'm **pre-registering my expectations** so I can't rationalize after the fact, and I will **not patch** based on the results — only measure.

**Pre-registered expectations** (key checks; several have multiple acceptable answers):
1. `make_optimizer` (all scalars) — annotate all (`float`/`bool`/`Optional[float]`); constraints (`in [0,1)`, `>=0`, `None=no clip`) in comments; one column *or* split the lone `Optional` — both fine. No invented shape comments.
2. `line_search` — `f: Callable` annotated + role note; `x0`/`direction` arrays aligned; scalars annotated; `f` is a type-outlier (own group or natural — judgment).
3. `total_variation` (1 arg) — single line, shape comment, `-> float` with comment; not over-grouped.
4. `reshape_blocks` — annotate `Tuple[int,...]` / `Dict[int,int]` (expressible!) with semantics in comments; reasonable grouping.
5. `cap_ranks` — `limit` is the long `Union` outlier among short args; group-by-length or one-column (4 args); `Union[...,None]` annotated, comment trimmed.

- 🔧 **Agent** (general-purpose) — Held-out generalization probe

Both done: (a) the magic-number threshold is now a principle (committed & pushed), and (b) the held-out probe is running on five varied, never-patched-against cases.

The methodology now matches your concern: this batch is a genuine **test set** (different domains, types, and arities from the dev cases), I **pre-registered** my expectations above, and per your guidance I'll treat any divergence as **data about the doc's bias/generalization**, not as something to reflexively patch. A clean pass is evidence the principle-first doc generalizes; a divergence on a *defensible-alternative* case tells us we should leave it to judgment, while a divergence that's a *clear readability error* would indicate a genuine remaining gap (and even then we'd weigh a principle-level fix against added dogmatism). I'll report the comparison when it finishes.

Results are in. Per our agreement I'm treating these as **measurement, not a patch trigger** — and the headline is that the doc **generalized well**, with the only divergences being genuine taste-level judgment calls.

## Scorecard vs. pre-registration

| Case | Result |
|---|---|
| 1 `make_optimizer` (all-scalar) | ✅ all annotated, constraints-in-comments, single column, `-> Optimizer`. No invented shape comments. |
| 2 `line_search` (callable+arrays+scalars) | ⚠️ correct annotations (incl. `Callable[[NDArray], float]`), but kept **one column** where I'd lean toward grouping (gaps ~17 on the scalars) |
| 3 `total_variation` (1 arg) | ✅ exact |
| 4 `reshape_blocks` (tuple/dict/bool/float) | ✅ `Tuple[int, ...]` / `Dict[int, int]` annotated, semantics in comments, single column |
| 5 `cap_ranks` (long `Union` outlier) | ⚠️ kept **one column** (big ~30-col gaps on `mode`/`inplace`) — but *explicitly flagged it borderline*, cited the ≳4-row threshold, and offered to split `limit` out |

## The strong signal: the annotation principle generalized

The agent correctly annotated **`Callable[[NDArray], float]`, `Tuple[int, ...]`, `Dict[int, int]`, `Optional[float]`, `Union[int, Sequence[int], None]`** — types that **never appeared in the dev cases** — and put only the inexpressible part (shapes, `len=d+1`, `'soft'|'hard'`, semantics) in the comments. That's the clearest possible evidence the doc encodes a *principle* ("annotate the expressible, comment the rest"), not memorized examples. Single-arg and all-scalar (no arrays) cases were handled right too. No overfitting visible.

## The two ⚠️ are not errors — they're the bias/variance tradeoff we deliberately bought

Both divergences are the same: on **borderline grouping calls**, the agent tolerates bigger gaps and keeps one column, where I'd lean toward splitting. Three things make this *healthy*, not a gap:

1. It's a **defensible point on the spectrum** — specifically the one-aligned-with-gaps style *you yourself* preferred for the frontend `t3m` early on. So "wrong" here means "picked Nick's original favorite," not "unreadable."
2. On case 5 the agent **reasoned explicitly**, cited the doc's own borderline criteria, and **offered the alternative** — exactly the judgment behavior we want instead of dogmatic rule-following.
3. This is the **direct, predicted consequence of removing the magic number** in (a). The number made grouping more decisive but was brittle; removing it shifts borderline calls toward one-column. We traded a little grouping-decisiveness for less overfitting — and the held-out test shows that's exactly what happened, in a defensible way.

## Recommendation: don't patch

The divergences are genuine ties left to judgment, and the agent resolves them defensibly. Adding firmer grouping guidance to force my preference on borderline cases would be **re-introducing the overfitting/dogmatism you flagged** — legislating taste and suppressing the reasoning that handled everything else correctly. The doc looks to be at a good bias–variance point: it nails the clear calls (alignment, annotation, single-arg, varied types) and leaves real ties open. So I'd **leave the doc as-is** (no commit needed; this run was measurement only).

One honest caveat: the agent slightly *under-estimated its own gaps* on case 5 (called ~30-col gaps "tolerable"). If anything nudged me, it'd be a tiny clarification that "one column for few rows" still has to respect the lose-the-row gap test — but that risks re-litigating the number, so I'd only do it if you've seen the borderline-skew bite in practice.

Want to leave it here, or should I gently strengthen the borderline guidance (at the documented risk of nudging back toward overfitting)?


## 🧑 Nick

Ok, great. Based on these results, I'm happy leaving the docs as-is. Let's move on to actually doing the function signature modifications for the other verified modules

## 🤖 Claude

Good. The doc is validated, so I can confidently delegate per-file (each agent reads the same spec, I review every diff + run tests). Let me first survey the remaining verified modules to plan the batches:

- 🔧 **Bash** — Survey remaining verified files for size and crammed signatures

This is a sizable, precision-sensitive sweep. Survey shows the work concentrates in `basis_variations_format.py` (18), `corewise.py` (11), `stacking.py` (6), `tangent_operations.py` (3), plus several 1–2 signature files; `contractions.py`/`linalg.py`/`ranks.py`/`t3_svd.py`/`bv_conversions.py` likely already conform.

**Plan:** delegate per-file to fresh agents (the validated approach — each reads the same doc spec), then I review every diff against the code (shapes correct, annotations right, no behavior change, alignment) and run tests before committing. I'll go in waves so review stays tractable.

Launching Wave 1 — three files of varied size/shape to confirm the real-file workflow:

- 🔧 **Agent** (general-purpose) — Signature cleanup: corewise.py
- 🔧 **Agent** (general-purpose) — Signature cleanup: tangent_operations.py
- 🔧 **Agent** (general-purpose) — Signature cleanup: stacking.py

Wave 1 is running in the background — three fresh agents reformatting `corewise.py`, `tangent_operations.py`, and `stacking.py` in parallel, each working only from the validated doc spec. They'll edit in place and report any shape comments they're unsure of.

When they complete I'll review each diff against the code (correct shapes, correct annotations, no behavior change, alignment), run the test suite, fix any slips, and commit per file — then kick off Wave 2 (`basis_variations_format.py` — the big one — plus the small 1–2 signature files, and a quick conformance check of `contractions.py`/`linalg.py`/`ranks.py` which likely need little or nothing). I'll report back with the review once Wave 1 lands.

Wave 1's first agent (corewise.py) is back — it couldn't run Bash itself, so I'll verify. Let me import-check, run tests, and review the diff:

- 🔧 **Bash** — Import-check corewise, find tests, diff stat

corewise imports clean; stacking.py agent also finished (Bash was denied for the sub-agents, so I'll verify all imports myself). Let me review the corewise diff in full:

- 🔧 **Bash** — Full corewise.py diff for review

All three Wave-1 agents are done (all hit Bash-denied in their sandboxes, so I verify everything). corewise looks clean — only nit: it left `corewise_dot/norm/stack_dot/stack_norm/err/relerr` return-annotation-free (comment-only), where my `t3_linalg` precedent added `-> NDArray:`. I'll add those for consistency. Let me check the other two diffs and import them:

- 🔧 **Bash** — Import-check and diff tangent_operations + stacking

Both clean. Two things to handle before committing Wave 1: (a) add `-> NDArray:` to corewise's 6 comment-only returns (consistency with `t3_linalg`), and (b) verify the stacking agent's flagged **uniform "axis-0 reserved"** comments, since that touches the deferred uniform layer. Let me read those bodies:

- 🔧 **Read** `t3toolbox/backend/stacking.py`

Verified: the bodies confirm the agent's "axis 0 reserved" comments are **correct** (`range(1, 1+num_stacking_axes)` for uniform vs `range(num_stacking_axes)` for ragged). So stacking and tangent_operations are good as-is. Now let me add the `-> NDArray:` returns to corewise. Let me get exact lines:

- 🔧 **Bash** — Find corewise no-annotation return lines
- 🔧 **Bash** — Add -> NDArray returns to corewise; reimport

corewise returns annotated, imports clean. Now run the suites that exercise these three modules, then commit Wave 1:

- 🔧 **Bash** — Run suites exercising corewise/tangent_operations/stacking

All 122 green. Committing Wave 1:

- 🔧 **Bash** — Commit and push Wave 1

Wave 1 pushed. Launching Wave 2 — `basis_variations_format.py` (the big one), `t3_operations.py`, and `common.py`:

- 🔧 **Agent** (general-purpose) — Signature cleanup: basis_variations_format.py
- 🔧 **Agent** (general-purpose) — Signature cleanup: t3_operations.py
- 🔧 **Agent** (general-purpose) — Signature cleanup: common.py

Wave 2 is running. While it works (on different files), let me handle the small remaining files myself — non-overlapping. Let me look at their crammed signatures:

- 🔧 **Bash** — Find crammed signatures in the small files

The Wave-2 agents are editing their files. The "crammed" hits in the small files are all **inner closures** (`_dev`, `_left_func`, `_func`) — which the doc explicitly exempts ("throwaway internal closures don't need the full treatment"). So those small files' public signatures likely already conform. Let me verify the public signatures in a couple:

- 🔧 **Bash** — List top-level public defs in the small files

All public defs use the multi-line form. Let me confirm they already conform (shape comments, expanded returns) — reading apply.py and orthogonal_representations.py signatures:

- 🔧 **Bash** — Read public signatures to assess conformance

The small files' public signatures already conform (the "crammed" hits were exempt inner closures); `orthogonalization.py` and `orthogonal_representations.py` are clean. Only `apply.py`/`entries.py` have minor gaps — one-line `*_transpose` returns (should expand) and bare `sum_over_probes`. Let me read `entries.py` and fix both:

- 🔧 **Read** `t3toolbox/backend/entries.py`

common.py agent is done too (I'll review it shortly). Let me finish my `apply.py`/`entries.py` touch-ups first. Editing both `*_transpose` signatures (align, normalize `shape=W+C`, add `sum_over_probes` note, expand returns):

- 🔧 **Edit** `t3toolbox/backend/entries.py`

All Wave 2 agents are back. Let me finish the apply.py edit, then review everything (Wave 2 + my apply/entries) together:

- 🔧 **Edit** `t3toolbox/backend/apply.py`
- 🔧 **Read** `t3toolbox/backend/apply.py`
- 🔧 **Edit** `t3toolbox/backend/apply.py`

All edits in. Let me import-check everything and run the full suite to catch any agent-introduced breakage:

- 🔧 **Bash** — Import-check Wave 2 + small files, run suite

The "error" is just my bad test path (`backend.test_contractions` → should be `tests.backend.test_contractions`); the 122 real tests passed. Let me re-run cleanly and check the `squash_tt_tails` variadic-return (which the agent said it kept un-expanded — good):

- 🔧 **Bash** — Clean test run + check squash_tt_tails return

151 tests OK; `squash_tt_tails` kept its variadic `Tuple[NDArray, ...]` un-expanded (correct). Let me review the three Wave-2 diffs for shape/annotation correctness — basis_variations_format first:

- 🔧 **Bash** — Review basis_variations_format diff

`basis_variations_format` is clean — one consistency fix: `random_orthogonal`'s `shape`/`tucker_ranks`/`tt_ranks` were left un-annotated (just `# len=d`), but they're expressible and my `manifold.random_orthogonal` annotates them. Let me match:

- 🔧 **Edit** `t3toolbox/basis_variations_format.py`
- 🔧 **Read** `t3toolbox/basis_variations_format.py`
- 🔧 **Edit** `t3toolbox/basis_variations_format.py`

Now the t3_operations and common diffs — let me review for any wrong shapes or broken annotations:

- 🔧 **Bash** — Review t3_operations diff

t3_operations is clean — accurate shapes, correct annotations, variadic returns preserved, sensible group-by-type (blank before `use_jax` in the constructors). Now common.py:

- 🔧 **Bash** — Review common.py diff

common.py is clean (the untyped `x` element arg correctly stays comment-only; the variadic return preserves its `...`). All five files reviewed and good. Let me re-run the suite (I edited 3 files after the last run) then commit this batch:

- 🔧 **Bash** — Re-run suite after final edits
- 🔧 **Bash** — Commit and push Wave 2 + apply/entries

Wave 2 pushed. Status: done — manifold, t3_linalg, corewise, tangent_operations, stacking, basis_variations_format, t3_operations, common, apply, entries (+ probing/orthogonalization/orthogonal_representations already conform). 

Remaining: the keystone `tucker_tensor_train.py` (big, mostly conforms) plus a conformance check of `t3_orthogonalization`/`t3_svd`/`ranks`/`linalg`/`contractions`/`bv_conversions`. Let me delegate the keystone (conservative, minimal-change) and spot-check the small backends myself while it runs:

- 🔧 **Agent** (general-purpose) — Signature cleanup: tucker_tensor_train.py

The keystone agent is running. While it works, let me spot-check the remaining "likely-conform" backends for any unexpanded fixed-tuple returns or crammed/comment-less public signatures:

- 🔧 **Bash** — Scan likely-conform backends for non-conforming signatures

`t3_svd`, `ranks`, `linalg`, `contractions`, `bv_conversions` all conform — no work. But `t3_orthogonalization.py` has 6 one-line fixed-tuple returns to expand — and I recall two of them (`left/right_orthogonalize_t3`) have **stale/wrong** return comments (say "tucker_variations, outer_tt_cores" but actually return up_tucker + left/right_tt). That's exactly the silent error this sweep should catch. It's not touched by the keystone agent, so let me fix it now. Reading the regions:

- 🔧 **Read** `t3toolbox/backend/t3_orthogonalization.py`

Confirmed: `left/right_orthogonalize_t3` return comments are **wrong** (say "tucker_variations, outer_tt_cores" but return up_tucker + left/right_tt), and `right_orthogonalize_t3`'s docstring says "Left" (copy-paste bug). Let me expand all four returns and fix the stale comments/docstring:

- 🔧 **Edit** `t3toolbox/backend/t3_orthogonalization.py`
- 🔧 **Edit** `t3toolbox/backend/t3_orthogonalization.py`
- 🔧 **Edit** `t3toolbox/backend/t3_orthogonalization.py`
- 🔧 **Edit** `t3toolbox/backend/t3_orthogonalization.py`

Now the two `orthogonalize_relative_to_*` functions. Let me read them:

- 🔧 **Bash** — Locate the relative orthogonalize functions
- 🔧 **Read** `t3toolbox/backend/t3_orthogonalization.py`

Let me read the second relative function to edit both:

- 🔧 **Read** `t3toolbox/backend/t3_orthogonalization.py`

The keystone agent was appropriately conservative (only `segment` + `from_vector` — explained well, flagged borderline cases it left). Let me finish my two `t3_orthogonalization` relative functions first:

- 🔧 **Edit** `t3toolbox/backend/t3_orthogonalization.py`
- 🔧 **Edit** `t3toolbox/backend/t3_orthogonalization.py`

Now review the keystone agent's 2 edits, import-check both files, and run the suite:

- 🔧 **Bash** — Review keystone diff + import-check

Both clean. Run the suite and commit Wave 3 (keystone + t3_orthogonalization, which also got a stale-comment + docstring fix):

- 🔧 **Bash** — Run suite, commit and push Wave 3

Wave 3 pushed — that completes the verified set. Let me double-check one I may have missed: `dense_t3svd.py` (relied on by `tucker_tensor_train.t3svd_dense`):

- 🔧 **Bash** — Check dense_t3svd conformance and list remaining backend files

`dense_t3svd.py` shows no crammed sigs/unexpanded returns. Let me confirm its three signatures actually carry shape comments (not just structurally split):

- 🔧 **Bash** — Inspect dense_t3svd signatures

`dense_t3svd.py` substantially conforms — one small gap: `ttsvd_dense`'s `T` lacks the shape comment its siblings have. (I'll leave `rtol/atol`/`min_ranks` as `float = None`/`Sequence[int] = None` to match the rest of the SVD layer — `linalg.truncated_svd` uses the same loose form, so changing only here would be inconsistent; an "Optional everywhere in the SVD layer" sweep is a separate call.) Let me add the one comment:

- 🔧 **Read** `t3toolbox/backend/dense_t3svd.py`
- 🔧 **Edit** `t3toolbox/backend/dense_t3svd.py`
- 🔧 **Bash** — Commit dense_t3svd comment fix

The signature-style sweep of the **entire verified set is complete**. Summary:

## Done (all verified modules now conform)

**Frontend:** `tucker_tensor_train.py`, `basis_variations_format.py`, `manifold.py`, `corewise.py`
**Backend:** `t3_linalg`, `probing`*, `tangent_operations`, `t3_operations`, `t3_orthogonalization`, `orthogonalization`*, `orthogonal_representations`*, `stacking`, `common`, `apply`, `entries`, `linalg`*, `ranks`*, `t3_svd`*, `dense_t3svd`, `contractions`*, `bv_conversions`*  (`*` = already conformed, verified, little/no change)

**Not touched (correctly):** the deferred layers — `bv_operations.py` (parked weighted), `ut3_*`/`ubv_*`/`uniform_*`/`wt3_*`, `OLD_*`.

## How it went

Bulk done by **doc-only sub-agents** (no conversation context — just `docs/signature_style.md` + `CLAUDE.md`), then I reviewed every diff against the code, ran the suite, and committed per wave. This was a real-world rerun of your validation idea: fresh agents reproduced the style well from the docs alone; my review caught the handful of divergences. All committed and pushed; the full suite (manifold + basis_variations_format + tucker_tensor_train + dispatch + contractions, 151 tests) stayed green throughout.

## Notable things the sweep caught beyond formatting

- **Two genuine silent bugs** in `t3_orthogonalization.py`: `left/right_orthogonalize_t3` had **wrong return comments** (claimed `(tucker_variations, outer_tt_cores)` but actually return `up_tucker` + `left/right_tt`), and `right_orthogonalize_t3`'s docstring said "Left". Fixed.
- **Variadic vs fixed tuples**: agents correctly left `Tuple[NDArray, ...]` returns un-expanded (only fixed-arity tuples get expanded) — one self-corrected mid-edit.
- **Consistency fixes** in review: `corewise` scalar/array returns got `-> NDArray`; `basis_variations_format.random_orthogonal` annotations aligned with `manifold.random_orthogonal`.

The convention is documented (`docs/signature_style.md`), validated (the held-out + bias checks), and now applied across the whole verified codebase. Want me to move on to something else — the deferred least-squares tutorial, repairing a deferred layer, or another item from the TODO list?


## 🧑 Nick

Excellent! Now I would like to move on to updating doctests in the verified modules. I would like to proceed similarly to how we did for the signature style. We work through some doctests functions one at a time, talk about what the doctests should do, look at deficiences or problems with the existing doctests, and then update them accordingly. From this process, we will hopefully learn general principles that can be written up in a file, for future automated application.

## 🤖 Claude

Good next target — and like signatures, worth aligning on *what doctests should be* before editing. Let me survey the current state to ground the discussion.

- 🔧 **Bash** — Survey doctest density and unseeded-random usage

Let me redo the random-usage check and actually *run* the doctests to see the real failure picture (the most direct way to find deficiencies):

- 🔧 **Bash** — Survey random/seed usage and run doctests on small modules

Very revealing: `corewise` (fixed inputs) and `manifold` (prints deterministic dims/ranks) **pass**, while `linalg` has 3 failures — all the "print a raw residual from unseeded random" pattern (`print(norm(A-A2))` → a machine-noise float that won't reproduce). Let me quantify across all verified modules:

- 🔧 **Bash** — Count doctest failures across all verified modules

Strong signal — here's the concrete picture across the verified set:

| module | failing / total examples | why |
|---|---|---|
| **manifold** | **0 / 102** | uses random but prints **deterministic facts** (dims, ranks) and **bool checks** / exact algebraic `0.0` |
| **corewise** | **0 / 54** | **fixed** inputs → deterministic outputs |
| **stacking** | **0 / 176** | deterministic |
| tucker_tensor_train | 190 / 903 | raw random residual prints |
| dense_t3svd | 16 / 30 | "" |
| basis_variations_format | 21 / 159 | "" |
| probing | 10 / 122 | "" |
| linalg | 3 / 40 | "" |

**~240 failing examples, and they're all one failure mode**: printing a *non-reproducible float* (a machine-noise residual from **unseeded** random, e.g. `print(np.linalg.norm(A - A2))` → `1.04e-13`). Everything deterministic — shapes, ranks, dims, exact `0.0`, `True/False` — reproduces and passes. **No module seeds its RNG**, yet manifold/corewise/stacking still hit 0 failures because they never print a raw random value.

## The exemplar already exists: `manifold.py`

Just like `probing.py` was the signature reference, **`manifold.py` is the doctest reference**. It does the right things: prints `manifold_dim` → `578` (deterministic); checks an algebraic identity `print(norm((2v-v) - v))` → `0.0` (exactly reproducible); and for a numerical-correctness check it prints a **boolean**, not the residual: `print(bool(max(... norm(a-b) ...) < 1e-9))` → `True`.

## The concrete deficiency (pilot: `linalg.truncated_svd`)

```python
>>> A = np.random.randn(2,3,4, 55,70)          # unseeded
>>> U, ss, Vt = linalg.truncated_svd(A)
>>> A2 = np.einsum('...ix,...x,...xj->...ij', U, ss, Vt)
>>> print(np.linalg.norm(A - A2))
1.0428742517412705e-13      # ✗ won't reproduce — different random A, BLAS noise; this exact float is fiction to a reader
>>> print(np.all(ss >= 0.0))
True                        # ✓ reproduces
```
(The *other* `truncated_svd` examples use a fixed Hilbert matrix and **pass** — fixed input → deterministic.) Proposed fix, manifold-style:
```python
>>> np.random.seed(0)
>>> A = np.random.randn(2,3,4, 55,70)
>>> U, ss, Vt = linalg.truncated_svd(A)
>>> A2 = np.einsum('...ix,...x,...xj->...ij', U, ss, Vt)
>>> print(bool(np.linalg.norm(A - A2) < 1e-9))   # exact reconstruction (no truncation)
True
>>> print(bool(np.all(ss >= 0.0)))
True
```

## The design question for you

This is the genuine decision before we proceed, because it changes everything we do: **do we make doctests reproducible / CI-runnable, or keep them "illustrative captured values" (the current stated convention)?**

My recommendation: **reproducible**, for three reasons — (1) there's already a TODO to wire doctests into CI, and reproducible doctests give us free regression tests; (2) a non-reproducing `1.04e-13` is actively *misleading* to a reader (they'll see a different number); (3) it's demonstrably achievable — manifold/corewise/stacking already do it. This does **change the convention** (the current one explicitly permits captured-but-non-reproducing values), so it's your call.

The principles I'd expect to fall out (to refine as we go, then write up):
- **Determinism**: seed (`np.random.seed(0)`) or use fixed inputs — every shown output must reproduce.
- **Numerical checks → print a boolean** (`bool(err < tol)` → `True`), not the raw residual (robust to RNG/BLAS/platform); put the magnitude in a comment if useful.
- **Prefer printing structure** (shapes, ranks, dims, exact algebraic `0.0`) — reproducible *and* pedagogically clearer.
- (secondary) **array reprs are version-fragile** — prefer `.tolist()`/scalars/bools over printing raw arrays.

Two things to confirm before I start the pilot: (a) adopt the **reproducible/CI-runnable** goal? and (b) is `linalg.truncated_svd` a good first function to work through together, or would you rather pilot on a frontend method (e.g. something in `tucker_tensor_train.py`)?


## 🧑 Nick

(a) yes, we should make the tests reproducible. But this is a target, not a hard rule. I don't want to rule out edge cases where a non-reproducible result may be more informative. (b) I would prefer to start in the frontend modules. But first, I have some principles I would like to lay out, and discuss. Please push back on things if you have a different view, or add additional things that may be useful. 1) The purpose of the doctests is as examples, not tests. The goal is to help a user of the library understand how to use the function. 2) The doctests should illustrate "gotchas" or failure modes. For example, if a numerical function requires an orthogonal input, it may be worth creating a doctest that shows it failing when a non-orthogonal input is supplied. 3) Output structure (lengths and shapes) can often be useful to show in doctests, so that the user has concrete examples of what they will get back when they call a function. 4) doctests should be short and to-the-point. It is usually better to have two doctests that illustrate two different things, rather than to combine them.

## 🤖 Claude

These are good principles and mostly match what the data shows works (manifold). Let me go through each — agreeing, refining, and pushing where I see it differently — then add a few.

## On your four

**1) Purpose = examples, not tests.** Agree on the *priority* (optimize for a reader learning the API), but I'd sharpen the framing, because "not tests" can be read two ways and one is risky:

- The healthy reading: **examples first** — design for pedagogy, not coverage. The crucial corollary is *don't pursue coverage in doctests*: edge-case/branch coverage is `tests/`' job, and if we start adding doctests to "test" things we'll bloat the docs. So "not tests" → **keep them minimal and illustrative, leave coverage to the suite.**
- The risk: "not tests" sliding into "doesn't need to be right." Since we agreed (a) reproducible, I'd state the bar explicitly: examples are held to a **reproducible + correct** bar so they don't *mislead*. 

The reconciling insight: once reproducible and wired to CI, the doctest-CI **guards the docs (against rot), not the code**. So they stay "examples," and CI just keeps the examples honest. That's how "examples, not tests" and "reproducible/CI" coexist without contradiction.

**2) Illustrate gotchas / failure modes.** Strongly agree — and this is *already* the house philosophy (CLAUDE.md: numerical problems warn rather than enforce, paired with "an illustrative doctest showing it failing when the property is absent"). Two refinements:
- **Distinguish the two failure kinds.** Structural violations *raise* → show the **traceback** in the doctest (doctest matches `Traceback ... ValueError: ...`, optionally `# doctest: +IGNORE_EXCEPTION_DETAIL`). Numerical violations *don't raise* (warn) → show the **wrong number next to the right one** (e.g. ungauged `inner` ≠ the true HS value), so the contrast teaches the contract.
- **Label the cautionary line unambiguously** (`# WRONG: basis not orthogonal`) so nobody copies it as correct usage. The gotcha is usually its own short block (ties to your #4).

**3) Show output structure (shapes/lengths/ranks).** Agree, and it's a double win here: structure is both the most *pedagogically* concrete thing (it makes the signature's shape-comments tangible) **and** the most *reproducible* thing. So my stronger version: **when you'd be tempted to print a value, print its structure instead** (`zz[0].shape`, `.tucker_ranks`, `len(...)`) — it teaches more and it's the natural reproducible substitute for the raw-residual prints we're killing. Especially valuable for **stacked** returns, where the shape is the whole lesson.

**4) Short and to-the-point; two focused over one combined.** Agree. One mechanical nuance that makes "short" cheaper: within a single docstring, the `>>>` blocks **share a namespace and run in order**, so do the **setup once** (imports + build the object) and let each later block reuse it — rather than the current habit of re-`import numpy as np` in every block. That keeps each block one-idea-short without paying the setup tax. (Counter-consideration: a fully self-contained block is more copy-pasteable for a user; I'd resolve it by one setup block at the top of the docstring, readable top-to-bottom, over repeating it.)

## Additions I'd propose

- **Reproducibility mechanics + your escape hatch.** Default: `np.random.seed(0)` once at the top (or fixed inputs); numerical checks print `bool(err < tol)`. But per your (a) "target not hard rule": when the *raw magnitude is the lesson* (e.g. "this is machine precision"), keep it informative without lying — either **bool + magnitude-in-comment** (`True   # ~1e-13`), or, in the rare case a non-reproducible value really is most informative, mark it `# doctest: +SKIP` so it stays illustrative without breaking CI. So the hierarchy is: prefer structure → else bool(+comment) → else seeded value → else `+SKIP`.
- **Small dimensions** in examples (fast, and the shapes/values stay readable) — unless a realistic size is itself the point.
- **Avoid printing raw arrays** (NumPy reprs drift across versions); prefer `.shape`, `.tolist()`, a scalar, or a bool.

## One thing I'd explicitly *not* do
Chase the 240 failures mechanically into `True`s. The point isn't "make doctests pass" — it's "make each example teach something and not lie." Some failing examples should become a **structure print** or a **gotcha**, not just a bool. So this is genuinely a per-function rewrite (your one-at-a-time process), not a find-replace.

## Pilot
You want frontend. I'd suggest **`TuckerTensorTrain.__mul__`** as the first: its current doctests have the exact failure mode (unseeded random → `8.55e-11` residual) *and* a natural structure lesson (`xy.tucker_ranks` → ranks **multiply** — itself a "gotcha" worth foregrounding), so it exercises principles 1/3/4 and the reproducibility fix in one small rewrite. Alternatives if you'd rather: **`t3svd`** (truncation: show ranks + a real gotcha that truncation loses info) or **`inner`/`norm`** (the orthogonal/gauged precondition — principle 2's poster child).

Want me to pull up `__mul__`'s current doctests and draft the reworked version for us to react to — or pick one of the others?


## 🧑 Nick

Ok, agree on all points except partially 4. Having self-contained doctests that are runnable via copy-paste is a strong value-add. So, let's temper this a little bit. In general, we aim to make each doctest self-contained. However, if that would make the doctests excessively long, we can, in a case-by-case judgement, do the setup once and reuse it. However, this re-use is the exception not the norm.

## 🤖 Claude

Got it — that's a meaningful correction: **self-contained, copy-paste-runnable blocks are the norm**; shared setup is the case-by-case exception only when self-containment gets excessively long. So each block carries its own imports + setup + seed by default. That cleanly fits per-block seeding too (each block is an independent reproducible unit).

Let me pull up `__mul__`'s actual doctests and draft a rework — and crucially I'll **run** the draft to capture the true deterministic outputs rather than hand-writing them. First the current state:

- 🔧 **Read** `t3toolbox/tucker_tensor_train.py`

Let me draft the reworked blocks and **run them** to capture the real outputs (ranks/shapes/bools) — never hand-writing values:

- 🔧 **Bash** — Execute proposed __mul__ doctest blocks to capture true outputs

Caught a real one: the original says `xy.tt_ranks` → `(1, 6, 6, 1)`, but the actual value is **`(3, 6, 6, 2)`** — a **stale, wrong, *deterministic*** output (boundary bonds 1·3 and 1·2, not 1). So even the "passing-looking" captured value was fiction. Let me read the tail of the block, then present the rework:

- 🔧 **Read** `t3toolbox/tucker_tensor_train.py`

Read it — confirmed: `(1, 6, 6, 1)` is the captured value, actual is `(3, 6, 6, 2)`. A stale, *deterministic*, wrong output that's been sitting in the rendered docs.

Here's the reworked `Examples` section (run-verified — I executed each block to capture the outputs). **Not applied yet** — for your reaction:

**Before** (3 blocks; 2 fail to reproduce, 1 deterministic value is wrong):
```python
>>> x = t3...randn((14,15,16), (4,5,6), (1,3,2,1), stack_shape=(2,3))
>>> sx = x * 3.2
>>> print(np.linalg.norm(s*x.to_dense() - sx.to_dense()))
1.6268482531988893e-13          # ✗ unseeded → won't reproduce
...
>>> print(xy.tt_ranks)
(1, 6, 6, 1)                    # ✗ WRONG — actual (3,6,6,2)
```

**After**:
```python
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> np.random.seed(0)
        >>> x = t3.TuckerTensorTrain.randn((14, 15, 16), (4, 5, 6), (1, 3, 2, 1), stack_shape=(2, 3))
        >>> sx = x * 3.2                                  # scale a T3 by a scalar -> T3
        >>> print(bool(np.linalg.norm(3.2 * x.to_dense() - sx.to_dense()) < 1e-9))
        True

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> np.random.seed(0)
        >>> x = t3.TuckerTensorTrain.randn((14, 15, 16), (4, 5, 6), (1, 3, 2, 1), stack_shape=(2, 3))
        >>> y = np.random.randn(*(x.stack_shape + x.shape))
        >>> xy = x * y                                    # T3 * ndarray -> dense ndarray (elementwise product)
        >>> print(xy.shape)
        (2, 3, 14, 15, 16)
        >>> print(np.linalg.norm(x.to_dense() * y - xy))  # exactly the dense product
        0.0

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> np.random.seed(0)
        >>> x = t3.TuckerTensorTrain.randn((14, 15, 16), (4, 5, 6), (1, 3, 2, 1), stack_shape=(2, 3))
        >>> y = t3.TuckerTensorTrain.randn((14, 15, 16), (2, 3, 4), (3, 2, 3, 2), stack_shape=(2, 3))
        >>> xy = x * y                                    # elementwise product of two T3s -> T3
        >>> print(bool(np.linalg.norm(x.to_dense() * y.to_dense() - xy.to_dense()) < 1e-9))
        True
        >>> print(xy.tucker_ranks)                        # Tucker ranks MULTIPLY: 4*2, 5*3, 6*4
        (8, 15, 24)
        >>> print(xy.tt_ranks)                            # and the TT bonds: 1*3, 3*2, 2*3, 1*2
        (3, 6, 6, 2)
```

What each change embodies:
- **Reproducible**: `np.random.seed(0)` per block; residuals → `bool(... < 1e-9)` → `True`.
- **Self-contained** (your refined #4): each block re-imports + re-seeds + rebuilds `x`, copy-paste runnable on its own.
- **Structure over values** (#3): block 2 shows the return `shape` (teaches "T3·ndarray → dense, stack+shape"); block 3 shows the ranks.
- **Gotcha foregrounded** (#2): block 3's comments make "ranks multiply" the lesson (with the *correct* `(3, 6, 6, 2)`), and the inline arithmetic (`4*2`, `1*3`…) shows *why*.
- **Caught a bug**: the stale `(1, 6, 6, 1)`.

One judgment call to flag: block 2 keeps an exact `0.0` (it's algebraically exact, not machine noise, so it reproduces and is more informative than a bool) — an instance of your "reproducible, but show the value when it's genuinely the lesson."

Does this match what you want a doctest to be? If so I'll apply it and we can pick the next frontend function.


## 🧑 Nick

Great, this matches what I want. For the 0.0, I would change it to numerical test like the first one. I'm not sure if exactness is guaranteed here. You can change the 0.0 one, apply it, and move to the next

## 🤖 Claude

Agreed — block 2's `0.0` → a bool check (safer than assuming exactness). Applying the rework:

- 🔧 **Edit** `t3toolbox/tucker_tensor_train.py`

Applied. Let me verify just `__mul__`'s doctests pass (runs only that docstring):

- 🔧 **Bash** — Verify __mul__ doctests pass

`__mul__` passes. Now the next frontend pilot — I'd like one that exercises the **gotcha/precondition** principle (#2), which `__mul__` didn't. Let me find a good candidate among the frontend methods:

- 🔧 **Bash** — List frontend methods that have doctests

`inner` is a good principle-2 candidate (HS inner product, with the `use_orthogonalization` numerical gotcha). Let me read its docstring:

- 🔧 **Read** `t3toolbox/tucker_tensor_train.py`

`inner` is a good "make-reproducible" pilot (4 blocks, 3 failing) and lets me add a principle-2 **gotcha** (shape-mismatch → `ValueError` traceback). Let me draft and run to capture outputs/tolerances:

- 🔧 **Bash** — Run proposed inner doctest blocks to capture outputs/tolerances

Two refinements emerged from running it:
1. `inner`'s value is **~3.6e5** (large), so an absolute `< 1e-6` check would be misleadingly "tight." `np.allclose(result, reference)` is the better idiom — relative+absolute, magnitude-robust, and reads as "≈". I'll adopt **`np.allclose` for value-match checks** (and retrofit `__mul__` for consistency, since it's not committed yet).
2. The shape-mismatch gotcha cleanly raises `ValueError` → a traceback block with `+IGNORE_EXCEPTION_DETAIL`.

Retrofitting `__mul__` to `np.allclose`:

- 🔧 **Edit** `t3toolbox/tucker_tensor_train.py`
- 🔧 **Edit** `t3toolbox/tucker_tensor_train.py`
- 🔧 **Edit** `t3toolbox/tucker_tensor_train.py`

Now the `inner` rework (5 blocks: 4 reproducible + a gotcha):

- 🔧 **Edit** `t3toolbox/tucker_tensor_train.py`

Let me verify both `__mul__` and `inner` doctests pass (including the traceback gotcha):

- 🔧 **Bash** — Verify __mul__ and inner doctests pass

Both pass (23 + 31 examples, 0 failures, including the `ValueError` traceback gotcha). Let me commit these two pilots as a checkpoint:

- 🔧 **Bash** — Commit and push the two doctest pilots

Two pilots done, committed, both run-verified (0 failures). Here's where we are and the decision/path to confirm.

## Principles that have emerged (ready to write up)

- **Examples first, held to a reproducible+correct bar.** Not for coverage (that's `tests/`); doctest-CI would guard the *docs*, not the code.
- **Self-contained blocks** (imports + `seed` + setup each) — copy-paste runnable is the norm; shared setup only when self-containment gets excessively long.
- **Reproducible** via `np.random.seed(0)` or fixed inputs (target, not hard rule — `# doctest: +SKIP` is the escape hatch for a genuinely-informative non-reproducible value).
- **Value-match → `np.allclose(result, reference)`** (see below). For a *tolerance-bound* check (e.g. "truncation error ≤ bound"), `bool(norm < bound)` instead.
- **Print structure** (shapes, ranks, dims) over raw values — reproducible *and* teaches the return contract; avoid raw array reprs (version-fragile).
- **Gotchas as their own short block**: structural failure → traceback (`+IGNORE_EXCEPTION_DETAIL`); numerical → show wrong-vs-right.
- **One idea per block; short.** Magnitude-in-comment when useful. Small dims unless realism is the point. Exemplar module: `manifold.py`.

## One thing to confirm

I adopted **`np.allclose(a, b)`** as the value-match idiom, evolving from the `bool(norm(a-b) < 1e-9)` you saw on `__mul__` (and retrofitted `__mul__` to match). Reason: `inner`'s value is ~3.6e5, so an *absolute* `< 1e-6` check reads as "tight" but is actually loose relative to the magnitude — `np.allclose` (relative+absolute, idiomatic) is more honest and robust, and reads as "≈". Good to standardize on, or do you prefer keeping the explicit `bool(norm < tol)` form?

## Path forward

This mirrors the signature work: a couple of hand-pilots → write up the principles → validate the doc with a fresh agent → delegate the bulk (the ~240 failing examples, 190 in `tucker_tensor_train` alone) per-module with review. I'd suggest:

1. (optional) one or two more pilots to surface any remaining patterns — e.g. a **truncation** function (`t3svd`: the tolerance-bound `bool(norm < bound)` case + a "truncation loses info" gotcha), which neither pilot covered.
2. Write **`docs/doctest_style.md`**.
3. Validate with a no-context agent, then delegate per-module.

Want me to (a) confirm `np.allclose` and do the `t3svd` pilot next, or (b) go straight to writing `docs/doctest_style.md` from what we have?


## 🧑 Nick

Confirm allclose adoption. I would like to talk through another principle which we haven't figured out yet: what options or configurations should we add doctests for, and which should we not? There is a tension here; if a user wants to use a certain set of options, having a doctest which uses those options can be immensely useful. However, if we add doctests for every possible combination of inputs, the number of doctests will be huge (growing conbinatorially with the number of independent options). What principle or principles would you suggest here?

## 🤖 Claude

Confirmed: **`np.allclose`** for value-match (tolerance-bound checks stay `bool(norm < bound)`). 

Now your combinatorial question — good one, and I think it resolves cleanly once we lean on the earlier "examples, not coverage" principle.

## The reframe: we teach *distinct behaviors*, not cover *input combinations*

Because doctests aren't for coverage (that's `tests/`), we never need the cross-product. We need enough examples to **teach each distinct usage/behavior once**. That turns the problem from combinatorial (`2^k`) into roughly **linear in the number of options that actually change something** — and most don't earn an example at all. The governing question for any option is: *does this change observable behavior in a way a user couldn't predict from the signature + the default example?*

## What earns an option its own example (roughly in priority order)

1. **The default / common path** — always one clean example. Non-negotiable; it's what most readers come for.
2. **Options that change the output contract** — shape, type, or semantics. These are the highest-value: `include_shift=True` (changes *what* is represented), `sum_over_probes` (changes the output stack), `method='swap'` returning the same thing but... no — see below. If flipping the option changes what comes back, show it.
3. **Gotchas / preconditions** — the failure-mode block (your principle 2). `rtol` requires unstacked → show it raising.
4. **Non-obvious behavior** — an option whose effect would *surprise* a reader.

## What does NOT earn one

- **Options with no observable effect on the result**: `use_jax=True/False` gives the same numbers → mention in prose, don't double the example.
- **Choices among equivalent outputs**: `t3m(method=...)` — all three methods return the *same product*. Showing all three is repetition with no teaching value; one default example + a prose line on *when to pick which*.
- **Anything obvious from the signature** (a self-evident `axis=`, a scalar tolerance).

## The two efficiency levers

- **Vary one option at a time, against defaults.** Demonstrate A's effect with B at default, and B's with A at default — never the grid.
- **Combinations *only* when the behavior is emergent** (the options genuinely couple, so the result isn't predictable from each alone). Example: `t3m(method='swap', oversample=2)` — `oversample` only matters *for* `swap`, so its one example lives in that combined context. If options are orthogonal, say so in prose ("these compose independently") and don't write a combination example.

## Two more things

- **Contrast in one block for binary behavior-changing options.** For `include_shift` / `sum_over_probes`, a *single* block showing both values and how they differ teaches the option more efficiently than two separate blocks.
- **The long tail lives in `Parameters`/prose.** Not having a doctest ≠ undocumented — every option is still described in the docstring; doctests are reserved for the few high-value *demonstrations*. A soft per-function cap (say ~2–5 blocks) forces this prioritization.

## Concrete: `t3m`

By this principle, `t3m` (`method` × `max_tucker_ranks` × `max_tt_ranks` × `rtol` × `atol` × `oversample` — a huge grid) collapses to ~3 blocks: (1) **exact product** (default), (2) **truncation reduces ranks** (one representative — max-rank), (3) **gotcha**: `rtol` + stacked → raises. `method` and the other tolerances → prose; `oversample` → one line, or folded into the swap-method note where it actually matters. Not the cross-product.

Net principle, in one line: **one example per distinct observable behavior (one option varied at a time); combinations only when emergent; everything else to prose — capped at a handful per function.** Does that match your intuition, or would you weight any of the "earns an example" triggers differently?


## 🧑 Nick

Matches my intuition, go write docs/doctest_style.md

## 🤖 Claude

- 🔧 **Write** `docs/doctest_style.md`

Now point to it from `CLAUDE.md` and update the (now-superseded) doctest note there:

- 🔧 **Bash** — Locate the CLAUDE.md doctest note
- 🔧 **Read** `CLAUDE.md`
- 🔧 **Edit** `CLAUDE.md`
- 🔧 **Bash** — Commit and push the doctest_style doc + CLAUDE.md update

`docs/doctest_style.md` is written and `CLAUDE.md` points to it (superseding the old "illustrative captured values" note). Both committed.

## Where we are
- **2 pilots done** (`__mul__`, `inner`) — reproducible, `np.allclose`, structure-prints, a traceback gotcha; both run-verified; caught a real stale-value bug.
- **Convention captured** in `docs/doctest_style.md` (purpose, reproducibility, mechanics, gotchas, the option-coverage principle, "run-and-paste-real-output").

## Proposed path forward (mirrors the signature work)
1. **Validate the doc** with a fresh no-context agent (your doc-quality test): hand it a function with the bad patterns + the doc, see if it reworks correctly. Cheap, and it catches doc gaps *before* we point many agents at it.
2. **Delegate the bulk** per-module — the ~240 failing examples concentrate in `tucker_tensor_train` (190), `dense_t3svd` (16), `basis_variations_format` (21), `probing` (10), `linalg` (3). Each agent: read the doc, rework that module's doctests, **run them to capture real outputs**, report. Then I review (run the doctests + sanity-check teaching value/gotchas) and commit per module.

A note on review burden: doctests are higher-stakes than signatures (a wrong pasted output is a silent lie), so my review will actually *run* each module's doctests rather than just eyeball — which is straightforward and catches any hand-written or stale value.

Want me to (a) run the doc-validation probe first, or (b) skip straight to delegating module 1 (I'd start with a mid-size one like `linalg` or `dense_t3svd` to confirm the agent workflow on real doctests before turning it loose on `tucker_tensor_train`)?


## 🧑 Nick

Let's try to do another pilot. For this one, I want you to try to find a "difficult" example. One where you feel some confusion or uncertainty about what should be done based on the principles we discussed

## 🤖 Claude

Good idea — stress-testing on a hard case will find where the principles run out. Let me hunt for one I'm genuinely unsure about. The pattern that makes me uncertain: **lossy / non-unique outputs**, where the `np.allclose(result, reference)` value-match *inverts* (the result is supposed to differ). Truncation is the prime suspect. Let me find a frontend truncation function:

- 🔧 **Bash** — Find frontend truncation methods
- 🔧 **Read** `t3toolbox/tucker_tensor_train.py`

This is a genuinely hard one — `t3svd`'s doctest. Let me run it to ground the discussion in real numbers before I articulate where I'm stuck:

- 🔧 **Bash** — Run t3svd to ground the difficulties in real numbers

This is the hard one I was looking for — `t3svd`'s doctest. Grounding it surfaced several genuine uncertainties where our principles don't cleanly decide. Let me lay them out.

## The current doctest (≈14 prints)
It constructs a random T3, runs `x.t3svd()` (no truncation), and then prints — and compares by eye — the t3svd singular values against the **full dense matricization SVDs** for *every* unfolding and matricization:
```python
>>> print(full_ss_tt1)
[1.86e+02 8.77e+01 1.87e+01 1.27e-14 5.91e-15]
>>> print(ss_tt[1])
[1.86e+02 8.77e+01 1.87e+01]            # same leading values; the ~0 tail dropped (-> minimal ranks)
```
Everything here is unseeded-random **and** printed as raw float arrays, so none of it reproduces (even seeded, those digits are BLAS/platform-fragile).

## Where I'm genuinely unsure

**1. Keep or cut the singular-value correspondence?** This is the doctest's whole centerpiece, and it's a beautiful property (t3svd recovers each matricization's spectrum and drops the numerically-zero tail). But it pulls three ways against our principles at once: it's non-reproducible as raw arrays; it's *algorithm verification* (does the method recover known spectra?) which by "examples-not-coverage" belongs in `tests/` or `docs/t3svd_verification.md` (which already exists); and it's ~14 prints (way over the brevity cap). 
- My lean: **cut the full correspondence to the verification doc**, keep the doctest's lesson as "re-represents the same tensor, at reduced/minimal ranks," plus maybe *one* `np.allclose(ss_tt[1], dense_unfold_svals[:k])` → `True` as a tasteful nod.
- My hesitation: a user genuinely benefits from *seeing* that `t3svd` hands them the matricization spectra with the zero-tail gone — converting it all to `True`s loses that vividness. I don't know where you'd draw the keep/cut line. **This is the main thing I want your read on.**

**2. How do you illustrate *lossy truncation* at all?** I tried the obvious `x.t3svd(rtol=1e-1)` on a random T3 and it **truncated nothing visibly** — `err=0.0`, ranks unchanged. Reason: a random T3 has a *sharp* spectrum (a few big values, the rest ~1e-15), so any tolerance either drops only the already-zero tail (no visible effect) or nukes everything. To show "truncation = controlled approximation with reduced ranks and bounded error," you need a **graded** spectrum, which is non-trivial to construct reproducibly in a doctest. Options I see:
   - (a) **force** a reduction with `max_tt_ranks=1` (always reduces ranks) and show the reduced ranks + that it's now an approximation (`relerr > 0`) — guaranteed visible, but the error magnitude is arbitrary/data-dependent;
   - (b) for the **rtol** path, assert the *documented* error bound — `bool(norm(x - x2) <= sqrt(2d-1) * rtol * norm(x))` → `True` — which is principled and reproducible, but on a sharp-spectrum input it holds *trivially* (err≈0), so it documents the contract without *illustrating* truncation;
   - (c) build a graded-spectrum input on purpose (e.g. a smooth sampled function) so truncation gradually reduces ranks — most illustrative, but the setup is elaborate and itself needs explaining.
   - I'm unsure which is right: (b) is the honest "here's the guarantee," (a) is the clearest "look, ranks dropped," (c) is the most faithful but heaviest.

**3. The value-match principle inverts here.** For no-truncation, `np.allclose(x.to_dense(), x2.to_dense())` → `True` works. For truncation it's `False` by design — so the "right assertion" becomes either the reduced-ranks observable or the error-bound check, not value-match. That's a real extension to the doc (which currently only has value-match + tolerance-bound).

## Candidate new principles falling out
- **Algorithm-verification (does the method recover known values/spectra?) is coverage → `tests/` or a verification doc, not a doctest.** (sharpens "examples, not coverage")
- **Lossy ops**: assert the *documented* error bound when one exists (rtol → the `sqrt(2d-1)` bound); otherwise **force a clear reduction** (e.g. `max_rank=1`) and show the reduced ranks; don't print gauge/spectrum-fragile raw arrays — check the *relationship* via `np.allclose` or show the resulting ranks.

How would you call #1 (keep/cut the spectrum display) and #2 (which truncation illustration)? Once we settle those, I'll rework `t3svd` and fold the new principles into `doctest_style.md`.


## 🧑 Nick

Ok, this is a really difficult one. Good find. I am actually unsure about these questions. For 1, I want to keep some kind of illustration of this singular value corerspondence. The connection to matrix unfoldings is important, and the user may not know about it. Can we do this in another way? If we can't, here is a case where the illustration is more important than the principles. For 2, perhaps we could include both the error upper bound, and the rank upper bound (rank of truncated matrix unfoldings, as we use in the tests to check correctness).

## 🤖 Claude

Both calls make sense. For #1, there *is* a reproducible way to illustrate the correspondence — verify the relationship (`np.allclose(ss_tt[i], dense_unfolding_svals[:k])`) plus the rank/tail-drop via counts — rather than printing the fragile raw values. For #2, you referenced "as we use in the tests" — let me look at exactly what bounds the t3svd truncation test asserts, so the doctest mirrors them:

- 🔧 **Bash** — Find t3svd truncation correctness checks in the tests
- 🔧 **Read** `tests/test_tucker_tensor_train.py`

The test's bound is clear and clean: **error ≤ √(total dropped singular-value energy)** (sum of dropped σ² over all unfoldings+matricizations — the generalized Oseledets bound), and **chosen rank ≤ cap** (and ≤ the unfolding's numerical rank). Both are reproducibly illustratable. Let me draft 3 blocks and run them to capture real outputs:

- 🔧 **Bash** — Run the proposed 3-block t3svd doctest rework

Both of your asks resolve cleanly, and the rework shrinks from ~14 fragile prints to **3 focused, fully-reproducible blocks** (run-verified, real outputs shown):

**Block 1 — basic usage (same tensor, minimal ranks):**
```python
>>> x2, ss_tucker, ss_tt = x.t3svd()                       # no truncation
>>> print(np.allclose(x.to_dense(), x2.to_dense()))        # represents the same tensor
True
>>> print(x2.tucker_ranks, x2.tt_ranks)                    # but reduced to minimal ranks
(3, 4, 2) (1, 3, 2, 1)
```

**Block 2 — your #1, the singular-value ↔ unfolding correspondence (illustrated *another way*):**
```python
>>> _, ss_tucker, ss_tt = x.t3svd()
>>> # the i-th TT singular values are the singular values of the i-th dense TT-unfolding,
>>> # with the numerically-zero tail dropped (which is what reduces the TT ranks):
>>> dense_svals = np.linalg.svd(x.to_dense().reshape(5, 6 * 3), compute_uv=False)
>>> print(np.allclose(ss_tt[1], dense_svals[:len(ss_tt[1])]))   # leading values match
True
>>> print(len(ss_tt[1]), int(np.sum(dense_svals > 1e-9)))       # kept TT rank == numerical rank of the unfolding
3 3
>>> # (the Tucker singular values relate to the mode-i matricizations the same way)
```
Instead of *printing* the fragile random values, it **verifies the relationship** (`allclose` against the dense unfolding's SVD) and shows the **rank/tail-drop via counts** — the connection a user needs to learn ("these *are* the unfolding spectra; the ~0 tail is dropped"), made reproducible. One representative unfolding + a comment that the Tucker side is analogous (the full per-mode correspondence stays in the verification doc).

**Block 3 — your #2, both bounds (truncation now visibly reduces ranks):**
```python
>>> _, full_tucker_ss, full_tt_ss = x.t3svd()                     # untruncated singular values
>>> x3, _, _ = x.t3svd(max_tucker_ranks=2, max_tt_ranks=2)        # truncate
>>> print(x3.tucker_ranks, x3.tt_ranks)                           # rank bound: reduced, capped at the max
(2, 2, 2) (1, 2, 2, 1)
>>> # error bound: ||x - x3|| <= sqrt(total dropped singular-value energy)  [generalized Oseledets]
>>> dropped_sq = (sum(float(np.sum(s[r:]**2)) for s, r in zip(full_tt_ss, x3.tt_ranks))
...             + sum(float(np.sum(s[r:]**2)) for s, r in zip(full_tucker_ss, x3.tucker_ranks)))
>>> err = np.linalg.norm(x.to_dense() - x3.to_dense())
>>> print(bool(err <= np.sqrt(dropped_sq)))      # err=28.49, bound=31.36
True
```
This forces a *visible* truncation `(3,4,2)→(2,2,2)` (a small cap, not `rtol` on a sharp-spectrum random input which truncates nothing), shows the **rank upper bound** (capped) and the **error upper bound** (the √dropped-energy Oseledets bound, mirroring the correctness test — and reusing the untruncated singular values as the dropped reference). The magnitudes are in a comment.

## New principles this surfaced (to fold into `doctest_style.md`)
1. **Spectrum/gauge-fragile or random arrays → illustrate the *relationship*, not the values**: `np.allclose(result, reference)` + counts (rank), never the raw array.
2. **A representative property-illustration is pedagogy, not coverage** — fine in a doctest *if* reproducible and limited to one representative case (+ "others analogous"); exhaustive verification stays in `tests/` / the verification doc. (nuances "examples not coverage")
3. **Lossy ops**: value-match *inverts* — instead show the controlled observable (reduced/capped ranks) **and** assert the documented error bound (Oseledets √dropped-energy here; `√(2d-1)·rtol·‖x‖` for the `rtol` path).
4. **To illustrate truncation, force a clear reduction with a small max-rank** — don't rely on `rtol` against a sharp random spectrum (truncates nothing visibly).

Does block 2's "verify the relationship + show the rank" satisfy your intent for keeping the unfolding correspondence? If so I'll apply the rework and fold these four principles into the doc.


## 🧑 Nick

I agree on 1, 2, 3. For 4, I do not think this is the right move on general principle. We should develop examples which rtol will truncate. Also, I think there is a misunderstanding of the rank bound I am talking about. Please have a look at docs/t3svd_verification.md

## 🤖 Claude

Let me read the verification doc to understand the rank bound you mean:

- 🔧 **Read** `docs/t3svd_verification.md`

That's the key correction — thank you. The **rank bound is parsimony**: the chosen rank `r̂_k ≤ #{σ⁽ᵏ⁾_j ≥ τ}` (count of *original*-unfolding singular values above the threshold `τ = max(rtol·‖X̃‖, atol)`), evaluated at the `rtol`/`atol` threshold — *not* just "≤ cap." And you're right that forcing with max-rank dodges the real lesson; I should build a **graded-spectrum** input so `rtol` actually truncates. Let me construct one (a smooth tensor) and compute both bounds the way the verification doc does:

- 🔧 **Bash** — Graded-spectrum rtol truncation; verify both Oseledets error bound and the parsimony rank…

The graded (smooth) tensor makes `rtol` truncate genuinely (`(1,8,8,1)→(1,3,3,1)`), and both bounds hold — I just had a slicing bug in the rank-bound comparison. Let me re-run it aligned to confirm the exact outputs:

- 🔧 **Bash** — Confirm corrected graded-rtol truncation block outputs

Both your corrections land cleanly, and the result is **fully reproducible** (the smooth tensor is deterministic — no seed needed). Blocks 1 & 2 are as you already approved; here's the corrected **Block 3** with `rtol` truncation on a graded input and the *parsimony* rank bound:

```python
        Truncation -- a smooth tensor has gradually decaying unfolding spectra, so ``rtol``
        truncates meaningfully (a sharp random spectrum would not):

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> i, j, k = np.ogrid[1:9, 1:9, 1:9]
        >>> x = t3.TuckerTensorTrain.t3svd_dense(1.0 / (i + j + k))[0]   # exact T3 of a smooth tensor
        >>> _, full_tucker_ss, full_tt_ss = x.t3svd()                    # original (untruncated) spectra
        >>> xt, _, _ = x.t3svd(rtol=1e-3)                                # truncate at rtol
        >>> print(x.tt_ranks, '->', xt.tt_ranks)                        # rtol drops the small singular values
        (1, 8, 8, 1) -> (1, 3, 3, 1)
        >>> # accuracy: ||x - xt|| <= sqrt(dropped singular-value energy at the chosen ranks) [Oseledets]
        >>> dropped_sq = (sum(float(np.sum(s[r:]**2)) for s, r in zip(full_tt_ss, xt.tt_ranks))
        ...             + sum(float(np.sum(s[r:]**2)) for s, r in zip(full_tucker_ss, xt.tucker_ranks)))
        >>> print(bool(np.linalg.norm(x.to_dense() - xt.to_dense()) <= np.sqrt(dropped_sq)))
        True
        >>> # parsimony: each chosen rank <= #{ original singular values >= tau },  tau = rtol * ||xt||
        >>> tau = 1e-3 * np.linalg.norm(xt.to_dense())
        >>> tt_ok = all(r <= max(1, int(np.sum(s >= tau))) for s, r in zip(full_tt_ss, xt.tt_ranks))
        >>> tk_ok = all(n <= max(1, int(np.sum(s >= tau))) for s, n in zip(full_tucker_ss, xt.tucker_ranks))
        >>> print(tt_ok, tk_ok)
        True True
```

This nails both points: `rtol` genuinely truncates `(1,8,8,1)→(1,3,3,1)`; it shows the **accuracy** bound (Oseledets √dropped-energy at the chosen ranks) *and* the **parsimony** bound from `t3svd_verification.md` (chosen rank ≤ count of original singular values ≥ `τ` — not "≤ cap"). It reuses the untruncated spectra as the "original singular values" (legit, since Block 2 establishes those equal the dense unfolding spectra).

## Corrected principles for the doc
1–3 as agreed (relationship-not-values; representative-illustration-is-pedagogy; lossy ops show observable + documented bounds). **Revised 4:**

> **To illustrate tolerance-based (`rtol`/`atol`) truncation, construct a graded-spectrum input** (a smooth sampled function like `1/(i+j+k)`) so the tolerance drops *meaningful* singular values. Don't force truncation with a small `max_rank` (dodges the real behavior), and don't use `rtol` on a sharp random spectrum (truncates nothing). For truncation, show **both** documented bounds: **accuracy** (Oseledets √dropped-energy, at the *chosen* ranks) and **parsimony** (chosen rank ≤ #{original σ ≥ `τ`}).

And one meta-note for the doc: a **foundational, math-rich function (like `t3svd`) earns a richer doctest** — the brevity cap bends "within reason" when the properties (the unfolding correspondence, both truncation bounds) are themselves the value.

Shall I apply the full 3-block `t3svd` rework and fold these principles into `docs/doctest_style.md`?


## 🧑 Nick

apply it and update the doc

## 🤖 Claude

Applying. Let me read the full current `t3svd` Examples section (there's a "Stacked T3s" block too) to replace it precisely:

- 🔧 **Read** `t3toolbox/tucker_tensor_train.py`

This reveals the full `t3svd` Examples is a mess: block 1 (14 non-reproducible prints), block 2 (stacked — *missing* its expected output), block 3 (rtol — right idea with a graded/preconditioned input, but unseeded → prints random ranks), block 4 (broken — uses `x.uniform_structure` on a raw tuple). I'll replace all four with 4 clean blocks, including the **`rtol`-on-stacked → `ValueError` gotcha** (real, in the code). Let me verify the stacking block's outputs:

- 🔧 **Bash** — Verify stacked t3svd block outputs + rtol-stacked gotcha

All confirmed. Now applying the full 4-block rework (replacing the messy original):

- 🔧 **Edit** `t3toolbox/tucker_tensor_train.py`

Let me verify the `t3svd` doctest passes:

- 🔧 **Bash** — Verify t3svd doctest passes

`t3svd` passes (35 examples). Now let me fold the new principles into `docs/doctest_style.md`. Let me read the anchor points:

- 🔧 **Read** `docs/doctest_style.md`
- 🔧 **Read** `docs/doctest_style.md`
- 🔧 **Edit** `docs/doctest_style.md`
- 🔧 **Edit** `docs/doctest_style.md`
- 🔧 **Edit** `docs/doctest_style.md`

Now the "rich function earns a richer doctest" note. Let me read the Within-reason section:

- 🔧 **Read** `docs/doctest_style.md`
- 🔧 **Read** `docs/doctest_style.md`
- 🔧 **Edit** `docs/doctest_style.md`

Doc updated. Let me commit the `t3svd` rework + the doc principles:

- 🔧 **Bash** — Commit t3svd doctest rework + doc updates

Done — the hard `t3svd` pilot is reworked (35 examples, all reproducible) and `docs/doctest_style.md` now carries the four principles it surfaced.

## Where the doctest work stands
- **3 pilots done & verified**: `__mul__`, `inner`, `t3svd` — the last being the difficult case you asked for, which paid off: it forced out the *relationship-not-values*, *representative-illustration-is-pedagogy*, *lossy-ops (both bounds + graded input)*, and *rich-function* principles.
- **Convention is now well-developed** in the doc, and it caught real bugs along the way (the stale `(1,6,6,1)`; the broken `uniform_structure`/incomplete stacked blocks in `t3svd`).
- **~237 failing examples remain** across `tucker_tensor_train` (now fewer after 3 reworks), `dense_t3svd`, `basis_variations_format`, `probing`, `linalg`.

## Proposed next step (mirrors the signature workflow)
1. **Validate the doc** with a fresh no-context agent — hand it a function with the bad patterns (unseeded residuals, raw-array prints, a lossy op) + `doctest_style.md`, and check it reworks correctly (running its examples). This is cheap and finds doc gaps before we point many agents at it — and the `t3svd` experience shows doctests have more subtle traps than signatures, so it's worth doing here.
2. Then **delegate the bulk** per-module: each agent reworks one module's doctests, **runs them to paste real outputs**, reports; I review by *running* each module's doctests (a wrong pasted value is a silent lie, so I'll execute, not eyeball) and commit per module.

Want me to run the doc-validation probe next, or go straight to delegating the first module (I'd start with `linalg` or `dense_t3svd` — small, to confirm the agent workflow on real doctests before `tucker_tensor_train`)?


## 🧑 Nick

run the doc-validation probe

## 🤖 Claude

Running the doc-validation probe. Note a real constraint I expect it to hit: doctests *require running code* to capture outputs, but earlier sub-agents reported Bash was denied in their sandbox. The probe will both validate the doc's principles **and** tell us whether a no-context agent can even run the examples (which decides how delegation must work). I'll point it only at `doctest_style.md` + the `manifold.py` exemplar (a fair doc-only test), on `linalg.py` — small, and `truncated_svd` exercises reproducibility, the *relationship-not-values* sign-ambiguity case, and lossy truncation at once.

My pre-registered expectations: seed the random block; convert the 2 residual prints → `np.allclose`/`bool` `True`; **preserve** the reconstruction+orthogonality checks (not try to print sign-ambiguous `U`/`Vt`); keep the fixed Hilbert-matrix truncation examples (deterministic).

- 🔧 **Agent** (general-purpose) — Doc-validation probe: linalg doctests

Probe launched (fresh, doc-only). It's reworking `linalg.py`'s doctests and will report whether it could actually *run* them — which is the operationally important question: doctests demand run-and-paste, so if a no-context agent can't execute code, delegation has to change shape (agent reworks structure + flags unverified outputs; I fill them in by running). I'll review its decisions against the pre-registered expectations (and run the doctests myself either way) when it returns, then patch the doc for any genuine gap.

The probe is highly informative. Let me verify independently (run the doctests myself) and review the diff before reporting:

- 🔧 **Bash** — Independently verify linalg doctests + assess diff size

Independently verified: **108 passed, 0 failed** — the agent's outputs are real, not hand-written. Let me look at a couple of its *added* examples to judge quality:

- 🔧 **Bash** — Inspect quality of agent-added doctests

The probe is a strong pass, and the *added* examples are genuinely good — e.g. `left_svd_pair` checks the **product-across-the-bond preservation** (the lossless re-gauge invariant) + left-orthonormality rather than the sign-ambiguous factors, and `pad_or_truncate` teaches signed widths via `.tolist()` (deterministic). Every principle from the doc was applied correctly **and run-verified** (I independently reran: 108 passed, 0 failed).

## Two takeaways

**1. Operational (big one): a no-context agent *can* run code.** Bash worked (only some heredoc/`;`-compound forms were sandbox-denied; it wrote scripts to `/tmp` instead). So the cardinal "run-and-paste real outputs" rule is achievable in delegation — **doctest delegation is viable**, with my review re-running each module.

**2. The doc validated cleanly** — seed, `np.allclose`, structure prints, relationship-not-values for gauge/sign-ambiguous factors, the lossy/Oseledets handling, the gotcha traceback: all correct from the doc alone.

## The one policy decision it surfaced (your call)

The agent **expanded scope**: beyond fixing `truncated_svd`'s broken doctests, it **added** a default-path example to **7 previously-undocumented public functions** (`left_svd`, `*_svd_pair`, …) and even a missing **docstring** to `pad_or_truncate`. It flagged this itself. It's defensible — the doc says the default path "always earns an example" — and the additions are high quality. But it materially changes the sweep: "fix the ~237 stale examples" is bounded; "also add examples to every undocumented public function across the verified modules" is open-ended (dozens of functions) and shades toward the coverage/gold-plating we said to avoid.

My recommendation: **scope this sweep to *fixing existing* doctests** (stop the docs from lying — the bounded, high-value goal), and treat "add examples to undocumented functions" as a **separate, opt-in** pass. For `linalg` specifically, the additions are already done, correct, and valuable, so I'd **keep them** here — but I'd tell the delegated agents on the other modules "**rework existing doctests; don't add new ones to undocumented functions**" unless you want the broader coverage. I'd also tighten the doc's "always earns an example" wording so it doesn't read as "every public function must get one in this pass."

So: for the remaining modules, do you want (a) **fix-existing only** (my lean), or (b) **also add default-path examples to undocumented public functions**? And either way — keep `linalg`'s good additions and commit it?


## 🧑 Nick

Ultimately, I do want to add doctests to currently undocumented functions. However, we should fix the existing ones first. Please make a note to come back to this, then proceed with the doctest editing. You can commit linalg's good additions

## 🤖 Claude

Got it — fix-existing first, defer adding new ones. Let me commit linalg's additions, record the deferred TODO, then delegate the existing-doctest fixes.

First, commit linalg (the probe's verified work):

- 🔧 **Bash** — Commit linalg doctest rework

Now record the deferred TODO + a scoping note. Let me find the CLAUDE.md TODO section:

- 🔧 **Bash** — Find the CLAUDE.md TODO/cleanup section
- 🔧 **Read** `CLAUDE.md`
- 🔧 **Edit** `CLAUDE.md`

I'll keep `doctest_style.md` as the timeless convention and scope the *pass* via the agent prompts (and the CLAUDE.md TODO). Committing the TODO note:

- 🔧 **Bash** — Commit the doctest-sweep TODO note

Now delegate the existing-doctest fixes. Launching a wave on the three mid-size modules (`dense_t3svd`, `probing`, `basis_variations_format`) — **fix-existing-only**, run-and-paste verified. I'll handle the big `tucker_tensor_train` after this wave confirms the cadence.

- 🔧 **Agent** (general-purpose) — Fix doctests: dense_t3svd.py
- 🔧 **Agent** (general-purpose) — Fix doctests: probing.py
- 🔧 **Agent** (general-purpose) — Fix doctests: basis_variations_format.py

Committed linalg + the deferred-TODO note. The doc-validation **passed** (convention transfers cleanly to a no-context agent, and agents *can* run code — so run-and-paste delegation works), so I've launched the existing-doctest fixes for three modules in parallel: `dense_t3svd`, `probing`, `basis_variations_format` — scoped to **fix-existing-only** (the "add to undocumented functions" pass is deferred per your call and noted in `CLAUDE.md`).

When they land I'll review each by independently re-running its doctests (a wrong pasted value is a silent lie, so I execute rather than eyeball) + checking the diffs, then commit per module. After this wave I'll take on the big remaining one, `tucker_tensor_train` (its ~180 remaining failing examples), likely splitting it so a single agent isn't grinding the whole 3,700-line file at once. I'll report back with the wave review.

The probing agent is done (the other two are still editing their files — I'll review those when they complete). Let me independently verify probing and check its diff:

- 🔧 **Bash** — Verify probing doctests + check scope adherence

probing: **127 passed, 0 failed**, scope respected (no new functions documented). Let me spot-check one reworked block, then commit probing while the other two agents finish:

- 🔧 **Bash** — Spot-check probing reworked doctest quality

probing is clean (127 pass, scope respected, correct `np.allclose`+shape patterns). Committing it while the other two agents finish:

- 🔧 **Bash** — Commit probing doctest fixes

dense_t3svd agent done — and it caught **broken stale code** (dead imports `t3toolbox.common`, wrong names like `t3svd.tt_svd_dense`, nonexistent module paths) that the old doctests masked. Let me independently verify + check scope:

- 🔧 **Bash** — Verify dense_t3svd doctests + scope

dense_t3svd: **92 passed, 0 failed**, scope respected. Committing:

- 🔧 **Bash** — Commit dense_t3svd doctest fixes

Two of three committed (probing, dense_t3svd). The third (`basis_variations_format`) is still editing. While it finishes, let me do non-overlapping prep — map the remaining failing doctests in `tucker_tensor_train.py` (read-only) to plan how to split that big one:

- 🔧 **Bash** — Map remaining failing doctests in tucker_tensor_train to plan the split

All three wave agents are done. basis_variations_format also caught real bugs (stale wrong `structure` order; `bv_to_t3` and `t3_orthogonal_representations` were *failing* — tuple `==` and a `NameError` on nonexistent `t3.t3_corewise_randn`). Let me verify + commit it:

- 🔧 **Bash** — Verify and commit basis_variations_format doctest fixes

Wave 1 complete — `probing`, `dense_t3svd`, `basis_variations_format` all verified (135/92/127 pass), scope respected, and they caught **multiple real bugs** the old doctests masked (stale wrong values, `bv_to_t3`'s ambiguous `==`, `t3_orthogonal_representations`' `NameError`).

`tucker_tensor_train.py` has **161 failing examples across ~25 methods** — too big to split across concurrent agents (they'd race on the same file), so I'll use **one** agent for the whole file (conflict-free), then a follow-up if it doesn't finish. Launching it (fix-existing-only; leave the already-done `__mul__`/`inner`/`t3svd` and other passing examples alone):

- 🔧 **Agent** (general-purpose) — Fix doctests: tucker_tensor_train.py

Wave 1 of the existing-doctest fixes is **done and committed** — `linalg`, `probing`, `dense_t3svd`, `basis_variations_format` (all independently re-verified: 108/127/92/135 examples pass, scope respected). Notably, the sweep is acting as a **stale-code detector**: it surfaced and fixed several real bugs the broken doctests were hiding (a wrong `structure` order, `bv_to_t3`'s ambiguous `==`, `t3_orthogonal_representations`/`dense_t3svd`'s dead imports & `NameError`s).

The big one, `tucker_tensor_train.py` (161 failing examples), is now running as a single agent (same-file concurrency would corrupt it). When it lands I'll review by running the whole module's doctests + scope-check + spot-check, and if it didn't finish all 161 I'll run a follow-up agent on the remainder. After that, the existing-doctest sweep across the verified modules is complete — leaving the deferred "add doctests to undocumented functions" pass (tracked in `CLAUDE.md`) for whenever you want it. I'll report back with the `tucker_tensor_train` review.


## 🧑 Nick

Ok, we have run out of context. Please update CLAUDE.md and write/update a handoff note.

## 🤖 Claude

Let me check the current working-tree state (the `tucker_tensor_train` agent edits in place) so the handoff is accurate:

- 🔧 **Bash** — Capture working-tree + commit state for the handoff

The agent has driven `tucker_tensor_train` from 161 → ~23 failures but is **still running** (uncommitted, mid-edit — I won't commit partial work). Let me update CLAUDE.md and write the handoff. First the CLAUDE.md doctest bullet:

- 🔧 **Bash** — Locate the CLAUDE.md doctest bullet
- 🔧 **Read** `CLAUDE.md`
- 🔧 **Edit** `CLAUDE.md`

Now the handoff note:

- 🔧 **Write** `docs/doctest_handoff.md`

Committing the CLAUDE.md update + handoff (leaving the running agent's uncommitted `tucker_tensor_train.py` for the next session to verify and finish):

- 🔧 **Bash** — Commit CLAUDE.md + handoff; show remaining uncommitted state

Done. Checkpoint committed and pushed.

## What I did
- **`CLAUDE.md`** — updated the doctest bullet to reflect the real state: existing-doctest fixes **done & committed** for `linalg`, `probing`, `dense_t3svd`, `basis_variations_format` (+ `manifold`/`corewise`/`stacking` already conformed, and the `__mul__`/`inner`/`t3svd` pilots); `tucker_tensor_train.py` **in progress/uncommitted**; the deferred "add to undocumented functions" pass; and that the sweep doubles as a stale-code detector. Points to the handoff.
- **`docs/doctest_handoff.md`** (new) — full resume note: status, exact steps to finish `tucker_tensor_train` (run doctests → fix the last ~23 → review-by-running + scope-check → commit), the deferred pass, and the hard-won delegation lessons (one-agent-per-file to avoid same-file races; agents *can* run code via `/tmp` scripts / `dangerouslyDisableSandbox`+`PYTHONPATH`; always re-verify by running; `+IGNORE_EXCEPTION_DETAIL` for traceback gotchas).

## State to be aware of on resume
- **`t3toolbox/tucker_tensor_train.py` is left UNCOMMITTED on purpose** — it holds a still-running sub-agent's partial doctest edits (161 → ~23 failing). The next session should verify the agent's outcome, finish the remaining failures, review by running, then commit it. I deliberately did **not** commit partial/mid-edit work. (The other `??` untracked files are the pre-existing strays from the start of the session — left alone.)

Everything else from this session (the T3M swap method, the signature-style sweep across all verified modules, and the doctest convention + Wave-1 fixes) is committed and pushed.

The agent finished — **799 passed, 0 failed**, and it caught a *lot* of real bugs (nonexistent `t3.t3_corewise_randn` across ~12 methods, `up_svd_tt_core` calling a nonexistent `down_svd_tt_core`, a self-vs-self comparison, `x.to_dense(x)`, a wrong method+einsum in `right_orthogonalize_tt_cores`, stale ranks). Since it's complete and didn't commit, let me independently verify and commit it now (a clean finish beats leaving it uncommitted):

- 🔧 **Bash** — Independently verify tucker_tensor_train doctests + scope

Verified: **0 failed**, pilots untouched, balanced rework diff (281/254). Committing:

- 🔧 **Bash** — Commit tucker_tensor_train doctest fixes

Sweep complete. Let me update CLAUDE.md and the handoff to reflect that `tucker_tensor_train` is done:

- 🔧 **Edit** `CLAUDE.md`

Now the handoff note — mark `tucker_tensor_train` done and refocus on the deferred pass:

- 🔧 **Edit** `docs/doctest_handoff.md`

Committing the final status updates:

- 🔧 **Bash** — Commit final doctest-sweep status; confirm clean tree

The agent landed cleanly and it's all committed — the working tree now has **no tracked modifications** (only the pre-existing untracked strays remain).

## Final state

**The existing-doctest sweep of every verified module is complete and pushed:**
- `manifold`/`corewise`/`stacking` (already conformed), `linalg`, `backend/probing`, `backend/dense_t3svd`, `basis_variations_format`, and `tucker_tensor_train` (**161 failing → 0**, independently reconfirmed), plus the `__mul__`/`inner`/`t3svd` hand pilots.
- All run-verified (`python -m doctest <module>` clean), scope-respected (fix-existing-only; no doctests added to undocumented functions).

**The sweep paid off as a stale-code detector** — across the modules it found and fixed real bugs the broken doctests were hiding: a nonexistent `t3.t3_corewise_randn` used in ~12 `tucker_tensor_train` examples, `up_svd_tt_core` calling a nonexistent method, a self-vs-self comparison, `bv_to_t3`'s ambiguous `==`, dead imports/`NameError`s in `dense_t3svd`/`t3_orthogonal_representations`, and several stale deterministic values (wrong ranks, missing captured outputs).

## Docs updated
- `CLAUDE.md` — doctest sweep marked ✅ done; the deferred "add doctests to undocumented public functions" pass noted; and the flagged `core_shapes` (strips stack) vs `get_core_shapes()` (includes stack) inconsistency recorded for follow-up.
- `docs/doctest_handoff.md` — now reflects the sweep complete; refocused on the deferred pass.

## Left for next time (both tracked in `CLAUDE.md` / the handoff)
1. The deferred **add-doctests-to-undocumented-functions** pass (the convention's "default path always earns an example").
2. The minor `core_shapes`/`get_core_shapes` stack inconsistency — worth a look since it's an apparent code (not doctest) discrepancy.

Everything from this session — the T3M swap method, the full signature-style sweep, and the doctest convention + complete existing-doctest sweep — is committed and pushed, working tree clean.


## 🧑 Nick

Ok, we have to conclude the session for today. Thank you for your help, we got a lot done. I particularly enjoyed out collaboration developing design principles.

## 🤖 Claude

Thank you — this was a genuinely satisfying session. A quick recap of what we landed, all committed and pushed with a clean tree:

- **T3M `swap` method (c)** — the TTM→T3 generalization, gauge-managed swaps + the `oversample`/cleanup resolution of the leaf-frame tension, with the `ttm_t3m_ht_note.tex` capturing the theory (and your honest "does converting to balanced HT help?" thought experiment recorded).
- **Signature-style sweep** — every verified module, plus `docs/signature_style.md` and the "comment *is* the type" principle, validated by a no-context agent.
- **Doctest sweep** — `docs/doctest_style.md` and all verified modules' existing doctests made reproducible, which doubled as a stale-code detector and caught a pile of real bugs.

The part I enjoyed most too was the principle work — and I think the reason it went well is worth noting: you kept pushing on the *why* (the type-length-spread cost model, the lossy-op bound, "examples not coverage"), insisted we *validate the docs against a fresh agent* rather than trust them, and flagged overfitting before we baked our few examples into dogma. That combination is what turned ad-hoc choices into conventions a future collaborator (human or AI) can actually apply.

Everything's documented for the next session: the deferred "add doctests to undocumented functions" pass and the small `core_shapes`/`get_core_shapes` inconsistency are in `CLAUDE.md` and the handoff. Enjoy the rest of your day, and thanks again.
