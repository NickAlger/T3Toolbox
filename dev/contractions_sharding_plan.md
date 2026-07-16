# `contractions.py`: stop folding the order axis into the W block — plan

_Started 2026-07-15. Source: the T3Polynomial survey, `/home/nick/repos/T3Polynomial/dev/t3toolbox-upstream-notes.md`
§"`contractions.py`: order-axis folding blocks W-sharding". **That note's survey is confirmed and its one
open caveat is now resolved — measured, not reasoned (§2).** Scope: 5 functions, a module-docstring
invariant, and a regression test. Numerically a provable no-op._

## 0. The problem, in one paragraph

Several contractions delegate to a lower-level twin by *renaming* their leading order axis `t` into the
`W` block: the callee infers its groups positionally ("everything left of `C` is `W`"), so handing it a
`t`-carrying array silently makes `W` mean `t+W`. That is numerically exact and was a deliberate
simplification — the docstrings say so ("with the order axis `t` folded into the outer W block"). But it
flattens `(t, W)` with **`t` major**, and a reshape is only sharding-free if the sharded axis is the
**major** member of the group being flattened. So a W-sharded array must be all-gathered.

The library targets GPU/jit (that is what the uniform layer is *for*), and data-parallel multi-GPU by
sharding the sample axis `W` is the natural next step. The fold blocks it, for nothing.

## 1. The invariant (the durable output — §6 S3)

> **A flatten is sharding-safe only if the sharded axis is the MAJOR (leftmost) member of the group
> being flattened.** Reshape moves no data on a device; it *reindexes* which logical elements live
> where. With `(t=2, W=4)` sharded on `W`, row-major flat index `t*4 + W`: dev0 holds `W ∈ {0,1}` →
> flat `{0,1,4,5}`; dev1 holds `W ∈ {2,3}` → flat `{2,3,6,7}`. A contiguous 2-way tiling is
> `{0..3}`/`{4..7}` — neither matches, so XLA must resolve it with a collective. Reversed, `(W=4, K=2)`:
> dev0 holds `W ∈ {0,1}` → flat `{0..3}` = exactly tile 0. Free.
>
> **So `W` may absorb blocks to its RIGHT (`K`, `C`) but must never be folded with anything on its LEFT
> (the order axis `t`, or `d`).**

This belongs in the module docstring. It is invisible today (nothing in the library shards), which is
precisely why it drifted — and why it will drift back without a written rule and a test.

**The detector** (mechanical, exhaustive):

> If a delegation's callee name **drops the leading `t`** while the caller has it, `t` has been folded
> into the W block and W is no longer major. Dropping `K` is always fine (`K` is to `W`'s right).

```
awk '/^def /{n=$0} /^    return [a-zA-Z_]+\(/{print n" -> "$0}' contractions.py
```

## 2. Evidence (measured this session — the note's open caveat, resolved)

The note flagged: *"The GSPMD claim is untested… Confirm with a two-device HLO dump on one lift before
refactoring — cheap, and it either justifies the whole exercise or kills it."* Done, on 4 virtual devices
(`XLA_FLAGS=--xla_force_host_platform_device_count=4`, jax 0.10.2), W sharded, cores replicated,
counting `all-gather` in the compiled HLO:

| site | current (folded) | explicit einsum |
|---|---|---|
| `dtWCi_dCio_to_dtWCo` | **3 all-gathers** | **0** |
| `dtWKCi_dCio_to_dtWKCo` | **3** | **0** |
| `dtWKCo_dCio_to_dtWKCi` | **3** | **0** |

**The claim is confirmed: it justifies the exercise.** Two further checks, both of which matter:

- **Numerical identity — all 5 sites are BIT-IDENTICAL to the explicit einsum** (`np.array_equal`, max
  diff exactly `0.0`). The refactor is a provable no-op, which is what makes it mechanical and safely
  testable.
- **The "leave alone" list is genuinely safe**: `dWKCi_dCio_to_dWKCo`, `WKCi_Cio_to_WKCo`,
  `tWKCi_Cio_to_tWKCo` all emit **0 all-gathers** under W-sharding. They fold `W+K`, and `K` is to `W`'s
  right, so `W` stays major. The design intent working correctly — they need a comment saying *why*, so
  nobody "fixes" them.

**Cost (measured, single-device — the library's default path):** the explicit form is a wash. numpy
`optimize=False` (what the 2-operand path actually does): folded 3896µs vs explicit 4098µs (~5% slower).
numpy `optimize=True`: 1089 vs 1181 (~8% slower). jax jit: 733 vs 705 (~4% *faster*). So: a small,
consistent, noise-level cost on the numpy path; nothing on jax. Acceptable for the sharding property —
but it is a real cost, not zero, and §7 asks whether you agree.

## 3. The five sites

All five are **passive-broadcast Tucker lifts** — `t` rides as a pure broadcast (no `trs`), which is
exactly why the fold was numerically free and why the fix is mechanical.

| # | function | line | delegates to | folds | write instead |
|---|---|---|---|---|---|
| 1 | `tWCi_KCio_to_tWKCo` | 1600 | `WCi_KCio_to_WKCo` | `t+W` | `'tWCi,KCio->tWKCo'` |
| 2 | `dtWCi_dCio_to_dtWCo` | 2519 | `dWCi_dCio_to_dWCo` | `t+W` | `'dtWCi,dCio->dtWCo'` |
| 3 | `dtWKCi_dCio_to_dtWKCo` | 2529 | `dWCi_dCio_to_dWCo` | `t+W+K` | `'dtWKCi,dCio->dtWKCo'` |
| 4 | `dtWCi_dKCio_to_dtWKCo` | 2538 | `dWCi_dKCio_to_dWKCo` | `t+W` | `'dtWCi,dKCio->dtWKCo'` |
| 5 | `dtWKCo_dCio_to_dtWKCi` | 2634 | `dWCo_dCio_to_dWCi` | `t+W+K` | `'dtWKCo,dCio->dtWKCi'` |

**The headline: the ragged layer already does this right; the uniform (`d`-prefixed) layer systematically
diverged.** Three of the four ragged lifts keep `t` explicit (`tWCi_Cio_to_tWCo`, and the two that
delegate to it); all four uniform twins fold it. Site #2's own ragged model is already the answer.

Verified exhaustive: the detector's 18 hits are these 5, four safe `K`-only/`W+K` folds, one
`_pairwise_path` false positive, and the eight `_assemble_*` helpers (which dispatch on a `keep_W` flag,
not re-grouping). External callers are clean — every `contractions.*` call in `sampling_derivatives.py`
uses the properly-named `t`-carrying entry point.

**Sites #3 and #5 fold `t+W+K`.** Splitting `t` out still leaves `W+K` fused, which is fine (`W` major).
Do not over-fix into three separate letters without a reason.

## 4. The fix

Copy the templates already in the file — do not invent:

- **`tWCi_Cio_to_tWCo` (L727)** — the explicit form: `t_shape = (tWCi.shape[0],)` held *out* of
  `W_shape`, einsum `'tWCi,Cio->tWCo'`, unflatten `t_shape + W_shape + C_shape + o_shape`. Site #2 is
  literally this plus a `d` prefix.
- **`trs_drWCa_dCaib_dsWCb_to_dtWCi` (L2493)** — the explicit `d`-prefixed form, for sites #3–#5.

Each becomes a self-contained ~15-line body with no delegation. The docstrings must lose the "folded into
the outer W block" sentence (it becomes false) and gain a pointer to the invariant.

## 5. Testing

1. **Bit-equality against the current implementation, per site, before deleting it.** The delegation is
   the oracle: capture `current(x, y)` and assert `np.array_equal` with the explicit form, over a stack
   matrix incl. empty `K`/`C`/`W` (the blocks may be empty — the `= 1 when empty` convention). Already
   spot-verified for all 5 (§2); this promotes it to a test.
2. **The sharding regression test** — the only thing that makes the invariant non-invisible. On 4
   virtual CPU devices, shard `W`, compile, assert **0 `all-gather`** in the HLO for each fixed site
   *and* for the four "leave alone" folds (which pins them as safe too). Without this, nothing stops the
   next delegation from re-breaking it; with it, the property is checked rather than asserted in prose.
   Cost: one `XLA_FLAGS` process env, ~seconds. **§7 asks whether you want this.**
3. Full suite + docs `-W` as usual. `sampling_derivatives` is the consumer; its existing numerical tests
   cover the lifts end-to-end.

## 6. Slices

1. **S1 — the 5 sites.** Explicit einsums + docstring corrections + bit-equality tests. Mechanical.
2. **S2 — the invariant.** Module docstring gets the rule (§1) + the detector; the four safe folds get a
   one-line *why-this-is-safe* comment. This is the durable half — without it S1 is a fix that regresses.
3. **S3 — the sharding regression test** (if §7.2 says yes).
4. **S4 — promote** the invariant to `docs/contributor/` (it is a durable design record, not a dev note)
   and archive this plan.

## 7. Decisions (Nick, 2026-07-15)

1. **Sharding-friendliness IS a library concern** — specifically w.r.t. the sample `W` group/stack. The
   plan proceeds.
2. **The multi-device HLO regression test is wanted** (S3). It is the difference between a documented
   invariant and an enforced one, and this issue exists *because* the invariant was invisible to every
   existing test.
3. The ~5–8% numpy cost of the explicit form (§2) rides along with (1) — it is noise-level next to the
   property it buys, and the jax path (where this matters) is a wash or slightly faster.

## 8. Adjacent: the numpy 2-operand `optimize` question — OUT OF SCOPE, and NOT the one-liner I first claimed

While measuring §2 I noticed `_grouped_einsum`'s claim:

> *"2-operand contractions are already BLAS, so they pass straight through on both."*

and first reported it as simply false with a ~2–7× one-line fix. **That was wrong** — or rather, half
right in a way that matters. Nick supplied the missing context (`optimize=True` was tried and was ~50×
*slower*), and measuring both regimes together reconciles it:

**Regime A — MULTI-operand (3+). The existing design is right, and the effect is large.** numpy's own
optimizer picks a FLOP-tied path that stays inside a single non-BLAS `c_einsum`; the forced greedy
pairwise path makes every step 2-operand and BLAS-eligible:

| contraction | opt=False | opt=True | `_pairwise_path` |
|---|---|---|---|
| `Wa,Caib,Wi->WCb` (3 ops) | 2757µs | 2789µs | **2201µs** |
| `WCa,Caib,WCi->WCb` (3 ops) | 3813µs | 2843µs | **1754µs** |
| `trs,rWCa,Caib,sWCb->tWCi` (4 ops) | 162903µs | 159744µs | **7388µs** (**22×**) |

That is Nick's 50× — confirmed in kind. `_pairwise_path` is doing real work and must not be touched.

**Regime B — 2-operand. The claim is false but the EFFECT is a defensible default**, because there is a
genuine crossover. The short-circuit hands numpy `optimize=False` → `c_einsum`, no BLAS. Forcing the path
(`_pairwise_path` already returns the correct `('einsum_path', (0,1))` for 2 operands) buys BLAS but pays
a **fixed ~20–25µs** dispatch overhead. Sweeping `W` on `'WCi,Cio->WCo'` (C=4, i=o=8):

| W | 1 | 8 | 32 | **64** | 128 | 512 | 2048 | 8192 |
|---|---|---|---|---|---|---|---|---|
| speedup of forcing the path | **0.1×** | **0.3×** | **0.7×** | 1.8× | 2.2× | 5.0× | 9.0× | 8.7× |

**It loses up to 10× below W≈64 and wins up to 9× above it.** On a degenerate contraction (C=1, i=o=1)
passthrough always wins. So this is a **regime-dependent tradeoff, not a bug with a one-line fix** — which
is exactly the shape of question that should not be settled by a laptop microbenchmark. The analytic part
is solid and portable (numpy einsum without a path is `c_einsum`, never BLAS; with a path it dispatches to
`tensordot`); the *crossover* is BLAS-build- and shape-dependent, and the library is general-purpose.

**Also not free numerically:** the BLAS path is **not** bit-identical (rel diff ~1.7e-16, one ULP —
`c_einsum` and `tensordot` sum in different orders). Within the dense-oracle tolerance, but any test
asserting bit-equality would move.

So: the docstring's *claim* needs correcting either way (it says "already BLAS"; they are not). Whether
the *behavior* should change needs a size-aware rule and an analytic argument — a real thread with a real
design question (where is the crossover, is `W` reliably large in the library's own inner loops, and does
a general-purpose library get to assume that?). **Not folded into this plan.**
