# Design rationale: `t3svd` vs `rank_adjustment_sweep`

Why T3-SVD and rank minimization are **two separate operations**, why `t3svd` makes no minimal-rank
guarantee, and why the minimizer is a directional *sweep* rather than a `minimize()` method. This is the
reasoning behind the current API — read it before "fixing" `t3svd` to return minimal ranks again.

(Companion: [`t3svd_minimal_ranks.md`](../t3svd_minimal_ranks.md) is the user-facing "what `t3svd` returns
and how to minimize"; this note is the *why*.)

## The starting point: `t3svd` was doing too much

The first version of the fix for "`t3svd` returns non-minimal ranks under truncation" (the orphan
mechanism — see the companion note) folded a rank **re-tightening** pass into `t3svd`, behind a
`minimize_ranks` flag. That bundled three concerns into one call: truncation, rank minimization, and
(implicitly) output-gauge management. It was complex and, worse, it made the **output gauge
unpredictable**.

## The gauge problem (and why reversing can't fix it)

The truncating sweep ends **left-orthogonal**. The cheapest re-tightening pass ends **right-orthogonal**
(it moves the orthogonality center the other way). The re-tighten was *gated* — it ran only when a hard
cap actually orphaned a rank. So the output gauge depended on whether an orphan happened to occur:
left-orthogonal usually, right-orthogonal when a cap bit a certain way. The user could not know which
without a numerical check (as costly as just re-orthogonalizing).

We asked whether a *reverse* (cheap — just rank-sized TT transposes) could realign the gauge. It cannot,
by a parity argument:

- A valid sweep **flips** the gauge (from a given gauge, only the flipping sweep direction is
  well-defined: L→R needs a right-orthogonal input and yields left-orthogonal; R→L the reverse).
- A reverse flips the gauge **and** the mode order.
- The no-orphan path is **one** sweep (→ left-orthogonal); the orphan path is **two** (truncate +
  re-tighten → right-orthogonal). They differ by one sweep, i.e. **opposite gauge parity**.
- To preserve the mode order you need an even number of reverses, which leaves gauge parity unchanged.

So no amount of reversing can make the two paths land in the same gauge. The only ways to a *consistent*
gauge are to **always** re-tighten (an extra sweep even when nothing is orphaned — wasteful) or to **not
re-tighten inside `t3svd` at all**. We chose the latter.

## The decision: separate the basic algorithm from minimization

- **`t3svd`** is the basic algorithm (Algorithm 10 / Oseledets' TT-SVD): orthogonalize + one
  left-to-right truncating sweep. It is **always left-orthogonal** and returns the raw sweep ranks, with
  **no minimal-rank guarantee** under truncation. One job, predictable gauge, no hidden sweeps.
- **`rank_adjustment_sweep(direction)`** is a separate, **opt-in** lossless reduction that drops
  structurally-redundant ranks. Minimization is now the caller's explicit choice.

This satisfies the three goals that were in tension: *no excessive sweeps* (`t3svd` does the minimum;
minimization is paid only when asked), *user control* (the user decides whether/when to minimize), and
*simplicity* (one fixed gauge out of `t3svd`, no option-dependent return).

## Why a directional *sweep*, not `minimize()`

A single directional reduction sweep reaches the minimal ranks **only if the input is already orthogonal
in the opposite direction**: a left-orthogonal input (e.g. a `t3svd` result) already satisfies the
forward bounds, so a `'right_to_left'` sweep adds the backward bounds and the result is fully minimal. On
a general input one sweep is a *partial* reduction. So `minimize_ranks()` would over-promise — the name
`rank_adjustment_sweep` says what it actually is, and the `direction` argument also picks the output
gauge (`'right_to_left'` → right-orthogonal, `'left_to_right'` → left-orthogonal). Compose both
directions for a guaranteed minimal result in a chosen gauge.

We chose the **single-pass** version (over a "robust" variant that orthogonalizes internally and always
minimizes) because it is cheaper for the common case (minimize a `t3svd` output — one pass) and keeps the
operation honest about being one sweep. The opposite-orthogonality **precondition is not enforced** — the
same treatment as `assume_orthogonal`; verify with `is_left_orthogonal` / `is_right_orthogonal` first.

### The ragged/uniform asymmetry on misuse

On the *wrong* direction the two layers behave differently, and this is by design:

- **Ragged** truncates each edge to `min(rows, cols)` with no cap, so it can only discard genuinely-zero
  directions: wrong direction → **under-minimized but lossless** (tensor intact).
- **Uniform** has *static* array shapes, so it must commit to the (minimal) output shape up front and
  force the supercore into it: wrong direction → **lossy** (it discards real content).

Uniform cannot match ragged's lossless-partial behavior here: its left-to-right scan structurally
requires a right-orthogonal starting gauge, so it can only *assume* the precondition (cheap, lossy if
violated) or *orthogonalize internally* (always minimal, but then it is no longer a single directional
pass). We keep the precondition required and not-enforced, consistent at the contract level, with the
failure modes documented (and doctested) on both methods.

## Other decisions

- **`assume_orthogonal` is one type** (a bool: "the input is already right-orthogonal — the form the L→R
  sweep needs"). Supporting the other gauge would mean reversing internally and returning a different
  gauge; a caller with a left-orthogonal T3 reverses it themselves. Not enforced.
- **Uniform has no `rtol`/`atol`.** A tolerance makes the truncated shape data-dependent, which the
  uniform (static-shape) layer forbids. Ragged keeps them.
- **`t3m` stays raw rounding.** It rounds via `t3svd`, so its output is now correct and within the max
  ranks but not auto-minimized. Re-minimizing every round is the caller's choice
  (`rank_adjustment_sweep`), not a hidden cost baked into `t3m`.

## See also

- [`t3svd_minimal_ranks.md`](../t3svd_minimal_ranks.md) — user-facing: why ranks may be non-minimal, how to
  minimize, the matricization/TT background.
- `backend/t3_svd.py` (`t3svd`, `rank_adjustment_sweep`), `backend/ut3_svd.py` (`ut3svd`,
  `ut3_rank_adjustment_sweep`).
