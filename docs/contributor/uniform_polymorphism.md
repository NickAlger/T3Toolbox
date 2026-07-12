# Uniform polymorphism: when an op shares the ragged code, and when it gets a `ut3_` twin

The triage framework used to decide, operation by operation, whether a uniform operation is
**polymorphic** (one function serves ragged and uniform operands — the ragged name is the
polymorphic name, per [`../naming_conventions.md`](../naming_conventions.md)) or **uniform-specific**
(a `ut3_`/`utv_` twin). These are **lenses, not hard rules** — apply with judgment; flag edge cases
rather than forcing them. (Promoted from the uniform-rebuild plan during the docs split; the
executed placements all follow it.)

## Lens 1 — what does it RETURN?

- Returns a **read-out** — vectors, scalars, a dense array — → **polymorphic**, via mask-once
  (below). Canonical case: `inner`/`norm` *touch* the whole T3 but *return a scalar* → polymorphic.
- Returns a **T3-like object** (a T3, frame, tangent; anything restructuring) → **uniform-specific**:
  the output needs masks, and mask bookkeeping entangles with the result's structure — a shared code
  path would have to maintain masks in-flight, which is exactly the uniform-only work.

Cruder phrasing that usually gives the same answer: acts-*on*-data (polymorphic) vs
acts-*to*-self (twin).

## Lens 2 — if polymorphic, HOW: scan vs map

- **Scans over cores** (carry state along the TT — sweeps) → one polymorphic scan via `xscan`;
  `supercore` and `cores` both index as "core `i`", so the per-core body is shared.
- **Maps over cores** (independent per-core work) → **not** a polymorphic `xmap`: the uniform path
  is a single **einsum over the leading `d` axis** (vectorized — the perf win the uniform layer
  exists for). The two paths share the per-core math but not the loop.

## The mask-once mechanism, and why it works

For read-outs: apply the masks to the supercores **once, up front** (zero the garbage), then run
the core algorithm **mask-free** — the same code serves ragged and uniform. This is cheaper than
masking intermediates in flight, and it is why the polymorphic sampling cores
(`probing`/`apply`/`entries`) contain no mask logic at all: the `ut3_sampling`/`utv_sampling`
wrappers mask once, pack, and delegate.

Why it works: the supercore's leading axis is the core index
([`../uniform_supercore_layout.md`](../uniform_supercore_layout.md)), so `supercore[i]` ≈ ragged
`cores[i]` and per-core code "just works"; and zeroed padding is inert in contractions
(`0 × garbage = 0`), so the read-out's real part matches the ragged result — which is exactly the
equivalence contract ([`../uniform_equivalence_contract.md`](../uniform_equivalence_contract.md)),
and the round-trip test is what certifies each op ([`testing_strategy.md`](testing_strategy.md)).

The same boundary principle governs the sampling **vectors**: pack once at the boundary
(`pack_vectors`/`unpack_vectors`), never inside the op, and a sampling op is **strict** about
matching operand types — uniform cores with packed vectors, ragged cores with ragged vectors; a
mismatch is a structural error, never a silent auto-pack.

## The exactness argument: orthogonalization *inside* a read-out is polymorphic-safe

`norm`/`inner` may orthogonalize for stability before reading out. That orthogonalization is
polymorphic-safe by an **exactness** argument, not by mask maintenance: every orthogonalization
step is an *exact* factorization contracted along an *internal* edge —
`Q·(R·C_{i+1}) = (QR)·C_{i+1}` — so `step(T3).to_dense() == T3.to_dense()` identically, and the
padded-slot garbage **provably contracts to zero** (otherwise `to_dense` would change). SVD is fine
as the factorization (`Q = U`, `R = SVᵀ` is exact; triangularity is irrelevant), preserving the
library's SVD-only convention.

**The load-bearing caveat: exactness requires NO truncation.** The inside-a-read-out
orthogonalization must be **shape-preserving** (keep all singular values, including the padded
zeros) and touch no masks. Dropping a *nonzero* σ changes `to_dense` (wrong); dropping the
*zero/padded* σ is exact but shrinks ranks → changes masks → uniform-specific bookkeeping that a
read-out must not do. So the read-out path uses a shape-preserving exact orthogonalization,
**distinct from the mask-recomputing sweeps** (`left_orthogonalize_tt_cores` and friends, which
return T3-like objects and are uniform-specific by Lens 1). In code: the frontend
left-orthogonalizes, then `ut3_norm_orthogonalized` masks-once, squashes, and reads the norm off
the last core — no mask is recomputed anywhere on the path. Correctness here is a proof modulo
roundoff, so a smoke test suffices where the round-trip test would be redundant.

## Dispatch and verification

Ragged-vs-uniform is **inferred at the lowest level** (`is_ndarray(...)` on the operand: one
stacked supercore is uniform, a tuple of cores is ragged), exactly like the numpy/jax dispatch —
a polymorphic op needs no flag and no `ut3_` twin (`utt_` stays empty by design;
[`naming_rules.md`](naming_rules.md)). Every placement decision is ultimately certified by the
equivalence contract's round-trip test, plus the mask-strictness and garbage-robustness checks in
[`testing_strategy.md`](testing_strategy.md).
