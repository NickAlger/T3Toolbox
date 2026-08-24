# Why uniform orthogonalization must be SVD-based (the prefix-mask contract)

> A short "why" note. The uniform layer represents per-stack ranks as **prefix** masks `[0, rank)` —
> i.e. it asserts the *real* orthonormal frame vectors occupy the **leading** slots of each (padded)
> frame core, with `[rank, pad)` being don't-care garbage. This note records why that assertion is
> correct **only because the orthogonalization is SVD-based**, and would break under (non-pivoted or
> pivoted) QR. Pairs with `docs/uniform_masks_vs_ranks.md` and `uniform_pytree_composition.md`.

## The contract

`ufv_make_frame_masks` builds the frame masks from the structural ranks alone — `arange(pad) < rank` — and
**never inspects the core contents**. So the masks are correct iff the orthogonalization actually places
the rank-many real content in the leading slots `[0, rank)`.

## SVD delivers this; QR does not

Each orthogonalization step factors a core `B = U·(S Vᵀ)`, keeps `U` as the (orthonormal) frame core, and
pushes `S Vᵀ` into the neighbour.

- **SVD sorts singular values descending**, so the nonzero-σ (real) content lands in the **leading**
  columns of `U` — **regardless of how the rank deficiency is arranged inside `B`**. Zero-σ columns
  (padding, or orthonormal rank-completion) go to the trailing slots, which the mask zeroes. The prefix
  mask therefore identifies the real frame vectors correctly **and deterministically** (the sort is a
  function of the rank, not of where the deficiency sits).

- **QR pushes the rank deficiency into the triangular remainder `R`** (which is shipped off to the
  neighbour), while `Q` stays full-rank orthonormal in **Gram–Schmidt / input-column order** — *not*
  sorted by importance. Whether the real content ends up in `Q`'s prefix depends entirely on column
  order. With *internal* dependence — which an orthogonalization sweep cannot preclude — the zero pivot
  lands mid-stream, not trailing, so a prefix mask drops real content.

  Column-pivoted (rank-revealing) QR *could* prefix-align, but its pivot order is **data-dependent**,
  which is fatal twice: a fixed structural prefix mask can't track a data-dependent permutation, and the
  structure would change run-to-run.

### Empirical confirmation

A rank-2 matrix with *internal* dependence, columns `[c1, c1, c2, c2]`:

```
singular values        : [5.263 2.484 0. 0.]        (sorted -> rank 2 in the prefix)
SVD  prefix err |M - U[:,:2] (SVᵀ)[:2,:]| = 0.0
QR   R diagonal        : [-3.701 0. -1.507 0.]      (zero pivot at position 1 -- INTERNAL)
QR   prefix err |M - Q[:,:2] R[:2,:]|     = 2.13    (prefix drops real content)
```

## Two properties, both load-bearing

SVD gives the prefix-mask design exactly the two things it needs:

1. **Correctness** — the rank-many real content sits in the masked prefix, for *any* input arrangement.
2. **Determinism** — at fixed rank the prefix structure is identical every time. This is the bridge to
   the jit-performance story (`uniform_pytree_composition.md`): in a manifold-optimization loop the
   frame is re-orthogonalized **every step**, yet the masks come out **identical** (same prefix), so they
   are loop-invariant and neither the backend (close-over) nor the frontend (value-hashed holder) path
   recompiles. Under pivoted QR the masks would shift each iteration and you would recompile *regardless*.

## The completion at zero-σ slots (pad-safe since 2026-08)

When the **structural** rank exceeds the **numerical** rank, the SVD completes the prefix with
orthonormal vectors at the zero-σ slots. This is intentional — it is how a rank-deficient frame escapes
its stratum (the examples' "completes the rank-deficient frame with orthonormal vectors"). The
completion vectors are non-unique, but they are **traced data, not the jit cache key**, so their wobble
is invisible to jit.

For the mask to mark them real, though, they must actually **live in the real slots** — and a black-box
SVD does not promise that: the zero-σ eigenspace *contains the pad coordinates*, and LAPACK may place
the completion there, which the masks then erase — a silently lost tangent direction (review S1b; it hit
every zero-padded `resize` warm start). Since 2026-08 every kept-basis SVD in the sweeps is therefore
the **pad-safe** one (`backend.linalg.pad_safe_svd`): the masks are handed to the factorization as
data, the completion is confined to the real rows bitwise, and the prefix property and the
real-support property are enforced by the same mechanism. Deterministic (fixed sketch), one jit
compile across mask patterns.

## Code

SVD throughout: `backend/ut3_orthogonalization.py` and `backend/t3_orthogonalization.py`; the uniform
mask-aware sites route through `backend.linalg.pad_safe_svd`, the rest through `xnp.linalg.svd`; the
functions are named `down_svd_*` / `left_svd_*` / `right_svd_*`; the pad-safe SVD's full
derivation is [`../pad_safe_svd.tex`](../pad_safe_svd.tex) (+pdf). There is **no QR** in any
orthogonalization path in the sense of this note — `pad_safe_svd` uses Householder QR *internally*
(pads permuted to the trailing pivots), but its output contract is the SVD's (sorted σ, prefix
content), which is what the mask design needs.
