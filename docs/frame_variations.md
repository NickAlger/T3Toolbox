# The frame-variations representation (and why the cores are numbered as they are)

A tangent vector to the fixed-rank manifold is stored as a **frame** plus **variations**
(`T3Frame` + `T3Variations`, bundled as a `T3Tangent`). This note explains what those cores *mean*, why
there are `d` of each but `d+1` rank/bond indices, and why the variations behave like plain **coordinates**
— which is what makes `corewise` linear algebra (and the weighted metric, `T3FrameWeights`) work.

## A tangent is a sum of `d` terms

The key picture: a tangent vector is the **sum of `d` tensor-train diagrams**, each with one **variation
core** (in parentheses) surrounded by **frame cores**:

```
1--(H0)--R1---R2--1     1---L0--(H1)--R2--1     1---L0---L1--(H2)--1
    |    |    |             |    |    |             |    |    |
    U0   U1   U2            U0   U1   U2            U0   U1   U2
```

(plus a matching set with the Tucker variation `(Vᵢ)` in place of `Uᵢ`.) The **frame cores encode the base
point** — where the tangent space attaches to the manifold — and the **variation cores encode the tangent
direction** relative to that frame. There are four frame families and two variation families:

| frame core | symbol | role | rank on its legs |
|---|---|---|---|
| up-Tucker `Uᵢ` | `U` | the Tucker factors (retained directions) | `nU` |
| down-TT `Oᵢ` | `O` | the Tucker *complement* (where `V` varies) | `nD` |
| left-TT `Lᵢ` | `L`/`P` | left environment (left of the variation) | `rL` |
| right-TT `Rᵢ` | `R`/`Q` | right environment (right of the variation) | `rR` |
| tucker-variation `Vᵢ` | `V` | Tucker-factor direction (in the `nD` complement) | `nD` |
| tt-variation `Hᵢ` | `H` | TT-core direction | `nU`, `rL`, `rR` |

## `d` cores, `d+1` bonds — and the "extra" `Ld`, `R0`

Each family has **`d` cores** (`U0…U(d-1)`, `L0…L(d-1)`, …), but the TT **bonds** run `rL0 … rLd` and
`rR0 … rRd` — **`d+1`** of them. Mathematically there are only `d` *natural* left/right cores: in the
diagrams above, **`Ld` and `R0` are never used** (the leftmost term needs no left environment; the
rightmost needs no right environment).

So why keep them? **To give the base point a home.** A frame is meant to *remember the point it was
constructed at*, and that point is stored as a single extra variation term — the base point written as a
gauged tangent, `v_X`, whose only nonzero variation is the **last TT variation** (`= P_last`, the
`Ld`-adjacent slot). Keeping the padding `Ld` / `R0` cores means:

- there is a **canonical, non-arbitrary place** to put the base-point term (rather than picking one of the
  `d` cores by fiat), so `T3Tangent.to_t3(include_shift=True)` can fold the point back in; and
- every family has a **uniform length `d`** with uniform bond indexing, which is an enormous win for **code
  reuse and off-by-one safety** — the numbering headaches (the "`0`-th right core sits in a different place
  than the `0`-th left core") largely disappear. This was a deliberate, hard-won choice: the austere
  `d`-core math versus the practical `d+1` bookkeeping — practicality won.

**Takeaway for reading the code:** `left_ranks` / `right_ranks` have length `d+1`, but the *last* left bond
and *first* right bond are frame-internal padding, "not really part of the frame." The `d` things that are
genuinely per-mode are the variation cores and their adjacent edges.

## Gauge conditions make the variations *coordinates*

When the frame is **orthonormal** and the variations are **gauged** (`Uᵢᵀ Vᵢ = 0`; `Lᵢ` left-orthogonal
against all but the last `Hᵢ` — Appendix A.3 of the T4S paper), something powerful happens: **plain
"dumb" corewise linear algebra on the variation cores faithfully equals Hilbert–Schmidt linear algebra on
the tangent vectors they represent.** Adding variations adds tangents; `corewise_norm`/`corewise_inner`
of the variations *is* the HS norm/inner (see `T3Tangent.corewise_norm` vs `MANIFOLD.norm`). The variation
cores are, in this gauge, an orthonormal **coordinate system** for the tangent space.

That is exactly why the minimal `T3Tangent` stores only `(frame, variations)` and derives everything else,
and why the `K`/`C` stack split is *recovered* rather than stored (see
[`batching_and_stacking.md`](batching_and_stacking.md)).

## Weights are a diagonal metric on those coordinates

This is where the weighted layer ([`weighting.md`](weighting.md)) plugs in, and it clarifies the whole
picture:

- A **base point** is a *tensor*; `T3Weights` weights all its internal edges (a weighted network).
- A **tangent** is a set of *coordinates*; `T3FrameWeights` is a **diagonal metric on those coordinates** —
  one weight per variation-core edge (`d` each: `up`↔`H.nU`, `down`↔`V.nD`, `left`↔`H.rL`, `right`↔`H.rR`).
  The weighted norm is just `corewise_norm` of the weight-scaled variations; the frame stays orthonormal.

The `d`-natural-edges count is not a coincidence — it is the same `d` as the variation cores, and the
padding `Ld`/`R0` bonds (being base-point machinery, not tangent directions) correctly carry **no** weight.
Putting inverse singular values on these coordinate edges penalises the poorly-informed tangent directions
— the Grasedyck–Kramer preconditioner.

## Where to look

- The core diagrams, orthogonality, and gauge conditions: the `T3Frame` docstring (`frame_variations_format.py`).
- The tangent bundle + geometries (`MANIFOLD` HS vs `COREWISE` coordinate metric): `manifold.py`.
- The doubled-rank point+tangent embedding (base point folded into the last core): `T3Tangent.to_t3`.
- The `K`/`C`/`W` stack blocks: [`batching_and_stacking.md`](batching_and_stacking.md).
