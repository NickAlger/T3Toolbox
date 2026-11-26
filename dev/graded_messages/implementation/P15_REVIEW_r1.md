NOT YET: the code is right, but the "equivalent" surviving mutant is not equivalent. The centers' tangent is `2d·X`, not `X`, and the docstrings call the centers "the point's own coordinates". Three cheap fixes.

# P15 review, round 1: the geometry returns the centers

Reviewer, 2026-11-21. This reviews the uncommitted P15 on top of `99f65dc7`, against the plan's P15, the rulings in
`LOG.md` ("P15 designed") and the handover. Everything was run single-process. The mutant copy has been deleted, and
the working tree is untouched apart from this file.

## What is right

**The four geometries.**
- Manifold returns `t3_orthogonal_representations`' second result.
- Uniform manifold returns `(Vc_sc, Hc_sc)`, the two supercores (ruled).
- The two corewise geometries return `None`, with the frame `(B, G, G, G)` unchanged.

**`aligned_padded_frame`** still returns a frame only, and uses `old, _ = self.frame(x_old)`.

**The call sites, against my own search** of `t3toolbox/`, `tests/`, `examples/` and `docs/` for every `frame(`:
- The **backend** geometry's `frame` is called in exactly three library places:
  - `geometry.py` (the aligned padded frame, `point_norm_sq`);
  - `optimizers._local_model_phases`.

  All three are updated.
- The **frontend** geometries (`MANIFOLD`, `COREWISE`, the uniform ones, `SharedGeometry`) have their own `frame`
  methods returning a `T3Frame` / `UT3Frame`, unchanged (O-4). The many `geom.frame(x)` hits in tests, docs and
  doctests are those.
- The test-side backend calls take `[0]`.
- The two test wrapper geometries delegate (`_CountingGeometry`), or do not define `frame`.
- No example calls a backend `frame`.

**The tests and runs.**
- `TestTheFrameReturnsTheCenters` checks bitwise equality with the orthogonal representations (ragged, stacked, shared;
  uniform with forced padding) and the consistency residuals.
- My run: `test_geometry`, `test_zero_reference` and `test_padded_start` pass (75); `test_geometry` on the floor (25).

**The P16 note: confirmed.** On a manifold frame, the engine's probe `z` without `Vc` equals `z` with `Vc` to 1.1e-15,
at `C = ()` and `C = (2,)`. So an oracle at a padded-start frame, which has no centers, is right without them.

## Finding: the surviving mutant is not equivalent, and the centers are not `X`'s coordinates

**The claim.** The handover keeps the mutant "`point_norm_sq` reading the centers in place of
`point_tangent(frame)`" as equivalent: "the point's tangent at its own frame **is** the centers, the same numbers by
construction". It suggests P16 or later make the change to save a sweep.

**Measured, it is false.** At `x = randn((5, 4, 6), (2, 3, 2), (1, 2, 3, 1))`:
- `UniformManifoldGeometryOps.point_norm_sq` is 4786.996, which is `|X|^2`;
- `inner(centers, centers)` is 28721.97, which is `6 |X|^2` (`2d = 6`);
- the ragged geometry gives the same pair.

**Why.**
- The dense tensor of the tangent `(frame, centers)` is `6 X`, to 3e-15: **every one of the `2d` terms reproduces
  `X`**. This is the doubled-rank representation, which is why the probe `z` can read `X` through one term's `Vc`.
- The point *as a gauged tangent*, `v_X`, puts `L_last` in the last TT slot and zeros elsewhere (`fv_base_point_tangent`).
- So the mutant makes the regularizer `1/2 lambda |X|^2` six times too large.

**Why it survives.** No test checks `point_norm_sq` against `|X|^2`. The uniform regularized-model test compares the
model with `R.value(ref, ...)`, which calls the same `point_norm_sq`. It is circular, so the mutant moves both sides.

**Fix, before P16:**
1. **A test** that `point_norm_sq(x)` equals the dense `|X|^2`, for all four backend geometries, uniform with forced
   padding. It kills the mutant.
2. **Retract the handover's P16 suggestion** in the handover and `LOG.md`. The centers cannot replace
   `point_tangent`. That is not a matter of a skipped sweep; it is a factor of `2d`.
3. **Make the docstrings precise.** In `ManifoldGeometryOps.frame` ("the point's own coordinates in the frame (its
   tangent-shaped representation)") and in the `Geometry` protocol text ("the point's own coordinates in it"), say
   instead:
   - "the centers `(Vc, Hc)`: the point's cores in the frame's doubled-rank representation, each of the `2d` terms
     reproducing `X` (so the tangent `(frame, centers)` is `2d·X`, not `X`); the point as a gauged tangent is
     `point_tangent(frame)`";
   - and keep the sentence that an oracle may read the probe `z` through `Vc`, which is right.

## Mutants

The implementer's four killed mutants stand. My only mutant was the one above, which I examined numerically rather
than re-running the suite. The suite's subset that touches the regularizer takes too long for one single-process
mutant run within the time limit, and the numeric result settles it.
