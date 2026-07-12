# Uniform layer — implementation internals

Contributor-facing design records for the uniform layer that user docs reference but do not carry.
(Started during the docs user/dev split, S1; the S2/S3 extractions and the P9 polymorphism-lenses
section land here too.)

## Optimizer design rule: masks as loop-invariant state

(The design rule behind the library's own uniform optimizers — MC-SGD, Newton-CG, … — executed in
the U1–U6 uniform-optimizers build; the user-facing recipe is `../uniform_backend_jit_recipe.md`.)

The backend optimization functions are **designed** so the masks are **loop-invariant state,
recomputed only at rank-continuation stage boundaries** (where a recompile is correct and rare),
while the per-step jitted kernel is pure supercore work with masks as fixed constants. The
frontend's value-hashed mask holders (`../uniform_pytree_composition.md`) give the OO path this
cache-stability automatically; the backend gets it by **object reuse**. (If the finest separation
is ever wanted, `backend/fv_conversions.t3_orthogonal_representations` already returns *just the
cores*, mask-free, so a kernel could orthogonalize supercores inside and attach held masks
outside — but the bundled `ut3_orthogonal_representations` inside a close-over kernel is already
recompile-free, so this is optional.)
