# Target decisions

The decisions that define the target design of the sampling layer, in the order they depend on each other. This
list asks questions. It proposes no answers. It was drafted on 2026-11-15 at the maintainer's request, after he
found that the review of 2026-11-14 had leaned on preserving old behaviour and on the sequencing of the refactor.

Nothing here is about how the refactor is carried out. Migration, staging and what a consumer has to change are
planned only after the target is settled.

## Where this stands (pause of 2026-11-15, 05:53)

- **Decided on 2026-11-15:** Taylor coefficients replace derivatives, with derivative mode as one adapter at the
  outer level of fitting (2.1, 2.2). Input jets are a point and one direction (2.6). Order 0 is the one-coefficient
  case of the same functions, and the order axis is an optional axis `T` (2.7, 9.4). Three kinds with an `order`
  in place of six (2.8). One function for the plain and the jetted computation, `probe(ww, pp=None, order=0)`
  (6.6). How the direction is passed: the fitting functions take a bundled sample, the frontend methods take `pp`
  as its own argument, and `order` decides, never the type (part of 6.2). The convention for arrows on the edges,
  and the words for their direction (9.5).
- **Since the pause:** 2.4 was decided on the afternoon of 2026-11-15 (see its row), and 9.6, the notation of the
  sampling variables, at 23:23 (`SAMPLE_VARIABLES.md`). What follows describes the state at the pause.
- **Held by the maintainer at the pause:** what a tilde jet is (2.4). His lean is the residual in forward order at the API
  seam. The material is `TYPE_CATALOG.md`: the types of every variable, his picture of arrows on the edges, and a
  type checker for every contraction. In its terms the remaining choice is between turning the order leg once at
  the seed (reversed inside, one product for every grade) and giving the order leg its second orientation (forward
  inside, a correlation beside the convolution).
- **Parked for another arc:** weights on the internal edges as inner products, and orthogonality with respect to
  them.
- **Nearest open questions:** 2.4 above; 2.5 and 2.3, for which the catalog says the misfit is a choice of inner
  product on data jets and derivative mode is another such choice; 2.9; the rest of 6.2; 6.3; 3.4.

## What remains, in groups (2026-11-15, 15:34)

Of the 56 rows in sections 1 to 9, 29 are decided, 3 are guidance and 1 is parked. The other 23 are below, in
seven groups that cut across the sections. The grouping and the suggested order are Claude's, made at the
maintainer's request; they decide nothing.

| Group | Rows | Open | Reopened | To confirm | Held |
| --- | --- | --- | --- | --- | --- |
| A. Jets and the misfit | 0 | | | | |
| B. The frame and the value | 2 | | | 2 | |
| C. What a kind is, and the public surface | 7 | 4 | 2 | 1 | |
| D. One program for every layout and stack | 5 | 3 | 1 | 1 | |
| E. Plans, memory and scale | 5 | 3 | 2 | | |
| F. The fitting layer above | 3 | 1 | | 2 | |
| G. What defines correct | 1 | 1 | | | |
| Total | 23 | 12 | 5 | 6 | |

- **A. Jets and the misfit.** (Decided on the afternoon of 2026-11-15: 2.4, what a tilde jet is; 2.5, the misfit;
  2.3, the derivative-mode adapter; and, at 19:32, 2.9, how a term is carried out over the orders.)
  (And on 2026-11-16: at 01:04, 3.6, the weight inside the engine; at 01:32, 3.7, what happens to the sample
  stack on each output; at 01:36, 3.3, which adjoint sweep a tilde name refers to.) Group A is finished.
- **B. The frame and the value.** (Decided on 2026-11-16: at 01:52, 4.1, what a frame is, with the centers' part
  of 4.2; at 01:59, 4.3, the route to the value at a trial point; at 02:07, the rest of 4.2, frame
  consistency; at 02:12, 4.4, tails and boundaries.) 3.2 whether `gamma` (now `nu_L`) is
  first-class (to confirm, depends on 4.1); 4.5 where the projection, the lift and the Hessian assembly live (to
  confirm).
- **C. What a kind is, and the public surface.** 3.4 what a sampling kind is and whether a user can define one
  (reopened); 8.2 the fitting layer's kind object (open); 6.2 the remaining arguments of `sampling_variables`
  (open; the sample and the direction are decided); 6.3 the other public functions (reopened); 6.4 which
  variables are public for each kind (to confirm); 6.5 who guarantees a held variable is valid (open); 9.2
  names chosen to match things that will go (open). (Decided on 2026-11-15 at 23:23, with the notation of 9.6:
  3.1, letters and grades as the primitives.)
- **D. One program for every layout and stack.** 5.2 one set of step functions for ragged and uniform (open); 5.4
  terms generic over the stack groups (open); 5.3 sites scanned or vectorized (reopened); 5.5 association order
  inside a term (to confirm); 5.6 the dtype rule (open). The cases the order-0 assessment did not cover are here.
  The maintainer gave rules for how the functions are written on 2026-11-15, 19:50 (notes to section 5).
- **E. Plans, memory and scale.** 7.3 where the plan lives (reopened); 7.2 the defaults (reopened); 7.5 inert
  padded samples for entries (open); 7.6 automatic sizing (open); 7.7 sharding and device placement (open).
- **F. The fitting layer above.** 8.1 what `LocalModel` is (to confirm); 8.3 the optimizers' contract with a model
  and what is compiled as one program (open); 8.4 what the display and diagnostics read (to confirm).
- **G. What defines correct.** 9.3 the reference when old behaviour is not it (open).

**How the groups depend on each other.** A stands alone and continues the morning's work. B feeds C: 6.2 and 6.4
wait on 4.1 and 4.5. C feeds F: 8.1 and 8.3 need the public surface. D and E are each mostly self-contained; E's
7.3 is tied to F's 8.1. G bears on how every other answer is checked.

**The six rows marked to confirm** are earlier decisions that were not taken for reasons of compatibility.
They may go quickly, several at a time.

**Outside the list:** 8.5 shared sample vectors (parked); weights on internal edges as inner products (parked for
another arc); review finding F17 (never discussed; listed under "After the target").

## How to read the status column

| Status | Meaning |
| --- | --- |
| Decided | The maintainer decided this on principle |
| To confirm | Decided on 2026-11-14 for a reason of principle, but quickly. Confirm or reopen |
| Reopened | Decided on 2026-11-14, but the stated reason was compatibility, cost of change, or "as the library does today". That reason no longer counts |
| Open | Not decided |
| Assessed | An assessment was asked for and is done (see the notes). The decision is still to be taken |
| Parked | Set aside by the maintainer |
| Guidance | The maintainer answered with principles to guide judgement, not hard rules |

"Decision n" refers to `REVIEW_DECISIONS.md`. Sections 1, 2 and 4 are the roots: most of the rest depends on them.

## 0. Ground rules

These are the maintainer's, and they are the criteria for every decision below.

| # | Rule | Status |
| --- | --- | --- |
| 0.1 | Compatibility with today's behaviour is not a criterion. This is a breaking change with a wide blast radius, and that is accepted | Decided |
| 0.2 | The target is decided before the sequencing | Decided |
| 0.3 | Design as if writing the library today, knowing what is known now | Decided |
| 0.4 | The end state must let a consumer (T3Polynomial first) use every addition without rewriting the optimizers and without changing T3Toolbox | Decided |
| 0.5 | Code is readable statically: immutable data of basic types, no containers whose contents must be inspected, each array its own argument | Decided |
| 0.6 | The framework gives users control of performance choices. The target scale is very large problems sharded across many GPUs. Small-scale CPU timings do not decide anything | Decided |
| 0.7 | Keep it simple, and avoid state | Decided |
| 0.8 | Tolerance tests for numerical operations, exact tests for structural things | Decided |

## 1. Scope

| # | Question | Status | Depends on |
| --- | --- | --- | --- |
| 1.1 | Which layers does the target redesign: the sampling kernels only, or also the fitting layer (sampling kinds, `Problem`, `LocalModel`), the geometry interface (frame, projection, lift), and the optimizers' contract with a model? | Guidance | |
| 1.2 | Which things are fixed inputs to the design, not open to change: the ragged and uniform layouts, the numpy and jax backends, the `W`, `K` and `C` stacks, the frame's four core families, Tucker-factor sharing as a geometry wrapper? | Guidance | |
| 1.3 | How do the criteria rank when they conflict: clarity for the reader, control of memory, throughput at large scale, latency on small problems, compile time, float32 accuracy, size of the public surface? | Guidance | |

The maintainer's answers (2026-11-15). In his words they are "not hard answers but more principles to help guide
judgement".

- **1.1.** Redesign may be considered for everything downstream of sampling: the optimization and fitting layers,
  and the frontend and backend sampling functions. Things unrelated to sampling are off limits, such as the
  T3-SVD, orthogonalization, or adding two T3s. The data structures of the flagship basic types are off limits:
  T3s, frames, tangent vectors, and their uniform twins.
- **1.2.** All of these are fixed inputs: the ragged and uniform layouts, the numpy and jax backends, the `W`, `K`
  and `C` stacks, the frame's four core families, and Tucker-factor sharing. The one near-exception is the way
  sharing is handled. Sharing as a feature must be preserved in full. Another way of handling it may be
  considered only if it loses no functionality and no performance.
- **1.3.** There is no hard ranking. When criteria conflict, each case gets thoughtful consideration.

What this settles elsewhere in the list:

- **3.2 and 4.1.** The frame keeps its four core families and its data structure. So a different frame convention
  is not an option, and a frame cannot be given the center cores. What remains of 4.1 is which family is
  authoritative for the base point, and whether anything travels beside the frame.
- **5.2 and 5.4.** Both layouts and all three stacks must be served.
- **6.3.** The existing public sampling functions, frontend and backend, are in scope, so none of their names has
  to remain.
- **Section 8.** The fitting layer and the optimizers are in scope.
- **8.5.** Sharing stays as it is unless an alternative loses nothing.

The geometry boundary (the maintainer, 2026-11-15):

- **Off limits:** basic frame construction, projection, lift and retraction.
- **On the table in principle:** the geometry singleton pattern that organizes those functions for optimization
  and fitting. He is not sure it can be improved. Simpler ways he tried ran into chains of mutual contradiction:
  some forced recompiling too often, some did not respect the frontend and backend split, some could not support
  numpy and jax seamlessly.
- **To preserve, even if the geometry changes:** what concerns a specific tangent vector lives in the tangent
  vector class, and what concerns the geometry in general lives in the geometry.

What that settles: "off limits" means the signature and behaviour of the existing functions. New functions may be
added beside them. So fusing retraction with frame construction is allowed as an additive feature, and every
route in 4.3 stays open. For 4.5 nothing is proposed for the projection or the lift themselves; the question
there is only which layer calls them. Any change to the organization has to pass the three tests his earlier
attempts failed.

## 2. Jets

| # | Question | Status | Depends on |
| --- | --- | --- | --- |
| 2.1 | Taylor coefficients replace derivatives everywhere, and jet edge variables are first-class, like the plain ones | Decided | |
| 2.2 | Derivative mode is one adapter at the outer level of fitting, done by scaling and weighting. Taylor is the default and the only form underneath | Decided | 2.1 |
| 2.3 | The derivative-mode adapter is standalone conversions plus one opt-in keyword on the fitting functions. Data and validation data are divided by `t!`; the weight is multiplied by `t!`; with the keyword and no weight the weight is `t!`, so that the objective is the plain sum of squares of the derivative residuals. Nothing is converted on the way out. The sampling methods have no such keyword, and the backend is Taylor only. This behaviour of the keyword is documented with a doctest | Decided | 2.2, 6.7 |
| 2.4 | A tilde jet is an adjoint variable: a Lagrange multiplier, a dual object. The data and the residual are of the type of `z`, in forward order. The misfit is the weighted sum of squares slot by slot. The dualization is one step at the seed: the weights, then the turn of the order leg. Tilde variables are stored turned, the ones a user holds included. `w_tilde` names the seed | Decided | 2.1 |
| 2.5 | The misfit is `1/2 sum omega^2 r^2` over modes, orders and samples, on Taylor coefficient residuals. With no weight given the weight is ones; that default may change in a later release if a more robust method that "just works" is found. The weight has the shape `(d,) + T` for probe and `T` for apply and entries, with no rule for a bare vector | Decided | 2.1, 2.2 |
| 2.6 | Is an input jet always affine (`w + s p`), or can it be general? This decides whether the two-term banded form is part of the structure | Decided | |
| 2.7 | Is order 0 the one-coefficient case of the same functions, with no order axis ever created? | Decided | 2.1, 2.6 |
| 2.8 | There are three sampling kinds, `probe`, `apply` and `entries`, each with an `order` (0 for the plain computation), in place of six. The fitting layer has the same shape as the frontend methods of 6.6 | Decided | 2.7, 6.6 |
| 2.9 | A term over the orders is the truncated product of its operands' jets. One operand carries the order axis `T` through the contraction; the other is taken one coefficient at a time: two contractions when it is affine. No dense order-by-order form. Where both operands are full jets there are two forms, chosen by the user through the plan (7.1): a scan over one operand's coefficients with the other jet carried and shifted one slot per step, which is the default; and a loop with slices. The block readouts compute only the terms whose orders sum to the top | Decided | 2.4, 2.6, 7.1 |

Notes:

- **2.7, the maintainer's conditions.** Do it only if order 0 loses no performance. His first jet implementation
  tried this and could not: there were extra contractions. That was with the dense `trs` tensors. The order axis
  must not be created and then squeezed; the preferred form never creates it when it is absent, perhaps through
  the contraction engine as a missing group.
- **2.7, one fact.** `contract` already defines an uppercase letter as a group of zero or more axes, and an empty
  group vanishes. So an absent order group vanishes in a term that is linear in one jet. The products of two jets
  are where the assessment has to look.
- **2.6, decided (2026-11-15).** The jets are Taylor series in `s` of functions of the form `f(w + s p)`. The input
  is always affine: a point `w` and one direction `p`. Several directions, `f(w + s p + t q)`, are not supported.
  A curved path in the one variable, `w + s p + s^2 q + ...`, is not wanted either. So an input never carries an
  order axis, and the two-term form for a product with an input jet is part of the structure.
- **2.7, assessment** (2026-11-15, `scripts/review/order0_assessment.py`).
  - **In the explicit Taylor form, order 0 is exactly the plain program.** Checked for the probe kind on ragged
    cores with one sample axis, for `z`, `Jp` and `g`. The same functions, called with order 0, make the same 124
    contractions on the same shapes in the same order as an implementation that knows nothing about jets. No
    array has an order axis. The results are bitwise equal on numpy, and under jax the two traced programs are
    identical character for character. At orders 2 and 4 the same functions match the library's derivative kind
    to 4e-16.
  - **How.** The order is a group in each contraction string, empty at order 0; this is the engine's missing
    group, with `len_T` passed at the two terms where the operand ranks cannot pin it. Each product term carries
    the order group on one operand and takes one coefficient of the other. The loop over those coefficients, and
    the start values, are four small helpers that branch on the static `order`.
  - **Costs.** `order` becomes a static argument of every step function, or is bound once. The branch in the
    helpers is on a static integer, not on which data is present. The input jet is `w` alone at order 0 and
    `(w, p)` above, so the form of the sample depends on the order.
  - **Today's library, for comparison.** Order 0 through the derivative kind is not the plain program. It carries
    an order axis of length 1, makes one more contraction per mode in `z` and `Jp` and none more in `g`, and has
    2.5 to 3.7 times as many primitives in the traced program (concatenations, multiplications, additions). `p`
    is not read.
  - **Not assessed.** Orders above 0 were checked for correctness only: the loops over coefficients are unrolled
    and unoptimized (304 and 452 contractions at orders 2 and 4), so this says nothing about the banded and
    scanned forms of 2.9. Not covered: the uniform layout, the `K` and `C` stacks, several sample axes, apply
    and entries, the tilde-dot grade, and whether one contraction per term is itself the best plain program.
- **2.7, decided in structure (2026-11-15).** The maintainer supports making the order a grouped axis that is
  either empty or has one entry, so that structurally it acts like the batching and stacking axes. He asks for
  care with the terminology, because a batching axis means a bunch of the same thing bundled together and has
  fewer restrictions; see 9.4. The cases the assessment did not cover still have to be shown to behave the same
  way.
- **2.9.** Decision 4 dropped the option for orders, because for two full jets "vectorized over orders" is the
  dense `trs` form the library removed for its memory use. That is a reason of principle, but the measurements
  behind it were made in the derivative normalization.
- **2.8, decided** (2026-11-15, 04:11). After 6.6, Claude noted that the frontend now has one function with an
  `order` while the fitting layer still has six kinds. The maintainer: "Yes, let's make the fitting layer
  consistent." So the `*Derivatives` kinds and the kind strings `'probe_derivatives'`, `'apply_derivatives'` and
  `'entries_derivatives'` go; a kind is one of three, and the order is a separate integer. In the consumer,
  T3Polynomial's call `newton_cg(geometry, 'probe_derivatives', (ww, pp), ..., order=jet_order)` names the kind
  `'probe'`.
- **2.8, what it leaves open.** What a kind is and whether a user can define one (3.4). How the fitting functions
  take the sample: today it is `ww` for the plain kinds and the pair `(ww, pp)` for the derivative kinds, so with
  one kind the sample's contents would depend on the order. That belongs to 6.2 and 8.1.
- **2.4, the maintainer's suggestion** (2026-11-15, 04:26): "I have not figured that out yet. Maybe some insight
  can be gained from the theory of convolutions of formal polynomials."
- **2.4, Claude's assessment: a candidate definition** (`scripts/review/tilde_jet_pairing_check.py`). Not decided.
  - A jet of order `n` is an element of the ring `R = R[s] / (s^(n+1))`, and every term of the program is the
    product of `R`, the truncated convolution.
  - `R` has a pairing made from its own product: `beta(a, b) = [s^n](a b) = sum_t a[t] b[n-t]`, the top
    coefficient of the product. It is nondegenerate, and it is invariant: `beta(a b, c) = beta(a, b c)`. (`R` is
    a Frobenius algebra, and this is its form.)
  - So, with jets and tilde jets paired by `beta`, multiplication by a jet is its own transpose. The adjoint of
    every term is the same product as the term.
  - **The candidate definition.** A tilde jet is a jet. It acts on a plain jet by the top coefficient of their
    product. Slot `t` of a tilde jet is the sensitivity to the coefficient `t` orders below the top:
    `a_tilde[t] = d f / d a[n-t]`.
  - **The two storage orders in this language.** A residual in natural order, `c[t] = d f / d z[t]`, is an element
    of the dual of `R`: a polynomial in `1/s`. `R` acts on it by multiplying and keeping the non-positive powers,
    which is a correlation; that is the "second kind of product". Multiplying by `s^n` turns it into an ordinary
    polynomial, the reciprocal polynomial `s^n c(1/s)`, which is `c` reversed. The reversal is the isomorphism
    between `R` and its dual.
  - **Where the pairing shows.** Only at the two ends. At the seed, the weighted residual is reversed once. At
    the readouts, a core does not depend on `s`, so its gradient block is the top coefficient of a product; that
    is `top2` and `top3` in `order0_assessment.py`. At order 0 the reversal is the identity and the top
    coefficient is the only coefficient.
  - **The same thing as a sweep direction.** The forward program is causal in the order: slot `t` reads slots
    `<= t`. Its adjoint runs the other way, as the adjoint of a left-to-right sweep runs right to left. Stored
    reversed, it is causal again: slot 0 of a tilde jet reads only the top residual.
  - **A side result for 2.1.** In derivative coordinates the product is a binomial convolution, and the flip
    alone is not its transpose; it needs the weights `t!(n-t)!`. Taylor coefficients are the coordinates in which
    the transpose is a plain flip.
  - **Checked on numbers** at `n = 4`: the invariance of `beta` (1e-16); flip times the transposed product matrix
    times flip equals the product matrix (exactly); a chain of two products against finite differences (2e-10);
    the readout for a constant factor (2e-11); the failure of the plain flip in derivative coordinates (relative
    difference 0.6) and its repair by `t!(n-t)!` (exactly). `jet_steps_check.py` had already matched the library
    with tilde jets stored this way.
  - **What the definition does not settle.** A tilde jet means something only relative to `n`. The public
    residual of the transposes pairs slot by slot with the output (6.6); whether the tilde variables a user can
    hold or ask for are in the reversed form is part of the decision.
- **2.4, held by the maintainer** (2026-11-15, 04:44). "Let's hold off on deciding this for now. It hinges on
  whether we can come up with a more natural meaning where the reversed order follows naturally. My lean is to
  keep the residual in forward order at the API seam." 2.4 stays Assessed.
- **2.4, a second reading offered by Claude, for when he returns to it. Not decided.** A tilde jet is an ordinary
  Taylor jet in `s`: the Taylor coefficients of the plain tilde variable, when the plain transpose is run at the
  point `w + s p` for each `s`. Nothing in it is reversed. The residual enters through one polynomial,
  `c(s) = sum_t c[t] s^(n-t)`, and the gradient is the coefficient of `s^n` of the result. The reason is one line:
  `z[t] = [s^t] z(s) = [s^n] (s^(n-t) z(s))`, so `sum_t c[t] z[t] = [s^n] (c(s) z(s))`. The factor `s^(n-t)` lifts
  the order-`t` output to the top order, where one coefficient reads all of them. Under this reading the residual
  is in forward order at the API seam, as the maintainer leans; the seed step builds `c(s)` from it; and the
  reversed array is nothing but the coefficient list of `c(s)`. Checked in part 6 of
  `tilde_jet_pairing_check.py`: the plain adjoint evaluated at 13 values of `s`, with no jets anywhere, has the
  tilde jet as its Taylor coefficients (4e-13), and the gradient is its coefficient of `s^n` (2e-11).
- **2.4, the maintainer's question** (2026-11-15, 04:51): "Is it a jet or some kind of dual functional on jets?
  Maybe that is where the reversal is coming from." Claude's answer, not a decision: it is a dual functional, and
  yes, that is where the reversal comes from.
  - A tilde variable is a linear functional on jets, an element of the dual space `R*`. That is the object with
    a meaning; the two arrays are two ways of writing it down.
  - **Written in the dual basis** (forward order): slot `t` is the functional's value on `s^t`, which is
    `d f / d z[t]`. The ring acts on functionals by `(a . phi)(z) = phi(a z)`. In these coordinates that action is
    the correlation, and `s` acts as a shift down: `s . delta_t = delta_(t-1)`, `s . delta_0 = 0`. The dual has
    degrees `-n .. 0`: polynomials in `1/s`.
  - **Written by a representing jet** (reversed order): `R*` is isomorphic to `R` as a module over `R`, and the
    isomorphism is fixed by choosing the generator `[s^n]` of `R*`: every functional is `z -> [s^n](c z)` for one
    jet `c`. The dual basis element `delta_t` is represented by `s^(n-t)`. That is the reversal: a shift of
    degree by `n`, carrying degrees `-n .. 0` onto `0 .. n`. In these coordinates the action is the product.
  - **Why order 0 hides it.** At order 0 the dual of the vector space `V` is identified with `V` by the dot
    product, silently. At order `n` the dual is `V* (x) R*`, and `R*` needs its own identification with `R`. The
    dot product on coefficients is one, but it is not compatible with multiplication. The compatible one is the
    top-coefficient form, whose matrix is the flip.
  - **The second reading above is this representation**, not something separate: the polynomial `c(s)` is the
    representing jet of the residual functional. Saying "nothing is reversed" there moved the reversal into
    `c(s)`; it did not remove it.
- **2.4, the type catalog** (2026-11-15, 04:57). The maintainer: "Let's catalog the types for all edge variables
  (jet vs dual jet), where are inner products vs dual pairings, and where any Riesz maps are occurring." It is in
  `TYPE_CATALOG.md`, checked by `scripts/review/type_catalog_check.py`. Its findings, none of them a decision:
  - The letter fixes the network factor (`w`, `xi`, `mu` vectors; `eta`, `nu`, `gamma` (now `nu_L`), `z` covectors)
    in every grade. The grade fixes the order factor: plain and dot are jets, tilde and tilde-dot are dual jets.
  - The multiplier of a variable lives in the dual of that variable's space. In the network the dual is the
    partner's own space, so the partner rule absorbs it. In the order both partners are jets, so nothing absorbs
    it. That leftover dual is the reversal.
  - Inside the sweeps the order index is never contracted: plain and dot terms are the ring product, tilde terms
    are the action of the ring on its dual. It is contracted in three places: the block readouts (a dual pairing),
    the misfit and its sums (an inner product), and the seed (the Riesz map of that inner product).
  - There is one Riesz map in the sampling layer: residual to seed, `omega^2 r`. No inner product of a bond or
    Tucker-rank space is used anywhere in it. The flip is not a Riesz map: its form is not positive.
  - For 2.5: the misfit is a choice of inner product on data jets. For 2.3: derivative mode is another such
    choice and needs nothing below the seed step.
- **2.4, the maintainer's arrow picture** (2026-11-15, 05:20). Arrows on each edge, contracted into a core only
  when they point towards it; on internal edges an object is "neither plain nor dual"; plain and dual are definite
  only at the exterior, where `w` points in and `z` and `r` point out; `r` "must be converted to a plain object by
  a Riesz map. For the unjetted version, this means applying the weights. For the jetted version, this also means
  reversing the jets." His full words and Claude's reading are in section 8 of `TYPE_CATALOG.md`. In short: the
  order index is one more leg whose core is the ring product; a forward-order residual points out on it and the
  reversal turns it round; the weights are the misfit's part of the map and the reversal is the network's part;
  and the choice in 2.4 is the choice between turning the arrow once at the seed and giving the order leg its
  second orientation (the correlation), as a bond has `nu_term` beside `mu_term`. 2.4 stays held.
- **2.4, arrows on the cores** (2026-11-15, 05:26). The maintainer: give the frame cores orientations too, `L` with
  left-to-right arrows on `a` and `b` and an up arrow on `i`, `R` with right-to-left arrows; "Then every
  contraction can be type checked." Done in `scripts/review/arrow_type_check.py` and written up in section 9 of
  `TYPE_CATALOG.md`. The whole explicit program type checks (424 contractions, 176 sums). The plain and tilde frame
  variables flow with the arrows; `gamma`, `z` and the dot variables flow against them (now: `nu_L`, `z`, and the
  frame's dots `mu_R_dot`, `xi_U_dot`, `nu_L_dot`, `eta_D_dot`; in the new names, what flows against the arrows is
  what carries a basis in its name, and the outputs `z` and `Jp`); each block fits its hole. The checker rejects
  the residual put back as it is, and the weighted residual with the order not reversed. The arrows coincide with
  the frame's orthogonality conditions. A naming hazard is noted there: the library's names say where a store sits,
  and its arrow points the other way. Not a decision.
- **2.4, the convention for the arrows** (2026-11-15, 05:39). Agreed by the maintainer: the typing rule is the
  two ends of one arrow ("I agree with the first rule change, this is right"). His lean for the one convention:
  an arrow entering a core marks a dual index ("contracting puts a vector 'in'"), an arrow leaving marks a plain
  one. Claude agrees: an arrow shows the way a vector travels. It is the convention of index notation and of
  tensors as maps (TensorKit.jl); TeNPy labels the other way round, so it must be stated wherever arrows are
  used. With it every frame core, as a map along its arrows, satisfies `M M^T = I`, checked on the library in
  `scripts/review/gauge_arrow_check.py`. Section 10 of `TYPE_CATALOG.md`. The convention is not yet recorded as
  decided.
- **Parked for another arc** (2026-11-15, 05:39): the maintainer's remark that the diagonal weight matrices
  between the cores of a T3 are (pseudo-)inner products on the edges, with hidden identities on a plain T3, and
  that orthogonality for weighted networks should then be taken with respect to the weighted inner products.
  Outside this refactor (scope guidance 1.1). Notes in section 10 of `TYPE_CATALOG.md`.
- **2.4, resumed** (2026-11-15, 15:49). The maintainer: "I think we have a clear picture now. It is an adjoint
  variable with respect to a Lagrangian, and the reversal of order within the jet comes in because it is a dual
  object which points the opposite way on its edge. Are there any more subtleties here I'm not seeing?" Claude
  raised six points, none of which changes the picture; they are in section 11 of `TYPE_CATALOG.md`. The two that
  touch the decision: on the manifold frame a tilde variable is the multiplier of the dot of its partner (checked
  with jets in `scripts/review/tilde_multiplier_check.py`), and the forward-order functional at the seam and the
  turned seed inside are two arrays, so `w_tilde` has to name one of them. Awaiting his confirmation of the three
  parts of the decision: forward order at the seam; turned once at the seed and stored that way inside,
  including the tilde variables a user holds; `w_tilde` names the turned seed.
- **2.4, the maintainer's statement of it** (2026-11-15, 16:04). "My thinking is that the residual is the same type
  of object as the output z. So, if the function that takes w's and pushes them through the network is f, then
  the residual is r = z_true - f(w). We can expect the user to supply z_true in the same order as if it had been
  generated by an f-like function. The dualization comes from the misfit function ||r||^2 = <r,r> (norm which is
  weighted on base vector and has the appropriate inner product on jet involving the reversal). Does this comport
  with your suggestions, and if not, what are the differences?"
- **2.4, Claude's answer: it comports, with three precisions.** Awaiting his reply.
  - **Where the reversal is.** The misfit's inner product itself has no reversal: it is the weighted sum of
    squares slot by slot, `sum_t omega[t]^2 r[t]^2`. It cannot have one, because the reversed form
    `[s^n](r r)` is not positive (at order 1 it is `2 r[0] r[1]`). The reversal is in the map from `r` to the
    seed: the functional `<r, .>` written as a jet that can enter the network, `flip(omega^2 r)`. The two are
    tied by `sum_t omega[t]^2 a[t] b[t] = [s^n](flip(omega^2 a) b)`.
  - **The sign.** The library and the prototypes use `r = f(w) - z_true`. With the other sign the gradient
    changes sign.
  - **Two seams.** In fitting, the user supplies the data and the weight, both in forward order and of the type
    of `z`; the dual object never crosses the seam. In the frontend transposes, the argument is an array of the
    type of `z` in forward order, and the function applies the plain slot-by-slot inner product, which is the
    identity on the array, then the turn. Under the maintainer's view that argument is a residual, not a tilde
    object, and `w_tilde` is what the dualization produces.
- **2.4, decided** (2026-11-15, 16:10). On Claude's three precisions the maintainer answered: "1. Agree. 2. No
  preference of z-f(w) vs f(w)-z. Keep the libraries sign convention, whatever it is. If the library is
  inconsistent, you can use the convention from your prototype. 3. Ok." What is fixed:
  - The data and the residual are of the type of `z`: jets in forward order. The user supplies the data in the
    order in which the function would have produced it.
  - The misfit's inner product is slot by slot, `sum_t omega[t]^2 r[t]^2`, with no reversal in it.
  - The dualization is one step at the seed: the weights, then the turn. Every tilde variable is stored turned,
    including those a user holds. Its pairing with a jet is the top coefficient of the product.
  - `w_tilde` names the seed. The argument of the frontend transposes is a residual, of the type of `z`, in
    forward order; the function applies the plain slot-by-slot inner product (the identity on the array) and the
    turn. Its present name `ztildes` is then a misnomer; renaming it is left to the naming questions.
  - **The sign** is the library's, `r = S(x) - data`. Checked: the library is consistent (`backend/fitting.py`,
    `backend/optimizers.py`), and the prototypes use the same sign.
  - This also settles the first half of 2.5, what the weight multiplies. The default weight is still open there.
- **2.5, opened** (2026-11-15, 16:10 to 16:19). After 2.4 only two parts remain: the default weight, and the shape
  of the weight under one kind. Claude recommended ones on the Taylor coefficients, and a weight of shape
  `(d,) + T` for probe and `T` for apply and entries, with no rule for a bare vector. The maintainer did not
  answer these. He set out three ideas that should guide the choice: the aim is to approximate a function with
  a polynomial; the importance of each term depends on a length scale, for which 1.0 is the natural choice; and,
  empirically, equalizing the influence of each term helps the fit a lot. He asked for the consumer's record
  first.
- **2.5, the survey** (`T3POLYNOMIAL_SCALING_SURVEY.md`). T3Polynomial today: whitened coordinates, unit probes,
  centering, an input scale `kappa` (automatic, an estimate of the Taylor radius), and block-equalizing weights
  `1 / (norm of the block)` per mode and order, with exact zeros for dead blocks. The history from per-order
  weights (July) through block weights, the selection score, centering and `kappa` (September), the deferred
  per-sample weight, and the negative result of whitening the directions. Claude's observations there, none a
  decision: under block-equalizing weights derivatives and Taylor coefficients give the same loss; the length
  scale is already an input, the length of the direction `p`; `kappa` acts on conditioning and is not a weight;
  equalizing is a policy computed from data, for which the library supplies the mechanism; a zero weight must
  remove a block exactly; a weight per sample has been wanted and is deferred in T3Toolbox
  (`dev/data_shaped_residual_weight_reviews.md`).
- **2.5, the maintainer's lean on the architecture** (2026-11-15, 16:33). "The big picture idea is that we want to
  give hooks to allow the user to play with different approaches for their problem, similar to our approach for
  performance options. It is not just a question of default, but also a question of code architecture. My lean
  would be to provide standalone functions which prepare the data and create weights, which the user can do on
  their own before passing the data to our fitting functions. We can also create nicety kwarg options which do it
  for the user within the fitting, by calling the preparation functions from within. For the default, I am torn
  between do nothing, and do everything." For nothing: the fit is likely to sit inside a rank continuation loop,
  and the data should be prepared once before it. For everything: the fit "just works" for a user who is not an
  expert, where raw data is "likely to have the fit immediately fail badly, when some simple scaling and
  equalization could have made it work."
- **2.5, Claude's analysis of that lean.** Awaiting his choice; nothing decided.
  - **Three classes of preparation.** (a) Those that change only the loss: a weight, the conversion between
    derivatives and Taylor coefficients, the length of the direction `p`. Nothing has to be undone afterwards. (b) Those
    that change the unknowns: a rescaling of a mode's input coordinates, as `kappa` does to the points. The
    starting point and the fitted tensor are then in other coordinates, and the function comes with an inverse.
    (c) Those that belong to the problem: centering, the homogenization coordinate, the pivot from measured jets
    to targets. These stay in the consumer.
  - **A rule for what may be done inside a fit:** only class (a), because it does not change the meaning of the
    starting point or of the tensor returned. Classes (b) and (c) stay standalone.
  - **For the misfit, "everything" is one thing.** Under equalizing weights the conversion and the length of the
    direction cancel. So the default is a choice between ones and the equalizing weight.
  - **A layered default, suggested.** The backend does nothing: the weight is what it is given, and none means
    ones. The frontend fitting functions equalize when no weight is given. An explicit weight always wins, so a
    rank continuation loop prepares once and passes it, as T3Polynomial does today. The library already has this
    split: the frontend owns the weight contract (`fitting._canonical_weight`) and the backend is lenient.
  - **The hazard of equalizing by default:** a block of data that is zero or nearly zero. T3Polynomial's record
    is that a floor turned a dead block into a weight of 1e12. The standalone function needs a stated rule.
  - **2.3 has the same shape.** The derivative-mode adapter of 2.2 is a standalone conversion with a keyword
    that calls it.
- **2.5, the default weight, decided** (2026-11-15, 16:49). The maintainer: "Let's go with do-nothing (ones) by
  default, with an asterisk that the default is subject to change in a later release if and when we figure out a
  more robust 'just works' method." So with no weight given the misfit is the plain sum of squares of the Taylor
  coefficient residuals, at both the frontend and the backend. The documentation of the default has to carry
  that asterisk. The layered default Claude suggested is not taken.
- **2.5, what is still open.** The shape of the weight under one kind (Claude's recommendation: `(d,) + T` for
  probe, `T` for apply and entries, no rule for a bare vector). 2.5 stays open until that is answered.
- **Carried to other rows, not decided.** The maintainer's lean, with Claude's agreement: standalone functions
  that prepare data and create weights, and keywords in the fitting functions that call them (now opt-in, since
  the default does nothing). Which functions exist is 6.3; the keywords are part of 6.2 and 8.x. Claude's
  suggested rule, that a keyword inside a fit does only loss-only preparation, and the rule the equalizing
  function needs for a zero or nearly zero block, are also unconfirmed.
- **2.5, decided** (2026-11-15, 17:00). The maintainer: "Agree with architecture and keyword-inside-fit rule, and
  shape weight proposal." With the default of 16:49 this closes 2.5. The weight is `(d,) + T` for probe (so `(d,)`
  at order 0 and `(d, order+1)` above) and `T` for apply and entries; a bare vector has no special meaning, and
  the user broadcasts a per-order or per-mode weight themselves. Claude's question whether a mode dimension of 1
  is still accepted and broadcast was not answered; the simpler reading, the exact shape only, is taken, to be
  corrected if wrong. The architecture and the rule for keywords are recorded as 6.7.
- **2.3, a proposal and an example** (2026-11-15, 17:04 to 18:32). Claude's proposal for the adapter, in the
  terms of 6.7: standalone conversions between Taylor coefficients and derivatives (slot `t` times or divided by
  `t!`), for anything of the type of `z`; a separate conversion for the weight, which goes the opposite way
  because it is a dual object; one opt-in keyword on the fitting functions that converts the data, the validation
  data and the weight on the way in and nothing on the way out; no such keyword on the sampling methods; the
  backend Taylor only. The maintainer asked to see a minimal example of it in use; it was shown in the
  conversation. A point the example raises: with the keyword and no weight, the weight has to be `t!` per order,
  so that the objective is the plain sum of squares of the derivative residuals. Checked on numbers: the
  conversions reproduce the derivative-mode objective exactly, with a weight and without. Not decided.
- **2.3, decided** (2026-11-15, 18:37). The maintainer, on the form of the adapter and its example: "Yes, this is
  good." On the keyword changing what "no weight" means: "yes, but let's make sure to document clearly. We can
  have a doctest example illustrating this behavior, showing what happens to the data in each case".
  - Three standalone conversions: derivatives to Taylor coefficients (slot `t` divided by `t!`), the reverse, and
    a derivative-mode weight to a Taylor weight (slot `t` multiplied by `t!`).
  - One opt-in keyword on the fitting functions that applies them to the data, the validation data and the
    weight. With the keyword and no weight, the weight is `t!` per order.
  - Required: clear documentation of that, with a doctest that shows what happens to the data in each case. A
    draft that runs is `scripts/review/adapter_doctest_draft.py` (three cases: Taylor data; derivative data with
    a weight; derivative data with no weight).
  - Not decided: the names of the functions and the spelling of the keyword (`jet_form='derivatives'` in the
    draft), and what the display shows in derivative mode (8.4).
- **2.9, decided** (2026-11-15, 19:08). The maintainer: "I confirm 2.9 as stated. However, I want to implement with
  a scan rather than a loop. Although the 4th order polynomial is my target, there can be other users with
  different problems who want higher order polynomials; for example to get higher accuracy on smaller problems."
  - Confirmed as stated: every term over the orders is the truncated product; one operand carries `T` through the
    contraction and the other is taken one coefficient at a time; no dense order-by-order form; no user option;
    the block readouts compute only the terms whose orders sum to the top.
  - His addition: where the second operand is a full jet, its coefficients are taken by a scan, not an unrolled
    loop, so that the compiled program does not grow with the order. Claude had proposed leaving that to group D.
  - Where a scan is needed: the products of two full jets, which are in the tilde and tilde-dot steps and their
    readouts. A product with an affine operand (`xi`, `w`, `xi_dot` (now `xi_U_dot`; the raw `xi_dot` is affine
    too)) is two contractions and needs none. At order 0 there is no order axis and no scan (2.7).
  - What a scan costs (`scripts/review/order_scan_check.py`). The body must have the same shapes at every
    iteration, so the truncation cannot be done by slicing; it is done by a shift along the order axis and a
    mask. The scan matches the explicit sum to rounding at orders 1 to 16, under jit. The traced program has the
    same size at every order (77 lines), where the unrolled loop grows from 56 lines at order 1 to 491 at order
    16. The mask throws work away: the scan does `(order+1)^2` contractions of one pair of orders where the
    triangle has `(order+1)(order+2)/2`, a factor 1.67 at order 4 that tends to 2.
- **2.9, the scan reconsidered** (2026-11-15, 19:14). The maintainer, on seeing that the scan does up to twice the
  contractions: "Oh hold on. We need to think more about the scan. Performance is important. Can you show me
  pseudocode of representative contractions, loop vs scan, showing how the scan doubles the contractions?" The
  parts of 2.9 confirmed as stated stand. How a full operand is taken is open again, so the row is marked to
  confirm.
- **2.9, how the library does it today** (2026-11-15, 19:22). The maintainer: "How does this compare with how the
  code currently implements the analogous operation for convolution of jets?" Read in
  `backend/sampling_derivatives.py` at 2026.5.0 (the note "The jet kernels" and `_eta_jets_order_step`,
  `_adj_src_order_step`, `_mu_jets_step`, `_adj_prop`):
  - **An affine operand:** the two shifted copies of the full jet are stacked on a pair axis and contracted
    against the two coefficients in ONE contraction. No scan.
  - **Two full jets:** a scan over the order `r` of one factor. Each step contracts that one order slice with the
    core, then contracts the result against ALL slots of the other jet, which rides the carry and is shifted one
    slot per step. The truncation comes from the zeros the shift brings in. No traced shift, no mask, no dynamic
    slice. The binomial weights come from the same shift; for Taylor coefficients they are gone.
  - **It is a real scan only on the uniform layout** (`jax.lax.scan`); on the ragged layout it is a Python loop,
    which unrolls under jit.
  - **The reason the code gives is memory**, not the size of the program: only one order slice of the spatial
    product is live at a time. Its docstring records a measured 14 to 28 times smaller peak than the vectorized
    dense form, constant in the order, and says: "The order loop being a real scan is what forces this -- an
    unrolled loop does not."
- **2.9, a correction to Claude's comparison.** The first table counted "pairs of orders" as if every contraction
  cost the same. Two contractions are involved. The expensive one is the core with one coefficient
  (`W ra n rb` multiplications); the cheap one pairs that product with one coefficient of the other jet
  (`W ra rb`, a factor `n` fewer). Counted properly (`scripts/review/order_scan_check.py`):

  | Form | Multiplications | Traced program |
  | --- | --- | --- |
  | Loop over `r`, with slices | `P E + P(P+1)/2 c` | grows with the order: 143 lines at order 4, 491 at order 16 |
  | Scan over `r` with the other jet carried and shifted one slot per step (the library's scheme, in Taylor form) | `P E + P^2 c` | 34 lines at every order |
  | Scan over a table of the needed pairs | about `P(P+1)/2 (E + c)` | 74 lines at every order |

  So the scan costs more than the loop only in the cheap contraction: 6 percent in all at order 4 with bonds of
  64 and a mode rank of 32, 20 percent at order 16; 29 percent at order 4 when every rank is 4. "The scan doubles
  the contractions" was true of the cheap ones only. The pair table repeats the EXPENSIVE contraction for every
  pair and costs 2 to 7 times the loop: Claude's recommendation of it is withdrawn. The scan with a shift and a
  mask that Claude wrote first is the library's scheme done less well (a traced shift and a mask where a carried
  shift needs neither). All three forms give the explicit sum under jit. Nothing was timed.
- **2.9, decided: two forms, chosen through the plan** (2026-11-15, 19:32). The maintainer: "Let's have two
  options, controllable by the user through the Plan abstraction. The analog of the libraries current scheme, and
  a loop with slices. The scan is the default. In the plentiful memory regime, I can imagine jax's jit compiler
  may find optimizations which increase speed,if it can see the whole loop."
  - **The scan (the default).** The library's present scheme in Taylor form: a scan over the coefficients of one
    operand; each step contracts that one coefficient with the core and then with every slot of the other jet,
    which rides the carry and is shifted one slot per step. One coefficient of the expensive product is live at
    a time, the compiled program has one size at every order, and it does the square of the cheap contractions
    where the triangle would do.
  - **The loop with slices.** A Python loop over the same coefficients, with the truncation done by slicing. It
    does only the triangle, the compiler sees the whole loop, the compiled program grows with the order, and
    nothing is promised about peak memory (the library's own note on its scan: "an unrolled loop does not" bound
    it).
  - This replaces "No user option" in the row. "No dense order-by-order form" stands: neither option is that
    form. It also supersedes decision 4 of 2026-11-14 ("no option for orders; drop `Plan.orders`") and section
    4.7 of `DESIGN_AFTER_REVIEW.md`: the option that decision removed was scan against the dense form, and the
    dense form stays out. The `SamplePlan` of that document's section 5 gains one field. It is one more of the performance options the framework hands to the user, and like the others it
    changes how the computation is carried out and never what is computed.
  - **Claude's readings, not said by the maintainer:**
    - The setting is one field of the plan and applies to every product of two full jets in the call, not term
      by term or site by site. The order axis is not an axis of the sample stack, so it is a setting of its own
      and not one more entry in the plan's tuple of scans (see 7.1).
    - ~~With the default, the scan is a real scan on every layout.~~ **Wrong; corrected by the maintainer at
      19:50** (next note). The scan is written through `xscan`, so it is a real scan with jax on the uniform
      layout and a Python loop on numpy or on the ragged layout, as in the library today.
    - The two settings agree to rounding, not bit for bit, so the test that they agree is a tolerance test.
    - A product with an affine operand, and order 0, are the same under both settings: neither has a loop over
      the orders.
    - The block readouts take only the pairs of orders that sum to the top, so there the two settings do the
      same contractions and differ only in the form of the program.
  - Not decided: the name of the setting and of its two values (`over_orders='scan'` and `'loop'` in the
    prototype are placeholders).
  - Checked (`scripts/review/order_scan_check.py`, part 3): the two settings as one function agree to rounding
    under jit at orders 1 to 16, in float64 and in float32, and keep the dtype.
- **2.9, how the scan is written** (2026-11-15, 19:50). The maintainer, correcting Claude's reading above:
  "The implementation is through the xscan abstraction, which dispatches to a jax scan if jax is used and the
  layer is uniform, and a loop if pure numpy or ragged. In general, things should respect the numpy vs jax dispatch
  rule, and scan type operations should be polymorphic in ragged vs uniform. This may have reasonable exceptions
  for performance optimizations and issues related to masking; though for uniform sampling the pattern is to mask
  first, then compute edge variables with the assumption that masking was already done. For map-type operations,
  uniform and ragged may have different behavior for performance and memory considerations; this is one of the
  things we are helping the user control with the Plan abstraction in this refactor arc. Dispatch at the lowest
  level within reason, use jax if any input is jax and numpy otherwise, write code generically with xscan and
  similar, which are obtained from the get_backend function and may also use the tree_contains_jax function.
  Uniform vs nonuniform is deduced from the input types. Look at other functions in the library for examples, for
  example things in the existing backend/probing.py."
  - For 2.9: the `'scan'` setting is the carried shift written with `xscan`. With jax on the uniform layout it
    is `jax.lax.scan`; on numpy or on the ragged layout it is a Python loop over the same body. The `'loop'`
    setting is the loop with slices on every backend.
  - The general rules in this statement are recorded under section 5, where they bear on 5.2, 5.3 and 5.6.
  - The prototype is rewritten in that pattern (`scripts/review/order_scan_check.py`): `get_backend`,
    `tree_contains_jax`, `xscan`, a closure-free body with the core on the carry, and the library's `contract`.
    The layout is deduced by the caller and handed down as `is_uniform`, as `_eta_jets_core` does, because one
    core's arrays look the same in both layouts. Checked on all four backends, in float64 and float32: both
    settings give the explicit sum, numpy in gives numpy out, jax in gives jax out, and the dtype is kept.
  - **What follows, Claude's observation.** Where `xscan` is a Python loop, the `'scan'` setting has neither of
    the scan's two properties: under jit on the ragged layout its traced program grows with the order (143 lines
    at order 4, 479 at order 16, against 34 at every order on the uniform layout), and nothing bounds the peak
    memory. It still does the square of the cheap contractions, where `'loop'` does the triangle (138 and 474
    lines). So on the ragged layout and on numpy, `'loop'` does less work and gives up nothing. The user can
    choose it through the plan. Whether the default should depend on the layout is a question for 7.2 (the
    default plan); as decided, the default is `'scan'`.
  - The line counts quoted in the earlier notes (143 and 491 for the loop) were of a first draft written in
    jax only. Written generically the loop has 138 and 474.

## 3. Variables and kinds

| # | Question | Status | Depends on |
| --- | --- | --- | --- |
| 3.1 | The primitives of a variable's name are a letter for the side contracted (`mu`, `nu`, `xi`, `eta`), the basis of the cut edge where it is not the usual one, and a grade (plain, dot, tilde, tilde-dot) | Decided | 9.6 |
| 3.2 | Is `gamma` (now `nu_L`) a first-class variable, or does a different frame convention remove the need for it? | To confirm | 4.1 |
| 3.3 | The tilde names always mean the adjoint sweep of the residual. The adjoint sweep of `Jp` has no names in a call that also holds the residual's; its variables are reached by a call with `Jp` given as the residual | Decided | 6.5 |
| 3.4 | What is a sampling kind in the target, and can a user define one? | Reopened | 1.1 |
| 3.5 | The output is `z` for every kind | Decided | |
| 3.6 | The engine takes the data `y` and an optional `weight` (ones by default). The weight acts on `f`, `g`, `JtJp` and `Kp` alike, through one public helper that turns a jet into a dual jet. The bare transpose takes a seed and no weight. The engine has no other notion of a loss | Decided | 1.1 |
| 3.7 | What happens to the sample stack `W` on each output. Kept: `z`, `r`, `Jp` and every edge variable. Summed by default, kept on request: the outputs shaped like cores, `g`, `JtJp`, `Kp`. Always summed: `f`, `r_block_sumsq`, `Jp_sumsq` | Decided | 3.6 |

Notes:

- **3.1, decided with the notation** (2026-11-15, 23:23; see 9.6). The question was whether letters and grades
  are the right primitives. The notation the maintainer locked in has three: the letter, the basis and the
  grade. That this closes 3.1 is Claude's reading of his decision, to be corrected if wrong.
- **3.2, what the notation changes.** The variable is now `nu_L`: the right of a bond in the `L` basis. The
  letter `gamma` is retired. It can be wanted and held like every variable (confirmed at 21:57). What stays to
  confirm is the second half of the question, which waits on 4.1: whether a different frame convention removes
  the need for a second basis.
- **3.2.** Decided on 2026-11-14 as public. It exists because the frame's right variable is swept through `R`
  while the value and the curvature term need it through `L`.
- **3.3.** Decided on 2026-11-14: only the copies seeded by the residual, because those depend on the point and
  the data only.
- **3.4.** Decision 8 left the `SamplingKind` protocol unchanged so that user-written kinds keep working. That is a
  compatibility reason. This question is also the library's standing open question on its extension surface,
  which the target can answer instead of deferring.
- **3.6, decided** (2026-11-16, 00:56 to 01:04).
  - **Why anything is inside.** The gradient is the transpose applied to a seed, and the seed is made from the
    output: `r = z - y`, then the weight. For `f` and `g` in one pass over the samples that step has to happen
    inside the pass, between the forward and the adjoint sweep of one chunk. Otherwise the plain variables are
    swept twice, or held for every sample.
  - **The maintainer's structure** (00:56): "we create a publicly accessible helper function which converts a
    dual jet to a jet using weights and jet reversal (and another function which does the inverse), then we pass
    the weights to the engine as needed, and inside the engine it calls the helper function. The helper function
    can also be used by the user for their own purposes."
  - **The weight on `f`** (01:04). Claude had said that `f` does not need the weight inside, since the block sums
    of squares could be weighted outside. The maintainer: "There needs to be consistency between f and its
    derivatives. Therefore, if g, JtJp, and Kp are weighted, then f must be weighted. Therefore I think we should
    add the weight to f. If the user supplies the same weight to all these functions, then the results they get
    are derivatives of each other without extra work. It should be optional on all of these, defaulting to ones.
    The transpose does not need weight. Really, JtJp is "JtWJP" where W is the Riesz operator (inverse mass
    matrix), and Jt is the unweighted transpose. We keep the name JtJp, because the weight is seen as optional."
  - **So:** with `omega` the weight and `omega^2` its Riesz operator, `f = 1/2 <r, omega^2 r>`,
    `g = Jt omega^2 r`, `JtJp = Jt omega^2 J p`, and `Kp` uses the seed `omega^2 r`, with one optional `weight`
    of the shape of 2.5. The bare transpose takes a seed and no weight. The operator is applied in one place, the
    public helper. (The operator is not written `W` here: `W` is the sample stack. The maintainer pointed out
    the clash at 01:26.) The regularizer, the derivative-mode adapter (2.3) and the data preparation (6.7) stay above
    the engine. Another loss is built above it from `z` and the bare transpose, at the cost of a second pass.
  - **Details noted by Claude, not answered:** the inverse helper divides by the weight squared, so it does not
    exist where a weight is zero (6.7 lets a weight be zero to mask a block); the helper works in Taylor
    coefficients, the `t!` factors being the adapter's; at order 0 it is the multiplication alone. The names of
    the two helpers are open (6.3).
  - **Considered and not taken:** a loss function passed by the caller and applied inside the pass.
- **3.3, decided** (2026-11-16, 01:36). The maintainer: "Proposal confirmed."
  - **The question, as put to him.** In one Newton step the adjoint sweep runs for two inputs. The residual's:
    the seed is made from `r`; it gives `g`, and `Kp` reads its variables; it changes only when the point or the
    data change. The direction's: the seed is made from `Jp`; it gives `JtJp`; it changes every CG iteration.
    When one call asks for `JtJp` and `Kp` together, both sweeps run in that pass, and a name like `mu_tilde`
    can mean only one of them.
  - **Decided.** The tilde names always mean the residual's sweep: those are the ones worth asking for and
    holding. The direction's sweep has no names in that call; its variables live inside `JtJp`. Nothing is
    unreachable: a call with `Jp` given as the residual has them under the ordinary names. That last part rests
    on 6.5 (who vouches for a held variable), which is open.
  - **Not taken:** a second set of names for the direction's sweep.
  - This is the decision of 2026-11-14 confirmed, with the route for the direction's variables added. The row
    used to read "Which seeded copies of the tilde variables are public?"; "public" has since come to mean every
    variable (6.4), so the row is stated as what a name refers to.
- **3.7, decided** (2026-11-16, 01:32). Put first as "which reductions over the samples are sampling
  variables", the question confused the maintainer (01:26). Put as a table of what happens to `W` on each
  output, he confirmed it: "I understand now. Confirmed. The table with columns "What happens to W" and
  "Outputs" in your message is what I want."

  | What happens to `W` | Outputs |
  | --- | --- |
  | Kept | `z`, `r`, `Jp`, and every edge variable |
  | Summed by default, kept on request | The outputs shaped like cores: `g`, `JtJp`, `Kp` |
  | Always summed | `f`, `r_block_sumsq`, `Jp_sumsq` |

  - The third row holds numbers, not arrays over the samples. They are formed inside the pass, because under a
    chunked plan `r` and `Jp` are dropped chunk by chunk.
  - `f = 1/2 <r, omega^2 r>` is the objective. `r_block_sumsq` is the sum of squares of `r` for each mode and
    order, unweighted, in the shape of the weight; the display's table of relative errors reads it.
    `Jp_sumsq = <Jp, omega^2 Jp>` is the Gauss-Newton quadratic: the same number as `p` paired with `JtJp`, with
    no adjoint sweep.
  - **Said with the question and not spoken of in his answer** (Claude's, to be corrected if wrong): `f` keeps
    `C`, `Jp_sumsq` keeps `K + C`, and `r_block_sumsq` keeps `C` (the library's `block_sumsq` sums `C` away
    today); and an unsummed `W` on a transpose becomes the tangent stack of the result in the order
    `W + K + C`, as in the library today.
  - The block sums of squares of the data are not sampling variables: they depend on the data alone (6.7).
- **The sample stack on the outputs** (the maintainer, 2026-11-16, 01:26; it bears on 3.7, 5.4, 6.2 and 8.1).
  - **The residual.** "The residual is f(x)-y or z-y, where y is the data, and z=f(x) is the model prediction of
    the sampling map. The data y will typically be stacked over the W axis group ..., while the model prediction
    z can be further batched over the base point batch axis group C, because the T3 itself can have a C batch.
    Then, in the formula z-y, the data y is broadcast over C."
  - **The transposes sum the samples by default.** "When it comes to the transposes which output cores, the
    W-axes from the sample data gets incorporated into their C or K stacks (forget the details of which), unless
    the option is given to sum the samples, in which case the W stack is summed. This is important since stacking
    the W results at the front (as simple jax autodiff approaches do) would put it before the site index which
    must come first, and necessitate a transpose and hence memory reordering. Right now, the default is to *not*
    sum the W stack on transposes. For this refactor I would like to change the default so that it does sum the
    stack, but keep the option to not sum it." (In the library today the flag is `sum_over_probes=False`, and an
    unsummed `W` becomes the tangent stack `K` of the result, in the order `W + K + C`:
    `docs/batching_and_stacking.md` section 11.)
  - **Which layer supports the stacks.** "The optimization and fitting layers can assume that there is no C or K
    stacks, but the sampling layer must support C and K stacking."
- **3.6, the earlier record.** Decided on 2026-11-14 as inside, because computing `f` and `g` in one pass needs the residual formed
  inside the chunk scan. A reviewer argued for outside.

## 4. The frame and the value

| # | Question | Status | Depends on |
| --- | --- | --- | --- |
| 4.1 | A frame is the four stores `(D, U, L, R)`, and `(D, L)` is the base point. The Tucker center cores may be supplied beside it. With them the value is read off the frame, `z[i]` being `eta[i]` through the center; without them it goes through `D` | Decided | 1.1, 1.2 |
| 4.2 | Frame consistency (the four stores describe one tensor; given centers agree with it) is a precondition, checked in safe mode at the moment a function that requires it is called, and never otherwise. No runtime safety check goes through a dense tensor | Decided | 4.1 |
| 4.3 | At an accepted point the value is read off the frame with `Vc` (4.1). At a trial point the optimizers offer three policies: the trial's raw cores; a frame at every trial, with its plain variables held for the accepted one (the default); a frame at every trial with the gradient computed with the value | Decided | 4.1 |
| 4.4 | Frames support arbitrary tails, and nothing checks for them. The boundary vector is ones, sized by the end bond of the cores passed. A function that requires squashed tails is to be looked at with suspicion | Decided | 4.1 |
| 4.5 | Where do the gauge projection, the lift, and the assembly of the exact Hessian product live? | To confirm | 1.1 |

Notes:

- **4.1, the first half of 4.3, and the centers' part of 4.2, decided** (2026-11-16, 01:49 to 01:52).
  - **The two routes to the value at a point that has a frame.** Through the base point's cores:
    `z[i] = z_term(eta_D[i], D[i])`, which costs one more right sweep (`nu_L`) and one more trunk per site
    (`eta_D`) beyond what `g` needs. Read off the frame: `z[i] = z_term(eta[i], center[i])`, with the Tucker
    center core of site `i`, which has the shape of `V[i]`; no extra sweep. The identity is checked
    (`scripts/reuse_check.py`, 1.7e-15).
  - **The maintainer** (01:49): "Can we make both options available? If the user supplies the Tucker centers, we
    use them. If not, we go through D. In the optimization routines that we control, we would supply the Tucker
    centers. Are there any downsides of supporting both? Also, currently I believe the orthogonalization creates
    the TT and the Tucker centers, can you check this? With the planned retract-and-orthogonalize function, we
    should make sure it can return the centers too."
  - **Checked in the library.** `t3_orthogonal_representations` (`backend/fv_conversions.py`) returns the frame
    and, as a second result, the TT centers (from the right-to-left sweep) and the Tucker centers (from the
    up-orthogonalization of those). The manifold geometry's `frame` keeps the first result only
    (`frame, _ = t3_orthogonal_representations(x_cores)`). So the centers cost nothing new.
  - **Downsides named by Claude.** A supplied center can be wrong or stale, and then `f` is silently wrong;
    two routes to one number have to be tested against each other; wrong shapes are a structural error. Only
    the probe kind gains: for apply and entries the value through `D` is `mu` at the last bond, which the left
    sweep already gives. `Kp` reads `nu_L` and its family on either route.
  - **Decided** (01:52: "Agree with proposal for 4.1 and first half of 4.3. Also, agree that wrong centers
    should error in safe mode."):
    - the engine takes optional Tucker centers; with them `z[i]` is `eta[i]` through the center, without them
      it goes through `D`;
    - the frame construction and the retract-and-orthogonalize function return the centers;
    - the optimizers the library controls supply them;
    - centers that do not agree with `(D, L)` are an error in safe mode. (The check is core-sized. Claude
      believes it needs an orthogonal frame; not written.)
  - **The name** (01:54). The maintainer: plain "tucker_centers" is unambiguous but long. Claude suggested
    `Vc` for the Tucker centers and `Hc` for the TT centers: each has the shape of the direction core it sits in
    place of, `V[i]` and `H[i]`, and `c` is for center. The maintainer: "Agree with Vc, Hc." The engine's
    keyword is `Vc=`. Checked on the library's own frame and centers, with jets
    (`scripts/review/sample_variables_check.py`, part E): the value with `Vc`, with `Hc`, and through `D` agree.
  - **Not decided by this:** what a trial point in a line search does (the second half of 4.3); whether the
    four stores' agreement among themselves is also a precondition (the rest of 4.2).
  - This reverses decision 7 of 2026-11-14 in part: reading the value off the frame is no longer parked. Its
    three objections were the missing guard (now the safe-mode check), the missing channel in the geometry
    protocol (now provided), and the library's caching audit (a fact about the present structure).
- **4.2, decided** (2026-11-16, 01:52 for the centers, 02:07 for the four stores).
  - **What agreement means.** A frame describes one tensor several times: `D` over `L`, `D` over `R`, and at each
    site `L` cores to the left, `U`, and `R` cores to the right. A frame the library builds agrees by
    construction; one assembled by hand need not. If the stores disagree, the value comes from one tensor and
    `Jp`, `g`, `JtJp` go through the cores of another, so `g` is not the gradient of `f`, silently.
  - **The library today.** `T3Frame.is_consistent` is a diagnostic that nothing calls automatically. It compares
    the `L` and the `R` reconstruction only, not `U`, and it forms both dense tensors
    (`fv_frame_consistency_residual`); its docstring calls it expensive. Decision 7 of 2026-11-14 left it a
    diagnostic.
  - **The maintainer** (02:07): "This is a precondition checked in safe mode at the moment that a function that
    requires it is called. Runtime safety checks must not go through dense; this will probably blow out the
    users memory. We can check ||A-B|| using the existing non-dense norm machinery. The is_consistent helper
    should be updated to avoid dense as well. Do not call the check when it is not needed; the user may do
    whatever they want with the extra core; if they are not using functions that depend on consistency, then we
    let them be."
  - **So:** a precondition in safe mode, skipped in unsafe mode and under jit like the others; checked only
    where a function needs it; computed through the cores, as the norm of the difference of two tensors in T3
    form, never through a dense tensor. `is_consistent` and `fv_frame_consistency_residual` are to be rewritten
    the same way. With 01:52: centers that are given and do not agree with `(D, L)` are the same kind of error.
  - **Open:** which functions require it. Claude's reading, to be settled with the table of variables (6.4): the
    fitting layer's local model always; `g` when its residual comes from the `z` of the same frame; `z` with
    `Vc` (the centers' agreement); and not an edge variable asked for on its own, nor `z` through `D`.
- **4.3, the trial point, decided** (2026-11-16, 01:59). The maintainer: "Add all three as options, default to
  (b)".
  - The engine does not care: a trial's value is `f` on whatever frame it is given. The policy belongs to the
    optimizers. With F one forward pass over the samples (the plain sweeps and `z`), A one adjoint pass, and `t`
    trials in an outer iteration:

    | Policy | At each trial | On acceptance | Sample work | Needs |
    | --- | --- | --- | --- | --- |
    | a. Raw cores at the trial (today) | F through the trial's own cores | Orthogonalize, then F and A at the new frame | `t F + F + A` | Nothing |
    | b. A frame at every trial (the default) | Retract-and-orthogonalize, then F through its frame with `Vc` | A only: the trial's plain variables are the frame's | `t F + A` | The plain variables of one trial held until it is accepted; `t` orthogonalizations |
    | c. A frame at every trial, the gradient with it | Retract-and-orthogonalize, then F and A together | Nothing | `t (F + A)` | `t` orthogonalizations; nothing held |

  - b saves one forward pass per outer iteration over a, for core-sized work per rejected trial and the memory
    of one trial's plain variables. Under a plan that holds nothing, b saves nothing; c is the one that works
    with nothing held, equal to b when the first trial is accepted and worse than a from the third trial on.
    That A costs about as much as F is Claude's estimate, not a measurement.
  - "Carrying `r` from an accepted trial" (decision 7, point 3) is overtaken: with `Vc`, forming `r` again at
    the frame costs one `z_term` per site.
  - Open: the name of the option and of its three values.
- **4.1 to 4.3.** Decision 7 chose the value through the base point's own cores and parked reading it off the
  frame. Two of the reasons were that the geometry protocol has no channel for the center cores and that the
  library's caching audit had declined it. Those are facts about the present structure. The counts of sample
  sweeps per outer iteration (`t + 2` today, `t + 1`, or `t` with a frame per trial) do not depend on them.
- **4.4, decided** (2026-11-16, 02:12). The maintainer: "Frames must support arbitrary tails. Allow and do not
  check. Functions which require squashed tails should be looked at with suspicion."
  - **Background.** A TT whose end bonds are larger than 1 (tails) stands for the tensor with those two end
    indices summed, that is, contracted with a vector of ones. Squashing sums the end index into the end core.
  - **Put with the question, and not objected to:** the boundary vector is ones, sized by the end bond of
    whatever cores are passed, so tails work everywhere in the engine with nothing special-cased (a requirement
    with a test); and each sweep's start value follows from its grade: ones for plain (with jets, ones at order
    0 and zero above), zeros for dot, tilde-dot and the probe kind's tilde, and the seed times the ones vector
    at the right end for the tilde of apply and entries.
  - **Noted by Claude with the choice:** on a frame with tails, variations of an end core that sum to zero over
    the tail index change nothing, so the tangent coordinates are not unique there. That is the caller's
    business.
  - **Not spoken of:** that the frames the library builds are squashed by default (`squash_tails=True` today).
  - **For the branch, from his last sentence:** an audit of the library's functions that squash or assume
    squashed tails. First candidates from a search for "squash": the default of
    `t3_orthogonal_representations` and its uniform twins; `geometry.point_norm_sq`, which squashes before it
    reads the norm off the last core; the uniform `+`, stack sum and SVD paths, which squash; and a comment in
    `backend/fv_operations.py` that reasons "on the usual squashed frame". Not examined.
- **4.4, the earlier record.** Adopted on 2026-11-14 "as the library does it".
- **4.3, after the geometry boundary.** A new function that retracts and builds the frame together may be added.
  The existing retraction and frame construction keep their signatures and behaviour.
- **4.5, what is and is not proposed.** Nothing is proposed for the projection or the lift. The suggestion to
  confirm is that they stay where they are: the geometry owns them and the fitting layer calls them, so the
  sampling layer returns the blocks before projection, takes the already-lifted direction for the curvature term,
  and never needs to know the geometry. That is also why Tucker-factor sharing can stay in a geometry wrapper.

## 5. Functions and layouts

| # | Question | Status | Depends on |
| --- | --- | --- | --- |
| 5.1 | One function per variable with explicit arguments, seven terms, no graded containers | Decided | |
| 5.2 | Is there one set of step functions for the ragged and the uniform layout, and how does the uniform layout's static data (shapes, masks) reach them? | Open | 1.2 |
| 5.3 | For each variable, are sites scanned or vectorized, what is the default, and which variables admit which? | Reopened | 5.2 |
| 5.4 | How are the terms generic over the stack groups: several `W` axes, `K`, `C`, and the order? | Open | 2.7 |
| 5.5 | Is there any choice of association order inside a term? | To confirm | |
| 5.6 | What is the dtype rule, so that a float32 problem stays float32? | Open | |

Notes:

- **5.3.** Decision 4 made the site choice per variable, which stands on an accounting argument. Its default,
  "what each kernel does today", is a reason from history.
- **5.5.** Decision 4 left association out, on the library's own measurement that one contraction per term is
  right on jax.
- **5.1, step functions and the basis** (the maintainer, 2026-11-16, 00:03 and 00:23). Asked whether a step
  function is named by letter and grade alone, with the basis following from the cores it is given, he first
  said: "I would prefer to have separate step functions. It is ok to have lots of small functions." Shown that
  the two would be one recurrence with different arguments, he said: "Oh, I see. There is no need for extra step
  functions if they literally do the exact same things with different arguments." So a step function is named by
  letter and grade only and serves both bases: `nu_step` given `L` cores returns `nu_L`, and `mu_dot_step` given
  `(R, U, H)` returns `mu_R_dot`. The prototype's 22 step functions stand.
- **Rules for how the functions are written** (the maintainer, 2026-11-15, 19:50, said while settling 2.9):
  "The implementation is through the xscan abstraction, which dispatches to a jax scan if jax is used and the
  layer is uniform, and a loop if pure numpy or ragged. In general, things should respect the numpy vs jax dispatch
  rule, and scan type operations should be polymorphic in ragged vs uniform. This may have reasonable exceptions
  for performance optimizations and issues related to masking; though for uniform sampling the pattern is to mask
  first, then compute edge variables with the assumption that masking was already done. For map-type operations,
  uniform and ragged may have different behavior for performance and memory considerations; this is one of the
  things we are helping the user control with the Plan abstraction in this refactor arc. Dispatch at the lowest
  level within reason, use jax if any input is jax and numpy otherwise, write code generically with xscan and
  similar, which are obtained from the get_backend function and may also use the tree_contains_jax function.
  Uniform vs nonuniform is deduced from the input types. Look at other functions in the library for examples, for
  example things in the existing backend/probing.py."
  Read as rules, by Claude:
  - **numpy or jax** is inferred from the inputs: jax if any input is a jax array, numpy otherwise
    (`tree_contains_jax`). The dispatch is made at the lowest level, within reason. No `use_jax` argument.
  - **Ragged or uniform** is deduced from the input types.
  - **Scan-type operations** are written once, generically, with `xscan` and its relatives from `get_backend`,
    and are polymorphic in the layout. Exceptions are allowed for performance and for masking.
  - **Masking on the uniform layout:** mask first; the edge variables are then computed on the assumption that
    the masking has been done.
  - **Map-type operations** may behave differently on the two layouts, for performance and memory. That
    difference is one of the things the plan lets the user control.
  - The existing `backend/probing.py` is the example to follow.
  These bear on 5.2 (one set of step functions), 5.3 (sites scanned or vectorized) and 5.6 (the dtype rule).
  The rows are not marked decided by this; they are to be taken up in group D with these rules as given.
- **The same rules as the library has them** (read on 2026-11-15, 20:00, at the maintainer's request: "the
  prototype is just that; the way the code is written there may be what we are going for, or it may be something
  that was done for expediency"). Read in full: `backend/probing.py`, `backend/sampling_derivatives.py`,
  `backend/common.py` (the scans and maps), `backend/ut3_sampling.py` (its head), and the notes they cite. The
  rules above are already written down in the library: `docs/contributor/uniform_polymorphism.md` (which loops
  are shared between the layouts, and masking once) and `docs/contributor/scan_body_principles.md` (how a scan or
  map body is written). Those two notes, not this folder's prototypes, are the reference for the form of the code.

  | Loop | Ragged layout | Uniform layout | Example in the library |
  | --- | --- | --- | --- |
  | A sweep along the chain | `xscan`: a Python loop | `xscan`: `jax.lax.scan` with jax | `compute_mu`, `compute_sigma`, `_adj_sweep_scanned` |
  | A per-site map in the plain kernels, and every lift | `xmap`: a Python loop | no map: one contraction with a leading `d` | `compute_eta`, `assemble_z`, `assemble_z_jets` |
  | A per-site map whose step has a product of two full jets | `xmap`: a Python loop | `xmap`: `jax.lax.map`, one site at a time | `compute_eta_jets`, `compute_deta_jets`, `compute_dxi_tilde_jets`, the TT assembly |
  | The scan over the orders inside one site | a Python loop | `jax.lax.scan` with jax | `_eta_jets_core(data, is_uniform)` |
  | Chunks of samples | none: the unchunked assembly | dynamic slices inside `jax.lax.scan`, with jax, in two assemblies only | `_wchunked_sum` |

  - **Dispatch.** Every function begins `use_jax = tree_contains_jax((its inputs))`, `is_uniform = is_ndarray(a
    core argument)`, `xnp, xmap, xscan = get_backend(is_uniform, use_jax)`. A body re-derives `xnp` from all of
    its own arguments.
  - **Masks.** The kernels have no mask logic. The wrappers in `backend/ut3_sampling.py` mask the supercores
    once, pack the vectors, call the shared kernel, and re-mask a core-shaped output. This is the library's
    present answer to the second half of 5.2.
  - **Bodies.** Module-level functions that close over nothing; statics are read off shapes; operands that do
    not change ride the carry. A static that no operand carries reaches a body through two step variants
    (`_eta_jets_step_uniform`, `_eta_jets_step_ragged`) or a memoized factory keyed on hashable scalars
    (`_tt_variation_jets_step(n_probe, sum_over_probes, is_uniform)`). A setting of the plan would reach a body
    the same way, which is why the plan has to be hashable.
  - **Dtype.** A start value is built in `xnp.result_type` of the sweep's operands in the jet kernels and the
    two-hole; the plain sweeps (`compute_mu`, `compute_sigma`, `compute_tau_tilde`) use the default dtype.

  What the reading bears on. Claude's observations, none a decision:
  - **5.3.** The site option is a choice only where `xmap` is a real map: on the uniform layout with jax. The
    library today has both behaviours there: vectorized over the sites in the plain kernels and the lifts,
    one site at a time where a step has a product of two full jets.
  - **2.9 with 5.3.** On the uniform layout the setting for the orders and the setting for the sites combine.
    The lever the library's own record names for the small-problem latency of the scanned kernels
    (`docs/contributor/deferred_and_rejected.md`) is one of the combinations: one scan over the orders whose
    body is vectorized over the sites. `scripts/eta_axes.py` timed four combinations for the eta combine, but
    its "vectorized over orders" is the dense form, not the loop with slices.
  - **A question for group E.** By the rule as stated, a scan is real only with jax on the uniform layout. Is a
    scan over chunks of samples (7.1) real on ragged cores with jax? The sample axis has the same length at
    every mode on both layouts. Decision 1 of 2026-11-14 kept a host loop "only for numpy and for streamed
    data", and `levels_jax_check.py` ran chunk scans on ragged cores under jit. The library's present chunking
    engages on the uniform layout with jax only.
  - **What the explicit prototypes are.** `jet_steps_check.py` and its relatives are ragged, numpy, with Python
    loops and one `W` axis. They establish the algebra and the names. They are not the form of the code.
  - **One difference that matters for the target.** The prototype forms every tilde variable at every order and
    takes the top coefficient of full blocks at the end. The library forms only what is read: `dxi_tilde` (the
    target's `eta_tilde`) in its two read rows only, because it is paired with the affine input `(w, p)` and
    nothing else, and each gradient block directly, summed over the orders. The row of 2.9 covers the block
    readouts. The two rows of `eta_tilde` are not recorded anywhere, and they bear on 3.3 and 6.4: if
    `eta_tilde` is public, the question is whether it is those two rows or all of them.

## 6. The interface

| # | Question | Status | Depends on |
| --- | --- | --- | --- |
| 6.1 | `sampling_variables` is a public backend function. It returns a tuple aligned with `want`, and held variables are keyword arguments | Decided | |
| 6.2 | What are its remaining arguments: the kind and the order, the directions, the data and the weight? | Open | 2.8, 3.4, 4.5 |
| 6.3 | What other public functions does the sampling layer have, and do any of the existing public kernel names remain? | Reopened | 1.1 |
| 6.4 | Which variables are public for each kind, and is the docstring table of reads generated from the declared graph? | To confirm | 3.1 to 3.4 |
| 6.5 | Who guarantees that a held variable is valid for this point, these samples, this data and this order? | Open | |
| 6.6 | The plain and the jetted sampling function are one function: `probe(ww, pp=None, order=0)`, returning `T + W + C + (Ni,)`. This holds for the frontend methods of the flagship classes and for the backend function behind each, which uses the engine | Decided | 2.6, 2.7 |
| 6.7 | Preparing the data and creating weights is done by standalone functions that the user calls before the fitting functions. The fitting functions have opt-in keywords that call those functions. A keyword inside a fit does only preparation that changes the loss alone; anything that changes the unknowns or belongs to the problem stays standalone | Decided | 2.5 |

Notes:

- **What "public" means for a variable, under discussion** (2026-11-15, 21:35). Not decided. It bears on 3.3,
  6.3, 6.4, 6.5 and 3.4.
  - **The maintainer's take.** "In the current codebase, there are handcoded functions that implement the
    computation of every edge variable, in addition to functions that do higher level operations like probing T3s
    and tangent vectors, or applying the transpose of the probing map. In the new codebase, the computation of
    edge variables is done through the sample variable engine, where you ask for what variables you want and
    supply what variables you have. In this sense, there is no need for individualized functions for computing
    each edge variable, since the engine does it better and more generally. This is my take; how do you see it?"
  - **Claude's answer: agreed, and it gives the word one meaning.** A variable is public when the engine accepts
    its name, in `want` and as a held keyword. It is not public by having a function of its own. The public
    functions of the sampling backend are then the engine and the operations of 6.6 that call it. The step and
    term functions are the engine's private implementation.
  - **Reasons, from the design and not from today's code.** (1) A function per variable makes the caller walk
    the dependency graph, which is the engine's knowledge. (2) Functions composed by the caller cannot follow the
    plan: each returns its variable for every sample, and the plan's memory bound comes from forming a chain
    inside one chunk and dropping it. So the composable route and the fast route would be different routes.
    (3) The count: today 46 of the 111 public names in `probing`, `sampling_derivatives`, `apply` and `entries`
    are per-variable functions (`compute_*`, `assemble_*`, `build_*`), plain and jetted. (4) The library's razor
    for the backend is met by the engine: a user of raw `.data` finds one function and asks it for `mu`.
  - **What a name promises, proposed as the test for a name being public.** A meaning that does not depend on the
    route the engine took (so a held value may come from anywhere); one shape; and a use outside: it is an output,
    or it is read by an output and is worth holding from one call to the next. By this test the residual-seeded
    tilde variables that `Kp` reads are public, and `eta_tilde` is not (only `g` reads it, once), which leaves the
    engine free to form its two read rows only. The tilde variables seeded by `Jp` and the dot variables of `Kp`
    stay internal, as in the review.
  - **Costs and conditions.** The shape contract of each variable has no signature to sit on, because the return
    is a tuple aligned with `want`: the table of variables in the docstring (6.4) becomes the contract and has to
    be generated and tested. Asking for one variable must compile to the program a hand-written chain would give.
    The engine has not been prototyped; only the step functions have.
  - **What it moves to 3.4.** With no function per variable, a user cannot assemble a new quantity by hand. If
    users may define kinds, the terms and steps are building blocks for the author of a kind, which is a second
    meaning of public, for a second audience.
  - **A consequence for 6.5, noted.** If `r` can be held, the transpose applied to any array of the type of `z`
    is `want=('g',)` with that array held as `r`. A held variable is then an input with a meaning, and its
    validity is the caller's.
- **What "public" means, the maintainer's two objections** (2026-11-15, 21:44). Not decided.
  - **His words.** "First, the edge variables like eta_tilde are themselves valid mathematical quantities with a
    definition (jetted adjoint with respect to a top level order) that user may want. This particular example
    seems unlikely, but I can easily see situations where the user would want, say, mu. And I don't have a clear
    sense of where to draw the line, which suggests that we provide all of them. Second, the shape contract has
    some nuance when it comes to jets, particularly since we are planning to unify the jetted and unjetted
    variables, so that the unjetted variable is just the jet of order zero. Perhaps this is fine given our
    definition of the optional T index group."
  - **Claude's answer to the first: the third part of the test is withdrawn.** "A use outside" drew a line that
    has no principle behind it. Every variable of a kind's graph, at every grade, can be wanted and held. What a
    name promises is then two things: a meaning that does not depend on the route, and one shape. Whether a
    variable is worth holding is guidance in the table, not a condition for having a name.
    - **A rule this needs:** wanting a variable never changes any value; not wanting it lets the engine skip
      work. `eta_tilde` is the case in point: wanted, it is formed at every order; not wanted, only the two
      slots that `g` reads are formed. Those are two forms of one step, and the short one has to equal a slice
      of the full one, by test.
    - **The seeds (3.3).** A tilde variable is linear in its seed. If `r` can be held, one name per tilde
      variable is enough: the copy seeded by `Jp` is the same name in a call with `Jp` held as `r`. Then "which
      seeded copies are public" has the answer "all, under one name". This rests on 6.5.
    - **Still without a name:** the `eta`-like variable through `L` that `z` reads, and the dot variables of `Kp`,
      which belong to the lifted direction. To be named or given a route at 6.4. Not checked. (Named at 23:23:
      `eta_D`; `mu_dot`, `xi_dot`, `nu_L_dot`, `eta_D_dot`, and `eta_D_tilde_dot`.)
  - **Claude's answer to the second: the optional group `T` carries it, with three things the table has to
    say.** Every variable has the shape `[d] + T + W + [K] + C + (leg,)`, with `T = (order+1,) if order else ()`,
    so at order 0 it is today's plain variable, and the rank is declared by `order`, never read off an array.
    - **Affine variables do not have `T`.** The input `w + s p`, `xi` and `xi_dot` (`xi_dot`: now `xi_U_dot`; the
      raw `xi_dot` too) have two coefficients at every order above 0 (catalog, point 6; 2.6; 2.9). Their order
      group is a second optional group, two slots or absent. The library stores `xi_jets` this way. The prototype
      pads them to `T` with zeros, which is expediency. Claude's reading; not said by the maintainer.
    - **The tilde grades have the same `T` with the order leg turned** (2.4), so the table needs a column for
      the orientation; the shape does not show it.
    - **A plain jet cut at a lower order is the jet of that order; a tilde jet is not** (catalog, point 4).
      This bears on what a held variable is valid for (6.5), not on the shape.
- **Public, two points confirmed** (2026-11-15, 21:57). The maintainer: "Confirmed on both counts."
  - Every variable of a kind's graph, at every grade, can be wanted and held. Wanting a variable never changes
    any value; not wanting it lets the engine skip work.
  - The affine variables (the input `w + s p`, `xi`, `xi_dot`; now `w`, `xi`, `xi_dot`, `xi_U_dot`:
    `SAMPLE_VARIABLES.md` section 7) carry a two-slot group of their own in place of `T`: two slots at every order
    above 0, absent at order 0.
  - This settles the first half of 6.4 (which variables are public: all of them) and the test of 21:35 is
    reduced to its first two parts. The rest of 6.4 (the table generated from the graph), 3.3 (one name per
    tilde variable, the seed being whatever `r` is) and 6.5 are not confirmed by this.
- **The variables that still had no name, and the naming that followed** (2026-11-15, 21:57 to 22:41).
  Deprecated: superseded on 2026-11-15 at 23:23 by target decision 9.6. The note that stood here is in
  `NOTATION_HISTORY.md`, split by topic:
  - 21:57: Claude listed six variables without a name (the maintainer's words of 21:57 and 21:59 with it); all
    were named at 23:23. `NOTATION_HISTORY.md` section 3.2.
  - 22:11: `zeta` for the down variable through the base point's cores; it became `eta_D` at 23:23.
    `NOTATION_HISTORY.md` section 4.
  - 22:11: options A and B for the dots of `xi` and `mu` along the raw core direction; neither was taken: the
    dots are `xi_dot`/`mu_dot` (raw form) and `xi_U_dot`/`mu_R_dot` (frame's form). `NOTATION_HISTORY.md`
    section 5.
  - 22:24 to 22:41: the naming was rethought from first principles; the result is 9.6 (`SAMPLE_VARIABLES.md`
    section 2). `NOTATION_HISTORY.md` section 6.
- **6.3.** The review kept the old kernels as public adapters and the new names as a second vocabulary. That was
  compatibility. With one vocabulary, the near miss between the new `xi_tilde` and today's `dxi_tilde` exists
  only during migration.
- **6.6.** Raised by the maintainer on 2026-11-15. It was a goal of his when he first tried to merge the jetted
  and plain functions. His condition: consider it again if it can be done cleanly; if it needs complicated
  hackery or is awkward, do not force it. The issue he met: how to tell whether an axis is part of the batch or is
  the derivative-order axis.
- **6.6, the library's record.** `deferred_and_rejected.md` lists "a kwarg-unified derivative interface"
  (`probe(..., order=)`) as a locked rejection, for four reasons: the output rank would depend on whether a
  keyword was passed; that fights the shape-comment-as-contract style; it is worse for the transposes, whose
  residual input rank becomes context-dependent; and a named `*_derivatives` method documents an advanced feature
  that would otherwise be buried.
- **6.6, a precedent in the library.** The batching convention already has groups whose rank varies and is never
  read off an array alone: the `K` and `C` split of a tangent is "recovered, not stored", from the pairing of the
  variations with the frame.
- **6.6, where the order axis is.** Today the derivative methods return `(order+1,) + W + K + C + ...`, order
  outermost, and on the uniform packed layout `(d,) + (order+1,) + W + C + (N,)`, order just after the site axis.
  The plain methods return the same shapes without that axis. The idea under discussion for 6.6 moves nothing:
  the order axis stays exactly there when the order is above 0 and is absent at order 0, as in the plain methods
  today.
- **6.6, a prototype of the signature** (`scripts/review/merged_frontend_prototype.py`): `probe(ww, pp=None, order=0)`
  returning `T + W + C + (Ni,)`, and the corewise transpose with the same three arguments. `pp` is given exactly
  when `order >= 1`. The order is declared, never read off an array; in the transpose it is also recovered by
  pairing the residual with the sample, and the two are checked against each other. The bodies wrap the library's
  existing methods, so the shapes are real: order 0 returns exactly `x.probe(ww)`, and order 3 times `t!` equals
  `x.probe_derivatives`.
- **6.6, decided** (2026-11-15, 04:04). The maintainer, on the prototype: "Perfect. This is good for a frontend
  method of the flagship classes, and a backend function which uses the engine under the hood." What this fixes:
  the order is declared by `order` and the dispatch is on it, not on the type of an input; `pp` is given exactly
  when `order >= 1`; the return carries the optional axis `T`, outermost; the values are Taylor coefficients; the
  transposes take the same `pp` and `order`, and their residual carries `T`. The `*_derivatives` method names go.
  The entry "a kwarg-unified derivative interface" in the library's `deferred_and_rejected.md` is reversed at the
  target, and that file has to say so when the refactor is done.
- **6.6, Claude's reading of the reach, to be corrected if wrong.** The prototype showed `probe` and its corewise
  transpose on `TuckerTensorTrain`. The same pattern is taken to cover `apply` and `entries`, the transposes, the
  `T3Tangent` methods, and the uniform twins. Which backend functions exist beside `sampling_variables` is
  question 6.3, still open; 6.6 fixes only that each one has the merged signature.
- **6.2, the sample and the direction, decided** (2026-11-15, 04:22). 6.2 as a whole stays open. The maintainer
  asked whether the sample can be of type `Union[ww_type, Tuple[ww_type, pp_type]]` with a dispatch on it. The
  answer he agreed to: that is the type, but the dispatch is on `order`, never on the type. Two reasons. A pair
  says the call is jetted but not how many coefficients, so `order` is an argument either way. And the type alone
  is ambiguous: a ragged `ww` with `d = 2` is a 2-tuple of arrays, and so is a jetted sample `(ww, pp)` on the
  uniform layout. With `order` deciding, the function unpacks `ww, pp = sample` when `order >= 1` and takes
  `ww = sample` otherwise, and inspects nothing.
  - **Fitting functions take the bundled sample**: `sample,  # ww if order == 0 else (ww, pp)`, with `index` in
    place of `ww` for entries. The sample is already opaque there and passes through many functions; a separate
    `pp` would have to follow it everywhere, with a second one beside `val_sample`.
  - **Frontend sampling methods take `pp` as its own argument**, as in 6.6.
  - `sampling_variables` is kind-generic and takes the bundled sample; the backend function behind each
    frontend method takes `ww` and `pp` separately, like its method. (Claude's reading; approved by the
    maintainer at 04:26.)
- **6.6, the wording of the comment.** The maintainer: the comment on `order` should relate it to `T`, and
  `T=(order+1)` is wrong at order 0. The form now in the prototype is a conditional written as Python, on the
  `order` line, with the return line using `T` like any stack letter:
  `order: int = 0,  # highest Taylor order.  T=(order+1,) if order else ()`. Approved at 04:26.
- **6.7, decided** (2026-11-15, 16:33 to 17:00). The maintainer's lean of 16:33, confirmed at 17:00 together with
  Claude's rule for what a keyword may do. The three classes of preparation are under 2.5. With the default of
  2.5 doing nothing, every such keyword is opt-in. Which standalone functions exist is part of 6.3.
- **6.7, the equalizing weight, under discussion** (17:00). The maintainer: the rule for a zero or nearly zero
  block "is an open research question. For a default, let's think of something simple. What do you think about a
  filtered inverse like 1/max(x, floor), where floor is a user-specified parameter that defaults to something
  like 1e-6? What I am concerned about is that for input data in a bad length scale, the norms are expected to
  vary hugely across derivative order, often way more than 1e-6, and this is expected and not a sign of numerical
  zeros. So, perhaps the guidance is that one should adjust the length scales first, before equalizing". Claude's
  answer, awaiting his reply: yes to the filtered inverse, with the floor relative to the largest block,
  `1 / max(x, floor * max(x))`, because an absolute floor depends on the units of the data and equalizing is
  meant not to. No special case for a block that is exactly zero; the user sets a weight to zero to mask a block.
  His guidance holds in a precise form: rescaling the direction `p` changes nothing under equalization except which blocks
  reach the floor, so doing it first is what lets the floor separate true zeros from small legitimate blocks; the
  rescaling that repairs a badly conditioned fit is the other one, of a mode's input coordinates. Suggested: one
  function for the block norms and one for the filtered inverse, since the norms are also the diagnostic of the
  length scale.
- **6.7, the equalizing weight, decided** (2026-11-15, 17:04). The maintainer: "Yes, agree with broadcasting
  decision (exact shape only), relative floor, and two-function split."
  - Two standalone functions. One returns the norm of each block of the data, in the shape of the weight. The
    other is the filtered inverse of those norms, `1 / max(x, floor * max(x))`, with `floor` set by the user and
    1e-6 by default. The floor is relative to the largest block.
  - A block that is exactly zero has no special case: it gets the floor weight. To mask a block the user sets
    its weight to zero in the array returned. (Part of Claude's proposal; the maintainer agreed to the relative
    floor without naming this point.)
  - The norms are also the diagnostic of the length scale. The guidance is to adjust the length scales first and
    equalize after.
  - The weight's shape is exact: a mode dimension of 1 is not broadcast (confirmed).
- **6.7, a required example** (17:04). The maintainer: "let's make sure to provide an example with badly scaled
  data, which shows the bad fit as given, and then a good fit after we process the data with these functions."
  A deliverable of the refactor. Not prototyped yet.
- **6.7, a correction by Claude to its own class (b).** For the probe kind, rescaling a mode's input coordinates
  is not loss-neutral: the data of that mode live in the same coordinates, so they are rescaled too, by the
  inverse, and the components of that mode's residual are reweighted. T3Polynomial's record says the same of
  `kappa` on its input modes. The class still "changes the unknowns and comes with an inverse"; for probe it also
  changes the loss. The example has to state this.
- **6.7, the example prototyped** (2026-11-15, 17:10; `scripts/review/bad_scaling_example.py`, against T3Toolbox
  2026.5.0). A tensor with 4 modes fitted from probe jets of order 3 by the library's Newton solver, from the zero
  tensor, with default settings and exact data. The data are badly scaled in two ways: the coordinates of the
  sample vectors span four decades, and the directions `p` are 100 times too short. The error is that of the fitted
  tensor applied to fresh samples in the original coordinates.

  | Preparation | 8 samples | 200 samples |
  | --- | --- | --- |
  | As given | bad (87) | bad (1.8) |
  | Equalizing weight only | bad (17) | bad (1.9) |
  | Coordinates only | bad (3.3) | good (9e-11) |
  | Coordinates, then the direction | good (1e-09) | not run |
  | Coordinates, then equalizing weight | good (2e-10) | good (4e-12) |
  | Coordinates and direction, then equalizing weight | good (3e-10) | not run |

  The numbers in brackets are from one run and are not reproducible to the digit; the script's checked output
  says only good or bad. With 8 samples the same pattern held for four seeds.
- **6.7, what the prototype shows.** Claude's reading.
  - The required example exists in prototype: bad as given, good after preparation.
  - **The equalizing weight alone does not rescue this fit.** What decides it is the rescaling of the
    coordinates, which changes the unknowns. The equalizing weight is needed as well when the higher orders are
    needed to determine the tensor (8 samples), and not when the order-0 data determine it alone (200 samples).
  - Rescaling the direction and equalizing are interchangeable here, as predicted: either works with the
    coordinates, and doing both gives the result of equalizing alone.
  - The bad fits also took longer: 14 to 20 seconds at 200 samples, against under 1 second for the good ones.
  - **The example needs more functions than the two decided.** Besides the block norms and the filtered inverse
    it uses a rescaling of the direction `p`, a rescaling of a mode's coordinates with its inverse on the fitted tensor,
    and two estimates read off the data (a length for the directions from the block norms, a coordinate scale from the
    samples). Which of these the library provides is part of 6.3, not decided.
  - Limits: exact data, one problem, the ragged layout, numpy, and today's derivative data. For probe the
    coordinate rescaling also changes the loss, so its effect here is not conditioning alone.
- **6.7, a word withdrawn** (2026-11-15, 17:47). Claude had written "the step" for the direction `p` of the input
  jet `w + s p`, and "rescaling the step" for multiplying `p` by a length and slot `t` of the data by that length
  to the power `t`. The maintainer asked: "What is the step rescaling? Is this the newton step length or something
  else?" It is not the Newton step. The word also collides with the step functions of 5.1. It is replaced by
  "direction" throughout, and the prototype's function is `rescale_direction`.
- **6.7, two more functions decided** (2026-11-15, 18:32). The maintainer: "Coordinate rescaling and direction
  rescaling become library functions." So the standalone preparation functions decided so far are five: the block
  norms; the filtered inverse of the norms; the rescaling of the direction `p` (loss-only); the rescaling of a
  mode's coordinates, which changes the unknowns and, for probe, the loss; and its inverse on the fitted tensor.
  Not decided: the two estimates the prototype reads off the data (a length for the directions from the block
  norms, a coordinate scale from the samples), the names, and where the functions live (6.3).

## 7. Memory and execution

| # | Question | Status | Depends on |
| --- | --- | --- | --- |
| 7.1 | A plan is an ordered tuple of scans over axes of the sample stack, each with a `hold`, plus a top-level `hold`. Since 2.9 it also carries how a product of two full jets is taken over the orders | Decided | |
| 7.2 | What are the defaults: the default plan, and what a model holds between calls? | Reopened | 7.3 |
| 7.3 | Where does the plan live: in the model, or as a solver parameter passed to each operation? | Reopened | 1.1 |
| 7.4 | Chunk sizes are array shapes: the user pads and reshapes, with helpers | Decided | |
| 7.5 | How are padded samples made inert for entries? | Open | 7.4 |
| 7.6 | What is automatic sizing, given that confirming a size by compiling is costly? | Open | 7.1 |
| 7.7 | What does the target promise about sharding, and who owns device placement? | Open | 7.1 |
| 7.8 | `chunk_size` leaves the sampling kind | Decided | |

Notes:

- **7.1, one more setting** (2026-11-15, 19:32). By 2.9 the plan also says how a product of two full jets is
  taken over the orders: a scan with a carried shift (the default) or a loop with slices. The plan stays what it
  was: settings that change how a computation is carried out and never what it computes. What the default plan
  is otherwise is still 7.2.
- **7.2.** The default plan was defined as "today's behaviour". The lean default for what the model holds was the
  maintainer's choice between two options framed against today's behaviour.
- **7.3.** Decision 3 put the plan in `LocalModel` because threading it through about 25 optimizer functions was
  "too much". That is a cost-of-change reason. The maintainer has also said that a chunk size is conceptually a
  solver parameter, not part of the local model.
- **7.5.** The 0/1 mask was accepted with the maintainer's reservation on record.

## 8. The fitting layer

By 1.1 this section is in scope.

| # | Question | Status | Depends on |
| --- | --- | --- | --- |
| 8.1 | What is `LocalModel` in the target, and what does it ask the sampling layer for? | To confirm | 1.1, 6.1 |
| 8.2 | What is the fitting layer's kind object, beside the kind in the sampling layer? | Open | 3.4 |
| 8.3 | What is the optimizers' contract with a model, and what is compiled as one program? | Open | 1.1 |
| 8.4 | What do the display and the diagnostics read? | To confirm | 3.7 |
| 8.5 | Shared sample vectors: in the target or not? | Parked | |

Notes:

- **8.5.** Parked by the maintainer on 2026-11-14 as low priority. T3Polynomial's main fit has the same sample
  vectors at every input mode.

## 9. Names and verification

| # | Question | Status | Depends on |
| --- | --- | --- | --- |
| 9.1 | The names settled on 2026-11-14 | Decided | |
| 9.2 | Which of those names were chosen to match something that will no longer exist: the kind strings, `'sweep'` as a unit name, the sizing function names? | Open | 3.4, 6.3, 7.6 |
| 9.3 | What defines correct when old behaviour is not the reference: the generic twin, adjoint identities, finite differences? | Open | |
| 9.4 | What is the order axis called as a group, so that it is not taken for a batching axis, and what letter does it have in contraction strings? | Decided | 2.7 |
| 9.5 | The convention for arrows on the edges. An arrow shows the way a vector travels: where it leaves an array the index is plain, where it enters the index is dual. A contraction joins the two ends of one arrow. Every arrow of a frame points away from its store, towards the hole. Arrows are described by the way they point in the diagram: left-to-right, right-to-left, bottom-to-top, top-to-bottom | Decided | |
| 9.6 | The notation for the sampling variables: four letters and a basis. `mu`, `nu`, `xi`, `eta` name the four sides of a cut edge that can be contracted; the store of the basis follows the letter where the basis is not the usual one (`nu_L`, `eta_D`, `mu_R`, `xi_U`); then the grade. The reference is `SAMPLE_VARIABLES.md` | Decided | |

Notes:

- **Names and the T4S paper** (the maintainer, 2026-11-15, 21:59). "Trying to keep consistency with the t4s paper's
  notation through this refactor is an impossible task. We do not need to attempt it." So a name is chosen for the
  target alone; `sigma`, `tau` and the other letters the refactor frees may be given new meanings. (The notation of
  9.6 uses neither.) The library's notation map from the paper to the code will need rewriting when the names land.
- **9.6, decided** (2026-11-15, 23:23). The maintainer, on the seven drawings in formula style: "This is excellent
  and further confirms that this notation is the right one. I want to lock in this new notation." At 23:02 he
  had called the scheme "the best so far", with "great potential for confusion", and asked for the drawings to
  meet that.
  - **What is decided.** A variable is named by the side of the cut edge that is contracted (`mu` the left of a
    bond, `nu` the right of a bond, `xi` the leaf below a Tucker edge, `eta` the trunk above it), by the basis
    of the cut edge where it is not the letter's usual one (`L` for `mu`, `R` for `nu`, `D` for `xi`, `U` for
    `eta`), and by its grade. In formulas and drawings: the Greek letter, a tilde or a dot above it, the basis
    as a subscript.
  - **What it replaces.** `gamma` is `nu_L` and `zeta` is `eta_D` (both letters are retired; `zeta` had been
    decided at 22:11). The frame's `mu_dot` and `xi_dot` are `mu_R_dot` and `xi_U_dot`, and the bare `mu_dot`
    and `xi_dot` now mean the base point's. The frame's `nu_dot` and `eta_dot` are `nu_L_dot` and `eta_D_dot`,
    the same names as the base point's, because they are the same arrays. The six variables that had no name
    (notes to section 6) all have one.
  - **The reference** is `SAMPLE_VARIABLES.md`: the taxonomy, every definition as a recurrence and as a
    subnetwork, the types and shapes, and the map from the old names and from the library's. Checked by
    `scripts/review/sample_variables_check.py`. The earlier names and schemes are in `NOTATION_HISTORY.md`.
  - **Not decided by this:** see section 11 of `SAMPLE_VARIABLES.md`.
- **9.4.** The maintainer's concern: a batching axis means a bunch of the same thing bundled together, and it has
  fewer restrictions. The order axis is neither. The words the library already has: the contraction engine's
  "group" (an uppercase letter standing for zero or more axes), which is structural and says nothing about
  batching; "batch block" and "stack" for `C`, `W` and `K`; and "order axis" or "jet axis", lowercase `t`, for the
  one axis.
- **9.4, the maintainer's suggestion (2026-11-15): "optional axis".** The order axis is an optional axis: absent at
  order 0, present with length `order + 1` above. The phrase is not used anywhere in the library today. Still
  open: its letter in contraction strings. The engine needs an uppercase letter for anything that can be absent;
  today's strings use only `C`, `W` and `K` in uppercase, and write the order axis as lowercase `t`.
- **9.4, decided (2026-11-15).** The term is "optional axis", and the optional order axis is written `T` in
  contraction strings. Lowercase `t` stays the letter for the order axis where it is always present.

- **9.5, decided** (2026-11-15, 05:39 to 05:47). The rule (two ends of one arrow) was agreed at 05:39. The
  meaning of the arrows was the maintainer's lean ("inward pointing arrows towards a core should mean dual
  indices ... outward facing arrows from a core should mean plain"), Claude agreed, and he confirmed: "All good
  except the name suggestion." The full account is in sections 8 to 10 of `TYPE_CATALOG.md`.
- **9.5, the direction words.** Claude had suggested keeping "left", "right", "up" and "down" for the names of
  the stores only, and describing arrows relative to the hole. Rejected by the maintainer: "Describing
  directionality with respect to diagrams (ascii in the documentation here and tikz in papers) is immensely
  valuable to humans, even if there is the risk of misinterpreting or switching it (as I have already done). It's
  a risk that is worth it in my opinion." So both uses stand: a name says where a store sits, an arrow is
  described by the way it points in the diagram, and the left cores carry left-to-right arrows. The risk is
  accepted. What it asks of the documents is that each diagram shows its arrows.
- **9.5, the wording for the direction of an arrow** (2026-11-15, 05:49). The maintainer: "For directionality, we
  can use 'left-to-right' and 'right-to-left' rather than rightward/leftward to avoid confusion with stores." So
  an arrow along a bond is "left-to-right" or "right-to-left", and "left" and "right" alone stay with the stores.
  For the vertical direction, "bottom-to-top" and "top-to-bottom": Claude's proposal, confirmed by the maintainer
  at 05:50. "Up" and "down" alone stay with the stores.
- **9.5, the arrows of the base point** (2026-11-15, 15:49 to 16:10). The maintainer's convention: a gauge
  convention used unless otherwise stated, bottom-to-top and left-to-right, so that every edge points towards the
  last TT core; with the last `L` core and the first `R` core as the known exception to gauge correctness. It is
  the `D` and `L` rows of the frame, and the checker already uses it. Asked to confirm it with 2.4 he said "I
  think so". Claude's wording of the exception, to be corrected if wrong: those two cores are typed like the
  others, and what fails for them is `M M^T = I`, since they carry the norm (checked). Section 11 of
  `TYPE_CATALOG.md`.

## After the target

Not in this list, on purpose: the order of work, staging, adapters for migration, what T3Polynomial has to change,
the reference paper, and the library backlog items of finding F17.
