# Scaling, weighting and normalization in T3Polynomial: a survey

**Why this exists.** On 2026-11-15 (16:19), before deciding target decision 2.5 (the misfit on Taylor
coefficients), the maintainer asked for context from the consumer: "can you look through the T3Polynomial
documents and history for content related to scaling, weighting, and normalization? Both how it is done
currently, and the historical approaches we tried. The current approach there should not be taken as an edict, but
it can be a piece of evidence that helps us make a good choice."

**His three guiding ideas, in his words.** "The first is what we are trying to do: approximate a function with a
polynomial. The second is that the different terms in the polynomial depend on the length-scale in the problem;
there is no a-priori way to say which is more important unless a length scale is given, and a natural choice of
length scale is 1.0. The third is the empirical observation that normalizing to try to equalize the influence of
each term in the objective seems to help the fit a lot."

**What was read.** T3Polynomial at commit `dbda0de` (2026-11-14), with its full history of 172 commits, read only:
`t3polynomial/jets.py`, the fit scripts, `docs/`, `dev/`, and the commit log. Also T3Toolbox's own record
`dev/data_shaped_residual_weight_reviews.md`. Nothing in either repository was changed. This is Claude's reading;
the numbers are quoted from those documents, not re-measured.

## 1. How it is done now

The current recipe has five separate steps. All but the last are exact changes of variables made when the data
are assembled; only the last is a weight.

| Step | What it does | Where |
| --- | --- | --- |
| Whitened coordinates | The inputs are covariance-whitened reduced coordinates `z`, homogenized as `(1, z)`. The outputs are whitened by a mass matrix. So a length of 1 in `z` is one prior standard deviation | `dev/plan.md`, sections 2 and 4 |
| Unit probes | Each sample's direction `v` and output functional are rescaled to unit norm, with the exact compensating rescaling of the jets (`y_t` scales as the norm of `v` to the power `t`). The sample points are never rescaled | `assemble_probe_fit_data(normalize=True)` |
| Centering | The training mean of the values is subtracted from the order-0 data, so the value block is measured against the map's variation, not its mean | `center=`, on by default in the fit scripts |
| Input scale `kappa` | The fit is of `f(kappa z')` in `z' = z / kappa`. The order-`t` jets are multiplied by `kappa^t`. The default is `kappa_auto = (norm of centered y_0 / norm of y_J)^(1/J)`, an estimate of the Taylor radius | `input_scale=`, `'auto'` by default in `d10` |
| Block-equalizing weights | `weight[mode, slot] = 1 / (norm of that training block)`. Every live block contributes its squared relative error equally. A structurally dead block gets weight exactly 0. Computed from the training data and then frozen | `per_block_weights`, passed as T3Toolbox's `weight=` |

Two more facts about the current state.

- **The jets are derivatives, with no factorials.** `y_t = D^t f(z) v^t`. The documents note that these are
  "the Taylor coefficients of the line ... up to factorials".
- **The score used to select a level is not the training loss.** Selection moved from the weighted misfit to
  forward-only and to "forward plus summed reverse" scores with equal weight per slot (section 2, 2026-08-27).

## 2. The history, in order

| Date | What was tried | What was found |
| --- | --- | --- |
| 2026-07-12, the plan and round one | Unit directions and functionals. One weight per order, `1 / RMS_t`, taken over all mode blocks of that order. It was the only weighting the library had | The jets decay steeply with order (dataset RMS 3.1e-2, 1.5e-3, 1.2e-4, 1.9e-5), so the weights reach 5e4 at order 3. The measured gradient components were 0.19 to 0.90 percent of the data energy per order: almost absent from the loss |
| 2026-07-12, proposed and not run | A reverse scale on the functional slot; a homogenization scale `(c, z)`; normalizing the sample points | Recorded as exact reweightings of the same problem. Point normalization was marked as an experiment because it reweights samples |
| 2026-07-13, round two | T3Toolbox gained `weight[mode, order]`. Block-equalizing weights replaced the per-order ones | Validation error of the values unchanged; the gap between training and validation much narrower; the symmetry of the fitted tensor improved; about 45 percent faster at high rank, "consistent with better GN conditioning from block-normalized residuals". What remained was an imbalance inside one row: the derived homogenization entry was 98.5 percent of the reverse block's energy at order 0 |
| 2026-07-13, discussed | The homogenization scale `c` | Two first-principles anchors conflict: `c = 1` is right for the representation, `c` near the typical point norm (about 10) is right for the information. "One knob cannot satisfy both"; the ideal was named as a per-component residual metric |
| 2026-08-27 | The weighted validation misfit as the selection score | It selected the wrong level: rank correlation with test error 0.56, against 0.999 for the forward-only order-0 error. Four of the five equalized blocks are reverse blocks, so the score was dominated by the parts that overfit first |
| 2026-08-27 | Counting unknowns per order ("face counting") | Each order receives the same amount of data, but the number of unknowns it determines grows by about `n / k` per order. The ratio of data to unknowns fell from 18,367 at order 0 to 0.6 at order 3, and the error graded with it |
| 2026-08-27 | The floor `1e-12` in the block weights | It turned a structurally dead block into a weight of 1e12. Replaced by an exact zero and a mask |
| 2026-09-02, deferred | Scaling each sample's direction by its own convergence radius | The one exact per-sample scaling the formulation admits: a direction of length `c` multiplies the order-`t` residual by `c^t`. "Trust each jet to its radius". Not run |
| 2026-09-03 | Centering | On the eigenproblem 84 percent of the value error was a constant offset, which cost the loss 40 times less than a derivative residual. Centered: 0.88 to 0.26 |
| 2026-09-03 | The input scale `kappa` | The jets spanned seven decades across the orders. "Block-relative weights remove that from the residuals but not from the parametrization". With `kappa_auto = 21` the same data went from 0.87 to 0.070, in a third of the time, with CG converging in 8 to 14 steps. `kappa` also turned out to be the homogenization scale of July |
| 2026-09-03, deferred upstream | A residual weight with the shape of the data, so that a weight per sample is possible | T3Toolbox reviewed a plan and deferred it. Its revised recommendation: keep the `[mode, order]` matrix as a static part of the kind, and add a second, separate argument for a data-shaped weight that is a traced array; the effective weight is their product |
| 2026-09-23 | Whitening the directions by the active subspace, to balance the samples by a change of coordinates | Negative: value error 0.0135 against 0.0060. It "rescales what the jets measure"; the estimated Taylor margin fell from 4.5 to 10 down to 0.8 to 0.9. "The per-row residual weight ... balances without touching what the jets measure, and is the mechanism to prefer" |
| 2026-10 and 2026-11 | The current scripts | Centering on, `kappa` automatic, block-equalizing weights, unit directions |

## 3. What the history separates

Read together, the record separates three things that the word "scaling" covers.

1. **The definition of the loss: which residuals count for how much.** This is what a weight is. The
   block-equalizing weights are a weight, and so is any choice between derivatives and Taylor coefficients.
2. **The conditioning of the unknowns.** `kappa` is not a weight. On the output mode it leaves the loss
   unchanged, because the block-relative weight cancels it. What it changes is the coordinates of the tensor:
   its degree-`k` corner is multiplied by `kappa^k`. Its large effect was on the solver.
3. **The score by which a fit is judged.** The weighted loss was a poor score. What tracked the test error was
   the error of the values and of the low orders.

## 4. Observations that bear on target decision 2.5

These are Claude's, offered as evidence for the maintainer's decision. None is a decision.

- **Under block-equalizing weights, derivatives and Taylor coefficients give the same loss.** A weight of
  `1 / (norm of the block)` is unchanged when the data and the model of one slot are both multiplied by a
  constant. So for T3Polynomial as it fits today, the choice of the default weight in 2.5 changes nothing: the
  adapter of 2.3 only has to convert the data, and the objective comes out the same.
- **A length scale is already an input: it is the length of the direction.** For an input jet `w + s p`, the
  Taylor coefficient of order `t` scales as the length of `p` to the power `t`. So the sum of squares of the
  Taylor coefficient residuals, with weight 1, is the error of the terms of the series at `s = 1`, the point
  `w + p`. A different length scale is a different length of `p`, or a rescaling of the inputs as `kappa` does. With
  derivatives the same sum carries `t!^2`, which no length scale produces.
- **`kappa` and the per-sample radius are the same move.** `kappa` rescales every direction together; the
  deferred idea of 2026-09-02 rescales each sample's step by its own radius. Both act through the length scale,
  not through the weight.
- **Equalizing is a policy computed from the data; the library supplies the mechanism.** The equalizing weights
  need the training data, centering first and `kappa` first, an exact zero for dead blocks, and freezing. All of
  that is in T3Polynomial today, and T3Toolbox only applies the matrix it is given.
- **Three requirements on the weight come out of the record.** A weight of exactly zero must remove a block
  entirely, since it is how dead cells are masked. A weight per sample has been wanted since September and is
  deferred upstream. And T3Polynomial's older per-order weights, and its own `weighted_misfit`, use today's rule
  that a bare vector means one weight per order; its current scripts pass the full matrix.
- **The loss and the score are different objects.** Whatever 2.5 fixes for the misfit, the record says not to
  read the weighted misfit as the measure of the fit. That bears on 8.4, what the display and diagnostics read.
