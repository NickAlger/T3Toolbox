NOT YET

# P3: review, round 1

Reviewer, 2026-11-17, on `graded-messages` at `a38d1a3` with the slice uncommitted in the working tree.

**Summary.** The mathematics is right everywhere I checked it by my own route, the two tiers are kept, the names
and signatures are the plan's, and the gate is valid for the tree as it stands. No blocking finding. Four
should-fix findings: one hole in a checker, one check in the frontend with no backend checker behind it (needs a
decision that is not the implementer's), one function that fails on a traced argument, one uncaught mutation.
Twelve minor findings. One question for P11 (the packed probe array) with my view.

---

## Findings

### Should fix

**1. `check_sample` accepts a sample whose `pp` is missing at order 1 and above, when it is given as `None`.**
- Place: `t3toolbox/backend/jets.py`, `check_sample`, `if pp is not None:` (line 762); `_sample_parts` returns
  `sample[1]` unexamined.
- Evidence: `jets.check_sample('apply', 2, (ww, None), shape)` returns nothing (also `'probe'`, and
  `(index, None)` for `'entries'`). The pair has length 2, so the form test passes, and the `None` skips every
  check of `pp`. The docstring says "at `order >= 1` it is the pair `(ww, pp)`".
- `(ww, None)` is the natural wrong input of a caller who builds the sample as `(ww, pp)` with `pp=None` left
  from an order-0 code path. No test covers it.
- Resolves it: at `order >= 1`, `pp is None` raises, naming it ("at order 2 the sample needs the directions pp;
  got None"); a test.

**2. The frontend checks `scales` with no backend checker behind it, and the same wrong `scales` pass unchecked
on the way in.** (The handover's issue 2 and choice 12.)
- Place: `t3toolbox/tucker_tensor_train.py`, `_check_coordinate_scales` (private, also called from
  `uniform_tucker_tensor_train.py` as `t3._check_coordinate_scales`); `backend/data_preparation.py`,
  `rescale_coordinates`, `t3_unscale_coordinates`, `ut3_unscale_coordinates`, which check nothing.
- Against: `TARGET_DECISIONS.md`, "Checking in the two tiers": "we must provide the *tools* to the backend user
  to do checks themselves"; "the requirement on the backend is that a checker exists for each thing the frontend
  checks". `PART1_PLAN.md` 7.5: "Every check the frontend makes calls a backend checker."
- Evidence that it matters in practice, beyond the letter of the rule: `rescale_coordinates` is a backend
  function that the frontend user reaches as `fitting.rescale_coordinates`, so nothing checks `scales` there for
  either user. `prep.rescale_coordinates('apply', 0, ww, y, scales[:2])` returns a sample of 2 modes with no
  error (the `zip` drops a mode), and a scale of length 1 broadcasts. The frontend user then gets the error only
  at `x.unscale_coordinates(scales)`, after the fit; the backend user never.
- My view: a backend checker belongs here. It is four lines, but the rule is explicit and the failure is silent.
  It would be called by both methods (replacing the private helper) and offered to the user beside
  `rescale_coordinates` in the docstrings.
- Whose call: it needs a new public name (for example `check_scales(scales, shape)` in
  `backend/data_preparation.py`), and every public name of Part 1 was approved by the maintainer (sections 3,
  11.3, 12). So the name is the maintainer's. Claude can instead record a stated exception ("within reason":
  trivial logic) and leave the code; that is Claude's to decide, not the implementer's and not mine. The
  implementer was right to raise it and not decide it. I hold the sign-off on it only until one of the two is
  recorded.

**3. `rescale_direction` fails when `length` is traced, and ignores a jax `length` in its dispatch.**
- Place: `backend/data_preparation.py`, `_scale_slots`: `powers = [float(length) ** t ...]`; dispatch from `y`
  alone (`is_jax_ndarray(y_TX)`).
- Evidence: `jax.jit(lambda L: prep.rescale_direction('apply', 2, (ww, pp), y, L))(2.0)` raises
  `ConcretizationTypeError`. With numpy data and an eager jax `length`, `pp` comes back a jax array and `y` a
  numpy array (mixed). The length is exactly the number a user computes inline from `data_block_norms` (O-14:
  "compute inline"), which on jax data is a jax scalar.
- Against: plan 7.3 ("`test_dispatch.py` jits every new public function"; "`tree_contains_jax` over every array
  input"). The dispatch test passes because it closes over `3.0`.
- Resolves it: the powers through `xnp` (for example `length ** xnp.arange(order + 1, dtype=...)` shaped to the
  jet), `length` counted in the dispatch; a case in `test_jit_jets_and_data_preparation` with `length` traced.
  If it is judged that `length` is a Python float by contract, say so in the signature comment and the
  docstring, and I withdraw this to minor.

**4. The exactness of the frontend's zero check is not pinned by any test.**
- Place: `tests/test_jets.py`, `TestFrontendDualJetToJet`.
- Evidence: the mutation `n_zero = int(np.sum(np.abs(np.asarray(weight)) <= 1e-8))` in `fitting.dual_jet_to_jet`
  passes all of `tests/test_jets.py`, `tests/test_data_preparation.py`, `tests/test_import_graph.py`. The
  handover's choice 3 ("exactly zero, with no tolerance ... a small weight is a legitimate weight") is therefore
  untested.
- Resolves it: a test that a small nonzero weight (say `1e-12`) is accepted in safe mode and gives the backend's
  numbers.

### Minor

**5. "Exactly zero" misses a weight whose square is zero.** The division is by `weight**2` in the result's dtype.
`fitting.dual_jet_to_jet(dual_f32, jnp.asarray([1., 1e-30, 2.], float32), 2)` passes the safe-mode check and
returns `inf` in slot 1; so does a float64 numpy weight of `1e-30` against a float32 dual jet (the backend casts
the weight down first), and `1e-200` in float64. O-20 says "where a weight is zero", so the present check is a
fair reading; the check that matches what the backend divides by is "`weight**2 == 0` in the dtype the backend
computes in", still with no tolerance. Claude's call.

**6. Two messages of `check_sample` do not say what is wrong.** `(ww, pp)` given at order 0 gives "ww must have
one array per mode: expected 3, got 2" (for entries: "index must be one integer array ...; got a tuple"); the
cause is that the order-0 sample is `ww` alone. With `d = 2`, `ww` alone at order 1 is read as the pair and gives
"ww must be a tuple or a list of 2 arrays ...; got a ndarray". Both are refused; the message could add "at order
0 the sample is the inputs alone, with no pp".

**7. `equalizing_weight` on all-zero norms warns on numpy and is silent on jax.** numpy: `RuntimeWarning: divide
by zero`, an error under `-W error`; jax: `inf`, silent. The backend `dual_jet_to_jet` was made silent on both
for this reason (choice 2). Also a 0-dimensional input gives a numpy scalar back, not an array
(`equalizing_weight(data_block_norms('apply', 0, y))` is `numpy.float64`), where `data_block_norms` took care to
return an array (choice 9). Same for `jet_to_dual_jet` and `dual_jet_to_jet` on a 0-dimensional numpy jet.

**8. A statement in the docstring of `ut3_unscale_coordinates` is false for packed scales.** "so the padding of
the mode axis comes out zero whatever it held" holds only for ragged `scales`. With `scales` already packed to
`(d, N)` and nonzero beyond `Ni`, garbage in the mode padding stays garbage (checked: real parts right to 2e-16,
mode padding nonzero). Either say it, or zero the packed scales beyond `Ni` from `shape` on the host. Accepting
packed scales is itself wider than the plan's "the same scales"; harmless.

**9. Dispatch is per array inside a probe tuple.** `jet_to_dual_jet([numpy, jax], weight, order)` returns
`(numpy, jax)`; `derivatives_to_taylor` the same; `data_block_norms` on the same input returns one jax array.
The house rule is one `tree_contains_jax` over all the inputs of the operation.

**10. A jax integer weight with a numpy float64 jet, x64 off,** warns ("Explicitly requested dtype float64 ...
not available") because the dtype is taken from the floating arrays only and then asked of jax. An edge.

**11. The asterisk of row 2.5 is missing.** The note to 2.5 says the documentation of the default weight "has to
carry that asterisk" (ones by default, subject to change in a later release). The module docstring of `jets`
and `check_weight` say "`None` means ones" without it. May be left to P19 if recorded.

**12. The references to the re-exported functions do not resolve in the rendered pages.** In the built
`autoapi/t3toolbox/fitting/index.html` the names `derivatives_to_taylor()`, ..., `t3_unscale_coordinates()`,
`jet_to_dual_jet()` are plain text (only the two module links and `dual_jet_to_jet` are links), and
`t3toolbox.fitting.rescale_coordinates()` in the docstrings of both `unscale_coordinates` methods is plain text
too: autoapi does not document an imported name under the importing module. A frontend reader gets no link to
the function's page. Targets `t3toolbox.backend.jets.<name>` and `t3toolbox.backend.data_preparation.<name>`
would link. (Row 9.7 and O-16 are about this reader.)

**13. Where a frontend reader learns to prepare data.** The teaching doctests are good (the three cases, the
weight trap, the probe kind, the bad-scaling walk) but they live in the backend modules and import
`t3toolbox.backend.jets`. The section added to the docstring of `t3toolbox.fitting` lists names. Enough for P3;
P19 should carry a frontend example that imports only `t3toolbox.fitting`.

**14. `prepare_derivative_data` (and the other functions that take `kind`) assume the ragged type.** With
`kind='probe'` and a packed array `(d,) + T + W + (N,)` it divides the mode axis by `t!`: a broadcast error when
`d != order + 1`, a silent wrong answer when `d == order + 1`. See the question for P11 below; a sentence in
these docstrings would do.

**15. `_KINDS` and `_require_kind` are duplicated** in `jets.py` and `data_preparation.py`, while
`_lowest_precision_dtype` is imported across the two. The dtype helper will be needed by the engine (P10); it
may belong in `common` then. Cross-module private imports have precedent in the backend (`entries` from
`apply`), so this is a note.

**16. `check_weight` cannot tell a transposed probe weight `T + (d,)` when `d == order + 1`.** Inherent in a
shape check; a sentence in the docstring would warn the user.

---

## The handover's 15 choices

Sound and the implementer's: 2 (silent division; the plan says "raises nothing"), 3 (exact zero; O-20; see
findings 4 and 5), 6 (`rescale_direction` at order 0 and `rescale_coordinates` for entries raise; the plan says
"probe and apply only"), 7, 8 (see finding 7), 9, 13 (the uniform method takes the ragged `scales`), 14, 15.

Sound, but they touch the public behaviour, so Claude should record them in the log:
- **1** (the type is read from tuple or not, where there is no `kind`). It follows from the decided signatures.
  Its consequence is the question for P11.
- **4** (the lower floating precision where two arrays meet). It is row 5.6 and O-11 applied to public helpers
  that the row does not name; consistent with both (the seed the engine makes through `jet_to_dual_jet` must
  follow 5.6 anyway). Checked: float32 with float64 gives float32 in every function, numpy and jax; a weight
  converted alone keeps its own dtype.
- **5** (`prepare_derivative_data` at order 0 returns the data and ones). The plan's sketch says `order >= 1`.
  Harmless, and in line with "order 0 as the same functions".
- **10** (`ut3_unscale_coordinates` also takes packed scales). Wider than the plan; see finding 8.
- **11** (the checkers take the ragged form only; a packed array is refused). Consistent with the plan's
  comments. Consequence for P14: `check_batch` is built from them, so a backend user holding a packed batch has
  no checker for it. Claude decides at P14 whether that is wanted.

Not the implementer's, and correctly not decided by them: **12** (finding 2).

## The handover's issue 1: the packed probe array (for P11)

The implementer is right. `jet_to_dual_jet(jet, weight, order)` reads one array as the apply type, order axis 0.
A packed probe array `(d,) + T + W + ... + (N,)` passed whole is misread, and silently:
- with a weight `(d,) + T`: the weight lands on the right axes (it is reshaped to `(d, T, 1, ...)`), and the turn
  is made on axis 0, so the result is the weighted jet with the MODES reversed and the orders not turned. No
  error. (Checked.)
- with `weight=None`: the modes reversed, no error.
- `derivatives_to_taylor` on a packed array: a broadcast error when `d != order + 1`, silently the mode axis
  scaled by `t!` when `d == order + 1`.
- at order 0 with a weight `(d,)` the packed array happens to come out right.
`tuple(packed)` is right in every case.

The ways out:
1. **A `kind` argument** on the four functions without one. Right on both layouts and no reading of types. Cost:
   four decided signatures change (plan 2.4; row 3.6's helper), every call site passes `kind`. The maintainer's.
2. **A rule read from the arguments.** There is none that works: the weight's rank fails with `weight=None`, and
   `d` against `order + 1` is ambiguous. Not an option.
3. **The engine passes mode by mode** (`tuple(r)` in, a stack out, or per site inside the map of O-19). No public
   change. Cost: a backend user on the uniform layout must know the step, which is against the razor's corollary
   ("don't leave a backend user one fiddly step short"), and the whole-array call stays a silent trap. The
   docstring names the step; it should also say what happens otherwise.
4. **A uniform twin under its own name.** A new public name. The maintainer's.

My view: 3 for Part 1, decided by Claude at P11 as a minor issue, because nothing public changes and the engine
maps one site at a time; with the trap stated in the docstring, and the maintainer told, since it is a silent
wrong answer on a public function. If P11 finds the engine holds the packed residual whole and has to split and
restack it at every seed, that is the moment to ask the maintainer for 1.

---

## What I checked and ran

All with `PYTHONPATH=/home/claude/t3toolbox PYTHONDONTWRITEBYTECODE=1 python3 -B`; scripts under `/tmp/p3rev`
(removed at the end); nothing run in parallel.

**The mathematics, by my own computation** (not the implementer's tests):
- *The dual jet.* Apply type at orders 0, 1, 2, 5, sample stacks `()`, `(4,)`, `(2,3)`, trailing axes `()`,
  `(2,)`, `(2,3)`; probe type at orders 0, 1, 3, three modes of sizes 3, 4, 5, the same stacks, tuples and lists.
  Against `dual[t] = weight[order-t]**2 * jet[order-t]` written out slot by slot; the round trip; no weight; the
  pairing identity entry by entry with `numpy.polynomial.polymul` (the coefficient of `s^order` of the product
  against `sum_t weight[t]**2 a[t] b[t]`). Worst difference 3.6e-15. Cases built to confuse the axes (`W`, `d`,
  `N` and `order + 1` all equal to 3): the order axis is axis 0 of each array, row `i` of a probe weight acts on
  mode `i`, at order 0 the probe weight is one number per mode.
- *Derivative data.* Derivatives of `dense_apply` and `dense_probe` along `w + s p` from a polynomial fitted at
  Chebyshev nodes and differentiated by `numpy.polynomial.polyder`, at orders 1, 2, 3, 5, three sample stacks:
  they equal the old `x.apply_derivatives` / `x.probe_derivatives` (2026.5.0 code still in the tree), and
  `derivatives_to_taylor` of them equals `dense_apply` / `dense_probe` with `order`; the reverse; slot `t` by
  `t!` by hand; entries through `x.entries_derivatives` against `dense_entries`. The objective: `1/2 sum
  (weight_der * (z_der - y_der))**2` computed on derivatives equals `1/2 sum (weight_taylor * (z_taylor -
  y_taylor))**2` after `prepare_derivative_data`, for apply and probe, with a random weight and with none
  (relative difference at rounding); the same value through `jet_to_dual_jet` and the pairing. No weight returns
  exactly `t!`, `(order+1,)` for apply and entries, `(d, order+1)` for probe; at order 0 ones in the shape of a
  weight, accepted by `check_weight`. Worst absolute difference 2.4e-12 (the polynomial fit).
- *Data preparation.* Block norms against sums of squares written out, the shapes for three kinds at orders 0 to
  3; `rescale_direction` against the dense functions at the rescaled directions (three kinds, two lengths);
  `rescale_coordinates` against the dense functions of the dense tensor divided by the outer product of the
  scales, at the rescaled sample, for probe and apply at orders 0 to 3: the rescaled sample and data belong to
  the tensor in rescaled coordinates; `t3_unscale_coordinates` returns the original Tucker cores and the TT cores
  as given; the probe values in the original coordinates come back; a stacked tensor. Worst relative difference
  1.9e-14. `equalizing_weight`: the floor is relative to the largest block; a zero block gets `1/(floor*max)`;
  all zeros gives `inf` (finding 7).
- *The uniform twin.* Three structures (one with `d = 1`), default and forced padding, garbage (1e3) written
  into every slot outside a real region that I derived from the ranks and mode sizes, not from the object's
  masks; ragged and packed scales: real parts equal the ragged function's to 2e-16; the masks are the input's
  own objects, host numpy bool; the TT supercore is returned as given; a stack of varying ranks element by
  element, structures equal. (Finding 8.)

**The two tiers.** `grep`: no import of `safety` under `t3toolbox/backend/`; no backend call of `check_sample`,
`check_data`, `check_weight`. The backend `dual_jet_to_jet` at a zero weight: `inf`/`nan`, no warning with
warnings as errors, on numpy and jax. The frontend one: raises in safe mode on numpy, eager jax, float32 jax, a
numpy dual with a jax weight, one zero in one slot of a probe weight, a 0-dimensional and a Python `0.0` weight,
`-0.0`, a list; divides in `safety.unsafe()` with the backend's numbers; divides under `jax.jit` with the weight
a tracer, the weight closed over (numpy and jax), and both closed over; divides under `grad` and `vmap` over the
dual jet. (Findings 4, 5.)

**The checkers.** Every valid input I tried is accepted: three kinds, orders 0, 1, 4, sample stacks `()`,
`(7,)`, `(2,3)`, `(1,)`, `(0,)`, tuples and lists, four integer dtypes of `index`, jax arrays, a numpy integer
order, a Python float weight at order 0. 31 wrong samples, 13 wrong data, 15 wrong weights: all refused except
`(ww, None)` (finding 1), an index out of range (documented), and the transposed weight of finding 16. Under
`jit` and `vmap` they run and raise at trace time. The two validators are textually the old ones
(`inspect.getsource` equal).

**The tests tested.** 86 mutations in a scratch copy (`/tmp/p3rev/scratch`, `python -B`), each run against
`tests/test_jets.py`, `tests/test_data_preparation.py`, `tests/test_import_graph.py`. 85 caught: a factorial
off by one (data; weight) and shifted down; the weight not squared; no turn, either direction; the turn on the
last axis and on axis 1; a turn at order 0 too; the weight along the trailing axes; the weight applied after the
turn; the probe weight's rows reversed or all row 0; the inverse multiplying; `errstate` removed; the backend
raising at zero; the conversions the wrong way; only mode 0 converted; the weight divided or by `t!` squared; no
weight giving ones; validation not converted or the wrong way; the default probe weight transposed; highest
precision for lowest; `check_weight` accepting a bare vector, a dimension of 1, the other kind's shape, `(1,)`
at order 0; `check_sample` without each of its six checks; `check_data` without each of its four; a changed
validator; block norms over all axes, keeping the sample axis, without the root, as absolute sums, stacked on
the wrong axis; the floor absolute, relative to the minimum, per row, squared; the powers of the length off by
one, reversed, inverted; `pp` not scaled or divided; `ww` scaled too; coordinates: `pp` not scaled, `ww`
divided, probe data multiplied or untouched, entries accepted; unscale dividing, doing nothing, squared, or
touching a TT core; uniform: ones in the padding of the scales, dividing, masks all true, modes reversed; the
frontend never checking, checking in unsafe mode, raising only when all are zero, being the backend function,
reading the trace from the dual jet alone; either method without its check; the check of `scales` allowing a
length of 1 or any count; a name missing from `fitting.__all__`; a backend import of `safety`; a backend
function calling a checker. **Not caught: 1**, the zero check with a tolerance (finding 4).

**The gate.** The implementer's run in `/tmp/gate_P3` started at 10:34:35; the newest file of the slice
(`backend/data_preparation.py`) was last modified at 10:21:59, and no file under `t3toolbox`, `tests`, `docs`
or `examples` is newer than the run's `PID` file. Its results: suite 1,562 passed, 5 skipped, 1 xfailed, 65,775
subtests; module doctests 302; quickstart, 48 pages, 14 examples pass; sphinx 0 warnings. So the gate stands for
this tree and I did not rerun it whole. Rerun by me: `tests/test_jets.py`, `tests/test_data_preparation.py`,
`tests/test_import_graph.py`, `tests/test_jax_absent.py` (65 passed, 499 subtests); the module doctests of
`backend/jets.py`, `backend/data_preparation.py`, `fitting.py`, `tucker_tensor_train.py`,
`uniform_tucker_tensor_train.py` (116 passed); the three new tests of `tests/test_dispatch.py` (3 passed, 121
subtests).

**Form.** Signatures in the house style (one argument per line, shape comments); body locals carry the layout;
`xnp` for data, masks passed through untouched as host numpy; `np.shape` and `np.dtype` only on static
information. O-14: no estimate is shipped; `rescale_direction` is in the plan (N9, row 6.7). Every public name
is in sections 2.4, 2.5, 3 or 11.3; none is new.

**The tree.** Left as found apart from this file; the scratch copy removed; nothing left running.
