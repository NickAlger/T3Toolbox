NOT YET

# P3: review, round 2

Reviewer, 2026-11-17, on `graded-messages` at `a16fb4a` with the slice uncommitted in the working tree. Against
`P3_RESPONSE_r1.md` (with its two addenda), the updated `P3_HANDOVER.md`, and the decisions made since round 1:
`kind` on the four functions (the maintainer, 12:02), `check_coordinate_scales` and fitting's own
`rescale_coordinates` (Claude).

**Summary.** Everything round 1 asked for is done and holds under my own tests, the `kind` change is right
(recomputed through the new signatures; the packed probe array comes out right on numpy, eager jax and under
`jit`), and the gate is valid for this tree. **One thing stands between this and a sign-off, and it is one line:**
`ut3_unscale_coordinates` fell out of `data_preparation.__all__` when the checker was added, and its page is gone
from the rendered API (finding 17). Four minor findings (18 to 21) may be fixed without another round. For 17 I
need no new round of review beyond seeing the line, a test that pins it, and the page back.

---

## The sixteen findings of round 1

| # | State | How I tested it |
| --- | --- | --- |
| 1 | Resolved | `(ww, None)`, `[ww, None]`, `(index, None)` at order 2, three kinds: each raises "needs the directions pp ... got (ww, None)". Good samples still pass. Mutant "pp None accepted again": caught |
| 2 | Resolved (decided: a public checker, and fitting's own `rescale_coordinates`) | Below, "The scales" |
| 3 | Resolved | `jax.jit(lambda L: rescale_direction(kind, 2, (ww, pp), y, L))(2.0)` runs for the three kinds and gives `(1, 2, 4)` times the slots; numpy data with an eager jax length gives jax `pp` and jax `y`. The inputs `ww` / `index` are returned as the object given (decided, documented): accepted. Mutants `float(length)` again, the length left out of the dispatch: caught (by the dispatch tests only) |
| 4 | Resolved | My round-1 mutant (the check on the weight with a tolerance of `1e-8`) now fails `tests/test_jets.py`. Also caught: the check on the weight exactly; a tolerance of `1e-250` on the divisor; the divisor computed from the weight in its own dtype |
| 5 | Resolved (decided: the check is on the divisor) | Raises in safe mode: float32 jax `1e-30`; float32 numpy `1e-30`; a float64 weight `1e-30` against a float32 dual jet, numpy and jax; `1e-200` in float64. Passes: `1e-8`, `1e-150`, and a float64 `1e-30` against a float64 dual jet. The frontend calls `backend_jets._squared_weights(kind, dual_jet, weight, xnp)` and the backend divides by the result of the same call, so the two cannot differ |
| 6 | Resolved | The pair at order 0 (probe, entries, `d = 2`) and `ww` alone at order 1 with `d = 2` each end with the sentence that names the cause; a wrong mode size gets no such sentence; good `d = 2` samples still pass. Both hints removed: caught |
| 7 | Resolved | `equalizing_weight` on zeros: `inf`, no warning with warnings as errors, numpy (1-d, 2-d, 0-d) and jax; a 0-dimensional numpy input gives an array, also from the four jet functions and the weight conversion. Mutants: caught |
| 8 | Resolved, by making the sentence true | Garbage (1e3) in every padded slot of both supercores and 777 in the packed scales beyond `Ni`, three structures: real entries equal the ragged function's exactly; the mode padding is zero for ragged and packed scales; the rank padding within `Ni` is the input's garbage times its scale; masks the same objects; TT supercore as given. Mutants (scales not zeroed; the mode mask one too wide, one too narrow): caught |
| 9 | Resolved | A probe tuple `[numpy, jax]`, a numpy jet with a jax weight, numpy data with jax validation data, one jax scale among numpy ones: every output leaf is jax, in ten calls. Mutants (dispatch per array; on the jet alone): caught (dispatch tests only) |
| 10 | Resolved | A jax integer weight, a jax float32 weight, a jax length, jax scales, each against numpy float64 with x64 off: float32 out, no warning. Mutant "no jax cap": caught |
| 11 | Resolved | Read: the module docstring of `jets`, the `weight` entries of the two dual-jet functions, the frontend one and `check_weight` carry the asterisk; `prepare_derivative_data` separates its own `None` from the fit's default |
| 12, 13 | Deferred to P19, as instructed | Listed in the handover under "For P19". See finding 20 for one link the handover says was not looked at |
| 14 | Resolved | Read in the five docstrings |
| 15 | Resolved | One `_KINDS`, in `jets` |
| 16 | Resolved | The docstring states the limit; a test pins it |

---

## The `kind` change, verified again

**Through the new signatures, by my own computation** (the method of round 1: the formula written out slot by
slot, and `numpy.polynomial` for the pairing and for the derivatives):
- *The dual jet, its inverse, the pairing identity.* 216 cases: three kinds, orders 0 to 5, sample stacks `()`,
  `(4,)`, `(2,3)`, trailing axes `()` and `(2,)`, with a weight and with `None`. Worst difference 3.6e-15. The
  frontend `dual_jet_to_jet` gives the backend's arrays exactly, in safe mode and in `safety.unsafe()`, in every
  case. Probe returns a tuple, apply and entries one array. With every size equal to 3 the order axis is still
  axis 0 of each array and row `i` of the weight still acts on mode `i`.
- *The derivative conversions.* Orders 0 to 5, three sample stacks, three kinds, against derivatives from
  `polyfit` and `polyder` along `w + s p` (for entries along `e_index + s p`), against the old
  `apply_derivatives`, `probe_derivatives`, `entries_derivatives`, and slot by slot by `t!`. The objective on
  derivatives equals the Taylor objective after `prepare_derivative_data`, with a weight and without; no weight
  returns exactly `t!` in the right shape. Worst relative difference 1.2e-14.
- *The frontend in the three modes.* Zero weight: raises in safe mode (numpy, eager jax, entries, one zero in one
  row of a probe weight, a packed probe dual jet, 0-dimensional, a Python `0.0`, lists); divides in `unsafe()` with
  the backend's numbers; divides under `jit` with the weight traced, closed over (numpy, jax) and both closed
  over, and under `vmap`.

**The packed probe array.** 81 cases: `(d, order)` in `(3,2)`, `(4,3)`, `(1,2)`, `(1,0)`, `(3,0)`, `(2,5)`,
`(6,5)`, `(3,1)`, among them `d == order + 1` (the case that was silently wrong) and one mode; sample stacks
`()`, `(4,)`, `(2,3)`; with a weight and without; `jet_to_dual_jet`, both `dual_jet_to_jet`,
`derivatives_to_taylor`, `taylor_to_derivatives`, `prepare_derivative_data`. Each against the definition I wrote
on the packed array (weight on axes 0 and 1, the turn on axis 1), against `tuple(packed)` and `list(packed)`
(equal exactly), on numpy, eager jax and under `jit`. Worst difference 8.9e-16. The result is always a tuple of
`d` arrays, for one mode too.

**A mismatched type.** The implementer's three descriptions are true:
- a tuple with `kind='apply'` whose modes differ in size: `ValueError` from numpy (inhomogeneous shape);
- equal sizes, `d != order + 1`, with a weight, or in a conversion: a broadcast `ValueError`;
- equal sizes, no weight: the modes turned, no error.
One more case, covered by the docstring's "or a wrong result": equal sizes with `d == order + 1` and a weight, or
in a conversion, gives a wrong result with no error. The reverse (an apply array given with `kind='probe'`) is
also silent in some shapes. "No guard" is the right reading of the rule: the backend gives checkers, and
there is one. `check_data(kind, order, sample, y)` refuses a tuple for `'apply'` and a single array for
`'probe'`. It takes the ragged form only, so for a packed array the user has no checker; that is the question
already recorded for P14.

**Nothing else is decided by the type of the object.** The remaining `isinstance(..., (tuple, list))` tests in
the two modules are in the checkers, where the form is what is checked, and `pack_if_ragged` on `scales` in
`ut3_unscale_coordinates` (the library's own packedness rule, not the type of `z`). `data_block_norms`,
`rescale_direction` and `prepare_derivative_data` iterate a probe object, so they take a packed array the same
way.

**Signatures.** `weight` is required with no default in `jet_to_dual_jet` and both `dual_jet_to_jet`
(`TypeError` when left out); the two conversions have no weight; `derivative_weight_to_taylor_weight(weight,
order)` is unchanged; all as the plan's updated sketches (2.4) and the note in `TARGET_DECISIONS.md`. An unknown
kind raises in all of them.

**Callers.** A search for the four names over `t3toolbox`, `tests`, `docs`, `examples`, `README.md` and
`/home/claude/t3polynomial` finds them only in `backend/jets.py`, `fitting.py`, `tests/test_jets.py`,
`tests/test_dispatch.py`, `tests/test_jax_absent.py`, all in the new form. Every other hit is a note under
`dev/`.

## The scales

- **`check_coordinate_scales(scales, shape)`.** Passes lists, tuples, jax arrays, integer scales, a numpy `shape`,
  one mode, and zero, negative, infinite and `nan` values (it looks at no value). Refuses, each with a message
  that says what is wrong: an array packed to `(d, N)`, a short and a long list, an empty list, a scale of length
  1, a wrong `N`, modes swapped, a number, a 0-dimensional array, two axes `(1, N)` and `(N, 1)`, an inner list,
  `None`, a number for the whole, a dict, a generator. Runs under `jit` and raises at trace time. No backend
  function calls it (two mutants that make one do so: caught).
- **The frontend `rescale_coordinates`.** Good input: the backend's arrays exactly, probe and apply, orders 0 and
  2, in safe and unsafe mode. Six wrong forms of `scales`, each in safe mode, in `unsafe()`, under `jit`, and
  under `jit` inside `unsafe()`: all raise. `'entries'` and an unknown kind raise the backend's messages. It is
  not the backend object; `fitting` has no `check_coordinate_scales` and no `ut3_unscale_coordinates`. For a
  sample of the wrong form see finding 18.
- **The two methods** refuse the wrong forms, in unsafe mode too, and agree with each other on good scales.

---

## New findings

### Should fix

**17. `ut3_unscale_coordinates` is no longer in `data_preparation.__all__`.**
- Place: `t3toolbox/backend/data_preparation.py`, lines 90 to 97. The list now ends `'t3_unscale_coordinates',
  'check_coordinate_scales'`; in round 1 it ended `'t3_unscale_coordinates', 'ut3_unscale_coordinates'`. The
  handover and the response do not mention the change, so I take it for an accident of adding the checker.
- Evidence: `data_preparation.__all__` at run time has six names without it. In the round-1 gate
  (`/tmp/gate_P3/sphinx_html/autoapi/t3toolbox/backend/data_preparation/`) there is
  `ut3_unscale_coordinates.html`; in `/tmp/gate_P3_r3/...` there is not. So a public function of the plan (2.1,
  2.4, 11.3; O-34) is absent from the rendered API and from `from ... import *`. The function itself works and
  its doctest still runs.
- No test sees it: dropping any name from that `__all__` passes the slice's tests, the slice's dispatch tests
  and `test_jax_absent.py` (two mutants, below). `tests/test_jets.py` pins `jets.__all__` through its doctest
  test, and nothing does the same for `data_preparation`.
- Resolves it: the name back in the list; a test that pins `data_preparation.__all__` (the seven names), as the
  one for `jets`.

### Minor (may be fixed without another round)

**18. The frontend `rescale_coordinates` blames `scales` for a sample of the wrong form.** `ww` alone at order 2
gives "scales must be a tuple or a list with one array per mode (len=d=7, ...); got length 3"; a pair at order 0
gives numpy's "inhomogeneous shape" error; `None` a `TypeError`; numbers in place of vectors an `IndexError`. The
function reads the mode sizes off the sample and takes the sample to be good. A sentence in the docstring would
do, or a guard on the form before the sizes are read.

**19. A 0-dimensional `length` does not count for the dtype in `rescale_direction`.** A numpy float32 length
with float64 data gives float64: the length is cast to the dtype of the array it multiplies. By O-11 an array
whose entries affect the output counts; by the engine's rule "constants are Python scalars" a number does not.
The docstring says what is done. Claude's call; no change needed if a length is held to be a number.

**20. The link the handover says was not looked at.** `t3toolbox.fitting.rescale_coordinates` now has its own
page, so the references in the two `unscale_coordinates` docstrings can resolve. For P19 with findings 12 and 13.

**21. The plan does not yet carry the two decisions of this round.** `PART1_PLAN.md` has the `kind` signatures,
but not `check_coordinate_scales` (2.1, 2.4, 7.1, 11.3) and not that `fitting.rescale_coordinates` is fitting's
own function (2.5 still lists it among the names that are the backend objects). They are recorded in
`TARGET_DECISIONS.md`. Claude's document, not the implementer's.

An observation, no action: with a float32 weight of `1e-22` the squared weight is a nonzero denormal, the check
passes, and the division overflows to `inf` with numpy's `RuntimeWarning: overflow`. That is floating point, not
a zero divisor, and the check does what it says.

---

## The mutations

81 in a scratch copy (`/tmp/p3rev2/scratch`, `python -B`, `PYTHONDONTWRITEBYTECODE=1`), each against
`tests/test_jets.py`, `tests/test_data_preparation.py`, `tests/test_import_graph.py`, and, when those passed,
against the six slice tests of `tests/test_dispatch.py` and `tests/test_jax_absent.py`.

**79 caught.**
- *`kind`:* the type read from the object again; a packed array read as one array under `'probe'`; the modes of
  a packed array taken along axis 1, and along the last axis; the modes reversed; an equal-shaped result returned
  packed; the return type taken from the count of arrays; a list returned; entries treated as the probe type; the
  weight's row always 0, and reversed; a mode axis on the default weight of entries; an unknown kind accepted
  (two functions); validation data converted as apply; the frontend passing `'apply'` on; the frontend checking
  only mode 0 of a probe weight.
- *The scales:* the frontend function with no check, with a check only in safe mode, with the check skipped under
  a trace (caught by the dispatch test only), with the sizes read off the scales, with a truncated shape, being
  the backend object, giving wrong numbers; the checker looking at values, not checking the count, allowing a
  length of 1, not checking the rank, allowing an inner list, allowing any sequence; either method without its
  check; `rescale_coordinates` and `t3_unscale_coordinates` calling the checker; `fitting` re-exporting it.
- *The divisor:* the four of finding 4; the check skipped for jax weights; never made; made in unsafe mode; the
  backend squaring the weight in its own dtype.
- *The other fixes:* those named in the table above.
- *The mathematics again, on the rewritten code:* the weight not squared; no turn, either direction; the turn on
  the last axis and at order 0; the weight along the trailing axes; the inverse multiplying; `errstate` removed;
  factorials off by one (data, weight); the conversions swapped; the weight conversion dividing; no weight
  giving ones; the floor absolute; block norms over all axes; `pp` not scaled; probe data multiplied; unscale
  dividing; uniform modes reversed; uniform masks all true; a backend import of `safety`.

**Not caught: 2**, both the same gap: `'t3_unscale_coordinates'` dropped from `data_preparation.__all__`, and
`'check_coordinate_scales'` dropped from it (finding 17).

## What I ran

All with `PYTHONPATH=/home/claude/t3toolbox PYTHONDONTWRITEBYTECODE=1 python3 -B`, nothing in parallel.

- My own scripts for everything above (the dual jet, the conversions, the packed array, the tiers and the
  underflow cases, dispatch, the checker, the frontend function, the uniform padding).
- `tests/test_jets.py`, `tests/test_data_preparation.py`, `tests/test_import_graph.py`,
  `tests/test_jax_absent.py`: 80 passed, 547 subtests.
- `tests/test_fitting.py`: 45 passed, 238 subtests.
- The module doctests of `backend/jets.py`, `backend/data_preparation.py`, `fitting.py`,
  `tucker_tensor_train.py`, `uniform_tucker_tensor_train.py`: 118 passed.
- The six slice tests of `tests/test_dispatch.py`: 6 passed, 173 subtests.
- **The gate, not rerun whole.** The implementer's run in `/tmp/gate_P3_r3` started at 12:10:25; the newest file
  of the slice (`tests/test_jets.py`) was last modified at 12:10:13, and no file under `t3toolbox`, `tests`,
  `docs`, `examples` or `README.md` is newer than the run's `PID` file. Its logs: suite 1,580 passed, 5 skipped,
  1 xfailed, 65,875 subtests; module doctests 304; quickstart, 48 pages and 14 examples pass; sphinx 0 warnings.
  It is valid for this tree. Finding 17 changes no test count; after the fix the sphinx step is worth running
  once, to see the page back and no warning from it.

**The tree.** Left as found apart from this file (checksums before and after agree); the scratch copy removed;
nothing left running; no commit.
