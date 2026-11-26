# Response r2 to the critic's second review

Written by the writer, 2026-11-17, against `e1e2354` (the revised draft `4f5b2b7` plus `REVIEW_r2.md`). All four
findings are accepted, and the note on O-37 is taken. The orchestrating session's guidance of this round is followed.

**1. (should fix) `pack_vectors`.** Accepted. I checked the code first. Today a mismatch in `W` makes `xnp.pad` (a
different rank) or `xnp.stack` (different sizes) raise (`backend/ut3_operations.py`, lines 128 to 152). P11 now gives
`pack_vectors` and `pack_if_ragged` a keyword `len_J=0`:
- the `len_J` leading jet axes are brought to their largest size across the modes with zeros;
- every other axis but the last must still agree, so a mismatch in `W` raises as today;
- with `len_J=0` it is today's function, so its existing calls are unchanged: 55 in the library and 36 in the tests,
  by an `ast` count of calls of the two names (`ut3_sampling.py` 27, `utv_sampling.py` 25, and one each in
  `uniform_fitting.py`, `data_preparation.py` and `ut3_operations.py`; in the tests, 20 in `test_uniform_curvature.py`,
  6 in `test_uniform_fitting.py`, 5 in `test_curvature.py`, 4 in `test_dispatch.py`, 1 in `test_uniform_probing.py`).

It takes the count, not the jet shape. The function needs to know how many leading axes may differ, and its target
sizes are the largest of the modes' own sizes, which are not `F`, since an input may have more slots than the box or
fewer. The count is also what `contract` takes for a group (`len_W=`, `len_C=`), so the name is `len_J` (proposed,
N41). Where it comes from at each call site: the engine (P11), the uniform twins (P12) and `pack_sample` (P14) all
have `order` and pass `len_J=len(order)`. `pack_data` passes nothing, because the data have the full jet sizes `F` in
every mode.

Where it is listed:
- section 2.4, beside the twins, as a change to two public functions of `ut3_operations`;
- P11's "What breaks": nothing, a new keyword with the old default;
- P11's three tests: unequal jet sizes packed with zeros (and unpacked back); a `W` mismatch raises at every
  `len_J`; `len_J=0` equals today's result. A fourth test: a packed array comes back as it is.
- Section 7.2, the account (section 3, the design page, the table of section 6) and 5.2 (P11 waits on N41) follow.

**2. (minor) O-44 (b) with O-46.** Accepted. Both points now name the interaction. O-44 gives three ways out, without
picking, each with its cost:
- (b1) no default for probe and apply only, through a sentinel the oracle resolves when it is built;
- (b2) the entries oracle takes no `order` at all;
- (b3) accept `order=()` for entries.

I add, as the writer, that with (b) I would take (b1). The maintainer's.

**3. (minor) O-44 (c)'s scope.** Accepted. (c) now names what it covers: the merged sampling functions and their
twins (P12), the frontend methods (P13), the dense functions (P4b), the engine (P5) and the oracle (P14). The
reference module (P4c) follows the engine. The functions of `jets` and `data_preparation`, `pack_batch` and the
checkers already have no default.

It gives a size: about 300 calls in `t3toolbox/`, `tests/` and `examples/`, by an `ast` count of calls of the probe
and apply names in all their forms. That count includes local helpers of the same names, so it is an upper bound.
There are also 124 lines of doctests and pages, by `grep`.

It names the slices that would change beyond their present text: P4b, P4c, P5 to P12, P13 (far more than its
doctest), P14, P16, P17, and section 9's row `T.probe(ww)`.

**4. (minor) The account's wording.** Fixed:
- "memoization on the jet shape and shapes" is now "on the jet shape and the operands' jet sizes";
- "a new input jet sizes recompiles" is now "input jets of other jet sizes recompile".

**On O-37.** Reworded. The reach from `t3toolbox.fitting` is decided, by the answers of 2026-11-16 at 20:15 ("The
preparation functions and the jet helpers are reachable from `t3toolbox.fitting` under the same names") and by O-16.
P4d does it. The open point is the top-level export only.

**Your note on finding 5 of round 1.** Taken: section 6's jet matrix now says, in a parenthesis, that the curve that
reaches the outputs is the `(2, 3)` mode's, along `s_2`, and that the third slot of the `(3, 2)` mode is clipped.

**Disagreements.** None.
