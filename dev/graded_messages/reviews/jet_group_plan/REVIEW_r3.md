# Review r3 of the jet-group plan change (60c354e)

**Verdict: AGREED.** I would let an implementer start on P4a from this text once the maintainer has answered the
open points. What remains are two optional notes on wording; neither blocks.

Written by the critic, 2026-11-17. I checked `RESPONSE_r2.md` against `git diff e1e2354 60c354e` and against the
code.

## The four findings of r2

**1. `pack_vectors`. Resolved.** P11 adds the keyword `len_J=0` to `pack_vectors` and `pack_if_ragged`. I checked
it against the code and the call sites:
- **Behaviour.** Only the `len_J` jet axes may differ across the modes; they are brought to their largest size with
  zeros. Every other axis but the last must still agree. A mismatch in `W` still raises, as it does today in the
  `xnp.pad` or the final `xnp.stack` of `backend/ut3_operations.py`, lines 128 to 152.
- **Default.** With `len_J=0` the function is today's function.
- **Counts.** My own `ast` count of calls of the two names is 55 in `t3toolbox/` (`ut3_sampling` 27, `utv_sampling`
  25, one each in `uniform_fitting`, `data_preparation`, `ut3_operations`) and 36 in `tests/`. This is exactly the
  response's breakdown (`/tmp/critic/pk.py`).
- **Placement.** The change is listed in section 2.4, and P11 has its tests and its "What breaks" entry.
- **The count, not the target sizes.** The reason given is right: the target is the largest of the modes' own
  sizes, which need not be `F`.
- **The name** is proposed as N41, and P11 now waits on it.

The two questions put to me:
- **"Leading" is right.** It means the leading axes of each mode's unpacked array, `J_in_i + W + (Ni,)`. The packed
  result puts `(d,)` in front of them, giving `(d,) + J + W + (N,)`, which the same paragraph states. (Optional
  wording: "the `len_J` leading axes of each mode's array".)
- **`pack_data` needs no `len_J`.** Every array of the type of `z` (the data, the residual, a probe output) has the
  full jet sizes `F` in every mode and differs between modes only in its last axis, `Ni`. So it pads the last axis
  only, as today. The same holds for the residuals the uniform twins pack, which pass nothing too.

**2. O-44 (b) with O-46. Resolved.** Both points now state that they interact. The three ways out (b1 to b3) are
stated with their costs, and the two points are recommended for one answer.
- On the brief's question: a sentinel default fits the frozen, value-hashed oracle and has precedent in the library.
  Today's frozen kinds in `backend/fitting.py` (lines 591 and 629) rewrite a field in `__post_init__` with
  `object.__setattr__`.
- Resolved there, before anything hashes it, the sentinel never reaches `ValueHashedFields`'s key. So
  `LeastSquaresOracle('entries')` and `LeastSquaresOracle('entries', order=())` would be one cache key, and
  `dc.replace` would carry the resolved value.
- The stated cost ("a signature that does not show the rule") is the real one. Optionally, b1 could cite this
  precedent; not needed.

**3. O-44 (c)'s scope. Resolved.** It names the functions it covers, the size of the change (an upper bound), and
the slices beyond their present text, section 9's row included.

**4. The account's wording. Fixed.**

**O-37** now asks only about the top-level export, and cites the decision that settles the `t3toolbox.fitting`
reach. Section 6's jet matrix carries the parenthesis I asked for on the `(3, 2)` and `(2, 3)` modes.

## A last read for consistency

- The changed passages agree with one another: P11, 2.4, 7.2, 5.2, N41, and the account's sections 1, 3, 6 and 8.
- No text assumes an answer to O-44 or O-46. P14's test is written for either answer, and the sketches use the
  `order=()` default that options (a) and (b) share, as 11.2 says.
- The open points (O-36, O-37, O-38, O-40 to O-47) and the proposed names (N32, N33, N35 to N41, with N31 and N34
  recorded as decided) are all and only the maintainer's.

## Not verified

- Nothing of the suite was run.
- The "about 300 calls" and "124 lines" of O-44 (c) were not recounted. They are stated as an upper bound and only
  size an option.
