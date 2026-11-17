# Response to review r3 of `PART1_PLAN.md`

Writer's response, 2026-11-16. The plan is now draft r4. All five findings (28 to 32) are accepted, each after
checking the code.

| # | Verdict | Where it is resolved in r4 |
| --- | --- | --- |
| 28 | Accepted. Verified: `T3Tangent.probe_transpose`, `apply_transpose`, `entries_transpose` and the `UT3Tangent` twins are `@staticmethod`s taking `frame` (`manifold.py:751`, `uniform_manifold.py:730` and neighbours). No row changes their kind. | Section 2.5 now states the rule: today's kind, positional order and names kept, `pp=None, order=0` added (6.6). The renames are tied to N19 and N25, and the default to the 3.7 note. A table of today's forms is followed by three house-style sketches, including the static `probe_transpose(residual, ww, frame, pp=None, order=0, sum_over_samples=True)`. No new open point is needed: keeping today's form decides nothing. |
| 29 | Accepted. Verified the five examples (`import ... sampling_derivatives as pd`, then `pd.dense_*_derivatives`). | P13 moves their dense ground truth, so P13 now needs O-27 (new). Section 10's list is rebuilt from an alias-aware search and now names `examples/`. The folder's scripts are O-26 (new; P19 needs it). |
| 30 | Accepted. | N16 now says what is kept (kind, positional order, names), what is renamed and by which row or name (N19; N3 and N25 with the 3.7 note), what is added (6.6), and which positional calls move: only `tests/test_uniform_probing.py:832-836` and `:906-910`, listed in P12. No frontend caller passes the sum flag positionally. Section 2.4 says the same. |
| 31 | Accepted. Verified `backend/regularization.py:99` (definition) and `backend/optimizers.py:44` (import). | Section 2.4's removal list and P16 part 2: the definition goes from `regularization.py`, the import from `optimizers.py`; the `fitting_internals.md` mention is P18's. |
| 32 | Accepted. | A sentence above the table of 5.2: section 3 must be settled before P1 (small item 6). |

## What the comparison of the section 2.5 sketches turned up

I compared every sampling method of the four classes with today's definitions, read with `ast`. For each I checked:
- the decorator;
- the positional arguments, their order and their names;
- the defaults;
- the place of `pp` and `order` in today's `*_derivatives` forms.

Beyond the static transposes of finding 28:
- `TuckerTensorTrain.apply` takes `vecs`, not `ww`. Now stated.
- `entries` takes `index`, as the sketches already had.
- The corewise transposes are instance methods `(self, ztildes|c, ww|index, sum_over_probes=False)`. They now have their own sketch, in today's form.
- Today's derivative transposes put `pp` before `frame`: `(ztildes, ww, pp, frame, order, ...)`. The merged form keeps the plain method's order and appends `pp` and `order`, which is what N16 says. These methods are removed, so no caller of the old order survives.
- The frontend optimizers: today `regularizer` sits after `use_jit` (`mc_sgd`, `adam`) or after `chunk_size` (`gradient_descent`, `newton_cg`). The sketch moves it right after the positional arguments, as 8.3 writes `newton_cg(geom, oracle, batch, x0, regularizer=None, ...)`; every other keyword keeps today's order. Now stated in sections 2.4 and 2.5, with the row cited.
- The renaming of `batch` to `batch_size` is cited to the note to 8.1 to 8.4.

No difference remains that no row or names-table entry covers.

## What the alias-aware search turned up

**Why r3 missed the examples.** The r3 name search did find them: its output listed the five examples with counts of 1 to 5. I left them out when I wrote section 10's list. It was not a blind spot of the search.

**What r4 searched.** The search maps every import alias of the modules P19 deletes from, in code and in every module and page doctest:
- `sampling_derivatives`, whole;
- the removed names of `probing`, `apply`, `entries`, `ut3_sampling` and `utv_sampling`;
- the old kinds of `backend/fitting.py`, and those of `uniform_fitting.py` apart from `pack_sample`, `pack_data` and `uniform_minimal`.

It then lists every attribute of each alias, and every `from ... import` of a removed name. A text grep adds string references such as `mock.patch.object(sd, 'estimate_chunk_size')`. The roots were `t3toolbox/`, `tests/`, `docs/`, `examples/`, `dev/graded_messages/scripts/` and `CLAUDE.md`.

**What it found beyond r3's list:**
- the five examples, with their dense ground truth;
- the lazy import in `t3toolbox/optimizers.py` (`_bfit_pd`);
- the doctest of `backend/optimizers.py` that calls `sd.t3_probe_derivatives`;
- the `mock.patch.object(sd, ...)` strings in `tests/test_optimizers_frontend.py`;
- old-kind uses in `tests/backend/test_optimizer_display.py`, `test_padded_start.py` and `test_zero_reference.py`. These are all files P16 already rewrites, now named in section 10.

`CLAUDE.md` has no code reference.

**`dev/graded_messages/scripts/`.** Fourteen scripts call old kernels or old kinds. They are not part of any gate, since CI runs only `tests/`, the doctests and `examples/`. Proposed: they stay as records run against `v2026.5.0`, and the folder README says so in P19. This is O-26.

**The examples' dense ground truth afterwards.** No row says what computes the jets of a dense tensor:
- 6.3 removes every `*_derivatives` function and keeps `dense_probe` as it is, at order 0;
- 9.7's reference module takes cores.

Proposed: the reference module's `z` also takes a dense tensor. This is O-27, needed before P13.

## Open points

- **Added:**
  - O-26: the folder's scripts; P19 needs it.
  - O-27: dense ground truth for jets; P13 needs it.
- **Changed:** none in substance. The "Needs" entries of P13 and P19 now include O-27 and O-26.
