# P11: Engine G, the uniform layout and sharding over `W` — handover

Slice P11 of `PART1_PLAN.md`, the last of the additive phase, built to `P11_DESIGN_NOTE.md`, `P11_DESIGN_REVIEW.md`
and the coordinator's rulings at the end of the plan's P11. Written by the implementer, 2026-11-20, against
`graded-messages` at `fcb20bcd` (P10 committed). Nothing is committed.

## 1. What changed

**`t3toolbox/backend/sampling_variables.py`.**
- **The structures** (section 2.2): the layout first (`is_ndarray(frame[0])`), then the count: ragged 2 or 4,
  uniform 4 (`(B, G, shape, masks)`) or 6 (`(D, U, L, R, shape, masks)`); any other count raises, naming the layout.
- **The mask families** (`_families`): `down`, `up`, `left`, `right` from the frame's masks (on a point, `up` is the
  Tucker mask and `right` the TT one), `N` the shape mask from the static `shape`; `require_concrete_masks` on entry
  (a traced mask raises).
- **One table of entry masks, `_UNIFORM_MASKS`**: per input, one `(family, part)` per core axis. The stores with their
  own; `V`, `H`, `Vc` with the frame's variation masks (`left` without the last bond, `right` without the first);
  `D_dot`, `L_dot` with the frame's `D` and `L` masks (never the point's); **the shape mask on `N` for every
  Tucker-shaped input** (`D`, `D_dot`, `V`, `Vc`: Finding A); **the sample's `N`** (its jet axes untouched, O-45); probe's
  `y` on `N` (apply's `y` has no `N`). `_masked(x, spec, masks)` applies a spec, the masks right-aligned against the
  stacks so `W`, `K` before `C` broadcast; `_DIRECTION_MASKS` (P8) stays as the record of which object each direction
  is masked by.
- **Held variables masked on entry** by their table (`_mask_variable`, `_SIZE_MASKS` by the size column: bond
  variables over the full `(d + 1)` bond mask, sites and modes over theirs, blocks over their core's axes); Finding B.
- **The start values** carry ones on the real slots of their end bond only (`_STORE_FAMILY`: the store a sweep goes
  through names its bond family), so every variable has exactly zero padding.
- **Exit:** every output masked by its table (`_mask_variable`): the variables, the blocks (`V`/`VD` with `N` too),
  `g`, `JtJp`, `Kp`; the `'one'` variables and the sums as they are. **This masks every variable on exit, not only the
  core-shaped outputs** (section 3, item 1).
- **The sample is packed** by the engine on a uniform call: `pack_if_ragged(sample, N, len_J=len(order))`; probe's
  `y` by `pack_if_ragged(y, N)`. A packed sample is taken as given.
- **The entries kind's index has one row per mode** (round 1, Finding 1): on both layouts, an index whose leading axis
  is not `d` raises a `ValueError` naming `d` (a uniform leading 1 broadcast to `T[i, i, i]`; a ragged leading 4 was
  silently cut since P9b). It is the sample's length, outside the removable `_check_stacks`.
- **The entries kind's index** (`_uniform_index`, Finding C): a numpy index outside `[-N_i, N_i)` raises an
  `IndexError` naming the mode and its size; a negative index counts from `N_i` (on the padded `N` it would read a
  padding column); a jax index is wrapped the same way but not checked (out of range it reads a zero column).
- **`get_backend(uniform, use_jax)`**: `xmap`/`xscan` over the leading `d` (`jax.lax.map`/`lax.scan` with jax); the
  terms with `over_orders='scan'` on uniform.
- **`_check_stacks`'s uniform branch** (removable with it): every mask's `C`; the packed sample's `d` and `N` equal
  `D`'s. The ragged checks (`C` prefix on every core, `W` on every mode) run on the supercores' per-site arrays as they
  are.
- **The held-shape table** on uniform: one array per variable, the index leading (`_expected`).
- **The `NotImplementedError` for the uniform layout is gone.**
- **Docstrings:** a section "The uniform layout" (structures, masks, packing, the index, layered masks, the return in
  the uniform layout); the parameters `frame`, `sample`, `held`.

**`t3toolbox/backend/sampling_steps.py`.** The whole steps infer the layout from their operands (ruling 3; no
`layout` static): `r_step` (probe) subtracts a packed `y` with `C` inserted before `N` when `z` is an array;
`w_tilde_step` stacks `jet_to_dual_jet`'s tuple when `r` was packed; `g_step` returns the pair as given when the
blocks are supercores; `w_one_hot_step` packs to the padded `N` when `D` is a supercore. The site and sweep steps are
unchanged.

**`t3toolbox/backend/ut3_operations.py`.** `pack_vectors(..., len_J=0)` and `pack_if_ragged(..., len_J=0)` (N41):
the `len_J` leading axes brought to their largest size with zero coefficients; every other axis but the last must
agree, and a mismatch raises a `ValueError` naming the vector (it was an error from inside `pad`/`stack`). With
`len_J=0` the function is what it was. A doctest example (outputs from a real run).

**Tests.**
- **`tests/test_sampling_uniform.py`** (new, 15 tests):
  - **the covering design** (`TestEquivalence`): 81 uniform calls, the nine configurations (no stack, `C`, `K`,
    `K + C`, forced padding, forced padding with `C`, `C = (2, 3)`, `K = (2, 3)`, ranks varying across `C`) with the four
    jet cases (`()`, a line at `(2,)`, `(1, 1)`, mixed input jet sizes at `(1, 2)`) for probe and apply, entries on every
    configuration, the call type alternating; **garbage (1e3) in every masked slot** of every input (the cores, `V`,
    `H`, `Vc`, `D_dot`, `L_dot`, the `N` padding of the sample and of `y`); every name compared with the ragged engine
    per element `(k, c)` and per site, the real block read off the RAGGED result's shape and **the rest asserted exactly
    zero** (non-circular); and the maximal case per kind (forced padding, ranks varying across `C = (2, 3)`, `K = (2, 3)`,
    `W = (2, 2)`, `(1, 1)`), with `sum_over_samples=False` on probe and entries;
  - **the packed jet zeros are data**: a nonzero written into a slot packing filled gives the ragged call with that
    slot added to the input jet (and differs from the call without it); a ragged sample of modes of different jet
    sizes is packed by the engine itself, the same as given packed; `pack_vectors(len_J)` (zero coefficients,
    unpacking by the ragged sizes, today's result at `len_J=0`, a packed array returned as it is, a `W` mismatch
    raising at `len_J` 0 and 1);
  - **held variables with garbage padding** (`mu`, `r`, `z`) give the clean `f`, `g`, `z` (Finding B);
  - **the index**: `N_i` and `-N_i - 1` raise on uniform as on ragged; a leading 1, 2 or 4 raises on both layouts
    (round 1); entries' one-hot `'w'` is one packed array (round 1) of the padded length, real columns as ragged, zero
    padding;
  - **`Kp` at TT slack** with two permanent negative controls (section 3, item 2);
  - **the stack check's uniform branch**, each family alone: a wrong `d`, a wrong `N`, a size-1 `C` on one supercore, a
    mask of another `C`.
- **`tests/test_dispatch.py`**: the engine jitted on uniform data (masks closed over), the numbers numpy's; **the
  program the same size at `(2,)` and `(6,)`** (the lines of its jaxpr; the compiled HLO text also depends on what the
  process compiled before); **an eager uniform call twice compiles nothing the second time** (backend-compile events
  through `jax.monitoring`; skipped if a jax records none).
- **`tests/test_contractions_sharding.py`**: `TestTheEngineShardsOverW`, `g` and `Kp` under `shard_map` with `W`
  sharded, in four programs (uniform at `()`, `(3,)`, `(1, 1)`; ragged at `(3,)`): no all-gather, the result equal to
  the unsharded one to `1e-14`.
- **`tests/test_sampling_graph.py`**: a uniform frame of three or five elements raises (the old `NotImplementedError`
  check).

**Verified before the tests asserted it:** the covering design and the maximal cases were run as the test bodies
first: 2,606 comparisons (every name, element, site), all within `1e-12` with exact zero padding; the jitted program
sizes (8,659 HLO lines at both `(2,)` and `(6,)` in a fresh process) and the numbers (7.8e-16); an eager second call
with zero compile events.

**Single-process runs:** the sampling tests (uniform, stacks, entries, apply, graph, variables, corewise), held,
`test_uniform_probing.py`, jax-absent, with the three modules' doctests: 148 passed (15,754 subtests) current, 148
passed on the floor; the sharding class on the floor (jax 0.4.30): passed; the two dispatch tests on the floor:
passed. (`test_a_ragged_sample_is_packed_by_the_engine` was added after the mutation pass, below.)

**`subTest` labels:** builtins (the guard ran).

**Mutation pass** (scratch copies, deleted), on `test_sampling_uniform.py` and the uniform jit test. 23 mutants,
all killed in the end. **The mask families one by one** (the reviewer's negative controls), each removed alone and
together with the exit mask:

| Family removed | Alone | With the exit mask |
| --- | --- | --- |
| the stores' (`L`) | killed (2,252 subtests) | killed |
| `V`, `H`, `Vc` (incl. `N`) | killed (108: `Vc`'s `N` columns reach `z`, so `r` and `f`) | killed |
| `D_dot` (incl. `N`) | killed (18) | killed |
| probe's `y` on `N` | killed (72: `f`) | killed |
| the sample's `N` | killed only by the permanent control (the exit mask covers it) | killed (360) |
| the start values' | killed only by the permanent control (the exit mask covers it) | killed (626) |
| the held variables' | killed (`f` with a garbage held `r`) | killed |
| the exit mask alone | killed only by the permanent control (the entry masks cover it) | |

So the stores', the directions', `y`'s and the held variables' masks are load-bearing alone; the sample's, the
starts' and the exit masks are layered, each covering the others, and the permanent controls (`TestKpAtTTSlack`)
remove one and the exit mask together.

**The others:** the index check removed; negative indices not wrapped; the loop over the orders on uniform (killed by
the program-size test); the uniform `d`/`N` check removed; `r` on uniform without its `C` axes; the start mask taken at
the wrong end; and **the jet axes not packed** (`pack_if_ragged` without `len_J`), which first **survived**: every
equivalence case passes the sample already packed. `test_a_ragged_sample_is_packed_by_the_engine` (modes of
different jet sizes given ragged) now kills it.

## 2. The gate

The whole gate at `-n 2` (`--detach`, then `run_gate.sh wait`), with nothing else running. The environment was python
3.11.15, numpy 2.4.6 and jax 0.10.2. The summary, copied:

    step                result counts                                                      seconds   peak_MB  minavl_MB
    suite               PASS   1799 passed, 5 skipped, 1 xfailed, 1 warning, 89969 subtests passed   1936.0      8475       3686
    doctest-modules     PASS   315 passed                                                     47.6      1722      10282
    doctest-quickstart  PASS   1 file(s), 0 with failures, 0 failed example(s)                 2.7       283      11706
    doctest-pages       PASS   48 file(s), 0 with failures, 0 failed example(s)               14.1       370      11634
    examples            PASS   14 examples, 0 failed                                           391       638      11369
    sphinx              PASS   0 warning/error line(s)                                      1018.8       630      11236

**No failure.** Against P10 (1,782), 17 more tests: `test_sampling_uniform.py` (14), the two dispatch tests, the
sharding class; one more module doctest (`pack_vectors`).

**Memory:** the lowest MemAvailable is **3.69 GB** (3.81 GB at P10); the peak RSS summed over the workers 8.5 GB.

**Time:** the suite took 1,936 s against 1,870 s at P10 (+66 s at `-n 2`): the uniform covering design (about 33 s
single-process for the 81 calls and the maximal cases), the sharding class (about 15 s of compilation) and the two
dispatch tests (about 8 s).

No process, lock or file of the gate is left in the repository.

**Status: P11 is COMPLETE** (built, verified, mutation pass done, full gate green), nothing committed. This closes the
additive phase. **Round 1** (`P11_RESPONSE_r1.md`): the index's leading `d` checked on both layouts, the one-hot `'w'`'s
packed form asserted, the docstring's jit and `'w'` sentences, the dispatch test's listener; the affected files and the
module doctests re-run single-process on both environments (67 passed; the three dispatch tests passed); no gate, as
ruled.

## 3. Departures and choices

1. **Every output is masked on exit, not only the core-shaped ones.** It is one multiply per output, it makes "exact
   zero padding" hold for every variable whatever was held or started, and the masks then are two full layers (entry
   and exit).
2. **The masks are layered in part.** The stores', the directions', `y`'s and the held variables' masks are each
   load-bearing alone (the mutation pass); the sample's mask on `N`, the start values' and the exit masks cover one
   another. So the permanent negative controls remove one entry family AND the exit mask: without the directions'
   masks the frame's padding of `mu_dot` is not zero; without the sample's mask on `N`, that of `Kp`. (The design note
   called the exit mask the only one needed for zero padding; with every output masked on exit, the entry masks give
   it too.)
3. **TT slack is modelled by garbage**, which contains every value a point's superset mask could leave in the frame's
   padding.
4. **A negative entries index** counts from `N_i` on uniform (not from the padded `N`), for numpy and jax indices alike.
5. **The jit program-size test counts the jaxpr's lines**, not the compiled HLO text, which differed by 12 lines when
   another test had compiled first.
6. **For P12:** the twins mirror the sample's form (the engine returns the uniform layout); a backend user who jits the
   engine on raw `.data` must keep the masks static (closed over, or the frame object), as the docstring says.
