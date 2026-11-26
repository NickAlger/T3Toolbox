# P4c: the reference module in several variables — handover

Slice P4c of `PART1_PLAN.md` (section 5.3; section 2.4, "One such function per name"; section 2.3, the table of
declared jet sizes). Written by the implementer, 2026-11-19, against `graded-messages` at `d26f1d4c` (P4b committed).
Nothing is committed.

## 1. What changed

**`t3toolbox/backend/sampling_definitions.py`.**

*The signature of all 37 functions.* `(kind, frame, sample, i=None, order=(), *, variations, core_directions, y,
weight, w_tilde)`:
- `order` is the jet shape, a tuple, defaulting to `()`;
- `sample` is the input jets `ww` (`J_in_i + W + (Ni,)`), or `index` for entries;
- there is no `pp`.

*`_read_jet_shape(order)`* is the module's own reader of the jet shape (no import of `contractions` or `jets`, by its
independence test), with the rule of P4a:
- a tuple only;
- entries Python ints or numpy integers (`(int, np.integer)` and not `bool`; `numpy.bool_` is not an `np.integer`),
  nonnegative;
- returned as Python ints.

It uses no `numbers` or `itertools` import, so the independence test's list of allowed outside modules is unchanged.

*`_setting(kind, frame, sample, order)`*:
- reads the jet shape;
- raises for an entries call with jets (`ValueError`, saying to use apply at an input jet whose slot 0 is the one-hot
  vectors);
- clips each input jet to the box `F`;
- records the leaves' jet sizes (the clipped input sizes);
- builds the grid: `M_j = d (a_j - 1) + n_j + 1` roots of unity along axis `j`, with `a_j` the largest clipped input
  size along that axis;
- evaluates the clipped input jets on it (`_on_grid`).

`_Setting` now holds `order`, `box`, `ww` (as given), `leaf_sizes`, `nodes` (one host array per axis), `inputs` (on the
grid) and `M`. `pp` is gone.

*The rest of the machinery:*
- `_on_grid(nodes, jet)` evaluates a jet on the grid: the sum over its slots of `jet[k] s^k`, with the powers built on
  the host and cast with `_complex`. It serves the inputs and the seed (`_at_nodes`).
- `_taylor(S, values, sizes=None)` takes the discrete Fourier transform along every grid axis (`fftn`), divided by
  the grid's size, then the real part on the box `sizes` (default `F`). The leaves (`xi`, `xi_dot`, `xi_U_dot`) pass
  their clipped input sizes. `affine=` is gone.
- `_turn` flips every jet axis.
- `w` returns the input jet as given.
- `r_block_sumsq` sums over everything after the `m` jet axes.
- `_dense_sample` passes the input jets and the jet shape to the dense functions; `_z` of entries goes through
  `dense_entries`.
- `_along` is deleted. `_Kp`'s dispatch no longer reads `pp`.

*The `_einsum` fix of P4b stays.* No new einsum of many operands was added: the grid axes ride the `...` of the
existing ones.

*Docstrings.* The module docstring is rewritten for jets in several variables:
- the inputs are input jets;
- the declared jet sizes;
- the grid and why `M_j` is exact (and tight for apply);
- the turn as a flip of every axis;
- the arguments;
- the entries kind has no jets.

The shape comments say `F` for the full jet sizes, `min(J_in_i,F)` for the leaves and `J_in_i` for `w`. The doctests
of `w`, `w_tilde`, `mu` (a new example in two variables), `z` (with two gotchas: an integer `order`, and entries with
jets) and `r_block_sumsq` use input jets. All 37 doctests pass, from real runs.

**Verified by script before the tests** (`scratchpad/implementer/check_defs.py`):
- *Against an independent route.* The no-jet call at the clipped input jets evaluated on a real grid, interpolated by
  a tensor-product Vandermonde solve. All 24 jetted names, probe and apply, `d = 3` and `d = 2`, at five jet cases: a
  line at `(1,)` and `(3,)`, mixed sizes at `(2,)`, a plane at `(1, 2)`, and modes `(3, 2)`, `(1, 1)`, `(2, 3)` at
  `(1, 2)`. Worst relative error 5e-14.
- *The grid* (point 2 of the brief):
  - one node more along each axis changes nothing anywhere (≤ 6e-16);
  - one node fewer changes the apply kind at full-size inputs (2.5e-2, in `nu_tilde`, the seed times all `d`
    inputs);
  - one node fewer does not change the probe kind (whose largest degree is `max(d (a - 1), (d - 1)(a - 1) + n)`,
    below the bound), nor the mixed sizes, whose largest degree is below the bound too.

  So `M_j` is the least grid that serves every kind, tight for apply; for probe and for mixed sizes it has a node or
  more to spare. (Not changed: one grid rule for every kind and variable keeps the module simple.)
- *The conversions to the old kernels* (point 5), probe, a line, orders 1 to 3:
  - `sampling_derivatives.t3_probe_derivatives` is `t!` times the reference's `z` (1.4e-14);
  - `tv_probe_derivatives_transpose`, given the FORWARD residual `weight[t]**2 r[t] / t!`, is the reference's `g`
    (3e-16), whose seed is the same residual weighted and turned;
  - the residual turned, or times `t!`, does not match (errors of order 1).

## 2. The tests

**`tests/test_sampling_definitions.py`.** Everything above the independence machinery is rewritten; 68 tests where
there were 65.

*The jet matrix.* `JetCase` and `JET_CASES` hold it:
- no jets;
- a line at `(1,)` and at `(3,)`;
- two directions at `(1, 2)` (`affine_jet`);
- modes `(3, 2)`, `(1, 1)`, `(2, 3)` at `(1, 2)`;
- a curve of three slots at `(2,)`;
- modes `(1,)`, `(4,)`, `(2,)` at `(2,)`.

Mixed sizes are cycled over the modes. Every input jet's slot 0 is the point `ww`. The entries kind runs with no jets.

*`Jets`* does the algebra on the box with explicit multi-index loops: `lift` (cut and pad), `tprod`, `top` (the
corner), `flip`, the terms of section 5, `V_top`, `H_top`, `pair`, `top_pair`, `times_seed`, `ones`, `weighted`.

*The tests' own dense route* (`sample_dense_jet`, `interpolate`, `degree_bound`, `evaluate`): `np.tensordot`, sample by
sample, at the input jets evaluated on a real grid, and a tensor-product Vandermonde solve. It shares nothing with
the reference's roots of unity.

| Class | Matrix | What |
| --- | --- | --- |
| `TestOutputsAgainstDense` (5) | the whole jet matrix × 5 problems × 3 kinds (the Jacobian tests: 2 problems × 4 jet cases) | `z`, `Jp` against the dense route; `r`, the seed (a flip of every axis), `f`, `r_block_sumsq` (the shape of the weight, exact), `Jp_norm_sq`; `g`, `JtJp` against the dense Jacobian transposed; a seed of one's own, turned |
| `TestRecurrences` (5) | the whole matrix | every recurrence of section 5, with truncated products on the box |
| `TestReadouts` (3) | the whole matrix | section 6 |
| `TestPairings` (3) | the whole matrix | section 8 at every cut edge; the multipliers |
| `TestApplyAndEntries` (4) | the whole matrix / 3 jet cases | zero or the seed times the untilded variable; the blocks; the two pairings of the probe kind only; tilde-dot as the derivative of tilde (`xi_tilde` keeps the full sizes `F`, checked on the unlifted array) |
| `TestJetsTwoWays` (3) | every jet case on `small` (d = 3); on `long` (d = 4) the line at `(3,)` and the mixed modes | **the jets against the Taylor expansion on two real grids**, with the declared jet sizes exact; slot `(0, ..., 0)` is the call with no jets; a second variable of order 0 (inputs `(4, 1)`, jet shape `(3, 0)`) is one variable, for every name |
| `TestKp` (5) | no jets, `(3,)`, mixed modes | jax against finite differences; the route; the derivative of the corewise `g`; the second derivative of the dense objective; symmetric and linear |
| `TestAgainstTheOldKernels` (1) | probe, a line, orders 1 to 3, three problems | **the bridge**: `z` against `t3_probe_derivatives / t!`; `g` against `tv_probe_derivatives_transpose` of the forward residual `weight**2 r / t!` |
| `TestCorewiseFrame` (2) | no jets, `(3,)`, mixed modes | section 9 |
| `TestStructure` (14) | | the names; the signature (`order=()`); the declared jet sizes of every name at every jet case (the leaves clipped, `w` as given, the block sums the shape of the weight); one index; the errors, now with entries-with-jets and five wrong jet shapes; **the module's reader agrees with `contractions._read_order` on 15 inputs**; sixteen sites; `w` returns the input jet exactly (a mode larger than the box included); probe and apply share the plain and dot variables; **entries equals apply at `index_to_one_hot`**; the independence test, unchanged, passes |
| `TestJax` (1) | probe at `(1,)`, apply with no jets and at `(1, 2)`, entries | jax in, jax out, every name |
| `TestDenseFunctions`, `TestInputJets` | | P4b's, unchanged |

**`tests/test_jax_absent.py`:** its script calls every name at a hand-built line `(2,)` (probe, apply) and with no
jets (entries).

**Mutations** (a scratch copy of `t3toolbox/` and `tests/`, each mutant of `sampling_definitions.py` run against a
fast subset of the test file: the dense-route `z` and the formulas, the recurrences, two readouts, the structure tests,
apply-and-entries, the bridge; the copy deleted). **16 of 17 caught.** Caught:
- the grid one node fewer; the grid from the first mode's sizes only; the inputs not clipped;
- the leaves given the full box; the turn on the first jet axis only, and no turn;
- the powers of the nodes on the wrong axis; the transform along one axis only; the block sums over one jet axis;
- `w` clipped; entries with jets accepted;
- the reader accepting a `bool` or a list, or not normalizing;
- the dense functions given no jet shape; the weight not squared.

**Not caught, and equivalent:** the grid sized from the unclipped input sizes. It is a larger grid, still exact, so the
results are the same to rounding; only the cost grows.

## 3. The gate

The whole gate on the final tree, nothing else running:
`PATH=/home/nick/miniconda3/envs/t3toolbox/bin:$PATH dev/graded_messages/implementation/run_gate.sh --detach -n 3 -l <scratchpad>/implementer/gate_P4c`,
then `wait`. The summary, copied:

    step                result counts                                                      seconds   peak_MB  minavl_MB
    suite               PASS   1660 passed, 5 skipped, 1 xfailed, 1 warning, 69649 subtests passed   1534.0      9926       2367
    doctest-modules     PASS   312 passed                                                     54.7      1556      10457
    doctest-quickstart  PASS   1 file(s), 0 with failures, 0 failed example(s)                 2.6       276      11737
    doctest-pages       PASS   48 file(s), 0 with failures, 0 failed example(s)               14.7       373      11653
    examples            PASS   14 examples, 0 failed                                           385       640      11388
    sphinx              PASS   0 warning/error line(s)                                      1010.7       637      11259

**No failure.** Against P4b's gate:
- The suite has 3 more tests (1,657 to 1,660: the reader's agreement, the bridge to the old kernels, and the second
  variable of order 0). The subtests go from 68,234 to 69,649.
- The same 5 skips, the same 1 warning, the same 1 xfailed.
- The module doctests are the same 312: `mu`'s and `z`'s gain examples, and every function keeps one doctest.
- The suite's peak memory is 9.9 GB, with 2.4 GB available at its lowest.

**The test file's time, single process** (point 3 of the brief), before (P4b) and after:

| Env | Before | After | Ratio |
| --- | --- | --- | --- |
| `t3toolbox` (py3.11, numpy 2.4.6, jax 0.10.2) | 132 s | 217 s | 1.6 |
| `tttt`, the floor (py3.9, numpy 1.22.4, jax 0.4.30) | 107 s | 224 s | 2.1 |

The floor command: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD timeout 1500 /home/nick/miniconda3/envs/tttt/bin/python -B -m pytest tests/test_sampling_definitions.py -p no:cacheprovider -q`
(68 passed).

The largest tests:

| Test | Current env | Floor |
| --- | --- | --- |
| `TestJetsTwoWays.test_every_variable` | 67 s | 101 s |
| `TestKp.test_jax_against_finite_differences` | 41 s | 17 s |
| `TestJax` | 38 s | 17 s |
| `TestReadouts.test_the_blocks_of_Kp` | 12 s | 16 s |

Before these numbers, the first version of the matrix took 245 s on the current env. It was cut as section 4,
choice 5 says. Nothing further was cut: the floor's ratio is about double, not much more. If it should be lower,
`TestJetsTwoWays` on `long` (the mixed modes give a grid of 6 x 11 nodes, twice, for 24 names) is the place.

The sphinx build was deleted after reading; no process, lock or file of the gate is left in the repository.

## 4. Departures; choices

**Departures:** none from what the plan builds. One thing the plan's P4c says, nuanced, is in section 1: `M_j` is
exact and tight for the apply kind at full-size inputs; for the probe kind one node fewer would still be exact.

**Choices:**
1. The module's reader is written with `(int, np.integer)`, not `numbers.Integral`. This keeps the independence test
   unchanged (it allows only `math`, `typing`, `dataclasses`, `numpy` and `jax` outside the library), and
   `np.ndindex` replaces `itertools.product` for the same reason. A test checks that the reader agrees with the
   library's.
2. One grid for every variable of a call: `M_j` from the largest clipped input size, the jet shape and `d`, enough for
   the variable of highest degree (the apply kind's seed times every input). Smaller grids per variable or per kind
   were not taken: they save time in a slow reference, at the cost of a rule per variable.
3. Inputs are clipped to the box before evaluation (so they set no larger grid); `w` still returns them as given (the
   declared jet sizes of `w` are the input's).
4. The doctests build input jets by hand (`np.stack([w, p])`), as the module does, rather than through
   `jets.affine_jet`; the tests use `affine_jet` (the tests are not under the independence rule).
5. The test matrix (point 3 of the brief). The structural and definitional tests run the whole jet matrix. The
   costliest tests run a subset where a case adds no distinct coverage:
   - `TestJetsTwoWays` on `long` (d = 4) takes the line at `(3,)` and the mixed modes only;
   - `TestKp`'s jax comparison takes no jets, `(3,)` and the mixed modes, dropping `(1,)` (as `(3,)`) and the plane
     (the mixed modes are the richer `m = 2` case).

## 5. Found outside the slice

1. The time of `tests/test_sampling_definitions.py`: see section 3. `TestJetsTwoWays` (about 67 s), `TestKp`'s jax
   comparison (41 s) and `TestJax` (38 s) are most of it.
