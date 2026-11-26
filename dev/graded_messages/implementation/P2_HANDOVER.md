# P2: handover

Slice P2 of `PART1_PLAN.md`: the reference module `sampling_definitions`, and the dense functions `dense_probe` (in
its new form), `dense_apply` and `dense_entries`. Written 2026-11-17 on `graded-messages` at `3fcfbe7` (P1).
Nothing is committed.

The work was interrupted twice (the machine restarted once). Nothing was lost; the tree was checked after each.

**Brought up to date after review round 1** (`P2_REVIEW_r1.md`; the answers are in `P2_RESPONSE_r1.md`). What
changed in that round: the jets of the reference come from the roots of unity and a Fourier transform; `Kp` takes
its route from the arrays; `dense_entries` indexes at order 0; the dense functions refuse a negative order; the
docstrings of the tilde grades say what holds for which kind; the guard of the module's independence is rebuilt;
five more tests and two more problems. This file describes the tree after that round.

## 1. What changed

| File | Change |
| --- | --- |
| `t3toolbox/backend/sampling_definitions.py` | **new**, 3,129 lines: 37 public functions, one per name of the plan's table (section 2.3) |
| `t3toolbox/backend/probing.py` | `dense_probe(T, ww, pp=None, order=0)` in place of `dense_probe(vectors, T)`; three private helpers, `_dense_contract`, `_dense_jet` and `_check_pp_and_order`; `import itertools`; the eight calls in the doctests of `t3_probe`, `tv_probe` and `dense_probe` |
| `t3toolbox/backend/apply.py` | **new** `dense_apply(T, ww, pp=None, order=0)`, in `__all__` |
| `t3toolbox/backend/entries.py` | **new** `dense_entries(T, index, pp=None, order=0)`, in `__all__` |
| `t3toolbox/manifold.py`, `t3toolbox/uniform_manifold.py` | one doctest call of `dense_probe` each (`T3Tangent.probe`, `UT3Tangent.probe`) |
| `tests/test_manifold.py` | one call of `dense_probe` and the comment above it |
| `tests/test_tucker_tensor_train.py` | one call of `dense_probe` |
| `tests/test_sampling_definitions.py` | **new**, 1,585 lines: 56 tests |
| `tests/test_dispatch.py` | one test, `test_jit_dense_sampling_functions` |
| `tests/test_jax_absent.py` | one test, `test_the_sampling_definitions_and_the_dense_functions_run_without_jax` |

Nothing is deleted. The 12 calls of the library's `dense_probe` are the 12 the plan lists (confirmed by a search
of `t3toolbox/`, `tests/`, `docs/`, `examples/` and `README.md` for the name in every form); the local helpers of
the same name in four test files and three examples are untouched.

### The dense functions

```python
def dense_probe(                                              # backend/probing.py
        T:      NDArray,                                        # shape=C+(N0,...,N(d-1))
        ww:     typ.Sequence[NDArray],                          # len=d, elm_shape=W+(Ni,)
        pp:     typ.Optional[typ.Sequence[NDArray]] = None,     # len=d, elm_shape=W+(Ni,); given iff order >= 1
        order:  int = 0,                                        # highest Taylor order.  T=(order+1,) if order else ()
) -> typ.Tuple[NDArray, ...]:                                   # z. len=d, elm_shape=T+W+C+(Ni,)

def dense_apply(T, ww, pp=None, order=0) -> NDArray:           # backend/apply.py.    z. shape=T+W+C
def dense_entries(T, index, pp=None, order=0) -> NDArray:      # backend/entries.py.  index int, (d,)+W.  z. T+W+C
```

- Taylor coefficients (slot `t` is the coefficient of `s^t`, no `t!`), the order axis leading, by the multilinear
  expansion over which modes take `pp`: exact, and exactly zero above the degree (`d - 1` for probe, `d` for the
  others).
- The code for the jets of a dense tensor exists once: `probing._dense_jet`, which the three call
  (`dense_entries` with the unit vectors `e_index`). The one einsum behind it is `probing._dense_contract`, which
  is today's `dense_probe` body with the open mode optional.
- `dense_entries` at order 0 indexes the tensor, with no arithmetic (an entry comes back as stored, whatever the
  others hold); with jets it goes through the unit vectors.
- Today's stacks are kept: `W` on the vectors, `C` on the tensor, the result `W + C`.
- They raise a `ValueError` on a negative `order`, on `pp` without `order` or `order` without `pp`
  (`probing._check_pp_and_order`), and on vectors of a wrong stack or length.

### The reference module

One function per name, named exactly as the variable, all with one signature:

```python
def mu(
        kind:             str,                               # 'probe' | 'apply' | 'entries'
        frame:            typ.Tuple,                         # (D, U, L, R), each len=d
        sample:           typ.Any,                           # ww | index if order == 0, else (ww, pp) | (index, pp)
        i:                typ.Optional[int] = None,          # the bond, 0 .. d; None = every bond
        order:            int = 0,                           # highest Taylor order.  T=(order+1,) if order else ()
        *,
        variations:       typ.Optional[typ.Tuple] = None,    # (V, H), elm_shapes (nUi,Ni), (rLi,nDi,rR(i+1))
        core_directions:  typ.Optional[typ.Tuple] = None,    # (D_dot, L_dot), the shapes of D and L
        y:                typ.Optional[typ.Any] = None,      # the data, of the type of z
        weight:           typ.Optional[NDArray] = None,      # probe (d,)+T; apply, entries T; None = ones
        w_tilde:          typ.Optional[typ.Any] = None,      # a seed of the type of z, stored turned
) -> typ.Union[NDArray, typ.Tuple[NDArray, ...]]:            # mu[i]. shape=T+W+(rLi,)
```

The 37 names, in the order of `__all__`:

| Group | Names | Index `i` |
| --- | --- | --- |
| Inputs and seed | `w`, `w_tilde` | site; mode (probe only) |
| Plain | `xi`, `mu`, `nu`, `eta`, `nu_L`, `eta_D` | site for `xi`, `eta`; bond for `mu`, `nu` |
| Dot, frame form | `xi_U_dot`, `mu_R_dot`, `nu_L_dot`, `eta_D_dot` | likewise |
| Dot, raw form | `xi_dot`, `mu_dot` | likewise |
| Tilde | `xi_tilde`, `mu_tilde`, `nu_tilde`, `eta_tilde`, `nu_L_tilde`, `eta_D_tilde` | likewise |
| Tilde-dot | `xi_tilde_dot`, `mu_tilde_dot`, `nu_L_tilde_dot`, `eta_D_tilde_dot` | likewise |
| Outputs | `z`, `r`, `Jp` (mode, probe only); `g`, `JtJp`, `Kp` (no index); `V_tilde`, `H_tilde`, `V_tilde_dot`, `H_tilde_dot` (site) | |
| Sums | `f`, `r_block_sumsq`, `Jp_norm_sq` | none |

How each is computed:

- **An edge variable** (the four letters, in every basis and grade): one dense `einsum` over all the cores and
  inputs of its side (`_left_side`, `_right_side`, `_trunk`; no site-by-site loop), summed over the places where
  the direction can be put (`_frame_form`, `_raw_form`) and where the seed can be put (`_seeded`). The network is
  given as three lists, the TT cores, the Tucker cores and the inputs, so "the direction at site `j`" is a change of
  the first two and "the seed at input `k`" a change of the third, and the two are independent (the case of both
  at one site falls out).
- **Jets**: the same contraction at `d + order + 1` values of `s`, the complex roots of unity, and the discrete
  Fourier transform of the values divided by their number (`_taylor`): exact for a polynomial of that degree, and
  with no loss of digits as the order grows (measured against exact arithmetic for `d = 4`: `2e-15` at order 3,
  `1e-14` at orders 16 and 32). The nodes ride as a leading axis of the inputs, so it is one `einsum` per place.
  At order 0 there is one node, `s = 0`, and no transform. The stored seed enters as the polynomial
  `sum_t w_tilde[t] s^t`: that is where "stored turned" is used.
- **`z`**: the dense tensor of `(D, L)` (`_tensor`, one `einsum`) through `dense_probe`, `dense_apply` or
  `dense_entries`. **`Jp`**: the same of the dense tangent tensor, the sum over the places of the direction.
- **`g`, `JtJp`, `V_tilde`, `H_tilde`**: the dense Jacobian transposed. The forward map is applied to a basis of
  directions (the unit cores of one block, stacked as a `C` stack of the dense function) and paired with the seed
  (`_jacobian_transpose`, `_pair`).
- **`Kp`, `V_tilde_dot`, `H_tilde_dot`**: the second derivative of the dense objective `phi(B, G) = <seed, z(B, G)>`
  at `(D, L)` along `(D_dot, L_dot)`. The route is read off the arrays, as everywhere in the library (Claude's
  decision on finding 6 of the review; the plan's sentence said "jax where present"). numpy arrays: central
  finite differences of the gradient (`_Kp_by_finite_differences`): the gradient in one core is `phi` with a basis
  in place of that core (`phi` is linear in each core), and along the direction it is a polynomial of degree at
  most `2d - 1`, so the central stencil on the `2d` points `+-h, ..., +-d h` is exact for it. A jax array among
  them: `jax.jvp` of `jax.grad` (`_Kp_by_jax`), in the precision jax is set to. The two agree to `1e-15` in double
  precision.
- **`r`, `w_tilde`, `f`, `r_block_sumsq`, `Jp_norm_sq`**: their formulas on the outputs.

It imports `backend.common`, and `probing`, `apply`, `entries` for the three dense functions only (a test reads
the module's syntax tree and fails on any other reach into the library, by any of 23 roads it is tried on). It
has no sweep, no `contract`, no `jet_contract`, no `safety`. It is for the ragged layout and real arrays.

For apply and entries the seed is one number per sample at bond `d`. The docstrings say for each tilde and
tilde-dot variable what it then is (zero, or the seed times a plain variable or a raw dot), and which of the four
multiplier pairings of `SAMPLE_VARIABLES.md` section 8 hold: two for every kind, two for the probe kind only.

## 2. The tests

`tests/test_sampling_definitions.py`, 56 tests (4,107 subtests), about 40 s alone with two workers. numpy only,
except `TestJax` and the jax half of `TestKp`. Five problems: `small` (three sites, every size different from its
neighbours, one sample axis), `tails` (end bonds larger than one, two sample axes), `long` (four sites),
`two sites, W = ()` (no sample axis) and `one site`. The stores are random: not orthogonal and not consistent with
each other, since no definition needs that. Kinds probe, apply, entries; orders 0, 1, 3.

| Class | Tests | What it establishes |
| --- | --- | --- |
| `TestOutputsAgainstDense` | 5 | `z` is `t3_to_dense((D, L))` contracted with the inputs; `Jp` is `tv_to_dense(frame, (V, H))` contracted with them. The contraction and its jets are the tests' own (`np.tensordot` one sample at a time, interpolation at nodes that are not the reference's), not the library's dense functions. `g`, `V_tilde`, `H_tilde`, `JtJp` are the dense Jacobian transposed: the Jacobian is built column by column from `tv_to_dense` at unit directions and transposed with the **plain** slot-by-slot inner product against `weight^2 r` in forward order (the reference pairs a turned seed at the top coefficient, so this also pins the turn). `r`, `w_tilde`, `f`, `r_block_sumsq`, `Jp_norm_sq` are their formulas; a given `w_tilde` is used as it is |
| `TestRecurrences` | 5 | Every recurrence of `SAMPLE_VARIABLES.md` section 5 (plain; dot in the frame form; dot in the raw form; tilde; tilde-dot with its six terms), written out in the test with truncated products as explicit loops, holds for the variables the reference contracts whole. For apply and entries: the start of the right-to-left tilde sweeps is the seed times the ones, and `xi_tilde`, `mu_tilde` are exactly zero |
| `TestReadouts` | 3 | Section 6 of the page: `z` and `Jp` (dense tensors) are `z_term(eta_D, D)` and `z_term(eta_D_dot, D) + z_term(eta, V)`, or `mu[d]`, `mu_R_dot[d]` summed for apply and entries; the blocks of `g` (dense Jacobian) are the top coefficients of the `V_term`s and `H_term`s of the tilde variables; the blocks of `Kp` (second derivative) are those of the tilde-dot variables, nine `H_term`s included. This is what ties the tilde edge variables to the dense Jacobian and the tilde-dot ones to the second derivative |
| `TestPairings` | 3 | Section 8, at every cut edge: the direction's network as one side dotted against the other plain, at the `d + 1` bonds and the `d` Tucker edges, all equal to `Jp` of apply; the four multiplier pairings of the probe kind (`mu_tilde` with `nu_L_dot`, `nu_tilde` with `mu_R_dot`, `xi_tilde` with `eta_D_dot`, `eta_tilde` with `xi_U_dot`), each with the direction wholly on one side of the edge; the two that exist for apply and entries |
| `TestApplyAndEntries` | 4 | What the tilde and tilde-dot variables are for apply and entries: `xi_tilde`, `mu_tilde`, `xi_tilde_dot`, `mu_tilde_dot` exactly zero; the others the seed times the plain variable or the raw dot, as polynomials. Their blocks have only the terms with the seed on the right. `mu_tilde` with `nu_L_dot` and `xi_tilde` with `eta_D_dot` are pairings of the probe kind only: for apply and entries they are zero and the multipliers are the seed times `mu` and `xi`. Every tilde-dot variable is the derivative of its tilde variable with the seed held, for every kind. These are the statements the docstrings make |
| `TestJetsTwoWays` | 2 | `w`, `z`, `Jp` and the 22 edge variables at orders 1 and 3 equal the interpolation, at two other sets of nodes, of the same function at order 0 along `ww + s pp`, the tilde grades with the stored seed given as the polynomial `sum_t w_tilde[t] s^t`. The affine variables have two slots. Slot 0 of a plain or dot jet is the order-0 variable |
| `TestKp` | 5 | jax arrays (in double precision, inside a context the test opens) against the finite differences of numpy arrays (`1e-10`); the route is read off the arrays (numpy never reaches the automatic differentiation, a jax array always does); `Kp` is the derivative of the corewise gradient with the seed held (`g` on `(B, G, G, G)`, a fourth-order difference, `1e-8`); `Kp` is the mixed second derivative of `<weight^2 r, z(B, G)>` computed from `t3_to_dense` by the tests' own route (exact, by interpolation in the two step sizes, `1e-9`); symmetric and linear |
| `TestCorewiseFrame` | 2 | On `(B, G, G, G)`: `nu` is `nu_L`, `eta` is `eta_D`, likewise with a tilde; the frame's dots are the raw ones; the two forms of `nu_L_dot` and `eta_D_dot` are one array. On a manifold frame with both directions given, the two shared names are the raw form (O-12) |
| `TestStructure` | 14 | Exact. `__all__` is the 37 names of the plan's table, in its order; every function has the one signature; every function has a doctest about itself; the module calls nothing of the library but the three dense functions, and the guard that says so sees each of 23 ways around it (a name imported from a module, `getattr`, the module renamed, `__dict__`, another module, a scan from `common`, ...); the shapes of section 7 of the page (`T` or `A`, `W`, the size of the edge in the variable's basis; `d + 1` per bond, `d` per site; the blocks), numpy and float64; one index is the entry of the whole, and an index out of range raises; an index where there is none raises; each missing argument raises, and each function runs with only what it reads; an unknown kind, a frame of two stores and 17 sites raise, and 16 sites run; `w` is returned exactly; the plain and dot variables do not depend on the kind; entries is apply at unit vectors |
| `TestJax` | 1 | jax in gives jax out for all 37 names, with numpy's numbers (to float32) |
| `TestDenseFunctions` | 12 | Order 0: `dense_probe` against hand-written einsums, with `W`, with `C`, with neither, and with one mode; `dense_apply` equals each probe contracted with the vector of its mode; `dense_entries` equals indexing (exactly), with a `C` stack, a negative index, a nested list, `W = ()`, and with a non-finite entry beside an indexed one. Jets equal the interpolation of order-0 calls at two sets of nodes, with `W` and `C` stacks, orders 1, 2, 3, 5; slot 0 is the order-0 call exactly and the slots above the degree are exactly zero; the slots are Taylor coefficients, not derivatives; `pp` without `order`, `order` without `pp` and a negative `order` raise; wrong shapes raise, naming the function; the signatures; float32 kept; jax in, jax out |

`tests/test_dispatch.py::TestDispatch::test_jit_dense_sampling_functions` (42 subtests): the three functions under
`jax.jit` at orders 0 and 3, float64 and float32, with a `W` and a `C` stack; jax in gives jax out, eagerly too; the
dtype is kept with x64 on; the numbers are numpy's; and with each array argument in turn a numpy array held outside
the trace (the dispatch is read off all of them).

`tests/test_jax_absent.py::...::test_the_sampling_definitions_and_the_dense_functions_run_without_jax`: in a
subprocess that hides jax, every one of the 37 functions runs for the three kinds at order 2, and the three dense
functions run.

**Doctests.** Every one of the 37 functions and the three dense functions has one, run and pasted. Those of the
reference show the shape and the variable's recurrence or formula from `SAMPLE_VARIABLES.md` against the
definition; `Jp`, `g`, `Kp`, `mu_dot` are shown as derivatives by a central difference.

**Testing the tests.** Twelve mutations of the code, each run against the test file with bytecode caching off,
each caught: the turn a no-op; `L` cores after the direction in the frame form; the seed of apply also on left
sides; `V` allowed at site `i` in `eta_D_dot`; the stencil's sign; the weight not squared; the frame form winning
where both directions are given; the seed allowed at input `i` in the tilde-dot trunk; one interpolation node too
few; the dense jets as derivatives; `pp` and `ww` swapped in a term of the dense jet; `Jp` without the `U`-over-`V`
term at the last site. After round 1, eight more, of the code that round changed, each caught (the list is in
`P2_RESPONSE_r1.md`), and the reviewer's two mutants of the independence guard, which passed before, now fail the
test file. (The reviewer's own 53 mutations: 51 caught in round 1, those two not.)

**A one-off check, not a test** (the bridge to the old kernels is P6's): `z`, `Jp`, `g`, `JtJp` and `Kp` of the
reference against `ProbeDerivativesKind`, `ApplyDerivativesKind` and `EntriesDerivativesKind` through the stated
conversions (`t!` per slot, the weight times `t!`), at orders 0 and 2, on a frame with tails: all within `5e-15`,
`Kp` by both of its routes.

## 3. The gate

**Round 2, the tree as it is now.** The whole gate, detached, two workers, one step after another, with the
bytecode cache removed first: it started at 09:13:40 and every file of the slice was last written before that (the
latest at 09:07:23); nothing of the slice changed afterwards.

    dev/graded_messages/implementation/run_gate.sh --detach -n 2 -l /tmp/gate_P2_r2
    dev/graded_messages/implementation/run_gate.sh wait /tmp/gate_P2_r2

    step                result counts                                                      seconds   peak_MB  minavl_MB
    suite               PASS   1501 passed, 5 skipped, 1 xfailed, 1 warning, 65157 subtests passed    937.7      3637       4256
    doctest-modules     PASS   282 passed                                                     24.5      1208       6529
    doctest-quickstart  PASS   1 file(s), 0 with failures, 0 failed example(s)                 1.3       226       7408
    doctest-pages       PASS   48 file(s), 0 with failures, 0 failed example(s)               10.5       467       7227
    examples            PASS   14 examples, 0 failed                                           188       370       7352
    sphinx              PASS   0 warning/error line(s)                                       188.9       702       6888
    logs: /tmp/gate_P2_r2

- **No failure, so nothing new against the baseline.** The five skips are the slow tier, the same five as in
  `P0_BASELINE.md`; the one expected failure is P1's marker for Part 2 (row 5.5); the one warning is the baseline's
  (Adam on the manifold geometry).
- **The counts.** The suite collects 1,507 tests, against 1,449 at the last commit `3fcfbe7` (counted there with
  `pytest --collect-only` on an export of the commit): 58 more, which are the 56 of
  `tests/test_sampling_definitions.py`, one in `tests/test_dispatch.py` and one in `tests/test_jax_absent.py`. (The
  log's figure for P1, 1,442 passed, is from its full run; P1's last pass on its test file added one test after
  that.) The module doctests are 282 against 243: 39 more, the 37 functions of the reference and the two new dense
  functions. Subtests: 65,157.
- **The slice's tests alone**, before the whole gate, with a clean cache and bytecode off: the test files the slice
  touches, 231 passed, 28,632 subtests; the module doctests of the six touched modules, 74 passed.
- **Round 1's gate** (`/tmp/gate_P2`, on the tree before the review): every step passed; 1,496 passed, 64,796
  subtests. It is superseded by the run above.
- **`/tmp/gate_P2_quick` holds a FAILED quick run** (247 failed) from round 1. It is not a result about the code:
  it ran on stale bytecode left by my own mutation run (section 5, first item). The same command passed once the
  cache was removed.

## 4. Departures from the plan, and the choices it left open

### Departures and ambiguities, for a decision

1. **The common signature has two more arguments than the plan's sketch, and `i` is optional.** The sketch
   (`PART1_PLAN.md` section 2.4) is `mu(kind, frame, sample, i, order=0, *, variations=None,
   core_directions=None, w_tilde=None)`. `r`, `f`, `r_block_sumsq` need the data, and the seed of the residual
   needs the data and the weight, so the signature has `y=None` and `weight=None` as well. `i` has the default
   `None`, which returns the whole variable (a tuple over the index), so that a test of the engine can compare a
   whole variable in one call; a call written as the sketch has it works unchanged. The variables with no index
   (`g`, `JtJp`, `Kp`, the three sums, and `z`, `r`, `Jp`, `w_tilde` of apply and entries) take `i` too, so that
   every function has the same arguments as the plan asks, and raise if it is not `None`.
2. **The tilde edge variables are built from subnetworks, not from the dense Jacobian.** (The reviewer found
   no objection.)  The plan says both: "each
   built as the dense contraction of its subnetwork" and "`g`, `JtJp` and the tilde variables from the dense
   Jacobian" (section 2.4; row 9.3 has the second). I took the first for the edge variables (the column "As a
   subnetwork": the seed in place of one input, summed) and the second for `g`, `JtJp` and the blocks. The tests
   tie the two: the blocks of `g` from the Jacobian equal the readouts of the tilde variables (`TestReadouts`), and
   the multiplier pairings hold (`TestPairings`).
3. **Apply and entries: where the seed goes is not in `SAMPLE_VARIABLES.md`**, whose definitions are written for
   the probe kind (its status line: "Not covered: apply and entries"). I took it from the note to row 4.4 ("the
   seed times the ones vector at the right end for the tilde of apply and entries"), the README's "Things learned"
   ("the adjoint needs only `nu_tilde` and `eta_tilde`. For them `nu_tilde` is the residual times the plain `nu`")
   and `scripts/review/kind_readout_check.py`: the seed is one number per sample and multiplies the ones that close
   bond `d`. So a side that reaches bond `d` (a right side, a trunk) has it as a factor, with jets as a product of
   polynomials in `s`; a side that does not (a left side, a leaf) has no place for it. **The second half is my
   inference:** `xi_tilde`, `mu_tilde`, `xi_tilde_dot`, `mu_tilde_dot` of apply and entries are defined here as
   zero, rather than raising. **Decided by Claude after the review: the zeros stay.** The reviewer confirmed that
   zero is the only value consistent with the readouts of section 6 of the page (the other candidate, the seed
   times `mu`, counts the seed twice in `H_tilde`). Which names the engine accepts for these kinds is P9's to
   settle, with a test that the reference and the engine agree. A consequence, now in the docstrings and the tests:
   two of the four pairings of section 8 are of the probe kind only.
4. **A manifold frame only.** `frame` is the four stores. A corewise point is passed as `(B, G, G, G)`, as
   `SAMPLE_VARIABLES.md` section 9 writes it, under the manifold names; a frame of two stores raises, with a
   message saying so. The four spellings that exist only on a corewise call (`nu_dot`, `eta_dot`, `nu_tilde_dot`,
   `eta_tilde_dot`) have no function: they are not in the plan's table. P8's tests of the corewise call would
   compare the engine's `nu_tilde_dot` with the reference's `nu_L_tilde_dot` on `(B, G, G, G)`. **Decided by Claude
   after the review: kept as it is**; the corewise spellings are to be listed in `SAMPLE_VARIABLES.md` before P8.
5. **`tests/test_jax_absent.py` gained a test in this slice**; the plan lists that file from P3. The brief for this
   slice asks that the module run without jax, and this is the test of it.

### Choices the plan left open

6. **No `K` and no `C` stack in the reference; `W` with any number of axes.** P10 says it compares "against the
   reference applied element by element". `W` costs nothing (an ellipsis). The blocks are summed over `W`.
7. **`nu_L_dot` and `eta_D_dot`**: the raw form where `core_directions` is given, also when `variations` is given
   too (O-12); else the frame's form; a `ValueError` with neither.
8. **The seed**: `w_tilde` where it is given, else made from `y` and `weight`; a `ValueError` with neither. `r`,
   `f` and `r_block_sumsq` always read `y`. `JtJp` makes its own seed from `Jp` and the weight.
9. **Interpolation**: `d + order + 1` nodes (no variable has a higher degree than `d + order`; one node fewer is
   caught by a test), the complex roots of unity, and a discrete Fourier transform. Round 1 changed this from
   Chebyshev nodes and a Vandermonde solve, which lost digits with the order (`2e-8` at order 16); the transform
   loses none, and the change was three small places. The price: the inputs at the nodes are complex, and complex
   cores are not supported. At order 0 no interpolation. `w` is not interpolated: it is `ww` and `pp` stacked,
   exactly.
10. **`Kp` by finite differences is exact for the polynomial** (item on `Kp` in section 1), so numpy arrays get a
    result that is right to rounding and does not depend on whether jax is installed.
11. **What the reference checks**: an unknown kind; a frame that is not four stores; more than 16 sites (the
    einsum letters); an index out of range, or given where there is none; a missing argument. All `ValueError`.
    Nothing numerical, and no shape of the sample, the data or the weight.
12. **Dispatch**: every helper infers numpy or jax from its own arrays, so jax in gives jax out. The reference is
    not written for `jit` and is not tested under it.
13. **The dense functions raise** on a negative `order`, and when `pp` is given at `order = 0` or missing at
    `order >= 1` (the plan's comment is "given iff order >= 1"), and keep today's checks of the vectors' stack and length, now also for `pp`; the
    messages name the calling function, and say "sample stack" where today's said "probe stack" (no test or page
    matched the old text).
14. **`dense_entries` indexes at order 0 and goes through unit vectors with jets** (round 1; before, unit vectors
    at every order). For entries the reference takes `z` through `dense_entries`, and samples the tangent tensors
    and the bases of `g` and `Kp` through `dense_apply` at the unit vectors, which are what it holds.
15. **The argument `T` of the dense functions** is the plan's name; the docstrings say that the group `T` in the
    shape comments is the order axis and the argument is the tensor.
16. **The doctests of the reference use random stores** (ten lines of setup, repeated so that each block runs on
    its own), not a frame of a point: the definitions do not need one, and unequal sizes make the shapes readable.
    The module is long (3,129 lines, about 600 of them code) because of them and of the 13-line signature on each of 37 functions.
17. **The private record `_Setting`** holds what every definition reads (the stores, the inputs at the nodes). It
    is private, has no behaviour and is never compared (`eq=False`).

## 5. Found outside the slice

- **A trap when perturbing code in place.** A mutation that leaves a file's size unchanged, written and restored
  within the same second, leaves Python's cached bytecode of the *mutated* file valid (the cache is keyed on the
  size and the modification time in whole seconds). I hit it: after a mutation run, `probing.py` was restored but
  its stale bytecode made 247 subtests fail in the next run. Removing `t3toolbox/backend/__pycache__/*.pyc` cured
  it. Whoever perturbs code to test the tests should run with `python -B` and `PYTHONDONTWRITEBYTECODE=1`, or
  delete the cache afterwards. The gate recorded in section 3 ran after the cache was cleared.
- **`docs/api_reference.rst` lists the backend modules by hand** (the "Sampling" item). The new module is not
  there; nothing tests the list. For P19.
- **Not run here:** the floor (Python 3.9, numpy 1.x, jax 0.4.30) and CI's current leg (Python 3.11, numpy 2.4,
  jax 0.10). The new code uses nothing I know to be newer than the floor. The one version-dependent line is now in
  a test, not in the library: the choice between `jax.enable_x64` and `jax.experimental.enable_x64`
  (`jax_in_double_precision` in `tests/test_sampling_definitions.py`), of which only the first could be run here.
- **From the review, for Claude:** section 8 of `SAMPLE_VARIABLES.md` should say that two of its four pairings are
  of the probe kind (for the page of P19), and P9's identity tests must not assume them for apply and entries.
