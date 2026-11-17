NOT YET

# P2 review, round 1

Slice P2 of `PART1_PLAN.md`: the reference module `backend/sampling_definitions.py` and the dense functions.
Reviewed 2026-11-17 on `graded-messages` at `3fcfbe7`, working tree as the implementer left it.

**In one paragraph.** I found no wrong definition. All 37 functions agree with my own computation of each
variable from `SAMPLE_VARIABLES.md` (worst relative difference 8e-14, at order 5), with the old library (6e-15),
and with hand-written einsums. The module is independent of every sweep and kernel. Of 53 real mutations of the code
the tests caught 51; the two that passed go around the test that guards the module's independence. Two small
should-fix findings keep this from a sign-off: that guard (finding 1), and one sentence of the reference's own
documentation that is false for apply and entries (finding 2). The rest is minor. Two things need a decision
from Claude before later slices, not a change now (section "The two decisions").

## Findings

### Should fix

**1. The test that guards the module's independence can be walked around.**
`tests/test_sampling_definitions.py`, `TestStructure.test_the_module_calls_nothing_of_the_library_but_the_dense_functions`.
- Evidence (scratch copy, `/tmp/p2rev/mutate2.py`, N04 and N05). With
  `from t3toolbox.backend.probing import t3_probe as _old_kernel` added to the module and used, the whole test
  file passes (51 passed). The same with `_old = getattr(probing, 't3_' + 'probe')`.
- Why: the test collects the *modules* named in `import` and `from ... import` statements, and the attributes
  reached as `probing.X`, `apply.X`, `entries.X`. A name imported *from* one of the three allowed modules is in
  neither set, and `getattr` on a module is not an `ast.Attribute`.
- The test is real for what it does see: it fails on `probing.t3_probe(...)`, on any other library module, and on
  the names `contract`, `jet_contract`, `safety`, `xscan`, `xmap`.
- Resolves it: assert that the only `ImportFrom` of the library is `from t3toolbox.backend.common import *`, and
  that the names `probing`, `apply`, `entries` occur in the syntax tree only as the value of the three allowed
  attributes (count the `ast.Name` nodes with those ids against the `ast.Attribute` nodes), which also shuts out
  `getattr(probing, ...)` and `probing.__dict__`.

**2. `mu_tilde`'s docstring says it is the multiplier of `nu_L_dot` without saying "probe only"; for apply and
entries that is false.** `sampling_definitions.py` line 1715 to 1717.
- Evidence (`/tmp/p2rev/pair_apply.py`; apply, order 2, the direction wholly to the right of bond 1):
  the seed paired with `Jp` is 30469.0086; the top coefficient of `mu_tilde[1]` with `nu_L_dot[1]` from the
  reference is 0.0; the top coefficient of `(seed * mu[1])` with `nu_L_dot[1]` is 30469.0086.
- So with the seed at the right end (the convention of the 4.4 note), two of the four pairings of
  `SAMPLE_VARIABLES.md` section 8 (`mu_tilde` with `nu_L_dot`, `xi_tilde` with `eta_D_dot`) are identities of the
  probe kind only. The tests already know this (`test_the_multipliers_of_apply_and_entries` checks only the other
  two), but the docstring reads as a statement about every kind, next to "for apply and entries `mu_tilde` is
  zero".
- Resolves it: say "(probe)" in that sentence, and say in the module docstring's `_tilde` bullet that for apply
  and entries only `nu_tilde` with `mu_R_dot` and `eta_tilde` with `xi_U_dot` are multiplier pairs. Claude should
  carry the same into the log for P9 ("the identities of P5 to P8 for both" cannot include those two) and for
  the page of P19.

### Minor

**3. "Exact" interpolation loses digits with the order, and the docstring does not say so.** Module docstring,
"Jets" paragraph; `_taylor`.
- Evidence (`/tmp/p2rev/misc.py`, d = 4, probe, against my exact polynomial arithmetic): `mu` is right to 1e-14 at
  order 3, 2e-12 at order 8, 9e-11 at order 12, 2e-8 at order 16. Slots that are exactly zero in truth (above the
  degree) come out as 1e-10 at order 8 and 5e-7 at order 16. `z`, `Jp` and the blocks do not have this (they go
  through the multilinear expansion): 3e-16 at every order.
- Cause: the monomial coefficients of a polynomial from its values on `[-1, 1]` are ill-conditioned whatever the
  nodes. The planned tests of P5 to P11 use orders 0, 1, 3, where it does not matter.
- Resolves it: one sentence in the docstring giving the accuracy (about 1e-14 up to order 3, falling to 1e-8 at
  order 16) so that nobody later tests the engine against it at order 16 with a tolerance of 1e-10. Better, if
  wanted: evaluate at the roots of unity (complex `s`) and read the coefficients by a discrete Fourier transform,
  which is perfectly conditioned (the P1 reviewer used this). Not needed for sign-off.

**4. `dense_entries` at order 0 is a contraction with unit vectors, not indexing.** `entries.py`, `dense_entries`.
- Evidence (`/tmp/p2rev/dense.py`): with one `inf` in the tensor at a position no index points to, 4 of 24
  entries come back `nan` (0 times inf). The cost is the whole tensor per sample, not one entry.
- It is exact for finite tensors (I checked `==` against indexing), and the handover says why (one code path).
  For a function whose first line says `T[index[0], ..., index[d-1]]`, indexing at order 0 would be both the
  more independent ground truth and cheap for the examples of P13. Claude's call; at least the docstring should
  say that a non-finite entry anywhere spoils the result.

**5. The dense functions accept a negative `order` when `pp` is given** and return a jet with one slot
(`dense_apply(T, ww, pp, -1).shape == (1,) + W + C`). The backend promises nothing here; a one-line check beside
the existing `pp`/`order` check would make the message they already have cover it.

**6. `Kp` takes its route from what is installed, not from the arrays.** `_Kp`: with jax installed, numpy inputs
go through `jax.jvp(jax.grad(...))` inside an x64 context and come back as numpy. This is what the plan says
("by jax where present, finite differences otherwise"), and I checked the two routes against each other and
against my own computation (3e-15), and that the global x64 setting is unchanged afterwards. But the finite
differences are exact and as fast (0.08 s against 0.10 s after a 2 s first trace), so the simpler rule "numpy in:
finite differences; jax in: jax" would follow the library's dispatch convention and make a numpy-only result the
same on every machine. Not a defect; for Claude to weigh.

**7. No test of the reference has `W = ()`, one site, or two sites.** I ran all 37 names for the three kinds at
d = 1 and d = 2 and with no sample axis (`/tmp/p2rev/indep.py` problem C, `/tmp/p2rev/d1.py`): all right (1e-15).
A fourth problem in `PROBLEMS` with `W = ()` and `d = 2` would pin it.

**8. Form, small.**
- `_subnetwork` and `_tensor` pass `optimize='greedy'` to `xnp.einsum` on the jax path too; `CLAUDE.md` says the
  jax path omits it. It works (jax accepts it).
- The einsum strings of the reference are built in code, not literals. Unavoidable for "any number of sites in
  one einsum", and the rule of literal strings is about the engine's terms; noted so that nobody is surprised.
- The return line's comment is not in the column of the arguments' comments (37 times).
- A uniform corewise `.data` is also a 4-tuple and would be read as four stores. The docstring says "ragged
  only" under "What is not here"; nothing more is needed unless P11 trips on it.
- `i` and `order` are both positional integers with defaults, in that order (the plan's sketch has the same):
  `sdef.mu(kind, frame, sample, 3)` is bond 3, not order 3. The doctests always write `order=`; good.

## The two decisions the implementer asks about

**(a) Zeros for `xi_tilde`, `mu_tilde`, `xi_tilde_dot`, `mu_tilde_dot` of apply and entries.**
- What it does, checked for every name and kind: for apply and entries those four are exact zeros of the right
  shapes (`T`, not `A`); `nu_tilde`, `nu_L_tilde` are the seed times `nu`, `nu_L` as polynomials; `eta_tilde`,
  `eta_D_tilde` the seed times `eta`, `eta_D`; `nu_L_tilde_dot` and `eta_D_tilde_dot` the seed times the raw dots;
  the start of each right-to-left tilde sweep is the seed times the ones, of the tilde-dot one zero.
- Is zero the true value? It is the only value consistent with section 6 of the page once the seed is at the
  right end, which the 4.4 note decides. I checked this without the reference: with zeros, the readouts of the
  page (`H_tilde` as three `H_term`s, `V_tilde`, the nine terms of `H_tilde_dot`, `V_tilde_dot`) equal my own
  dense Jacobian transposed and my own second derivative, for apply and entries, at orders 0 to 5 (3e-15). With
  the other candidate, `mu_tilde = seed * mu` (what the multiplier pairing of section 8 would give), `H_tilde`
  would count the seed twice. So zero is right as the value; what does not carry over to these kinds is two rows
  of section 8 (finding 2). The page itself is silent: its status line says apply and entries are not covered.
- Zeros or raise? Keep the zeros in P2. They are true, they are stated in the docstrings, and raising would
  decide now which names the graphs of apply and entries have, which is P9's content. If P9's graphs leave the
  four names out, the engine rejects them and the reference's zeros are never compared; if P9 keeps them, they
  are compared. Either way P9 should add a test that the set of names per kind is the same in the reference and
  the engine, and bring the reference in line then.
- Whose call: Claude's, as a minor issue, for P2 (no decided row, public name or signature changes). Whether
  the engine accepts those four names for apply and entries is user-visible; row 6.4 says "every variable of a
  kind's graph" and leaves the graph to the plan, and P9 only says what the kinds *need*. I would tell the
  maintainer at P9 which way it went.

**(b) `frame` is the four stores; a corewise point is `(B, G, G, G)`; no functions for the corewise spellings.**
- Enough for P8? Yes, with a table in P8's test. On `(B, G, G, G)` the reference gives everything a corewise
  call of the engine returns: `nu`, `eta`, `nu_tilde`, `eta_tilde` under those same names (on that frame they equal `nu_L`,
  `eta_D` and so on: `TestCorewiseFrame` checks it);
  `xi_dot`, `mu_dot`, `xi_tilde_dot`, `mu_tilde_dot` under the same names; the raw `nu_L_dot`, `eta_D_dot`,
  `nu_L_tilde_dot`, `eta_D_tilde_dot` for whatever the corewise call names them; `g` and `Kp` with the shapes
  of `(B, G)`.
- Honest documentation? Yes for what it says: the module docstring states the convention and that `nu_L` and
  `nu` are one array there, and a frame of two stores raises with a message that says what to pass.
- **A gap in the plan, found here, for Claude before P8.** The handover speaks of "the four spellings that exist
  only on a corewise call (`nu_dot`, `eta_dot`, `nu_tilde_dot`, `eta_tilde_dot`)". No table of the plan or of
  `SAMPLE_VARIABLES.md` has those names: section 2.3 lists the 37, and O-13 says the names with a basis are not
  accepted on a corewise call. Taken together, a corewise call has no accepted name for the right-side raw dot
  unless `nu_dot` and the three others exist, which is an inference from section 9 ("its dots are those of the
  raw form"), not a decided name. New public names are the maintainer's. When they are settled, N17 ("one
  function named exactly as the variable") decides whether the reference gets functions of those names or a
  documented table; either is small.

## What I checked, and how

**1. Every definition, by my own computation** (`/tmp/p2rev/indep.py`; nothing of the implementer's tests).
- My route: the recurrences of sections 5.1 to 5.4 of the page, written by me, in exact polynomial arithmetic
  on coefficient arrays of full degree (no interpolation, no truncation until the comparison). The reference
  contracts each subnetwork whole and interpolates, so the two routes share nothing.
- The outputs by a third route: my own dense tensor and dense tangent tensor (sequential `tensordot`), sampled
  mode by mode in polynomial arithmetic; `g` and `JtJp` from my own dense Jacobian, entry by entry, paired with
  `weight^2 r` in forward order by the plain slot-by-slot sum; `Kp` as the exact double sum over ordered pairs
  of distinct cores of `phi` with one core replaced by a unit core and another by its direction (`phi` is
  multilinear), with no differentiation at all.
- The page against itself: the readouts of section 6 computed from my recurrence variables equal my dense
  outputs and blocks (4e-15), for the three kinds. So the page is consistent, and for apply and entries the
  readouts hold with the zeros of 3(a).
- Problems: d = 3 with tails on both ends and `rL_d != rR_d`, `W = (3,)`; d = 4 with unequal sizes, `W = (2, 2)`;
  d = 2, `W = ()`; d = 1. Stores random, not orthogonal, not consistent. Kinds probe, apply, entries. Orders
  0, 1, 2, 4, 5. The seed from `y` and a weight, and a seed of my own passed as `w_tilde`.
- Compared, each at every index, with the shape checked exactly (`T` or `A`, `W`, the size of the edge in the
  variable's basis) and the single-index call equal to the entry of the whole: `w`, `w_tilde`; `xi`, `mu`, `nu`,
  `eta`, `nu_L`, `eta_D`; `xi_U_dot`, `mu_R_dot`, `nu_L_dot` and `eta_D_dot` in the frame form, in the raw form,
  and with both directions given (the raw form, O-12); `xi_dot`, `mu_dot`; the six tildes; the four tilde-dots;
  `z`, `r`, `Jp`, `g`, `JtJp`, `Kp`, `V_tilde`, `H_tilde`, `V_tilde_dot`, `H_tilde_dot`; `f`, `r_block_sumsq`
  (its shape is the weight's), `Jp_norm_sq`.
- Result: every one agrees. Worst relative difference 3e-14 at orders 0 and 1, 8e-14 at order 5 (the
  interpolation; the outputs and blocks stay at 6e-15).
- By hand-written einsums over the subnetwork, d = 3, order 0 (`/tmp/p2rev/hand.py`): `mu[2]`, `nu[1]`, `nu_L[1]`,
  `eta[0]`, `eta[1]`, `eta[2]`, `eta_D[1]`, `mu_R_dot[2]`, `nu_L_dot[1]` (frame), `eta_D_dot[1]` (frame, the five
  places, with `R[1]` or `L[1]` at the site as the page says, and raw), `eta_tilde[1]`, `mu_tilde[2]`, the apply
  `nu_tilde[1]` and `eta_tilde[1]`, `z`. All within 4e-15.
- The turn, from first principles (`/tmp/p2rev/slot.py`): with no notion of a turn in my code,
  `d(sum over modes left of bond i of <weight^2 r, z>) / d(right side of bond i at slot m)` equals
  `mu_tilde[i][order - m]`, and the mirror for `nu_tilde`, at every bond and slot (relative 2e-15). So slot `t`
  of a stored tilde variable is the sensitivity to slot `order - t`, as row 2.4 says.
- The weight enters squared, once, at the seed; `r`, `r_block_sumsq` are unweighted; `f` and `Jp_norm_sq` carry
  `weight^2`.
- Against the old library (`/tmp/p2rev/old.py`): `z`, `Jp`, `g`, `JtJp`, `Kp` against `ProbeDerivativesKind`,
  `ApplyDerivativesKind`, `EntriesDerivativesKind` at orders 0, 1, 3 (conversions: `t!` per slot, the old weight
  is the new one divided by `t!`) and against the plain `ProbeKind`, `ApplyKind`, `EntriesKind`, on a frame with
  tails and on d = 4; `Kp` with the corewise sweep and with the manifold frame's sweep. Worst 6e-15.
- Where the page is ambiguous or the implementer chose: the place of the seed for apply and entries (3(a));
  section 8 for those kinds (finding 2); which form the two shared dot names take with both directions (O-12
  decides: raw; done so). I found no disagreement with the page.

**2. Independence.** Read every function. Each edge variable is one `einsum` over all the cores and inputs of
its side (`_left_side`, `_right_side`, `_trunk`, `_leaf`), summed over the places of the direction and the seed;
no variable is built from another. `z`, `Jp` go through the dense functions (the multilinear expansion over
subsets of modes, not a banded recurrence); `g`, `JtJp` through a basis of directions; `Kp` through jax or the
stencil. The module imports `common` and the three sampling modules, and uses of those only the three dense
functions. No `contract`, `jet_contract`, `xscan`, `xmap`, `safety`. Nothing in it is the recurrence the engine
will use. The one shared piece is `probing._dense_jet`, behind `z`, `Jp`, `g`, `JtJp`, `Kp`; I checked it
separately (below).

**4. The dense functions** (`/tmp/p2rev/dense.py`). Against my own polynomial arithmetic one sample and one
stack element at a time, d = 4, `W = (2, 2)`, `C = (2, 3)`, orders 0, 1, 2, 3, 4, 6: 2e-14; shapes `T + W + C
(+ (Ni,))` exactly; order 0 of `dense_entries` equal to indexing with `==`; against the old
`dense_probe_derivatives` and `dense_apply_derivatives` divided by `t!`. `pp` at order 0 raises; an order
without `pp` raises; both name the function. The form is N16's and O-27's: `(T, ww | index, pp=None, order=0)`.
The 12 calls of the library's `dense_probe` are changed (8 doctests in `probing.py`, one each in `manifold.py`
and `uniform_manifold.py`, `tests/test_manifold.py`, `tests/test_tucker_tensor_train.py`) and nothing else
calls it: I searched `t3toolbox/`, `tests/`, `docs/`, `examples/`, `README.md` for the name in every form; the
other hits are local helpers and `dense_probe_derivatives`.

**5. The tests.** Read all of `tests/test_sampling_definitions.py`. Its routes are its own (`np.tensordot` one
sample at a time, interpolation at nodes that are not the reference's, truncated products as loops,
`tv_to_dense` and `t3_to_dense`, the plain forward pairing for `g`). Tolerances are sensible (1e-10 relative,
1e-8 where an interpolation of interpolations is compared, exact for structure).
Mutations, in a scratch copy under `/tmp/p2rev/tree` made without `__pycache__`, run with `python -B` and
`PYTHONDONTWRITEBYTECODE=1`, the tree untouched (`/tmp/p2rev/mutate.py`, `mutate2.py`; logs `mut.log`, `mut2.log`):
- Caught (51): `nu_L` through `R`; `r = y - z`; the seed polynomial not turned; the seed without the weight;
  `L` cores after the direction in either term of the frame form; `L_dot` and `D_dot` together at a site;
  Taylor slots shifted by one; the seed allowed at input `i` in `eta_tilde` and in `eta_D_tilde`; no seed at
  the right end for `nu_L_tilde` of apply; `f` without the half; `Jp_norm_sq` unweighted; the pairing without
  the turn back; `pp` counted on the open mode in the dense jet; `C` before `W`; the objective of `Kp` doubled;
  the seed of `JtJp` not turned, unweighted, with the weight not squared; `D_dot` allowed at site `i` in
  `eta_D_tilde_dot`; the empty sum of `mu_R_dot` sized by `L`; `r_block_sumsq` weighted; `xi_tilde_dot` through
  `D`; the seed not allowed at the last site in `mu_tilde_dot`; entries from reversed samples; the weight of
  mode 0 for every mode; a given `w_tilde` turned again; the stencil without `1/k`; the `H` blocks of the jax
  `Kp` zeroed; every entries sample at one index; `ww` and `pp` swapped in `w`; one node fewer; the frame form
  winning with both directions; the dense jet in derivatives; `s` scaled by 2; a mode dropped from `f`; the
  wrong route of `Kp`; `mu_tilde = seed * mu` for apply; errors of 1e-7 in `z` of entries and in the raw
  direction; a nonzero slot above the degree; a lost doctest; a numpy `tensordot` on the jax path; `Kp` of jax
  inputs returned as numpy. In `tests/test_dispatch.py`: `np.zeros_like` and `np.einsum` on a tracer, a float64
  `eye`, the dispatch read off `T` only. In `tests/test_jax_absent.py`: an unconditional `import jax`; `Kp`
  always by jax.
- Not caught (2): the two of finding 1.
- Two of my mutants were equivalent (a no-op control, and a branch that cannot be taken) and passed, as they
  should.

**6. Form.** The one signature is on all 37 functions (a test holds it). `y` and `weight` are needed (`r`, `f`,
`r_block_sumsq`, and the seed of the residual cannot be formed without them) and are the engine's names for
the same things; `i = None` for the whole variable is an addition that a call in the plan's form does not
notice. Names are those of `SAMPLE_VARIABLES.md`. No site group. No `safety`, no numerical check that raises
(the `ValueError`s are an unknown kind, the count of stores, 17 sites, an index out of range, a missing
argument). Dispatch is inferred from the arrays in every helper (but see finding 6). The doctests run (37 of
the reference, 74 with the five other modules touched) and one changed expected output fails them. On the size:
about 600 of the 3,071 lines are code, the rest docstrings; the code has no repetition that hides anything (the
37 bodies are two to eight lines each on shared helpers). The doctests repeat ten lines of setup in every
block, which `docs/contributor/doctest_style.md` asks for ("Self-contained blocks are the norm"), so it is the
house style and I do not ask for a change. Writing the four stores with literal shapes
(`D = [rn((2, 4)), rn((3, 5)), rn((2, 6))]`) would halve it and be easier to read than the five-tuple of size
lists; a suggestion only.

**7. The gate.** The file times agree with the handover: the last write to any file of the slice is 07:53:36
(`sampling_definitions.py`), the gate's `PID` file is 07:56:50, its last log 08:18:27, and only the handover
is newer. `/tmp/gate_P2/results.tsv` has every step PASS with the counts the handover gives (1,496 passed, 5
skipped, 1 xfailed, 64,796 subtests; 282 module doctests; sphinx 0 warnings).
The failed quick run: I believe the explanation. All 247 failures of `/tmp/gate_P2_quick/quick.log` are at
orders 1, 2, 3, 5 and none at order 0, all in tests that go through the jets of a dense tensor, which is what a
live mutant of `probing._dense_jet` gives; the handover's mutation "`pp` and `ww` swapped" changes no file
size, which is what the trap needs. Today every one of the 69 bytecode files under `t3toolbox/` matches a
fresh compile of its source (I compared them; two old files under `tests/__pycache__` have headers that do not
match their sources, so Python ignores them).
I did not run the whole gate again: nothing has changed since it ran, and the bytecode is clean. I ran, in the
tree, with no bytecode written and no pytest cache:
`tests/test_sampling_definitions.py`, `tests/test_jax_absent.py`, `tests/test_import_graph.py`,
`tests/test_tucker_tensor_train.py` and `test_jit_dense_sampling_functions`: 140 passed, 28,015 subtests;
the probe tests of `tests/test_manifold.py`: 2 passed; the module doctests of `sampling_definitions`, `probing`,
`apply`, `entries`, `manifold`, `uniform_manifold`: 74 passed. And the test file alone in the scratch copy: 51
passed, 3,746 subtests.

**Half-done work.** None found: no marker, no debug print, every public function has its Returns and Examples
sections, `__all__` is the 37 names, no stash, no stray file.

## Outside the slice

- The gap in the plan about the corewise spellings (section (b) above), for Claude before P8.
- Section 8 of `SAMPLE_VARIABLES.md` should say that two of its four pairings are for the probe kind (finding
  2), when the page of P19 is written; and P9's identity tests must not assume them.
- `docs/api_reference.rst` does not list the new module (the handover says so too; P19).

## State of the tree

Unchanged apart from this file (`git status` before and after my runs is the same; no worktree; nothing left
running). My scripts and logs are in `/tmp/p2rev/`; the scratch copy of the tree there is removed.
