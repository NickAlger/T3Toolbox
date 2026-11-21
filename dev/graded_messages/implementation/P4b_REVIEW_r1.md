NOT YET

# P4b review, round 1: the input jets and the dense functions

Reviewer, 2026-11-19, on the working tree of `graded-messages` (the implementer's uncommitted changes on `45a92b1f`,
the gate already passed). One should-fix finding: the out-of-range semantics of `index_to_one_hot`, as ruled by
Claude. Apart from that, I could not fault the dense functions, the ground truth of the refactor. They agree with
an oracle of my own to 2e-15 over about 640 outputs, slots beyond the box are never read, and every non-equivalent
mutant I made is caught.

## What is confirmed

- **Against the plan (P4b, section 2.4) and the decided rows.**
  - `dense_probe(T, ww, order=())` and `dense_apply(T, ww, order=())` take input jets `J_in_i + W + (Ni,)` and
    return `F + W + C (+ (Ni,))`.
  - `dense_entries(T, index)` is indexing only (entry 120).
  - `affine_jet(ww, *directions)` and `index_to_one_hot(index, shape)` are in `backend/jets.py` and its
    `__all__`.
  - `order` is read by the one reader, `contractions._read_order`: a tuple only; an integer, a list or a float
    raises.
  - The dense functions never call `jet_contract`.
  - `sampling_definitions` builds its own two-slot input jets (`xnp.stack([w, p])`) and imports nothing of `jets`.
    Its independence test passes unchanged.
- **`_dense_jet`, read line by line.** Correct:
  - It contracts one mode at a time, keeping the partial polynomial's coefficients by multi-index.
  - The input slots are clipped to the box; only sums `r + k` inside the box are formed; one einsum per pair.
  - The contracted axis is placed by `left.index(jj)`.
  - With `d = 1` and its one mode open, it broadcasts the tensor over `W`.
  - A slot no product reaches is `zeros_like(first)`, so the dtype is kept. The result is stacked row-major.
  - Its structural checks: at least `m + 1` axes per input, jet axes of at least one slot, one `W`, each mode's
    length.
- **`affine_jet`.**
  - Its slots go in row-major order, and the zero slots are made as zeros, not `0 * w`.
  - The dtype is the promotion of the point and the directions.
  - A packed input is stacked on axis 1 and reshaped to `(d,) + (2,) * m + W + (N,)`. The row-major split of the
    `2^m` axis matches the slot order.
  - With no direction it returns `ww` itself. It checks only the structure.
- **The `_einsum` change** (`optimize='greedy'` on both backends) is right, and its docstring says why it departs
  from the house convention: jax 0.4.30's default `'optimal'` is exponential in the number of operands. The numpy
  branch's bare `np.einsum` is in a branch already dispatched to numpy.

## Finding

**1. Should fix (ruled by Claude, 2026-11-19). `index_to_one_hot` turns an out-of-range index into a zero vector,
silently.**
- *Place:* `t3toolbox/backend/jets.py`, `index_to_one_hot`: `index_i = xnp.where(index[i] < 0, index[i] + N,
  index[i])`, then compared with `arange(N)`. Its docstring: "an index outside `[-N_i, N_i)` gives a zero vector,
  not an error". `tests/test_sampling_definitions.py`, `TestInputJets.test_index_to_one_hot_is_the_rows_of_the_identity`,
  asserts it.
- *Evidence:* my run, `index_to_one_hot(np.array([[-1, 2, 3, -4, -5]]), (4,))`, gives rows `[3, 2, 3, 0, 0]` with
  sums `[1, 1, 1, 1, 0]`. The index `-5` comes back as a zero vector, with no error. Everywhere else in the library
  an entries index follows numpy indexing: `dense_entries` raises on numpy (and clamps on jax), as its own docstring
  says. An index outside the grid has no one-hot vector. A zero vector would turn a bad index into plausible data
  (apply at the zero vector is 0): a silent wrong answer.
- *Resolve (the ruling):*
  - An index in `[-N, 0)` wraps, as now.
  - A concrete numpy index outside `[-N, N)` raises `IndexError`, naming the mode and the bad value or values (one
    comparison on the array).
  - A jax index may be a tracer and cannot be checked, so its out-of-range vector is zeros. The docstring says so
    in one sentence, beside the contrast with numpy.
  - The test is changed accordingly: an `IndexError` for an index past each end, and the jax case stated.

## A note for P4d (not a finding against P4b)

`data_preparation`'s doctests keep comments in the old order-group notation: `elm_shape=T+W+(Ni,)` in the module
docstring, `T + W = (3, 7)` in `rescale_direction`'s. P4d rewrites those doctests and should write `F` there.

## Independent checks

- **The dense functions against an oracle of my own** (`dense_oracle.py`, seeds 0, 1 and 2).
  - *How it works:*
    - The input jets, unclipped (all their slots), are evaluated at a grid of roots of unity, with
      `M_j = 1 + sum (size_j - 1)` nodes over the contracted modes, so the whole product is interpolated exactly.
    - At each node the tensor is contracted with all the evaluated vectors but the open one, in one plain
      `np.einsum` built in the script.
    - The coefficients come back by an inverse DFT on each axis and are cut to the box.

    This shares nothing with `_dense_jet`, which works one mode at a time, nor with the tests' oracle, which calls
    the same functions at `order = ()`.
  - *What it covers:* 60 random problems per seed:
    - `d` from 1 to 4 (with `d = 1` and its mode open) and `m` from 1 to 3;
    - jet sizes per mode drawn from `{1, F-1, F, F+2}`, and discrete modes of size 1 on every jet axis;
    - stacks `C` in `()`, `(2,)`, `(2, 1)` and `W` in `()`, `(3,)`, `(2, 2)`.
  - *Result:* 210, 212 and 214 outputs; worst relative error 1.9e-15, 1.9e-15 and 2.0e-15.
  - *Slots beyond the box:* NaN in every input slot beyond the box gives results bit-identical and NaN-free, for
    every probe and apply.
- **The recipe of entry 120**, by the same oracle: `dense_apply(T, affine_jet(index_to_one_hot(index, shape), pp),
  (n,))` at `n` = 1, 3 and 5 matches the oracle, and its slot 0 equals `dense_entries` exactly. The implementer's
  test checks the same recipe against the old `dense_entries_derivatives / t!`.
- **`affine_jet` packed at `m = 3`:** the shape is `(3, 2, 2, 2, 2, 4)`; slot 0 is the point; slot `e_2` is the
  second direction; the slots `(1, 1, 0)` and `(1, 1, 1)` are zero.
- **Mutations** (`mutate_p4b.py`, a scratch copy of `t3toolbox/` and `tests/`, `python -B`, deleted after), each
  against `TestDenseFunctions`, `TestInputJets` and the two dispatch tests:
  - *Caught (5):*
    - terms overwritten instead of added;
    - the unreached slots' zeros in the default dtype;
    - the contracted axis taken by its mode number;
    - `index % N` in place of the wrap;
    - no wrap.
  - *Missed, and each equivalent (3):*
    - The box check off by one (`a > n + 1`): the extra coefficient beyond the box is formed and then never read
      into the output, because the input slots stay clipped. Only wasted work.
    - `affine_jet`'s zero made in the default dtype: every slot is then cast to the promoted dtype by
      `xnp.asarray(slot, dtype=dtype)`.
    - `index_to_one_hot` always in `np.float64`: with x64 on, jax's default float is float64; with it off, jax
      truncates float64 to float32. Either way the dtype is the same.

## What I ran (single process)

- The slice's tests: `tests/test_sampling_definitions.py`, `test_jets.py`, `test_data_preparation.py`,
  `test_jax_absent.py`, `test_import_graph.py`, and the two new and changed dispatch tests. 150 passed, 4,707
  subtests, 127 s.
- The module doctests of `probing`, `apply`, `entries`, `jets`, `data_preparation` and `sampling_definitions`: 66
  passed. The docstring examples are real runs.
- `dense_oracle.py` (three seeds) and `mutate_p4b.py`, in
  `/tmp/claude-1000/-home-nick-repos-T3Toolbox/352ad37b-5ac8-40a1-a326-a7fc3a369cb2/scratchpad/reviewer/`.

The working tree is as I found it, apart from this file. The scratch copy is deleted, and nothing of mine is left
running.
