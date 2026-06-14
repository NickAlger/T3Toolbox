# Signature style — shape comments as the real type system

A house convention for T3Toolbox function signatures. Read this once; apply the **principle**
everywhere, the mechanics **within reason**. Reference exemplar: `t3toolbox/backend/probing.py`.
Shape vocabulary (`C`/`W`/`K` blocks, `elm_shape`, single-axis letters): `docs/batching_and_stacking.md`.

## The principle (this is the part that matters)

**In array code the nominal type is nearly useless — the shape *is* the type.** Every argument is
`NDArray` (or `Union[Sequence[NDArray], NDArray]`), so Python's annotations tell a caller almost
nothing: not the rank, not which axes are batch vs core, not the length of a core tuple, not which
bond matches which. We recover that lost contract with a **trailing comment after each argument and
each return element**. The comment is not prose decoration — it *is* the type annotation that the
language cannot express.

Two goals, both about the reader:
1. **Fill the gap in Python's type system** for array/tensor code.
2. **Make a caller understand the arguments at a glance** — jump-to-definition should answer "what
   shapes does this need, and what comes back?" without reading the docstring or the body. That is
   where people actually look.

If an edit serves those two goals, it's in the spirit of the rule. If a mechanical detail ever works
*against* them, the goals win — see "Within reason" below.

## Annotations vs comments: express what the type system *can*, comment the rest

The comment-as-type technique is a patch for a *specific* deficiency — array **shape**. It is **not** a
license to write ordinary types as comments. The division of labor:

- **Whatever Python can express, put in the annotation.** A plain `int | Sequence[int] | None`,
  `Optional[float]`, `bool`, a callable — fully expressible, so annotate them (real, checkable,
  IDE-visible; this is a library for outside users whose tooling consumes annotations). Be
  correct-and-explicit: include `None` in the union / use `Optional`.
- **Whatever it can't, put in the comment.** Array shapes (`NDArray` is content-free), the *length*
  of a sequence (`Sequence[int]` can't say `len=d+1`), cross-argument constraints
  (`1 <= max_rank <= min(N,M)`), and semantics (`requires unstacked`).

The two coexist on one argument, exactly like array args do (a weak `NDArray` / `Union[...]` annotation
*plus* a load-bearing shape comment). Example:

```python
max_tt_ranks:  typ.Union[int, typ.Sequence[int], None] = None,  # scalar caps all, or len=d+1
rtol:          typ.Optional[float] = None,                      # requires unstacked
```

Why not comment-only? Because annotating the expressible part keeps the **presence of a comment
meaningful**: a `#` then signals "there is a contract here the type system cannot capture." If comments
also carried ordinary types, you couldn't distinguish load-bearing shape comments from
types-written-as-comments. (It also keeps the annotation column uniform — every arg carries its Python
type; the comment carries the rest.)

## The mechanics

```python
def compute_mus(
        left_tt_cores:  typ.Union[typ.Sequence[NDArray], NDArray],  # len=d-1, elm_shape=C+(rLi,nUi,rL(i+1))
        xis:            typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=W+C+(nUi,)
) -> typ.Union[
    typ.Sequence[NDArray],  # mus. len=d, elm_shape=W+C+(rLi,)
    NDArray,
]:
```

1. `def name(` on its own line; **one argument per line** as `name: type, # contract`.
2. The closing `) -> ...:` carries the **return shape(s)**, each return element on **its own line**
   with its own trailing contract — expand return tuples one-element-per-line *even when they would
   fit on one line*.
3. **Three vertically-aligned columns**: argument name · type annotation · `#` contract. The block
   reads as a table; a single-arg change touches a single line.
4. The contract uses the **shared shape vocabulary** — the same letters as the body-local suffixes
   (`mu_WCa`, `B0_b_j_c`) and the contraction names (`WCa_Caib_WCi_to_WCb`). One vocabulary across
   signature, body, and helper names: learn it once, read it everywhere.

## Micro-grammar (keep the table cells uniform)

- Arrays: **`# len=<n>, elm_shape=<...>`** — `len` first (for core tuples), then `elm_shape`, comma
  separator. Use `shape=<...>` when there is no tuple wrapper (a bare array).
- Non-array / scalar args: a short **role or constraint** note, e.g. `# 1 <= max_rank <= min(N,M)`,
  `# 'left' | 'right'`, `# requires unstacked`. If a scalar's role is obvious from its name, a note
  may add nothing — then keep it short or omit (see below).
- Prefer the comma form over a period (`# len=d, elm_shape=...`, not `# len=d. elm_shape=...`).

## Within reason (principle over dogma)

This is a convention, not a contract enforced by anything. Apply it where it earns its keep; handle
the exceptions case by case. Reasonable deviations:

- **Trivial scalars** (`use_jax: bool`, an obvious `axis: int`) may need no comment, or only a brief
  role note. Don't manufacture filler to fill a column.
- **Self-evident or non-array types** (a callable, a plain Python `int` count, an enum-like string)
  need only a short note, or none.
- **Pathologically long types/contracts** (a deep nested `Union` of two full representations, as in
  `probe_t3`'s `x`) may wrap across several lines; perfect single-line alignment is impossible there,
  so keep it readable rather than rigidly tabular.
- **One-off / throwaway internal closures** (`_func` inside an `xmap`) don't need the full treatment.

The two costs we accept knowingly: the comments are **unverified** (they can drift — tests and review
are the only guard, so update them when shapes change), and the **alignment is hand-maintained** (no
autoformatter; one long name re-flows the block). Both are the price of using comments as types; we
pay it on purpose.

## Scope notes

- These trailing comments are the **source-reader / jump-to-def** surface. The numpydoc
  `Parameters`/`Returns` docstring is the **rendered-docs** surface. They may both carry shapes today;
  if so, keep them consistent. (Whether docstrings should stop repeating shapes is a separate,
  undecided question — don't conflate it with signature cleanup.)
- Applies to the **frontend** (`TuckerTensorTrain` & friends) too, not only the backend.
- The deferred uniform/weighted layers (`ut3_*`, `ubv_*`, `wt3_*`, `OLD_*`) are not held to this until
  they're repaired.
