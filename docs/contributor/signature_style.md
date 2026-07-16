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
def compute_mu(
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
- **Placement / dtype, when load-bearing:** `NDArray` says nothing about *dtype* or *host-vs-device*
  either — so when those are part of the contract (not just shape), put them in the comment too. The
  motivating case is the uniform **masks**, tagged **`HOST bool, static`** alongside their shape. The
  `HOST` token means *"must be a host (numpy) array, never jax/traced — it is static structure,
  runtime-enforced by the mask guard"* (under jit a traced mask breaks; see
  `uniform_pytree_composition.md`). **No tag = the default**: dtype-agnostic and `xnp`/backend-polymorphic
  (numpy *or* jax). Spell a dtype (`bool`, `int`) only when it actually matters. This keeps the old
  "a bare `np.` is a backend-agnosticism tell" intuition usable: a `HOST` arg is *meant* to be numpy.
- Prefer the comma form over a period (`# len=d, elm_shape=...`, not `# len=d. elm_shape=...`).

## Aligning the comment column: one block, or grouped?

**Names and types are *always* column-aligned** — each is an O(1) jump for the eye. The only open
question is the **comment** column. The cost of forcing every comment into a single column is the
**type-length spread**: the gap an argument gets is `(longest type+default) − (its type+default)`. So:

- **Small spread → one aligned comment column.** Cheap, tidy, every comment O(1). The common case
  (e.g. `probing.py`, where every arg is the same array type — the column falls out for free).
- **Large spread → group, and align *within* each group.** Split the args into runs of similar-length
  types (which usually coincide with roles: operands / rank caps / tolerances / flags), put a **blank
  line** between groups, and align the comment column *within* each group. **Group by type-*length*,
  not role** — length is what sets the gap; role is only the usual proxy. When they disagree (e.g. a
  short `str` selector sitting among long `Union` caps), an argument whose type-length matches no group
  gets **its own** blank-delimited group rather than being force-aligned into a column it doesn't fit —
  otherwise you reintroduce the very gap grouping exists to remove. The test is the reader's eye, not a
  column count: split an arg out when force-aligning it would open a gap big enough to lose the row (a
  few spaces is fine; a tab-plus usually is not). The comment column resets
  per group; **names/types stay aligned across the whole signature** (still O(1)) — only the comment
  column is per-group. This keeps every comment O(1) *within its group*, keeps lines short, and makes
  the grouping visible. It is strictly better than one column with 30-space gaps, and far better than
  the **staircase** (each comment 2 spaces after its own type, no cross-row alignment): the staircase
  minimizes gaps but makes the comment column *unpredictable* — O(n) to find — which is the worst to
  scan. **Don't use the staircase.**

Rule of thumb: group when the args span clearly different type-lengths (a long array/`Union` next to a
bare `float`/`bool`) **and** there are enough rows for a staircase to bite (≳4). For two or three args
a single column is already fine. (`t3m`, frontend and backend, is the worked example: groups
operands / `max_*` caps / `rtol`,`atol` / `oversample`, blank-line-separated.)

## Within reason (principle over dogma)

This is a convention, not a contract enforced by anything.

> **The iron imperative for every comment is that it be useful to a user of the code or a maintainer
> of the library.** A comment that only repeats what the signature already says is not neutral — it is
> a liability: a second, unverified copy of the contract, free to drift out of sync with the first.
> The shape comment exists for one reason: to carry **type information the language cannot express**
> (shapes, sequence lengths, which bond matches which). Where that information is *already* expressed —
> by an annotation, or by the name itself — the comment has no job to do.

Apply the convention where it earns its keep; handle the exceptions case by case. **These rules cannot
anticipate every case: the deviations below are illustrative, not exhaustive, and it matters more to
follow their spirit than their wording.** Reasonable deviations:

- **Trivial scalars** (`use_jax: bool`, an obvious `axis: int`) may need no comment, or only a brief
  role note. Don't manufacture filler to fill a column.
- **Self-evident or non-array types** (a callable, a plain Python `int` count, an enum-like string)
  need only a short note, or none.
- **Pathologically long types/contracts** (a deep nested `Union` of two full representations, as in
  `t3_probe`'s `x`) may wrap across several lines; perfect single-line alignment is impossible there,
  so keep it readable rather than rigidly tabular.
- **One-off / throwaway internal closures** (`_func` inside an `xmap`) don't need the full treatment.
- **The named contractions in `backend/contractions.py`** — an explicit exception. There the function's
  *name* already **is** the shape contract (`WKCa_Caib_WCi_to_WKCb` says operand 1 is `W+K+C+(a,)`,
  operand 2 is `C+(a,i,b)`), so a comment restating it (`# W + K + C + (a,)` beside `WKCa`) tells the
  reader nothing the signature has not already told them — exactly the liability above. Leave those
  signatures bare; the module's original style is the reference. Prose describing an operand's *role at
  one call site* ("mu jet", "probe vector w") is worse than redundant there: `contractions.py` is a
  deliberately caller-agnostic mechanical workaround for einsum's lack of grouped indices, so naming one
  caller's variables in it misleads a reader about what the function is for — that context belongs in the
  docstring, if anywhere. The exception is specific to the **named contractions**: a parameter whose name
  does *not* state its contract (`n_probe: int  # len(W)`) still needs the information — though the better
  fix is usually a name that states it (`len_W`) than a comment compensating for one that doesn't.

The two costs we accept knowingly: the comments are **unverified** (they can drift — tests and review
are the only guard, so update them when shapes change), and the **alignment is hand-maintained** (no
autoformatter; one long name re-flows the block). Both are the price of using comments as types; we
pay it on purpose.

## Scope notes

- These trailing comments are the **source-reader / jump-to-def** surface. The numpydoc
  `Parameters`/`Returns` docstring is the **rendered-docs** surface. They may both carry shapes today;
  if so, keep them consistent. (Whether docstrings should stop repeating shapes is a separate,
  undecided question — don't conflate it with signature cleanup.)
- Applies to the **frontend** (`TuckerTensorTrain` & friends) too, not only the backend — and to the
  uniform layer (whose `HOST bool, static` mask tags are the motivating case above).
- The parked weighted layer (`wt3_*`) is not held to this until its post-1.0 revival.
- The reader-side half of this convention (the decode table users need) is
  [`../reading_signatures.md`](../reading_signatures.md).
