# Grouped einsum — the `contract` interpreter

`backend.contractions.contract` is an einsum whose **UPPERCASE letters each stand for a *group* of
zero or more axes**:

```python
>>> import numpy as np
>>> from t3toolbox.backend.contractions import contract
>>> rng = np.random.default_rng(0)
>>> A, x = rng.standard_normal((2, 4, 3)), rng.standard_normal((5, 2, 3))

>>> y = contract('Cio,WCo->WCi', A, x)  # C and W are groups; i, o are ordinary single axes
>>> y.shape                             # here W=(5,), C=(2,) -- both ride along
(5, 2, 4)

```

One call handles every batch configuration — no batch, one batch axis, several, on any subset of
the operands — because a group may be empty, single, or multi-axis, and its size is worked out from
the operands. This page is the user guide: what problem it solves, how to use it, and what it
guarantees. (How it works under the hood — the length solve, the letter expansion — is contributor
material: [`contributor/contractions_internals`](contributor/contractions_internals.md).)

## 1. The problem it solves

Throughout the library one starts from a plain contraction, say

```python
>>> x, G, w = rng.standard_normal(3), rng.standard_normal((3, 4, 5)), rng.standard_normal(4)
>>> np.einsum('a,aib,i->b', x, G, w).shape
(5,)

```

and then needs batched variants of *the same* contraction. A batch axis `u` riding on the first and
third operands:

```python
>>> x, w = rng.standard_normal((2, 3)), rng.standard_normal((2, 4))       # u = 2
>>> np.einsum('ua,aib,ui->ub', x, G, w).shape
(2, 5)

```

Then an independent batch axis `n` on the middle operand:

```python
>>> G = rng.standard_normal((6, 3, 4, 5))                                 # n = 6
>>> np.einsum('ua,naib,ui->unb', x, G, w).shape
(2, 6, 5)

```

Then the first batch becomes three axes `u,v,w` and the second becomes two:

```python
>>> x, w = rng.standard_normal((2, 3, 2, 3)), rng.standard_normal((2, 3, 2, 4))   # u,v,w = 2,3,2
>>> G = rng.standard_normal((6, 7, 3, 4, 5))                                      # n,m = 6,7
>>> np.einsum('uvwa,nmaib,uvwi->uvwnmb', x, G, w).shape
(2, 3, 2, 6, 7, 5)

```

The *idea* is clear and mechanical — some axis groups follow along for the ride, independently of
each other — but a fixed einsum string cannot say "zero or more axes here", so every batch
configuration needs its own subscripts. einsum's own `'...'` covers exactly **one** such group, and
only when its axes broadcast-align across every operand that carries it; two independent groups on
*different* operand subsets (the case above, and the canonical library case — the frame stack `C`
on the cores vs the probe stack `W` on the probe vectors only) are out of reach. `vmap` is not an
option either: the backend is dual numpy/jax, and numpy has no `vmap`.

`contract` closes the gap by making the group a first-class part of the subscripts:

```python
>>> y = contract('Wa,Caib,Wi->WCb', x, G, w)   # W = (), (u,), or (u,v,w)...; C = (), (n,), (n,m)...
>>> y.shape                                    # the operands above: W=(2,3,2), C=(6,7)
(2, 3, 2, 6, 7, 5)
>>> bool(np.allclose(y, np.einsum('uvwa,nmaib,uvwi->uvwnmb', x, G, w)))
True

```

## 2. The dialect

The subscripts are standard einsum format with one extension:

- **UPPERCASE letter = a group of zero or more axes.** In this library the letters follow the
  batching legend — `W` probe stack, `K` tangent stack, `C` frame/core stack, frame-inner order —
  see [`batching_and_stacking`](batching_and_stacking.md) §2–4. (The interpreter itself attaches no
  meaning to the letters; any uppercase letter is a group.)
- **lowercase letter = one axis**, exactly as in einsum.
- An explicit `->` output is required; whitespace is allowed; `'...'` is not part of the dialect
  (an uppercase group subsumes it); no symbol may repeat within one term.
- Group axes follow einsum's usual extent rules: the same group's axes must agree across the
  operands that carry it (checked per axis), except that a size-1 axis broadcasts.

Dispatch follows the house convention: numpy-vs-jax is inferred from the operands, the numpy path
uses the greedy pairwise BLAS-friendly contraction order when there are **more than two operands**
(with two or fewer there is nothing to order, so no `optimize=` is passed), and jax gets a single
fused einsum for XLA to optimize.

## 3. Usage

A batched matvec where the matrix batch `C` is shared by both operands and the vector batch `W`
rides on one:

```python
>>> rng = np.random.default_rng(0)
>>> A = rng.standard_normal((2, 4, 3))      # C=(2,), legs (i, o)
>>> x = rng.standard_normal((5, 2, 3))      # W=(5,), C=(2,), leg o

>>> y = contract('Cio,WCo->WCi', A, x)
>>> y.shape                                 # frame-inner: W outer, C inner
(5, 2, 4)
>>> bool(np.allclose(y, np.einsum('cio,wco->wci', A, x)))    # the |W|=|C|=1 case, by hand
True

```

The **same string** covers every other batch configuration, including none at all — an empty group
simply contributes no axes:

```python
>>> contract('Cio,WCo->WCi', np.ones((4, 3)), np.ones(3)).shape         # W=(), C=()
(4,)
>>> contract('Cio,WCo->WCi', np.ones((2, 2, 4, 3)),
...                          np.ones((7, 5, 2, 2, 3))).shape            # W=(7,5), C=(2,2)
(7, 5, 2, 2, 4)

```

A group summed away is a group absent from the output:

```python
>>> v = rng.standard_normal((6, 3, 4))      # W=(6,), C=(3,), leg a
>>> s = contract('WCa->Ca', v, len_W=1)
>>> s.shape
(3, 4)
>>> bool(np.allclose(s, v.sum(axis=0)))     # summing away W is summing over its axes
True

```

### When the shapes cannot pin a split: `len_<G>=`

Group sizes are solved from the operand ndims. Sometimes the ndims genuinely cannot pin a split —
in `'WCo,WCa->Cao'` the groups `W` and `C` appear only together on every operand, so only their
combined total is knowable. Then `contract` refuses, and the error says exactly what to pass:

```python
>>> zo, za = np.ones((5, 2, 3)), np.ones((5, 2, 6))     # W=(5,), C=(2,)
>>> contract('WCo,WCa->Cao', zo, za)                    # doctest: +IGNORE_EXCEPTION_DETAIL
Traceback (most recent call last):
    ...
ValueError: contract('WCo,WCa->Cao'): the operand ndims do not determine the axis counts of
group(s) W, C -- only combined totals are pinned. Supply one of: len_W, len_C (repeat with
another if still underdetermined). Note this is decided from the subscripts alone, so a call
site either always needs it or never does.

>>> contract('WCo,WCa->Cao', zo, za, len_W=1).shape     # summed over the probe stack W
(2, 6, 3)

```

`len_W=1` means "the group `W` has one axis". Three guarantees make this pleasant to live with:

- **Whether a supplement is needed is a property of the *subscripts*, never of the shapes.** A call
  site either always needs its `len_<G>` or never does — code that works unstacked cannot start
  silently misgrouping when a stack appears (the interpreter deliberately refuses to use
  shape-specific accidents like a zero-length prefix to resolve a split).
- **Redundant supplements are verified, not ignored.** Passing `len_C=1` where the shapes already
  pin `C` is a free assertion; passing a wrong value raises.
- **Groups that always travel together need no split at all.** In `'WKCi,Cio->WKCo'` nothing ever
  separates `W` from `K`, so the boundary between them is unobservable — the call needs no
  supplement, and any explicit split you do pass (`len_W=0`, `1`, or `2` here) is verified and
  yields the identical result:

```python
>>> ops = (np.ones((5, 3, 2, 4)), np.ones((2, 4, 6)))
>>> contract('WKCi,Cio->WKCo', *ops).shape                       # no supplement needed
(5, 3, 2, 6)
>>> [contract('WKCi,Cio->WKCo', *ops, len_W=n).shape for n in (0, 1, 2)]     # all identical
[(5, 3, 2, 6), (5, 3, 2, 6), (5, 3, 2, 6)]

```

### Decision rule (from the batching guide)

If your batch axes form **one** broadcast-aligned prefix shared by the operands, a plain `'...'`
einsum is all you need. If independent groups live on **different operand subsets** — or you want
one call that covers every batch configuration — use `contract`. See
[`batching_and_stacking`](batching_and_stacking.md) §4.

## 4. What you get beyond convenience

- **No reshape, ever.** The interpreter expands each group into ordinary single-axis letters and
  runs one einsum on the operands exactly as given. There is no flatten/unflatten bookkeeping, no
  data movement, and nothing that fixes a memory layout.
- **Any axis of any group is shardable.** Because nothing is reshaped, every group sub-axis is an
  honest einsum axis; under jax sharding, batch and free axes cost zero collectives wherever they
  sit in their group, and a summed axis costs exactly the all-reduce the mathematics requires.
  (Compiler-verified over every subscripts string in the library:
  `tests/test_contractions_sharding.py`.)
- **Per-axis shape checking.** Each group axis is a named einsum index, so mismatched extents
  between operands raise instead of silently computing garbage — including the treacherous case
  where a transposed batch (`W=(2,3)` vs `(3,2)`) has the same total size.
- **Negligible overhead.** The parse/solve/expansion is cached on `(subscripts, operand ndims,
  supplements)`; a warm call is a dictionary lookup plus the einsum. Under `jit` all of it happens
  at trace time, so the compiled program contains only the einsum.

## 5. Where it came from (motivation, in brief)

Until 2026-07-17 this machinery was ~104 hand-enumerated *named functions*
(`WCa_Caib_WCi_to_WCb(...)` — the subscripts spelled in the function name, a hand-written
flatten-to-one-axis body per name). That design was auditable but had two structural costs: the
enumeration grew without bound (every new block combination was a new function plus a hand-written
test), and the name was a promise the body could drift from — twice, exactly-numerically-invisible
fusions of two groups were caught only by compiling under sharding. The interpreter keeps the
readable subscripts (now at the call site) and lets the machine derive the behavior from them, so
the name-vs-body gap no longer exists, and the flatten — whose one-axis layout was the only reason
sharding was ever restricted — is gone entirely. The full decision record, including what was
tried, measured and rejected along the way, is in
[`contributor/batching_internals`](contributor/batching_internals.md); the implementation itself is
in [`contributor/contractions_internals`](contributor/contractions_internals.md).
