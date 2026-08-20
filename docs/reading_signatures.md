# How to read T3Toolbox signatures

T3Toolbox signatures carry their real contract in **trailing `#` comments**, because in array code
the nominal type is nearly useless: every operand is `NDArray` (or `Union[Sequence[NDArray],
NDArray]`), which says nothing about rank, batch-vs-core axes, tuple lengths, or which bond matches
which. The comment is not decoration — **it is the type annotation the language cannot express.**
The API reference renders each function's *verbatim source signature* (comments included) for
exactly this reason.

```python
def compute_mu(
        left_tt_cores:  typ.Union[typ.Sequence[NDArray], NDArray],  # len=d-1, elm_shape=C+(rLi,nUi,rL(i+1))
        xis:            typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=W+C+(nUi,)
) -> typ.Union[
    typ.Sequence[NDArray],  # mus. len=d, elm_shape=W+C+(rLi,)
    NDArray,
]:
```

## The decode table

| you see | it means |
|---|---|
| `# len=d, elm_shape=C+(ri,ni,r(i+1))` | a length-`d` tuple of arrays, each shaped `C + (ri, ni, r(i+1))` |
| `# shape=W+(Ni,)` | a bare array (no tuple wrapper) of that shape |
| `C`, `W`, `K` | the three batch **blocks**: frame/core stack, probe stack, tangent stack — the same letters used in body-variable suffixes (`mu_WCa`) and grouped-contraction subscripts (`contract('WCa,Caib,WCi->WCb', ...)`). Full legend: [`batching_and_stacking.md`](batching_and_stacking.md) |
| lowercase letters (`ri`, `nUi`, `Ni`) | single axes (ranks, mode sizes); a leading `d` is the stacked core / derivative axis |
| `# 1 <= max_rank <= min(N,M)`, `# requires unstacked`, `# 'left' \| 'right'` | a constraint or role note — a contract the type can't carry |
| `# HOST bool, static` | the array **must be host numpy, never jax/traced** — it is static structure (the uniform masks; runtime-enforced by the mask guard). **No tag is the default**: dtype-agnostic and numpy-or-jax polymorphic |

Two reading rules that make the system trustworthy:

- **The presence of a `#` is itself information.** Whatever Python *can* express is in the real
  annotation (`Optional[float]`, `int | Sequence[int] | None`, …); the comment appears only when
  there is a contract the type system cannot capture. No comment on a trivial scalar means there is
  nothing more to know.
- **One vocabulary everywhere.** The signature letters, the body-local name suffixes, and the
  grouped-contraction subscripts all use the same shape vocabulary — learn it once from the
  [batching legend](batching_and_stacking.md) and every signature in the library reads the same way.

(The authoring side of this convention — alignment mechanics, grouping rules, when to deviate — is
contributor documentation: [`contributor/signature_style.md`](contributor/signature_style.md).)
