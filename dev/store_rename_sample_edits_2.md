# Store rename: ten more sample edits (batch 2), and the least-determined one

Purpose: a second test of `dev/store_rename_plan_2026-11-10.md` (draft 16) on real code, chosen to differ from batch 1
(`dev/store_rename_sample_edits.md`) and from each other, and to sit where a literal implementer is most likely to go
wrong. Nothing in the tree was modified.

How to read it (as batch 1):

- **BEFORE** is verbatim from the working tree. `t3toolbox/`, `tests/` and `examples/` equal `v2026.4.0`
  (`git diff --stat v2026.4.0` touches only `docs/glossary.md` and `docs/index.rst`), so the line numbers are the plan's.
- **AFTER** is the final text once every slice touching the lines has run; the slice is named. Where the literal plan
  and the right answer differ, both are shown and labelled.
- **Governs** cites the plan. **What could go wrong** names the slip, what in the plan prevents it, and says so when
  nothing does.

| # | site | mechanism | does the plan determine it? |
|---|---|---|---|
| 1 | `UT3Variations` structure tuple; `ufv_apply_frame_masks` | A3, B1, A2; physical order | names yes; the mask locals `UM_k`/`UM_t`/`DM_t` and `WU`/`WD` no (they read backwards) |
| 2 | `_SEED_TAGS`, its test copy, two docs | B2, tier E, R5 | yes |
| 3 | `_deta_tilde_tangent_jets_step` (`D`) vs its oracle twin (`B`) | §1.2a, tier C | NAMEs yes (on a reason the call graph refutes); `st['dU']`, `:262`, `(Ud, Gd)` no; the C4 prose no (§11) |
| 4 | `_apply_transpose_adjoint` (`dD`) vs `_apply_corewise_hessian_adjoint` (`dB`) | §1.2a | NAMEs yes; prose letters only by the reviewer |
| 5 | `_apply_transpose_assemble_core`: σ̂ `sh` and the shift `sh` | D-1, σ̂ row, tier D | yes, but the σ̂ row also renames a shape tuple |
| 6 | `compute_sigma_hat`: `up-index` → `upward` beside `nUi` → `nDi` | §3.3, A3, `Q→R` | yes |
| 7 | `shared_t3_math.tex` `K_i`; `ttm_t3m_ht_note.tex` `X/Y/Z` | §4.8 | partly: the `K_i` range is one line too wide, `F` is not free, a new `B_t` collision |
| 8 | `ufv_padded_frame`'s mask message and its test | B2, §4.6, A3 scoped, A2 | yes |
| 9 | HISTORICAL comments and `_down_message` | §1.5 item 6, tier E | partly: inv 1's row keeps `U+`; `padded_frames.py`'s `kU`/`kD` comments unmapped |
| 10 | `utv_corewise_lift` doctest, `nU=4, nD=4` | A3, K15, bijection | yes; `(U, P)` → `(D, L)` meets the lift `L` |

---

## 1. Uniform layer: the structure tuple keeps its positions; the kernel's mask locals do not follow

**Where:** `t3toolbox/uniform_frame_variations_format.py:574–587` (the `UT3Variations` doctest) and `:622–632`
(`uniform_structure`); its positional consumer `tests/test_uniform_weighted.py:604–605`; the kernel
`t3toolbox/backend/ufv_masking.py:126–138` and `:44–50`.

**Governs:** A3 (`nU`↔`nD` on NAMEs and comments, doctest examples included), B1 (the doctest locals `up`/`down`), A1/A2
(the supercore names, `up_mask`↔`down_mask`), §0 "Physical order is unchanged", D-8, §2 (silent width properties),
K15 and the bijection. No row of §1.2 maps `UM_k`, `UM_t`, `DM_t`, or `ufv_masking.py`'s `WU`/`WD`.

**BEFORE**

*`t3toolbox/uniform_frame_variations_format.py:574–587`*
~~~python
    >>> d, N, nU, nD, rL, rR = 3, 6, 4, 5, 3, 2
    >>> tkv = np.random.randn(d, nD, N)
    >>> ttv = np.random.randn(d, rL, nU, rR)
    >>> shape = (4, 5, 6)
    >>> up    = np.arange(nU) < np.array([[2],[3],[4]])       # (d, nU)
    >>> down  = np.arange(nD) < np.array([[3],[4],[5]])       # (d, nD)
    >>> left  = np.arange(rL) < np.array([[1],[2],[3]])       # (d, rL) -- note d, not d+1
    >>> right = np.arange(rR) < np.array([[1],[2],[2]])       # (d, rR)
    >>> masks = ubcf.UT3VariationsMasks(up, down, left, right)
    >>> V = ubcf.UT3Variations(tkv, ttv, shape, masks)
    >>> V.uniform_structure
    (3, 6, 4, 5, 3, 2, ())
    >>> V.uniform_variation_shapes
    ((3, 5, 6), (3, 3, 4, 2))
~~~

**AFTER (R2; the `=` columns realigned, see the last bullet below)**
~~~python
    >>> d, N, nD, nU, rL, rR = 3, 6, 4, 5, 3, 2
    >>> tkv = np.random.randn(d, nU, N)
    >>> ttv = np.random.randn(d, rL, nD, rR)
    >>> shape = (4, 5, 6)
    >>> down  = np.arange(nD) < np.array([[2],[3],[4]])       # (d, nD)
    >>> up    = np.arange(nU) < np.array([[3],[4],[5]])       # (d, nU)
    >>> left  = np.arange(rL) < np.array([[1],[2],[3]])       # (d, rL) -- note d, not d+1
    >>> right = np.arange(rR) < np.array([[1],[2],[2]])       # (d, rR)
    >>> masks = ubcf.UT3VariationsMasks(down, up, left, right)
    >>> V = ubcf.UT3Variations(tkv, ttv, shape, masks)
    >>> V.uniform_structure
    (3, 6, 4, 5, 3, 2, ())
    >>> V.uniform_variation_shapes
    ((3, 5, 6), (3, 3, 4, 2))
~~~

The printed tuple is unchanged and now means `(d, N, nD, nU, rL, rR, C)`: position 2 is still the Tucker-side width
(4), position 3 the TT-shaped one (5). The names moved; the slots did not.

*`t3toolbox/uniform_frame_variations_format.py:622–632`* — BEFORE, then AFTER (R2):
~~~python
    def uniform_structure(self) -> typ.Tuple[
        int, # d
        int, # N
        int, # nU
        int, # nD
        int, # rL
        int, # rR
        typ.Tuple[int,...], # stack_shape
    ]:
        return self.d, self.N, self.nU, self.nD, self.rL, self.rR, self.stack_shape
~~~
~~~python
    def uniform_structure(self) -> typ.Tuple[
        int, # d
        int, # N
        int, # nD
        int, # nU
        int, # rL
        int, # rR
        typ.Tuple[int,...], # stack_shape
    ]:
        return self.d, self.N, self.nD, self.nU, self.rL, self.rR, self.stack_shape
~~~
(The properties above it swap names in place: `def nD` now returns `tt_variations.shape[-2]`, `def nU`
`tucker_variations.shape[-2]`, which is D-8: the TT variations' middle leg is `nD`, the Tucker variations' rows `nU`.)

*`tests/test_uniform_weighted.py:604–605`* — BEFORE, then AFTER (R2):
~~~python
    _, _, nU, nD, rL, rR, _ = uv.variations.uniform_structure
    return uv, ubvf.UT3FrameWeights.from_t3frameweights(W, nU=nU, nD=nD, rL=rL, rR=rR)
~~~
~~~python
    _, _, nD, nU, rL, rR, _ = uv.variations.uniform_structure
    return uv, ubvf.UT3FrameWeights.from_t3frameweights(W, nD=nD, nU=nU, rL=rL, rR=rR)
~~~

*`t3toolbox/backend/ufv_masking.py:126–138`*
~~~python
    SM_k = shape_mask.reshape(           (d,) + (1,)*len(ss) + (1,)  + (N,))
    UM_k = up_mask.reshape(              (d,) + ss           + (nU,) + (1,))
    UM_t = up_mask.reshape(              (d,) + ss           + (1,)  + (nU,) + (1,))
    DM_t = down_mask.reshape(            (d,) + ss           + (1,)  + (nD,) + (1,))
    LM_l = frame_left_mask[:-1].reshape( (d,) + ss           + (rL,) + (1,)  + (1,))
    LM_r = frame_left_mask[1:].reshape(  (d,) + ss           + (1,)  + (1,)  + (rL,))
    RM_l = frame_right_mask[:-1].reshape((d,) + ss           + (rR,) + (1,)  + (1,))
    RM_r = frame_right_mask[1:].reshape( (d,) + ss           + (1,)  + (1,)  + (rR,))

    masked_up_tucker_supercore = up_tucker_supercore * (SM_k * UM_k)
    masked_down_tt_supercore   = down_tt_supercore   * (LM_l * DM_t * RM_r)
    masked_left_tt_supercore   = left_tt_supercore   * (LM_l * UM_t * LM_r)
    masked_right_tt_supercore  = right_tt_supercore  * (RM_l * UM_t * RM_r)
~~~

**AFTER, literal plan (R2; nothing maps the three mask locals)**
~~~python
    SM_k = shape_mask.reshape(           (d,) + (1,)*len(ss) + (1,)  + (N,))
    UM_k = down_mask.reshape(              (d,) + ss           + (nD,) + (1,))
    UM_t = down_mask.reshape(              (d,) + ss           + (1,)  + (nD,) + (1,))
    DM_t = up_mask.reshape(            (d,) + ss           + (1,)  + (nU,) + (1,))
    ...
    masked_down_tucker_supercore = down_tucker_supercore * (SM_k * UM_k)
    masked_up_tt_supercore   = up_tt_supercore   * (LM_l * DM_t * RM_r)
    masked_left_tt_supercore   = left_tt_supercore   * (LM_l * UM_t * LM_r)
    masked_right_tt_supercore  = right_tt_supercore  * (RM_l * UM_t * RM_r)
~~~

**AFTER, recommended (needs a tier C row: `ufv_masking.py: UM_k→DM_k, UM_t↔DM_t`; realigned)**
~~~python
    SM_k = shape_mask.reshape(           (d,) + (1,)*len(ss) + (1,)  + (N,))
    DM_k = down_mask.reshape(            (d,) + ss           + (nD,) + (1,))
    DM_t = down_mask.reshape(            (d,) + ss           + (1,)  + (nD,) + (1,))
    UM_t = up_mask.reshape(              (d,) + ss           + (1,)  + (nU,) + (1,))
    ...
    masked_down_tucker_supercore = down_tucker_supercore * (SM_k * DM_k)
    masked_up_tt_supercore       = up_tt_supercore       * (LM_l * UM_t * RM_r)
    masked_left_tt_supercore     = left_tt_supercore     * (LM_l * DM_t * LM_r)
    masked_right_tt_supercore    = right_tt_supercore    * (RM_l * DM_t * RM_r)
~~~
The recommended form is self-checking against D-8: the left and right cores' middle legs carry the down mask.

*`t3toolbox/backend/ufv_masking.py:45–50`*
~~~python
    int,   # WU = min(n, N)
    int,   # WD = min(n, N, r*r)
    int,   # WL = r
    int,   # WR = r
]:
    """The padded widths ``(nU, nD, rL, rR)`` of the frame ``ut3_orthogonal_representations`` gives a point padded to
~~~
**AFTER, literal plan (R2):** the docstring becomes ``(nD, nU, rL, rR)`` (A3) and the return comments still say
`WU = min(n, N)`, `WD = min(n, N, r*r)`. **Recommended:** `WD = min(n, N)`, `WU = min(n, N, r*r)`.

**What could go wrong.**

- **The mask locals read backwards, and the plan cannot fix them.** Literally applied, `UM_k` holds the down mask and
  `DM_t` the up mask, and the left/right cores are masked by "`UM_t`" on their Tucker leg, the opposite of D-8. No gate
  reads local mask names. §3.4's widened scan (every NAME with a `U` letter segment) surfaces `UM_k` and `UM_t`, but
  not `DM_t`, and R1's freeness check would then reject `UM_t→DM_t` because `DM_t` is bound. The swap needs its own
  row, as the frontend already has (`UM_good↔DM_good`).
- **`WU`/`WD` contradict the docstring above them.** A3's scoped `WU↔WD` covers only the uniform padded-frame files.
  Both doctests print equal first and second widths (`(5, 5, 3, 3)`, `(4, 4, 2, 2)`), so nothing can tell. Add
  `ufv_masking.py` (comments) to that scoped swap.
- **Restoring the familiar order.** Writing `return self.d, self.N, self.nU, self.nD, …` after the rename changes
  what position 2 holds; `test_uniform_weighted.py:604` would then pad the weights with swapped widths. Caught: K15
  (positional NAMEs), and this doctest, whose widths differ (4 vs 5).
- **Ragged columns.** In-place swaps leave `down    =`/`up  =` here and `.reshape(` columns ±2 off in the kernel;
  §4.10 realigns signature blocks only. Cosmetic; the plan should say whether R4a realigns bodies and doctest examples.

---

## 2. `_SEED_TAGS`: the words swap, the integers stay with the slot, and the test keeps its own copy

**Where:** `t3toolbox/backend/padded_frames.py:139, 143, 148, 167, 197`; `tests/backend/test_padded_frame.py:91–92,
132–141`; `docs/contributor/padded_frame_internals.md:86`; `docs/padded_frame_cone.tex:339`.

**Governs:** B2 (the keys, `_seed_rows` arguments, the test copy: "The `_SEED_TAGS` integers stay with the physical
store"); B1 (test locals `up`/`down`); tier C (`padded_frames.py` and `test_padded_frame.py:135–139`: `U0→D0`); A3;
tier E (inv 2 `:143, 148, 164, 167, 171`, inv 1 `:91`); R5 for the two docs; battery B7.

**BEFORE**

*`t3toolbox/backend/padded_frames.py:139–151`*
~~~python
_SEED_TAGS = {'up': 0, 'down': 1, 'left': 2, 'right': 3}   # the store-kind part of a seed stream's key


def _seed_rows(
        store:  str,   # 'up' | 'down' | 'left' | 'right'
        key:    int,   # the core's mode; for a Tucker factor, its sharing group's first mode
        seed:   int,
        w:      int,   # the old width: the slots already filled
        k:      int,   # the number of new slots
        p:      int,   # the seed length: the mode size (up) or the real rows of the core's unfolding
~~~
*`:167`* and *`:197`*
~~~python
        store:   str,                         # 'down' | 'left' | 'right'
    seeds_Nk = _seed_rows('up', key, seed, s, n - s, N).T
~~~
*`tests/backend/test_padded_frame.py:91–92`* and *`:135–136`*
~~~python
    """Rows w .. w+k-1 of the keyed stream RandomState([seed, store tag, key]) (tags: up 0, down 1, left 2, right 3)."""
    tag = {'up': 0, 'down': 1, 'left': 2, 'right': 3}[store]
        U0 = up0[f]
        new, piv = seeded_qr(U0.T, stream_rows('up', f, seed, U0.shape[0], nU[f] - U0.shape[0], shape[f]).T)
~~~
*`docs/contributor/padded_frame_internals.md:86`* and *`docs/padded_frame_cone.tex:339`*
~~~text
`RandomState([seed, store tag, key])` stream (`_seed_rows`; tags up 0, down 1, left 2, right 3; the key is the
stream, \code{RandomState([seed, store tag, key])} (tags up 0, down 1, left 2, right 3; the key is the mode, or a
~~~

**AFTER** (R2 for the STRING tokens and NAMEs; R4a for the comments and test docstring; R5 for the docs)
~~~python
_SEED_TAGS = {'down': 0, 'up': 1, 'left': 2, 'right': 3}   # the store-kind part of a seed stream's key
        store:  str,   # 'down' | 'up' | 'left' | 'right'
        p:      int,   # the seed length: the mode size (down) or the real rows of the core's unfolding
        store:   str,                         # 'up' | 'left' | 'right'
    seeds_Nk = _seed_rows('down', key, seed, s, n - s, N).T
~~~
~~~python
    """Rows w .. w+k-1 of the keyed stream RandomState([seed, store tag, key]) (tags: down 0, up 1, left 2, right 3)."""
    tag = {'down': 0, 'up': 1, 'left': 2, 'right': 3}[store]
        D0 = down0[f]
        new, piv = seeded_qr(D0.T, stream_rows('down', f, seed, D0.shape[0], nD[f] - D0.shape[0], shape[f]).T)
~~~
~~~text
`RandomState([seed, store tag, key])` stream (`_seed_rows`; tags down 0, up 1, left 2, right 3; the key is the
stream, \code{RandomState([seed, store tag, key])} (tags down 0, up 1, left 2, right 3; the key is the mode, or a
~~~
Also on these lines' neighbours: `:164`/`:171` "an old TT or down core" → "an old TT or up core"; the test's
`up = []` / `down = [...]` locals (`:132`, `:141`) swap by B1.

The integers are the slot: tag 0 is physical slot 0 (the Tucker factor) before and after. Because the physical order
now reads `(down, up, left, right)`, the swapped map happens to read in order, as the old one did.

**What could go wrong.**

- **Moving the integers with the words** ("up keeps 0"). Every Tucker completion then draws from stream 1, and every
  padded frame changes. If the test copy gets the same edit, the suite stays green: the test only checks that the two
  copies agree (inv 1 §1.2). The battery's B7, bitwise against `v2026.4.0` and part of R2's gate, is the only net.
- **Editing one copy.** The test fails at R2. Good.
- **The comment at `:143` is not a STRING.** `# 'up' | 'down' | …` is a COMMENT token, so B2 skips it and K9 catches
  it at R4a. §1.1's example cites `padded_frames.py:143` with `:167`'s text (`'down' | 'left' | 'right'`); rows are
  located by content, so this is harmless, but the citation should be fixed.
- **The prose tag lists** need a word swap with the numbers kept ("down 0, up 1"), not a renumbering.

---

## 3. B vs D on one object: the kernel calls the corewise frame's down slot `D`, its oracle twin calls the point's cores `B`

**Where:** the kernel `t3toolbox/backend/sampling_derivatives.py:2482–2498`; its oracle twin and caller
`tests/backend/_curvature_oracles.py:261–263, 305–307, 397–400, 416, 423`; the prose pair `apply.py:320–322` /
`utv_operations.py:381–382, 391`.

**Governs:** D-12 and §1.2a (D forms: `_deta_tilde_tangent_jets_step`, "`Ud→Dd`, beside `U→D`"; B forms:
`deta_tilde_tangent_jets_trs` and `probe_derivatives_two_hole_states_trs`, `Ud→Bd`); tier C (`sampling_derivatives.py`:
`U→D, Ud→Dd`); tier C's B rows (`_curvature_oracles.py`: `U→B` in `probe_derivatives_two_hole_states_trs`); A1, A3;
tier D (`(U, G, G, G)`); the "Frame-generic kernels" bullet; K8, K15.

**The rule** (§1.2a): a name takes the letter of the object it moves. A point's own core or its direction → `B`
forms; a frame's down core or its variation → `D` forms; a kernel serving both geometries is frame-generic and named
for the frame's down store.

**(a) The kernel — `D`.** *`sampling_derivatives.py:2482–2498`*
~~~python
def _deta_tilde_tangent_jets_step(
        data: typ.Tuple[NDArray, NDArray, NDArray],  # (U, Udot, ztilde) for one core
) -> typ.Tuple[NDArray]:                             # (deta_tilde_dot,) elm_shape=(order+1,)+W+K+C+(nUi,)
    '''One core of :py:func:`compute_deta_tilde_tangent_jets` (ragged path only). The C-only ``U`` is
    passed only to recover ``len(C)``. Closure-free map body -- ``docs/contributor/scan_body_principles.md``.'''
    U, Ud, zt = data
    return (contractions.contract('tWCo,KCio->tWKCi', zt, Ud, len_C=U.ndim - 2),)


def compute_deta_tilde_tangent_jets(
        up_tucker_cores:        typ.Sequence[NDArray],  # U.    len=d, elm_shape=C+(nUi,Ni)
        up_tucker_cores_dot:    typ.Sequence[NDArray],  # Udot. len=d, elm_shape=K+C+(nUi,Ni)
        ztildes:                typ.Sequence[NDArray],  # residual jets, len=d, elm_shape=(order+1,)+W+C+(Ni,)
) -> typ.Tuple[NDArray, ...]:                           # deta_tildes_dot. len=d, elm_shape=(order+1,)+W+K+C+(nUi,)
    '''The derivative of :py:func:`compute_deta_tilde_jets` along the Tucker directions:
    ``deta_tilde_dot_i = Udot_i z~_i`` (the residual is fixed, so only the core moves). The order axis
    rides through; the tangent stack ``K`` comes from ``Udot``.'''
~~~
**AFTER** (R2 NAMEs and widths; R4b comments, the `Udot`s forced by K8, letter by the function's D form)
~~~python
def _deta_tilde_tangent_jets_step(
        data: typ.Tuple[NDArray, NDArray, NDArray],  # (D, Ddot, ztilde) for one core
) -> typ.Tuple[NDArray]:                             # (deta_tilde_dot,) elm_shape=(order+1,)+W+K+C+(nDi,)
    '''One core of :py:func:`compute_deta_tilde_tangent_jets` (ragged path only). The C-only ``D`` is
    passed only to recover ``len(C)``. Closure-free map body -- ``docs/contributor/scan_body_principles.md``.'''
    D, Dd, zt = data
    return (contractions.contract('tWCo,KCio->tWKCi', zt, Dd, len_C=D.ndim - 2),)


def compute_deta_tilde_tangent_jets(
        down_tucker_cores:      typ.Sequence[NDArray],  # D.    len=d, elm_shape=C+(nDi,Ni)
        down_tucker_cores_dot:  typ.Sequence[NDArray],  # Ddot. len=d, elm_shape=K+C+(nDi,Ni)
        ztildes:                typ.Sequence[NDArray],  # residual jets, len=d, elm_shape=(order+1,)+W+C+(Ni,)
) -> typ.Tuple[NDArray, ...]:                           # deta_tildes_dot. len=d, elm_shape=(order+1,)+W+K+C+(nDi,)
    '''The derivative of :py:func:`compute_deta_tilde_jets` along the Tucker directions:
    ``deta_tilde_dot_i = Ddot_i z~_i`` (the residual is fixed, so only the core moves). The order axis
    rides through; the tangent stack ``K`` comes from ``Ddot``.'''
~~~

**(b) The oracle — `B`.** *`_curvature_oracles.py:261–263, 305–307, 397–400, 416, 423`*
~~~python
# Notation: a trailing ``d`` is the derivative along theta_hat (it carries the tangent stack K); the undotted
# operands carry W + C only (the residual jets z~ have no K). theta = (U, G) is the corewise point, so its
# corewise frame is (U, G, G, G): every frame core of the GN transpose is G and its derivative is Gd.

def deta_tilde_tangent_jets_trs(Ud, ztildes, n_frame):
    """Reference twin of ``sd.compute_deta_tilde_tangent_jets``: ``Udot_i z~_i`` per core."""
    return tuple(contract('tWCo,KCio->tWKCi', z, u, len_C=n_frame) for z, u in zip(ztildes, Ud))

    tangent forward at ``Q = O = P = G`` along ``(Ud, Gd)``. Every jet kernel here is a dense ``*_trs`` twin."""
    U, G = core_pair
    Ud, Gd = core_directions
    n_frame = U[0].ndim - 2
    st['deta_td'] = deta_tilde_tangent_jets_trs(Ud, ztildes, n_frame)
    st['dU'] = tucker_tangent_assembly_trs(ztildes, st['dxi_td'], ww, pp, st['etad'], n_frame)
~~~
**AFTER, recommended** (R2 NAMEs; R4b prose)
~~~python
# Notation: a trailing ``d`` is the derivative along theta_hat (it carries the tangent stack K); the undotted
# operands carry W + C only (the residual jets z~ have no K). theta = (B, G) is the corewise point, so its
# corewise frame is (D, G, G, G): every frame core of the GN transpose is G and its derivative is Gd.

def deta_tilde_tangent_jets_trs(Bd, ztildes, n_frame):
    """Reference twin of ``sd.compute_deta_tilde_tangent_jets``: ``Bdot_i z~_i`` per core."""
    return tuple(contract('tWCo,KCio->tWKCi', z, u, len_C=n_frame) for z, u in zip(ztildes, Bd))

    tangent forward at ``R = U = L = G`` along ``(Bd, Gd)``. Every jet kernel here is a dense ``*_trs`` twin."""
    B, G = core_pair
    Bd, Gd = core_directions
    n_frame = B[0].ndim - 2
    st['deta_td'] = deta_tilde_tangent_jets_trs(Bd, ztildes, n_frame)
    st['dB'] = tucker_tangent_assembly_trs(ztildes, st['dxi_td'], ww, pp, st['etad'], n_frame)
~~~
The literal plan differs at three places: `:262` keeps `theta = (U, G)`, `:397` keeps `(Ud, Gd)`, and `:423` keeps
`st['dU']` (reasons below). The same arrays are `B`/`Bd` here and
`D`/`Dd` inside the kernel, and the twin's docstring now says `Bdot_i z~_i` where the kernel's says `Ddot_i z~_i`.
The comprehension `u` stays: §1.2a lists `u→b` only for other functions.

**(c) The prose that names the slot's content** (`apply.py:320–322`, `utv_operations.py:381–382, 391`): see §11.

**What could go wrong.**

- **The plan's reason for (a) is false.** §1.2a says these kernels "serve the manifold tangent forward and the corewise
  Hessians alike". `compute_deta_tilde_tangent_jets` has one library caller, the corewise Hessian
  (`sampling_derivatives.py:2997`), and probing's `compute_deta_tilde_tangent` likewise (`probing.py:1729`); only
  `_tangent_z_jets_step` (`assemble_tangent_z_jets`, `z = U deta + dU eta`) is the manifold forward. A reviewer who
  applies the stated test gets `B, Bd` and "fixes" the kernel. The enumerated row governs, and the kernel's A1 keywords
  (`down_tucker_cores[_dot]`) support `D`, but §1.2a should give that as the reason.
- **`:262` stays `theta = (U, G)`.** inv 1 files it under App. B (keep), and §3.4 scans NAMEs and the U family, not a
  bare `U` in a comment. Tier D converts `(U, G, G, G)` on the next line. The sentence then calls θ's Tucker cores the
  up core. No gate catches a bare `U`.
- **`(Ud, Gd)` in prose escapes K8.** K8's code-name pattern is `Udot\w*|dU…|gU|Ug`; it has no `Ud` (probably because
  `o1_ragged.py`'s "u dense" `Ud` stays). So `:397` and `:471` keep `Ud` after the NAME became `Bd`.
- **`st['dU']` cannot pass R4b.** The key is a STRING at nine sites (`_curvature_oracles.py:423, 464, 543`;
  `test_curvature.py:740, 751, 756, 1328, 1335, 1338`). K8's `dU(?![a-z])` matches it, no row maps it (B4 and B5 are
  scoped to other files), and the brief forbids renaming it without asking. Add a scoped string row `'dU'→'dB'` for
  the two files, written and read together like B5.

---

## 4. `dD` and `dB` from the same contraction, one function apart

**Where:** `t3toolbox/backend/apply.py:460–467, 492–503` (`_apply_transpose_adjoint`) and `:506–516, 536–545`
(`_apply_corewise_hessian_adjoint`).

**Governs:** §1.2a (D: `_apply_transpose_adjoint`, `dU_tildes→dD_tildes`; B: `_apply_corewise_hessian_adjoint`,
`dU→dB`); tier C `apply.py` (`O→U`); the σ̂ row (`sh`, `shd`); A1 (`down_tt_cores`); A3; tier D; §4A (the `:465–467`
equation block is a candidate); K8, K15.

**The test that decides:** what the array moves, read off the call graph. `_apply_transpose_adjoint` is reached from
`tv_apply_transpose_from_sweep` (`apply.py:296`) and `entries.tv_entries_transpose_from_sweep` (`entries.py:239`), so
from the manifold geometry and from `t3_apply_corewise_transpose`, which is `tv_apply_transpose` at
`t3_corewise_frame(core_pair)` (`apply.py:329`). It is frame-generic, so `D`. `_apply_corewise_hessian_adjoint` is
reached only from the corewise Hessians (`apply.py:362`, `entries.py:293`), so `B`.

**BEFORE**

*`apply.py:460–467`*
~~~python
def _apply_transpose_adjoint(c, ww, xis, mus, down_tt_cores, right_tt_cores, sum_over_probes):
    '''Adjoint-state assembly shared by apply/tv_entries_transpose_from_sweep -- the **low-memory,
    K-aware** transpose (replaces the old scatter). The scalar residual ``c`` seeds one reverse
    ``sigma_hat`` sweep (recomputing the right context, so ``nu``/``eta`` are never stored); then

        dxi-hat_k  = mu-hat_{k-1} . O_k . sigma-hat_k                 # over the down mode nO
        dG-tilde_k = mu-hat_{k-1} (x) xi-hat_k (x) sigma-hat_k        # over (rL, nU, rR)
        dU-tilde_k = dxi-hat_k (x) w_k                               # over (nO, N)
~~~
*`apply.py:492–503`*
~~~python
    n_probe = ww[0].ndim - 1
    dxi_hats = tuple(contractions.contract('WCa,Caib,WKCb->WKCi', mu, O, sh)        # dxi_hat = mu . O . sigma_hat
                     for mu, O, sh in zip(mus, down_tt_cores, sigma_hats))
    if sum_over_probes:
        dG_tildes = tuple(contractions.contract('WCa,WCi,WKCb->KCaib', mu, xi, sh, len_W=n_probe)
                          for mu, xi, sh in zip(mus, xis, sigma_hats))
        dU_tildes = tuple(contractions.contract('Wo,WKCa->KCao', w, dxh) for w, dxh in zip(ww, dxi_hats))
    else:
        dG_tildes = tuple(contractions.contract('WCa,WCi,WKCb->WKCaib', mu, xi, sh, len_W=n_probe)
                          for mu, xi, sh in zip(mus, xis, sigma_hats))
        dU_tildes = tuple(contractions.contract('Wo,WKCa->WKCao', w, dxh) for w, dxh in zip(ww, dxi_hats))
    return dU_tildes, dG_tildes   # (var_tucker, var_tt) = T3Variations.data
~~~
*`apply.py:506–508, 516`* and *`:536–545`*
~~~python
def _apply_corewise_hessian_adjoint(c, ww, xis, xis_dot, mus, mus_dot, tt_cores, tt_dirs):
    '''The derivative of :py:func:`_apply_transpose_adjoint` (``sum_over_probes=True``) at the corewise frame
    ``(U, G, G, G)`` along ``(Udot, Gdot)``, shared by :py:func:`t3_apply_corewise_hessian_from_sweep` and
        dU              = sum_W  w (x) dxi_hat_dot
    n_probe = ww[0].ndim - 1
    dU = tuple(contractions.contract('Wo,WKCa->KCao', w, dxd) for w, dxd in zip(ww, dxi_hats_dot))
    dG = []
    for mu, mud, xi, xid, sh, shd in zip(mus, mus_dot, xis, xis_dot, sigma_hats, sigma_hats_dot):
        E = contractions.contract('WCi,WCb->WCib', xi, sh)
        E_dot = (contractions.contract('WKCi,WCb->WKCib', xid, sh, len_W=n_probe)
                 + contractions.contract('WCi,WKCb->WKCib', xi, shd, len_W=n_probe))
        dG.append(contractions.contract('WKCa,WCib->KCaib', mud, E, len_W=n_probe)
                  + contractions.contract('WCa,WKCib->KCaib', mu, E_dot, len_W=n_probe))
    return dU, tuple(dG)
~~~

**AFTER** (R2 NAMEs and widths outside a fence; R4b the docstrings and, if fenced, `:465–467` by hand)
~~~python
def _apply_transpose_adjoint(c, ww, xis, mus, up_tt_cores, right_tt_cores, sum_over_probes):
    ...
        dxi-hat_k  = mu-hat_{k-1} . U_k . sigma-hat_k                 # over the up mode nU
        dG-tilde_k = mu-hat_{k-1} (x) xi-hat_k (x) sigma-hat_k        # over (rL, nD, rR)
        dD-tilde_k = dxi-hat_k (x) w_k                               # over (nU, N)
~~~
~~~python
    n_probe = ww[0].ndim - 1
    dxi_hats = tuple(contractions.contract('WCa,Caib,WKCb->WKCi', mu, U, sig_hat)        # dxi_hat = mu . U . sigma_hat
                     for mu, U, sig_hat in zip(mus, up_tt_cores, sigma_hats))
    if sum_over_probes:
        dG_tildes = tuple(contractions.contract('WCa,WCi,WKCb->KCaib', mu, xi, sig_hat, len_W=n_probe)
                          for mu, xi, sig_hat in zip(mus, xis, sigma_hats))
        dD_tildes = tuple(contractions.contract('Wo,WKCa->KCao', w, dxh) for w, dxh in zip(ww, dxi_hats))
    else:
        dG_tildes = tuple(contractions.contract('WCa,WCi,WKCb->WKCaib', mu, xi, sig_hat, len_W=n_probe)
                          for mu, xi, sig_hat in zip(mus, xis, sigma_hats))
        dD_tildes = tuple(contractions.contract('Wo,WKCa->WKCao', w, dxh) for w, dxh in zip(ww, dxi_hats))
    return dD_tildes, dG_tildes   # (var_tucker, var_tt) = T3Variations.data
~~~
~~~python
def _apply_corewise_hessian_adjoint(c, ww, xis, xis_dot, mus, mus_dot, tt_cores, tt_dirs):
    '''The derivative of :py:func:`_apply_transpose_adjoint` (``sum_over_probes=True``) at the corewise frame
    ``(D, G, G, G)`` along ``(Bdot, Gdot)``, shared by :py:func:`t3_apply_corewise_hessian_from_sweep` and
        dB              = sum_W  w (x) dxi_hat_dot
    n_probe = ww[0].ndim - 1
    dB = tuple(contractions.contract('Wo,WKCa->KCao', w, dxd) for w, dxd in zip(ww, dxi_hats_dot))
    dG = []
    for mu, mud, xi, xid, sig_hat, sig_hat_d in zip(mus, mus_dot, xis, xis_dot, sigma_hats, sigma_hats_dot):
        E = contractions.contract('WCi,WCb->WCib', xi, sig_hat)
        E_dot = (contractions.contract('WKCi,WCb->WKCib', xid, sig_hat, len_W=n_probe)
                 + contractions.contract('WCi,WKCb->WKCib', xi, sig_hat_d, len_W=n_probe))
        dG.append(contractions.contract('WKCa,WCib->KCaib', mud, E, len_W=n_probe)
                  + contractions.contract('WCa,WKCib->KCaib', mu, E_dot, len_W=n_probe))
    return dB, tuple(dG)
~~~
(The uniform branches, `:486, 489` and `:528, 534`, follow the same letters.) Every contraction string is
byte-identical (§1.5 item 6, K14); the longest changed line is 119 characters.

**What could go wrong.**

- **One contraction, two letters.** `'Wo,WKCa->KCao'` gives `dD_tildes` in one function and `dB` in the next, and the
  second's docstring calls itself the derivative of the first. A uniform choice is the natural slip. K15 catches it for
  NAMEs, because R1's hit table is built from §1.2a. It does not for prose: K8 forces `dU-tilde_k`, `(Udot, Gdot)` and
  `dU = sum_W` to change but cannot choose the letter. That is the reviewer's.
- **`(D, G, G, G)` along `(Bdot, Gdot)`** reads like a mismatch. It is §1.2a's "Where the two letters meet", but not
  one of its two listed sites. List it, or say once in the convention paragraph that at the corewise frame of
  θ = (B, G) the down slot holds `B`.
- **The equation block mixes a one-way and a swap.** `:465` and `:467` take `nO→nU`, `:466` takes `nU→nD`. Done in
  sequence, all three end as `nD`. Sentinels prevent it in the script; if R1 fences the block, the hand rewrite must
  get it right, and it is a K10b re-layout ("down mode" → "up mode" shortens the line).

---

## 5. The σ̂ local and the shift trade the spelling `sh` across one docstring and its code

**Where:** `t3toolbox/backend/sampling_derivatives.py:2153–2170` (`_apply_transpose_assemble_core`); its oracle twin
`tests/_jet_trs_twins.py:206–216`; and `sampling_derivatives.py:1689–1695` (`_flatten_w`).

**Governs:** D-1 FINAL and §3.2; the σ̂ row (`sh→sig_hat`, function-scoped); tier D (`[U^` → `[(shᵀ)^`); §1.5 item 8
(`_shift_up_pair` keeps its name); §1.2a (`_apply_transpose_assemble_core`: `dU→dD`, `dU_t→dD_t`; the twin:
`gU→gD`, `dU→dD`); tier C twin (`O→U`); K7, K14, K15.

**BEFORE** *`sampling_derivatives.py:2153–2170`*
~~~python
) -> typ.Tuple[NDArray, NDArray]:  # (dU_tilde, dG_tilde); W summed (K+C+...) or kept (W+K+C+...)
    '''One core of the gradient-assembly map of :py:func:`_apply_derivatives_transpose_from_jets`
    (ragged and uniform alike), shared by the two closure-free map bodies below --
    ``docs/contributor/scan_body_principles.md``. ``dG = sum_{r, s<2} C(r+s, r) mu^(r) (x) xi^(s) (x)
    sigma^^(r+s)`` is banded in the affine ``xi``: the pair ``sum_r mu^(r) (x) [U^k sigma^]^(r)``
    (:py:func:`_shift_up_pair`) meets ``xi^(k)``.'''
    xi, mu, sh, dxh, wj = x
    xnp, _, _ = get_backend(True, tree_contains_jax(x))      # only xnp
    s_size = min(2, mu.shape[0])    # the w / dxi input jets carry orders {0, 1}, capped at order
    n_probe = wj.ndim - 2           # len(W), from the W-only ambient jet w_jet ((2,)+W+(Ni,))
    if sum_over_probes:
        dG, dU = 'kWKCab,kWCi->KCaib', 'uWKCa,uWo->KCao'
    else:
        dG, dU = 'kWKCab,kWCi->WKCaib', 'uWKCa,uWo->WKCao'
    pair = contractions.contract('krWKCb,rWCa->kWKCab', _shift_up_pair(sh, xnp), mu, len_W=n_probe)
    dG_t = contractions.contract(dG, pair, xi, len_W=n_probe)
    dU_t = contractions.contract(dU, dxh[:s_size], wj[:s_size])
    return (dU_t, dG_t)
~~~

**AFTER** (R2 NAMEs; R4b the docstring shift letter and the return comment)
~~~python
) -> typ.Tuple[NDArray, NDArray]:  # (dD_tilde, dG_tilde); W summed (K+C+...) or kept (W+K+C+...)
    '''One core of the gradient-assembly map of :py:func:`_apply_derivatives_transpose_from_jets`
    (ragged and uniform alike), shared by the two closure-free map bodies below --
    ``docs/contributor/scan_body_principles.md``. ``dG = sum_{r, s<2} C(r+s, r) mu^(r) (x) xi^(s) (x)
    sigma^^(r+s)`` is banded in the affine ``xi``: the pair ``sum_r mu^(r) (x) [(shᵀ)^k sigma^]^(r)``
    (:py:func:`_shift_up_pair`) meets ``xi^(k)``.'''
    xi, mu, sig_hat, dxh, wj = x
    xnp, _, _ = get_backend(True, tree_contains_jax(x))      # only xnp
    s_size = min(2, mu.shape[0])    # the w / dxi input jets carry orders {0, 1}, capped at order
    n_probe = wj.ndim - 2           # len(W), from the W-only ambient jet w_jet ((2,)+W+(Ni,))
    if sum_over_probes:
        dG, dD = 'kWKCab,kWCi->KCaib', 'uWKCa,uWo->KCao'
    else:
        dG, dD = 'kWKCab,kWCi->WKCaib', 'uWKCa,uWo->WKCao'
    pair = contractions.contract('krWKCb,rWCa->kWKCab', _shift_up_pair(sig_hat, xnp), mu, len_W=n_probe)
    dG_t = contractions.contract(dG, pair, xi, len_W=n_probe)
    dD_t = contractions.contract(dD, dxh[:s_size], wj[:s_size])
    return (dD_t, dG_t)
~~~
Before, the docstring's `U` was the shift and the code's `sh` was σ̂; after, the docstring's `sh` is the shift and the
code's `sig_hat` is σ̂, five lines apart, with `_shift_up_pair` (the function that applies `shᵀ`) between them. `dU`
and `dG` here are NAMEs holding contraction specs: the NAME is on §1.2a's list, the strings are not touched.

*`tests/_jet_trs_twins.py:207, 211–216`* — BEFORE, then AFTER (R2; the docstring by R4b):
~~~python
    ``mu O sigma^`` (all rows) and the single-term assembly ``dG = mu (x) xi (x) sigma^``, ``dU = dxi^ (x) w``.
    gU, gG = ('KCao', 'KCaib') if sum_over_probes else ('WKCao', 'WKCaib')
    dU, dG = [], []
    for O, xi, mu, sh, wj in zip(down_tt_cores, xi_jets, mu_jets, sigma_hats, w_jets):
        dxh = contract('trs,rWCa,Caib,tWKCb->sWKCi', trs, mu, O, sh)
        dG.append(contract('trs,rWCa,sWCi,tWKCb->' + gG, trs[:, :, :s2], mu, xi[:s2], sh, len_W=n_probe))
        dU.append(contract('uWKCa,uWo->' + gU, dxh[:s2], wj[:s2]))
~~~
~~~python
    ``mu U sigma^`` (all rows) and the single-term assembly ``dG = mu (x) xi (x) sigma^``, ``dD = dxi^ (x) w``.
    gD, gG = ('KCao', 'KCaib') if sum_over_probes else ('WKCao', 'WKCaib')
    dD, dG = [], []
    for U, xi, mu, sig_hat, wj in zip(up_tt_cores, xi_jets, mu_jets, sigma_hats, w_jets):
        dxh = contract('trs,rWCa,Caib,tWKCb->sWKCi', trs, mu, U, sig_hat)
        dG.append(contract('trs,rWCa,sWCi,tWKCb->' + gG, trs[:, :, :s2], mu, xi[:s2], sig_hat, len_W=n_probe))
        dD.append(contract('uWKCa,uWo->' + gD, dxh[:s2], wj[:s2]))
~~~

*`sampling_derivatives.py:1689–1695`* (`_flatten_w`) — the literal plan renames this; it should not change:
~~~python
    if n_probe == 1:
        return tuple(ops), w_axes
    flat = []
    for o, w in zip(ops, w_axes):
        sh = tuple(o.shape)
        flat.append(o.reshape(sh[:w] + (math.prod(sh[w:w + n_probe]),) + sh[w + n_probe:]))
    return tuple(flat), w_axes
~~~

**What could go wrong.**

- **The plan renames a shape tuple.** The σ̂ row lists `sampling_derivatives.py` `sh` at "1693–1694", and §3.2 counts
  14 tokens there. Four of the 14 are `_flatten_w`'s `sh = tuple(o.shape)`. Applied literally:
  `sig_hat = tuple(o.shape)`. Nothing catches it: the row puts it in R1's hit table (K15), and no DO-NOT-TOUCH list
  names it (K14). Fix: drop 1693–1694 from the row (10 + 5 tokens), and list `_flatten_w`'s `sh` with the shape locals
  in §3.2.
- **Pattern completion on neighbours.** `sh_pair_j` (`:2190`) and `psh`/`pshd` (`:3220`) are not on the row;
  `test_curvature.py:1063`'s `sh` is a σ̂ local in a file the row does not list; `sharing.py:873`'s doctest
  `sh = (0, 0, 1)` is a sharing-label tuple. All stay. K15 catches a rename of any of them.
- **Following CHANGES draft 14**, which renamed the functions to `_shift_t_pair`: §1.5 item 8 and K14 forbid it, and
  the CHANGES header says history is not instruction.

---

## 6. `-ward` beside widths that flip, and beside ones that don't

**Where:** `t3toolbox/backend/probing.py:812–827` (`_sigma_hat_step`, `compute_sigma_hat`), `:346–350`
(`compute_dxi`), `:866–870` (`compute_deta_tilde`); `t3toolbox/backend/apply.py:215`.

**Governs:** §3.3's travel rows (`up-index` → `upward` at `probing.py:824`; `down eta` → `downward eta` at
`apply.py:215`), which override inv 2's keep/AMBIG rows (§1.5 item 9); "Directions already spelled as directions
stay"; tier C `probing.py` (`Q→R`); D-7 for `Q` in comments (§3.4); A1, A3; K9 and its converse.

**BEFORE** *`probing.py:812–827`*
~~~python
def _sigma_hat_step(
        carry: NDArray,                      # W+K+C+(rR(i+1),)
        data:  typ.Tuple[NDArray, NDArray],  # (Q, xi): C+(rRi,nUi,rR(i+1)) ; W+C+(nUi,)
) -> typ.Tuple[NDArray, typ.Tuple[NDArray]]:   # (next carry, (carry,))
    '''One edge of the adjoint reverse sweep of :py:func:`compute_sigma_hat`. Closure-free scan body
    -- ``docs/contributor/scan_body_principles.md``.'''
    Q, xi = data
    return contractions.contract('WKCa,Caib,WCi->WKCb', carry, Q, xi), (carry,)


def compute_sigma_hat(
        right_tt_cores: typ.Sequence[NDArray],  # Q.  len=d, elm_shape=C+(rRi,nUi,rR(i+1))
        xis:            typ.Sequence[NDArray],  # frame up-index edge vars, len=d, elm_shape=W+C+(nUi,)
        c:              NDArray,                # residual (scalar), shape=W+K+C
) -> typ.Sequence[NDArray]:                     # sigma_hats. len=d, elm_shape=W+K+C+(rR(i+1),)
    '''Propagation-only adjoint **reverse** sweep via ``Q``, seeded at the terminal bond by the residual
~~~
**AFTER** (R2 NAMEs and widths; R4b the travel word and the `Q`s in prose)
~~~python
def _sigma_hat_step(
        carry: NDArray,                      # W+K+C+(rR(i+1),)
        data:  typ.Tuple[NDArray, NDArray],  # (R, xi): C+(rRi,nDi,rR(i+1)) ; W+C+(nDi,)
) -> typ.Tuple[NDArray, typ.Tuple[NDArray]]:   # (next carry, (carry,))
    '''One edge of the adjoint reverse sweep of :py:func:`compute_sigma_hat`. Closure-free scan body
    -- ``docs/contributor/scan_body_principles.md``.'''
    R, xi = data
    return contractions.contract('WKCa,Caib,WCi->WKCb', carry, R, xi), (carry,)


def compute_sigma_hat(
        right_tt_cores: typ.Sequence[NDArray],  # R.  len=d, elm_shape=C+(rRi,nDi,rR(i+1))
        xis:            typ.Sequence[NDArray],  # frame upward edge vars, len=d, elm_shape=W+C+(nDi,)
        c:              NDArray,                # residual (scalar), shape=W+K+C
) -> typ.Sequence[NDArray]:                     # sigma_hats. len=d, elm_shape=W+K+C+(rR(i+1),)
    '''Propagation-only adjoint **reverse** sweep via ``R``, seeded at the terminal bond by the residual
~~~

*`probing.py:347, 349–350`* and *`:867, 869–870`* — BEFORE, then AFTER (R2). The words already spelled as directions
stay; only the widths and the A1 name change (realigned, the type column kept):
~~~python
        var_tucker_cores:       typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(nOi,Ni)
) -> typ.Union[typ.Sequence[NDArray], NDArray]: # dxis. len=d, elm_shape=(...,nOi)
    '''Compute var-upward edge variables dxi.

        up_tucker_cores:    typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=(nUi,Ni)
) -> typ.Union[typ.Sequence[NDArray], NDArray]: # len=d, elm_shape=(...,nUi)
    '''Adjoint-var-upward edge variables deta_tilde.
~~~
~~~python
        var_tucker_cores:       typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(nUi,Ni)
) -> typ.Union[typ.Sequence[NDArray], NDArray]: # dxis. len=d, elm_shape=(...,nUi)
    '''Compute var-upward edge variables dxi.

        down_tucker_cores:  typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=(nDi,Ni)
) -> typ.Union[typ.Sequence[NDArray], NDArray]: # len=d, elm_shape=(...,nDi)
    '''Adjoint-var-upward edge variables deta_tilde.
~~~

*`apply.py:215`* — BEFORE, then AFTER (R4b):
~~~text
    **Lean ``(xis, mus)`` only** (not the right ``nu`` / down ``eta`` sweeps): the all-modes apply
    **Lean ``(xis, mus)`` only** (not the right ``nu`` / downward ``eta`` sweeps): the all-modes apply
~~~

Three upward edge variables, two widths: `xi` and `deta_tilde` sit on the down store's width (`nD`), `dxi = V w` on
the up width (`nU`, the Tucker variations' rows). The travel word never predicts the width letter.

**What could go wrong.**

- **The store swap on a travel word.** Old "up" is new "down", so a word swap gives "down-index", which reads as a
  store and is backwards as a direction. §3.3's row prevents it. inv 2 files `:824` as AMBIG/keep and `apply.py:215`
  as "keep 'down' … generated text wrongly flips it", so an implementer reading the inventory alone keeps both; §1.5
  item 9's override is the only thing that says otherwise.
- **K9's converse would fail the right edit** unless §3.3's rows are registered as tier-E texts: both lines are keep
  rows, and the converse demands their bare `up`/`down` survive. The plan says so ("e.g. §3.3's rewordings"); R1's
  `keeps.txt` must carry the exemption.
- **Harmonizing the width with the word** (`upward … (nUi,)`): the width is script-made, so this is a hand edit
  outside the map. K15 reads NAMEs only and K13 does not recognise `xis`; the bijection count and the reviewer catch it.
- **Flipping the existing `-ward` words.** A pass that re-derives travel words from the new store names would turn
  "var-upward" into "var-downward". "Directions already spelled as directions stay" forbids it; K9 cannot see it
  (no bare word).

---

## 7. `.tex`: `K_i` frees `B_i` in the shared-factor note, but its range is one line too wide; `X/Y/Z` leaves a `B_t`

**Where:** `docs/shared_t3_math.tex:273–292, 304–306, 566`; `docs/ttm_t3m_ht_note.tex:85, 93, 217–218, 238–240`.

**Governs:** §4.8 (a T3's own factors → `B_i`, `B_g` in the point algorithms; blocks `B_i` → `K_i` "(:279–306)"; tied
coordinates `B` → `F`; "up-leg" → "Tucker leg"; in the ttm note `U_i` → `B_i` and the operand superscripts
`A`, `B`, `C` → `X`, `Y`, `Z`, "so `B` is not both a factor and an operand"); D-6; §4A(a) (every letter-bearing
equation line inventoried); R5 by hand; Nick on the PDFs.

**BEFORE** *`shared_t3_math.tex:273–292`* (the initialization algorithm, a point algorithm)
~~~tex
Input: T3 with orthonormal but distinct factors $U_i$, $i \in g$.
For each group $g$:
\begin{enumerate}[label=\emph{(\arabic*)},itemsep=1pt,topsep=2pt]
\item Stack factors and compute one tall QR:
  $[\,U_{i_1} | \cdots | U_{i_k}\,] = Q R$, with
  $Q \in \R^{N_g \times m}$, $m \leq \sum_{i\in g} n_i$.
\item Express each block $B_i \coloneqq U_i W_i$ (never formed at size
  $N_g$) in the $Q$ basis: $Q^\top B_i = R_{[i]} W_i$, where $R_{[i]}$ is the
  corresponding column block of $R$. SVD the small stacked matrix
  $[\,R_{[i_1]} W_{i_1} | \cdots | R_{[i_k]} W_{i_k}\,] \in
  \R^{m \times \sum_i r_{i-1} r_i}$; retain leading left singular vectors
  $Y$.
\item Set $U_g \coloneqq Q Y$; project each mode by applying
  $(U_g^\top U_i)$ to the up-leg of core $i$ and replacing the factor by
  $U_g$. Follow with Algorithm~\ref{alg:rounding}.
\end{enumerate}
\end{algorithm}

\begin{lemma}\label{lem:init}
The column space of $[\,B_{i_1} | \cdots | B_{i_k}\,]$ lies in
~~~
**AFTER (R5)**
~~~tex
Input: T3 with orthonormal but distinct factors $B_i$, $i \in g$.
For each group $g$:
\begin{enumerate}[label=\emph{(\arabic*)},itemsep=1pt,topsep=2pt]
\item Stack factors and compute one tall QR:
  $[\,B_{i_1} | \cdots | B_{i_k}\,] = Q R$, with
  $Q \in \R^{N_g \times m}$, $m \leq \sum_{i\in g} n_i$.
\item Express each block $K_i \coloneqq B_i W_i$ (never formed at size
  $N_g$) in the $Q$ basis: $Q^\top K_i = R_{[i]} W_i$, where $R_{[i]}$ is the
  corresponding column block of $R$. SVD the small stacked matrix
  $[\,R_{[i_1]} W_{i_1} | \cdots | R_{[i_k]} W_{i_k}\,] \in
  \R^{m \times \sum_i r_{i-1} r_i}$; retain leading left singular vectors
  $Y$.
\item Set $B_g \coloneqq Q Y$; project each mode by applying
  $(B_g^\top B_i)$ to the Tucker leg of core $i$ and replacing the factor by
  $B_g$. Follow with Algorithm~\ref{alg:rounding}.
\end{enumerate}
\end{algorithm}

\begin{lemma}\label{lem:init}
The column space of $[\,K_{i_1} | \cdots | K_{i_k}\,]$ lies in
~~~
**The collision `K_i` resolves:** with `U_i` → `B_i` alone, `:279` reads `B_i \coloneqq B_i W_i` and `:280`
`Q^\top B_i = R_{[i]} W_i`: a factor defined as a block. `R`, `R_{[i]}` and `Q` are QR's (§1.5 item 2).

*`shared_t3_math.tex:304–306`* — **stays as it is**, though §4.8's range ":279–306" covers it:
~~~tex
T3Toolbox implements only step (1)'s stacked-factor SVD, as an \emph{exact}
common-span rewrite: the factor coefficients are the SVD's own row blocks
($B_i = W_{[i]} \operatorname{diag}(s)\, V^\top$, valid for arbitrary --- not
~~~

**BEFORE** *`ttm_t3m_ht_note.tex:85, 93, 217–218, 238–240`*
~~~tex
(U^A\krrow U^B)[(a,b),\,x] &= U^A[a,x]\,U^B[b,x] && \text{(row-wise Khatri--Rao, ``COPY'')},\\
W_i \;=\; U^A_i\krrow U^B_i \in\R^{(n^A_i n^B_i)\times N_i},
An HT tensor is a \emph{dimension tree} with leaf frames $U_i\in\R^{n_i\times N_i}$
and transfer tensors $B_t\in\R^{n_{t_1}\times n_{t_2}\times n_t}$ at internal nodes
\text{leaf frames } \;W_i = U^A_i\krrow U^B_i,
\qquad
\text{transfer tensors } \; B^C_t = B^A_t\kron B^B_t,
~~~
**AFTER, literal plan (R5)**
~~~tex
(B^X\krrow B^Y)[(a,b),\,x] &= B^X[a,x]\,B^Y[b,x] && \text{(row-wise Khatri--Rao, ``COPY'')},\\
W_i \;=\; B^X_i\krrow B^Y_i \in\R^{(n^X_i n^Y_i)\times N_i},
An HT tensor is a \emph{dimension tree} with leaf frames $B_i\in\R^{n_i\times N_i}$
and transfer tensors $B_t\in\R^{n_{t_1}\times n_{t_2}\times n_t}$ at internal nodes
\text{leaf frames } \;W_i = B^X_i\krrow B^Y_i,
\qquad
\text{transfer tensors } \; B^Z_t = B^X_t\kron B^Y_t,
~~~
`X/Y/Z` resolves `U^B_i` → `B^B_i` (factor `B` of operand `B`).

**What could go wrong.**

- **`:306` renamed to `K_i`.** There `B_i` is the factor itself, in the library's row layout: "the factor
  coefficients are the SVD's own row blocks", and `t3_svd.py:384–387` writes the same identity with `B_i` the factor
  (`[B_{i_1}; ...; B_{i_k}] = W diag(s) V^T`). Following §4.8's range makes the remark false. Fix the range to
  `:279–292`.
- **`F` is not free.** §4.8 says neither `K` nor `F` is used in the note; `\|\cdot\|_F` (Frobenius) is at `:236`,
  `:257`, `:393` and on `:566` itself, which becomes `\sum_i \|FS_i\|_F^2 = \tr(F(\sum_i S_iS_i^\top)F^\top)`.
  Readable, but the claim is wrong; pick another letter or accept it and say so.
- **The ttm rename creates a `B` collision of its own.** Leaf frames `B_i` and HT transfer tensors `B_t` meet in one
  sentence (`:217–218`), and `B^X_i` (`:238`) sits two lines above `B^X_t` (`:240`). §4.8 lists neither. In HT
  usage `U_t` is the frame and `B_t` the transfer tensor, so the HT section is where `U_i` → `B_i` hurts most.
- **The macros are not in "59 superscripts".** `\A`, `\B`, `\C` (calligraphic, 29 uses) stay, so `:64`
  `\C=\A\had\B` sits beside `:95` `G^Z_i = G^X_i\kron G^Y_i`; and `\C` is also the T3's core tensor in `eq:t3`
  (`:72–75`). Ask whether the macros become `\X`, `\Y`, `\Z`, and what the transfer tensors become.
- **No gate reads any of this.** K4/K7/K8 don't see `B_i`/`K_i`/`B_t`, and K16 lists widths only. R1's inventory of
  letter-bearing lines and Nick's PDF review are the guard.

---

## 8. A backend message whose `%s` flips by script and whose prose word flips by hand, and its test

**Where:** `t3toolbox/backend/uniform_padded_frames.py:194–197, 223–226`;
`tests/backend/test_uniform_padded_frame.py:298–303, 318–321`.

**Governs:** B2 (the exact literals in the `zip`); §4.6 (store-naming messages and their 12 tests, in R2:
`test_uniform_padded_frame.py:303, 320`); A3 scoped (`mU`↔`mD`, `mU0`↔`mD0`, `kU`↔`kD` in both files); A2 explicit
(the test name); tier C (`old_U→old_D`); tier E (inv 2 `:196, :225`; inv 1 `:301, 303, 319, 320`); K9.

**BEFORE**
~~~python
    for name, have, want in zip(('up', 'down', 'left', 'right'), old, layout):
        if not np.array_equal(have, want):
            raise ValueError('old_frame_data: the %s masks count %s; an orthogonal representation with these up and '
                             'left counts has %s' % (name, have.tolist(), want.tolist()))
        if any(not np.array_equal(mU0[i], mU0[group[0]]) or not np.array_equal(kU[i], kU[group[0]]) for i in group):
            raise ValueError('sharing group %r: unequal old up masks or Tucker counts; a shared factor needs one '
                             'layout per group' % (group,))
~~~
~~~python
    def test_old_masks_not_an_orthogonal_representation(self):
        masks = list(self.uold[5])
        mD = masks[1].copy()
        mD[1, np.flatnonzero(~mD[1])[0]] = True                         # one extra real down slot at mode 1
        masks[1] = mD
        with self.assertRaisesRegex(ValueError, 'the down masks count'):

    def test_unequal_old_up_masks_in_a_group(self):
        """Sharing labels on an untied scatter (the members' up slots at different positions)."""
        with self.assertRaisesRegex(ValueError, 'unequal old up masks'):
~~~

**AFTER** (R2: the B2 literals, the hand-edited message words and test regexes, the test name, `mU`/`mD`/`kU`;
R4a: the test's comment and docstring)
~~~python
    for name, have, want in zip(('down', 'up', 'left', 'right'), old, layout):
        if not np.array_equal(have, want):
            raise ValueError('old_frame_data: the %s masks count %s; an orthogonal representation with these down and '
                             'left counts has %s' % (name, have.tolist(), want.tolist()))
        if any(not np.array_equal(mD0[i], mD0[group[0]]) or not np.array_equal(kD[i], kD[group[0]]) for i in group):
            raise ValueError('sharing group %r: unequal old down masks or Tucker counts; a shared factor needs one '
                             'layout per group' % (group,))
~~~
~~~python
    def test_old_masks_not_an_orthogonal_representation(self):
        masks = list(self.uold[5])
        mU = masks[1].copy()
        mU[1, np.flatnonzero(~mU[1])[0]] = True                         # one extra real up slot at mode 1
        masks[1] = mU
        with self.assertRaisesRegex(ValueError, 'the up masks count'):

    def test_unequal_old_down_masks_in_a_group(self):
        """Sharing labels on an untied scatter (the members' down slots at different positions)."""
        with self.assertRaisesRegex(ValueError, 'unequal old down masks'):
~~~
The `%s` word follows physical slot 1 (`masks[1]`, the TT-shaped store, "down" before and "up" after). The literal
word "these up and left counts" names the counts passed to `compute_orthogonal_representation_ranks` (`old[0]`, the
Tucker side, and `old[2]`), so it becomes "down". The `:196` line grows to 119 characters.

**What could go wrong.**

- **Taking the regex for a B2 literal.** `'the down masks count'` is not an exact `'down'`, so the script leaves it;
  the backend now says "the up masks count", and the suite fails at R2. Caught.
- **Missing the prose half of the message.** After B2 alone it reads "the up masks count …; … with these up and left
  counts": it looks consistent and is wrong, since the hint means slot 0, now down. R2's gate (K0, K11, K13–K15, suite)
  passes it; K9 flags it only at R4a. §4.6 puts it in R2, so the R2 reviewer has to read the message text, not just
  run the suite.
- **"Fixing" the test the other way**, by corrupting `masks[0]` so the word can stay "down": that changes what the
  test checks. inv 1's rows keep `masks[1]` and change the regex.

---

## 9. HISTORICAL table comments: the keys are frozen, the prose flips, and one `U+` the rows miss

**Where:** `tests/test_contractions_interpreter.py:207–209, 214–215, 227–228, 289–290, 359–366`, and the code they
describe, `t3toolbox/backend/padded_frames.py:370–379` (`_down_message`).

**Governs:** §1.5 item 6 (contraction strings and the HISTORICAL table; K14's string-multiset check); §1.5 item 9
(inv 1 App. B files `:207, 227, 289, 335` as keeps; the overrides come through §3.3 for `:227`/`:335`, §1.2a's text
list for `:227`/`:336`, tier C's B-row text for `:214`, and §3.4's `O` scan); inv 1's tier-E rows (`:360, 361, 363`);
A2 (`_down_message`↔`_up_message`); tier C `padded_frames.py`; A3.

**BEFORE**
~~~python
    # ... the frame's center factors (C only): H = L x3 T, S^T = <O, H>
    'Caib,Cbc->Caic': '',
    'Caxb,Caub->Cxu': '',
    # ... the T3-T3 sweep (C only: BU, G2, the one-hole environment) and its dotted two-hole terms
    'Czn,Cun->Czu': '',
    # The dotted adjoint-up lift (Udot z~), ragged and d-prefixed: no C-only operand, so len_C.
    'tWCo,KCio->tWKCi': 'C',
    # two-row dxi~ correlation per order slice (mu O and O nu, and the dotted terms; the tilde pair carries u):
    'Caib,WCb->WCai': '',
    # ... the alignment step (backend.padded_frames.tv_align_padded_frame, the plan's Appendix C, ragged forms c1-c31;
    # c19 'Caxb,Caub->Cxu' and c22 / c23 above are reused). Far-side messages: down (c1, c2), left (c3-c8), right
    # (c9-c14), up (c15-c20); new completions: the content g_old Phi / g_old Xi (c21), U+ g_new (c24), a slot in the
    # new coordinates (c8 first axis, c25 Tucker axis, c14 last axis), the near terms of the left (c26, c27), right
    # (c28, c29) and down (c30, c31) edges. Every term carries one group C and fixed lowercase counts, so C is solved
    # from any operand: ''.
    'Ckn,Cun->Cku': '',
    'Cun,Cku->Cnk': '',
~~~
**AFTER** (R4a; every key byte-identical)
~~~python
    # ... the frame's center factors (C only): H = L x3 T, S^T = <U, H>
    'Caib,Cbc->Caic': '',
    'Caxb,Caub->Cxu': '',
    # ... the T3-T3 sweep (C only: zB, G2, the one-hole environment) and its dotted two-hole terms
    'Czn,Cun->Czu': '',
    # The dotted adjoint-upward lift (Bdot z~), ragged and d-prefixed: no C-only operand, so len_C.
    'tWCo,KCio->tWKCi': 'C',
    # two-row dxi~ correlation per order slice (mu U and U nu, and the dotted terms; the tilde pair carries u):
    'Caib,WCb->WCai': '',
    # ... the alignment step (backend.padded_frames.tv_align_padded_frame, the plan's Appendix C, ragged forms c1-c31;
    # c19 'Caxb,Caub->Cxu' and c22 / c23 above are reused). Far-side messages: up (c1, c2), left (c3-c8), right
    # (c9-c14), down (c15-c20); new completions: the content g_old Phi / g_old Xi (c21), D+ g_new (c24), a slot in the
    # new coordinates (c8 first axis, c25 Tucker axis, c14 last axis), the near terms of the left (c26, c27), right
    # (c28, c29) and up (c30, c31) edges. Every term carries one group C and fixed lowercase counts, so C is solved
    # from any operand: ''.
    'Ckn,Cun->Cku': '',
    'Cun,Cku->Cnk': '',
~~~
The keys keep their letters though the mnemonics now point the other way: `u` in c1 `'Ckn,Cun->Cku'` and c19
`'Caxb,Caub->Cxu'` indexes the Tucker width (now `nD`), and c19's `x` the up core's middle leg. They are labels, and
K14's string multiset fails on any change. The code side, *`padded_frames.py:370–379`*, after R2:
~~~python
def _up_message(
        x: typ.Tuple[NDArray, NDArray],   # (VP, D): C+(kDi, Ni), the variation's completion rows ; C+(nDi, Ni)
) -> typ.Tuple[
    NDArray,   # PhiU. C+(kD, nD)
    NDArray,   # XiU.  C+(kD, z)
]:
    """The message of the edge core -> leaf (map body)."""
    VP, D = x
    PhiU = contractions.contract('Ckn,Cun->Cku', VP, D)
    return PhiU, _root(VP.swapaxes(-2, -1) - contractions.contract('Cun,Cku->Cnk', D, PhiU))
~~~
(`kDi`/`kD` are shown as the literal plan leaves them; see the last bullet. The docstring's arrow stays, per inv 2.)

**What could go wrong.**

- **`U+ g_new (c24)` stays `U+` by the rows.** inv 1's row for `:361` swaps only the word ("down (c15-c20)") and keeps
  `U+`; §1.4's L2 doesn't fire (`U+` is not a U-family form); K8 can't see it. The code's comment for the same object
  (`padded_frames.py:452`, `# U+.`) becomes `# D+.` by inv 2's row, and inv 2 §6 also says `D+` for this test line. So
  the literal result has the test calling the new Tucker factor `U+` and the code calling it `D+`. Fix inv 1's row.
- **`S^T = <O, H>` (`:207`) escapes K8**: angle brackets are not among K8's frame-core contexts, and inv 1 files the
  line as a keep. Only §3.4's text scan for `O` finds it. (`:289`'s `mu O`/`O nu` are caught by K8.)
- **"up (c30, c31) edges" is a both-senses site** off §4.2's list. The word swap matches the renamed `_up_edge`, but
  `_up_edge` (new) is the edge core → leaf, and "up edges" reads as edges pointing up. Either "the up store's (c30,
  c31) edges" or accept the function-name reading; R1 should say which.
- **`kD` in `padded_frames.py`'s comments is unmapped.** A3's scoped `k` swap covers only the uniform files, and every
  `kU`/`kD`/`k1U`/… in `padded_frames.py` is in a comment (32 tokens, 0 NAMEs). inv 2 has rows for them
  (`kDi→k_up_i`), but they are plain WIDTH rows, not tier E. So `_up_message` reads `C+(kD, nD)` for the up store's
  completion count, beside the correctly swapped `nD`. Add `backend/padded_frames.py` (comments) to the scoped swap.

---

## 10. A doctest input that changes while its output cannot: equal padded widths

**Where:** `t3toolbox/backend/utv_operations.py:704–709` (the `utv_corewise_lift` doctest), with the docstring
`:669–673` and body `:716–723`.

**Governs:** A3 (the keyword NAMEs `nU=`/`nD=` and the example's comment); §2 (silent width keywords); §4A(e)
(expected outputs unchanged); §1.2a (`utv_corewise_lift`: `Udot→Bdot`); A2 (`up_mask`, `_up`); §3.3 and §3.4 (the
point `(U, P)`); K15 and the bijection.

**BEFORE**
~~~python
    >>> uframe = ufc.t3frame_to_ut3frame(frame, N=8, nU=4, nD=4, rL=3, rR=3)
    >>> uv = ufc.t3variations_to_ut3variations(v, N=8, nU=4, nD=4, rL=3, rR=3)
    >>> theta_hat = utvo.utv_corewise_lift(uframe, uv, ufo.ufv_lift_factors(uframe, 0.0))
    >>> ragged = tvo.tv_corewise_lift(frame, v, fvo.fv_lift_factors(frame, 0.0))
    >>> theta_hat[0].shape, theta_hat[1].shape                       # (d,)+(nU, N), (d,)+(rL, nU, rL)
    ((3, 4, 8), (3, 3, 4, 3))
~~~
**AFTER (R2)**
~~~python
    >>> uframe = ufc.t3frame_to_ut3frame(frame, N=8, nD=4, nU=4, rL=3, rR=3)
    >>> uv = ufc.t3variations_to_ut3variations(v, N=8, nD=4, nU=4, rL=3, rR=3)
    >>> theta_hat = utvo.utv_corewise_lift(uframe, uv, ufo.ufv_lift_factors(uframe, 0.0))
    >>> ragged = tvo.tv_corewise_lift(frame, v, fvo.fv_lift_factors(frame, 0.0))
    >>> theta_hat[0].shape, theta_hat[1].shape                       # (d,)+(nD, N), (d,)+(rL, nD, rL)
    ((3, 4, 8), (3, 3, 4, 3))
~~~
The keywords stay in their positions, so the call now reads `nD=4, nU=4`.

**Why the doctest cannot see a stale name.** The padded widths are equal (4, 4), and so are the stores: `x` is at
Tucker ranks (2, 3, 2) and TT ranks (1, 2, 2, 1), whose up widths `min(n_i, r_i r_{i+1})` are (2, 3, 2) too. Leaving
`nU=4, nD=4` unrenamed, or "tidying" the renamed call back into that order, passes the same values, and the output is
identical.

**What catches it.** K15 after R2: the hit table lists both keyword positions with their images, so a stale or
reordered `nU` at the first position is an unlisted value. And the per-file bijection (count_after(`nD`) =
count_before(`nU`)). The battery's distinct widths (8, 7, 5, 6, 3) and §5.2's assertions test the keywords' meaning
through the frontend forwarders; they do not read this line.

The rest of the function (R2 for NAMEs, R4a for prose):
~~~python
):  # -> plain-UT3 .data of the raw direction theta_hat = (Bdot, Gdot, shape, (down, left) masks over K + C)
    raw core variations of the base point ``(D, L)``, ``theta_hat = L v``,
    ``Bdot_i = (S_i^T)^{+eps} V_i``,   ``Gdot_i = H_i x_3 T_i^{+eps}`` (``i <= d-2``),   ``Gdot_{d-1} = H_{d-1}``,
    Bdot = contractions.contract('dCux,dKCxn->dKCun', lift_factors.tucker_pinvs, tkv)
    down_mask, _up, left_mask, _right = frame_data[5]
    masks = ufv_masking._masks_over_stack((down_mask, left_mask), frame_data[0].shape[1:-2], tkv.shape[1:-2])
~~~

**What could go wrong.**

- **`(U, P)` → `(D, L)` meets the lift `L`.** `:671` is not on §3.3's list of points given by a frame's stores; §3.4's
  `P` scan finds it, and by `tv_operations.py:504`'s row it becomes `(D, L)`. It then sits beside `theta_hat = L v`, `L`
  the lift, in one line. D-7 creates this collision (in `tv_operations.py:504–510` it already exists, across lines);
  §3.1 doesn't list it. Write "the lift" in words here, or ask.
- **`Udot_i`, `Udot_g` (`:673, :682`)** are not on §1.2a's text list; K8 forces the change and the function's `B` form
  picks `Bdot`.

---

## 11. Least sure: edit 3(c), the corewise substitution sentences

The sentences that describe what sits in the corewise frame's down slot:

*`apply.py:320–322`*
~~~python
    Implemented by the Section 6.3 ("corewise simplification") substitution into the tangent transpose:
    feed the frame's own cores in place of the orthogonal frames (``P, Q, O -> G_i``), with ``U_i`` no
    longer required orthogonal -- i.e. :py:func:`tv_apply_transpose` at frame ``(U, G, G, G)``. No
~~~
*`utv_operations.py:381–382, 391`*
~~~python
    The uniform mirror of the additive retraction on the corewise frame ``(U, G, G, G)`` (Section 6.3,
    Alger et al. 2026 -- the ``(P, Q, O) -> G`` substitution): recovers the point ``(U, G)`` from the frame
    up_sc = frame_data[0]       # (d,)   + C + (nU, N)        -- the point's Tucker core U
~~~
and `probing.py:1427` ("``P, Q, O -> G_i`` (``U`` non-orthogonal)"), with the other C4 rows of inv 2.

**What the plan gives:**

- `utv_operations.py`: §1.2a rules `B` ("the point's Tucker core U", "the point `(U, G)`" → `B`, `(B, G)`), so
  `down_sc = frame_data[0]  # … -- the point's Tucker core B` and "recovers the point ``(B, G)``".
- `apply.py:321` and `probing.py:1427`: inv 2's C4 rows (tier E) give `U_i→D_i`, `U→D`. §1.4's L2 then layers "`D`, or
  `B` for a T3's own Tucker cores", and the object is exactly that: θ's own non-orthogonal cores in the down slot.
- D-6 fixes the tuple as `(D, G, G, G)`; "Where the two letters meet" lists only `tv_corewise_lift` and
  `curvature_internals.md`; no gate reads a bare `U`, `B` or `D` in prose.

So two mechanisms give opposite letters for one object, and the layering rule that should arbitrate is itself the
question. **My answer:** `B` for the content, `D` for the slot in the tuple ("with ``B_i`` no longer required
orthogonal -- i.e. … at frame ``(D, G, G, G)``", "(``B`` non-orthogonal)"), plus one sentence, once, in the convention
paragraph: "at the corewise frame of θ = (B, G) the down slot holds B". `D_i … no longer required orthogonal` would
contradict the down store's definition. But this is a ruling, not a reading of the plan: add the C4 sites to
§1.2a's "Where the two letters meet" with Nick's letter.

---

## 12. Plan issues surfaced

1. **The σ̂ row renames a shape tuple** (§5): drop `sampling_derivatives.py:1693–1694` (`_flatten_w`); the count is
   10 + 5, not 14 + 5.
2. **`ufv_masking.py`'s mask locals and width labels** (§1): add `UM_k→DM_k, UM_t↔DM_t` to tier C, and `ufv_masking.py`
   (comments) to the scoped `W` swap. §3.4's scan cannot produce the `DM_t` half.
3. **`padded_frames.py`'s `kU`/`kD` comments** (§9): 32 tokens outside A3's scoped swap; add the file.
4. **The frame-generic reason for the two `deta_tilde_tangent` kernels is false** by the call graph (§3); give the A1
   keyword as the reason, or move them to `B`.
5. **`st['dU']`** (§3): 9 string keys that K8 matches and no row maps; add a scoped string row.
6. **`_curvature_oracles.py:262` "theta = (U, G)"** and prose **`(Ud, Gd)`** (`:397, :471`) (§3): the first is an App.
   B keep outside every scan; the second escapes K8, whose pattern lacks `Ud`.
7. **The C4 corewise sentences** (§11): inv 2 says `D`, §1.2a's precedent says `B`; needs a ruling and a list.
8. **`shared_t3_math.tex`** (§7): the `K_i` range should be `:279–292` (`:306`'s `B_i` is the factor); `F` is not free
   (Frobenius `_F`).
9. **`ttm_t3m_ht_note.tex`** (§7): `B_i` vs the HT transfer tensors `B_t`; the macros `\A`, `\B`, `\C` (29 uses) are
   outside the "59 superscripts", and `\C` has two meanings.
10. **HISTORICAL comments** (§9): inv 1's `:361` row keeps `U+` (should be `D+`); `<O, H>` (`:207`) escapes K8; "up
    (c30, c31) edges" is an unlisted both-senses site.
11. **`utv_operations.py:671`** (§10): `(U, P)` is not on §3.3's list, and `(D, L)` collides with the lift `L` on the
    same line; §3.1 should list D-7's `L`-vs-lift collisions.
12. **Alignment in bodies and doctest examples** (§1): §4.10 covers signature blocks only; say whether R4a realigns
    the rest or leaves ±2 columns.
13. **§1.1 cites `padded_frames.py:143`** with `:167`'s text (§2).
14. **Not among the ten, found on the way:** `utv_operations.py:315–323` writes "`[Q | P] = [R | L]`", "the Q block",
    "P-part", and `tests/test_uniform_manifold.py:108` "`[Q-block | P-block]`". D-7 turns the first into
    `[R | L] = [R | L]`, and "P-block" is on §1.5 item 3's DO-NOT-TOUCH list, where it means the padded frame's
    completion block, a different object. §1.5 item 3 or §3.4 should say which "P-block" each is.
15. **Item 12's generic `n`** at `sampling_derivatives.py:3236` (`tt_cores`) leaves its twin `tt_dirs` (`:3237`) and the
    outputs (`:3239–3240`) on store letters: one size, three labels (`ni`, `nDi`, `nUi`) in four lines.
