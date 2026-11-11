# Store rename: ten sample edits, and the riskiest one

Purpose: show, before any implementation, what the store-rename plan (`dev/store_rename_plan_2026-11-10.md`,
draft 11, `e556f750`) does to real code and docs. Ten edits, chosen to differ as much as possible, plus the one edit I
think is riskiest. Nothing in the tree was modified to produce this.

How to read it:

- **BEFORE** blocks are quoted verbatim from the working tree. For `t3toolbox/`, `tests/`, `examples/` and the
  quoted notes and pages, this equals `v2026.4.0` (checked with `git diff --stat v2026.4.0`).
- **AFTER** is the final text once every slice that touches the lines has run. Where two slices touch one line
  (e.g. R2's script renames a NAME and R4a rewrites the comment beside it), the interim R2 state is also shown.
- "Governs" cites the plan section, tier and decision. "Gates" names the checks that would catch a mistake, and
  says so when none would.

A coverage table is at the end (§13), followed by plan issues that came up while drafting these (§12).

---

## 1. A public dataclass field and its shape comment: `T3Frame.up_tucker_cores` → `down_tucker_cores`

**Where:** `t3toolbox/frame_variations_format.py:156–159` (the four `T3Frame` fields; the field comments form one
§4A block).

**Governs:**

- The field names: tier A1 (`up_tucker_cores`→`down_tucker_cores`, `down_tt_cores`→`up_tt_cores`), applied by R2's
  script as NAME tokens. This is a loud break (§2): `AttributeError`, and a `TypeError` for the constructor and
  `dataclasses.replace` keywords.
- The comments: a §4A **field-comment relation** block (seed list `156–159`), fenced by a column range and rewritten
  by hand in R4a. §4.4 names the letters: `B_xo`→`D_xo`, the TT store's `R_ixj` (which clashes with *right*)
  →`U_ixj`, and `P`/`Q`→`L`/`R` (D-7).
- The widths: D-8, swapped by hand inside the fence (A3: "the §4A fences included (by hand there)"). The
  signature block is realigned per §4.10, and K13 checks the result.

**BEFORE**

*`t3toolbox/frame_variations_format.py:156–159`*
~~~python
    up_tucker_cores:    typ.Tuple[NDArray,...]  # len=d. B_xo B_yo   = I_xy, Bi.shape = stack_shape+(nUi, Ni)
    down_tt_cores:      typ.Tuple[NDArray,...]  # len=d. R_ixj R_iyj = I_xy  Ri.shape = stack_shape+(rLi, nDi, rR(i+1))
    left_tt_cores:      typ.Tuple[NDArray,...]  # len=d. P_iax P_iay = I_xy, Pi.shape = stack_shape+(rLi, nUi, rL(i+1))
    right_tt_cores:     typ.Tuple[NDArray,...]  # len=d. Q_xaj Q_yaj = I_xy  Qi.shape = stack_shape+(rRi, nUi, rR(i+1))
~~~

**After R2 (interim).** The NAMEs are renamed in place, and the fenced comments are byte-identical (K11), so they
shift by ±2 columns:

~~~python
    down_tucker_cores:    typ.Tuple[NDArray,...]  # len=d. B_xo B_yo   = I_xy, Bi.shape = stack_shape+(nUi, Ni)
    up_tt_cores:      typ.Tuple[NDArray,...]  # len=d. R_ixj R_iyj = I_xy  Ri.shape = stack_shape+(rLi, nDi, rR(i+1))
    left_tt_cores:      typ.Tuple[NDArray,...]  # len=d. P_iax P_iay = I_xy, Pi.shape = stack_shape+(rLi, nUi, rL(i+1))
    right_tt_cores:     typ.Tuple[NDArray,...]  # len=d. Q_xaj Q_yaj = I_xy  Qi.shape = stack_shape+(rRi, nUi, rR(i+1))
~~~

**AFTER (R4a: realigned, comment rewritten by hand)**

~~~python
    down_tucker_cores:  typ.Tuple[NDArray,...]  # len=d. D_xo D_yo   = I_xy, Di.shape = stack_shape+(nDi, Ni)
    up_tt_cores:        typ.Tuple[NDArray,...]  # len=d. U_ixj U_iyj = I_xy  Ui.shape = stack_shape+(rLi, nUi, rR(i+1))
    left_tt_cores:      typ.Tuple[NDArray,...]  # len=d. L_iax L_iay = I_xy, Li.shape = stack_shape+(rLi, nDi, rL(i+1))
    right_tt_cores:     typ.Tuple[NDArray,...]  # len=d. R_xaj R_yaj = I_xy  Ri.shape = stack_shape+(rRi, nDi, rR(i+1))
~~~

Realignment puts `typ` back at column 24 (`down_tucker_cores:` + 2 spaces), so every comment starts at its old
column 48. The comment span then passes the strict in-place check K10 (same lengths; only label characters differ),
which I checked mechanically. The relations, re-derived per §4A(d):

- `D` is `(nDi, Ni)`. The rows `x, y` are on the edge and the leaf `o` is summed, below the edge: down-orthogonal.
- `U` is `(rLi, nUi, rR(i+1))`. Both bonds `i, j` are summed and the middle `x, y` is on the Tucker edge, above it:
  up-orthogonal.
- `L` sums its left bond and mode `i, a`; `R` sums its mode and right bond `a, j`.

K13's table is satisfied: `nD` for `down_*` and for the middle leg of `left_tt_*`/`right_tt_*`, and `nU` for `up_*`.

**What could go wrong.**

- **The plan's R2 gate cannot pass here.** R2 must leave the fenced comment byte-identical (K11), yet R2's gate also
  requires K13 clean. After R2, K13 sees `down_tucker_cores` with `nUi`, `up_tt_cores` with `nDi` and
  `left/right_tt_cores` with `nUi`, and fails. The same happens on `uniform_frame_variations_format.py:110–113`:
  eight lines in all. Either K13 must skip fenced spans at R2, or these blocks should be split from their width
  tokens (see §12).
- **Replacing `R` by sequential rewrites.** Doing `R`→`U` and then `Q`→`R` in sequence would leave both TT relations
  labelled `R`. The block is hand-only, so the risk is a human slip.
  - K8's `\b[PQ]_(?:i|last|\{)` catches a leftover `P_iax`.
  - It does not catch `Q_xaj`, `Pi.shape` or `Qi.shape`.
  - Nothing catches `R_ixj` left where `U_ixj` belongs.
- **Copying the inventory's proposal text.** It uses `n_down_i` (withdrawn by D-8; K2 would catch it) and does not
  realign.
- **The two shape-comment spellings.** The dataclass comment spells the shape `Bi.shape = stack_shape+(nUi, Ni)`. The
  `elm_shape=C+(nUi, Ni)` spelling belongs to the backend signatures, which R2's script converts (item 3), not a
  hand rewrite.

---

## 2. A same-spelling silent swap: the `up_ranks` / `down_ranks` properties

**Where:** `t3toolbox/frame_variations_format.py:169–175` (`T3Frame`), plus the repr at `:288` and the doctest at
`:338`.

**Governs:**

- Tier A2 (`up_ranks`↔`down_ranks`, "everywhere"), together with A1 for the bodies, all by R2's script.
- §2 lists these among the 44 **silent** sites. §0 explains why the spelling stays: the up store has ranks, so
  `up_ranks` must exist, and it must now name the TT-shaped store. The old meaning moves to `down_ranks`. A name
  cannot alias two meanings, so there are no aliases, and the release notes lead with the list.
- D-4 (rename only) covers the doctest.

**BEFORE**

*`t3toolbox/frame_variations_format.py:169–175`*
~~~python
    @ft.cached_property
    def up_ranks(self) -> typ.Tuple[int,...]:
        return tuple([U.shape[-2] for U in self.up_tucker_cores])

    @ft.cached_property
    def down_ranks(self) -> typ.Tuple[int,...]:
        return tuple([G.shape[-2] for G in self.down_tt_cores])
~~~

**AFTER (R2)**

~~~python
    @ft.cached_property
    def down_ranks(self) -> typ.Tuple[int,...]:
        return tuple([U.shape[-2] for U in self.down_tucker_cores])

    @ft.cached_property
    def up_ranks(self) -> typ.Tuple[int,...]:
        return tuple([G.shape[-2] for G in self.up_tt_cores])
~~~

The definitions keep their physical order (Tucker first). Only the names permute, and each body follows its field's
loud A1 rename. The comprehension letter `U` stays: tier C's `U↔D` for this file is scoped to `483–573`, and inv 1
files `:171` as SW only. "`U` … in `down_tucker_cores`" is therefore an **ask** item, not a rename.

**BEFORE** (repr)

*`t3toolbox/frame_variations_format.py:288`*
~~~python
        return (f"T3Frame(shape={self.shape}, up_ranks={self.up_ranks}, "
~~~

**AFTER (R2)**

~~~python
        return (f"T3Frame(shape={self.shape}, down_ranks={self.down_ranks}, "
~~~

**BEFORE** (doctest; expected output `((3, 4, 3), True)`)

*`t3toolbox/frame_variations_format.py:338`*
~~~python
        >>> F.up_ranks, bool(F.is_orthogonal())
~~~

**AFTER (R2)**, with identical expected output:

~~~python
        >>> F.down_ranks, bool(F.is_orthogonal())
~~~

**What could go wrong.**

- **Nothing in the suite sees a missed swap.** A padded frame is minimal, so `up_ranks == down_ranks == (3, 4, 3)`,
  and the doctest prints the same thing whether or not `:338` was renamed. Only the battery's slack fixtures
  (`xS`: down `(3, 6, 2)` vs up `(2, 4, 2)`) and §5.2's new unequal-store assertions distinguish the two stores.
- **The f-string at `:288` is a trap on Python ≥ 3.12.** Tokenize splits it, so the NAME `self.up_ranks` gets
  renamed while the label `up_ranks=` does not. The repr would then print the Tucker ranks under the up label. §1.3
  pins Python 3.11 for this reason, and B2's repr comparison is the net.
- **`T3FrameWeights` is worse** (`:1456–1457`, `:1495–1501`). Every name on that class keeps its spelling, so there is
  no loud break at all. `T3FrameWeights(up_weights=w, …)` written for 2026.4.0 silently puts `w` in the other
  physical store.

---

## 3. Width symbols inside a backend signature's shape comments: `tv_corewise_lift`

**Where:** `t3toolbox/backend/tv_operations.py:482–497`, plus the derivative line `:510` in the same docstring.

**Governs:**

- The comment labels: A1, `\b`-bounded in COMMENT tokens, by R2's script.
- The widths: the A3 width family by R2's script. `nU`↔`nD` with the suffix kept; `rL`/`rR` untouched (D-3).
- `Udot`: a comment letter, so not part of tier C's NAME map. R4a renames it (§3.4, §4.4), and K8 catches a leftover.
- `DPhi`: tier D's enumerated derivative line `tv_operations.py:510` (D-11), in R4a.
- §4.10 realigns the comment columns, and K13 checks them.

**BEFORE**

*`t3toolbox/backend/tv_operations.py:482–497`*
~~~python
def tv_corewise_lift(
        frame:          typ.Tuple[
            typ.Sequence[NDArray],              # up_tucker_cores. len=d, elm_shape=C+(nUi, Ni)
            typ.Sequence[NDArray],              # down_tt_cores.   len=d, elm_shape=C+(rLi, nDi, rR(i+1))
            typ.Sequence[NDArray],              # left_tt_cores.   len=d, elm_shape=C+(rLi, nUi, rL(i+1))
            typ.Sequence[NDArray],              # right_tt_cores.  len=d, elm_shape=C+(rRi, nUi, rR(i+1))
        ],
        variations:     typ.Tuple[
            typ.Sequence[NDArray],              # tucker_variations. len=d, elm_shape=K+C+(nDi, Ni)
            typ.Sequence[NDArray],              # tt_variations.     len=d, elm_shape=K+C+(rLi, nUi, rR(i+1))
        ],
        lift_factors:   fv_operations.LiftFactors,  # the frame's, fv_operations.fv_lift_factors; stack C
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],                    # Udot. len=d, elm_shape=K+C+(nUi, Ni)
    typ.Tuple[NDArray, ...],                    # Gdot. len=d, elm_shape=K+C+(rLi, nUi, rL(i+1))
]:
~~~

**AFTER (R2, then R4a for `Ddot` and the column realignment)**

~~~python
def tv_corewise_lift(
        frame:          typ.Tuple[
            typ.Sequence[NDArray],              # down_tucker_cores. len=d, elm_shape=C+(nDi, Ni)
            typ.Sequence[NDArray],              # up_tt_cores.       len=d, elm_shape=C+(rLi, nUi, rR(i+1))
            typ.Sequence[NDArray],              # left_tt_cores.     len=d, elm_shape=C+(rLi, nDi, rL(i+1))
            typ.Sequence[NDArray],              # right_tt_cores.    len=d, elm_shape=C+(rRi, nDi, rR(i+1))
        ],
        variations:     typ.Tuple[
            typ.Sequence[NDArray],              # tucker_variations. len=d, elm_shape=K+C+(nUi, Ni)
            typ.Sequence[NDArray],              # tt_variations.     len=d, elm_shape=K+C+(rLi, nDi, rR(i+1))
        ],
        lift_factors:   fv_operations.LiftFactors,  # the frame's, fv_operations.fv_lift_factors; stack C
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],                    # Ddot. len=d, elm_shape=K+C+(nDi, Ni)
    typ.Tuple[NDArray, ...],                    # Gdot. len=d, elm_shape=K+C+(rLi, nDi, rL(i+1))
]:
~~~

**BEFORE** (`:510`)

*`t3toolbox/backend/tv_operations.py:510`*
~~~python
    there); at a rank-deficient point the clipped directions drop out, ``DPhi[L v] = E(Q v)`` (see
~~~

**AFTER (R4a)**

~~~python
    there); at a rank-deficient point the clipped directions drop out, ``dPhi[L v] = E(Q v)`` (see
~~~

Each width follows D-8's statement:

- The Tucker factor's rows are `nDi`.
- The up core's middle leg is `nUi`.
- The left/right cores' and the TT variation's middle legs are `nDi`.
- The Tucker variation's rows are `nUi`.

Every bracket keeps its `rL…`/`rR…`, and `L` (the lift) and `Q` (the projection off the clipped directions,
`fv_operations.py:579`) in `:510` are not frame letters. The script's in-place edit leaves the frame group's inner
`len=d` column ragged (69, 65, 67, 67, where it was 67 throughout). The realignment moves the group to 69, which is
also the variations group's column.

**What could go wrong.**

- **K13 may not parse this form.** K13's stated forms are a signature or field comment `NAME: type  # …`, a
  docstring listing `- NAME = (…), elm_shape=(…)`, and their uniform supercore forms. The commonest backend form, a
  tuple element `typ.Sequence[NDArray],  # down_tucker_cores. len=d, elm_shape=…`, is none of these. If K13's parser
  misses it, the most frequent frame-store shape comment in the backend goes unchecked (§12).
- **Over-reach in the same docstring.** A careless D-7 pass turning `E(Q v)` into `E(R v)` violates D-7's scope.
  §3.4 must classify this `Q` as kept, and no gate reads it.
- **`theta = (U, L)` at `:504, 506, 549` is easy to miss.** It changes to `(D, L)` only because inv 2 files those
  lines as CL rows (§4.4). It is not on §3.3's list of "points given by a frame's stores".

---

## 4. The `update` keyword: three forms of D-2 on one call site

**Where:** `t3toolbox/backend/padded_frames.py`:

- the step's signature `:554` and validation `:560–562`;
- the truncation check `:581`;
- the store gates `:624`, `:655`;
- the two-colour loop `:755–756`.

The test is `tests/backend/test_padded_frame.py:730–731`.

**Governs:**

- §1.3 and §4.5, applied in R3 to the LETTER-API rows (37 backend, 93 frontend/test).
- D-2 is open, with (a) recommended.
- K3 checks the result, and the battery's B7 runs all 15 store subsets through the `upd()` adapter.
- R2 has already run on these lines. A3 has swapped the widths (`nU0`↔`nD0`, …), and A2 has swapped the edge
  functions (`_up_edge`↔`_down_edge`), but **the strings are untouched**: `'U' in update` stays correct for the
  still-old letter API until R3.

**BEFORE**

*`t3toolbox/backend/padded_frames.py:554`*
~~~python
        update:            str = 'ULRO',                             # stores updated: 'U' up, 'L' left, 'R' right, 'O' down; the rest transported
~~~
*`t3toolbox/backend/padded_frames.py:560–562`*
~~~python
    if not update or len(set(update)) != len(update) or not set(update) <= set('ULRO'):
        raise ValueError("update must be a non-empty set of the stores 'U', 'L', 'R', 'O' without repeats; got %r"
                         % (update,))
~~~
*`t3toolbox/backend/padded_frames.py:581`*
~~~python
    for store, letter, old, new, own in zip(_STORE_NAMES, 'UOLR', (nU0, nD0, rL0, rR0), out, (nU, nD, rL, rR)):
~~~
*`t3toolbox/backend/padded_frames.py:624`*
~~~python
        tuple(nU1[g[0]] - nU0[g[0]] for g in groups), ('U' in update,) * len(groups)))
~~~
*`t3toolbox/backend/padded_frames.py:655`*
~~~python
        tuple(nD1[i] - nD0[i] for i in range(d)), ('O' in update,) * d))
~~~
*`t3toolbox/backend/padded_frames.py:755–756`*
~~~python
            for update in ('UL', 'RO'):
                frame, movement = _aligning_pass(step, frame, hint, project, old, target, sharing, update)
~~~

**AFTER, form (a), recommended: letter strings in the new meaning.** The helper, `_STORE_LETTERS = 'DULR'` and
`_COLOURS = ('DL', 'UR')` are added beside `_STORE_NAMES` (`:51`) exactly as §1.3 gives them.

~~~python
        update:            str = 'DULR',                             # store letters, any order: 'D' down, 'U' up, 'L' left, 'R' right; the rest transported
~~~
~~~python
    update = _update_stores(update)
~~~
~~~python
    for store, letter, old, new, own in zip(_STORE_NAMES, _STORE_LETTERS, (nD0, nU0, rL0, rR0), out, (nD, nU, rL, rR)):
~~~
~~~python
        tuple(nD1[g[0]] - nD0[g[0]] for g in groups), ('D' in update,) * len(groups)))
~~~
~~~python
        tuple(nU1[i] - nU0[i] for i in range(d)), ('U' in update,) * d))
~~~
~~~python
            for update in _COLOURS:
                frame, movement = _aligning_pass(step, frame, hint, project, old, target, sharing, update)
~~~

`:750`/`:765` pass `_STORE_LETTERS` and `:760` passes `_COLOURS[0]`. Note that `'U'` leaves `:624` and reappears on
`:655` with the other meaning.

**Form (b): a tuple of words.** These lines differ from (a), with `_COLOURS = (('down', 'left'), ('right', 'up'))`:

~~~python
        update:            typ.Collection[str] = ('down', 'up', 'left', 'right'),   # store words, any order; the rest transported
    ...
    for store, old, new, own in zip(_STORE_NAMES, (nD0, nU0, rL0, rR0), out, (nD, nU, rL, rR)):
            ...
            if b < c and store not in update:                                       # :586, the letter column gone
    ...
        tuple(nD1[g[0]] - nD0[g[0]] for g in groups), ('down' in update,) * len(groups)))
    ...
        tuple(nU1[i] - nU0[i] for i in range(d)), ('up' in update,) * d))
~~~

**Form (c): a collection of single letters.** These lines differ from (a), with
`_STORE_LETTERS = ('D', 'U', 'L', 'R')` and `_COLOURS = (('D', 'L'), ('U', 'R'))`:

~~~python
        update:            typ.Collection[str] = ('D', 'U', 'L', 'R'),   # store letters, any order: 'D' down, 'U' up, 'L' left, 'R' right; the rest transported
~~~

**Validation messages.** All keep `update must be` and `without repeats`, which the tests' regexes match.

- (a), `update='ULRO'`: `ValueError: update must be a non-empty string of the store letters 'D', 'U', 'L', 'R'
  without repeats (D down, U up, L left, R right), e.g. 'DL'; got 'ULRO'; 'O' was the 2026.4.0 letter for the
  TT-shaped store, now 'U' (and 2026.4.0's 'U', the Tucker store, is now 'D')`.
- (a), `update='UL'`: **accepted.** It now updates up+left. In 2026.4.0 it meant Tucker+left.
- (b), `update='DL'`: `ValueError: update must be a non-empty tuple of the store words 'down', 'up', 'left', 'right'
  without repeats, e.g. ('down', 'left'); got 'DL'`.
- (c), `update='DL'`: `ValueError: update must be a non-empty collection of the store letters 'D', 'U', 'L', 'R'
  without repeats, e.g. ('D', 'L'), not a str; got 'DL'`. `'ULRO'` adds (a)'s `'O'` sentence.

**The test across the two slices.**

*`tests/backend/test_padded_frame.py:730–731`*
~~~python
        for missing, store in (('U', 'up'), ('O', 'down'), ('L', 'left'), ('R', 'right')):
            update = ''.join(c for c in 'ULRO' if c != missing)
~~~

After R2, B2 has swapped only the words. This interim is correct for the old letter API, so the suite stays green:

~~~python
        for missing, store in (('U', 'down'), ('O', 'up'), ('L', 'left'), ('R', 'right')):
            update = ''.join(c for c in 'ULRO' if c != missing)
~~~

After R3, form (a):

~~~python
        for missing, store in (('D', 'down'), ('U', 'up'), ('L', 'left'), ('R', 'right')):
            update = ''.join(c for c in 'DULR' if c != missing)
~~~

**What could go wrong.**

- **"Fixing" a string in R2.** A reviewer who corrects `'U' in update` → `'D'` on `:624` in R2, because it "reads
  wrong" next to `PhiD`/`XiD`, breaks the old API mid-sequence. Under `'ULRO'` the down store then silently
  transports. The R2 battery catches this bitwise; the suite might not.
- **Converting by initials.** Mapping `U`→"up" instead of applying the simultaneous `'U'`→`'D'`, `'O'`→`'U'` gives
  colours `('UL', 'DR')`. That is a different Gauss–Seidel pairing; the battery catches it, but only bitwise.
- **Form (b)'s constant** `('right', 'up')` is not in its own canonical order (`('up', 'right')`). This is harmless,
  since the loop always passes the same constant, but it is visible in the docs.

---

## 5. An ASCII edge-rank diagram, rewritten whole

**Where:** `t3toolbox/frame_variations_format.py:86–108` (the `T3Frame` docstring).

**Governs:**

- §4A: an ascii-diagram block, rewritten whole by hand in R4a and fenced from every script (K11).
- K10 checks it in place, and K12 checks completion.
- D-8 sets the widths and §4.4 the letters: `D` below, `U` above, widths by adjacency.

**BEFORE**

*`t3toolbox/frame_variations_format.py:86–108`*
~~~text
    The edge ranks are shown in the following diagrams::

           rL0       rL1       rR2      rR(d-1)         rRd
        1 ------ L0 ----- (H1) ----- ... ------ R(d-1) ------ 1
                 |         |                    |
                 | nU0     | nU1                | nU(d-1)
                 |         |                    |
                 U0        U1                   Ud
                 |         |                    |
                 | N0      | N1                 | N(d-1)
                 |         |                    |

    and::

           rL0       rL1       rR2      rR(d-1)         rRd
        1 ------ L0 ------ D1 ------ ... ------ R(d-1) ------ 1
                 |         |                    |
                 | nU0     | nO1                | nU(d-1)
                 |         |                    |
                 U0       (V1)                   Ud
                 |         |                    |
                 | N0      | N1                 | N(d-1)
                 |         |                    |
~~~

**AFTER (R4a)**

~~~text
    The edge ranks are shown in the following diagrams::

           rL0       rL1       rR2      rR(d-1)         rRd
        1 ------ L0 ----- (H1) ----- ... ------ R(d-1) ------ 1
                 |         |                    |
                 | nD0     | nD1                | nD(d-1)
                 |         |                    |
                 D0        D1                   Dd
                 |         |                    |
                 | N0      | N1                 | N(d-1)
                 |         |                    |

    and::

           rL0       rL1       rR2      rR(d-1)         rRd
        1 ------ L0 ------ U1 ------ ... ------ R(d-1) ------ 1
                 |         |                    |
                 | nD0     | nU1                | nD(d-1)
                 |         |                    |
                 D0       (V1)                   Dd
                 |         |                    |
                 | N0      | N1                 | N(d-1)
                 |         |                    |
~~~

Changes:

- **Top diagram:** every Tucker factor `U`→`D` and every vertical width `nU`→`nD`. All three edges touch a Tucker
  factor.
- **Bottom diagram:** the old TT store `D1` becomes the up core `U1`. Its leg down to `V1` becomes `nO1`→`nU1`, since
  it is the up core's middle axis and `V1`'s rows.
- **Unchanged:** `rL`/`rR`, `N`, `H1`, `V1`.

The block passes K10 in place (23 lines, equal lengths, differing columns all label characters), checked
mechanically. `Ud` is the known typo for `U(d-1)` (glossary E17). The in-place version keeps it as `Dd`, because
`D(d-1)` would lengthen the line and turn the block into a K10b re-layout.

**What could go wrong.**

- **K12 cannot see a half-done block.** K12 fails a block only if its new text equals its old text with no reason
  recorded. If R1 inventories `86–108` as one block, rewriting only the bottom diagram (which has the forcing `nO1`)
  satisfies K12 and leaves the top diagram reading backwards. The plan's own K12 example treats `86–96` as a
  separate block, so R1 should make each diagram its own block.
- **The adjacency rule in §4A(f) is ambiguous.** "an edge touching a Tucker factor is `nD`, one touching a
  TT-shaped frame core is `nU`" conflicts on the `L0`–`D0` edge: `L0` is a TT-shaped frame core, and the edge is
  `nD`. D-8's own wording is right; §4A(f) should say "touching an up core".

---

## 6. An orthogonality relation, with its indices re-derived

**Where:** `t3toolbox/frame_variations_format.py:116–124` (the `T3Frame` docstring), plus the same convention at
`:1370–1372`, which is backwards today.

**Governs:**

- §4A: an index-equation block, hand-only, and a re-layout (the plan lists `116–124` under K10b).
- §4A(d): every relation is re-derived.
- §4.1: a convention statement, rewritten "from the rule".
- K8 `outer[- ]orthogonal`, `O_aib`.
- `:1371–1372` are §4.3 **backwards adjectives** (tier E), exempt from K9's converse.

**BEFORE**

*`t3toolbox/frame_variations_format.py:116–124`*
~~~text
    Often, it is desirable for the frame cores to be **orthogonal** as follows:
        - up_tucker_cores   = (U0,...,U(d-1)), orthogonal:       U_ia U_ja = delta_ij
        - down_tt_cores     = (O0,...,O(d-1)), outer-orthogonal  O_aib O_ajb = delta_ij
        - left_tt_cores     = (L0,...,L(d-1)), left-orthogonal:  L_abi L_abj = delta_ij
        - right_tt_cores    = (R0,...,R(d-1)), right-orthogonal  R_ibc R_jbc = delta_ij

    Often, it is desirable for the variations to satisfy the following **Gauge conditions**:
        - U_ia V_ja = 0    (all V)
        - L_abi H_abj = 0  (all but the last H)
~~~

**AFTER (R4a)**

~~~text
    Often, it is desirable for the frame cores to be **orthogonal** as follows:
        - down_tucker_cores = (D0,...,D(d-1)), down-orthogonal:  D_ia D_ja = delta_ij
        - up_tt_cores       = (U0,...,U(d-1)), up-orthogonal:    U_aib U_ajb = delta_ij
        - left_tt_cores     = (L0,...,L(d-1)), left-orthogonal:  L_abi L_abj = delta_ij
        - right_tt_cores    = (R0,...,R(d-1)), right-orthogonal  R_ibc R_jbc = delta_ij

    Often, it is desirable for the variations to satisfy the following **Gauge conditions**:
        - D_ia V_ja = 0    (all V)
        - L_abi H_abj = 0  (all but the last H)
~~~

**Derivations (for the R4a handoff).**

- `D_ia D_ja = δ_ij`: `D_i` is `(nDi, Ni)`. `i, j` index its rows on the Tucker edge, and the leaf index `a` is
  summed, below the edge: **down**-orthogonal.
- `U_aib U_ajb = δ_ij`: `U_i` is `(rLi, nUi, rR(i+1))`. Both bonds `a, b` are summed, and `i, j` are the middle
  index on the Tucker edge, above it: **up**-orthogonal.
- The `L`/`R` relations are unchanged.
- Gauge `D_ia V_ja = 0`: `V_i` is `(nUi, Ni)` and `D_i` is `(nDi, Ni)`. Contracting the leaf `a` gives an `nD × nU`
  block, which must vanish.
- The code agrees: after tier C, `fv_operations.py:219–220` reads `einsum('...io,...jo->...ij', D, D)` and
  `einsum('...iaj,...ibj->...ab', U, U)`, with the subscripts byte-identical (§1.5 item 6, K14).

Every alignment column (`=` at 28, `(` at 30, the equations at 65) maps to itself, so K10b passes. K10 does not:
`orthogonal:` → `down-orthogonal:` puts a `-` where there was a letter. The inventory's own row text for `:117` has
`down_tucker_cores   =` and would push `=` to column 30. Lines 120 and 124 are not rows, so they are left as they
are; that includes the missing colon after `right-orthogonal`, which no row licenses changing.

**BEFORE** (`:1370–1372`)

*`t3toolbox/frame_variations_format.py:1370–1372`*
~~~text
    The "frame cores" are:
        - tucker_cores      = (U0,U1, U2, U3), up orthogonal
        - down_tt_cores     = (O0, O1, O2, O3), down orthogonal
~~~

**AFTER (R4a)**, inv 1's BACKWARDS rows applied inside the fenced block:

~~~text
    The "frame cores" are:
        - down_tucker_cores = (D0, D1, D2, D3), down orthogonal
        - up_tt_cores       = (U0, U1, U2, U3), up orthogonal
~~~

**What could go wrong.**

- **A partial rewrite of `:116–124` passes every gate.** Suppose `:118` is fixed (forced by K8 on `O_aib` and
  `outer-orthogonal`), but `:117` keeps `(U0,…)`/`U_ia U_ja` and `:123` keeps `U_ia V_ja = 0`. `U` is a valid new
  letter, the block differs from its old text (K12 is satisfied), and the result calls the up core down-orthogonal.
  Only the reviewer's re-derivation catches it.
- **`:1371–1372` must flip both words.** Here the adjective has to change. A K9 converse built from inv 2/inv 1 keep
  rows must exempt these lines as tier-E text, or it will demand that the backwards "up orthogonal" survive.

---

## 7. A jet-shift site: the shift becomes `Λ`/`Λᵀ` while the frame core becomes `U`

**Where:** `t3toolbox/backend/sampling_derivatives.py`:

- `:327–332`: the definition of `_shift_up`;
- `:1377–1391`: `_dxi_tilde_order_step`, the collision inv 2 C3 calls "severe";
- `:1476–1479`: the signature of the next kernel.

**Governs:**

- The shift letters: D-1 (`Λ`, `Λᵀ`), the tier-D shift patterns (part of the 41 lines), by hand in R4b.
- The code names: tier C for `sampling_derivatives.py` (`O→U`, `mO→mU`, `On→Un`), applied by R2 as NAME tokens.
- The widths: A3 in R2 (`nOi`→`nUi`, one-way).
- §1.5 item 8: `_shift_up`/`_shift_up_pair` keep their names. Item 3: the direction `P` stays.
- §3.3: "frame down jets" → "the η jets".
- K7 and K8.

**BEFORE** (`:327–332`)

*`t3toolbox/backend/sampling_derivatives.py:327–332`*
~~~python
def _shift_up(
        jet:    NDArray,    # (order+1,)+...; axis 0 is the order axis
        xnp,                # numpy or jax.numpy
) -> NDArray:               # U jet, same shape: (U jet)^(s) = (s+1) jet^(s+1), zero at s = order
    '''The reverse Leibniz shift ``(U x)^(s) = (s+1) x^(s+1)``, the adjoint of :py:func:`_shift_down`.
    Iterated, ``(U^r x)^(s) = r! C(s+r, r) x^(s+r)`` (zero once ``s + r > order``).'''
~~~

**AFTER (R4b)**

~~~python
def _shift_up(
        jet:    NDArray,    # (order+1,)+...; axis 0 is the order axis
        xnp,                # numpy or jax.numpy
) -> NDArray:               # Λᵀ jet, same shape: (Λᵀ jet)^(s) = (s+1) jet^(s+1), zero at s = order
    '''The reverse Leibniz shift ``(Λᵀ x)^(s) = (s+1) x^(s+1)``, the adjoint of :py:func:`_shift_down`.
    Iterated, ``((Λᵀ)^r x)^(s) = r! C(s+r, r) x^(s+r)`` (zero once ``s + r > order``).'''
~~~

The function keeps its name, and only its letter changes. `:331` already says "the adjoint of `_shift_down`", which
is what `Λᵀ` encodes.

**BEFORE** (`:1377–1391`)

*`t3toolbox/backend/sampling_derivatives.py:1377–1391`*
~~~python
def _dxi_tilde_order_step(
        carry:  typ.Tuple[NDArray, NDArray],   # (dxi~, O): (2,)+W+K+C+(nOi,) ; C+(rLi,nOi,rR(i+1))
        xs:     typ.Tuple[NDArray, NDArray, NDArray, NDArray],  # (mu^(j), nu^(j), [U^u sigma~]^(j), [U^u tau~]^(j)): W+C+(rLi,) ; W+C+(rR(i+1),) ; (2,)+W+K+C+(rR(i+1),) ; (2,)+W+K+C+(rLi,)
) -> typ.Tuple[typ.Tuple[NDArray, NDArray], typ.Tuple]:   # (next carry, ())
    '''One order ``j`` of the two-row dxi~ correlation: ``dxi~^(u) += (mu^(j) O) . [U^u sigma~]^(j) +
    [U^u tau~]^(j) . (O nu^(j))`` for ``u in {0, 1}`` -- one order slice of ``mu O`` / ``O nu`` live (peak
    ``W*r*n``). Closure-free scan body (the core ``O`` rides the carry).'''
    dxi, O = carry
    mu_j, nu_j, sig_pair_j, tau_pair_j = xs
    nf = O.ndim - 3                                                    # O is C-only: pins len(C)
    mO = contractions.contract('WCa,Caib->WCib', mu_j, O)
    On = contractions.contract('Caib,WCb->WCai', O, nu_j)
    dxi = (dxi + contractions.contract('WCib,uWKCb->uWKCi', mO, sig_pair_j, len_C=nf)
           + contractions.contract('uWKCa,WCai->uWKCi', tau_pair_j, On, len_C=nf))
    return (dxi, O), ()
~~~

**After R2 (interim).** Code and widths are renamed, and the prose still says `O`:

~~~python
        carry:  typ.Tuple[NDArray, NDArray],   # (dxi~, O): (2,)+W+K+C+(nUi,) ; C+(rLi,nUi,rR(i+1))
    ...
    '''One order ``j`` of the two-row dxi~ correlation: ``dxi~^(u) += (mu^(j) O) . [U^u sigma~]^(j) +
    ...
    dxi, U = carry
    ...
    mU = contractions.contract('WCa,Caib->WCib', mu_j, U)
    Un = contractions.contract('Caib,WCb->WCai', U, nu_j)
~~~

**AFTER (R4b)**

~~~python
def _dxi_tilde_order_step(
        carry:  typ.Tuple[NDArray, NDArray],   # (dxi~, U): (2,)+W+K+C+(nUi,) ; C+(rLi,nUi,rR(i+1))
        xs:     typ.Tuple[NDArray, NDArray, NDArray, NDArray],  # (mu^(j), nu^(j), [(Λᵀ)^u sigma~]^(j), [(Λᵀ)^u tau~]^(j)): W+C+(rLi,) ; W+C+(rR(i+1),) ; (2,)+W+K+C+(rR(i+1),) ; (2,)+W+K+C+(rLi,)
) -> typ.Tuple[typ.Tuple[NDArray, NDArray], typ.Tuple]:   # (next carry, ())
    '''One order ``j`` of the two-row dxi~ correlation: ``dxi~^(u) += (mu^(j) U) . [(Λᵀ)^u sigma~]^(j) +
    [(Λᵀ)^u tau~]^(j) . (U nu^(j))`` for ``u in {0, 1}`` -- one order slice of ``mu U`` / ``U nu`` live (peak
    ``W*r*n``). Closure-free scan body (the core ``U`` rides the carry).'''
    dxi, U = carry
    mu_j, nu_j, sig_pair_j, tau_pair_j = xs
    nf = U.ndim - 3                                                    # U is C-only: pins len(C)
    mU = contractions.contract('WCa,Caib->WCib', mu_j, U)
    Un = contractions.contract('Caib,WCb->WCai', U, nu_j)
    dxi = (dxi + contractions.contract('WCib,uWKCb->uWKCi', mU, sig_pair_j, len_C=nf)
           + contractions.contract('uWKCa,WCai->uWKCi', tau_pair_j, Un, len_C=nf))
    return (dxi, U), ()
~~~

This is §3.2's example made real: `(mu^(j) U)·[(Λᵀ)^u sigma~]`, with one `U` and one shift, now distinguishable. The
contraction strings are byte-identical; their `u` is the jet row and is not a store (§1.5 item 6, K14). At
`:1403–1404` the carry `(dxi0, O)` becomes `(dxi0, U)`, while `_shift_up_pair(st, …)` on the next line stays. At
`:1431–1433` the docstring equation converts the same way, and its gloss becomes ``Λᵀ x = [(s+1) x^(s+1)]_s`` (row 1
of :py:func:`_shift_up_pair`). The public keyword of `compute_dxi_tilde_jets` at `:1419`, `down_tt_cores`, becomes
`up_tt_cores` (A1, a loud break listed in §2). Its comment `# O.` becomes `# U.` in R4b, and the parameter column
stays at 24.

**BEFORE** (`:1476–1479`)

*`t3toolbox/backend/sampling_derivatives.py:1476–1479`*
~~~python
        dxi_tildes:     typ.Sequence[NDArray],  # adjoint-var-down jets, len=d, elm_shape=(2,)+W+K+C+(nOi,)
        ww:             typ.Sequence[NDArray],  # probe vectors X,        len=d, elm_shape=W+(Ni,)
        pp:             typ.Sequence[NDArray],  # perturbation vectors P, len=d, elm_shape=W+(Ni,)
        etas:           typ.Sequence[NDArray],  # frame down jets, len=d, elm_shape=(order+1,)+W+C+(nOi,)
~~~

**AFTER (R2 widths, R4b wording)**

~~~python
        dxi_tildes:     typ.Sequence[NDArray],  # adjoint-var-down jets, len=d, elm_shape=(2,)+W+K+C+(nUi,)
        ww:             typ.Sequence[NDArray],  # probe vectors X,        len=d, elm_shape=W+(Ni,)
        pp:             typ.Sequence[NDArray],  # perturbation vectors P, len=d, elm_shape=W+(Ni,)
        etas:           typ.Sequence[NDArray],  # the η jets, len=d, elm_shape=(order+1,)+W+C+(nUi,)
~~~

On these lines:

- "frame down jets" is reworded, because it is on §3.3's closed list.
- "adjoint-var-down" stays. Inv 2 files it as an UNRELATED keep (a T4S travel direction), §3.3 does not list it, so
  K9's converse requires its `down` to survive, two lines above a width that now says `nU`.
- `P`, "perturbation vectors P", is the derivative direction and stays (§1.5 item 3). It is not the left core.

**What could go wrong.**

- **Rewrite order in the prose.** If a text pass applied `O→U` before the shift rewrite, the line would hold two
  `U`s of different meaning. §1.3 requires sentinels for exactly this. K7 (`\[U\^`, `\b[DU]\^[rku]\b`,
  `\b[DU] (?:x|z|jet)\b`) catches a leftover or mis-lettered shift.
- **K8 cannot see a bare `O`.** Its pattern needs a suffix (`\bO(?:\d|_[a-z]|ᵢ|i\b|\(d-1\))`), so
  `(mu^(j) O)`, `mu O`, `O nu` or "the core ``O`` rides the carry" surviving R4b passes every gate (§12).
- **Item 9 vs item 8.** §1.5 item 9 ("everything the inventories file as UNRELATED … stays") reads as protecting
  `:330–332`, which inv 2 files UNRELATED/keep. Item 8 and D-1 override it for the letters only. An implementer who
  reads item 9 alone will refuse the edit.

---

## 8. A LaTeX line in a math note: letters, `n^D_i`, and the T4S line

**Where:** `docs/exact_hessian_two_hole.tex:115–122` (§"Setup and notation", the frame paragraph).

**Governs:**

- §4.8 (R5, by hand, with sentinels): frame letters in the new meaning and `n^D_i`→`n^U_i`.
- D-1: no exempt note. D-7: `P`/`Q`→`L`/`R`.
- §3: the T4S line, placed in "each tex note's notation paragraph".
- §3.3/§3.4: `δU`→`δD`, and the point `(U,P)`→`(D,L)`.
- The listing `(up, down, left, right)` → `(down, up, left, right)` (tier-D pattern, applied by hand in a note).

**BEFORE**

*`docs/exact_hessian_two_hole.tex:115–122`*
~~~tex
\paragraph{The frame.} At a point $X$ the orthogonal frame (T4S \cite{T4S} App.~A.1, A.3, Algorithm~11; built by
\code{t3_orthogonal_representations}, which sweeps left-then-right where Algorithm~11 sweeps right-then-left ---
the representations are equivalent, and everything below is recomputed from the stored cores) is $(U_i, O_i, P_i, Q_i)$ = \code{T3Frame.data} = (up, down, left, right):
$U_i\in\R^{n_i\times N_i}$ with orthonormal rows, left-orthogonal $P_i\in\R^{r^L_{i-1}\times n_i\times r^L_i}$,
right-orthogonal $Q_i\in\R^{r^R_{i-1}\times n_i\times r^R_i}$, and $O_i\in\R^{r^L_{i-1}\times n^D_i\times r^R_i}$
orthonormal over its middle index ($n^D_i\le n_i$; $n^D_i<n_i$ is \emph{Tucker slack}, $r^R_i<r^L_i$ \emph{TT slack}).
The base point in its left-canonical gauge is $\th=(U,P)$ (\code{ManifoldGeometryOps.base_point}).
A tangent vector has gauged coordinates $v=(\delta U_i,\delta G_i)$, $\delta U_i\in\R^{n^D_i\times N_i}$,
~~~

**AFTER (R5)**

~~~tex
\paragraph{The frame.} At a point $X$ the orthogonal frame (T4S \cite{T4S} App.~A.1, A.3, Algorithm~11; built by
\code{t3_orthogonal_representations}, which sweeps left-then-right where Algorithm~11 sweeps right-then-left ---
the representations are equivalent, and everything below is recomputed from the stored cores) is $(D_i, U_i, L_i, R_i)$ = \code{T3Frame.data} = (down, up, left, right):
$D_i\in\R^{n_i\times N_i}$ with orthonormal rows, left-orthogonal $L_i\in\R^{r^L_{i-1}\times n_i\times r^L_i}$,
right-orthogonal $R_i\in\R^{r^R_{i-1}\times n_i\times r^R_i}$, and $U_i\in\R^{r^L_{i-1}\times n^U_i\times r^R_i}$
orthonormal over its middle index ($n^U_i\le n_i$; $n^U_i<n_i$ is \emph{Tucker slack}, $r^R_i<r^L_i$ \emph{TT slack}).
T4S writes $U$, $O$, $P$, $Q$ (and $\delta U$, $\delta G$) for our $D$, $U$, $L$, $R$ (and $V$, $H$).
The base point in its left-canonical gauge is $\th=(D,L)$ (\code{ManifoldGeometryOps.base_point}).
A tangent vector has gauged coordinates $v=(\delta D_i,\delta G_i)$, $\delta D_i\in\R^{n^U_i\times N_i}$,
~~~

The same R5 pass must also convert:

- `:103`'s raw cores $\th=((U_i),(G_i))$ → $((D_i),(G_i))$, or the note has two different `U_i` sixteen lines apart;
- `:110–111`'s $D\Phi$, $D^2\Phi$ → $\mathrm d\Phi$, $\mathrm d^2\Phi$ (D-11);
- the Lagrangian $\Lambda$ → $\ell$, **before** `\Dsh`/`\Ush` become `\Sh` = `\Lambda`, as §4.8 orders.

**What could go wrong.**

- **The gates see only part of this.** K8 catches a leftover `O_i`, `P_i`, `Q_i`, and K2 a leftover `n^O`. Nothing
  catches a missed `U_i`→`D_i` or `n^D_i`→`n^U_i`, since both spellings are valid in the new convention. Nothing
  catches `\delta U_i` or `\dot U_i` either: K8 has the Unicode `δU`/`U̇` but not their LaTeX spellings (§12).
- **Two collisions are missing from §4.8's list for this note.**
  - The new right store `R_i` meets the retraction $R_X(t\zeta)$ (`:288`), $R_x(p)$ (`:974, 976`) and Psenka &
    Boumal's $R_k^{-1}$ (`:382`).
  - The fixed T4S line says T4S's `δU` is our `V`, while §3.3/§3.4 make the note write `δD_i`. The note then gives
    our name for T4S's `δU` as `V` in one sentence and `δD_i` in the next. Ask Nick which to keep.

---

## 9. A sentence that uses both senses, rewritten from the rule

**Where:** `docs/contributor/padded_frame_internals.md:27–31`.

**Governs:**

- §4.2 (both senses, inv 3 C19). Inv 3 §6.1's row `:29-30` is tier E, applied in R5.
- The function names in the parentheses swap by tier A2 in **R2** (`_up_message`↔`_down_message`,
  `_up_edge`↔`_down_edge`; A tiers run over `docs/contributor/*`).
- D-5 changes the far frame `U′`→`F′`.
- §1.5 item 1 (`Φ`, `Ξ`) and item 3 (the completion `P`) stay.

**BEFORE**

*`docs/contributor/padded_frame_internals.md:27–31`*
~~~md
1. **Far-side messages, at the old frames** (`_down_message`, `_left_message_step`, `_right_message_step`,
   `_up_message`). On each directed edge `a → b`, `Φ = Pᵀ Z_far U′` (what the far side sends through the old far
   frame) and `Ξ`, a square root of the rest (`_root`, from a thin SVD; only `ΞΞᵀ` is used). The order is the down
   edges, the left messages from the far end, the right messages from the near end, then the up edges.
2. **New completions, in message order** (`_up_edge`, `_left_edge_step`, `_right_edge_step`, `_down_edge`). At
~~~

**AFTER (R2 for the names, R5 for the sentence and `F′`)**

~~~md
1. **Far-side messages, at the old frames** (`_up_message`, `_left_message_step`, `_right_message_step`,
   `_down_message`). On each directed edge `a → b`, `Φ = Pᵀ Z_far F′` (what the far side sends through the old far
   frame) and `Ξ`, a square root of the rest (`_root`, from a thin SVD; only `ΞΞᵀ` is used). The order is the up
   store's edges (`c_i → l_i`), the left messages from the far end, the right messages from the near end, then the
   down store's edges (`l_i → c_i`).
2. **New completions, in message order** (`_down_edge`, `_left_edge_step`, `_right_edge_step`, `_up_edge`). At
~~~

**Why this is a rewrite and not a swap.** Under the old rule a store was named for where its edge points, so "the
down edges" meant the edges `c_i → l_i`. Under the new rule a store is named for the side its cores occupy, and its
directed edge points *away* from that name. The up store's edges point down, from chain to leaf.

The word swap gives "The order is the up edges, …, then the down edges". It is literally consistent with the store
names, but a reader takes "up edge" as an edge pointing up, `l_i → c_i`, which is the down store's edge: backwards.
Naming the store *and* the arrow removes the ambiguity. The order itself matches the code after R2: `_up_message`
(formerly `_down_message`) runs first.

**What could go wrong.**

- **K9 passes the naive swap.** Clause (b) accepts any word whose aligned old token was the opposite store word. So
  "then the down edges" passes, and only the tier-E row and the reviewer stop it.
- **The same trap recurs.** It appears at `:42–43` ("Its up edge stacks the members' contents", which should become
  "Its down store's edge (`l_i → c_i`) stacks …") and in `padded_frame_cone.tex:403–404` (§4.2's own example).
- **`Φ` and `P` must stay.** D-5 renames `U′` but not the message `Φ`, and `P` here is the completion.

---

## 10. A DO-NOT-TOUCH name a naive implementer would rename: a T3's own `U` in `t3_linalg.py`

**Where:** `t3toolbox/backend/t3_linalg.py:556–558, 575–582` (`t3_mult` with truncation). For contrast, `:671–673`
and `:769–770` sit inside and beside tier C's range.

**Governs:**

- §1.5 item 11: `Ux`, `Uy`, `Uy_rev` in `t3_linalg.py` outside `660–760` are a T3's own Tucker cores.
- §1.5 item 7: `t3_down_orthogonalize_tucker_cores` stays.
- Tier C's `t3_linalg.py (660–760)` row: `U→D, O→U, BU→BD` inside the range only.
- Tier D (`(U, G, G, G)`, the §6.3 substitution) applies by pattern regardless of range.
- K15, and K14 with a caveat.

**BEFORE**, unchanged by the rename:

*`t3toolbox/backend/t3_linalg.py:556–558`*
~~~python
    Ux, Gx = x
    Uy, Gy = y
    d = len(Ux)
~~~
*`t3toolbox/backend/t3_linalg.py:575–582`*
~~~python
    # Leaves -> root (fold Tucker weight into the central TTs), then build/gauge the chain.
    Ux, Gx = ragged_orth.t3_down_orthogonalize_tucker_cores((Ux, Gx))
    Uy, Gy = ragged_orth.t3_down_orthogonalize_tucker_cores((Uy, Gy))
    Gx_o = orth.tt_left_orthogonalize(list(Gx))
    Gy_o = orth.tt_right_orthogonalize(tt_operations.tt_reverse(list(Gy)))
    Uy_rev = list(Uy[::-1])
    chain = [[Gx_o[i], Ux[i], i] for i in range(d)] \
          + [[Gy_o[j], Uy_rev[j], d - 1 - j] for j in range(d)]
~~~

**AFTER: byte-identical.** The tempting **wrong** edit reasons "these Tucker cores are being down-orthogonalized,
and the down store's letter is now `D`":

~~~python
    Dx, Gx = x                                                          # WRONG: T3-level, §1.5 item 11
    Dy, Gy = y
    ...
    Dx, Gx = ragged_orth.t3_down_orthogonalize_tucker_cores((Dx, Gx))
    Dy, Gy = ragged_orth.t3_down_orthogonalize_tucker_cores((Dy, Gy))
~~~

**Contrast:** 100 lines down, the same idiom is renamed, because it is inside tier C's range (the frame's down store):

*`t3toolbox/backend/t3_linalg.py:671–673`*
~~~python
    G_other, B_other, U = x
    xnp, _, _ = get_backend(False, tree_contains_jax(x))
    BU = xnp.einsum('...iz,...xz->...ix', B_other, U)
~~~

~~~python
    G_other, B_other, D = x
    xnp, _, _ = get_backend(False, tree_contains_jax(x))
    BD = xnp.einsum('...iz,...xz->...ix', B_other, D)
~~~

Tier D also rewrites the docstring just outside the range (R4a):

*`t3toolbox/backend/t3_linalg.py:769–770`*
~~~python
    The corewise transpose of the inner product: the frame transpose of ``<z, .>`` at the corewise frame
    ``(U, G, G, G)`` -- the Section 6.3 substitution ``(P, Q, O) -> G`` of Alger et al. (2026), "Tucker
~~~

~~~python
    The corewise transpose of the inner product: the frame transpose of ``<z, .>`` at the corewise frame
    ``(D, G, G, G)`` -- the Section 6.3 substitution ``(L, R, U) -> G`` of Alger et al. (2026), "Tucker
~~~

**Which gate catches the wrong rename.**

- **K15, certainly.** After R2 it is a positional NAME check: `Dx` at `:556` is a changed NAME at a
  (file, line, column) that R1's hit table does not list, since tier C's row is scoped to `660–760`. After R3–R5, it
  is an unlisted `difflib` hunk.
- **K14 only if R1 puts item 11 on `dnt.txt`.** §1.5's K14 paragraph says "the multiset of NAME tokens drawn from
  **items 1–10**". Items 11 and 12 were added in draft 10 and the range was not updated, so as written K14 does not
  cover `Ux`/`Uy`/`Uy_rev` (§12).
- **No gate, if the over-reach is in prose.** K14 and K15 are NAME-only. Renaming the docstring's
  ``BU_i = B_z,i U_i^T`` at `:815` reaches neither, and only the reviewer's trace rule catches it.

**What could go wrong even when done right.** After the rename the module has `BD = einsum(B_other, D)` in the
frame transpose (`:673`), and `BU = contract(z_tucker, tucker_cores)` in `t3_precompute_inner_product_sweep`
(`:837–838`) for θ's own cores at the corewise frame. Two names for one idiom, 160 lines apart, is correct by §1.5
item 11 and §3.3, but a reviewer will report it as a miss. Nick may want to rule on it, as D-6's residual case.

---

## 11. The riskiest edit: `n_{D,i}` → `n_{U,i}` in `shared_t3_math.tex`

**Where:** `docs/shared_t3_math.tex:365–367`, and the same symbol at `:406` and `:491`. The same R5 pass changes
`:357–358`'s `U_g`, `\dot U` → `D_g`, `\dot D`, with the complement basis `O_g` kept.

**Governs:** §4.8 ("`n_{D,i}`→`n_{U,i}`", "`U_i`, `U_g`, `U̇`→`D…`", "`O_g` kept"), D-8 (LaTeX `n_{D,i}` form) and
§4A(a) (every LaTeX equation with a width symbol), in R5.

**BEFORE**

*`docs/shared_t3_math.tex:365–367`*
~~~tex
\eqref{eq:tied} and in the projection \eqref{eq:gramsolve}. Note also that in
implementation $S_i \in \R^{n_g \times n_{D,i}}$ with
$n_{D,i} = \min(n_g,\ r_{i-1} r_i)$ --- rectangular in general, so an
~~~

**AFTER (R5)**

~~~tex
\eqref{eq:tied} and in the projection \eqref{eq:gramsolve}. Note also that in
implementation $S_i \in \R^{n_g \times n_{U,i}}$ with
$n_{U,i} = \min(n_g,\ r_{i-1} r_i)$ --- rectangular in general, so an
~~~

The same change applies at `:406`, `(\sum_{i\in g} n_{U,i}) \times n_g`, and at `:491`, "when the $n_{U,i}$ differ".

**Why this one.** It is the only edit where every weakness of the plan meets.

1. **The spelling survives and is valid in both conventions.** Left unconverted, `n_{D,i}` silently reads as the
   new down width, the Tucker factor's rows. The note would then state that the Tucker width is
   `min(n_g, r_{i-1} r_i)`, a false bound, with `M_g`'s row count at `:406` and the shape remark at `:491` inheriting
   it. A partial edit (`:366–367` done, `:406` missed) reads just as plausibly.
2. **No automated check reads it.** K2's LaTeX patterns ban only `n_\{O` and `n\^O`. K13 reads Python shape comments
   only. The battery and doctests never run a note, and `latexmk` only proves the PDF compiles. Unlike the code
   widths, this swap is not the script's (A3 says "LaTeX in its `n_{D,i}` form", by hand), so the bijection check
   does not count it either.
3. **It may never reach a human.** It reaches review only through §4A's inventory: K12 and Nick's R5 before/after
   cover only inventoried blocks. The scanner's candidate rules name "width symbols" by their code spelling, and
   this is an implementation remark about `S_i`, not one of the seeded "frame equations". The inventory row that
   does exist (inv 3 §6.8) is stale on both counts: it proposes the withdrawn word form `n_{\mathrm{up},i}`, and
   "keep" for the letters.
4. **It is entangled with the letter conversion around it.** The same lines must turn `U_g`/`\dot U` into
   `D_g`/`\dot D` while keeping `O_g`, in a note where `O` otherwise disappears. A slip either way (renaming `O_g`
   to `U_g`, or leaving `n_{D,i}` beside the new `D_g`) reads as consistent.

I rank this above the code edits because every code rename in the plan is under the bitwise battery, the suite and
K14/K15. The `update` letters and the silent attributes are user-facing risks the plan accepts on purpose.

**Recommendation.** Add the LaTeX width forms (`n_\{[DUO],`, `n\^[DUO]`) to the §4A scanner's candidate rules, and
require every hit in the six notes to be an inventoried block, so K12 and Nick's R5 review see each one.
Alternatively, add a report-mode check that lists every `n_{D`, `n^D`, `n_{U`, `n^U` in the notes for explicit
sign-off in the R5 handoff.

---

## 12. Plan issues surfaced while drafting these

1. **K13 vs K11 at R2** (item 1). R2's gate requires K13 clean, but the fenced field comments keep their old widths
   until R4a. That is eight failing lines: `frame_variations_format.py:156–159` and
   `uniform_frame_variations_format.py:110–113`. Fix: run K13 at R2 with fenced spans skipped.
2. **K14 says "items 1–10"** (item 10). §1.5 now has 12 items, and items 11 (a T3's own `U`) and 12 (`U` as
   uniform) are exactly the over-reach targets. Fix: "items 1–12".
3. **K8 has three blind spots** (items 7, 8, 10).
   - A bare `O` (`mu O`, `(dxi~, O)`, "the core ``O``"), because its `O` pattern needs a suffix.
   - The LaTeX spellings `\delta U`, `\dot U`.
   - The cited substitution written `(P, Q, O) -> G`, which K4 also misses.
4. **K12 granularity** (item 5). One block per diagram, or a half-done block passes.
5. **§4A(f)'s adjacency rule** (item 5). "Touching a TT-shaped frame core is `nU`" is wrong for the `L`/`R`–`D`
   edges. Say "touching an up core".
6. **K13's recognised forms** (item 3). These are `NAME: type  # …` and `- NAME = (…)`. The backend's commonest
   form, a tuple-element comment `typ.Sequence[NDArray],  # down_tucker_cores. len=d, elm_shape=…`, must be in the
   parser too.
7. **The exact-Hessian note** (item 8).
   - The collisions `R_i` vs the retraction `R_X`/`R_x` and `R_k^{-1}` are missing from §4.8's list.
   - The T4S line's "(δU, δG) for our (V, H)" disagrees with §3.3's `δD`.
8. **Item 9 vs D-1** (item 7). §1.5 item 9 protects everything inv 2 filed UNRELATED, including the shift letters
   (`U x`, `D^r`: inv 2 M7) that D-1 now changes. One sentence in item 9 would remove the conflict.
9. **The convention paragraph's source is stale.** Inv 3 §1(a), which §6 imports into `frame_variations.md`, still
   says the widths are `n_down`/`n_up`/`r_left`/`r_right` (K2 will catch this). It also says "the math notes written
   in its notation" keep T4S's letters, which D-1 reversed and no gate catches. This is the close second for §11.

## 13. Coverage: which plan mechanism each edit exercises

| mechanism | edits |
|---|---|
| A1 loud rename (field, keyword, comment label) | 1, 2 (bodies), 3, 7 (`down_tt_cores` keyword at `:1419`) |
| A2 same-spelling swap | 2, 9 (`_up_message`↔`_down_message` in docs) |
| A3 widths by script / by hand (fence, LaTeX) | script 3, 4, 7 · hand 1, 5, 8, 11 |
| B2 literals (`_STORE_NAMES`, test store words) | 4 |
| C NAME-letter maps (range- and file-scoped) | 7 (`O→U`, `mO→mU`, `On→Un`), 10 (`BU→BD` inside the range only) |
| D text patterns | 3 (`DPhi`), 10 (`(U, G, G, G)`, `(P, Q, O) -> G`), 8 (the store listing) |
| E hand rows (backwards adjectives, rewordings, both senses) | 6 (`:1371–1372`), 7 (§3.3), 9 |
| §4A blocks: field-comment / ascii / equation listing | 1 / 5 / 6 |
| D-1 `Λ` · D-2 `update` · D-5 `F′` · D-6 `(D,G,G,G)` · D-7 `L`,`R` · D-8 widths · D-11 `d` | 7 · 4 · 9 · 10 · 1, 8 · 1, 3, 5, 8, 11 · 3, 8 |
| §1.5 DO-NOT-TOUCH | 10 (main), 7 (`_shift_up*`, direction `P`), 9 (`Φ`, completion `P`), 6 (einsum strings), 3 (`rL`/`rR`, projector `Q`) |
| §4.10 realignment | 1, 3, 7 |
| Gates exercised | K2 (8, 11), K3 (4), K7 (7), K8 (6, 7, 8), K9 and its converse (6, 7, 9), K10 (1, 5), K10b (6), K11 (1, 5), K12 (5, 11), K13 (1, 3), K14/K15 (10), battery (2, 4) |
