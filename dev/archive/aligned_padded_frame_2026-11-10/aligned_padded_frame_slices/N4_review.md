# N4 review: the uniform start

Reviewer: Claude Opus 5.5, 2026-11-09. I reviewed the uncommitted working tree on top of cccc6f7c and the plan round 7
(97267ffd) against:
- the plan: §3.2, §3.4 (layout, start, `pad_safe_pinned_completion`), §4.5, §5.2, §5.3, §6 T7/T8/T16/T17, §12;
- the prototype `P/uniform_start.py` (`method='gs'`);
- `N4_handoff.md`.

## Summary

Correct, and in the house style. `ufv_padded_frame` reproduces the ragged start BITWISE on my own sweep of 172
randomized cases:
- old frames scattered at random own widths with gappy masks;
- old frames that are themselves in block layout, gappy in BOTH blocks;
- the zero tensor and sharing groups;
- `C = (3,)` stacks with ranks varying per element;
- garbage in every junk entry.

In every case the junk is exactly 0. It also equals the prototype bitwise. `pad_safe_pinned_completion` equals
`pinned_completion` on the real block to 1.1e-16 over 400 randomized cases, with interior pads in rows, pins and
seeds, NaN garbage, and degenerate seeds. The junk rows and unselected columns are bitwise 0.

The judgment call that drops the cumsum ordering key is right; the proof is below. There are only NITs.

## Style (checked first)

- **Basic types.**
  - In: uniform frame data (supercores, a static `shape` tuple, HOST bool masks), and target ranks as a sequence or a
    HOST int `(d,)+C` array.
  - Out: frame data plus the static `(Na, Rb)` ints.
  - `pad_safe_pinned_completion` takes arrays and host bool masks. `ufv_frame_widths` takes three ints and returns
    four.
- **Masks are the only truth about junk.**
  - The old counts are mask sums, per element.
  - The old frame is masked once (`ufv_apply_frame_masks`).
  - Output masks are `[old mask as given, padded | prefix(k)]`.
  - `pad_safe_pinned_completion` masks its inputs at entry with `where`, so even NaN junk is harmless; I checked.
- **No per-mode Python loop in the mathematics.** Each store is one batched completion over its modes and the stack,
  and the Tucker store is one completion batched over the groups. The only loops are host bookkeeping, as §4.5
  sanctions: the keyed Haar draws per group, the group checks, and `group_of`.
- **No object read for its shapes.** Only the old frame's own arrays are read, and they are used.
- **The QR** is confined to `pad_safe_pinned_completion`, with the justification of §3.4 in its docstring. Its host
  permutations come from host masks (`require_concrete_masks`; a traced mask raises the actionable error, tested in
  T16).
- **No contraction**, so no HISTORICAL rows. The only array operations are `where`, `take_along_axis`,
  `concatenate`, core-axis reshapes and `pad_or_truncate`.
- **Imports.** `common`, `linalg`, `ranks`, `sharing`, `ufv_masking`, `fv_operations` (`ZERO_FRAME_SEED`) and
  `padded_frames` (`_haar_seed_rows`, the shared host bookkeeping of §3.2). T18's allowed set holds.
- **Labels and docstrings.** No plan labels. Docstrings: one line on `ufv_padded_frame`, which gets a twin in N8;
  doctests on the untwinned `ufv_empty_frame`, `pad_safe_pinned_completion` and `ufv_frame_widths` (§3.6).
- **Signatures.** §5.2 and §5.3 match in content, except for the alignment noted in finding 1.

## Correctness

- **Layout (§3.4).**
  - `Na = max(U°, O°)` and `Rb = max(L°, R°)` over the padded widths; `Ka = max(kU, kD)` and `Kb = max(kL, kR)`.
  - The narrower old supercores get junk appended, and nothing moves.
  - The empty frame matches the §4.5 table.
- **Tucker.** One `pad_safe_pinned_completion` per group, with:
  - pins = the first mode's old rows;
  - seeds = the keyed Haar rows. The first `kU` rows of the `Ka`-row draw are the ragged `kU`-row draw, because the
    legacy stream is row-major.
  - row mask = the shape prefix.

  Every member takes its group's A-block and completion, so groups are tied bitwise.
- **TT and down.** Outside units are placed exactly at the real rows outside the A×A region (or at all real rows when
  the edge has no old column, as the ragged `c = 0` branch does). The rest are completed inside the A sub-grid from
  its real coordinate units. Carriers hold old content only.
- **Judgment call 1: no cumsum. Correct.**
  - A uniform slot axis is `[A-block | P-block]`. The real slots of each axis, in mask order, are order-isomorphic
    to the ragged indices (`ut3frame_to_t3frame` reads them in mask order): the A-block's real slots map to the
    ragged old indices `0..q−1`, and the P-block's real slots to the new ones.
  - Lexicographic order on pairs of order-isomorphic coordinates is preserved. So the flat order of the real rows of
    an unfolding equals the ragged row-major order, and "inside the old block" (`slot < split` on both axes)
    corresponds exactly to the ragged `[:q1, :q2]`.
  - Gaps anywhere in either block do not matter. Neither does an old block whose real slots are not a prefix.
  - My block-layout sweep (35 cases, gappy in both blocks) confirms this bitwise.
- **Judgment call 6: the dead check.** It was dead, and removing it is right.
  - The target store is at most the real rows of its unfolding: `rL(i+1) ≤ rL_i·nU_i` by the left sweep,
    `rR_i ≤ nU_i·rR(i+1)`, and `nD ≤ rL·rR`.
  - So `c + k_in ≤ q1·q2` whenever the outside units run out.
  - `pad_safe_pinned_completion`'s count check stands behind it anyway.
- **Judgment call 2** (jit vs numpy at 1e-12, plus bitwise on structural facts) follows the house rule (D-24).
  Accepted.
- **Judgment calls 3, 4, 5** accepted.

## Findings

### 1. NIT: `ufv_padded_frame`'s signature is misaligned and lacks §5.3's group break

`t3toolbox/backend/uniform_padded_frames.py:194`.

**Problem.** The `sharing:` line is one column short of the others (`sharing:` plus 8 spaces where 9 are needed).
§5.3 also has a blank line between `tt_ranks` and `sharing`. `signature_style.md` says names and types are always
aligned.

**Fix.** A whitespace edit.

### 2. NIT: two private helpers have two-sentence docstrings

`_coordinate_seeds` and `_store_completion`.

**Problem.** §3.1(4) asks for one-line docstrings in the backend.

**Fix.** Shorten each to one line. The detail fits in the return comments.

### 3. NIT, for the session: plan text to update

- §3.4 ("in their ragged row-major order (a cumsum of the masks)") and §3.2 / §4.2 ("the coordinate-seed order
  shared with the uniform start"): the uniform start uses the flat slot order of the real rows, provably the same,
  and does not import `_coordinate_seed_order`.
- §6 / §7 / §8 name the helper `tests/backend/_uniform_padded_scatter.py`; it is
  `_uniform_padded_frame_testing.py`.

## What I ran (env `t3toolbox`: py3.11, numpy 2.x, jax 0.10.2; `OPENBLAS_NUM_THREADS=1` for the scripts)

1. **My own scatter sweep,** `n4_scatter.py`. It uses its own scatter code, not the implementer's helper, and reads
   back through the library's `ut3frame_to_t3frame`.
   - 142 random cases at d = 1..4: 85 old frames at random own widths with gappy masks, 35 old frames in block layout
     (gappy in both blocks, junk in both), and 22 zero tensors. Every fifth case is shared (two groups), and the
     Haar seed varies.
   - Garbage of size 7·N(0,1) in every junk entry; N padded by 0 or 1.
   - Real parts against `fv_padded_frame`: **worst 0.0**. All junk exactly 0.
   - Plus 30 stacked cases (`C = (3,)`, ranks and gaps per element): **worst 0.0**, junk 0.
2. **Against the prototype,** `n4_proto.py`: `uniform_start(..., method='gs')` against `ufv_padded_frame` on d = 5,
   on outside-and-inside, and on sharing. Supercores **bitwise equal**, and masks and split equal. The prototype
   itself fails at d = 1, which it never supported; the library handles d = 1.
3. **`pad_safe_pinned_completion`,** `n4_psc.py`. 400 random cases with `C ∈ {(), (3,), (2,2)}`, random interior pads
   in rows, pins and seeds, 10 % NaN and 1e3-scale garbage, and seeds in span or zero.
   - Real block against `pinned_completion`: **1.1e-16**.
   - Junk rows bitwise 0; unselected columns 0.
4. **Tests.**
   - `test_uniform_padded_frame.py`, `test_linalg.py` and `test_import_graph.py`: **43 passed, 697 subtests**.
   - The four T16 rows: **passed** (the step's start row takes 10 s).
   - Doctests of `uniform_padded_frames`, `linalg` and `ufv_masking`: **17 passed**.
5. **Floor env** `tttt` (py3.9, numpy 1.22.4, jax 0.4.30): T7/T8/T17 **38 passed**. The four T16 rows: **4
   passed**.

The full suite was not run; the implementer reports 1,271 passed, 0 failed.

## Counts

MAJOR 0, MINOR 0, NIT 3.

VERDICT: APPROVE
