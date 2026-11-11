# The store rename plan — critic review, round 5 (diagrams and equations)

Reviewed: draft 5 (789 lines, 01:12): amended D-8 and the new §4A, with the slices' and checks' rows, read with
the diagrams in mind. For an independent baseline I scanned the tree myself: every `::` literal block in a docstring;
every docstring or comment line with `--`/`|` art, box characters, `delta_ij`/`I_xy`/`= 0` relations, einsum relations
or core letters with indices (about 115 equation lines in 26 files); every code fence and `::` block in the docs. Then
I tested the plan's gates against what I found.

**Sound.** (b) whole-block hand rewrites with before/after; (c) K10's rule — same line count and lengths, every
differing column a label character on both sides — is real and mechanical for blocks relabelled in place, and amended
D-8 keeps those labels the same length (`U0`→`D0`, `nU0`→`nD0`, `nO1`→`nU1`); (d) the example derivations are right
(`D_ia D_ja` summed over the leaf index; `U_aib U_ajb` over both bonds; they agree with `is_orthogonal`'s einsums);
(e) the DocTestFinder comparison; (f) a full read by the reviewer plus Nick's before/after at R4b and R5; (g)
fences for pass 1, tier D and the R4/R5 regexes, plus K11's no-sentinel check.

## MINOR

1. **K11 cannot pass on the field-comment blocks, and their column moves.** Fences are line ranges, and NAME tokens
   on a fenced line are renamed on purpose. Yet K11 requires the fenced block to be byte-identical to `v2026.4.0` after
   R2 and R3, located by its anchor, which is the first line's text. On the seeded field comments
   (`frame_variations_format.py:156-159`, `uniform_frame_variations_format.py:110-113`) the field names on those lines
   are A1 NAMEs, so the lines and the anchor change in R2 and K11 fails. R2 also moves the comments: `up_tucker_cores`
   grows by 2 and `down_tt_cores` shrinks by 2, so the four aligned `# len=d. B_xo B_yo = I_xy, …` comments go ragged
   until R4a.
   Fix:
   - Fence the COMMENT/STRING span (line, start column, end column), not the line. K11 compares the spans, and the
     anchor is the span's text.
   - §4.10's realignment absorbs a grown name into the padding (`up_tucker_cores:    ` → `down_tucker_cores:  `).
   - K10 checks such blocks for an unchanged comment start column.
2. **No gate catches a fenced block left unrewritten.** Inside a fence the old and the new letters are both legal
   (amended D-8). K2 carves fences out, K10 passes identical text, and K11 even demands identity through R3. Only K2a
   (`nO`) and K8 (`O`) can fire, and only on a block that has them. The first diagram of `T3Frame`
   (`frame_variations_format.py:86-96`: `nU0`, `U0`, `U1`, `Ud`, no `O`) left untouched passes every gate and reads
   backwards. Add a completion gate (K12) after R4b for the code and after R5 for the docs:
   - every inventoried block's "new lines" are filled, and the file's text equals them verbatim;
   - a block whose new text equals its `v2026.4.0` text carries a recorded reason (e.g. a T3 diagram in `B`, like
     `tucker_tensor_train.py:61-72`).
3. **Re-laid-out blocks have no mechanical check, and they are most of the equation blocks.** Any block holding an
   identifier or word that changes length is exempt from K10:
   - `frame_variations_format.py:70-78, 116-124, 353-357`: `up_tucker_cores` (15 characters) becomes
     `down_tucker_cores` (17), and `orthogonal:` becomes `down-orthogonal:`;
   - `probing.py:640-651`;
   - `ufv_operations.py:81-84`.

   Add K10b for those blocks. Same line count, and every alignment column of the old block must map to one column of
   the new block, the same for all its lines. An alignment column is one where at least two lines start a field after
   two or more spaces, or hold the same separator (`=`, `:`, `|`, `#`). It is mechanical and catches a ragged rewrite.
4. **Tier E's scripted locator can reach into a fence.** §1.4's locate/replace has no fence exclusion, and inv 1 §7
   has per-line rows inside the diagrams (e.g. `frame_variations_format.py:1266, 1271, 1273`, and the width labels at
   `:86-108`). The locator would rewrite fenced lines one at a time, against §4A(b) and (g). State that tier-E rows
   inside a fence feed the hand rewrite only and the locator skips fenced lines. §1.4's "tier D then runs over every
   line" should read "every unfenced line".
5. **K8's new `O` pattern hits uninventoried and kept sites.**
   - Hits with no disposition: `t3_orthogonalization.py:118-120` (`O_ab_x`, `Oaxb`: the TT core after a TT-up SVD)
     and `ut3_orthogonalization.py:201-204` (`O_ab_x`). These sit beside the T3's own `Uio`, so a mechanical
     `O`→`U` would give `U` two meanings in one function.
   - A deliberate keep it flags: `shared_t3_math.tex:358`'s `O_g` (kept by §4.8), which `\bO_[a-z]` matches.

   Extend §3.4's inventory to the letter `O`, with dispositions, and add `O_g` to K8's carve-outs.

## NIT

- **Blocks the seed list misses (the scanner will find most):**
  - diagrams: `frame_variations_format.py:843-853`, `:1262-1268` (beside the seeded :1271), `:1346-1359` (beside
    :1362); `uniform_frame_variations_format.py:1097-1110`; `tucker_tensor_train.py:61-72` (a T3, to dismiss with a
    reason);
  - a literal block of widths and store words: `ufv_operations.py:81-84`;
  - field-style relations: `probing.py:640-651, 1215-1226` (`U_xo U_yo = I_xy`);
  - a comment equation: `utv_operations.py:314-315` (`[U | dU] = [up | down]`);
  - the rank recurrences `ranks.py:644-647`, written in store words (`up_i = min(n_i, N_i)`, `down_i = …`). The
    scanner's rules see no letter, width or relation in them, and K9's `\bup\b` does not match `up_i`. Add equations
    with store-word variables to the scope.
  - Also, `frame_variations_format.py:1360-1374` already calls one store both `D2` (diagram) and `O0…O3` (listing);
    the rewrite unifies them.
- **Companion doctests.** `frame_variations_format.py:1302-1322` builds `(U0, U1, U2)`/`(D0, D1, D2)` for the
  diagram above it. These are CL rows, applied by §4.4 in R4a, and unfenced, yet the seed list names "1305–1322" while
  §4A(g) says no `>>>` line is fenced. Inventory them as companions, not fences, and have the reviewer check each
  companion's letters against its diagram: a swapped doctest still passes.
- **K10 should compare raw text, with no tab expansion.** The `user_guide.rst` diagrams are tab-indented.

VERDICT: REVISE (0 MAJOR, 5 MINOR)
