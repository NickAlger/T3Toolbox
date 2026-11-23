SIGN OFF

# M5 review, round 1: the types (comments only)

Reviewer (the sub-arc's second), 2026-11-23. Reviewed: the uncommitted M5 on `graded-messages` (over `0e249a6c`),
together with `M5_HANDOVER.md`, against plan §9.0 and §9.1 and `M_ARC_DECISIONS.md` 25.

My scripts are in `scratchpad/m2rev2/`:
- `tok_equal.py`: an independent check that no code changed;
- `mutants_m5.sh` and `mutants_m5.out`: my mutants.

**Verdict: SIGN OFF.**
- No library code changed, by a check independent of the implementer's.
- The types agree with §9.0, §9.1 and ruling 25.
- The law tests are independent and have teeth; my four mutants are killed.
- The engine's type column cannot drift silently.
- The gate and the tally check out.

Three minor points follow.

## 1. No library code changed

**The implementer's check.** `m5/ast_equal.py` (rerun by me) prints IDENTICAL CODE: 60 files, 17 changed in text
only.

**My own check, by a different route** (`tok_equal.py`):
- **The method.** Every changed `.py` under `t3toolbox/` at HEAD and in the tree is cut into tokens by `tokenize`.
  Comments and non-logical newlines are dropped, and the string tokens that are docstrings (located by the AST) are
  replaced by one marker. Every other token is compared, including indentation, defaults, annotations and string
  literals in code.
- **The result.** 17 files changed, all **token-identical**. No untracked file sits under `t3toolbox/`.
- **Two further checks.**
  - No `>>>` or `...` doctest line changed (`git diff`), so the doctests are the same code.
  - No library code reads a docstring at run time (`grep __doc__`), so a docstring edit cannot change behaviour.
    The only reader is the new test.

## 2. The types, sampled against §9.0 and §9.1 and the ruling

| Where | Written | Judgement |
| --- | --- | --- |
| forwards (`t3_probe`, TT `.probe`, …) | point `X`; `ww: P_in_i⊗V_i`; `z: P⊗V_i*` / `P` / `ℝ`; entries `index^d` | §9.1's `∏ P_in,j⊗V_j -> ⊕ P⊗V_i*`, and `index^d -> ℝ`: right |
| tangents (`tv_probe`, `*_corewise_tangent`) | directions `X`; the frame a role only; `Jp` / `J x_dot: P⊗V_i*` | `X × inputs -> Y*`, linear in `X`: right. The frame as a role only is ruling 25 (it is not in its own `X`) |
| the 42 (the lines M4 left) | blocks `X*`; ambient factors `E*` with `ww: V_i`; `T3Tangent` / `UT3Tangent` transposes "`X*`, in a T3Tangent at frame (before the projection)" | right. A frame tangent's coordinates are the `X*` blocks until projected, and "blocks: `X*`" replacing "gradients" is correct: they are `Jᵀc`, a gradient only at `c = M r` |
| `dense_*` | `T` in `E` | `E × inputs -> Y*`: right |
| converters | `derivatives_to_taylor`: "`Y*` in derivative coordinates (or `Y` in Taylor ones: the contragredient)"; `taylor_to_derivatives` the mirror | right. Dividing by `k!` takes a measurement from derivatives to Taylor, and an element of `Y` from Taylor to derivatives (`c_t = k! c_d`) |
| `prepare_derivative_data`, `data_block_norms`, `equalizing_weight` | `Y*` in, `Y*` out; `omega` a role; blocks `ℝ` per block | right; `omega` with a role only, as ruled (`omega` is not in `Y`; `omega**2` is `M`'s diagonal) |
| `affine_jet` | `ww`, `directions: V_i`; output `P_in_i⊗V_i` | right |
| the oracle (`query`, `LeastSquaresBatch`) | `r`, `Jp` `Y*`; `g`, `JtJp`, `Kp` `X*` (before the projection); `f`, `Jp_norm_sq`, `f_resolution` `ℝ`; block sums `ℝ` per block; `sample: P_in_i⊗V_i \| index^d`; `minibatch_scale: gamma` | §9.1's last row, word for word |
| `LocalModel` | `Jᵀ(M r)`, `<M JΠp, JΠp>`, `K_{Aᵀ M r}`, with `M = gamma * omega**2` once; the objective `½<r, M r>` (the letter `c` freed) | right |

**The engine's column.** Its values are `P`, `Y*`, `Y turned`, `P* turned`, `X*`.
- `w_tilde` is `Y turned`: per mode an element of `Y` (`P*⊗V_i`), with its jet axes turned.
- The other tilde variables are `P* turned`. They are dual jets **on edges** (paired with the edge jets `P`), not
  elements of `Y`. That refines plan §9.1's "every tilde variable is `Y` in turned form", and is correct: only
  `w_tilde` lives on the modes.
- The paragraph under the table gives each type's jet law, and the prose types `f`, `Jp_norm_sq`, `r_block_sumsq`
  (`ℝ`) and `g`, `JtJp`, `Kp` (`X*`).

## 3. The law tests

**Independence.** The movers live in `tests/_typing_laws.py`; the two new ones are `variations` (`V_i Δ_i^-1`) and
`dense` (`⊗Δ_i^-1`). None calls the library function under test.
- The contragredient check of the converters is a pairing identity, not a homogeneity:
  `<t2d(c), r> = <c, t2d(r)>` and `<d2t(c), t2d(r)> = <c, r>`, with the wrong pairing shown to differ. It catches a
  wrong factor, which a homogeneity law cannot.

**My mutants** (on a scratch copy, against `tests/test_sampling_typed.py`; unmutated 14 passed, 559 subtests):

| Mutant | Result |
| --- | --- |
| library: `equalizing_weight` returns `1/norm²` (a wrong power) | killed (4) |
| library: `taylor_to_derivatives` multiplies by `(k!)²` (a constant per slot, which no homogeneity sees) | killed (8: the contragredient pairing) |
| docstring: `nu_tilde` typed `P` in the engine's column | killed (9: the structural rule and both laws) |
| test side: the input law moves the probe weight as `V_i*` (not `V_i`) | killed (56) |

The implementer's nine (test-side movers, column entries, the converters swapped, `affine_jet`, the seed's flip)
complement these.

## 4. The column test: can it drift?

**Not silently.**
- **Completeness.** `type_column()` reads the table from the module docstring, and its name set must equal
  `sampling_variables._SHAPES`, an independent dict in the code. A new engine variable without a type fails.
- **Structure.** The structural rules (`mode` ⇔ `Y*`/`Y turned`; a core-shaped size ⇔ `X*`; a tilde name ⇔ turned)
  **determine** the column from each row's name, index and size. So no entry can change without failing them.
- **The laws.** They then confirm that this rule is the right one, for every public name of probe and apply, on a
  frame and on a point.
- **The other cells.** `test_sampling_graph` already pins the Index and Size cells the rules read.

The one blind spot: a duplicated row would be collapsed by the dict and still pass (minor 3).

## 5. The departures

- **The counts.** 99 signatures by script plus two classes by hand, against the note's 105. The six already typed in
  M1 and M2 needed nothing, and the edit log lists 219 `LEFT` comments. Fine.
- **Untyped:** `ut3_full_sum` and the unscale functions. They have no data-space argument. Fine. The unscale
  functions are the input law's own instance (`X -> X`) and could carry `X` later; not required.
- **"blocks: `X*`"** for the TT frontend's transpose returns, in place of "gradients". Right (§2).
- **The input law on edge variables only for probe and apply** (an entries index does not move). Correct, and stated
  in the test.

## 6. The gate and the tally

**The logs postdate the work.** The suite log (05:24) and the doc steps (05:25 on) come after the last edit (04:46).
- Suite: 1,979 passed, 0 failed.
- Module doctests: 325.
- `getting_started.rst` and the 47 pages: 0 failures.
- Examples: 14 pass.
- Sphinx `-W`: 0 warnings.

**My tally against M4's log.** 1,971 → 1,985 nodes, and exactly 14 differ, all new, all in `test_sampling_typed.py`.

## Minor

1. **One line over the house width.** `docs/contributor/signature_style.md:98`, rewrapped in this slice, is 130
   characters. Rewrap it to 120.
2. **For P19's list,** beside the handover's §5:
   - the glossary's `K` row and the pages' §4.12 and §4.14 still write `K_{Aᵀω²r}`, where the docstrings now say
     `K_{Aᵀ M r}`;
   - §9.1's table of function types (ruling 25, D1).
3. **`type_column()` could assert that no name repeats** (`len(rows) == len(set(names))`), so that a duplicated row
   cannot hide behind the dict. Optional.

Nothing is left running; the scratch copies are deleted. The working tree is as I found it, apart from this file.
