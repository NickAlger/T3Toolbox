# Diagram / equation inventory (store rename, R1; plan §4A)

Built by `dev/store_rename_slices/diagram_scanner.py` with the hand curation in `diagram_curation.py`; every
candidate line and its disposition is in `diagram_candidates.tsv`, the fence list the scripts consume is
`fences.tsv`. Old spans are of `v2026.4.0` (`docs/glossary.md`: of `88fbbedb`, absent at the tag).

**Totals:** 3464 candidate lines: 595 inside blocks, 2869 dismissed with a reason. **301 blocks**
(code 54, docs 11, tex 236; LaTeX equation line 229, LaTeX table or figure 7, ascii diagram 20, companion doctest 1, field-comment relation 6, index equation 22, markdown table 4, shape listing 12). **K16: 16 LaTeX width lines.**

How to use it (R4a/R4b/R5): each block is rewritten by hand, as a whole; fill `new span` and `new text` (verbatim),
and for a block whose new text equals its old text record the reason. K10 checks in-place blocks, K10b re-laid-out
ones, K11 that no script touched a fenced span, K12 that every block is filled
(`dev/store_rename_diagram_check.py`). Fenced `n` (the companion doctest) is inventoried for the reviewer but
renamed by R2's NAME pass (letter_sites rows). Kinds: ascii diagram, index equation, shape listing,
field-comment relation, markdown table (an added kind: symbol/shape tables in the docs; not column-aligned, so
K10/K10b do not apply), companion doctest, LaTeX equation line (one LaTeX display environment, or one paragraph run
of consecutive letter-bearing lines), LaTeX table or figure.


## Seed list resolution (§4A a)

| seed entry | resolved as |
|---|---|
| `frame_variations_format.py:56–66` | FVF-0058a/b/c (58–61, the TT-variation row) and FVF-0063a/b/c (63–66, the Tucker-variation row): six side-by-side diagrams, one column-ranged block per diagram (columns 8–31, 32–55, 56–end), so a half-done row cannot pass K12 |
| `:70–78` | **FVF-0071** (71–74, the frame listing) and **FVF-0077** (77–78, the variation listing): two listings, two blocks |
| `:86–96` | **FVF-0088** (88–96; 86–87 are the lead-in sentence and blank) |
| `:98–108` | **FVF-0100** (100–108) |
| `:116–124` | **FVF-0117** (117–120, the orthogonality relations) and **FVF-0123** (123–124, the gauge conditions); 116 and 122 are lead-in sentences |
| `:156–159` | **FVF-0156** (columns 48–end: the field comments; the field NAMEs stay outside the fence) |
| `:219–224` | **FVF-0217** (217–220) and **FVF-0222** (222–225): the two hole diagrams of `T3Frame.variation_shapes` |
| `:353–357` | **FVF-0354** (354–357) |
| `:843–853` | **FVF-0845** (845–848) and **FVF-0850** (850–853): the two hole diagrams of `T3Variations` fitting |
| `:1262–1274` | **FVF-1264** (1264–1267) and **FVF-1271** (1271–1274): the two `fv_to_t3` diagrams |
| `:1305–1322` / companion `:1302–1322` | **FVF-1302** (1302–1322), companion doctest, NOT fenced |
| `:1346–1365` | **FVF-1355** (1355–1358) and **FVF-1362** (1362–1365); the input-T3 diagram 1348–1351 (B/G only) is dismissed |
| `:1370–1374` | **FVF-1371** (1371–1374); the x listing 1368–1369 (B/G) and the variation listing 1376–1377 (V/H) carry no store letter |
| `uniform_frame_variations_format.py:110–113` | **UFVF-0110** (columns 49–end) |
| `uniform_frame_variations_format.py:1097–1125` | **UFVF-1106** (1106–1109), **UFVF-1113** (1113–1116), **UFVF-1122** (1122–1125); the input diagram 1099–1102 (B/G) dismissed |
| `manifold.py:685` | **M-0685** (685–686) |
| `uniform_manifold.py:478` | **UNMA-0478** (478–479) |
| `probing.py:640–651` | **P-0640** (640–643, cols 40–end), **P-0646** (646–649, cols 26–end), **P-0651** (651, cols 11–end) |
| `probing.py:1215–1226` | **P-1215** (1215–1218, cols 40–end), **P-1221** (1221–1224, cols 26–end), **P-1226** (1226, cols 11–end) |
| `ufv_operations.py:81–84` | **UFOP-0083** (83; 81 is the lead-in, 82/84 blank) |
| `utv_operations.py:314–315` | **UTVOPE-0314** (314–315) |
| `ranks.py:644–647` | **RA-0644** (644–647) |
| `tucker_tensor_train.py:61–72` | dismissed: the B/G diagram of a T3 (n, r, N, B, G only; no store letter or width), as the plan says |
| companion doctests | **FVF-1302** is the only one: no other docstring holds both a store diagram and example lines with frame letters (the scan of every doctest example line with a core letter found only this one and linalg's SVD U) |
| `frame_variations.md:13–17` and its symbol table | **DOC-FV-0014** (14–16; 13 and 17 are the code-fence markers) and **DOC-FV-0023** (23–30, the symbol table) |
| `user_guide.rst:124–142` | **DOC-UG-0125** (125–133, one displayed sum over two rows) and **DOC-UG-0139** (139–142, the frame-core list), as sample batch 3 edit 8 |
| `padded_frame_cone.tex` Table 1 | **TEX-PFC-0102** (102–116, the table float) |
| the exact-Hessian note's two tikz figures and its equation ranges | **TEX-EHTH-0151** (151–200) and **TEX-EHTH-0338** (338–379), the two figures; every letter-bearing equation line of the note is a block of its own |
| the four notes' frame equations | every letter-bearing equation line of the six notes is a TEX-* block (environments whole, text-mode runs by paragraph) |

## K16: every LaTeX width form in the notes (sign-off list for R5)

| block | file:line | text |
|---|---|---|
| TEX-EHTH-0117 | `docs/exact_hessian_two_hole.tex:118` | `$U_i\in\R^{n_i\times N_i}$ with orthonormal rows, left-orthogonal $P_i\in\R^{r^L_{i-1}\times n_i\times r^L_i}$,` |
| TEX-EHTH-0117 | `docs/exact_hessian_two_hole.tex:119` | `right-orthogonal $Q_i\in\R^{r^R_{i-1}\times n_i\times r^R_i}$, and $O_i\in\R^{r^L_{i-1}\times n^D_i\times r^R_i}$` |
| TEX-EHTH-0117 | `docs/exact_hessian_two_hole.tex:120` | `orthonormal over its middle index ($n^D_i\le n_i$; $n^D_i<n_i$ is \emph{Tucker slack}, $r^R_i<r^L_i$ \emph{TT slack}).` |
| TEX-EHTH-0117 | `docs/exact_hessian_two_hole.tex:122` | `A tangent vector has gauged coordinates $v=(\delta U_i,\delta G_i)$, $\delta U_i\in\R^{n^D_i\times N_i}$,` |
| TEX-EHTH-0117 | `docs/exact_hessian_two_hole.tex:123` | `$\delta G_i\in\R^{r^L_{i-1}\times n_i\times r^R_i}$ (code: \code{tucker_variations} $V$, \code{tt_variations} $H$),` |
| TEX-EHTH-0302 | `docs/exact_hessian_two_hole.tex:304` | `T_i = \ip{P_{>i}}{Q_{>i}}\in\R^{r^L_i\times r^R_i}\ (i<d),\qquad` |
| TEX-EHTH-0302 | `docs/exact_hessian_two_hole.tex:306` | `S_i^\T = \ip{O_i}{H_i}_{\{1,3\}}\in\R^{n^D_i\times n_i},` |
| TEX-EHTH-0888 | `docs/exact_hessian_two_hole.tex:893` | `\State $f\gets f_\eps(s_g)$, clip $\mathrm{rcond}=\mathrm{dtype}\,\eps\cdot\max(\sum_{i\in g}n^D_i,n_g)$ \Comment{\code{_tied_filter}}` |
| TEX-PFC-0102 | `docs/padded_frame_cone.tex:108` | `$\ell_i\to c_i$ & \code{up[i]} & $U_i\in\R^{n_{U,i}\times N_i}$ & the Tucker index \\` |
| TEX-PFC-0102 | `docs/padded_frame_cone.tex:109` | `$c_i\to \ell_i$ & \code{down[i]} & $O_i\in\R^{r_{L,i}\times n_{D,i}\times r_{R,i+1}}$ & the down index \\` |
| TEX-PFC-0102 | `docs/padded_frame_cone.tex:110` | `$c_i\to c_{i+1}$ & \code{left[i]}, $i\le d-2$ & $L_i\in\R^{r_{L,i}\times n_{U,i}\times r_{L,i+1}}$ & bond $i+1$ \\` |
| TEX-PFC-0102 | `docs/padded_frame_cone.tex:111` | `$c_i\to c_{i-1}$ & \code{right[i]}, $i\ge 1$ & $R_i\in\R^{r_{R,i}\times n_{U,i}\times r_{R,i+1}}$ & bond $i$ \\` |
| TEX-STM-0366 | `docs/shared_t3_math.tex:366` | `implementation $S_i \in \R^{n_g \times n_{D,i}}$ with` |
| TEX-STM-0366 | `docs/shared_t3_math.tex:367` | `$n_{D,i} = \min(n_g,\ r_{i-1} r_i)$ --- rectangular in general, so an` |
| TEX-STM-0404 | `docs/shared_t3_math.tex:406` | `\in \R^{(\sum_{i\in g} n_{D,i}) \times n_g},` |
| TEX-STM-0491 | `docs/shared_t3_math.tex:491` | `and, when the $n_{D,i}$ differ, even in shape, and no per-mode averaging` |

## Blocks


### `docs/batching_and_stacking.md`

#### DOC-BAS-0040

- file: `docs/batching_and_stacking.md`; old span: 40–46; kind: markdown table; fenced: y
- letters and widths: letters: Pᵢ Uᵢ
- roles: frame Tucker core `Uᵢ` → `Dᵢ`; frame TT core `Pᵢ` → `Lᵢ` (D-7); variation core `δUᵢ` → `δDᵢ` (D-12; or `Vᵢ`, the rewriter's choice); shapes use the T3's nᵢ, rᵢ (stay)
- re-laid out: n/a: a markdown table; K12 applies
- new == old possible: possible: record the reason at R4 if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
| Array | Blocks | Shape |
|---|---|---|
| frame Tucker core `Uᵢ` (`T3Frame`) | `C` | `(2,) + (nᵢ, Nᵢ)` |
| frame TT core `Pᵢ` (`T3Frame`) | `C` | `(2,) + (rᵢ, nᵢ, rᵢ₊₁)` |
| variation core `δUᵢ` (`T3Variations`) | `K + C` | `(3, 2) + (nᵢ, Nᵢ)` |
| probe vector `wᵢ` (`ww`) | `W` | `(4,) + (Nᵢ,)` |
| forward probe `zᵢ` (`tangent.probe(ww)`) | `W + K + C` | `(4, 3, 2) + (Nᵢ,)` |
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~


### `docs/exact_hessian_two_hole.tex`

#### TEX-EHTH-0103

- file: `docs/exact_hessian_two_hole.tex`; old span: 103; kind: LaTeX equation line; fenced: y
- letters and widths: U_i
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\paragraph{The format.} A T3 with raw cores $\th = \big((U_i)_{i=1}^d, (G_i)_{i=1}^d\big)$, $U_i\in\R^{n_i\times N_i}$,
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-0105

- file: `docs/exact_hessian_two_hole.tex`; old span: 105; kind: LaTeX equation line; fenced: y
- letters and widths: U_1, U_d
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
$X = \Phi(\th) = \mathrm{TT}(G_1,\dots,G_d)\times_1 U_1\cdots\times_d U_d$ (code: \code{TuckerTensorTrain.data}
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-0108

- file: `docs/exact_hessian_two_hole.tex`; old span: 108–112; kind: LaTeX equation line; fenced: y
- letters and widths: D, D^2
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\begin{equation}
  \label{eq:Dphi}
  D\Phi(\th)[\thh] = \sum_{a} \Phi(\th;\, a\leftarrow \thh_a), \qquad
  D^2\Phi(\th)[\thh,\thh'] = \sum_{a\neq b} \Phi(\th;\, a\leftarrow\thh_a,\, b\leftarrow\thh'_b),
\end{equation}
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-0117  (K16)

- file: `docs/exact_hessian_two_hole.tex`; old span: 117–123; kind: LaTeX equation line; fenced: y
- letters and widths: O_i, O_i,, P, P_i, P_i,, Q_i, U, U_i, U_i,, n^D_i, r^L_i, r^L_i}, r^L_{i-1}, r^R_i, r^R_i}, r^R_{i-1}
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- K16 width lines: 118, 119, 120, 122, 123
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
the representations are equivalent, and everything below is recomputed from the stored cores) is $(U_i, O_i, P_i, Q_i)$ = \code{T3Frame.data} = (up, down, left, right):
$U_i\in\R^{n_i\times N_i}$ with orthonormal rows, left-orthogonal $P_i\in\R^{r^L_{i-1}\times n_i\times r^L_i}$,
right-orthogonal $Q_i\in\R^{r^R_{i-1}\times n_i\times r^R_i}$, and $O_i\in\R^{r^L_{i-1}\times n^D_i\times r^R_i}$
orthonormal over its middle index ($n^D_i\le n_i$; $n^D_i<n_i$ is \emph{Tucker slack}, $r^R_i<r^L_i$ \emph{TT slack}).
The base point in its left-canonical gauge is $\th=(U,P)$ (\code{ManifoldGeometryOps.base_point}).
A tangent vector has gauged coordinates $v=(\delta U_i,\delta G_i)$, $\delta U_i\in\R^{n^D_i\times N_i}$,
$\delta G_i\in\R^{r^L_{i-1}\times n_i\times r^R_i}$ (code: \code{tucker_variations} $V$, \code{tt_variations} $H$),
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-0125

- file: `docs/exact_hessian_two_hole.tex`; old span: 125–129; kind: LaTeX equation line; fenced: y
- letters and widths: O_i, P, Q, U_i, U_j
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\begin{equation}
  \label{eq:E}
  \Emap v = \sum_i \big[P_{<i}\,\delta G_i\,Q_{>i}\big]\times_j U_j
          + \sum_i \big[P_{<i}\,O_i\,Q_{>i}\big]\times_i \delta U_i \times_{j\neq i} U_j ,
\end{equation}
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-0130

- file: `docs/exact_hessian_two_hole.tex`; old span: 130–132; kind: LaTeX equation line; fenced: y
- letters and widths: P, P_1, P_i, P_{i, Q, Q_d, Q_{i, U_1, U_i
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
where $P_{<i}=P_1\cdots P_{i-1}$ and $Q_{>i}=Q_{i+1}\cdots Q_d$ are TT chains, and $C\times_kM$ contracts
index $k$ of a core $C$ with the \emph{rows} of a matrix $M$ (so $\Phi(\th)=\mathrm{TT}(G)\times_1U_1\cdots$). The gauge conditions
$\delta U_i U_i^\T=0$ and $(P_i^{\langle 2\rangle})^\T\delta G_i^{\langle 2\rangle}=0$ ($i<d$; T4S (47)--(48)) are
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-0145

- file: `docs/exact_hessian_two_hole.tex`; old span: 145–146; kind: LaTeX equation line; fenced: y
- letters and widths: P_X
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
$\grad f = P_X\nabla f$ ($P_X=\Emap\Pi\Emap^\T$, coordinates $g=\Pi\Jac^\T(\omega^2\odot r)$, the model's
\code{misfit_gradient}), and the \emph{normal part} $N = (I-P_X)\nabla f$. The jets of the derivative kinds carry
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-0151

- file: `docs/exact_hessian_two_hole.tex`; old span: 151–200; kind: LaTeX table or figure; fenced: y
- letters and widths: U_1,2, U_2,3, U_2w_2, U_3,4, U_3}, U_4, U_4}
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\begin{figure}[t]
\centering
\begin{tikzpicture}[x=9.5mm,y=9.5mm]
  % (a) the sampled network
  \begin{scope}
    \foreach \i/\n in {1/G_1,2/G_2,3/G_3,4/G_4} \node[ttc] (g\i) at (\i,0) {$\n$};
    \foreach \i/\n in {1/U_1,2/U_2,3/U_3,4/U_4} \node[tkc] (u\i) at (\i,-1.15) {$\n$};
    \node[vecn] (w1) at (1,-2.1) {$w_1$}; \node[vecn] (w2) at (2,-2.1) {$w_2$};
    \node[resn] (z3) at (3,-2.1) {$c_3$}; \node[vecn] (w4) at (4,-2.1) {$w_4$};
    \foreach \i in {1,2,3} { \pgfmathtruncatemacro\j{\i+1} \draw[bnd] (g\i)--(g\j); }
    \foreach \i in {1,2,3,4} { \draw[bnd] (g\i)--(u\i); }
    \draw[bnd] (u1)--(w1); \draw[bnd] (u2)--(w2); \draw[bnd] (u3)--(z3); \draw[bnd] (u4)--(w4);
    \node[font=\small] at (2.5,0.85) {(a) $\langle c, A\,\Phi(\theta)\rangle$ (one probe term)};
  \end{scope}
  % (b) one hole
  \begin{scope}[xshift=52mm]
    \node[ttc] (g1) at (1,0) {$G_1$}; \node[holec] (g2) at (2,0) {}; \node[ttc] (g3) at (3,0) {$G_3$};
    \node[ttc] (g4) at (4,0) {$G_4$};
    \foreach \i/\n in {1/U_1,2/U_2,3/U_3,4/U_4} \node[tkc] (u\i) at (\i,-1.15) {$\n$};
    \node[vecn] (w1) at (1,-2.1) {$w_1$}; \node[vecn] (w2) at (2,-2.1) {$w_2$};
    \node[resn] (z3) at (3,-2.1) {$c_3$}; \node[vecn] (w4) at (4,-2.1) {$w_4$};
    \draw[bnd] (g1)--(g2); \draw[bnd] (g2)--(g3); \draw[bnd] (g3)--(g4);
    \foreach \i in {1,2,3,4} { \draw[bnd] (g\i)--(u\i); }
    \draw[bnd] (u1)--(w1); \draw[bnd] (u2)--(w2); \draw[bnd] (u3)--(z3); \draw[bnd] (u4)--(w4);
    \draw[dashed,gray,rounded corners=3pt] (0.55,0.45)--(1.45,0.45)--(1.45,-2.45)--(0.55,-2.45)--cycle;
    \node[font=\scriptsize,anchor=east] at (0.55,-1.15) {$\mu_1$};
    \node[font=\small] at (2.5,0.85) {(b) one hole: $\nabla_{G_2}$};
  \end{scope}
  % (c) two holes
  \begin{scope}[xshift=104mm]
    \node[ttc] (g1) at (1,0) {$G_1$}; \node[holec] (g2) at (2,0) {}; \node[ttc] (g3) at (3,0) {$G_3$};
    \node[dotc] (g4) at (4,0) {$\dot G_4$};
    \foreach \i/\n in {1/U_1,2/U_2,3/U_3} \node[tkc] (u\i) at (\i,-1.15) {$\n$};
    \node[tkc] (u4) at (4,-1.15) {$U_4$};
    \node[vecn] (w1) at (1,-2.1) {$w_1$}; \node[vecn] (w2) at (2,-2.1) {$w_2$};
    \node[resn] (z3) at (3,-2.1) {$c_3$}; \node[vecn] (w4) at (4,-2.1) {$w_4$};
    \draw[bnd] (g1)--(g2); \draw[bnd] (g2)--(g3); \draw[bnd] (g3)--(g4);
    \foreach \i in {1,2,3,4} { \draw[bnd] (g\i)--(u\i); }
    \draw[bnd] (u1)--(w1); \draw[bnd] (u2)--(w2); \draw[bnd] (u3)--(z3); \draw[bnd] (u4)--(w4);
    \node[font=\small] at (2.5,0.85) {(c) two holes: one term of $(\K_c\hat\theta)_{G_2}$};
  \end{scope}
\end{tikzpicture}
\caption{Cores as nodes, bonds as edges (TT cores on top, Tucker factors below, sample vectors as squares).
(a) One term of $\ip{c}{A\Phi(\th)}$ for a probe: the weighted residual $c_3$ sits on the free mode.
(b) The one-hole sweep (the corewise transpose) leaves one core out: the open legs are its gradient block, here
the left environment $\mu_1$ (dashed box), the projected probe $\xi_2=U_2w_2$ and a right environment.
(c) The two-hole sweep leaves one core out \emph{and} replaces another by its direction ($\dot G_4$, red); summing
over the replaced core $b\neq a$ gives block $a$ of $\K_c\thh$ (eq.~\eqref{eq:Kdef}).}
\label{fig:holes}
\end{figure}
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-0210

- file: `docs/exact_hessian_two_hole.tex`; old span: 210–211; kind: LaTeX equation line; fenced: y
- letters and widths: D
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
Let $\zeta,\zeta'\in T_X\M$ and let $\thh,\thh'$ be any raw directions with $D\Phi(\th)[\thh]=\zeta$,
$D\Phi(\th)[\thh']=\zeta'$. Then
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-0212

- file: `docs/exact_hessian_two_hole.tex`; old span: 212–216; kind: LaTeX equation line; fenced: y
- letters and widths: D^2
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\begin{equation}
  \label{eq:hess}
  \ip{\Hess f(X)[\zeta]}{\zeta'} = \ip{\omega\odot A\zeta}{\omega\odot A\zeta'}
     + \ip{N}{D^2\Phi(\th)[\thh,\thh']},
\end{equation}
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-0221

- file: `docs/exact_hessian_two_hole.tex`; old span: 221–222; kind: LaTeX equation line; fenced: y
- letters and widths: D, P, P_X
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
For an embedded submanifold, $\Hess f(X)[\zeta]=P_X\nabla^2 f\,\zeta+\W_X(\zeta,N)$ with the Weingarten map
$\W_X(\zeta,N)=P_X\,(D_\zeta P)\,N$, and $\ip{\W_X(\zeta,N)}{\zeta'}=\ip{N}{\mathrm{II}_X(\zeta,\zeta')}$
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-0225

- file: `docs/exact_hessian_two_hole.tex`; old span: 225–226; kind: LaTeX equation line; fenced: y
- letters and widths: D^2, P_XA
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
$\ip{N}{D^2\Phi[\thh,\thh']}=\ip{N}{\mathrm{II}_X(\zeta,\zeta')}$, which depends on $\zeta,\zeta'$ only. The first
term is $\ip{\zeta'}{P_XA^\T\omega^2A\zeta}=\ip{\omega A\zeta}{\omega A\zeta'}$ since $\zeta'$ is tangent.
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-0230

- file: `docs/exact_hessian_two_hole.tex`; old span: 230–232; kind: LaTeX equation line; fenced: y
- letters and widths: D, D^2
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
the gauge action ($D\Phi(\th)[V\th]=0$ for \emph{every} $\th$), satisfies, after differentiating that identity along
$\thh'$, $D^2\Phi[\thh_v,\thh']=-D\Phi[V\thh']\in T_X\M$, which $N$ annihilates. (At a slack frame
$\ker D\Phi(\th)$ is larger than the gauge orbit's tangent; Proposition~\ref{prop:weingarten} covers every lift.) No derivative of the frame (no SVD perturbation) enters --- this is the reason for the route
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-0237

- file: `docs/exact_hessian_two_hole.tex`; old span: 237–241; kind: LaTeX equation line; fenced: y
- letters and widths: D^2
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\begin{equation}
  \label{eq:Kdef}
  \K_Z\,\thh \;=\; \frac{d}{ds}\nabla_\th\ip{Z}{\Phi(\th+s\thh)}\Big|_{s=0},
  \qquad \ip{\thh'}{\K_Z\thh}=\ip{Z}{D^2\Phi(\th)[\thh,\thh']},
\end{equation}
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-0244

- file: `docs/exact_hessian_two_hole.tex`; old span: 244; kind: LaTeX equation line; fenced: y
- letters and widths: D^2
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
(Figure~\ref{fig:holes}c). Two instances are needed, and $\ip{N}{D^2\Phi}$ is their difference:
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-0258

- file: `docs/exact_hessian_two_hole.tex`; old span: 258–263; kind: LaTeX equation line; fenced: y
- letters and widths: D
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\begin{equation}
  \label{eq:lagrangian}
  \Lambda(\th) = f(\Phi(\th)) - \ip{\grad f(X)}{\Phi(\th)}, \qquad
  \nabla_\th\Lambda = D\Phi^\T N = 0,\qquad
  \nabla^2_\th\Lambda = D\Phi^\T\,\Hess f(X)\,D\Phi,
\end{equation}
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-0283

- file: `docs/exact_hessian_two_hole.tex`; old span: 283; kind: LaTeX equation line; fenced: y
- letters and widths: O, P_XD
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
$\ip{\grad f}{P_XD^2\Phi[\cdot,\cdot]}=O(\|\grad f\|)$ and depends on the lift. Level 2 is what the library ships;
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-0288

- file: `docs/exact_hessian_two_hole.tex`; old span: 288; kind: LaTeX equation line; fenced: y
- letters and widths: R_X
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\emph{second order}: $\frac{d^2}{dt^2}f(R_X(t\zeta))|_0=\ip{\Hess f[\zeta]}{\zeta}$. The archived design note
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-0302  (K16)

- file: `docs/exact_hessian_two_hole.tex`; old span: 302–307; kind: LaTeX equation line; fenced: y
- letters and widths: O_i}, P, P_d, P_i, Q, n^D_i, r^L_i, r^R_i}
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- K16 width lines: 304, 306
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\begin{equation}
  \label{eq:factors}
  T_i = \ip{P_{>i}}{Q_{>i}}\in\R^{r^L_i\times r^R_i}\ (i<d),\qquad
  H_i = P_i\times_3T_i\ (H_d=P_d),\qquad
  S_i^\T = \ip{O_i}{H_i}_{\{1,3\}}\in\R^{n^D_i\times n_i},
\end{equation}
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-0309

- file: `docs/exact_hessian_two_hole.tex`; old span: 309–310; kind: LaTeX equation line; fenced: y
- letters and widths: O_i, P, Q, U
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\code{tt_zipper_right_to_left}). Then $H_i=O_i\times_2S_i^\T$, the centers are the point's own core with the
orthogonality center at $i$ ($X=P_{<i}H_iQ_{>i}\times U$), $P_{>i}=T_i\,Q_{>i}$ (the left-canonical tail is
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-0316

- file: `docs/exact_hessian_two_hole.tex`; old span: 316; kind: LaTeX equation line; fenced: y
- letters and widths: P, U, U_i,
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
Define $\Lift v=\thh=(\dot U_i,\dot G_i)$, a variation of the base point $\th=(U,P)$, by
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-0317

- file: `docs/exact_hessian_two_hole.tex`; old span: 317–321; kind: LaTeX equation line; fenced: y
- letters and widths: U_i, U_i,
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\begin{equation}
  \label{eq:lift}
  \dot U_i = (S_i^\T)^{+}\,\delta U_i,\qquad
  \dot G_i = \delta G_i\times_3 T_i^{+}\ (i<d),\qquad \dot G_d=\delta G_d .
\end{equation}
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-0322

- file: `docs/exact_hessian_two_hole.tex`; old span: 322; kind: LaTeX equation line; fenced: y
- letters and widths: D
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
If every $S_i^\T$ has full row rank and every $T_i$ full column rank, then $D\Phi(\th)[\Lift v]=\Emap v$ for
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-0327

- file: `docs/exact_hessian_two_hole.tex`; old span: 327–329; kind: LaTeX equation line; fenced: y
- letters and widths: O_i, P, Q, U, U_i
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
TT slot: $P_{<i}\dot G_iP_{>i}=P_{<i}\,(\dot G_iT_i)\,Q_{>i}=P_{<i}\,\delta G_i T_i^+T_i\,Q_{>i}$, and $T_i^+T_i=I$
for full column rank. Tucker slot: replacing $U_i$ by $\dot U_i$ in $X=P_{<i}H_iQ_{>i}\times U$ gives
$P_{<i}(H_i\times_2\dot U_i)Q_{>i}=P_{<i}\big(O_i\times_2(S_i^\T\dot U_i)\big)Q_{>i}$, and $S_i^\T(S_i^\T)^+=I$
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-0338

- file: `docs/exact_hessian_two_hole.tex`; old span: 338–379; kind: LaTeX table or figure; fenced: y
- letters and widths: O_i, P, P_{i, Q, Q_{i, U_i
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\begin{figure}[t]
\centering
\begin{tikzpicture}[x=10mm,y=10mm]
  % (a) TT slot
  \begin{scope}
    \node[ttc] (a1) at (0,0) {$P_{i-1}$}; \node[dotc] (a2) at (1.2,0) {$\delta G_i$};
    \node[ttc] (a3) at (2.4,0) {$Q_{i+1}$}; \node[ttc] (a4) at (3.6,0) {$Q_{i+2}$};
    \draw[bnd] (-0.7,0)--(a1)--(a2)--(a3)--(a4)--(4.25,0);
    \foreach \x in {0,1.2,2.4,3.6} \draw[bnd] (\x,-0.38)--(\x,-0.8);
    \node at (4.75,0) {$=$};
    \node[ttc] (b1) at (5.8,0) {$P_{i-1}$}; \node[dotc] (b2) at (6.9,0) {$\dot G_i$};
    \node[matn] (bt) at (7.85,0) {$T_i$};
    \node[ttc] (b3) at (8.8,0) {$Q_{i+1}$}; \node[ttc] (b4) at (10.0,0) {$Q_{i+2}$};
    \draw[bnd] (5.2,0)--(b1)--(b2)--(bt)--(b3)--(b4)--(10.7,0);
    \foreach \x in {5.8,6.9,8.8,10.0} \draw[bnd] (\x,-0.38)--(\x,-0.8);
    \draw[decorate,decoration={brace,amplitude=4pt}] (7.6,0.45)--(10.4,0.45)
      node[midway,above=3pt,font=\scriptsize]{$=P_{i+1}P_{i+2}=P_{>i}$};
    \node[font=\small,anchor=west] at (-0.7,-1.25) {(a) TT slot: $\dot G_i=\delta G_i\times_3T_i^{+\eps}$, exact when $T_i^+T_i=I$ (TT slack: $T_i$ tall)};
  \end{scope}
  % (b) Tucker slot
  \begin{scope}[yshift=-24mm]
    \node[ttc] (c1) at (0,0) {$P_{i-1}$}; \node[ttc] (c2) at (1.2,0) {$O_i$}; \node[ttc] (c3) at (2.4,0) {$Q_{i+1}$};
    \node[dotc] (cu) at (1.2,-1.1) {$\delta U_i$};
    \draw[bnd] (-0.7,0)--(c1)--(c2)--(c3)--(3.05,0); \draw[bnd] (c2)--(cu)--(1.2,-1.75);
    \draw[bnd] (0,-0.38)--(0,-0.8); \draw[bnd] (2.4,-0.38)--(2.4,-0.8);
    \node at (3.55,0) {$=$};
    \node[ttc] (d1) at (4.5,0) {$P_{i-1}$}; \node[ttc] (d2) at (5.7,0) {$O_i$}; \node[ttc] (d3) at (6.9,0) {$Q_{i+1}$};
    \node[matn] (ds) at (5.7,-0.95) {$S_i^\T$};
    \node[dotc] (du) at (5.7,-1.9) {$\dot U_i$};
    \draw[bnd] (3.95,0)--(d1)--(d2)--(d3)--(7.6,0); \draw[bnd] (d2)--(ds)--(du)--(5.7,-2.55);
    \draw[bnd] (4.5,-0.38)--(4.5,-0.8); \draw[bnd] (6.9,-0.38)--(6.9,-0.8);
    \draw[dashed,rounded corners=3pt,gray] (5.2,0.45)--(6.25,0.45)--(6.25,-1.3)--(5.2,-1.3)--cycle;
    \node[font=\scriptsize,anchor=west] at (6.3,-1.15) {$=H_i$ (the center)};
    \node[font=\small,anchor=west] at (-0.7,-2.9) {(b) Tucker slot: $\dot U_i=(S_i^\T)^{+\eps}\delta U_i$, exact when $S_i^\T(S_i^\T)^+=I$ (Tucker slack: $S_i^\T$ wide)};
  \end{scope}
\end{tikzpicture}
\caption{The lift, slot by slot (red: the variation). (a) The frame's TT slot $P_{<i}\,\delta G_i\,Q_{>i}$ equals the
raw slot $P_{<i}\,\dot G_i\,P_{>i}$ because the left-canonical tail is $P_{>i}=T_iQ_{>i}$. (b) The frame's Tucker
slot pairs $\delta U_i$ with the orthonormal $O_i$; the raw slot pairs $\dot U_i$ with the center
$H_i=O_i\times_2S_i^\T$. The pseudo-inverses are the only place an inverse singular value enters the Hessian.}
\label{fig:lift}
\end{figure}
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-0382

- file: `docs/exact_hessian_two_hole.tex`; old span: 382; kind: LaTeX equation line; fenced: y
- letters and widths: R_k
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
$R_k^{-1}$ \cite{PB20}), so it is damped by the Tikhonov filter
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-0383

- file: `docs/exact_hessian_two_hole.tex`; old span: 383–387; kind: LaTeX equation line; fenced: y
- letters and widths: U
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\begin{equation}
  \label{eq:filter}
  f_\eps(\sigma)=\frac{\sigma}{\sigma^2+\eps^2}\ \ (0\text{ for }\sigma\le\mathrm{rcond}\cdot\sigma_1),\qquad
  M^{+\eps}=V f_\eps(\Sigma)U^\T=(M^\T M+\eps^2I)^{-1}M^\T,
\end{equation}
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-0394

- file: `docs/exact_hessian_two_hole.tex`; old span: 394; kind: LaTeX equation line; fenced: y
- letters and widths: D, Q
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
$\sim10^{-17}$) needs no special case: there the lift satisfies $D\Phi[\Lift v]=\Emap(Q_{\mathrm c}v)$ with $Q_{\mathrm c}$ the
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-0399

- file: `docs/exact_hessian_two_hole.tex`; old span: 399–400; kind: LaTeX equation line; fenced: y
- letters and widths: U_i, U_i,Y
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\paragraph{The transpose.} $\Lift^\T y$ maps raw core variations $y=(Y^U_i,Y^G_i)$ to \emph{ungauged}
coordinates, $\delta U_i=((S_i^\T)^{+\eps})^\T Y^U_i$, $\delta G_i=Y^G_i\times_3(T_i^{+\eps})^\T$,
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-0404

- file: `docs/exact_hessian_two_hole.tex`; old span: 404–425; kind: LaTeX equation line; fenced: y
- letters and widths: O, O_i}, P, P_i, Q, U, U_i, U_i,
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\begin{algorithm}[t]
\caption{The lift $\thh=\Lift_\eps v$ and its adjoint $\Lift_\eps^\T$ (per-frame factors once, per matvec two
contractions per mode)}
\label{alg:lift}
\begin{algorithmic}[1]
\Require orthogonal frame $(U,O,P,Q)$; relative damping $c\ge0$; coordinates $v=(\delta U_i,\delta G_i)$ (stack $\mathtt{K+C}$)
\Ensure raw directions $\thh=(\dot U_i,\dot G_i)$ of $\th=(U,P)$
\State $T_i\gets\ip{P_{>i}}{Q_{>i}}$ (zipper);\ $H_i\gets P_i\times_3T_i$;\ $S_i^\T\gets\ip{O_i}{H_i}_{\{1,3\}}$
   \Comment{\code{fv_center_factors}}
\State $\eps\gets c\,\|S_1^\T\|_F$;\ $(S_i^\T)^{+\eps},\ T_i^{+\eps}\gets$ one thin SVD each, filter \eqref{eq:filter}
   \Comment{\code{fv_lift_factors}}
\For{$i=1,\dots,d$} \Comment{\code{tv_corewise_lift}}
  \State $\dot U_i\gets(S_i^\T)^{+\eps}\,\delta U_i$ \Comment{\texttt{'Cux,KCxn->KCun'}}
  \State $\dot G_i\gets\delta G_i\times_3T_i^{+\eps}$ if $i<d$, else $\dot G_d\gets\delta G_d$ \Comment{\texttt{'KCaib,Cbc->KCaic'}}
\EndFor
\State adjoint: $\delta U_i\gets((S_i^\T)^{+\eps})^\T Y^U_i$,\ $\delta G_i\gets Y^G_i\times_3(T_i^{+\eps})^\T$
   \Comment{\code{tv_corewise_lift_transpose}}
\Impl{\code{fv_operations.fv_center_factors}, \code{fv_operations.fv_lift_factors} (\code{LiftFactors}),
\code{linalg.damped_pseudo_inverse}, \code{linalg.filtered_reciprocal}, \code{tv_operations.tv_corewise_lift[_transpose]};
uniform: \code{ufv_operations.ufv_lift_factors}, \code{utv_operations.utv_corewise_lift[_transpose]}.}
\end{algorithmic}
\end{algorithm}
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-0478

- file: `docs/exact_hessian_two_hole.tex`; old span: 478; kind: LaTeX equation line; fenced: y
- letters and widths: U
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
At the \emph{corewise frame} $(U,G,G,G)$ of raw cores $\th=(U,G)$ (\code{t3_corewise_frame}) the tangent
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-0480

- file: `docs/exact_hessian_two_hole.tex`; old span: 480; kind: LaTeX equation line; fenced: y
- letters and widths: O, P, Q
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
substitution $P,Q,O\to G$; \code{t3_probe_corewise_transpose}). With edge variables on bonds $0,\dots,d$ (core $i$
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-0482

- file: `docs/exact_hessian_two_hole.tex`; old span: 482–495; kind: LaTeX equation line; fenced: y
- letters and widths: U_i, U_ic_i,, U_iw_i,
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\begin{align}
  \label{eq:fwd}
  &\xi_i=U_iw_i,\quad \mu_i=\mu_{i-1}G_i(\xi_i),\ \mu_0=1,\quad \nu_{i-1}=G_i(\xi_i)\nu_i,\ \nu_d=1,\quad
   \eta_i=\mu_{i-1}G_i\nu_i, \\
  \label{eq:adj}
  &\ddet_i=U_ic_i,\quad \tilde\tau_i=\tilde\tau_{i-1}G_i(\xi_i)+\mu_{i-1}G_i(\ddet_i),\ \tilde\tau_0=0,\quad
   \tilde\sigma_{i-1}=G_i(\xi_i)\tilde\sigma_i+G_i(\ddet_i)\nu_i,\ \tilde\sigma_d=0, \\
  \label{eq:dxt}
  &\dxt_i=\tilde\tau_{i-1}G_i\nu_i+\mu_{i-1}G_i\tilde\sigma_i, \\
  \label{eq:asm}
  &\delta\tilde U_i=\textstyle\sum_{\mathtt W}\big(\eta_i\otimes c_i+\dxt_i\otimes w_i\big),\qquad
   \delta\tilde G_i=\sum_{\mathtt W}\big(\mu_{i-1}\otimes\xi_i\otimes\tilde\sigma_i
   +\tilde\tau_{i-1}\otimes\xi_i\otimes\nu_i+\mu_{i-1}\otimes\ddet_i\otimes\nu_i\big).
\end{align}
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-0500

- file: `docs/exact_hessian_two_hole.tex`; old span: 500; kind: LaTeX equation line; fenced: y
- letters and widths: U_i
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\xi_i\otimes\hat\sigma_i$, $\delta\tilde U_i=\sum_{\mathtt W}\delta\hat\xi_i\otimes w_i$ with
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-0503

- file: `docs/exact_hessian_two_hole.tex`; old span: 503–521; kind: LaTeX equation line; fenced: y
- letters and widths: U_i,, U_i,G_i, U_ic_i
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\begin{algorithm}[t]
\caption{The one-hole sweep (reference): the corewise probe transpose $\nabla_\th\ip{c}{A\Phi(\th)}$}
\label{alg:onehole}
\begin{algorithmic}[1]
\Require $\th=(U_i,G_i)$; probe vectors $w_i$ (stack $\mathtt W$); weighted residuals $c_i=\omega_i^2\odot r_i$
\Ensure $(\delta\tilde U_i,\delta\tilde G_i)$, $\mathtt W$ summed
\State $\xi_i,\mu_i,\nu_i,\eta_i$ by \eqref{eq:fwd} \Comment{frame sweep (T4S Alg.~5); \code{tv_precompute_probe_frame_sweep}}
\State $\ddet_i\gets U_ic_i$ \Comment{\code{compute_deta_tilde}}
\For{$i=1,\dots,d$} $\tilde\tau_i\gets\tilde\tau_{i-1}G_i(\xi_i)+\mu_{i-1}G_i(\ddet_i)$ \Comment{\code{compute_tau_tilde}}
\EndFor
\For{$i=d,\dots,1$} $\tilde\sigma_{i-1}\gets G_i(\xi_i)\tilde\sigma_i+G_i(\ddet_i)\nu_i$ \Comment{\code{compute_sigma_tilde}}
\EndFor
\State $\dxt_i\gets\tilde\tau_{i-1}G_i\nu_i+\mu_{i-1}G_i\tilde\sigma_i$ \Comment{\code{compute_dxi_tilde}}
\State $\delta\tilde U_i,\ \delta\tilde G_i$ by \eqref{eq:asm} \Comment{\code{assemble_tucker_variations}, \code{assemble_tt_variations}}
\Impl{\code{probing.t3_probe_corewise_transpose} = \code{probing.tv_probe_transpose} at \code{fv_conversions.t3_corewise_frame}
(T4S \S6.2.3, Algorithm~8; \S6.3). Jets: \code{sampling_derivatives.t3_probe_corewise_derivatives_transpose},
\code{tv_probe_transpose_derivatives_from_sweep}.}
\end{algorithmic}
\end{algorithm}
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-0527

- file: `docs/exact_hessian_two_hole.tex`; old span: 527; kind: LaTeX equation line; fenced: y
- letters and widths: U_i,
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
derivative along $\thh=(\dot U_i,\dot G_i)$, i.e.\ the \emph{product rule} applied to every line of
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-0535

- file: `docs/exact_hessian_two_hole.tex`; old span: 535; kind: LaTeX equation line; fenced: y
- letters and widths: U_iw_i
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
  $\dot\xi_i=\dot U_iw_i$,\
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-0539

- file: `docs/exact_hessian_two_hole.tex`; old span: 539; kind: LaTeX equation line; fenced: y
- letters and widths: O, P, Q, U
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
  --- exactly T4S Algorithm~7 ($\sigma,\tau,\delta\eta$) with $Q=O=P=G$ and variation $(\dot U,\dot G)$. The
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-0543

- file: `docs/exact_hessian_two_hole.tex`; old span: 543; kind: LaTeX equation line; fenced: y
- letters and widths: U_i
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
  $\frac{d}{ds}\delta\tilde U_i=\sum_{\mathtt W}(\dot\eta_i\otimes c_i+\dot{\dxt}_i\otimes w_i)$, and $\delta\tilde G_i$
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-0545

- file: `docs/exact_hessian_two_hole.tex`; old span: 545; kind: LaTeX equation line; fenced: y
- letters and widths: O, P, Q, U
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
  the manifold frame $(U,O,P,Q)$ and the corewise frame $(U,P,P,P)$ of the base point), $\nu$ is re-swept through
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-0551

- file: `docs/exact_hessian_two_hole.tex`; old span: 551; kind: LaTeX equation line; fenced: y
- letters and widths: D
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\paragraph{The block-sum oracle.} Since $D\Phi(\th)[\thh]=\Phi(Y)$ for the doubled-rank T3 $Y$ that
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-0554

- file: `docs/exact_hessian_two_hole.tex`; old span: 554–557; kind: LaTeX equation line; fenced: y
- letters and widths: B
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\begin{equation}
  \label{eq:blocksum}
  \K_c\thh = B^\T\,\nabla_Y\ip{c}{A\Phi(Y)},
\end{equation}
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-0558

- file: `docs/exact_hessian_two_hole.tex`; old span: 558; kind: LaTeX equation line; fenced: y
- letters and widths: B
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
$B^\T$ summing the gradient blocks where $\th$'s cores sit. This one-hole-only identity (check C9b, $3.4\times10^{-16}$)
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-0573

- file: `docs/exact_hessian_two_hole.tex`; old span: 573–576; kind: LaTeX equation line; fenced: y
- letters and widths: U_ip_i, U_iw_i,, U_iw_i,U_ip_i, \Dsh, \Ush
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
Write the down-shift $(\Dsh x)^{(t)}=t\,x^{(t-1)}$ and its adjoint, the up-shift $(\Ush x)^{(s)}=(s+1)\,x^{(s+1)}$
($x^{(m+1)}=0$), so that $\Dsh^r x=r!\binom{t}{r}x^{(t-r)}$ and $(\Ush^rx)^{(s)}=r!\binom{r+s}{r}x^{(r+s)}$
(\code{_shift_down}, \code{_shift_up}). The input jets are affine, $\xi_i=(U_iw_i,U_ip_i)$,
$\dot\xi_i=(\dot U_iw_i,\dot U_ip_i)$ (\code{build_input_jets}). Every jet product is one of four forms:
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-0577

- file: `docs/exact_hessian_two_hole.tex`; old span: 577–584; kind: LaTeX equation line; fenced: y
- letters and widths: \Dsh, \Ush
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\begin{align}
  \label{eq:band}
  &\text{banded (one affine factor } b\text{):} && (a\star b)^{(t)}=a^{(t)}b^{(0)}+(\Dsh a)^{(t)}b^{(1)},
     && (c\star^\T b)^{(s)}=c^{(s)}b^{(0)}+(\Ush c)^{(s)}b^{(1)},\\
  \label{eq:conv}
  &\text{full (both jets):} && (a\ast b)^{(t)}=\textstyle\sum_r\frac{a^{(r)}}{r!}(\Dsh^r b)^{(t)},
     && (a\ast^\T z)^{(s)}=\textstyle\sum_r\frac{a^{(r)}}{r!}(\Ush^r z)^{(s)},
\end{align}
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-0600

- file: `docs/exact_hessian_two_hole.tex`; old span: 600; kind: LaTeX equation line; fenced: y
- letters and widths: U_ic_i
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
and its product rule along $(\dot G,\dot\xi,\dot\mu,\dot{\ddet})$, with $\dot{\ddet}_i=\dot U_ic_i$, is the dotted step
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-0612

- file: `docs/exact_hessian_two_hole.tex`; old span: 612; kind: LaTeX equation line; fenced: y
- letters and widths: \Ush
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
pairs them with $(w,p)$), and only those are formed: with $[\Ush^0x]=x$, $[\Ush^1x]=\Ush x$,
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-0613

- file: `docs/exact_hessian_two_hole.tex`; old span: 613–617; kind: LaTeX equation line; fenced: y
- letters and widths: \Ush
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\begin{equation}
  \label{eq:dxitilde}
  \dxt_i^{(u)}=\sum_j\mu_{i-1}^{(j)}G_i\,[\Ush^u\tilde\sigma_i]^{(j)}+[\Ush^u\tilde\tau_{i-1}]^{(j)}G_i\,\nu_i^{(j)},
  \qquad u\in\{0,1\},
\end{equation}
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-0619

- file: `docs/exact_hessian_two_hole.tex`; old span: 619; kind: LaTeX equation line; fenced: y
- letters and widths: \Ush
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
$d(\mu G)[\Ush^u\tilde\sigma]+\mu G[\Ush^u\dot{\tilde\sigma}]+[\Ush^u\dot{\tilde\tau}]G\nu+[\Ush^u\tilde\tau]d(G\nu)$,
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-0622

- file: `docs/exact_hessian_two_hole.tex`; old span: 622–643; kind: LaTeX equation line; fenced: y
- letters and widths: O, P, Q, U, U_ip_i, U_iw_i,
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\begin{algorithm}[t]
\caption{Two-hole, forward half: the dotted forward jets along $\thh=(\dot U,\dot G)$}
\label{alg:fwd}
\begin{algorithmic}[1]
\Require $\th=(U,G)$; $\thh$ (stack $\mathtt{K+C}$); $w_i,p_i$ (stack $\mathtt W$); order $m$; model sweep $(\xi,\mu)$
\Ensure $\dot\xi_i,\dot\mu_i,\dot\nu_i,\dot\eta_i$ (orders $0..m$) and the undotted $\nu_i$
\State $\dot\xi_i\gets(\dot U_iw_i,\ \dot U_ip_i)$ \Comment{\code{build_input_jets}, \code{compute_xi}}
\State $\nu_{i-1}\gets G_i(\xi_i)\star\nu_i$ \Comment{re-swept through $G$: \code{compute_nu_jets}}
\For{$i=1,\dots,d$}
  \State $\dot\mu_i\gets\dot\mu_{i-1}\star G_i(\xi_i)+\mu_{i-1}\star\dot G_i(\xi_i)+\mu_{i-1}\star G_i(\dot\xi_i)$
  \Comment{\code{compute_sigma_jets}$(\dot G,G,G,\dots)$}
\EndFor
\For{$i=d,\dots,1$}
  \State $\dot\nu_{i-1}\gets G_i(\xi_i)\star\dot\nu_i+\dot G_i(\xi_i)\star\nu_i+G_i(\dot\xi_i)\star\nu_i$
  \Comment{\code{compute_tau_jets}}
\EndFor
\State $\dot\eta_i\gets\dot\mu_{i-1}G_i\ast\nu_i+\mu_{i-1}\dot G_i\ast\nu_i+\mu_{i-1}G_i\ast\dot\nu_i$
  \Comment{\code{compute_deta_jets} (order scan)}
\Impl{\code{sampling_derivatives}: the GN tangent forward (T4S Alg.~7 jet-ified) at $Q=O=P=G$; plain kinds
\code{probing.compute_dxi}, \code{compute_sigma}, \code{compute_tau}, \code{compute_deta}.}
\end{algorithmic}
\end{algorithm}
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-0645

- file: `docs/exact_hessian_two_hole.tex`; old span: 645–662; kind: LaTeX equation line; fenced: y
- letters and widths: U_ic_i
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\begin{algorithm}[t]
\caption{Two-hole, adjoint half: the dotted adjoint sweeps}
\label{alg:adj}
\begin{algorithmic}[1]
\Require undotted $\xi,\mu,\nu$ and adjoint state $\ddet,\tilde\tau,\tilde\sigma$ at $\th$; dotted $\dot\xi,\dot\mu,\dot\nu$; residual jets $c_i$
\Ensure $\dot{\ddet}_i,\dot{\tilde\tau}_i,\dot{\tilde\sigma}_i$ and the two read rows $\dot{\dxt}_i^{(0,1)}$
\State $\dot{\ddet}_i\gets\dot U_ic_i$ \Comment{order-diagonal; \code{compute_deta_tilde_tangent_jets}}
\For{$i=1,\dots,d$} \Comment{carry $\dot{\tilde\tau}$; the undotted $\tilde\tau_{i-1}$ rides as scan data}
  \State $\dot{\tilde\tau}_i\gets$ \eqref{eq:tautildedot} \Comment{\code{compute_tau_tilde_tangent_jets}, step \code{_adj_tilde_tangent_step}}
\EndFor
\State $\dot{\tilde\sigma}\gets$ the same sweep on the reversed train \Comment{\code{compute_sigma_tilde_tangent_jets}}
\State $\dot{\dxt}_i^{(u)}\gets$ the product rule of \eqref{eq:dxitilde}, $u\in\{0,1\}$ \Comment{\code{compute_dxi_tilde_tangent_jets}}
\Impl{\code{sampling_derivatives}: \code{_adj_prop_tangent}, \code{_adj_tangent_src_order_step},
\code{_adj_tangent_sweep_scanned}, \code{_dxi_tilde_tangent_order_step}; plain probe: \code{probing.compute_deta_tilde_tangent},
\code{compute_tau_tilde_tangent} (two \code{_sigma_step} calls per core), \code{compute_sigma_tilde_tangent},
\code{compute_dxi_tilde_tangent} (two \code{compute_deta} calls).}
\end{algorithmic}
\end{algorithm}
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-0668

- file: `docs/exact_hessian_two_hole.tex`; old span: 668–672; kind: LaTeX equation line; fenced: y
- letters and widths: U_i}
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\begin{equation}
  \label{eq:tuckerdot}
  (\K_c\thh)_{U_i}=\sum_{\mathtt W}\Big(\sum_t\dot\eta_i^{(t)}\otimes c_i^{(t)}+\sum_{u\in\{0,1\}}\dot{\dxt}_i^{(u)}\otimes w_i^{(u)}\Big),
  \qquad w^{(0)}=w,\ w^{(1)}=p .
\end{equation}
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-0696

- file: `docs/exact_hessian_two_hole.tex`; old span: 696–701; kind: LaTeX equation line; fenced: y
- letters and widths: \Ush
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\begin{equation}
  \label{eq:asmjet}
  \delta\tilde G_i=\sum_{\mathtt W}\Big(\sum_{k\in\{0,1\}}\Pi_{i,k}\otimes_2\xi_i^{(k)}
     +\sum_r\mu_{i-1}^{(r)}\otimes F_i^{(r)}\Big),\quad
  \Pi_{i,k}=\sum_r\mu_{i-1}^{(r)}\otimes[\Ush^k\tilde\sigma_i]^{(r)}+\sum_s[\Ush^k\tilde\tau_{i-1}]^{(s)}\otimes\nu_i^{(s)},
\end{equation}
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-0702

- file: `docs/exact_hessian_two_hole.tex`; old span: 702–703; kind: LaTeX equation line; fenced: y
- letters and widths: \Ush
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
with the fold $F_i^{(r)}=\sum_s\binom{r+s}{r}\ddet_i^{(r+s)}\otimes\nu_i^{(s)}=\sum_s\frac{1}{r!}(\Ush^r\ddet_i)^{(s)}\otimes\nu_i^{(s)}$
(\code{_tt_det_lag_step}, which carries $\Ush^r\ddet$). Its derivative (\code{_tt_tangent_variation_jets_step},
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-0744

- file: `docs/exact_hessian_two_hole.tex`; old span: 744–764; kind: LaTeX equation line; fenced: y
- letters and widths: U
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\begin{algorithm}[t]
\caption{The probe-derivative two-hole $\K_c\thh$ (driver)}
\label{alg:driver}
\begin{algorithmic}[1]
\Require $c_i=\omega^2\odot r_i$ (jets, $\mathtt{W+C}$); $w,p$; $\th=(U,G)$; $\thh$ ($\mathtt{K+C}$); order $m$; model sweep $(\xi,\mu)$
\Ensure $\K_c\thh$, raw core variations ($\mathtt{K+C}$, $\mathtt W$ summed)
\State undotted: $\nu$ re-swept through $G$; $\ddet=Uc$; $\tilde\tau$, $\tilde\sigma$ \Comment{GN kernels, recompute (I-11)}
\State dotted forward $\dot\xi,\dot\mu,\dot\nu,\dot\eta$ \Comment{Algorithm~\ref{alg:fwd}}
\State dotted adjoint $\dot{\ddet},\dot{\tilde\tau},\dot{\tilde\sigma},\dot{\dxt}^{(0,1)}$ \Comment{Algorithm~\ref{alg:adj}}
\State Tucker block \eqref{eq:tuckerdot}; TT block \eqref{eq:asmjetdot}, $\mathtt W$-chunked \Comment{Section~\ref{sec:assembly}}
\Impl{\code{sampling_derivatives.t3_probe_corewise_derivatives_hessian[_from_sweep]}; step 1 \code{compute_nu_jets},
\code{compute_deta_tilde_jets}, \code{compute_tau_tilde_jets}, \code{compute_sigma_tilde_jets}; step 4
\code{assemble_tucker_tangent_variation_jets}, \code{assemble_tt_tangent_variation_jets}; the other kinds:
\code{probing.t3_probe_corewise_hessian[_from_sweep]}, \code{apply.t3_apply_corewise_hessian[_from_sweep]}
(\code{_apply_corewise_hessian_adjoint}), \code{entries.t3_entries_corewise_hessian[_from_sweep]},
\code{sampling_derivatives.t3_apply_corewise_derivatives_hessian[_from_sweep]}, \code{t3_entries_corewise_derivatives_hessian[_from_sweep]}
(\code{_apply_derivatives_hessian_from_jets}, \code{compute_sigma_hat_tangent_jets} with step
\code{_sigma_hat_tangent_jets_step}); the hook \code{SamplingKind.corewise_hessian} (the required sixth operation)
in \code{fitting.py}, \code{uniform_fitting.py}.}
\end{algorithmic}
\end{algorithm}
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-0776

- file: `docs/exact_hessian_two_hole.tex`; old span: 776; kind: LaTeX equation line; fenced: y
- letters and widths: B_z,G_z
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
$\K_{\grad f}$ is \eqref{eq:Kdef} for $Z=\grad f$ given as a T3 $(B_z,G_z)$ of doubled rank. The one-hole
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-0779

- file: `docs/exact_hessian_two_hole.tex`; old span: 779–784; kind: LaTeX equation line; fenced: y
- letters and widths: B_{z,i, L, L_i, R, R_i,, U_i
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\begin{equation}
  \label{eq:t3t3}
  BU_i=B_{z,i}U_i^\T,\quad \hat G_i=G_{z,i}\times_2BU_i,\quad
  Z^L=\mathrm{zip}_\to(\hat G,G),\quad Z^R=\mathrm{zip}_\leftarrow(\hat G,G),\quad
  \mathrm{env}_i=Z^L_i\,G_{z,i}\,Z^R_i,
\end{equation}
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-0785

- file: `docs/exact_hessian_two_hole.tex`; old span: 785–788; kind: LaTeX equation line; fenced: y
- letters and widths: B, B_i, B_{z,i}, L, L_i, L_iA_i, L_iA_iB_i, U_i
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
with gradient blocks $\mathrm{env}_i\times_2BU_i$ (TT) and $\ip{\mathrm{env}_i}{G_i}^\T B_{z,i}$ (Tucker). The
two-hole adds the ``one core replaced'' companions: $\dot{BU}_i=B_{z,i}\dot U_i^\T$, $\dot{\hat G}_i$, the
\emph{tangent zippers} $\dot Z^{L}_{i+1}=\dot Z^L_iA_iB_i+Z^L_i\dot A_iB_i+Z^L_iA_i\dot B_i$ (for the chains
$A=\hat G$, $B=G$; \code{tt_zipper_tangent_left_to_right}, step \code{_tt_zipper_tangent_step}), and the product rule
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-0789

- file: `docs/exact_hessian_two_hole.tex`; old span: 789–796; kind: LaTeX equation line; fenced: y
- letters and widths: B_{z,i}, L_iG_, R_i, R_i,, U_i}
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\begin{equation}
  \label{eq:t3t3dot}
  \begin{gathered}
  \dot{\mathrm{env}}_i=\dot Z^L_iG_{z,i}Z^R_i+Z^L_iG_{z,i}\dot Z^R_i,\\
  (\K_Z\thh)_{G_i}=\dot{\mathrm{env}}_i\times_2BU_i+\mathrm{env}_i\times_2\dot{BU}_i,\qquad
  (\K_Z\thh)_{U_i}=\big(\ip{\dot{\mathrm{env}}_i}{G_i}+\ip{\mathrm{env}_i}{\dot G_i}\big)^\T B_{z,i}.
  \end{gathered}
\end{equation}
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-0801

- file: `docs/exact_hessian_two_hole.tex`; old span: 801–823; kind: LaTeX table or figure; fenced: y
- letters and widths: L_4, R_4
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\begin{figure}[t]
\centering
\begin{tikzpicture}[x=10mm,y=10mm]
  \foreach \i in {1,...,5} {
    \ifnum\i=4 \node[holec] (b\i) at (\i,0) {}; \else
      \ifnum\i=2 \node[dotc] (b\i) at (\i,0) {$\dot G_{\i}$}; \else \node[ttc] (b\i) at (\i,0) {$G_{\i}$}; \fi \fi
    \ifnum\i=4 \node[resn] (a\i) at (\i,1.3) {$G_{z,\i}$}; \else \node[ttc] (a\i) at (\i,1.3) {$\hat G_{\i}$}; \fi
    \draw[bnd] (a\i)--(b\i);
  }
  \foreach \i in {1,...,4} { \pgfmathtruncatemacro\j{\i+1} \draw[bnd] (a\i)--(a\j); \draw[bnd] (b\i)--(b\j); }
  \draw[decorate,decoration={brace,amplitude=4pt,mirror}] (0.6,-0.6)--(3.4,-0.6)
     node[midway,below=4pt,font=\scriptsize]{$\dot Z^L_4$ (one core replaced)};
  \draw[decorate,decoration={brace,amplitude=4pt,mirror}] (4.6,-0.6)--(5.4,-0.6)
     node[midway,below=4pt,font=\scriptsize]{$Z^R_4$};
  \node[font=\scriptsize,anchor=west] at (5.6,1.3) {$Z$'s chain in $\theta$'s basis ($\hat G=G_z\times_2BU$)};
  \node[font=\scriptsize,anchor=west] at (5.6,0) {$\theta$'s chain};
  \node[font=\scriptsize,anchor=west] at (5.6,0.65) {mode legs contracted};
\end{tikzpicture}
\caption{The T3-to-T3 two-hole as a ladder: $Z$'s re-based TT cores on top, $\theta$'s below. The hole at core 4 is
the gradient block; the bottom-left $\dot G_2$ makes $\dot Z^L_4$ one term of the tangent zipper. Core 4's own
top rung is $G_{z,4}$ itself, contracted with $BU_4$ (or $\dot{BU}_4$) on the mode leg.}
\label{fig:t3t3}
\end{figure}
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-0825

- file: `docs/exact_hessian_two_hole.tex`; old span: 825–840; kind: LaTeX equation line; fenced: y
- letters and widths: B_z,G_z, B_{z,i}, L, R, U, U_i
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\begin{algorithm}[t]
\caption{The T3-to-T3 two-hole $\K_Z\thh$}
\label{alg:t3t3}
\begin{algorithmic}[1]
\Require $Z=(B_z,G_z)$ (tails squashed); $\th=(U,G)$ ($\mathtt C$); $\thh$ ($\mathtt{K+C}$)
\Ensure $\K_Z\thh$
\State $BU,\hat G,Z^L,Z^R,\mathrm{env}$ by \eqref{eq:t3t3} \Comment{per frame: \code{t3_precompute_inner_product_sweep}}
\State $\dot{BU}_i\gets B_{z,i}\dot U_i^\T$;\ $\dot{\hat G}_i\gets G_{z,i}\times_2\dot{BU}_i$
\State $\dot Z^L\gets$ \code{tt_zipper_tangent_left_to_right}$(\hat G,G,\dot{\hat G},\dot G;Z^L)$;\ $\dot Z^R$ likewise from the right
\State blocks by \eqref{eq:t3t3dot}
\Impl{\code{t3_linalg.t3_inner_product_corewise_hessian[_from_sweep]}, \code{t3_linalg.t3_inner_product_corewise_transpose},
\code{tt_operations.tt_zipper_tangent_left_to_right}, \code{tt_zipper_tangent_right_to_left}; called by
\code{ManifoldGeometryOps.tangential_curvature}; uniform: \code{ut3_linalg.ut3_precompute_inner_product_sweep},
\code{ut3_inner_product_corewise_hessian[_from_sweep]}.}
\end{algorithmic}
\end{algorithm}
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-0860

- file: `docs/exact_hessian_two_hole.tex`; old span: 860–861; kind: LaTeX equation line; fenced: y
- letters and widths: U_g, U_i
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
$\delta U_i=S_i^\T\dot U_g$ (\texttt{docs/shared\_t3\_math.tex}, Prop.\ ``tied tangent space is a tilted subspace'';
that note writes the factors $N\times n$, so its $C_i=\dot US_i$ is this relation transposed) has one raw
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-0863

- file: `docs/exact_hessian_two_hole.tex`; old span: 863; kind: LaTeX equation line; fenced: y
- letters and widths: U_M
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
$M_g=[S_{i_1}^\T;\dots;S_{i_k}^\T]$ and its SVD $M_g=U_M\,\mathrm{diag}(s_g)\,V_M^\T$ (the companion
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-0865

- file: `docs/exact_hessian_two_hole.tex`; old span: 865–870; kind: LaTeX equation line; fenced: y
- letters and widths: U_M, U_g, U_i, U_i,, U_{i_1}, U_{i_k}, U_{i}
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\begin{equation}
  \label{eq:tied}
  \dot U_g=M_g^{+\eps}\begin{bmatrix}\delta U_{i_1}\\ \vdots\\ \delta U_{i_k}\end{bmatrix}
   =V_Mf_\eps(s_g)U_M^\T[\delta U_{i}]_{i\in g},\qquad
  \Lift^\T:\ [\delta U_i]_{i\in g}=(M_g^{+\eps})^\T\textstyle\sum_{i\in g}Y^U_i,
\end{equation}
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-0871

- file: `docs/exact_hessian_two_hole.tex`; old span: 871; kind: LaTeX equation line; fenced: y
- letters and widths: U_g
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
the same $\dot U_g$ at every group mode; the adjoint sums the group's copies ($\mathsf T^\T$) before the solve's
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-0888  (K16)

- file: `docs/exact_hessian_two_hole.tex`; old span: 888–901; kind: LaTeX equation line; fenced: y
- letters and widths: U_M, U_M,s_g,V_M,, U_g, U_i, n^D_i
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- K16 width lines: 893
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\begin{algorithm}[t]
\caption{The tied lift and its adjoint (one nontrivial group $g$)}
\label{alg:tied}
\begin{algorithmic}[1]
\Require companion $(U_M,s_g,V_M,\text{row splits})$ of $M_g$; $\eps$; coordinates $[\delta U_i]_{i\in g}$, or raw $[Y^U_i]_{i\in g}$
\State $f\gets f_\eps(s_g)$, clip $\mathrm{rcond}=\mathrm{dtype}\,\eps\cdot\max(\sum_{i\in g}n^D_i,n_g)$ \Comment{\code{_tied_filter}}
\State lift: $\dot U_g\gets V_M\,\mathrm{diag}(f)\,U_M^\T\,[\delta U_i]_{i\in g}$; $\dot U_i\gets\dot U_g$ for $i\in g$ \Comment{\code{_tied_filtered_solve}}
\State adjoint: $Y_g\gets\sum_{i\in g}Y^U_i$;\ $[\delta U_i]\gets U_M\,\mathrm{diag}(f)\,V_M^\T Y_g$, split by rows \Comment{its transpose}
\Impl{\code{sharing.fv_tied_lift_factors} (fills \code{LiftFactors.shared_data}, \code{tied_filters}),
\code{sharing._tied_filter}, \code{_tied_filtered_solve[_transpose]} (with \code{eps=None} bit-identical to the
tied projection's solve), dispatched inside \code{tv_corewise_lift[_transpose]}; uniform:
\code{ufv_operations.ufv_tied_lift_factors}.}
\end{algorithmic}
\end{algorithm}
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-0913

- file: `docs/exact_hessian_two_hole.tex`; old span: 913–927; kind: LaTeX equation line; fenced: y
- letters and widths: O_i,P_i,Q_i, P_d, Q, Q_1, R, U_g
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\begin{algorithm}[h]
\caption{The canonical zero-point frame}
\label{alg:zero}
\begin{algorithmic}[1]
\Require a frame's \emph{structure} (core shapes; unstacked); sharing groups (singletons their own); seed $=0$
\For{each group $g$ with key $k_g=\min g$}
  \State $G\gets$ \code{RandomState([seed, k_g]).standard_normal((n_g, N_g))} \Comment{keyed legacy stream}
  \State $Q,R\gets\mathrm{qr}(G^\T)$ (reduced);\ $U_g\gets(Q\,\mathrm{sign}(\mathrm{diag}R))^\T$, replicated over $g$ \Comment{sign-fixed Haar rows}
\EndFor
\State $O_i,P_i,Q_i\gets$ identity prefixes (orthonormal), $P_d=0$, $Q_1=0$ \Comment{base point $=0$}
\Impl{\code{fv_operations.fv_zero_point_frame} (\code{_haar_rows}, \code{_identity_prefix}, \code{ZERO_FRAME_SEED});
uniform \code{ufv_operations.ufv_zero_point_frame} (unpack, build, pack); geometry member \code{zero_point_frame};
the reference: \code{optimizers._resolve_g0_reference}, \code{_zero_reference_gnorm}.}
\end{algorithmic}
\end{algorithm}
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-0934

- file: `docs/exact_hessian_two_hole.tex`; old span: 934; kind: LaTeX equation line; fenced: y
- letters and widths: U
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
Haar only for $U$: the Tucker factor is the only part of the frame that meets the data, and any
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-0961

- file: `docs/exact_hessian_two_hole.tex`; old span: 961–988; kind: LaTeX equation line; fenced: y
- letters and widths: R_x
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\begin{algorithm}[t]
\caption{Newton--CG with the trust region (one row per trial)}
\label{alg:tr}
\begin{algorithmic}[1]
\Require $x_0$; references $\|g_0\|$ (Section~\ref{sec:zero}); forcing power $\kappa$ (default $\frac12$; $1$ for a q-quadratic tail)
\State build the local model at $x$; $g$, $f$ \Comment{\code{Problem.local_model(hessian='exact')}}
\Loop
  \If{$\|g\|\le\texttt{gtol\_rel}\,\|g_0\|$} \Return $x$ (\code{'gradient'}) \EndIf
  \State $\eta\gets\min(\frac12,(\|g\|/\|g_0\|_{cg})^\kappa)$ \Comment{forcing term}
  \State first step: $\Delta_0\gets\|g\|^3/g^\T\Hop_{GN}g$ (or $\|g\|$), $\Delta_{\max}\gets10^4\Delta_0$ \Comment{\code{_initial_radius}}
  \State $p\gets$ Steihaug CG on $\Hop p=-g$ to $\eta\|g\|$ inside $\Delta$ \Comment{\code{_cg_solve}}
  \State if $g^\T p\ge0$: $p\gets-g\min(1,\Delta/\|g\|)$ (a boundary step if scaled)
  \State $\mathrm{pred}\gets g^\T p+\frac12(p^\T\Hop_{GN}p+p^\T\Lift^\T\K_N\Lift p)$ \Comment{one two-hole}
  \State $\rho\gets(f(R_x(p))-f)/\mathrm{pred}$ if $\mathrm{pred}<0$, else NaN
  \State \textbf{if} $\rho<\frac14$ or NaN: $\Delta\gets\frac14\|p\|$;\ \textbf{elif} $\rho>\frac34$ and boundary: $\Delta\gets\min(2\Delta,\Delta_{\max})$
  \If{$\rho>0.1$} accept: $x\gets R_x(p)$, rebuild the model
  \Else \ keep $x$ and its model
    \State \textbf{stop} \code{'objective_floor'} if $|\mathrm{pred}|\le10\,\epsilon_{\mathrm{res}}(\|\omega\odot y\|\,\|\omega\odot r\|+|f|)$
    \State \textbf{stop} \code{'trust_region_floor'} if $\Delta<10^{-12}\Delta_0$ (checked second)
  \EndIf
\EndLoop
\Impl{\code{optimizers.newton_cg}; CG \code{_cg_solve}, \code{_cg_core}, \code{_cg_body}; \code{LocalModel.gn_quadratic},
\code{curvature_quadratic} (constants \code{_TR_ACCEPT}, \code{_TR_SHRINK_BELOW}, \code{_TR_SHRINK},
\code{_TR_EXPAND_ABOVE}, \code{_TR_EXPAND}, \code{_TR_RADIUS_MAX_REL}, \code{_TR_RESOLUTION_FACTOR},
\code{_TR_RADIUS_FLOOR_REL}); $\epsilon_{\mathrm{res}}$ = \code{_resolution_eps} (host-side, dtype-only); per-row
\code{NewtonInfo}.}
\end{algorithmic}
\end{algorithm}
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-1006

- file: `docs/exact_hessian_two_hole.tex`; old span: 1006–1033; kind: LaTeX table or figure; fenced: y
- letters and widths: D, D^2, O_i, P, P_X, R, U_i
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\begin{table}[t]
\centering
\small
\begin{tabular}{@{}l p{0.68\textwidth} r@{}}
\toprule
& check & residual \\
\midrule
C1 & \eqref{eq:factors}: $H_i=O_i\times_2S_i^\T$; $P_{>i}=T_iQ_{>i}$; spectra; $\|X\|^2=\|S_1^\T\|_F^2$ & $\le8.4{\times}10^{-16}$\\
C2 & Prop.~\ref{prop:lift}: $D\Phi[\Lift v]=\Emap v$ on minimal / Tucker-slack / TT-slack frames & $\le2.4{\times}10^{-15}$\\
C3--4 & $\Lift_\eps^\T$ is the adjoint at $\eps=10^{-2}\|X\|$; $\max f_\eps=1/(2\eps)$; $M^{+\eps}=(M^\T M+\eps^2I)^{-1}M^\T$ & $\le1.9{\times}10^{-15}$\\
C5 & $\grad f=P_X\nabla f$; $N$ is normal & $\le9.9{\times}10^{-16}$\\
C6 & $\ip{N}{D^2\Phi[\thh,\thh']}=\ip{\thh'}{\K_N\thh}$ (dense double replacement); a vertical $\thh_v$ has
  $\K_N\thh_v=0$ and does not change the form, while $\|\K_{A^\T r}\thh_v\|\approx10^4$ & $\le1.2{\times}10^{-13}$\\
C7 & \eqref{eq:newton} $=\Pi\Lift^\T\nabla^2_\th\Lambda\Lift\Pi$ (\code{jax.jvp} of \code{jax.grad}); $\nabla_\th\Lambda=0$;
  level 1 $=\Pi\Lift^\T\nabla^2_\th(f\circ\Phi)\Lift\Pi$ & $\le4.5{\times}10^{-14}$\\
C8 & $\ip{p}{\Hop p}$ vs $\frac{d^2}{dt^2}f(R(tp))$ (Richardson; level 2) / vs $\frac{d^2}{dt^2}f(\Phi(\th+t\Lift p))$ (level 1) & $3.7{\times}10^{-6}$ / $2.5{\times}10^{-10}$\\
C9 & the two-holes (probe derivatives, apply derivatives, T3--T3) $=$ complex-step $d/ds$ of their one-holes; the
  block sum \eqref{eq:blocksum}; symmetry; order 0 $=$ plain probe & $\le4.6{\times}10^{-16}$\\
C10 & the written recurrences $\dot\mu$, \eqref{eq:tautilde}, \eqref{eq:tautildedot}, $\dot{\dxt}$ vs the kernels & $\le4.3{\times}10^{-16}$\\
C11 & the factored assemblies \eqref{eq:pairs}, \eqref{eq:asmjetdot} $=$ the nine-term binomial product rule & $\le6.3{\times}10^{-16}$\\
C12 & tied lift \eqref{eq:tied}: $D\Phi[\Lift v]=\Emap v$ at a group-ceiling point; $=M_g^+[\delta U_i]$; adjoint & $\le5.1{\times}10^{-16}$\\
C13 & zero frame: nested, rank-independent, base point 0, the recipe by hand (orthonormal to $6.7{\times}10^{-16}$) & $0$\\
C14 & $\Hop=$ GN $+$ \code{curvature_hvp}; quadratic split; exact gradient $\equiv$ GN gradient; default $\eps$ & $\le4.4{\times}10^{-16}$\\
\bottomrule
\end{tabular}
\caption{The equation checks (scratch script against \texttt{76aa37f}; relative residuals).}
\label{tab:checks}
\end{table}
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-EHTH-1053

- file: `docs/exact_hessian_two_hole.tex`; old span: 1053; kind: LaTeX equation line; fenced: y
- letters and widths: R_k
- roles: T4S letters, per §4.8: the frame's U (Tucker store) → D, O → U, P/Q → L/R; θ's raw own cores U_i (:103–105 and on) → B_i; BU_i, \dot{BU}_i (:781–832) → \hat B_i; DΦ, D²Φ, D_ζP → \mathrm d; \Dsh/\Ush → \Sh/\Sh^\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \mathcal R; PB20's R_k^{-1} (:382) stays, marked "(PB20's notation)"; \Lift differs by font; a plain R is the right store
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
  arXiv:2011.13395 (2020) --- the TT Weingarten map with $R_k^{-1}$, the analogue of $T_i^{+}$.
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~


### `docs/frame_variations.md`

#### DOC-FV-0014

- file: `docs/frame_variations.md`; old span: 14–16; kind: ascii diagram; fenced: y
- letters and widths: letters: H0 H1 H2 L0 L1 R1 R2 U0 U1 U2
- roles: U0–U2 the Tucker store (→ D); (H) the TT variation; L, R stay
- re-laid out: n expected: labels keep their length (U0→D0, nU0→nD0, nO1→nU1); K10 in place
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
1--(H0)--R1---R2--1     1---L0--(H1)--R2--1     1---L0---L1--(H2)--1
    |    |    |             |    |    |             |    |    |
    U0   U1   U2            U0   U1   U2            U0   U1   U2
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### DOC-FV-0023

- file: `docs/frame_variations.md`; old span: 23–30; kind: markdown table; fenced: y
- letters and widths: letters: H Hᵢ L Lᵢ O Oᵢ P Q R Rᵢ U Uᵢ V Vᵢ; widths: nD nU rL rR; store words: down up
- roles: the symbol table: up-Tucker Uᵢ/U/nU → down-Tucker Dᵢ/D/nD; down-TT Oᵢ/O/nD → up-TT Uᵢ/U/nU; `L`/`P`, `R`/`Q` → L, R (D-7); V's width nD → nU, H's nU → nD (D-8); the T4S letters go to the correspondence line (§3)
- re-laid out: n/a: a markdown table (not column-aligned); K10/K10b do not apply, K12 does
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
| frame core | symbol | role | rank on its legs |
|---|---|---|---|
| up-Tucker `Uᵢ` | `U` | the Tucker factors (retained directions) | `nU` |
| down-TT `Oᵢ` | `O` | the Tucker *complement* (where `V` varies) | `nD` |
| left-TT `Lᵢ` | `L`/`P` | left environment (left of the variation) | `rL` |
| right-TT `Rᵢ` | `R`/`Q` | right environment (right of the variation) | `rR` |
| tucker-variation `Vᵢ` | `V` | Tucker-factor direction (in the `nD` complement) | `nD` |
| tt-variation `Hᵢ` | `H` | TT-core direction | `nU`, `rL`, `rR` |
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### DOC-FV-0057

- file: `docs/frame_variations.md`; old span: 57; kind: index equation; fenced: y
- letters and widths: letters: Lᵢ Vᵢ
- roles: `Uᵢᵀ Vᵢ = 0` → `Dᵢᵀ Vᵢ = 0` (an inline gauge relation in prose)
- re-laid out: n expected unless a label grows; K10 in place
- new == old possible: possible: record the reason at R4 if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
When the frame is **orthonormal** and the variations are **gauged** (`Uᵢᵀ Vᵢ = 0`; `Lᵢ` left-orthogonal
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### DOC-FV-0080

- file: `docs/frame_variations.md`; old span: 80–85; kind: markdown table; fenced: y
- letters and widths: letters: Lᵢ Oᵢ Rᵢ Uᵢ; store words: down up
- roles: `Uᵢ` (up / Tucker) → `Dᵢ` (down / Tucker); `Oᵢ` (down / outer) → `Uᵢ` (up); L, R stay; the rank caps use the T3's nᵢ, rᵢ (stay)
- re-laid out: n/a: a markdown table; K12 applies
- new == old possible: possible: record the reason at R4 if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
| core | orthonormality | the rank it caps |
|---|---|---|
| `Uᵢ` (up / Tucker) | rows orthonormal | `nᵢ ≤ Nᵢ` |
| `Oᵢ` (down / outer) | `nᵢ × (rᵢ·rᵢ₊₁)` unfolding, rows orthonormal | `nᵢ ≤ rᵢ·rᵢ₊₁` |
| `Lᵢ` (left) | `(rᵢ·nᵢ) × rᵢ₊₁`, columns orthonormal | `rᵢ₊₁ ≤ rᵢ·nᵢ` |
| `Rᵢ` (right) | `rᵢ × (nᵢ·rᵢ₊₁)`, rows orthonormal | `rᵢ ≤ nᵢ·rᵢ₊₁` |
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~


### `docs/glossary.md` (read at 88fbbedb: absent at v2026.4.0)

#### DOC-G-0106

- file: `docs/glossary.md`; old span: 106; kind: index equation; fenced: y
- letters and widths: letters: B; store words: down_orthogonalize_tucker_cores
- re-laid out: n expected unless a label grows; K10 in place
- new == old possible: yes expected: `B Bᵀ = I` is a T3's own Tucker core (B stays); the line's prose is the glossary's, read at 88fbbedb
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
  `B Bᵀ = I` (`down_orthogonalize_tucker_cores`). The adjective names the side of the cut edge whose indices the
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### DOC-G-0273

- file: `docs/glossary.md`; old span: 273; kind: index equation; fenced: y
- letters and widths: letters: Ui
- roles: `Ui Viᵀ = 0` → `Di Viᵀ = 0` (the frame's Tucker store)
- re-laid out: n expected unless a label grows; K10 in place
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
- **gauge conditions** (`is_gauged`, `gauge_residual`; also: gauged). `Ui Viᵀ = 0` for every `i` (code layout), and
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~


### `docs/pad_safe_svd.tex`

#### TEX-PSS-0049

- file: `docs/pad_safe_svd.tex`; old span: 49; kind: LaTeX equation line; fenced: y
- letters and widths: U
- roles: SVD / QR factors (U Σ Vᵀ, Q R) and the padding projector: §1.5 item 2, they stay; the note's only planned change is prose ("the down step" → "the TT up-orthogonalization", ×2, §6)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
right factor rebuilt from $\Apad^{\top}U = V\Sigma$ --- which realizes the contract exactly,
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-PSS-0065

- file: `docs/pad_safe_svd.tex`; old span: 65–67; kind: LaTeX equation line; fenced: y
- letters and widths: P_c,, P_r,
- roles: SVD / QR factors (U Σ Vᵀ, Q R) and the padding projector: §1.5 item 2, they stay; the note's only planned change is prose ("the down step" → "the TT up-orthogonalization", ×2, §6)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\[
  \Apad\in\R^{N\times M},\qquad N = n + P_r,\quad M = m + P_c,
\]
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-PSS-0068

- file: `docs/pad_safe_svd.tex`; old span: 68; kind: LaTeX equation line; fenced: y
- letters and widths: P_c, P_r
- roles: SVD / QR factors (U Σ Vᵀ, Q R) and the padding projector: §1.5 item 2, they stay; the note's only planned change is prose ("the down step" → "the TT up-orthogonalization", ×2, §6)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
where $P_r$ row positions and $P_c$ column positions --- at \emph{arbitrary, possibly interior}
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-PSS-0086

- file: `docs/pad_safe_svd.tex`; old span: 86–87; kind: LaTeX equation line; fenced: y
- letters and widths: U
- roles: SVD / QR factors (U Σ Vᵀ, Q R) and the padding projector: §1.5 item 2, they stay; the note's only planned change is prose ("the down step" → "the TT up-orthogonalization", ×2, §6)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
arbitrary orthonormal basis of it, so the $\sigma\approx 0$ columns of $U$ generically place mass on
pad rows. Restricted to the real rows --- which is what the masks do downstream --- $U$ is then
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-PSS-0120

- file: `docs/pad_safe_svd.tex`; old span: 120; kind: LaTeX equation line; fenced: y
- letters and widths: U_0
- roles: SVD / QR factors (U Σ Vᵀ, Q R) and the padding projector: §1.5 item 2, they stay; the note's only planned change is prose ("the down step" → "the TT up-orthogonalization", ×2, §6)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\item The \textbf{rank-adaptive BUG integrator} \cite{CerutiKuschLubich2022} augments $(K(t_1),U_0)$
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-PSS-0144

- file: `docs/pad_safe_svd.tex`; old span: 144; kind: LaTeX equation line; fenced: y
- letters and widths: U
- roles: SVD / QR factors (U Σ Vᵀ, Q R) and the padding projector: §1.5 item 2, they stay; the note's only planned change is prose ("the down step" → "the TT up-orthogonalization", ×2, §6)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
(padded entries bitwise zero), a \emph{pad-safe SVD} returns $U\in\R^{N\times K}$,
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-PSS-0147

- file: `docs/pad_safe_svd.tex`; old span: 147; kind: LaTeX equation line; fenced: y
- letters and widths: U
- roles: SVD / QR factors (U Σ Vᵀ, Q R) and the padding projector: §1.5 item 2, they stay; the note's only planned change is prose ("the down step" → "the TT up-orthogonalization", ×2, §6)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\item $\Apad = U\,\diag(s)\,V^{\top}$ --- a genuine economy SVD of the padded matrix;
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-PSS-0149

- file: `docs/pad_safe_svd.tex`; old span: 149; kind: LaTeX equation line; fenced: y
- letters and widths: U
- roles: SVD / QR factors (U Σ Vᵀ, Q R) and the padding projector: §1.5 item 2, they stay; the note's only planned change is prose ("the down step" → "the TT up-orthogonalization", ×2, §6)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
singular values, then real-supported null completions, with $U[:,{:}q]$ \textbf{bitwise} zero on
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-PSS-0163

- file: `docs/pad_safe_svd.tex`; old span: 163; kind: LaTeX equation line; fenced: y
- letters and widths: U
- roles: SVD / QR factors (U Σ Vᵀ, Q R) and the padding projector: §1.5 item 2, they stay; the note's only planned change is prose ("the down step" → "the TT up-orthogonalization", ×2, §6)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
direction; the completion is gauge. Demanding instead that \emph{all} $M$ columns of $U$ avoid the
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-PSS-0183

- file: `docs/pad_safe_svd.tex`; old span: 183; kind: LaTeX equation line; fenced: y
- letters and widths: Q
- roles: SVD / QR factors (U Σ Vᵀ, Q R) and the padding projector: §1.5 item 2, they stay; the note's only planned change is prose ("the down step" → "the TT up-orthogonalization", ×2, §6)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
reflector is bitwise zero on the padded rows, hence so is every data-supported column of $Q$; and
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-PSS-0185

- file: `docs/pad_safe_svd.tex`; old span: 185; kind: LaTeX equation line; fenced: y
- letters and widths: Q
- roles: SVD / QR factors (U Σ Vᵀ, Q R) and the padding projector: §1.5 item 2, they stay; the note's only planned change is prose ("the down step" → "the TT up-orthogonalization", ×2, §6)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
reflectors are the identity and the surplus columns of $Q$ are \emph{exact} pad coordinate
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-PSS-0195

- file: `docs/pad_safe_svd.tex`; old span: 195; kind: LaTeX equation line; fenced: y
- letters and widths: Q
- roles: SVD / QR factors (U Σ Vᵀ, Q R) and the padding projector: §1.5 item 2, they stay; the note's only planned change is prose ("the down step" → "the TT up-orthogonalization", ×2, §6)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\code{larfg} produces $\tau = 0$, $H = I$, and $Q$'s surplus columns are
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-PSS-0197

- file: `docs/pad_safe_svd.tex`; old span: 197; kind: LaTeX equation line; fenced: y
- letters and widths: Q_{ij}
- roles: SVD / QR factors (U Σ Vᵀ, Q R) and the padding projector: §1.5 item 2, they stay; the note's only planned change is prose ("the down step" → "the TT up-orthogonalization", ×2, §6)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
$t_j = \max_i |Q_{ij}|\,[i\text{ padded}] \in \{0,1\}$.
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-PSS-0205

- file: `docs/pad_safe_svd.tex`; old span: 205–206; kind: LaTeX equation line; fenced: y
- letters and widths: B, Q
- roles: SVD / QR factors (U Σ Vᵀ, Q R) and the padding projector: §1.5 item 2, they stay; the note's only planned change is prose ("the down step" → "the TT up-orthogonalization", ×2, §6)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\emph{later} repair is possible: the projected core $B = Q^{\top}\Apad$ is bit-for-bit identical
for a clean and a poisoned $Q$, so the contamination is invisible to everything downstream of the
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-PSS-0221

- file: `docs/pad_safe_svd.tex`; old span: 221; kind: LaTeX equation line; fenced: y
- letters and widths: Q
- roles: SVD / QR factors (U Σ Vᵀ, Q R) and the padding projector: §1.5 item 2, they stay; the note's only planned change is prose ("the down step" → "the TT up-orthogonalization", ×2, §6)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
un-permute $\to Q$; compute the surplus indicator $t$ (Lemma~\ref{lem:perm}).
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-PSS-0223

- file: `docs/pad_safe_svd.tex`; old span: 223; kind: LaTeX equation line; fenced: y
- letters and widths: B, Q
- roles: SVD / QR factors (U Σ Vᵀ, Q R) and the padding projector: §1.5 item 2, they stay; the note's only planned change is prose ("the down step" → "the TT up-orthogonalization", ×2, §6)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
$[\,B \;\; c\,\diag(t)\,]$ with $B = Q^{\top}\Apad$ and $c = 4\norm{\Apad}_F$
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-PSS-0226

- file: `docs/pad_safe_svd.tex`; old span: 226; kind: LaTeX equation line; fenced: y
- letters and widths: B
- roles: SVD / QR factors (U Σ Vᵀ, Q R) and the padding projector: §1.5 item 2, they stay; the note's only planned change is prose ("the down step" → "the TT up-orthogonalization", ×2, §6)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
triplet (row $i$ of $B$ is bitwise zero); where $t_i=0$ it contributes a zero column, harmless
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-PSS-0235

- file: `docs/pad_safe_svd.tex`; old span: 235–236; kind: LaTeX equation line; fenced: y
- letters and widths: U
- roles: SVD / QR factors (U Σ Vᵀ, Q R) and the padding projector: §1.5 item 2, they stay; the note's only planned change is prose ("the down step" → "the TT up-orthogonalization", ×2, §6)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
the (exact-arithmetic-zero) surplus coordinates of the kept columns of $W_1$; set $U = QW_1$ and
zero $s$ \emph{by the pin flag}, then beyond $q$. The first $q$ columns of $U$ are bitwise zero on
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-PSS-0240

- file: `docs/pad_safe_svd.tex`; old span: 240; kind: LaTeX equation line; fenced: y
- letters and widths: U
- roles: SVD / QR factors (U Σ Vᵀ, Q R) and the padding projector: §1.5 item 2, they stay; the note's only planned change is prose ("the down step" → "the TT up-orthogonalization", ×2, §6)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\item \textbf{Rebuild $V$ from $U$.} $W = \Apad^{\top}U$ equals $V\Sigma$ \emph{exactly}, with
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-PSS-0243

- file: `docs/pad_safe_svd.tex`; old span: 243; kind: LaTeX equation line; fenced: y
- letters and widths: R
- roles: SVD / QR factors (U Σ Vᵀ, Q R) and the padding projector: §1.5 item 2, they stay; the note's only planned change is prose ("the down step" → "the TT up-orthogonalization", ×2, §6)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
signs, recovered from $\mathrm{sign}(\diag R)$: the permuted QR of $W$ (padded columns to the
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-PSS-0274

- file: `docs/pad_safe_svd.tex`; old span: 274; kind: LaTeX equation line; fenced: y
- letters and widths: B
- roles: SVD / QR factors (U Σ Vᵀ, Q R) and the padding projector: §1.5 item 2, they stay; the note's only planned change is prose ("the down step" → "the TT up-orthogonalization", ×2, §6)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
present or not. The pins occupy \emph{bitwise-zero rows} of $B$, so the augmented core is bitwise
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-PSS-0276

- file: `docs/pad_safe_svd.tex`; old span: 276; kind: LaTeX equation line; fenced: y
- letters and widths: B
- roles: SVD / QR factors (U Σ Vᵀ, Q R) and the padding projector: §1.5 item 2, they stay; the note's only planned change is prose ("the down step" → "the TT up-orthogonalization", ×2, §6)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
effective backward error is $\varepsilon\norm{B}$, not $\varepsilon c$. The same zero-preservation
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-PSS-0322

- file: `docs/pad_safe_svd.tex`; old span: 322; kind: LaTeX equation line; fenced: y
- letters and widths: U
- roles: SVD / QR factors (U Σ Vᵀ, Q R) and the padding projector: §1.5 item 2, they stay; the note's only planned change is prose ("the down step" → "the TT up-orthogonalization", ×2, §6)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
comparisons must be gauge-invariant (norms, projectors $U^{\top}U$, dense tensors) or transport
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-PSS-0350

- file: `docs/pad_safe_svd.tex`; old span: 350; kind: LaTeX equation line; fenced: y
- letters and widths: D, O, P_r
- roles: SVD / QR factors (U Σ Vᵀ, Q R) and the padding projector: §1.5 item 2, they stay; the note's only planned change is prose ("the down step" → "the TT up-orthogonalization", ×2, §6)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
with $P_r\sim N/2$ the QR costs $O(N^3)$ against $O(NM^2)$ unaugmented. Method~D applies the same
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-PSS-0354

- file: `docs/pad_safe_svd.tex`; old span: 354–355; kind: LaTeX equation line; fenced: y
- letters and widths: B, P
- roles: SVD / QR factors (U Σ Vᵀ, Q R) and the padding projector: §1.5 item 2, they stay; the note's only planned change is prose ("the down step" → "the TT up-orthogonalization", ×2, §6)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
generalized SVD of the pair $(\!\Apad^{\top}, B)$ with $B^{\top}B = P_{\mathrm{real}}$
\cite{VanLoan1976,PaigeSaunders1981}. Because $P_{\mathrm{real}}\Apad = \Apad$ the pair
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-PSS-0358

- file: `docs/pad_safe_svd.tex`; old span: 358–359; kind: LaTeX equation line; fenced: y
- letters and widths: B, P
- roles: SVD / QR factors (U Σ Vᵀ, Q R) and the padding projector: §1.5 item 2, they stay; the note's only planned change is prose ("the down step" → "the TT up-orthogonalization", ×2, §6)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
augmented matrix $[\Apad \;\; \varepsilon B]$ has Gram
$\Apad\Apad^{\top} + \varepsilon^2 P_{\mathrm{real}}$ --- the same commuting pair folded into one
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-PSS-0362

- file: `docs/pad_safe_svd.tex`; old span: 362; kind: LaTeX equation line; fenced: y
- letters and widths: B
- roles: SVD / QR factors (U Σ Vᵀ, Q R) and the padding projector: §1.5 item 2, they stay; the note's only planned change is prose ("the down step" → "the TT up-orthogonalization", ×2, §6)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
there is no threshold. A thin variant for suffix-padded sites ($B=$ the first $n$ masked
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-PSS-0384

- file: `docs/pad_safe_svd.tex`; old span: 384; kind: LaTeX equation line; fenced: y
- letters and widths: B
- roles: SVD / QR factors (U Σ Vᵀ, Q R) and the padding projector: §1.5 item 2, they stay; the note's only planned change is prose ("the down step" → "the TT up-orthogonalization", ×2, §6)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
second order (and not at all for the commuting $B$) --- values $\sigma\lesssim\eta$ are erased
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-PSS-0424

- file: `docs/pad_safe_svd.tex`; old span: 424; kind: LaTeX equation line; fenced: y
- letters and widths: Q
- roles: SVD / QR factors (U Σ Vᵀ, Q R) and the padding projector: §1.5 item 2, they stay; the note's only planned change is prose ("the down step" → "the TT up-orthogonalization", ×2, §6)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
permutation, $Q$'s null column lands on the zero \emph{real} row (allowed --- the requirement is
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-PSS-0426

- file: `docs/pad_safe_svd.tex`; old span: 426; kind: LaTeX equation line; fenced: y
- letters and widths: B, Q
- roles: SVD / QR factors (U Σ Vᵀ, Q R) and the padding projector: §1.5 item 2, they stay; the note's only planned change is prose ("the down step" → "the TT up-orthogonalization", ×2, §6)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
deterministically. But in \emph{both} cases the projected core $B = Q^{\top}\Apad$ is bit-for-bit
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-PSS-0427

- file: `docs/pad_safe_svd.tex`; old span: 427–429; kind: LaTeX equation line; fenced: y
- letters and widths: B
- roles: SVD / QR factors (U Σ Vᵀ, Q R) and the padding projector: §1.5 item 2, they stay; the note's only planned change is prose ("the down step" → "the TT up-orthogonalization", ×2, §6)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\[
B = \begin{pmatrix} 5 & 10 & 0\\ 0&0&0\\ 0&0&0\end{pmatrix}:
\]
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-PSS-0430

- file: `docs/pad_safe_svd.tex`; old span: 430–431; kind: LaTeX equation line; fenced: y
- letters and widths: B, Q
- roles: SVD / QR factors (U Σ Vᵀ, Q R) and the padding projector: §1.5 item 2, they stay; the note's only planned change is prose ("the down step" → "the TT up-orthogonalization", ×2, §6)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
a zero row of $B$ says only ``this column of $Q$ is orthogonal to $\col(\Apad)$'' --- it cannot
distinguish a clean null column from a poisoned one. The contamination lives entirely in $Q$,
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~


### `docs/padded_frame_cone.tex`

#### TEX-PFC-0096

- file: `docs/padded_frame_cone.tex`; old span: 96–97; kind: LaTeX equation line; fenced: y
- letters and widths: U_{x, U_{y
- roles: per §6/D-5: Table 1, the frame letters (U Tucker store → D, O → U), widths n_{U,i} → n_{D,i}, n_{D,i} → n_{U,i}; D products/messages → Ψ; U_{x→t}, U′ → F_{x→t}, F′; R(\theta) (:217–219) is a residual function and stays; the matrix-case proof's SVD U may stay
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
$x\to t$, the frame $U_{x\to t}$ is an orthonormal basis of the column space of the tensor unfolded with $x$'s side
as rows. Frames are nested: $U_{x\to t}$ lies in $H_x\otimes\bigotimes_{y\in\mathrm{nb}(x)\setminus t}U_{y\to x}$,
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-PFC-0102  (K16)

- file: `docs/padded_frame_cone.tex`; old span: 102–116; kind: LaTeX table or figure; fenced: y
- letters and widths: L_i, L_{d, O_i, R_0, R_i, U_i, n_{D,i}, n_{U,i}, r_{L,i+1}, r_{L,i}, r_{R,i+1}, r_{R,i}
- roles: per §6/D-5: Table 1, the frame letters (U Tucker store → D, O → U), widths n_{U,i} → n_{D,i}, n_{D,i} → n_{U,i}; D products/messages → Ψ; U_{x→t}, U′ → F_{x→t}, F′; R(\theta) (:217–219) is a residual function and stays; the matrix-case proof's SVD U may stay
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- K16 width lines: 108, 109, 110, 111
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\begin{table}[h]
\centering
\begin{tabular}{@{}llll@{}}
\toprule
directed edge & store & core & output slot \\
\midrule
$\ell_i\to c_i$ & \code{up[i]} & $U_i\in\R^{n_{U,i}\times N_i}$ & the Tucker index \\
$c_i\to \ell_i$ & \code{down[i]} & $O_i\in\R^{r_{L,i}\times n_{D,i}\times r_{R,i+1}}$ & the down index \\
$c_i\to c_{i+1}$ & \code{left[i]}, $i\le d-2$ & $L_i\in\R^{r_{L,i}\times n_{U,i}\times r_{L,i+1}}$ & bond $i+1$ \\
$c_i\to c_{i-1}$ & \code{right[i]}, $i\ge 1$ & $R_i\in\R^{r_{R,i}\times n_{U,i}\times r_{R,i+1}}$ & bond $i$ \\
\bottomrule
\end{tabular}
\caption{Directed edges of a T3 and the frame stores. $L_{d-1}$ and $R_0$ are the \emph{carriers} of the base
point, not frame cores of an edge.}\label{tab:stores}
\end{table}
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-PFC-0120

- file: `docs/padded_frame_cone.tex`; old span: 120; kind: LaTeX equation line; fenced: y
- letters and widths: P_{x
- roles: per §6/D-5: Table 1, the frame letters (U Tucker store → D, O → U), widths n_{U,i} → n_{D,i}, n_{D,i} → n_{U,i}; D products/messages → Ψ; U_{x→t}, U′ → F_{x→t}, F′; R(\theta) (:217–219) is a residual function and stays; the matrix-case proof's SVD U may stay
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
(the \emph{A-block}, $\Abar_{x\to t}$, of width $r_e$) and an orthonormal \emph{completion} $P_{x\to t}$ of width
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-PFC-0140

- file: `docs/padded_frame_cone.tex`; old span: 140–141; kind: LaTeX equation line; fenced: y
- letters and widths: O, U
- roles: per §6/D-5: Table 1, the frame letters (U Tucker store → D, O → U), widths n_{U,i} → n_{D,i}, n_{D,i} → n_{U,i}; D products/messages → Ψ; U_{x→t}, U′ → F_{x→t}, F′; R(\theta) (:217–219) is a residual function and stays; the matrix-case proof's SVD U may stay
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
$p$ and then $c$). One trap remains. For $A=U\Sigma V^\T$, the leading frame of $A+\eps S$ is
$U+\eps(I-UU^\T)SV\Sigma^{-1}+O(\eps^2)$: the tangent part of $S$ \emph{tilts} $A$'s frame, and that tilt is not a
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-PFC-0152

- file: `docs/padded_frame_cone.tex`; old span: 152–153; kind: LaTeX equation line; fenced: y
- letters and widths: U_{t, U_{x
- roles: per §6/D-5: Table 1, the frame letters (U Tucker store → D, O → U), widths n_{U,i} → n_{D,i}, n_{D,i} → n_{U,i}; D products/messages → Ψ; U_{x→t}, U′ → F_{x→t}, F′; R(\theta) (:217–219) is a residual function and stays; the matrix-case proof's SVD U may stay
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
space: at edge $x\to t$, the rows in $C'_{x\to t}\ominus U_{x\to t}$ with $C'$ built from the completed child frames,
and the columns in the complement of $A$'s frame $U_{t\to x}$. Its completions depend on the root: each edge sees the
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-PFC-0171

- file: `docs/padded_frame_cone.tex`; old span: 171–172; kind: LaTeX equation line; fenced: y
- letters and widths: Q_e, U_{t
- roles: per §6/D-5: Table 1, the frame letters (U Tucker store → D, O → U), widths n_{U,i} → n_{D,i}, n_{D,i} → n_{U,i}; D products/messages → Ψ; U_{x→t}, U′ → F_{x→t}, F′; R(\theta) (:217–219) is a residual function and stays; the matrix-case proof's SVD U may stay
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
edge, tensors whose far side lies in the \emph{far completion} $Q_e$ (the completion of $U_{t\to x}$). Through them
the normal block of $S$ applied to $Q_e$ enters the step: information from outside the current completions.
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-PFC-0174

- file: `docs/padded_frame_cone.tex`; old span: 174–175; kind: LaTeX equation line; fenced: y
- letters and widths: P, Q, U
- roles: per §6/D-5: Table 1, the frame letters (U Tucker store → D, O → U), widths n_{U,i} → n_{D,i}, n_{D,i} → n_{U,i}; D products/messages → Ψ; U_{x→t}, U′ → F_{x→t}, F′; R(\theta) (:217–219) is a residual function and stays; the matrix-case proof's SVD U may stay
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\paragraph{The matrix case.} With $P,Q$ completing $U,V$, the projection of $S$ onto $T(F)=\{U'X+YV'^\T\}$ has normal
block $PP^\T N+(I-PP^\T)NQQ^\T$. The new $P$ is its top-$k$ left singular subspace, and the new $Q$ the top-$k$ right
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-PFC-0188

- file: `docs/padded_frame_cone.tex`; old span: 188; kind: LaTeX equation line; fenced: y
- letters and widths: P_{a, P_{b
- roles: per §6/D-5: Table 1, the frame letters (U Tucker store → D, O → U), widths n_{U,i} → n_{D,i}, n_{D,i} → n_{U,i}; D products/messages → Ψ; U_{x→t}, U′ → F_{x→t}, F′; R(\theta) (:217–219) is a residual function and stays; the matrix-case proof's SVD U may stay
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
lies in the near completion, provided the $k\times k$ matrix $P_{a\to b}^\T S P_{b\to a}$ is invertible. In the
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-PFC-0195

- file: `docs/padded_frame_cone.tex`; old span: 195; kind: LaTeX equation line; fenced: y
- letters and widths: P, Q, U
- roles: per §6/D-5: Table 1, the frame letters (U Tucker store → D, O → U), widths n_{U,i} → n_{D,i}, n_{D,i} → n_{U,i}; D products/messages → Ψ; U_{x→t}, U′ → F_{x→t}, F′; R(\theta) (:217–219) is a residual function and stays; the matrix-case proof's SVD U may stay
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
Let $N=(I-UU^\T)S(I-VV^\T)$, and let $P\perp U$, $Q\perp V$ be completions of width $k$. Then
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-PFC-0196

- file: `docs/padded_frame_cone.tex`; old span: 196–198; kind: LaTeX equation line; fenced: y
- letters and widths: P, Q
- roles: per §6/D-5: Table 1, the frame letters (U Tucker store → D, O → U), widths n_{U,i} → n_{D,i}, n_{D,i} → n_{U,i}; D products/messages → Ψ; U_{x→t}, U′ → F_{x→t}, F′; R(\theta) (:217–219) is a residual function and stays; the matrix-case proof's SVD U may stay
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\begin{equation}\label{eq:energy}
  E(P,Q)\;:=\;\|\Pi_{T(F)}S\|^2 \;=\; \|S\|^2-\bigl\|(I-PP^\T)\,N\,(I-QQ^\T)\bigr\|^2 .
\end{equation}
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-PFC-0199

- file: `docs/padded_frame_cone.tex`; old span: 199–200; kind: LaTeX equation line; fenced: y
- letters and widths: P, Q
- roles: per §6/D-5: Table 1, the frame letters (U Tucker store → D, O → U), widths n_{U,i} → n_{D,i}, n_{D,i} → n_{U,i}; D products/messages → Ψ; U_{x→t}, U′ → F_{x→t}, F′; R(\theta) (:217–219) is a residual function and stays; the matrix-case proof's SVD U may stay
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
If $\sigma_k>\sigma_{k+1}>0$, the fixed point $(P,Q)=$ the top-$k$ singular pair of $N$ is a stationary point of $E$
and a saddle. Rotating $P$ alone (or $Q$ alone) out of its span never decreases $E$, and increases it at second order
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-PFC-0206

- file: `docs/padded_frame_cone.tex`; old span: 206–207; kind: LaTeX equation line; fenced: y
- letters and widths: P, Q, U
- roles: per §6/D-5: Table 1, the frame letters (U Tucker store → D, O → U), widths n_{U,i} → n_{D,i}, n_{D,i} → n_{U,i}; D products/messages → Ψ; U_{x→t}, U′ → F_{x→t}, F′; R(\theta) (:217–219) is a residual function and stays; the matrix-case proof's SVD U may stay
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
$\Pi_{T(F)}=I-(I-U'U'^\T)(\cdot)(I-V'V'^\T)$ with $U'=[U\;P]$, $V'=[V\;Q]$, and
$(I-U'U'^\T)S(I-V'V'^\T)=(I-PP^\T)N(I-QQ^\T)$ since $P\perp U$, $Q\perp V$; this gives \eqref{eq:energy}. With $Q$
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-PFC-0209

- file: `docs/padded_frame_cone.tex`; old span: 209–212; kind: LaTeX equation line; fenced: y
- letters and widths: O, P, Q, U
- roles: per §6/D-5: Table 1, the frame letters (U Tucker store → D, O → U), widths n_{U,i} → n_{D,i}, n_{D,i} → n_{U,i}; D products/messages → Ψ; U_{x→t}, U′ → F_{x→t}, F′; R(\theta) (:217–219) is a residual function and stays; the matrix-case proof's SVD U may stay
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
$E=\|S\|^2-\|M\|^2+\|P^\T M\|^2$. At $P=[u_1\cdots u_k]$, $P^\T M=0$, the minimum of $\|P^\T M\|^2\ge 0$: so $E$ is
stationary in $P$ and does not decrease under any rotation of $P$ alone. A rotation of $u_k$ by $h$ toward a unit
vector $w\perp[U\;P]$ raises $\|P^\T M\|^2$ by $h^2\|w^\T M\|^2+O(h^3)$, which is positive when $w$ has a component
along some $u_j$, $j>k$ (a $w$ in the left null space of $N$ leaves $E$ unchanged). Symmetrically in $Q$. For the
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-PFC-0216

- file: `docs/padded_frame_cone.tex`; old span: 216–220; kind: LaTeX equation line; fenced: y
- letters and widths: O, R
- roles: per §6/D-5: Table 1, the frame letters (U Tucker store → D, O → U), widths n_{U,i} → n_{D,i}, n_{D,i} → n_{U,i}; D products/messages → Ψ; U_{x→t}, U′ → F_{x→t}, F′; R(\theta) (:217–219) is a residual function and stays; the matrix-case proof's SVD U may stay
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\begin{align*}
  R(\theta):=\bigl\|(I-PP^\T)N(I-QQ^\T)\bigr\|^2
  &=\sum_{j>k+1}\sigma_j^2+\bigl(\sigma_{k+1}+(\sigma_k-\sigma_{k+1})\sin^2\theta\bigr)^2\\
  &= R(0)+2\sigma_{k+1}(\sigma_k-\sigma_{k+1})\,\theta^2+O(\theta^4).
\end{align*}
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-PFC-0221

- file: `docs/padded_frame_cone.tex`; old span: 221; kind: LaTeX equation line; fenced: y
- letters and widths: R
- roles: per §6/D-5: Table 1, the frame letters (U Tucker store → D, O → U), widths n_{U,i} → n_{D,i}, n_{D,i} → n_{U,i}; D products/messages → Ψ; U_{x→t}, U′ → F_{x→t}, F′; R(\theta) (:217–219) is a residual function and stays; the matrix-case proof's SVD U may stay
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
$R$ grows at second order, so $E$ falls, exactly when $\sigma_k>\sigma_{k+1}>0$; with $\sigma_{k+1}=0$ the decrease is
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-PFC-0225

- file: `docs/padded_frame_cone.tex`; old span: 225; kind: LaTeX equation line; fenced: y
- letters and widths: P, Q
- roles: per §6/D-5: Table 1, the frame letters (U Tucker store → D, O → U), widths n_{U,i} → n_{D,i}, n_{D,i} → n_{U,i}; D products/messages → Ψ; U_{x→t}, U′ → F_{x→t}, F′; R(\theta) (:217–219) is a residual function and stays; the matrix-case proof's SVD U may stay
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
$E$ counts content of rank up to $2k$ (its maximum puts $P$ and $Q$ on \emph{different} singular directions), so it
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-PFC-0242

- file: `docs/padded_frame_cone.tex`; old span: 242; kind: LaTeX equation line; fenced: y
- letters and widths: P
- roles: per §6/D-5: Table 1, the frame letters (U Tucker store → D, O → U), widths n_{U,i} → n_{D,i}, n_{D,i} → n_{U,i}; D products/messages → Ψ; U_{x→t}, U′ → F_{x→t}, F′; R(\theta) (:217–219) is a residual function and stays; the matrix-case proof's SVD U may stay
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
$m=\|P^\T N\|^2/\|N\|^2$, maximized by the top-$k$ singular subspace, with every other stationary point a saddle.
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-PFC-0258

- file: `docs/padded_frame_cone.tex`; old span: 258–259; kind: LaTeX equation line; fenced: y
- letters and widths: L
- roles: per §6/D-5: Table 1, the frame letters (U Tucker store → D, O → U), widths n_{U,i} → n_{D,i}, n_{D,i} → n_{U,i}; D products/messages → Ψ; U_{x→t}, U′ → F_{x→t}, F′; R(\theta) (:217–219) is a residual function and stays; the matrix-case proof's SVD U may stay
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
$k$ directions, and the excursions of $N$ compete for them. An excursion across $L$ edges is captured only if all $L$
completions align with it, so its captured energy is a product of $L$ alignment factors, of order $2L$ in the
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-PFC-0261

- file: `docs/padded_frame_cone.tex`; old span: 261; kind: LaTeX equation line; fenced: y
- letters and widths: L
- roles: per §6/D-5: Table 1, the frame letters (U Tucker store → D, O → U), widths n_{U,i} → n_{D,i}, n_{D,i} → n_{U,i}; D products/messages → Ψ; U_{x→t}, U′ → F_{x→t}, F′; R(\theta) (:217–219) is a residual function and stays; the matrix-case proof's SVD U may stay
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
gains only at order $2L$: that configuration is a strict local maximum, however large the ignored excursion. The
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-PFC-0278

- file: `docs/padded_frame_cone.tex`; old span: 278; kind: LaTeX equation line; fenced: y
- letters and widths: P
- roles: per §6/D-5: Table 1, the frame letters (U Tucker store → D, O → U), widths n_{U,i} → n_{D,i}, n_{D,i} → n_{U,i}; D products/messages → Ψ; U_{x→t}, U′ → F_{x→t}, F′; R(\theta) (:217–219) is a residual function and stays; the matrix-case proof's SVD U may stay
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
node terms on $a$'s side and on $b$'s side, $P$ the old completion and $P^+$ the new one:
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-PFC-0280

- file: `docs/padded_frame_cone.tex`; old span: 280–289; kind: LaTeX table or figure; fenced: y
- letters and widths: D, D_{a, P, P_{a, P_{b, R, U
- roles: per §6/D-5: Table 1, the frame letters (U Tucker store → D, O → U), widths n_{U,i} → n_{D,i}, n_{D,i} → n_{U,i}; D products/messages → Ψ; U_{x→t}, U′ → F_{x→t}, F′; R(\theta) (:217–219) is a residual function and stays; the matrix-case proof's SVD U may stay
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\begin{tabular}{@{}llll@{}}
\toprule
message & size & definition & built \\
\midrule
$\Phi_{a\to b}$ & $k\times R$ & $P_{a\to b}^\T Z_{\mathrm{far}}U'_{b\to a}$ & stage 1, far leaves inward\\
$\Xi_{a\to b}$ & $k\times\cdot$ & $\Xi\Xi^\T=P_{a\to b}^\T Z_{\mathrm{far}}(I-U'_{b\to a}U'^\T_{b\to a})Z_{\mathrm{far}}^\T P_{a\to b}$ & stage 1, with $\Phi$\\
$D_{a\to b}$ & $R\times k$ & $U'^{+\T}_{a\to b}Z_{\mathrm{near}}P_{b\to a}$ & stage 2, message order\\
$o_{a\to b}$ & $k^+\times k$ & $P^{+\T}_{a\to b}P_{a\to b}$ & stage 2, with $D$\\
\bottomrule
\end{tabular}
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-PFC-0292

- file: `docs/padded_frame_cone.tex`; old span: 292; kind: LaTeX equation line; fenced: y
- letters and widths: D
- roles: per §6/D-5: Table 1, the frame letters (U Tucker store → D, O → U), widths n_{U,i} → n_{D,i}, n_{D,i} → n_{U,i}; D products/messages → Ψ; U_{x→t}, U′ → F_{x→t}, F′; R(\theta) (:217–219) is a residual function and stays; the matrix-case proof's SVD U may stay
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
square root of the rest (any factor; only $\Xi\Xi^\T$ enters). $D$ and $o$ carry the near side, already rotated into
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-PFC-0298

- file: `docs/padded_frame_cone.tex`; old span: 298–300; kind: LaTeX equation line; fenced: y
- letters and widths: P
- roles: per §6/D-5: Table 1, the frame letters (U Tucker store → D, O → U), widths n_{U,i} → n_{D,i}, n_{D,i} → n_{U,i}; D products/messages → Ψ; U_{x→t}, U′ → F_{x→t}, F′; R(\theta) (:217–219) is a residual function and stays; the matrix-case proof's SVD U may stay
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\begin{equation}\label{eq:edge}
  Y=\bigl[\,g_{\mathrm{old}}\Phi_P+g_{\mathrm{new}}\;\big|\;g_{\mathrm{old}}\Xi\,\bigr]\,\Theta^\T ,
\end{equation}
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-PFC-0303

- file: `docs/padded_frame_cone.tex`; old span: 303; kind: LaTeX equation line; fenced: y
- letters and widths: P
- roles: per §6/D-5: Table 1, the frame letters (U Tucker store → D, O → U), widths n_{U,i} → n_{D,i}, n_{D,i} → n_{U,i}; D products/messages → Ψ; U_{x→t}, U′ → F_{x→t}, F′; R(\theta) (:217–219) is a residual function and stays; the matrix-case proof's SVD U may stay
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
$\Abar$, and the edge's \emph{movement} is $\|g_{\mathrm{old}}-P^+P^{+\T}g_{\mathrm{old}}\|$: zero exactly when the
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-PFC-0306

- file: `docs/padded_frame_cone.tex`; old span: 306–323; kind: LaTeX equation line; fenced: y
- letters and widths: D, P
- roles: per §6/D-5: Table 1, the frame letters (U Tucker store → D, O → U), widths n_{U,i} → n_{D,i}, n_{D,i} → n_{U,i}; D products/messages → Ψ; U_{x→t}, U′ → F_{x→t}, F′; R(\theta) (:217–219) is a residual function and stays; the matrix-case proof's SVD U may stay
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\begin{algorithm}[t]
\caption{One alignment step (ragged)}\label{alg:step}
\begin{algorithmic}[1]
\Require padded frame $F$; variations $(V,H)$ of $Z$ at $F$; old ranks; output ranks; \code{update}
\State Stage 1 (old frames): down messages; left messages from the far end; right messages from the near end; up
messages; each $(\Phi,\Xi)$.
\Impl{\code{_down_message}, \code{_left_message_step}, \code{_right_message_step} (scan bodies), \code{_up_message};
$\Xi$ by \code{_root}, a factor from the thin SVD, never a Gram matrix.}
\State Stage 2 (message order): up edges (one per sharing group, the members' contents side by side); the left sweep
$i=0..d-2$; the right sweep $i=d-1..1$; down edges. At each: new coordinates, $Y$ by \eqref{eq:edge}, $P^+$, $o$, $D$,
movement.
\Impl{\code{_new_coordinates}, \code{_edge_content}, \code{_edge} $\to$ \code{linalg.pinned_leading_columns};
\code{_up_edge}, \code{_left_edge_step}, \code{_right_edge_step}, \code{_down_edge}.}
\State A store not in \code{update} is transported: $Y=g_{\mathrm{old}}$.
\State Carriers: the old content, embedded. Return the frame and the largest movement.
\Impl{\code{padded_frames.tv_align_padded_frame}; frontend one pass: \code{T3Tangent.align_padded}.}
\end{algorithmic}
\end{algorithm}
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-PFC-0330

- file: `docs/padded_frame_cone.tex`; old span: 330; kind: LaTeX equation line; fenced: y
- letters and widths: P, R_b
- roles: per §6/D-5: Table 1, the frame letters (U Tucker store → D, O → U), widths n_{U,i} → n_{D,i}, n_{D,i} → n_{U,i}; D products/messages → Ψ; U_{x→t}, U′ → F_{x→t}, F′; R(\theta) (:217–219) is a residual function and stays; the matrix-case proof's SVD U may stay
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
supercores in block layout ($[\text{A-block}\mid\text{P-block}]$ per slot, the split $(N_a,R_b)$ static). The edge is
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-PFC-0405

- file: `docs/padded_frame_cone.tex`; old span: 405; kind: LaTeX equation line; fenced: y
- letters and widths: P
- roles: per §6/D-5: Table 1, the frame letters (U Tucker store → D, O → U), widths n_{U,i} → n_{D,i}, n_{D,i} → n_{U,i}; D products/messages → Ψ; U_{x→t}, U′ → F_{x→t}, F′; R(\theta) (:217–219) is a residual function and stays; the matrix-case proof's SVD U may stay
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
halves. Each updated store maximizes the captured energy $\|P^{+\T}Y\|^2$ of its own content $Y$ given the others;
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-PFC-0451

- file: `docs/padded_frame_cone.tex`; old span: 451; kind: LaTeX equation line; fenced: y
- letters and widths: P, U_k
- roles: per §6/D-5: Table 1, the frame letters (U Tucker store → D, O → U), widths n_{U,i} → n_{D,i}, n_{D,i} → n_{U,i}; D products/messages → Ψ; U_{x→t}, U′ → F_{x→t}, F′; R(\theta) (:217–219) is a residual function and stays; the matrix-case proof's SVD U may stay
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
$[\Abar\mid U_k]$ then makes $P^+$ orthogonal to $\Abar$ to roundoff.
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~


### `docs/shared_t3_math.tex`

#### TEX-STM-0066

- file: `docs/shared_t3_math.tex`; old span: 66–70; kind: LaTeX equation line; fenced: y
- letters and widths: U_1,, U_d
- roles: per §4.8: the T3's own factors → B_i, B_g (the §2 point algorithms, the corewise section :440–450); in the tangent sections (:318–420, :469–495 with rem:embedding as one range) the frame's down store → D_i, D_g, \dot D; the block B_i := U_i W_i (:279–292) → K_i; :306's B_i (the factor) stays; the tied coordinate B (:358, 551–566) → Z; n_{D,i} → n_{U,i}; O_g (a complement basis) stays; R(\T,\xi) (:474) stays; Q_i at :363 stays (session ruling)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\[
\T = \bigl[\bigl[\, G ;\; U_1, \ldots, U_d \,\bigr]\bigr],
\qquad
G = \mathrm{TT}(G_1, \ldots, G_d),
\]
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-STM-0071

- file: `docs/shared_t3_math.tex`; old span: 71; kind: LaTeX equation line; fenced: y
- letters and widths: U_i
- roles: per §4.8: the T3's own factors → B_i, B_g (the §2 point algorithms, the corewise section :440–450); in the tangent sections (:318–420, :469–495 with rem:embedding as one range) the frame's down store → D_i, D_g, \dot D; the block B_i := U_i W_i (:279–292) → K_i; :306's B_i (the factor) stays; the tied coordinate B (:358, 551–566) → Z; n_{D,i} → n_{U,i}; O_g (a complement basis) stays; R(\T,\xi) (:474) stays; Q_i at :363 stays (session ruling)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
where $U_i \in \R^{N_i \times n_i}$ are the Tucker factors,
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-STM-0082

- file: `docs/shared_t3_math.tex`; old span: 82; kind: LaTeX equation line; fenced: y
- letters and widths: U_i
- roles: per §4.8: the T3's own factors → B_i, B_g (the §2 point algorithms, the corewise section :440–450); in the tangent sections (:318–420, :469–495 with rem:embedding as one range) the frame's down store → D_i, D_g, \dot D; the block B_i := U_i W_i (:279–292) → K_i; :306's B_i (the factor) stays; the tied coordinate B (:358, 551–566) → Z; n_{D,i} → n_{U,i}; O_g (a complement basis) stays; R(\T,\xi) (:474) stays; Q_i at :363 stays (session ruling)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
$U_i$ have orthonormal columns, and, when a statement refers to mode $i$, the
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-STM-0104

- file: `docs/shared_t3_math.tex`; old span: 104; kind: LaTeX equation line; fenced: y
- letters and widths: U_{g}, U_{i}
- roles: per §4.8: the T3's own factors → B_i, B_g (the §2 point algorithms, the corewise section :440–450); in the tangent sections (:318–420, :469–495 with rem:embedding as one range) the frame's down store → D_i, D_g, \dot D; the block B_i := U_i W_i (:279–292) → K_i; :306's B_i (the factor) stays; the tied coordinate B (:358, 551–566) → Z; n_{D,i} → n_{U,i}; O_g (a complement basis) stays; R(\T,\xi) (:474) stays; Q_i at :363 stays (session ruling)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
denoted $N_g$. A T3 is \emph{shared} (an SF-T3) if $U_{i} = U_{g}$ and
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-STM-0118

- file: `docs/shared_t3_math.tex`; old span: 118–120; kind: LaTeX equation line; fenced: y
- letters and widths: U_g
- roles: per §4.8: the T3's own factors → B_i, B_g (the §2 point algorithms, the corewise section :440–450); in the tangent sections (:318–420, :469–495 with rem:embedding as one range) the frame's down store → D_i, D_g, \dot D; the block B_i := U_i W_i (:279–292) → K_i; :306's B_i (the factor) stays; the tied coordinate B (:358, 551–566) → Z; n_{D,i} → n_{U,i}; O_g (a complement basis) stays; R(\T,\xi) (:474) stays; Q_i at :363 stays (session ruling)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\begin{equation}\label{eq:skeleton}
\T_{(i)} = U_g\, W_i\, M_i,
\end{equation}
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-STM-0122

- file: `docs/shared_t3_math.tex`; old span: 122–126; kind: LaTeX equation line; fenced: y
- letters and widths: U_g
- roles: per §4.8: the T3's own factors → B_i, B_g (the §2 point algorithms, the corewise section :440–450); in the tangent sections (:318–420, :469–495 with rem:embedding as one range) the frame's down store → D_i, D_g, \dot D; the block B_i := U_i W_i (:279–292) → K_i; :306's B_i (the factor) stays; the tied coordinate B (:358, 551–566) → Z; n_{D,i} → n_{U,i}; O_g (a complement basis) stays; R(\T,\xi) (:474) stays; Q_i at :363 stays (session ruling)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\begin{equation}\label{eq:stacked}
\bigl[\,\T_{(i_1)} \,|\, \cdots \,|\, \T_{(i_k)}\,\bigr]
= U_g \,\bigl[\, W_{i_1} \,|\, \cdots \,|\, W_{i_k} \,\bigr]\,
\operatorname{blockdiag}\!\bigl(M_{i_1}, \ldots, M_{i_k}\bigr).
\end{equation}
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-STM-0147

- file: `docs/shared_t3_math.tex`; old span: 147; kind: LaTeX equation line; fenced: y
- letters and widths: U_g
- roles: per §4.8: the T3's own factors → B_i, B_g (the §2 point algorithms, the corewise section :440–450); in the tangent sections (:318–420, :469–495 with rem:embedding as one range) the frame's down store → D_i, D_g, \dot D; the block B_i := U_i W_i (:279–292) → K_i; :306's B_i (the factor) stays; the tied coordinate B (:358, 551–566) → Z; n_{D,i} → n_{U,i}; O_g (a complement basis) stays; R(\T,\xi) (:474) stays; Q_i at :363 stays (session ruling)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
factor $U_g$ is common by sharedness, \eqref{eq:stacked} follows by placing
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-STM-0168

- file: `docs/shared_t3_math.tex`; old span: 168; kind: LaTeX equation line; fenced: y
- letters and widths: U_g
- roles: per §4.8: the T3's own factors → B_i, B_g (the §2 point algorithms, the corewise section :440–450); in the tangent sections (:318–420, :469–495 with rem:embedding as one range) the frame's down store → D_i, D_g, \dot D; the block B_i := U_i W_i (:279–292) → K_i; :306's B_i (the factor) stays; the tied coordinate B (:358, 551–566) → Z; n_{D,i} → n_{U,i}; O_g (a complement basis) stays; R(\T,\xi) (:474) stays; Q_i at :363 stays (session ruling)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
column space contained in $\operatorname{range}(U_g)$, hence so does the
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-STM-0170

- file: `docs/shared_t3_math.tex`; old span: 170–171; kind: LaTeX equation line; fenced: y
- letters and widths: U_g
- roles: per §4.8: the T3's own factors → B_i, B_g (the §2 point algorithms, the corewise section :440–450); in the tangent sections (:318–420, :469–495 with rem:embedding as one range) the frame's down store → D_i, D_g, \dot D; the block B_i := U_i W_i (:279–292) → K_i; :306's B_i (the factor) stays; the tied coordinate B (:358, 551–566) → Z; n_{D,i} → n_{U,i}; O_g (a complement basis) stays; R(\T,\xi) (:474) stays; Q_i at :363 stays (session ruling)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
$(\leq)$: take $U_g$ to be an orthonormal basis of the concatenation's column
space; projecting each mode ($U_g U_g^\top \T_{(i)} = \T_{(i)}$) yields an
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-STM-0179

- file: `docs/shared_t3_math.tex`; old span: 179–202; kind: LaTeX equation line; fenced: y
- letters and widths: O, U_g
- roles: per §4.8: the T3's own factors → B_i, B_g (the §2 point algorithms, the corewise section :440–450); in the tangent sections (:318–420, :469–495 with rem:embedding as one range) the frame's down store → D_i, D_g, \dot D; the block B_i := U_i W_i (:279–292) → K_i; :306's B_i (the factor) stays; the tied coordinate B (:358, 551–566) → Z; n_{D,i} → n_{U,i}; O_g (a complement basis) stays; R(\T,\xi) (:474) stays; Q_i at :363 stays (session ruling)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\begin{algorithm}[Grouped T3-SVD / SF-rounding; cf.~{\cite[Alg.~1]{MR2026}}]
\label{alg:rounding}
Input: shared T3 with orthonormal factors; target ranks. Output: shared T3
with target ranks.
\begin{enumerate}[label=\emph{(\arabic*)},itemsep=1pt,topsep=2pt]
\item TT-round the cores to the target TT ranks (standard sweep; factors
  untouched). In an interlaced implementation, steps (2)--(4) are woven into
  the same sweep; the interlacing order changes truncated results but not the
  bounds below.
\item For each mode $i$, obtain $W_i$ in the $i$-orthogonal representation
  (one orthogonalization sweep).
\item Singleton modes: truncated SVD of $W_i$; rotate factor and core as
  usual.
\item Each group $g$: one truncated SVD of the concatenation
  $[\,W_{i_1} | \cdots | W_{i_k}\,] \in \R^{n_g \times \sum_{i\in g}
  r_{i-1} r_i}$, yielding leading left singular vectors $Y \in \R^{n_g \times
  n_g'}$. Update $U_g \leftarrow U_g Y$ \emph{once} (assigned to every mode of
  $g$) and apply $Y^\top$ to the up-leg of every core in $g$.
\end{enumerate}
By Lemma~\ref{lem:skeleton}, step (4) computes the optimal rank-$n_g'$
truncation of the concatenation while touching the large dimension $N_g$ only
in the single product $U_g Y$, at cost $O(N_g n_g^2)$ per group; the SVD
itself acts on an $n_g \times \sum_i r_{i-1} r_i$ matrix.
\end{algorithm}
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-STM-0205

- file: `docs/shared_t3_math.tex`; old span: 205; kind: LaTeX equation line; fenced: y
- letters and widths: P
- roles: per §4.8: the T3's own factors → B_i, B_g (the §2 point algorithms, the corewise section :440–450); in the tangent sections (:318–420, :469–495 with rem:embedding as one range) the frame's down store → D_i, D_g, \dot D; the block B_i := U_i W_i (:279–292) → K_i; :306's B_i (the factor) stays; the tied coordinate B (:358, 551–566) → Z; n_{D,i} → n_{U,i}; O_g (a complement basis) stays; R(\T,\xi) (:474) stays; Q_i at :363 stays (session ruling)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
Let $P_{\mathrm{sf}}(\T)$ denote the output of
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-STM-0208

- file: `docs/shared_t3_math.tex`; old span: 208–217; kind: LaTeX equation line; fenced: y
- letters and widths: P
- roles: per §4.8: the T3's own factors → B_i, B_g (the §2 point algorithms, the corewise section :440–450); in the tangent sections (:318–420, :469–495 with rem:embedding as one range) the frame's down store → D_i, D_g, \dot D; the block B_i := U_i W_i (:279–292) → K_i; :306's B_i (the factor) stays; the tied coordinate B (:358, 551–566) → Z; n_{D,i} → n_{U,i}; O_g (a complement basis) stays; R(\T,\xi) (:474) stays; Q_i at :363 stays (session ruling)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\[
\bigl\| \T - P_{\mathrm{sf}}(\T) \bigr\|_{\HS}
\;\leq\;
C(d)\,
\inf_{\substack{\boldsymbol{Y} \text{ shared},\\
\mathrm{ranks}(\boldsymbol{Y}) \preceq (\boldsymbol{n},\boldsymbol{r})}}
\bigl\| \T - \boldsymbol{Y} \bigr\|_{\HS},
\qquad
C(d) = \sqrt{d} + \sqrt{d}\sqrt{d-1} + \sqrt{d-1}.
\]
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-STM-0221

- file: `docs/shared_t3_math.tex`; old span: 221–222; kind: LaTeX equation line; fenced: y
- letters and widths: P
- roles: per §4.8: the T3's own factors → B_i, B_g (the §2 point algorithms, the corewise section :440–450); in the tangent sections (:318–420, :469–495 with rem:embedding as one range) the frame's down store → D_i, D_g, \dot D; the block B_i := U_i W_i (:279–292) → K_i; :306's B_i (the factor) stays; the tied coordinate B (:358, 551–566) → Z; n_{D,i} → n_{U,i}; O_g (a complement basis) stays; R(\T,\xi) (:474) stays; Q_i at :363 stays (session ruling)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
Split $P_{\mathrm{sf}} = P_{\mathrm{SF\text{-}Tucker}} \circ
P_{\mathrm{TT}}$ (either order; \cite[Rem.~1]{MR2026}). Each stage is
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-STM-0272

- file: `docs/shared_t3_math.tex`; old span: 272–289; kind: LaTeX equation line; fenced: y
- letters and widths: B_i, Q, R, U_g, U_i, U_{i_1}, U_{i_k}
- roles: per §4.8: the T3's own factors → B_i, B_g (the §2 point algorithms, the corewise section :440–450); in the tangent sections (:318–420, :469–495 with rem:embedding as one range) the frame's down store → D_i, D_g, \dot D; the block B_i := U_i W_i (:279–292) → K_i; :306's B_i (the factor) stays; the tied coordinate B (:358, 551–566) → Z; n_{D,i} → n_{U,i}; O_g (a complement basis) stays; R(\T,\xi) (:474) stays; Q_i at :363 stays (session ruling)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\begin{algorithm}[Shared projection of an unshared T3]\label{alg:init}
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
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-STM-0292

- file: `docs/shared_t3_math.tex`; old span: 292–293; kind: LaTeX equation line; fenced: y
- letters and widths: B_{i_1}, B_{i_k}, Q
- roles: per §4.8: the T3's own factors → B_i, B_g (the §2 point algorithms, the corewise section :440–450); in the tangent sections (:318–420, :469–495 with rem:embedding as one range) the frame's down store → D_i, D_g, \dot D; the block B_i := U_i W_i (:279–292) → K_i; :306's B_i (the factor) stays; the tied coordinate B (:358, 551–566) → Z; n_{D,i} → n_{U,i}; O_g (a complement basis) stays; R(\T,\xi) (:474) stays; Q_i at :363 stays (session ruling)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
The column space of $[\,B_{i_1} | \cdots | B_{i_k}\,]$ lies in
$\operatorname{range}(Q)$, so step (2) computes the exact dominant left
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-STM-0306

- file: `docs/shared_t3_math.tex`; old span: 306; kind: LaTeX equation line; fenced: y
- letters and widths: B_i
- roles: per §4.8: the T3's own factors → B_i, B_g (the §2 point algorithms, the corewise section :440–450); in the tangent sections (:318–420, :469–495 with rem:embedding as one range) the frame's down store → D_i, D_g, \dot D; the block B_i := U_i W_i (:279–292) → K_i; :306's B_i (the factor) stays; the tied coordinate B (:358, 551–566) → Z; n_{D,i} → n_{U,i}; O_g (a complement basis) stays; R(\T,\xi) (:474) stays; Q_i at :363 stays (session ruling)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
($B_i = W_{[i]} \operatorname{diag}(s)\, V^\top$, valid for arbitrary --- not
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-STM-0325

- file: `docs/shared_t3_math.tex`; old span: 325; kind: LaTeX equation line; fenced: y
- letters and widths: U_i
- roles: per §4.8: the T3's own factors → B_i, B_g (the §2 point algorithms, the corewise section :440–450); in the tangent sections (:318–420, :469–495 with rem:embedding as one range) the frame's down store → D_i, D_g, \dot D; the block B_i := U_i W_i (:279–292) → K_i; :306's B_i (the factor) stays; the tied coordinate B (:358, 551–566) → Z; n_{D,i} → n_{U,i}; O_g (a complement basis) stays; R(\T,\xi) (:474) stays; Q_i at :363 stays (session ruling)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
$i$ satisfies $U_i^\top \dot V_i = 0$, and gauge conditions make all cross
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-STM-0347

- file: `docs/shared_t3_math.tex`; old span: 347–348; kind: LaTeX equation line; fenced: y
- letters and widths: U, U_g
- roles: per §4.8: the T3's own factors → B_i, B_g (the §2 point algorithms, the corewise section :440–450); in the tangent sections (:318–420, :469–495 with rem:embedding as one range) the frame's down store → D_i, D_g, \dot D; the block B_i := U_i W_i (:279–292) → K_i; :306's B_i (the factor) stays; the tied coordinate B (:358, 551–566) → Z; n_{D,i} → n_{U,i}; O_g (a complement basis) stays; R(\T,\xi) (:474) stays; Q_i at :363 stays (session ruling)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
a single \emph{gauged ambient direction} $\dot U \in \R^{N_g \times n_g}$
with $U_g^\top \dot U = 0$: in the ($S$-absorbed) frame coordinates,
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-STM-0349

- file: `docs/shared_t3_math.tex`; old span: 349–351; kind: LaTeX equation line; fenced: y
- letters and widths: U
- roles: per §4.8: the T3's own factors → B_i, B_g (the §2 point algorithms, the corewise section :440–450); in the tangent sections (:318–420, :469–495 with rem:embedding as one range) the frame's down store → D_i, D_g, \dot D; the block B_i := U_i W_i (:279–292) → K_i; :306's B_i (the factor) stays; the tied coordinate B (:358, 551–566) → Z; n_{D,i} → n_{U,i}; O_g (a complement basis) stays; R(\T,\xi) (:474) stays; Q_i at :363 stays (session ruling)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\begin{equation}\label{eq:tied}
C_i = \dot U\, S_i \quad \text{for a common gauged } \dot U, \qquad i \in g .
\end{equation}
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-STM-0358

- file: `docs/shared_t3_math.tex`; old span: 358; kind: LaTeX equation line; fenced: y
- letters and widths: B, O_g, U, U_g
- roles: per §4.8: the T3's own factors → B_i, B_g (the §2 point algorithms, the corewise section :440–450); in the tangent sections (:318–420, :469–495 with rem:embedding as one range) the frame's down store → D_i, D_g, \dot D; the block B_i := U_i W_i (:279–292) → K_i; :306's B_i (the factor) stays; the tied coordinate B (:358, 551–566) → Z; n_{D,i} → n_{U,i}; O_g (a complement basis) stays; R(\T,\xi) (:474) stays; Q_i at :363 stays (session ruling)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
$O_g$ of $U_g$ ($\dot U = O_g B$, coordinates $C_i = B S_i$), suggesting a
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-STM-0360

- file: `docs/shared_t3_math.tex`; old span: 360; kind: LaTeX equation line; fenced: y
- letters and widths: U_g
- roles: per §4.8: the T3's own factors → B_i, B_g (the §2 point algorithms, the corewise section :440–450); in the tangent sections (:318–420, :469–495 with rem:embedding as one range) the frame's down store → D_i, D_g, \dot D; the block B_i := U_i W_i (:279–292) → K_i; :306's B_i (the factor) stays; the tied coordinate B (:358, 551–566) → Z; n_{D,i} → n_{U,i}; O_g (a complement basis) stays; R(\T,\xi) (:474) stays; Q_i at :363 stays (session ruling)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
subspace depends only on $U_g$ (tied by hypothesis) and the $S_i$; no
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-STM-0363

- file: `docs/shared_t3_math.tex`; old span: 363; kind: LaTeX equation line; fenced: y
- letters and widths: Q_i
- roles: per §4.8: the T3's own factors → B_i, B_g (the §2 point algorithms, the corewise section :440–450); in the tangent sections (:318–420, :469–495 with rem:embedding as one range) the frame's down store → D_i, D_g, \dot D; the block B_i := U_i W_i (:279–292) → K_i; :306's B_i (the factor) stays; the tied coordinate B (:358, 551–566) → Z; n_{D,i} → n_{U,i}; O_g (a complement basis) stays; R(\T,\xi) (:474) stays; Q_i at :363 stays (session ruling)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
$\Gamma_i = S_i S_i^\top$ (replacing $S_i$ by $S_i Q_i$ together with the
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-STM-0366  (K16)

- file: `docs/shared_t3_math.tex`; old span: 366–367; kind: LaTeX equation line; fenced: y
- letters and widths: n_{D,i}
- roles: per §4.8: the T3's own factors → B_i, B_g (the §2 point algorithms, the corewise section :440–450); in the tangent sections (:318–420, :469–495 with rem:embedding as one range) the frame's down store → D_i, D_g, \dot D; the block B_i := U_i W_i (:279–292) → K_i; :306's B_i (the factor) stays; the tied coordinate B (:358, 551–566) → Z; n_{D,i} → n_{U,i}; O_g (a complement basis) stays; R(\T,\xi) (:474) stays; Q_i at :363 stays (session ruling)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- K16 width lines: 366, 367
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
implementation $S_i \in \R^{n_g \times n_{D,i}}$ with
$n_{D,i} = \min(n_g,\ r_{i-1} r_i)$ --- rectangular in general, so an
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-STM-0377

- file: `docs/shared_t3_math.tex`; old span: 377–384; kind: LaTeX equation line; fenced: y
- letters and widths: U
- roles: per §4.8: the T3's own factors → B_i, B_g (the §2 point algorithms, the corewise section :440–450); in the tangent sections (:318–420, :469–495 with rem:embedding as one range) the frame's down store → D_i, D_g, \dot D; the block B_i := U_i W_i (:279–292) → K_i; :306's B_i (the factor) stays; the tied coordinate B (:358, 551–566) → Z; n_{D,i} → n_{U,i}; O_g (a complement basis) stays; R(\T,\xi) (:474) stays; Q_i at :363 stays (session ruling)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\begin{equation}\label{eq:gramsolve}
\dot U \;=\; \Bigl(\sum_{i \in g} C_i S_i^\top\Bigr)
        \Bigl(\sum_{i \in g} S_i S_i^\top\Bigr)^{-1}
   \;=\; \Bigl(\sum_{i \in g} C_i S_i^\top\Bigr)
        \Bigl(\sum_{i \in g} \Gamma_i\Bigr)^{-1},
\qquad
C_i \leftarrow \dot U S_i .
\end{equation}
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-STM-0385

- file: `docs/shared_t3_math.tex`; old span: 385; kind: LaTeX equation line; fenced: y
- letters and widths: U
- roles: per §4.8: the T3's own factors → B_i, B_g (the §2 point algorithms, the corewise section :440–450); in the tangent sections (:318–420, :469–495 with rem:embedding as one range) the frame's down store → D_i, D_g, \dot D; the block B_i := U_i W_i (:279–292) → K_i; :306's B_i (the factor) stays; the tied coordinate B (:358, 551–566) → Z; n_{D,i} → n_{U,i}; O_g (a complement basis) stays; R(\T,\xi) (:474) stays; Q_i at :363 stays (session ruling)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
TT variations and singleton modes are unaffected, and $\dot U$ is
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-STM-0393

- file: `docs/shared_t3_math.tex`; old span: 393; kind: LaTeX equation line; fenced: y
- letters and widths: U
- roles: per §4.8: the T3's own factors → B_i, B_g (the §2 point algorithms, the corewise section :440–450); in the tangent sections (:318–420, :469–495 with rem:embedding as one range) the frame's down store → D_i, D_g, \dot D; the block B_i := U_i W_i (:279–292) → K_i; :306's B_i (the factor) stays; the tied coordinate B (:358, 551–566) → Z; n_{D,i} → n_{U,i}; O_g (a complement basis) stays; R(\T,\xi) (:474) stays; Q_i at :363 stays (session ruling)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
Minimize $\sum_{i\in g} \| C_i - \dot U S_i \|_F^2$ over $\dot U$; setting the
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-STM-0395

- file: `docs/shared_t3_math.tex`; old span: 395; kind: LaTeX equation line; fenced: y
- letters and widths: U
- roles: per §4.8: the T3's own factors → B_i, B_g (the §2 point algorithms, the corewise section :440–450); in the tangent sections (:318–420, :469–495 with rem:embedding as one range) the frame's down store → D_i, D_g, \dot D; the block B_i := U_i W_i (:279–292) → K_i; :306's B_i (the factor) stays; the tied coordinate B (:358, 551–566) → Z; n_{D,i} → n_{U,i}; O_g (a complement basis) stays; R(\T,\xi) (:474) stays; Q_i at :363 stays (session ruling)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
$\sum_i C_i S_i^\top = \dot U \sum_i S_i S_i^\top$, invertible whenever the
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-STM-0397

- file: `docs/shared_t3_math.tex`; old span: 397–398; kind: LaTeX equation line; fenced: y
- letters and widths: U, U_g
- roles: per §4.8: the T3's own factors → B_i, B_g (the §2 point algorithms, the corewise section :440–450); in the tangent sections (:318–420, :469–495 with rem:embedding as one range) the frame's down store → D_i, D_g, \dot D; the block B_i := U_i W_i (:279–292) → K_i; :306's B_i (the factor) stays; the tied coordinate B (:358, 551–566) → Z; n_{D,i} → n_{U,i}; O_g (a complement basis) stays; R(\T,\xi) (:474) stays; Q_i at :363 stays (session ruling)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
$U_g^\top \sum_i C_i S_i^\top = \sum_i (U_g^\top C_i) S_i^\top = 0$, and the
Gram sum acts on the other side, so $U_g^\top \dot U = 0$.
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-STM-0404  (K16)

- file: `docs/shared_t3_math.tex`; old span: 404–410; kind: LaTeX equation line; fenced: y
- letters and widths: n_{D,i}
- roles: per §4.8: the T3's own factors → B_i, B_g (the §2 point algorithms, the corewise section :440–450); in the tangent sections (:318–420, :469–495 with rem:embedding as one range) the frame's down store → D_i, D_g, \dot D; the block B_i := U_i W_i (:279–292) → K_i; :306's B_i (the factor) stays; the tied coordinate B (:358, 551–566) → Z; n_{D,i} → n_{U,i}; O_g (a complement basis) stays; R(\T,\xi) (:474) stays; Q_i at :363 stays (session ruling)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- K16 width lines: 406
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\[
M_g = \begin{bmatrix} S_{i_1}^\top \\ \vdots \\ S_{i_k}^\top \end{bmatrix}
\in \R^{(\sum_{i\in g} n_{D,i}) \times n_g},
\qquad
C^{\mathrm{stack}} = \begin{bmatrix} C_{i_1}^\top \\ \vdots \\
C_{i_k}^\top \end{bmatrix},
\]
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-STM-0412

- file: `docs/shared_t3_math.tex`; old span: 412–414; kind: LaTeX equation line; fenced: y
- letters and widths: U
- roles: per §4.8: the T3's own factors → B_i, B_g (the §2 point algorithms, the corewise section :440–450); in the tangent sections (:318–420, :469–495 with rem:embedding as one range) the frame's down store → D_i, D_g, \dot D; the block B_i := U_i W_i (:279–292) → K_i; :306's B_i (the factor) stays; the tied coordinate B (:358, 551–566) → Z; n_{D,i} → n_{U,i}; O_g (a complement basis) stays; R(\T,\xi) (:474) stays; Q_i at :363 stays (session ruling)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
$\dot U^\top = M_g^{+}\, C^{\mathrm{stack}}$ (then
$C_i^\top \leftarrow S_i^\top \dot U^\top$, the corresponding row block of
$M_g \dot U^\top$), computed from one thin SVD of $M_g$ with a relative
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-STM-0442

- file: `docs/shared_t3_math.tex`; old span: 442; kind: LaTeX equation line; fenced: y
- letters and widths: U_i
- roles: per §4.8: the T3's own factors → B_i, B_g (the §2 point algorithms, the corewise section :440–450); in the tangent sections (:318–420, :469–495 with rem:embedding as one range) the frame's down store → D_i, D_g, \dot D; the block B_i := U_i W_i (:279–292) → K_i; :306's B_i (the factor) stays; the tied coordinate B (:358, 551–566) → Z; n_{D,i} → n_{U,i}; O_g (a complement basis) stays; R(\T,\xi) (:474) stays; Q_i at :363 stays (session ruling)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
the raw cores $((U_i)_i, (G_i)_i)$, the metric is Euclidean on the core
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-STM-0444

- file: `docs/shared_t3_math.tex`; old span: 444; kind: LaTeX equation line; fenced: y
- letters and widths: U_i, U_j,
- roles: per §4.8: the T3's own factors → B_i, B_g (the §2 point algorithms, the corewise section :440–450); in the tangent sections (:318–420, :469–495 with rem:embedding as one range) the frame's down store → D_i, D_g, \dot D; the block B_i := U_i W_i (:279–292) → K_i; :306's B_i (the factor) stays; the tied coordinate B (:358, 551–566) → Z; n_{D,i} → n_{U,i}; O_g (a complement basis) stays; R(\T,\xi) (:474) stays; Q_i at :363 stays (session ruling)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
parameter set $\{\,U_i = U_j,\ i, j \in g,\ \forall g\,\}$ is a linear
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-STM-0446

- file: `docs/shared_t3_math.tex`; old span: 446; kind: LaTeX equation line; fenced: y
- letters and widths: U_i
- roles: per §4.8: the T3's own factors → B_i, B_g (the §2 point algorithms, the corewise section :440–450); in the tangent sections (:318–420, :469–495 with rem:embedding as one range) the frame's down store → D_i, D_g, \dot D; the block B_i := U_i W_i (:279–292) → K_i; :306's B_i (the factor) stays; the tied coordinate B (:358, 551–566) → Z; n_{D,i} → n_{U,i}; O_g (a complement basis) stays; R(\T,\xi) (:474) stays; Q_i at :363 stays (session ruling)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
$\{\dot U_i \text{ equal within each group}\}$ (TT-core perturbations
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-STM-0449

- file: `docs/shared_t3_math.tex`; old span: 449; kind: LaTeX equation line; fenced: y
- letters and widths: U_i, U_j
- roles: per §4.8: the T3's own factors → B_i, B_g (the §2 point algorithms, the corewise section :440–450); in the tangent sections (:318–420, :469–495 with rem:embedding as one range) the frame's down store → D_i, D_g, \dot D; the block B_i := U_i W_i (:279–292) → K_i; :306's B_i (the factor) stays; the tied coordinate B (:358, 551–566) → Z; n_{D,i} → n_{U,i}; O_g (a complement basis) stays; R(\T,\xi) (:474) stays; Q_i at :363 stays (session ruling)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
$\dot U_i \mapsto \tfrac{1}{k_g} \sum_{j \in g} \dot U_j$. The additive
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-STM-0472

- file: `docs/shared_t3_math.tex`; old span: 472; kind: LaTeX equation line; fenced: y
- letters and widths: U_g
- roles: per §4.8: the T3's own factors → B_i, B_g (the §2 point algorithms, the corewise section :440–450); in the tangent sections (:318–420, :469–495 with rem:embedding as one range) the frame's down store → D_i, D_g, \dot D; the block B_i := U_i W_i (:279–292) → K_i; :306's B_i (the factor) stays; the tied coordinate B (:358, 551–566) → Z; n_{D,i} → n_{U,i}; O_g (a complement basis) stays; R(\T,\xi) (:474) stays; Q_i at :363 stays (session ruling)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
factor is $[\,U_g \,|\, \dot U_g\,]$, identical across each group), and
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-STM-0473

- file: `docs/shared_t3_math.tex`; old span: 473–475; kind: LaTeX equation line; fenced: y
- letters and widths: P, R
- roles: per §4.8: the T3's own factors → B_i, B_g (the §2 point algorithms, the corewise section :440–450); in the tangent sections (:318–420, :469–495 with rem:embedding as one range) the frame's down store → D_i, D_g, \dot D; the block B_i := U_i W_i (:279–292) → K_i; :306's B_i (the factor) stays; the tied coordinate B (:358, 551–566) → Z; n_{D,i} → n_{U,i}; O_g (a complement basis) stays; R(\T,\xi) (:474) stays; Q_i at :363 stays (session ruling)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\[
R(\T, \xi) \coloneqq P_{\mathrm{sf}}(\T + \xi)
\]
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-STM-0484

- file: `docs/shared_t3_math.tex`; old span: 484; kind: LaTeX equation line; fenced: y
- letters and widths: U, U_g
- roles: per §4.8: the T3's own factors → B_i, B_g (the §2 point algorithms, the corewise section :440–450); in the tangent sections (:318–420, :469–495 with rem:embedding as one range) the frame's down store → D_i, D_g, \dot D; the block B_i := U_i W_i (:279–292) → K_i; :306's B_i (the factor) stays; the tied coordinate B (:358, 551–566) → Z; n_{D,i} → n_{U,i}; O_g (a complement basis) stays; R(\T,\xi) (:474) stays; Q_i at :363 stays (session ruling)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
after the fact: per group, the doubled factor is $[\,U_g \,|\, \dot U\,]$
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-STM-0489

- file: `docs/shared_t3_math.tex`; old span: 489; kind: LaTeX equation line; fenced: y
- letters and widths: U, U_g
- roles: per §4.8: the T3's own factors → B_i, B_g (the §2 point algorithms, the corewise section :440–450); in the tangent sections (:318–420, :469–495 with rem:embedding as one range) the frame's down store → D_i, D_g, \dot D; the block B_i := U_i W_i (:279–292) → K_i; :306's B_i (the factor) stays; the tied coordinate B (:358, 551–566) → Z; n_{D,i} → n_{U,i}; O_g (a complement basis) stays; R(\T,\xi) (:474) stays; Q_i at :363 stays (session ruling)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
per-mode factor blocks $[\,U_g \,|\, \dot U S_i\,]$ (the $S$-absorbed
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-STM-0491  (K16)

- file: `docs/shared_t3_math.tex`; old span: 491–492; kind: LaTeX equation line; fenced: y
- letters and widths: U, n_{D,i}
- roles: per §4.8: the T3's own factors → B_i, B_g (the §2 point algorithms, the corewise section :440–450); in the tangent sections (:318–420, :469–495 with rem:embedding as one range) the frame's down store → D_i, D_g, \dot D; the block B_i := U_i W_i (:279–292) → K_i; :306's B_i (the factor) stays; the tied coordinate B (:358, 551–566) → Z; n_{D,i} → n_{U,i}; O_g (a complement basis) stays; R(\T,\xi) (:474) stays; Q_i at :363 stays (session ruling)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- K16 width lines: 491
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
and, when the $n_{D,i}$ differ, even in shape, and no per-mode averaging
repairs this. $\dot U$ is recovered from tied coordinates by the
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-STM-0551

- file: `docs/shared_t3_math.tex`; old span: 551; kind: LaTeX equation line; fenced: y
- letters and widths: B
- roles: per §4.8: the T3's own factors → B_i, B_g (the §2 point algorithms, the corewise section :440–450); in the tangent sections (:318–420, :469–495 with rem:embedding as one range) the frame's down store → D_i, D_g, \dot D; the block B_i := U_i W_i (:279–292) → K_i; :306's B_i (the factor) stays; the tied coordinate B (:358, 551–566) → Z; n_{D,i} → n_{U,i}; O_g (a complement basis) stays; R(\T,\xi) (:474) stays; Q_i at :363 stays (session ruling)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\item The Gram of the tied parameterization $B \mapsto (BS_i)_{i \in g}$ is
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-STM-0553

- file: `docs/shared_t3_math.tex`; old span: 553; kind: LaTeX equation line; fenced: y
- letters and widths: B
- roles: per §4.8: the T3's own factors → B_i, B_g (the §2 point algorithms, the corewise section :440–450); in the tangent sections (:318–420, :469–495 with rem:embedding as one range) the frame's down store → D_i, D_g, \dot D; the block B_i := U_i W_i (:279–292) → K_i; :306's B_i (the factor) stays; the tied coordinate B (:358, 551–566) → Z; n_{D,i} → n_{U,i}; O_g (a complement basis) stays; R(\T,\xi) (:474) stays; Q_i at :363 stays (session ruling)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
  $\|\xi\|_{\HS}^2 = \tr\bigl(B (\sum_i \Gamma_i) B^\top\bigr)$. Its
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-STM-0566

- file: `docs/shared_t3_math.tex`; old span: 566; kind: LaTeX equation line; fenced: y
- letters and widths: B
- roles: per §4.8: the T3's own factors → B_i, B_g (the §2 point algorithms, the corewise section :440–450); in the tangent sections (:318–420, :469–495 with rem:embedding as one range) the frame's down store → D_i, D_g, \dot D; the block B_i := U_i W_i (:279–292) → K_i; :306's B_i (the factor) stays; the tied coordinate B (:358, 551–566) → Z; n_{D,i} → n_{U,i}; O_g (a complement basis) stays; R(\T,\xi) (:474) stays; Q_i at :363 stays (session ruling)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
frame coordinates: $\sum_i \|BS_i\|_F^2 = \tr(B(\sum_i S_iS_i^\top)B^\top)$.
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~


### `docs/symmetric_probe_derivatives.tex`

#### TEX-SPD-0048

- file: `docs/symmetric_probe_derivatives.tex`; old span: 48; kind: LaTeX equation line; fenced: y
- letters and widths: P
- roles: per §4.8: frame U/O/P/Q → D/U/L/R; D^t (:111) → \mathrm d^t (its P^t is the direction and stays); the T3's own \mathbf U_i → \mathbf B_i; the plain B of :158–169 (a bilinear map) stays; the corewise sentence :556–559 per §1.2a
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\emph{same} direction $\mathbf P$, repeated $t$ times --- and show they are computed by a single
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-SPD-0066

- file: `docs/symmetric_probe_derivatives.tex`; old span: 66–67; kind: LaTeX equation line; fenced: y
- letters and widths: U_i
- roles: per §4.8: frame U/O/P/Q → D/U/L/R; D^t (:111) → \mathrm d^t (its P^t is the direction and stays); the T3's own \mathbf U_i → \mathbf B_i; the plain B of :158–169 (a bilinear map) stays; the corewise sentence :556–559 per §1.2a
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
A \emph{Tucker tensor train} is a pair of core families $\mathbb T = \big((\mathbf U_i)_{i=1}^d,\,
(\mathbf G_i)_{i=1}^d\big)$, where each \emph{Tucker factor} $\mathbf U_i \in \R^{n_i \times N_i}$
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-SPD-0085

- file: `docs/symmetric_probe_derivatives.tex`; old span: 85–98; kind: LaTeX equation line; fenced: y
- letters and widths: U_i
- roles: per §4.8: frame U/O/P/Q → D/U/L/R; D^t (:111) → \mathrm d^t (its P^t is the direction and stays); the T3's own \mathbf U_i → \mathbf B_i; the plain B of :158–169 (a bilinear map) stays; the corewise sentence :556–559 per §1.2a
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\begin{align}
  \boldsymbol\xi_i      &= \mathbf U_i \mathbf w_i \in \R^{n_i},
  & &\text{(projected probe)} \label{eq:xi}\\
  \boldsymbol\mu_0^\top  &= 1, &
  \boldsymbol\mu_i^\top &= \boldsymbol\mu_{i-1}^\top\, \mathbf G_i(\boldsymbol\xi_i) \in \R^{r_i},
  & &\text{(left pushthrough)} \label{eq:mu}\\
  \boldsymbol\nu_d      &= 1, &
  \boldsymbol\nu_i      &= \mathbf G_{i+1}(\boldsymbol\xi_{i+1})\, \boldsymbol\nu_{i+1} \in \R^{r_i},
  & &\text{(right pushthrough)} \label{eq:nu}\\
  \boldsymbol\eta_i     &= \boldsymbol\mu_{i-1}^\top\, \mathbf G_i\, \boldsymbol\nu_i \in \R^{n_i},
  & &\text{(combine, mode $i$ free)} \label{eq:eta}\\
  \mathbf z_i           &= \mathbf U_i^\top \boldsymbol\eta_i \in \R^{N_i}.
  & &\text{(lift)} \label{eq:z}
\end{align}
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-SPD-0101

- file: `docs/symmetric_probe_derivatives.tex`; old span: 101; kind: LaTeX equation line; fenced: y
- letters and widths: U_i
- roles: per §4.8: frame U/O/P/Q → D/U/L/R; D^t (:111) → \mathrm d^t (its P^t is the direction and stays); the T3's own \mathbf U_i → \mathbf B_i; the plain B of :158–169 (a bilinear map) stays; the corewise sentence :556–559 per §1.2a
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
$\mathbf G_i$, leaving the middle index free. The single Tucker factor $\mathbf U_i$ is used twice:
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-SPD-0105

- file: `docs/symmetric_probe_derivatives.tex`; old span: 105; kind: LaTeX equation line; fenced: y
- letters and widths: P
- roles: per §4.8: frame U/O/P/Q → D/U/L/R; D^t (:111) → \mathrm d^t (its P^t is the direction and stays); the T3's own \mathbf U_i → \mathbf B_i; the plain B of :158–169 (a bilinear map) stays; the corewise sentence :556–559 per §1.2a
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
Fix a direction $\mathbf P = (\mathbf p_1, \dots, \mathbf p_d)$, $\mathbf p_i \in \R^{N_i}$. We seek
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-SPD-0107

- file: `docs/symmetric_probe_derivatives.tex`; old span: 107; kind: LaTeX equation line; fenced: y
- letters and widths: P
- roles: per §4.8: frame U/O/P/Q → D/U/L/R; D^t (:111) → \mathrm d^t (its P^t is the direction and stays); the T3's own \mathbf U_i → \mathbf B_i; the plain B of :158–169 (a bilinear map) stays; the corewise sentence :556–559 per §1.2a
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
times in the single repeated direction $\mathbf P$:
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-SPD-0108

- file: `docs/symmetric_probe_derivatives.tex`; old span: 108–113; kind: LaTeX equation line; fenced: y
- letters and widths: D^t, P, P^{t}
- roles: per §4.8: frame U/O/P/Q → D/U/L/R; D^t (:111) → \mathrm d^t (its P^t is the direction and stays); the T3's own \mathbf U_i → \mathbf B_i; the plain B of :158–169 (a bilinear map) stays; the corewise sentence :556–559 per §1.2a
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\begin{equation}
  \label{eq:goal}
  \mathbf z_i^{(t)} \;:=\; \frac{\dd{^t}}{\dd s^t}\, \mathbf z_i(\mathbf W + s\mathbf P)\Big|_{s=0}
  \;=\; D^t \mathbf z_i(\mathbf W)\,\mathbf P^{t},
  \qquad t = 0, 1, \dots, m.
\end{equation}
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-SPD-0121

- file: `docs/symmetric_probe_derivatives.tex`; old span: 121; kind: LaTeX equation line; fenced: y
- letters and widths: P
- roles: per §4.8: frame U/O/P/Q → D/U/L/R; D^t (:111) → \mathrm d^t (its P^t is the direction and stays); the T3's own \mathbf U_i → \mathbf B_i; the plain B of :158–169 (a bilinear map) stays; the corewise sentence :556–559 per §1.2a
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
$\mathbf W + s\mathbf P$ and tracking its Taylor data.
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-SPD-0126

- file: `docs/symmetric_probe_derivatives.tex`; old span: 126; kind: LaTeX equation line; fenced: y
- letters and widths: P
- roles: per §4.8: frame U/O/P/Q → D/U/L/R; D^t (:111) → \mathrm d^t (its P^t is the direction and stays); the T3's own \mathbf U_i → \mathbf B_i; the plain B of :158–169 (a bilinear map) stays; the corewise sentence :556–559 per §1.2a
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
(along $\mathbf P$, at $\mathbf W$) is the tuple of symmetric derivatives
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-SPD-0127

- file: `docs/symmetric_probe_derivatives.tex`; old span: 127–131; kind: LaTeX equation line; fenced: y
- letters and widths: P
- roles: per §4.8: frame U/O/P/Q → D/U/L/R; D^t (:111) → \mathrm d^t (its P^t is the direction and stays); the T3's own \mathbf U_i → \mathbf B_i; the plain B of :158–169 (a bilinear map) stays; the corewise sentence :556–559 per §1.2a
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\[
  \jet{q} \;=\; \big(q^{(0)}, q^{(1)}, \dots, q^{(m)}\big),
  \qquad
  q^{(t)} := \frac{\dd{^t}}{\dd s^t}\, q(\mathbf W + s\mathbf P)\Big|_{s=0}.
\]
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-SPD-0144

- file: `docs/symmetric_probe_derivatives.tex`; old span: 144–147; kind: LaTeX equation line; fenced: y
- letters and widths: U_i
- roles: per §4.8: frame U/O/P/Q → D/U/L/R; D^t (:111) → \mathrm d^t (its P^t is the direction and stays); the T3's own \mathbf U_i → \mathbf B_i; the plain B of :158–169 (a bilinear map) stays; the corewise sentence :556–559 per §1.2a
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\[
  \jet{\boldsymbol\xi_i} = (\boldsymbol\xi_i,\ \delta\boldsymbol\xi_i,\ 0, \dots, 0),
  \qquad \delta\boldsymbol\xi_i := \mathbf U_i \mathbf p_i.
\]
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-SPD-0152

- file: `docs/symmetric_probe_derivatives.tex`; old span: 152; kind: LaTeX equation line; fenced: y
- letters and widths: U_i
- roles: per §4.8: frame U/O/P/Q → D/U/L/R; D^t (:111) → \mathrm d^t (its P^t is the direction and stays); the T3's own \mathbf U_i → \mathbf B_i; the plain B of :158–169 (a bilinear map) stays; the corewise sentence :556–559 per §1.2a
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
$\boldsymbol\xi_i(s) = \mathbf U_i \mathbf w_i(s) = \mathbf U_i \mathbf w_i + s\,\mathbf U_i \mathbf
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-SPD-0158

- file: `docs/symmetric_probe_derivatives.tex`; old span: 158–159; kind: LaTeX equation line; fenced: y
- letters and widths: B
- roles: per §4.8: frame U/O/P/Q → D/U/L/R; D^t (:111) → \mathrm d^t (its P^t is the direction and stays); the T3's own \mathbf U_i → \mathbf B_i; the plain B of :158–169 (a bilinear map) stays; the corewise sentence :556–559 per §1.2a
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
Let $B : \R^{A} \times \R^{B} \to \R^{C}$ be a fixed bilinear map, and let $a(s), b(s)$ depend
smoothly on $s$. Then the jet of $s \mapsto B(a(s), b(s))$ is the binomial convolution of the jets of
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-SPD-0161

- file: `docs/symmetric_probe_derivatives.tex`; old span: 161–165; kind: LaTeX equation line; fenced: y
- letters and widths: B
- roles: per §4.8: frame U/O/P/Q → D/U/L/R; D^t (:111) → \mathrm d^t (its P^t is the direction and stays); the T3's own \mathbf U_i → \mathbf B_i; the plain B of :158–169 (a bilinear map) stays; the corewise sentence :556–559 per §1.2a
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\begin{equation}
  \label{eq:leibniz}
  \big[B(a,b)\big]^{(t)} \;=\; \sum_{r=0}^{t} \binom{t}{r}\, B\!\big(a^{(r)},\, b^{(t-r)}\big),
  \qquad t = 0, 1, \dots, m.
\end{equation}
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-SPD-0169

- file: `docs/symmetric_probe_derivatives.tex`; old span: 169–170; kind: LaTeX equation line; fenced: y
- letters and widths: B
- roles: per §4.8: frame U/O/P/Q → D/U/L/R; D^t (:111) → \mathrm d^t (its P^t is the direction and stays); the T3's own \mathbf U_i → \mathbf B_i; the plain B of :158–169 (a bilinear map) stays; the corewise sentence :556–559 per §1.2a
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
By bilinearity, $\tfrac{\dd}{\dd s} B(a,b) = B(a',b) + B(a,b')$. This is the ordinary product rule
for the bilinear pairing $B$, so the general Leibniz rule for the $t$th derivative of a product gives
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-SPD-0195

- file: `docs/symmetric_probe_derivatives.tex`; old span: 195; kind: LaTeX equation line; fenced: y
- letters and widths: U_i,
- roles: per §4.8: frame U/O/P/Q → D/U/L/R; D^t (:111) → \mathrm d^t (its P^t is the direction and stays); the T3's own \mathbf U_i → \mathbf B_i; the plain B of :158–169 (a bilinear map) stays; the corewise sentence :556–559 per §1.2a
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
U_i, \mathbf G_i$ are constants (independent of $s$), so the only $s$-dependence is carried by the
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-SPD-0217

- file: `docs/symmetric_probe_derivatives.tex`; old span: 217; kind: LaTeX equation line; fenced: y
- letters and widths: B
- roles: per §4.8: frame U/O/P/Q → D/U/L/R; D^t (:111) → \mathrm d^t (its P^t is the direction and stays); the T3's own \mathbf U_i → \mathbf B_i; the plain B of :158–169 (a bilinear map) stays; the corewise sentence :556–559 per §1.2a
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
The recursion \eqref{eq:mu} reads $\boldsymbol\mu_i^\top = B(\boldsymbol\mu_{i-1},
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-SPD-0241

- file: `docs/symmetric_probe_derivatives.tex`; old span: 241–248; kind: LaTeX equation line; fenced: y
- letters and widths: U_i
- roles: per §4.8: frame U/O/P/Q → D/U/L/R; D^t (:111) → \mathrm d^t (its P^t is the direction and stays); the T3's own \mathbf U_i → \mathbf B_i; the plain B of :158–169 (a bilinear map) stays; the corewise sentence :556–559 per §1.2a
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\begin{align}
  \boldsymbol\eta_i^{(t)}
    &= \sum_{r=0}^{t} \binom{t}{r}\,
       \big(\boldsymbol\mu_{i-1}^{(r)}\big)^\top \mathbf G_i\, \boldsymbol\nu_i^{(t-r)},
    \label{eq:deta}\\[2pt]
  \mathbf z_i^{(t)} &= \mathbf U_i^\top \boldsymbol\eta_i^{(t)}.
    \label{eq:dz}
\end{align}
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-SPD-0252

- file: `docs/symmetric_probe_derivatives.tex`; old span: 252–253; kind: LaTeX equation line; fenced: y
- letters and widths: B
- roles: per §4.8: frame U/O/P/Q → D/U/L/R; D^t (:111) → \mathrm d^t (its P^t is the direction and stays); the T3's own \mathbf U_i → \mathbf B_i; the plain B of :158–169 (a bilinear map) stays; the corewise sentence :556–559 per §1.2a
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
The combine \eqref{eq:eta} is $\boldsymbol\eta_i = B(\boldsymbol\mu_{i-1}, \boldsymbol\nu_i)$ with the
bilinear map $B(\mathbf u, \mathbf v) = \mathbf u^\top \mathbf G_i \mathbf v$ (mode $i$ free), so
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-SPD-0256

- file: `docs/symmetric_probe_derivatives.tex`; old span: 256–257; kind: LaTeX equation line; fenced: y
- letters and widths: U_i
- roles: per §4.8: frame U/O/P/Q → D/U/L/R; D^t (:111) → \mathrm d^t (its P^t is the direction and stays); the T3's own \mathbf U_i → \mathbf B_i; the plain B of :158–169 (a bilinear map) stays; the corewise sentence :556–559 per §1.2a
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
linear map $\boldsymbol\eta_i \mapsto \mathbf U_i^\top \boldsymbol\eta_i$, whose jet acts
componentwise: $\mathbf z_i^{(t)} = \mathbf U_i^\top \boldsymbol\eta_i^{(t)}$.
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-SPD-0276

- file: `docs/symmetric_probe_derivatives.tex`; old span: 276–302; kind: LaTeX equation line; fenced: y
- letters and widths: P, U_i
- roles: per §4.8: frame U/O/P/Q → D/U/L/R; D^t (:111) → \mathrm d^t (its P^t is the direction and stays); the T3's own \mathbf U_i → \mathbf B_i; the plain B of :158–169 (a bilinear map) stays; the corewise sentence :556–559 per §1.2a
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\begin{algorithm}[h]
\caption{Symmetric probe derivatives of a Tucker tensor train}
\label{alg:sweep}
\begin{algorithmic}[1]
\Require T3 $\big((\mathbf U_i),(\mathbf G_i)\big)$; inputs $\mathbf W$; direction $\mathbf P$; order $m$
\Ensure derivative jets $\jet{\mathbf z_i}$ for $i=1,\dots,d$
\For{$i = 1,\dots,d$} \Comment{project, forming the order-one input jets}
  \State $\boldsymbol\xi_i \gets \mathbf U_i \mathbf w_i$,\quad
         $\delta\boldsymbol\xi_i \gets \mathbf U_i \mathbf p_i$
\EndFor
\State $\jet{\boldsymbol\mu_0} \gets (1,0,\dots,0)$
\For{$i = 1,\dots,d$} \Comment{left sweep, Eq.~\eqref{eq:dmu}}
  \State $\boldsymbol\mu_i^{(t)} \gets \boldsymbol\mu_{i-1}^{(t)}\mathbf G_i(\boldsymbol\xi_i)
         + t\,\boldsymbol\mu_{i-1}^{(t-1)}\mathbf G_i(\delta\boldsymbol\xi_i)$ \textbf{ for } $t=0,\dots,m$
\EndFor
\State $\jet{\boldsymbol\nu_d} \gets (1,0,\dots,0)$
\For{$i = d,\dots,1$} \Comment{right sweep, Eq.~\eqref{eq:dnu}}
  \State $\boldsymbol\nu_{i-1}^{(t)} \gets \mathbf G_i(\boldsymbol\xi_i)\boldsymbol\nu_i^{(t)}
         + t\,\mathbf G_i(\delta\boldsymbol\xi_i)\boldsymbol\nu_i^{(t-1)}$ \textbf{ for } $t=0,\dots,m$
\EndFor
\For{$i = 1,\dots,d$} \Comment{combine and lift, Eqs.~\eqref{eq:deta}--\eqref{eq:dz}}
  \State $\boldsymbol\eta_i^{(t)} \gets \sum_{r=0}^{t}\binom{t}{r}
         \big(\boldsymbol\mu_{i-1}^{(r)}\big)^\top \mathbf G_i\,\boldsymbol\nu_i^{(t-r)}$ \textbf{ for } $t=0,\dots,m$
  \State $\mathbf z_i^{(t)} \gets \mathbf U_i^\top \boldsymbol\eta_i^{(t)}$ \textbf{ for } $t=0,\dots,m$
\EndFor
\end{algorithmic}
\end{algorithm}
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-SPD-0349

- file: `docs/symmetric_probe_derivatives.tex`; old span: 349; kind: LaTeX equation line; fenced: y
- letters and widths: P
- roles: per §4.8: frame U/O/P/Q → D/U/L/R; D^t (:111) → \mathrm d^t (its P^t is the direction and stays); the T3's own \mathbf U_i → \mathbf B_i; the plain B of :158–169 (a bilinear map) stays; the corewise sentence :556–559 per §1.2a
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
$(\mathbf W, \mathbf P)$ pairs at once: each sample is a paired input/direction, so to sweep many
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-SPD-0353

- file: `docs/symmetric_probe_derivatives.tex`; old span: 353; kind: LaTeX equation line; fenced: y
- letters and widths: P
- roles: per §4.8: frame U/O/P/Q → D/U/L/R; D^t (:111) → \mathrm d^t (its P^t is the direction and stays); the T3's own \mathbf U_i → \mathbf B_i; the plain B of :158–169 (a bilinear map) stays; the corewise sentence :556–559 per §1.2a
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
Each action is multilinear in the off-mode inputs, so $\mathbf z_i(\mathbf W + s\mathbf P)$ is a
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-SPD-0373

- file: `docs/symmetric_probe_derivatives.tex`; old span: 373–374; kind: LaTeX equation line; fenced: y
- letters and widths: O_i, P_i, Q_i, U_i
- roles: per §4.8: frame U/O/P/Q → D/U/L/R; D^t (:111) → \mathrm d^t (its P^t is the direction and stays); the T3's own \mathbf U_i → \mathbf B_i; the plain B of :158–169 (a bilinear map) stays; the corewise sentence :556–559 per §1.2a
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
$\big((\mathbf U_i),(\mathbf O_i),(\mathbf P_i),(\mathbf Q_i)\big)$ (up-Tucker, down-, left-, and
right-TT cores) and a gauged variation $\delta\mathbb V = \big((\delta\mathbf U_i),
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-SPD-0377

- file: `docs/symmetric_probe_derivatives.tex`; old span: 377–383; kind: LaTeX equation line; fenced: y
- letters and widths: O_i, P_i, Q_{i, U_i
- roles: per §4.8: frame U/O/P/Q → D/U/L/R; D^t (:111) → \mathrm d^t (its P^t is the direction and stays); the T3's own \mathbf U_i → \mathbf B_i; the plain B of :158–169 (a bilinear map) stays; the corewise sentence :556–559 per §1.2a
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\begin{equation}
  \label{eq:rbase}
  \boldsymbol\xi_i = \mathbf U_i \mathbf w_i, \quad
  \boldsymbol\mu_i^\top = \boldsymbol\mu_{i-1}^\top \mathbf P_i(\boldsymbol\xi_i), \quad
  \boldsymbol\nu_i = \mathbf Q_{i+1}(\boldsymbol\xi_{i+1})\,\boldsymbol\nu_{i+1}, \quad
  \boldsymbol\eta_i = \boldsymbol\mu_{i-1}^\top \mathbf O_i\, \boldsymbol\nu_i,
\end{equation}
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-SPD-0384

- file: `docs/symmetric_probe_derivatives.tex`; old span: 384; kind: LaTeX equation line; fenced: y
- letters and widths: O, P, Q
- roles: per §4.8: frame U/O/P/Q → D/U/L/R; D^t (:111) → \mathrm d^t (its P^t is the direction and stays); the T3's own \mathbf U_i → \mathbf B_i; the plain B of :158–169 (a bilinear map) stays; the corewise sentence :556–559 per §1.2a
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
(the plain pipeline \eqref{eq:xi}--\eqref{eq:z} with the frame cores $\mathbf P,\mathbf Q,\mathbf O$
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-SPD-0386

- file: `docs/symmetric_probe_derivatives.tex`; old span: 386–399; kind: LaTeX equation line; fenced: y
- letters and widths: O_i, O_{i, P_i, P_{i, Q_i, U_i
- roles: per §4.8: frame U/O/P/Q → D/U/L/R; D^t (:111) → \mathrm d^t (its P^t is the direction and stays); the T3's own \mathbf U_i → \mathbf B_i; the plain B of :158–169 (a bilinear map) stays; the corewise sentence :556–559 per §1.2a
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\begin{align}
  \delta\boldsymbol\xi_i &= \delta\mathbf U_i \mathbf w_i, \label{eq:rdxi}\\
  \boldsymbol\sigma_i^\top &= \boldsymbol\sigma_{i-1}^\top \mathbf Q_i(\boldsymbol\xi_i)
    + \boldsymbol\mu_{i-1}^\top \delta\mathbf G_i(\boldsymbol\xi_i)
    + \boldsymbol\mu_{i-1}^\top \mathbf O_i(\delta\boldsymbol\xi_i), \label{eq:rsigma}\\
  \boldsymbol\tau_i &= \delta\mathbf G_{i+1}(\boldsymbol\xi_{i+1})\boldsymbol\nu_{i+1}
    + \mathbf O_{i+1}(\delta\boldsymbol\xi_{i+1})\boldsymbol\nu_{i+1}
    + \mathbf P_{i+1}(\boldsymbol\xi_{i+1})\boldsymbol\tau_{i+1}, \label{eq:rtau}\\
  \delta\boldsymbol\eta_i &= \boldsymbol\sigma_{i-1}^\top \mathbf Q_i \boldsymbol\nu_i
    + \boldsymbol\mu_{i-1}^\top \delta\mathbf G_i \boldsymbol\nu_i
    + \boldsymbol\mu_{i-1}^\top \mathbf P_i \boldsymbol\tau_i, \label{eq:rdeta}\\
  \mathbf z_i &= \mathbf U_i^\top \delta\boldsymbol\eta_i + \delta\mathbf U_i^\top \boldsymbol\eta_i,
    \label{eq:rz}
\end{align}
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-SPD-0403

- file: `docs/symmetric_probe_derivatives.tex`; old span: 403; kind: LaTeX equation line; fenced: y
- letters and widths: P
- roles: per §4.8: frame U/O/P/Q → D/U/L/R; D^t (:111) → \mathrm d^t (its P^t is the direction and stays); the T3's own \mathbf U_i → \mathbf B_i; the plain B of :158–169 (a bilinear map) stays; the corewise sentence :556–559 per §1.2a
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
Differentiating \eqref{eq:rbase}--\eqref{eq:rz} with respect to $\mathbf W$ along $\mathbf P$ uses
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-SPD-0407

- file: `docs/symmetric_probe_derivatives.tex`; old span: 407–408; kind: LaTeX equation line; fenced: y
- letters and widths: U_i
- roles: per §4.8: frame U/O/P/Q → D/U/L/R; D^t (:111) → \mathrm d^t (its P^t is the direction and stays); the T3's own \mathbf U_i → \mathbf B_i; the plain B of :158–169 (a bilinear map) stays; the corewise sentence :556–559 per §1.2a
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
order-one input jets $\jet{\boldsymbol\xi_i} = (\boldsymbol\xi_i, \mathbf U_i\mathbf p_i, 0, \dots)$
and $\jet{\delta\boldsymbol\xi_i} = (\delta\boldsymbol\xi_i, \delta\mathbf U_i\mathbf p_i, 0, \dots)$,
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-SPD-0410

- file: `docs/symmetric_probe_derivatives.tex`; old span: 410–419; kind: LaTeX equation line; fenced: y
- letters and widths: O_i, Q_i, U_i
- roles: per §4.8: frame U/O/P/Q → D/U/L/R; D^t (:111) → \mathrm d^t (its P^t is the direction and stays); the T3's own \mathbf U_i → \mathbf B_i; the plain B of :158–169 (a bilinear map) stays; the corewise sentence :556–559 per §1.2a
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\begin{align}
  \big(\boldsymbol\sigma_i^{(t)}\big)^\top
  &= \big(\boldsymbol\sigma_{i-1}^{(t)}\big)^\top \mathbf Q_i(\boldsymbol\xi_i)
   + t \big(\boldsymbol\sigma_{i-1}^{(t-1)}\big)^\top \mathbf Q_i(\mathbf U_i\mathbf p_i) \nonumber\\
  &\;+ \big(\boldsymbol\mu_{i-1}^{(t)}\big)^\top \delta\mathbf G_i(\boldsymbol\xi_i)
     + t \big(\boldsymbol\mu_{i-1}^{(t-1)}\big)^\top \delta\mathbf G_i(\mathbf U_i\mathbf p_i) \nonumber\\
  &\;+ \big(\boldsymbol\mu_{i-1}^{(t)}\big)^\top \mathbf O_i(\delta\boldsymbol\xi_i)
     + t \big(\boldsymbol\mu_{i-1}^{(t-1)}\big)^\top \mathbf O_i(\delta\mathbf U_i\mathbf p_i),
  \label{eq:rdsigma}
\end{align}
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-SPD-0428

- file: `docs/symmetric_probe_derivatives.tex`; old span: 428; kind: LaTeX equation line; fenced: y
- letters and widths: O, P, Q
- roles: per §4.8: frame U/O/P/Q → D/U/L/R; D^t (:111) → \mathrm d^t (its P^t is the direction and stays); the T3's own \mathbf U_i → \mathbf B_i; the plain B of :158–169 (a bilinear map) stays; the corewise sentence :556–559 per §1.2a
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
Section~\ref{sec:impl}, fed a frame core ($\mathbf P,\mathbf Q,\mathbf O$) or the variation core
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-SPD-0438

- file: `docs/symmetric_probe_derivatives.tex`; old span: 438; kind: LaTeX equation line; fenced: y
- letters and widths: U
- roles: per §4.8: frame U/O/P/Q → D/U/L/R; D^t (:111) → \mathrm d^t (its P^t is the direction and stays); the T3's own \mathbf U_i → \mathbf B_i; the plain B of :158–169 (a bilinear map) stays; the corewise sentence :556–559 per §1.2a
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
the action-jet space, return the variation gradient $\big((\delta\widetilde{\mathbf U}_i),
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-SPD-0445

- file: `docs/symmetric_probe_derivatives.tex`; old span: 445–449; kind: LaTeX equation line; fenced: y
- letters and widths: L
- roles: per §4.8: frame U/O/P/Q → D/U/L/R; D^t (:111) → \mathrm d^t (its P^t is the direction and stays); the T3's own \mathbf U_i → \mathbf B_i; the plain B of :158–169 (a bilinear map) stays; the corewise sentence :556–559 per §1.2a
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\begin{equation}
  \label{eq:lagr}
  \mathcal L = \big\langle \mathbf r,\ \mathbf z \big\rangle
    + \big\langle \widehat\Xi,\ E(\Xi) \big\rangle,
\end{equation}
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-SPD-0452

- file: `docs/symmetric_probe_derivatives.tex`; old span: 452; kind: LaTeX equation line; fenced: y
- letters and widths: L
- roles: per §4.8: frame U/O/P/Q → D/U/L/R; D^t (:111) → \mathrm d^t (its P^t is the direction and stays); the T3's own \mathbf U_i → \mathbf B_i; the plain B of :158–169 (a bilinear map) stays; the corewise sentence :556–559 per §1.2a
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
T4S paper with one change: \emph{every contraction in $\mathcal L$ is replaced by its $\texttt{trs}$
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-SPD-0454

- file: `docs/symmetric_probe_derivatives.tex`; old span: 454; kind: LaTeX equation line; fenced: y
- letters and widths: L
- roles: per §4.8: frame U/O/P/Q → D/U/L/R; D^t (:111) → \mathrm d^t (its P^t is the direction and stays); the T3's own \mathbf U_i → \mathbf B_i; the plain B of :158–169 (a bilinear map) stays; the corewise sentence :556–559 per §1.2a
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
Section~\ref{sec:impl}. At a stationary point the constraints hold ($\partial_{\widehat\Xi}\mathcal L
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-SPD-0456

- file: `docs/symmetric_probe_derivatives.tex`; old span: 456; kind: LaTeX equation line; fenced: y
- letters and widths: L
- roles: per §4.8: frame U/O/P/Q → D/U/L/R; D^t (:111) → \mathrm d^t (its P^t is the direction and stays); the T3's own \mathbf U_i → \mathbf B_i; the plain B of :158–169 (a bilinear map) stays; the corewise sentence :556–559 per §1.2a
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
($\partial_\Xi\mathcal L = 0$), and
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-SPD-0457

- file: `docs/symmetric_probe_derivatives.tex`; old span: 457–461; kind: LaTeX equation line; fenced: y
- letters and widths: L, U
- roles: per §4.8: frame U/O/P/Q → D/U/L/R; D^t (:111) → \mathrm d^t (its P^t is the direction and stays); the T3's own \mathbf U_i → \mathbf B_i; the plain B of :158–169 (a bilinear map) stays; the corewise sentence :556–559 per §1.2a
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\begin{equation}
  \label{eq:grad}
  \big((\delta\widetilde{\mathbf U}_i),(\delta\widetilde{\mathbf G}_i)\big)
  = \partial_{\delta\mathbb V}\, \mathcal L = \big(J^{(s)}\big)^\top(\mathbf r).
\end{equation}
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-SPD-0466

- file: `docs/symmetric_probe_derivatives.tex`; old span: 466; kind: LaTeX equation line; fenced: y
- letters and widths: L
- roles: per §4.8: frame U/O/P/Q → D/U/L/R; D^t (:111) → \mathrm d^t (its P^t is the direction and stays); the T3's own \mathbf U_i → \mathbf B_i; the plain B of :158–169 (a bilinear map) stays; the corewise sentence :556–559 per §1.2a
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
The multiplier equations $\partial_\Xi\mathcal L = 0$ are the adjoint sweeps. Differentiating a forward
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-SPD-0482

- file: `docs/symmetric_probe_derivatives.tex`; old span: 482; kind: LaTeX equation line; fenced: y
- letters and widths: U_i
- roles: per §4.8: frame U/O/P/Q → D/U/L/R; D^t (:111) → \mathrm d^t (its P^t is the direction and stays); the T3's own \mathbf U_i → \mathbf B_i; the plain B of :158–169 (a bilinear map) stays; the corewise sentence :556–559 per §1.2a
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\eta}_i = \mathbf U_i \mathbf r_i$ is the order-diagonal lift.
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-SPD-0485

- file: `docs/symmetric_probe_derivatives.tex`; old span: 485; kind: LaTeX equation line; fenced: y
- letters and widths: L, U_i}
- roles: per §4.8: frame U/O/P/Q → D/U/L/R; D^t (:111) → \mathrm d^t (its P^t is the direction and stays); the T3's own \mathbf U_i → \mathbf B_i; the plain B of :158–169 (a bilinear map) stays; the corewise sentence :556–559 per §1.2a
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
The variation gradients $\partial_{\delta\mathbf G_i}\mathcal L$ and $\partial_{\delta\mathbf U_i}
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-SPD-0500

- file: `docs/symmetric_probe_derivatives.tex`; old span: 500; kind: LaTeX equation line; fenced: y
- letters and widths: U_i
- roles: per §4.8: frame U/O/P/Q → D/U/L/R; D^t (:111) → \mathrm d^t (its P^t is the direction and stays); the T3's own \mathbf U_i → \mathbf B_i; the plain B of :158–169 (a bilinear map) stays; the corewise sentence :556–559 per §1.2a
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
  \item the Tucker core $\delta\mathbf U_i$ has one internal edge (the down-edge to its TT core), so
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-SPD-0502

- file: `docs/symmetric_probe_derivatives.tex`; old span: 502–507; kind: LaTeX equation line; fenced: y
- letters and widths: U
- roles: per §4.8: frame U/O/P/Q → D/U/L/R; D^t (:111) → \mathrm d^t (its P^t is the direction and stays); the T3's own \mathbf U_i → \mathbf B_i; the plain B of :158–169 (a bilinear map) stays; the corewise sentence :556–559 per §1.2a
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
    \[
      \delta\widetilde{\mathbf U}_i
      = \sum_t \boldsymbol\eta_i^{(t)} \otimes \mathbf r_i^{(t)}
      + \sum_u \widehat{\delta\boldsymbol\xi}_i^{(u)} \otimes \mathbf w_i^{(u)},
      \qquad \jet{\mathbf w_i} = (\mathbf w_i, \mathbf p_i, 0, \dots).
    \]
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-SPD-0540

- file: `docs/symmetric_probe_derivatives.tex`; old span: 540; kind: LaTeX equation line; fenced: y
- letters and widths: U
- roles: per §4.8: frame U/O/P/Q → D/U/L/R; D^t (:111) → \mathrm d^t (its P^t is the direction and stays); the T3's own \mathbf U_i → \mathbf B_i; the plain B of :158–169 (a bilinear map) stays; the corewise sentence :556–559 per §1.2a
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\sigma}_i$, $\delta\widetilde{\mathbf U}_i = \widehat{\delta\boldsymbol\xi}_i\otimes\mathbf w_i$ (about
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-SPD-0542

- file: `docs/symmetric_probe_derivatives.tex`; old span: 542–543; kind: LaTeX equation line; fenced: y
- letters and widths: P
- roles: per §4.8: frame U/O/P/Q → D/U/L/R; D^t (:111) → \mathrm d^t (its P^t is the direction and stays); the T3's own \mathbf U_i → \mathbf B_i; the plain B of :158–169 (a bilinear map) stays; the corewise sentence :556–559 per §1.2a
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
e_{\mathrm{index}}$ in the base slot and a general direction $\mathbf P$ (fiber-slice the base projection,
contract $\mathbf P$ at order one); its transpose scatters onto the indexed rows.
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-SPD-0547

- file: `docs/symmetric_probe_derivatives.tex`; old span: 547; kind: LaTeX equation line; fenced: y
- letters and widths: P
- roles: per §4.8: frame U/O/P/Q → D/U/L/R; D^t (:111) → \mathrm d^t (its P^t is the direction and stays); the T3's own \mathbf U_i → \mathbf B_i; the plain B of :158–169 (a bilinear map) stays; the corewise sentence :556–559 per §1.2a
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
the sample stack $W$ (the paired $(\mathbf X,\mathbf P)$ samples, on the inputs), the tangent stack $K$
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-SPD-0559

- file: `docs/symmetric_probe_derivatives.tex`; old span: 559; kind: LaTeX equation line; fenced: y
- letters and widths: O, P, Q, U
- roles: per §4.8: frame U/O/P/Q → D/U/L/R; D^t (:111) → \mathrm d^t (its P^t is the direction and stays); the T3's own \mathbf U_i → \mathbf B_i; the plain B of :158–169 (a bilinear map) stays; the corewise sentence :556–559 per §1.2a
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
($\mathbf P,\mathbf Q,\mathbf O\to\mathbf G_i$, $\mathbf U$ no longer required orthogonal). It is a
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-SPD-0631

- file: `docs/symmetric_probe_derivatives.tex`; old span: 631; kind: LaTeX equation line; fenced: y
- letters and widths: U
- roles: per §4.8: frame U/O/P/Q → D/U/L/R; D^t (:111) → \mathrm d^t (its P^t is the direction and stays); the T3's own \mathbf U_i → \mathbf B_i; the plain B of :158–169 (a bilinear map) stays; the corewise sentence :556–559 per §1.2a
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\delta\widetilde{\mathbf U}_i$: order-summed outer products of the tilde edge variables with the base
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-SPD-0633

- file: `docs/symmetric_probe_derivatives.tex`; old span: 633; kind: LaTeX equation line; fenced: y
- letters and widths: L, R
- roles: per §4.8: frame U/O/P/Q → D/U/L/R; D^t (:111) → \mathrm d^t (its P^t is the direction and stays); the T3's own \mathbf U_i → \mathbf B_i; the plain B of :158–169 (a bilinear map) stays; the corewise sentence :556–559 per §1.2a
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
$(\mathrm{r_L},\mathrm{n},\mathrm{r_R})$, so the intermediate is the largest in the chain --- and it is
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~


### `docs/ttm_t3m_ht_note.tex`

#### TEX-TTHN-0073

- file: `docs/ttm_t3m_ht_note.tex`; old span: 73–77; kind: LaTeX equation line; fenced: y
- letters and widths: U_i
- roles: per §4.8: the Tucker factors ("leaf frames") U_i → B_i; operand labels A, B, C (superscripts) → X, Y, Z; the HT transfer tensors B_t (:217–240) → T_t; the macros \A, \B, \C → \X, \Y, \Z; \C keeps one meaning (eq:t3)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\begin{equation}
\T[x]=\sum_{n_0,\dots,n_{d-1}} \Big(\prod_{i=0}^{d-1} U_i[n_i,x_i]\Big)\,\C[n_0,\dots,n_{d-1}],
\qquad \C=\TT(G),
\label{eq:t3}
\end{equation}
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-TTHN-0078

- file: `docs/ttm_t3m_ht_note.tex`; old span: 78; kind: LaTeX equation line; fenced: y
- letters and widths: U_i
- roles: per §4.8: the Tucker factors ("leaf frames") U_i → B_i; operand labels A, B, C (superscripts) → X, Y, Z; the HT transfer tensors B_t (:217–240) → T_t; the macros \A, \B, \C → \X, \Y, \Z; \C keeps one meaning (eq:t3)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
with \emph{Tucker factors} (leaf frames) $U_i\in\R^{n_i\times N_i}$ and central TT
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-TTHN-0080

- file: `docs/ttm_t3m_ht_note.tex`; old span: 80; kind: LaTeX equation line; fenced: y
- letters and widths: U_i
- roles: per §4.8: the Tucker factors ("leaf frames") U_i → B_i; operand labels A, B, C (superscripts) → X, Y, Z; the HT transfer tensors B_t (:217–240) → T_t; the macros \A, \B, \C → \X, \Y, \Z; \C keeps one meaning (eq:t3)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
Plain TT is the special case $n_i=N_i$, $U_i=I$.
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-TTHN-0084

- file: `docs/ttm_t3m_ht_note.tex`; old span: 84–87; kind: LaTeX equation line; fenced: y
- letters and widths: B, U^A, U^B
- roles: per §4.8: the Tucker factors ("leaf frames") U_i → B_i; operand labels A, B, C (superscripts) → X, Y, Z; the HT transfer tensors B_t (:217–240) → T_t; the macros \A, \B, \C → \X, \Y, \Z; \C keeps one meaning (eq:t3)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\begin{align}
(U^A\krrow U^B)[(a,b),\,x] &= U^A[a,x]\,U^B[b,x] && \text{(row-wise Khatri--Rao, ``COPY'')},\\
(G^A\kron G^B)[(\alpha,\alpha'),(a,b),(\beta,\beta')] &= G^A[\alpha,a,\beta]\,G^B[\alpha',a',\beta'] && \text{(Kronecker)} .
\end{align}
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-TTHN-0092

- file: `docs/ttm_t3m_ht_note.tex`; old span: 92–97; kind: LaTeX equation line; fenced: y
- letters and widths: B_i, B_{i, U^A_i, U^B_i
- roles: per §4.8: the Tucker factors ("leaf frames") U_i → B_i; operand labels A, B, C (superscripts) → X, Y, Z; the HT transfer tensors B_t (:217–240) → T_t; the macros \A, \B, \C → \X, \Y, \Z; \C keeps one meaning (eq:t3)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\begin{equation}
W_i \;=\; U^A_i\krrow U^B_i \in\R^{(n^A_i n^B_i)\times N_i},
\qquad
G^C_i \;=\; G^A_i\kron G^B_i \in\R^{(r^A_i r^B_i)\times(n^A_i n^B_i)\times(r^A_{i+1}r^B_{i+1})}.
\label{eq:prod}
\end{equation}
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-TTHN-0099

- file: `docs/ttm_t3m_ht_note.tex`; old span: 99; kind: LaTeX equation line; fenced: y
- letters and widths: B_i
- roles: per §4.8: the Tucker factors ("leaf frames") U_i → B_i; operand labels A, B, C (superscripts) → X, Y, Z; the HT transfer tensors B_t (:217–240) → T_t; the macros \A, \B, \C → \X, \Y, \Z; \C keeps one meaning (eq:t3)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
$n^A_i n^B_i$ and TT bond $r^A_i r^B_i$. The point of every algorithm below is to
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-TTHN-0114

- file: `docs/ttm_t3m_ht_note.tex`; old span: 114–130; kind: LaTeX equation line; fenced: y
- letters and widths: B, B_i, O
- roles: per §4.8: the Tucker factors ("leaf frames") U_i → B_i; operand labels A, B, C (superscripts) → X, Y, Z; the HT transfer tensors B_t (:217–240) → T_t; the macros \A, \B, \C → \X, \Y, \Z; \C keeps one meaning (eq:t3)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\begin{algorithm}[t]
\caption{\textsc{TTM} --- Hadamard product of two tensor trains (swap form)}
\label{alg:ttm}
\begin{algorithmic}[1]
\Require TT cores $\{G^A_i\}$, $\{G^B_i\}$; per-step tolerance / target bond $\rtil$
\State $H \gets [\,G^A_0,\dots,G^A_{d-1},\ \textsc{reverse}(G^B)\,]$ \Comment{length-$2d$ chain $=\A\kron\B$; reversal makes mode $d{-}1$ adjacent}
\State left-orthogonalize the $A$-half, right-orthogonalize the reversed-$B$ half \Comment{mixed-canonical center at the seam}
\For{$i=d-1$ \textbf{downto} $0$}
  \State move the orthogonality center to the right end of the merged block
  \For{each step bringing the mode-$i$ $B$-core toward its $A$-partner}
    \State \textsc{swap} the adjacent pair by a \emph{centered} truncated SVD \Comment{bond stays $O(\rtil)$}
  \EndFor
  \State \textsc{merge} the adjacent matching pair with the copy tensor $\delta$ (diagonal in $x_i$); single bonds in/out
\EndFor
\State \Return the length-$d$ chain $\{G^C_i\}$
\end{algorithmic}
\end{algorithm}
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-TTHN-0133

- file: `docs/ttm_t3m_ht_note.tex`; old span: 133; kind: LaTeX equation line; fenced: y
- letters and widths: B
- roles: per §4.8: the Tucker factors ("leaf frames") U_i → B_i; operand labels A, B, C (superscripts) → X, Y, Z; the HT transfer tensors B_t (:217–240) → T_t; the macros \A, \B, \C → \X, \Y, \Z; \C keeps one meaning (eq:t3)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
``bubbled past'' by later $B$-cores. Because every swap and merge is performed in
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-TTHN-0149

- file: `docs/ttm_t3m_ht_note.tex`; old span: 149; kind: LaTeX equation line; fenced: y
- letters and widths: B_i, U^A_i, U^B_i
- roles: per §4.8: the Tucker factors ("leaf frames") U_i → B_i; operand labels A, B, C (superscripts) → X, Y, Z; the HT transfer tensors B_t (:217–240) → T_t; the macros \A, \B, \C → \X, \Y, \Z; \C keeps one meaning (eq:t3)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
is the pair $(n^A_i,n^B_i)$ carrying the Tucker factor $W_i=U^A_i\krrow U^B_i$ from
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-TTHN-0151

- file: `docs/ttm_t3m_ht_note.tex`; old span: 151–153; kind: LaTeX equation line; fenced: y
- letters and widths: B_i,
- roles: per §4.8: the Tucker factors ("leaf frames") U_i → B_i; operand labels A, B, C (superscripts) → X, Y, Z; the HT transfer tensors B_t (:217–240) → T_t; the macros \A, \B, \C → \X, \Y, \Z; \C keeps one meaning (eq:t3)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\begin{equation}
\rank(W_i)\le \min(n^A_i n^B_i,\;N_i),\qquad\text{truncate to } \ntil_i,
\end{equation}
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-TTHN-0158

- file: `docs/ttm_t3m_ht_note.tex`; old span: 158–178; kind: LaTeX equation line; fenced: y
- letters and widths: B, U^A,G, U^A_i, U^B,G, U^B_i, U^C_i,G, U_i,
- roles: per §4.8: the Tucker factors ("leaf frames") U_i → B_i; operand labels A, B, C (superscripts) → X, Y, Z; the HT transfer tensors B_t (:217–240) → T_t; the macros \A, \B, \C → \X, \Y, \Z; \C keeps one meaning (eq:t3)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\begin{algorithm}[t]
\caption{\textsc{T3M-swap} --- Hadamard product of two Tucker tensor trains}
\label{alg:t3m}
\begin{algorithmic}[1]
\Require T3s $(U^A,G^A)$, $(U^B,G^B)$; targets $\ntil,\rtil$ (and/or $\mathrm{rtol},\mathrm{atol}$); oversample $k\ge1$
\State down-orthogonalize the Tucker factors of each input \Comment{HT: ``orthogonalize leaves toward the root''}
\State $H \gets [\,G^A_0,\dots,G^A_{d-1},\ \textsc{reverse}(G^B)\,]$ with each $G_i$ carrying $(U_i,\text{mode }i)$
\State left-orthogonalize the $A$-half, right-orthogonalize the reversed-$B$ half \Comment{center at the seam}
\For{$i=d-1$ \textbf{downto} $0$}
  \State move the center to the right end of the merged block
  \For{each step bringing the mode-$i$ $B$-core to its $A$-partner}
    \State centered truncated-SVD \textsc{swap}, intermediate bond capped at $k\,\rtil$ (or tol $/k$) \Comment{memory only}
  \EndFor
  \State \textsc{merge}: $W_i\gets U^A_i\krrow U^B_i$; centered weighted-SVD compress to $\le k\,\ntil$; Tucker factors/modes ride along
\EndFor
\If{$k>1$ \textbf{or} a non-uniform per-position bond cap was requested}
  \State $\{(U^C_i,G^C_i)\}\gets$ \textsc{T3-SVD-round}$(\{(U^C_i,G^C_i)\};\ \ntil,\rtil,\mathrm{rtol},\mathrm{atol})$ \Comment{the decisive truncation}
\EndIf
\State \Return $\{(U^C_i,G^C_i)\}$
\end{algorithmic}
\end{algorithm}
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-TTHN-0186

- file: `docs/ttm_t3m_ht_note.tex`; old span: 186; kind: LaTeX equation line; fenced: y
- letters and widths: U^A_i, U^B_i
- roles: per §4.8: the Tucker factors ("leaf frames") U_i → B_i; operand labels A, B, C (superscripts) → X, Y, Z; the HT transfer tensors B_t (:217–240) → T_t; the macros \A, \B, \C → \X, \Y, \Z; \C keeps one meaning (eq:t3)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
the Khatri--Rao compression $W_i\!=\!U^A_i\krrow U^B_i$ \textbf{couples} the two
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-TTHN-0217

- file: `docs/ttm_t3m_ht_note.tex`; old span: 217–218; kind: LaTeX equation line; fenced: y
- letters and widths: B_t, U_i
- roles: per §4.8: the Tucker factors ("leaf frames") U_i → B_i; operand labels A, B, C (superscripts) → X, Y, Z; the HT transfer tensors B_t (:217–240) → T_t; the macros \A, \B, \C → \X, \Y, \Z; \C keeps one meaning (eq:t3)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
An HT tensor is a \emph{dimension tree} with leaf frames $U_i\in\R^{n_i\times N_i}$
and transfer tensors $B_t\in\R^{n_{t_1}\times n_{t_2}\times n_t}$ at internal nodes
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-TTHN-0222

- file: `docs/ttm_t3m_ht_note.tex`; old span: 222–230; kind: LaTeX table or figure; fenced: y
- letters and widths: U_i
- roles: per §4.8: the Tucker factors ("leaf frames") U_i → B_i; operand labels A, B, C (superscripts) → X, Y, Z; the HT transfer tensors B_t (:217–240) → T_t; the macros \A, \B, \C → \X, \Y, \Z; \C keeps one meaning (eq:t3)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\begin{tabular}{ll}
\toprule
format & HT realization\\
\midrule
Tensor train & linear (caterpillar) tree, \textbf{trivial} leaf frames ($U_i=I$)\\
Tucker & star (one root transfer tensor, $d$ leaf frames)\\
Tucker tensor train (T3) & linear tree, \textbf{nontrivial} leaf frames\\
\bottomrule
\end{tabular}
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TEX-TTHN-0237

- file: `docs/ttm_t3m_ht_note.tex`; old span: 237–241; kind: LaTeX equation line; fenced: y
- letters and widths: B^A_t, B^B_t,, B^C_t, U^A_i, U^B_i,
- roles: per §4.8: the Tucker factors ("leaf frames") U_i → B_i; operand labels A, B, C (superscripts) → X, Y, Z; the HT transfer tensors B_t (:217–240) → T_t; the macros \A, \B, \C → \X, \Y, \Z; \C keeps one meaning (eq:t3)
- re-laid out: n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)
- new == old possible: possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
\[
\text{leaf frames } \;W_i = U^A_i\krrow U^B_i,
\qquad
\text{transfer tensors } \; B^C_t = B^A_t\kron B^B_t,
\]
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~


### `docs/user_guide.rst`

#### DOC-UG-0125

- file: `docs/user_guide.rst`; old span: 125–133; kind: ascii diagram; fenced: y
- letters and widths: letters: H0 H1 H2 L0 L1 O0 O1 O2 R1 R2 U0 U1 U2 V0 V1 V2
- roles: U0–U2 the Tucker store (→ D, both rows); O0–O2 the TT-shaped store (→ U); H, V the variations; leading TABs kept (K10 on raw text; sample batch 3 edit 8)
- re-laid out: n expected: labels keep their length (U0→D0, nU0→nD0, nO1→nU1); K10 in place
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
	      1--H0--R1--R2--1   1--L0--H1--R2--1   1--L0--L1--H2--1
	         |   |   |          |   |   |          |   |   |
	v  =     U0  U1  U2    +    U0  U1  U2    +    U0  U1  U2
	         |   |   |          |   |   |          |   |   |

	      1--O0--R1--R2--1   1--L0--O1--R2--1   1--L0--L1--O2--1
	         |   |   |          |   |   |          |   |   |
	   +     V0  U1  U2    +    U0  V1  U2    +    U0  U1  V2
	         |   |   |          |   |   |          |   |   |
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### DOC-UG-0139

- file: `docs/user_guide.rst`; old span: 139–142; kind: shape listing; fenced: y
- letters and widths: letters: L(d-1) L0 O(d-1) O0 R(d-1) R0 U(d-1) U0; store words: down down_tt_cores left_tt_cores right_tt_cores up_tucker_cores
- roles: reordered to (down, up, left, right) with down-orthogonal / up-orthogonal; `:142`'s "down-orthogonal" on the TT-shaped core was backwards (sample batch 3 edit 8)
- re-laid out: y: reordered and the names/adjectives grow (K10b)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
  - up_tucker_cores   = (U0,...,U(d-1)), orthogonal
  - left_tt_cores     = (L0,...,L(d-1)), left-orthogonal
  - right_tt_cores    = (R0,...,R(d-1)), right-orthogonal
  - down_tt_cores     = (O0,...,O(d-1)), down-orthogonal
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### DOC-UG-0215

- file: `docs/user_guide.rst`; old span: 215–218; kind: index equation; fenced: y
- letters and widths: letters: U_i
- roles: frame Tucker core `U_i` → `D_i`; variation core `dU_i` → `dD_i` (D-12) or `V_i`; shapes use the T3's n_i (stay)
- re-laid out: n expected unless a label grows; K10 in place
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
    frame Tucker core  U_i   (T3Frame)        C            (2,)      + (n_i, N_i)
    variation core    dU_i  (T3Variations)   K + C        (3, 2)    + (n_i, N_i)
    probe vector      w_i   (ww)             W            (4,)      + (N_i,)
    forward probe     z_i   (tangent.probe)  W + K + C     (4, 3, 2) + (N_i,)
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~


### `docs/weighting.md`

#### DOC-W-0069

- file: `docs/weighting.md`; old span: 69–74; kind: markdown table; fenced: y
- letters and widths: letters: H V; widths: nD nU rL rR; store words: down up
- roles: the weight families and the legs they act on: `up` | H's Tucker (nU) leg → `down` | H's Tucker (nD) leg; `down` | V's complement (nD) leg → `up` | V's complement (nU) leg; rows in physical order (down, up, left, right)
- re-laid out: n/a: a markdown table; K12 applies
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
| family | weights | length |
|---|---|---|
| `up` | `H`'s Tucker (`nU`) leg | `d` |
| `down` | `V`'s complement (`nD`) leg | `d` |
| `left` | `H`'s left bond (`rL`) | `d` |
| `right` | `H`'s right bond (`rR`) | `d` |
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~


### `t3toolbox/backend/apply.py`

#### A-0465

- file: `t3toolbox/backend/apply.py`; old span: 465–467; kind: index equation; fenced: y
- letters and widths: letters: O_k; widths: nO nU rL rR; store words: down
- roles: O_k the TT-shaped store (→ U_k); "over the down mode nO" → up mode nU; (rL, nU, rR) → (rL, nD, rR); dU-tilde_k (§1.2a: `_apply_transpose_adjoint` takes D forms) → dD-tilde_k over (nU, N)
- re-laid out: n expected unless a label grows; K10 in place
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
        dxi-hat_k  = mu-hat_{k-1} . O_k . sigma-hat_k                 # over the down mode nO
        dG-tilde_k = mu-hat_{k-1} (x) xi-hat_k (x) sigma-hat_k        # over (rL, nU, rR)
        dU-tilde_k = dxi-hat_k (x) w_k                               # over (nO, N)
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~


### `t3toolbox/backend/fv_operations.py`

#### FO-0201

- file: `t3toolbox/backend/fv_operations.py`; old span: 201–202; kind: shape listing; fenced: y
- letters and widths: letters: D_i L_i R_i U_i; store words: down up_tucker
- roles: "up_tucker U_i, outer/down D_i" → down_tucker D_i, up U_i (this docstring's D is the TT-shaped store, inv 2 C2); L_i, R_i stay
- re-laid out: y expected: the store identifiers change length (up_tucker_* → down_tucker_*, down_tt_* → up_tt_*); K10b
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
    - up_tucker U_i (all i), outer/down D_i (all i),
    - left L_i (i=0..d-2), right R_i (i=1..d-1).
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~


### `t3toolbox/backend/padded_frames.py`

#### PF-0041

- file: `t3toolbox/backend/padded_frames.py`; old span: 41; kind: shape listing; fenced: y
- letters and widths: widths: nDi nUi rL(i+1) rLi rR(i+1) rRi; store words: down up
- roles: `(up, down, left, right)` → `(down, up, left, right)` (tier D); widths swap: C+(nUi, Ni) → C+(nDi, Ni), C+(rLi, nDi, rR(i+1)) → nUi, left/right nUi → nDi
- re-laid out: n expected unless a label grows; K10 in place
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
#   (up, down, left, right), each len=d, elm_shapes C+(nUi, Ni), C+(rLi, nDi, rR(i+1)), C+(rLi, nUi, rL(i+1)), C+(rRi, nUi, rR(i+1))
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### PF-0043

- file: `t3toolbox/backend/padded_frames.py`; old span: 43; kind: shape listing; fenced: y
- letters and widths: widths: nDi nUi rLi rR(i+1)
- roles: variations: C+(nDi, Ni) → C+(nUi, Ni); C+(rLi, nUi, rR(i+1)) → nDi
- re-laid out: n expected unless a label grows; K10 in place
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
#   (tucker_variations, tt_variations), each len=d, elm_shapes C+(nDi, Ni), C+(rLi, nUi, rR(i+1))
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~


### `t3toolbox/backend/probing.py`

#### P-0640

- file: `t3toolbox/backend/probing.py`; old span: 640–643, columns 40–end; kind: field-comment relation; fenced: y
- letters and widths: letters: O O_ixj O_iyj P P_iax P_iay Q Q_xaj Q_yaj U U_xo U_yo; widths: nO nU rL rR; store words: down_tt_cores left_tt_cores right_tt_cores up_tucker_cores
- roles: tuple-element comments: up_tucker_cores U_xo → down_tucker_cores D_xo; down_tt_cores O_ixj → up_tt_cores U_ixj; P → L, Q → R; nU → nD, nO → nU
- re-laid out: y: the store labels change length (K10b on the comment span)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
# up_tucker_cores. len=d. U_xo U_yo   = I_xy, U.shape = (nU, N)
# down_tt_cores.   len=d. O_ixj O_iyj = I_xy  O.shape = (rL, nO, rR)
# left_tt_cores.   len=d. P_iax P_iay = I_xy, P.shape = (rL, nU, rR)
# right_tt_cores.  len=d. Q_xaj Q_yaj = I_xy  Q.shape = (rL, nU, rR)
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### P-0646

- file: `t3toolbox/backend/probing.py`; old span: 646–649, columns 26–end; kind: field-comment relation; fenced: y
- letters and widths: widths: nO nU rL rR; store words: down down_tt_supercore left_tt_supercore right_tt_supercore up up_tucker_supercore
- roles: uniform supercore labels (up_tucker_supercore → down_tucker_supercore, down_tt_supercore → up_tt_supercore); widths nU → nD, nO → nU; adjectives "up orthogonal elements" / "down orthogonal elements" are backwards today (§4.3 probing.py:646–647)
- re-laid out: y: labels change length (K10b on the comment span)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
# up_tucker_supercore. shape=(d, nU, N),      up orthogonal elements
# down_tt_supercore.   shape=(d, rL, nO, rR), down orthogonal elements
# left_tt_supercore.   shape=(d, rL, nU, rR), left orthogonal elements
# right_tt_supercore.  shape=(d, rL, nU, rR), right orthogonal elements
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### P-0651

- file: `t3toolbox/backend/probing.py`; old span: 651, columns 11–end; kind: index equation; fenced: y
- letters and widths: letters: O P Q U; store words: down up
- roles: `frame order = T3Frame.data = (up, down, left, right) = (U, O, P, Q)` → `(down, up, left, right) = (D, U, L, R)` (tier D's patterns, applied by hand inside the fence)
- re-laid out: n expected (same lengths)
- new == old possible: possible: record the reason at R4 if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
# frame order = T3Frame.data = (up, down, left, right) = (U, O, P, Q)
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### P-1215

- file: `t3toolbox/backend/probing.py`; old span: 1215–1218, columns 40–end; kind: field-comment relation; fenced: y
- letters and widths: letters: O O_ixj O_iyj P P_iax P_iay Q Q_xaj Q_yaj U U_xo U_yo; widths: nO nU rL rR; store words: down_tt_cores left_tt_cores right_tt_cores up_tucker_cores
- roles: as P-0640
- re-laid out: y: the store labels change length (K10b on the comment span)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
# up_tucker_cores. len=d. U_xo U_yo   = I_xy, U.shape = (nU, N)
# down_tt_cores.   len=d. O_ixj O_iyj = I_xy  O.shape = (rL, nO, rR)
# left_tt_cores.   len=d. P_iax P_iay = I_xy, P.shape = (rL, nU, rR)
# right_tt_cores.  len=d. Q_xaj Q_yaj = I_xy  Q.shape = (rL, nU, rR)
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### P-1221

- file: `t3toolbox/backend/probing.py`; old span: 1221–1224, columns 26–end; kind: field-comment relation; fenced: y
- letters and widths: widths: nO nU rL rR; store words: down down_tt_supercore left_tt_supercore right_tt_supercore up up_tucker_supercore
- roles: as P-0646 (§4.3 probing.py:1221–1222)
- re-laid out: y: labels change length (K10b on the comment span)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
# up_tucker_supercore. shape=(d, nU, N),      up orthogonal elements
# down_tt_supercore.   shape=(d, rL, nO, rR), down orthogonal elements
# left_tt_supercore.   shape=(d, rL, nU, rR), left orthogonal elements
# right_tt_supercore.  shape=(d, rL, nU, rR), right orthogonal elements
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### P-1226

- file: `t3toolbox/backend/probing.py`; old span: 1226, columns 11–end; kind: index equation; fenced: y
- letters and widths: letters: O P Q U; store words: down up
- roles: as P-0651
- re-laid out: n expected (same lengths)
- new == old possible: possible: record the reason at R4 if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
# frame order = T3Frame.data = (up, down, left, right) = (U, O, P, Q)
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~


### `t3toolbox/backend/ranks.py`

#### RA-0644

- file: `t3toolbox/backend/ranks.py`; old span: 644–647; kind: index equation; fenced: y
- letters and widths: store words: down down_i left_i right_i up_i
- roles: store-word index variables: up_i (the Tucker store's rank) → down_i, down_i (the TT-shaped store's) → up_i; left_i, right_i stay; the recurrence's step order stays
- re-laid out: n expected unless a label grows; K10 in place
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
    1. Tucker down-orthogonalization: ``up_i = min(n_i, N_i)``;
    2. a LEFT sweep on the original TT bonds: ``left_{i+1} = min(left_i * up_i, r_{i+1})``;
    3. a RIGHT sweep over the left-orthogonal chain: ``right_i = min(left_i, up_i * right_{i+1})``;
    4. the down cores: ``down_i = min(up_i, left_i * right_{i+1})``.
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~


### `t3toolbox/backend/t3_orthogonalization.py`

#### T3OR-0044

- file: `t3toolbox/backend/t3_orthogonalization.py`; old span: 44–46; kind: index equation; fenced: y
- letters and widths: letters: B B_i G G_i
- re-laid out: n expected unless a label grows; K10 in place
- new == old possible: yes expected: B and G are a T3's own cores (§3.3: a B already written stays); no store letter or width
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
    - Tucker B_i, all i:                 ``einsum('...io,...jo->...ij', B, B) = I``
    - TT G_i, left  (i = 0..d-2):        ``einsum('...aib,...aic->...bc', G, G) = I``
    - TT G_i, right (i = 1..d-1):        ``einsum('...aib,...cib->...ac', G, G) = I``
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~


### `t3toolbox/backend/tv_operations.py`

#### TVOP-0108

- file: `t3toolbox/backend/tv_operations.py`; old span: 108; kind: index equation; fenced: y
- letters and widths: letters: H_i L_i U_i V_i
- roles: ``U_i V_i^T = 0`` → D_i; L_i, H_i stay
- re-laid out: n expected unless a label grows; K10 in place
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
    ``U_i V_i^T = 0`` (all i) and ``einsum('...abi,...abj->...ij', L_i, H_i) = 0`` (i = 0..d-2).
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~


### `t3toolbox/backend/ufv_operations.py`

#### UFOP-0083

- file: `t3toolbox/backend/ufv_operations.py`; old span: 83; kind: index equation; fenced: y
- letters and widths: widths: nD nU rL rR; store words: down up
- roles: `nU = up.shape[-2]  nD = down.shape[-2] ... N = up.shape[-1]`: the width names follow their stores (nD = down.shape[-2] for the Tucker supercore, nU = up.shape[-2], N = down.shape[-1]); the rewriter keeps or reorders by physical order
- re-laid out: n expected unless a label grows; K10 in place
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
        nU = up.shape[-2]    nD = down.shape[-2]    rL = left.shape[-1]    rR = right.shape[-1]    N = up.shape[-1]
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~


### `t3toolbox/backend/uniform_padded_frames.py`

#### UPF-0047

- file: `t3toolbox/backend/uniform_padded_frames.py`; old span: 47–48; kind: shape listing; fenced: y
- letters and widths: widths: nD nU rL rR; store words: down up
- roles: the store words up/down (free prose, tier E) and widths swap together: "up (d,)+C+(nU, N), down (d,)+C+(rL, nD, rR)" → "down (d,)+C+(nD, N), up (d,)+C+(rL, nU, rR)"; left/right middle legs nU → nD; the mask line likewise; "block layout: nU = nD" stays true either way
- re-laid out: n expected unless a label grows; K10 in place
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
#   up (d,)+C+(nU, N), down (d,)+C+(rL, nD, rR), left (d,)+C+(rL, nU, rL), right (d,)+C+(rR, nU, rR), shape (len=d),
#   masks HOST bool (up (d,)+C+(nU,), down (d,)+C+(nD,), left (d+1,)+C+(rL,), right (d+1,)+C+(rR,)); block layout: nU = nD, rL = rR
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### UPF-0050

- file: `t3toolbox/backend/uniform_padded_frames.py`; old span: 50; kind: shape listing; fenced: y
- letters and widths: letters: H V; widths: nD nU rL rR
- roles: V (d,)+C+(nD, N) → nU; H (d,)+C+(rL, nU, rR) → nD
- re-laid out: n expected unless a label grows; K10 in place
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
#   V (d,)+C+(nD, N), H (d,)+C+(rL, nU, rR), shape, variation masks; only V, H are read (masked with the frame's own masks)
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~


### `t3toolbox/backend/utv_operations.py`

#### UTVOPE-0314

- file: `t3toolbox/backend/utv_operations.py`; old span: 314–315; kind: index equation; fenced: y
- letters and widths: letters: L P Q R U; store words: down up
- roles: [U | dU] = [up | down] → [D | dD] = [down | up]; [Q | P] = [R | L] → [R | L] (T4S's [Q | P]) (§1.5 item 3's rows for utv_operations.py:315–322)
- re-laid out: n expected unless a label grows; K10 in place
- new == old possible: possible: record the reason at R4 if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
    # ---- Doubled rank masks (HOST numpy; carry K+C from the variation masks). The order respects the
    # block structure (T4S (49)-(52)): Tucker axis [U | dU] = [up | down]; the TT bond is [Q | P] = [R | L], so
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### UTVOPE-0538

- file: `t3toolbox/backend/utv_operations.py`; old span: 538–539; kind: index equation; fenced: y
- letters and widths: letters: L P U; store words: up
- roles: ``U^T dU = 0`` (the frame's Tucker store and its variation) → ``D^T dD = 0`` (D-12); "up-core" → down core; ``(P^L)^T dG^L = 0`` → L (D-7)
- re-laid out: n expected unless a label grows; K10 in place
- new == old possible: possible: record the reason at R4 if nothing changes
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
    Each Tucker variation must be orthogonal to its up-core (``U^T dU = 0``) and each left-interior TT
    variation orthogonal to its left-core (``(P^L)^T dG^L = 0``). Mask-once, then the gauge grams vectorized
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~


### `t3toolbox/frame_variations_format.py`

#### FVF-0058a

- file: `t3toolbox/frame_variations_format.py`; old span: 58–61, columns 8–32; kind: ascii diagram; fenced: y
- letters and widths: letters: H0 R1 R2 U0 U1 U2
- roles: U0–U2 the frame's Tucker store (→ D); L, R left/right (stay); (H) the TT variation
- re-laid out: n expected: labels keep their length (U0→D0, nU0→nD0, nO1→nU1); K10 in place
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
1--(H0)--R1---R2---1    
    |    |    |         
    U0   U1   U2        
    |    |    |         
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### FVF-0058b

- file: `t3toolbox/frame_variations_format.py`; old span: 58–61, columns 32–56; kind: ascii diagram; fenced: y
- letters and widths: letters: H1 L0 R2 U0 U1 U2
- roles: U0–U2 the frame's Tucker store (→ D); L, R left/right (stay); (H) the TT variation
- re-laid out: n expected: labels keep their length (U0→D0, nU0→nD0, nO1→nU1); K10 in place
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
1---L0--(H1)--R2---1    
    |    |    |         
    U0   U1   U2        
    |    |    |         
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### FVF-0058c

- file: `t3toolbox/frame_variations_format.py`; old span: 58–61, columns 56–end; kind: ascii diagram; fenced: y
- letters and widths: letters: H2 L0 L1 U0 U1 U2
- roles: U0–U2 the frame's Tucker store (→ D); L, R left/right (stay); (H) the TT variation
- re-laid out: n expected: labels keep their length (U0→D0, nU0→nD0, nO1→nU1); K10 in place
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
1---L0---L1--(H2)--1
    |    |    |
    U0   U1   U2
    |    |    |
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### FVF-0063a

- file: `t3toolbox/frame_variations_format.py`; old span: 63–66, columns 8–32; kind: ascii diagram; fenced: y
- letters and widths: letters: D0 R1 R2 U1 U2 V0
- roles: D0–D2 the TT-shaped store (this docstring's own D, → U); U the Tucker store (→ D); (V) the Tucker variation
- re-laid out: n expected: labels keep their length (U0→D0, nU0→nD0, nO1→nU1); K10 in place
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
1---D0---R1---R2---1    
    |    |    |         
   (V0)  U1   U2        
    |    |    |         
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### FVF-0063b

- file: `t3toolbox/frame_variations_format.py`; old span: 63–66, columns 32–56; kind: ascii diagram; fenced: y
- letters and widths: letters: D1 L0 R2 U0 U2 V1
- roles: D0–D2 the TT-shaped store (this docstring's own D, → U); U the Tucker store (→ D); (V) the Tucker variation
- re-laid out: n expected: labels keep their length (U0→D0, nU0→nD0, nO1→nU1); K10 in place
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
1---L0---D1---R2---1    
    |    |    |         
    U0  (V1)  U2        
    |    |    |         
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### FVF-0063c

- file: `t3toolbox/frame_variations_format.py`; old span: 63–66, columns 56–end; kind: ascii diagram; fenced: y
- letters and widths: letters: D2 L0 L1 U0 U1 V2
- roles: D0–D2 the TT-shaped store (this docstring's own D, → U); U the Tucker store (→ D); (V) the Tucker variation
- re-laid out: n expected: labels keep their length (U0→D0, nU0→nD0, nO1→nU1); K10 in place
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
1---L0---L1---D2---1
    |    |    |
    U0   U1  (V2)
    |    |    |
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### FVF-0071

- file: `t3toolbox/frame_variations_format.py`; old span: 71–74; kind: shape listing; fenced: y
- letters and widths: letters: D(d-1) D0 L(d-1) L0 R(d-1) R0 U(d-1) U0; widths: nDi nUi rL(i+1) rLi rR(i+1) rRi; store words: down_tt_cores left_tt_cores right_tt_cores up_tucker_cores
- roles: up_tucker_cores (U, nUi) → down_tucker_cores (D, nDi); down_tt_cores (D, nDi) → up_tt_cores (U, nUi); left/right write a generic `ni` (D-8 makes their middle leg nD: the rewriter decides whether to keep `ni`); re-laid out
- re-laid out: y expected: the store identifiers change length (up_tucker_* → down_tucker_*, down_tt_* → up_tt_*); K10b
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
        - up_tucker_cores   = (U0, ..., U(d-1)), elm_shape=(nUi, Ni)
        - down_tt_cores     = (D0, ..., D(d-1)), elm_shape=(rLi, nDi, rR(i+1))
        - left_tt_cores     = (L0, ..., L(d-1)), elm_shape=(rLi, ni, rL(i+1))
        - right_tt_cores    = (R0, ..., R(d-1)), elm_shape=(rRi, ni, rR(i+1))
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### FVF-0077

- file: `t3toolbox/frame_variations_format.py`; old span: 77–78; kind: shape listing; fenced: y
- letters and widths: letters: H(d-1) H0 V(d-1) V0; widths: nDi nUi rLi rRi
- roles: tucker_variations rows nDi → nUi; tt_variations middle leg nUi → nDi (D-8)
- re-laid out: n expected unless a label grows; K10 in place
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
        - tucker_variations = (V0, ..., V(d-1)), elm_shape=(nDi, Ni)
        - tt_variations     = (H0, ..., H(d-1)), elm_shape=(rLi, nUi, rRi)
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### FVF-0088

- file: `t3toolbox/frame_variations_format.py`; old span: 88–96; kind: ascii diagram; fenced: y
- letters and widths: letters: H1 L0 R(d-1) U0 U1 Ud; widths: nU(d-1) nU0 nU1 rL0 rL1 rR(d-1) rR2 rRd
- roles: U0, U1, Ud the Tucker store (→ D0, D1, Dd; `Ud` is the known typo for U(d-1), glossary E17); vertical widths nU → nD; rL, rR, N, (H1) stay (sample edit 5)
- re-laid out: n expected: labels keep their length (U0→D0, nU0→nD0, nO1→nU1); K10 in place
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
           rL0       rL1       rR2      rR(d-1)         rRd
        1 ------ L0 ----- (H1) ----- ... ------ R(d-1) ------ 1
                 |         |                    |
                 | nU0     | nU1                | nU(d-1)
                 |         |                    |
                 U0        U1                   Ud
                 |         |                    |
                 | N0      | N1                 | N(d-1)
                 |         |                    |
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### FVF-0100

- file: `t3toolbox/frame_variations_format.py`; old span: 100–108; kind: ascii diagram; fenced: y
- letters and widths: letters: D1 L0 R(d-1) U0 Ud V1; widths: nO1 nU(d-1) nU0 rL0 rL1 rR(d-1) rR2 rRd
- roles: D1 the TT-shaped store (→ U1); nO1 → nU1; nU → nD; U0, Ud → D0, Dd; (V1) stays (sample edit 5)
- re-laid out: n expected: labels keep their length (U0→D0, nU0→nD0, nO1→nU1); K10 in place
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
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

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### FVF-0117

- file: `t3toolbox/frame_variations_format.py`; old span: 117–120; kind: index equation; fenced: y
- letters and widths: letters: L(d-1) L0 L_abi L_abj O(d-1) O0 O_aib O_ajb R(d-1) R0 R_ibc R_jbc U(d-1) U0 U_ia U_ja; store words: down_tt_cores left_tt_cores right_tt_cores up_tucker_cores
- roles: U_ia (Tucker store) → D_ia; O_aib (TT-shaped store) → U_aib; L, R stay; adjectives (orthogonal → down-orthogonal, outer-orthogonal → up-orthogonal) and the missing colon after right-orthogonal (§4A d); relations re-derived (sample edit 6)
- re-laid out: y: the store names grow and the adjectives change (K10b; sample edit 6)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
        - up_tucker_cores   = (U0,...,U(d-1)), orthogonal:       U_ia U_ja = delta_ij
        - down_tt_cores     = (O0,...,O(d-1)), outer-orthogonal  O_aib O_ajb = delta_ij
        - left_tt_cores     = (L0,...,L(d-1)), left-orthogonal:  L_abi L_abj = delta_ij
        - right_tt_cores    = (R0,...,R(d-1)), right-orthogonal  R_ibc R_jbc = delta_ij
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### FVF-0123

- file: `t3toolbox/frame_variations_format.py`; old span: 123–124; kind: index equation; fenced: y
- letters and widths: letters: H H_abj L_abi U_ia V V_ja
- roles: gauge U_ia V_ja = 0 → D_ia V_ja = 0; L_abi H_abj stays
- re-laid out: n expected unless a label grows; K10 in place
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
        - U_ia V_ja = 0    (all V)
        - L_abi H_abj = 0  (all but the last H)
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### FVF-0156

- file: `t3toolbox/frame_variations_format.py`; old span: 156–159, columns 48–end; kind: field-comment relation; fenced: y
- letters and widths: letters: B_xo B_yo Bi P_iax P_iay Pi Q_xaj Q_yaj Qi R_ixj R_iyj Ri; widths: nDi nUi rL(i+1) rLi rR(i+1) rRi
- roles: B_xo (Tucker store) → D_xo; R_ixj (TT store, clashes with right) → U_ixj; P → L, Q → R; widths nUi ↔ nDi; the model is sample edit 1's approved AFTER
- re-laid out: n: in place after §4.10's realignment of the field names (K10 on the comment span; sample edit 1)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
# len=d. B_xo B_yo   = I_xy, Bi.shape = stack_shape+(nUi, Ni)
# len=d. R_ixj R_iyj = I_xy  Ri.shape = stack_shape+(rLi, nDi, rR(i+1))
# len=d. P_iax P_iay = I_xy, Pi.shape = stack_shape+(rLi, nUi, rL(i+1))
# len=d. Q_xaj Q_yaj = I_xy  Qi.shape = stack_shape+(rRi, nUi, rR(i+1))
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### FVF-0217

- file: `t3toolbox/frame_variations_format.py`; old span: 217–220; kind: ascii diagram; fenced: y
- letters and widths: letters: L0 R2 R3 U0 U1 U2 U3
- roles: U0–U3 the Tucker store (→ D); ( ) the hole; L, R stay
- re-laid out: n expected: labels keep their length (U0→D0, nU0→nD0, nO1→nU1); K10 in place
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
            1 -- L0 -- ( ) -- R2 -- R3 -- 1
                 |      |      |      |
                 U0     U1     U2     U3
                 |      |      |      |
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### FVF-0222

- file: `t3toolbox/frame_variations_format.py`; old span: 222–225; kind: ascii diagram; fenced: y
- letters and widths: letters: L0 L1 O2 R3 U0 U1 U3
- roles: O2 the TT-shaped store (→ U2); U0, U1, U3 → D; ( ) the hole
- re-laid out: n expected: labels keep their length (U0→D0, nU0→nD0, nO1→nU1); K10 in place
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
            1 -- L0 -- L1 -- O2 -- R3 -- 1
                 |     |     |     |
                 U0    U1    ( )   U3
                 |     |     |     |
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### FVF-0354

- file: `t3toolbox/frame_variations_format.py`; old span: 354–357; kind: index equation; fenced: y
- letters and widths: letters: D D_i L L_i R R_i U U_i; store words: down up_tucker
- roles: "up_tucker U_i" → down_tucker D_i; "down/outer D_i" → up U_i; einsum subscripts byte-identical (§1.5 item 6), only the operand letters change
- re-laid out: n expected unless a label grows; K10 in place
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
            - up_tucker U_i (all i):    ``einsum('...io,...jo->...ij', U, U) = I``
            - down/outer D_i (all i):   ``einsum('...iaj,...ibj->...ab', D, D) = I``
            - left L_i (i = 0..d-2):    ``einsum('...iaj,...iak->...jk', L, L) = I``
            - right R_i (i = 1..d-1):   ``einsum('...iaj,...kaj->...ik', R, R) = I``
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### FVF-0784

- file: `t3toolbox/frame_variations_format.py`; old span: 784–785; kind: shape listing; fenced: y
- letters and widths: letters: H(d-1) H0 V(d-1) V0; widths: nDi nUi rLi rRi
- roles: tucker_variations rows nDi → nUi; tt_variations middle leg nUi → nDi
- re-laid out: n expected unless a label grows; K10 in place
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
        - tucker_variations    = (V0, ..., V(d-1)), elm_shape=stack_shape+(nDi, Ni)
        - tt_variations        = (H0, ..., H(d-1)), elm_shape=stack_shape+(rLi, nUi, rRi)
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### FVF-0845

- file: `t3toolbox/frame_variations_format.py`; old span: 845–848; kind: ascii diagram; fenced: y
- letters and widths: letters: L0 R2 R3 U0 U1 U2 U3
- roles: as FVF-0217
- re-laid out: n expected: labels keep their length (U0→D0, nU0→nD0, nO1→nU1); K10 in place
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
            1 -- L0 -- ( ) -- R2 -- R3 -- 1
                 |      |      |      |
                 U0     U1     U2     U3
                 |      |      |      |
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### FVF-0850

- file: `t3toolbox/frame_variations_format.py`; old span: 850–853; kind: ascii diagram; fenced: y
- letters and widths: letters: L0 L1 O2 R3 U0 U1 U3
- roles: as FVF-0222
- re-laid out: n expected: labels keep their length (U0→D0, nU0→nD0, nO1→nU1); K10 in place
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
            1 -- L0 -- L1 -- O2 -- R3 -- 1
                 |     |     |     |
                 U0    U1    ( )   U3
                 |     |     |     |
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### FVF-1264

- file: `t3toolbox/frame_variations_format.py`; old span: 1264–1267; kind: ascii diagram; fenced: y
- letters and widths: letters: H1 L0 R2 R3 U0 U1 U2 U3
- roles: U0–U3 the Tucker store (→ D); (H1) the TT variation; L, R stay
- re-laid out: n expected: labels keep their length (U0→D0, nU0→nD0, nO1→nU1); K10 in place
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
        1 -- L0 --(H1)-- R2 -- R3 -- 1
             |     |     |     |
             U0    U1    U2    U3
             |     |     |     |
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### FVF-1271

- file: `t3toolbox/frame_variations_format.py`; old span: 1271–1274; kind: ascii diagram; fenced: y
- letters and widths: letters: D2 L0 L1 R3 U0 U1 U3 V2
- roles: D2 the TT-shaped store (→ U2); U → D; (V2) the Tucker variation
- re-laid out: n expected: labels keep their length (U0→D0, nU0→nD0, nO1→nU1); K10 in place
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
        1 -- L0 -- L1 -- D2 -- R3 -- 1
             |     |     |     |
             U0    U1   (V2)   U3
             |     |     |     |
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### FVF-1302

- file: `t3toolbox/frame_variations_format.py`; old span: 1302–1322; kind: companion doctest; fenced: n
- letters and widths: letters: D0 D1 D2 H0 H1 H2 L0 L1 L2 R0 R1 R2 U0 U1 U2 V0 V1 V2; store words: down up
- roles: example letters: U0–U2 the Tucker store (→ D0–D2), D0–D2 the TT-shaped store (→ U0–U2) — letter_sites rows (R2, K15); prose "Tucker (up) cores", "up cores", "down core D1" are tier E rows; expected outputs unchanged
- re-laid out: n/a: doctest example lines (realigned by §4.10 if a name grows)
- new == old possible: no
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
    >>> (U0, U1, U2) = (randn(10, 14), randn(11, 15), randn(12, 16))
    >>> (L0, L1, L2) = (randn(1, 10, 2), randn(2, 11, 3), randn(3, 12, 4))
    >>> (R0, R1, R2) = (randn(2, 10, 4), randn(4, 11, 5), randn(5, 12, 1))
    >>> (D0, D1, D2) = (randn(1, 9, 4), randn(2, 8, 5), randn(3, 7, 1))
    >>> frame = bvf.T3Frame((U0, U1, U2), (D0, D1, D2), (L0, L1, L2), (R0, R1, R2))
    >>> (V0, V1, V2) = (randn(9, 14), randn(8, 15), randn(7, 16))
    >>> (H0, H1, H2) = (randn(1, 10, 4), randn(2, 11, 5), randn(3, 12, 1))
    >>> variations = bvf.T3Variations((V0, V1, V2), (H0, H1, H2))

    Replacing the index-1 TT-core swaps ``H1`` into the right-orthogonal chain ``L0, ?, R2``; the
    Tucker (up) cores are unchanged:

    >>> tt_term = bvf.fv_to_t3((True, 1), frame, variations)
    >>> expected = ((U0, U1, U2), (L0, H1, R2))      # up cores untouched; TT chain = L0, H1, R2
    >>> print(np.allclose(cw.corewise_norm(cw.corewise_sub(tt_term.data, expected)), 0.0))
    True

    Replacing the index-1 Tucker core swaps ``V1`` into the up cores and the down core ``D1`` into the chain:

    >>> tucker_term = bvf.fv_to_t3((False, 1), frame, variations)
    >>> expected = ((U0, V1, U2), (L0, D1, R2))
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### FVF-1355

- file: `t3toolbox/frame_variations_format.py`; old span: 1355–1358; kind: ascii diagram; fenced: y
- letters and widths: letters: H1 L0 R2 R3 U0 U1 U2 U3
- roles: U0–U3 the Tucker store (→ D); H1 the TT variation; L, R stay
- re-laid out: n expected: labels keep their length (U0→D0, nU0→nD0, nO1→nU1); K10 in place
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
                  1 -- L0 -- H1 -- R2 -- R3 -- 1
        X    =         |     |     |     |
                       U0    U1    U2    U3
                       |     |     |     |
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### FVF-1362

- file: `t3toolbox/frame_variations_format.py`; old span: 1362–1365; kind: ascii diagram; fenced: y
- letters and widths: letters: D2 L0 L1 R3 U0 U1 U3 V2
- roles: D2 the TT-shaped store (→ U2); U → D; V2 the Tucker variation
- re-laid out: n expected: labels keep their length (U0→D0, nU0→nD0, nO1→nU1); K10 in place
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
                  1 -- L0 -- L1 -- D2 -- R3 -- 1
        X    =         |     |     |     |
                       U0    U1    V2    U3
                       |     |     |     |
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### FVF-1371

- file: `t3toolbox/frame_variations_format.py`; old span: 1371–1374; kind: shape listing; fenced: y
- letters and widths: letters: L0 L1 L2 O0 O1 O2 O3 R1 R2 R3 U0 U1 U2 U3; store words: down down_tt_cores left_tt_cores right_tt_cores up
- roles: tucker_cores (U, "up orthogonal": backwards, §4.3 :1371) → D, down orthogonal; down_tt_cores (O, "down orthogonal": backwards, §4.3 :1372) → up_tt_cores (U), up orthogonal; L, R stay
- re-laid out: y expected: the store identifiers change length (up_tucker_* → down_tucker_*, down_tt_* → up_tt_*); K10b
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
        - tucker_cores      = (U0,U1, U2, U3), up orthogonal
        - down_tt_cores     = (O0, O1, O2, O3), down orthogonal
        - left_tt_cores     = (L0, L1, L2),     left orthogonal
        - right_tt_cores    = (R1, R2, R3),     right orthogonal
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~


### `t3toolbox/manifold.py`

#### M-0685

- file: `t3toolbox/manifold.py`; old span: 685–686; kind: index equation; fenced: y
- letters and widths: letters: H_i L L_i U U_i V_i
- roles: `einsum(..., U_i, V_i) = 0` → D_i; "(Tucker variations ⟂ U)" → ⟂ D; L_i, H_i stay
- re-laid out: n expected unless a label grows; K10 in place
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
        - ``einsum('...ia,...ja->...ij', U_i, V_i) = 0`` for all i (Tucker variations ⟂ U).
        - ``einsum('...abi,...abj->...ij', L_i, H_i) = 0`` for i = 0..d-2 (TT variations ⟂ L).
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~


### `t3toolbox/tucker_tensor_train.py`

#### TTT-2765

- file: `t3toolbox/tucker_tensor_train.py`; old span: 2765; kind: index equation; fenced: y
- letters and widths: letters: B
- re-laid out: n expected unless a label grows; K10 in place
- new == old possible: yes expected: B and G are a T3's own cores (§3.3: a B already written stays); no store letter or width
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
            I.e., ``np.einsum('...io,jo->...ij', B, B) = (stacked) identity matrix``, where ``B=new_x.tucker_cores[ii]``.
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TTT-2837

- file: `t3toolbox/tucker_tensor_train.py`; old span: 2837; kind: index equation; fenced: y
- letters and widths: letters: G
- re-laid out: n expected unless a label grows; K10 in place
- new == old possible: yes expected: B and G are a T3's own cores (§3.3: a B already written stays); no store letter or width
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
            I.e., ``einsum('...iaj,...iak->...jk', G, G) = (stacked) identity matrix``, where ``G=new_x.tt_cores[ii]``.
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TTT-2908

- file: `t3toolbox/tucker_tensor_train.py`; old span: 2908; kind: index equation; fenced: y
- letters and widths: letters: G
- re-laid out: n expected unless a label grows; K10 in place
- new == old possible: yes expected: B and G are a T3's own cores (§3.3: a B already written stays); no store letter or width
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
            I.e., ``einsum('...iaj,...kaj->...ik', G, G) = (stacked) identity matrix``, where ``G=new_tt_cores[ii]``.
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### TTT-2979

- file: `t3toolbox/tucker_tensor_train.py`; old span: 2979; kind: index equation; fenced: y
- letters and widths: letters: G
- re-laid out: n expected unless a label grows; K10 in place
- new == old possible: yes expected: B and G are a T3's own cores (§3.3: a B already written stays); no store letter or width (the backwards adjective at :2978 is a separate §4.3 row)
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
            I.e., ``einsum('...iaj,...ibj->...ab', G, G) = (stacked) identity matrix``, where ``G=new_tt_cores[ii]``.
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~


### `t3toolbox/uniform_frame_variations_format.py`

#### UFVF-0110

- file: `t3toolbox/uniform_frame_variations_format.py`; old span: 110–113, columns 49–end; kind: field-comment relation; fenced: y
- letters and widths: letters: B_dxo B_dyo P_diax P_diay Q_dxaj Q_dyaj R_dixj R_diyj; widths: nD nU rL rR
- roles: as FVF-0156, supercore form: B_dxo → D_dxo, R_dixj → U_dixj, P → L, Q → R, widths nU ↔ nD
- re-laid out: n: in place after §4.10's realignment of the field names (K10 on the comment span)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
# B_dxo B_dyo   = I_dxy, shape = (d,)+stack_shape+(nU, N)
# R_dixj R_diyj = I_dxy  shape = (d,)+stack_shape+(rL, nD, rR)
# P_diax P_diay = I_dxy, shape = (d,)+stack_shape+(rL, nU, rL)
# Q_dxaj Q_dyaj = I_dxy  shape = (d,)+stack_shape+(rR, nU, rR)
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### UFVF-1106

- file: `t3toolbox/uniform_frame_variations_format.py`; old span: 1106–1109; kind: ascii diagram; fenced: y
- letters and widths: letters: H1 L0 R2 R3 U0 U1 U2 U3
- roles: U0–U3 the Tucker store (→ D); H1 the TT variation
- re-laid out: n expected: labels keep their length (U0→D0, nU0→nD0, nO1→nU1); K10 in place
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
                  1 -- L0 -- H1 -- R2 -- R3 -- 1
        X    =         |     |     |     |
                       U0    U1    U2    U3
                       |     |     |     |
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### UFVF-1113

- file: `t3toolbox/uniform_frame_variations_format.py`; old span: 1113–1116; kind: ascii diagram; fenced: y
- letters and widths: letters: L0 L1 O2 R3 U0 U1 U3 V2
- roles: O2 the TT-shaped store (→ U2); U → D; V2 the Tucker variation
- re-laid out: n expected: labels keep their length (U0→D0, nU0→nD0, nO1→nU1); K10 in place
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
                  1 -- L0 -- L1 -- O2 -- R3 -- 1
        X    =         |     |     |     |
                       U0    U1    V2    U3
                       |     |     |     |
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

#### UFVF-1122

- file: `t3toolbox/uniform_frame_variations_format.py`; old span: 1122–1125; kind: shape listing; fenced: y
- letters and widths: letters: L0 L1 L2 O0 O1 O2 O3 R1 R2 R3 U0 U1 U2 U3; store words: down left_tt_cores outer_tt_cores right_tt_cores up
- roles: tucker_cores (U, backwards "up orthogonal", §4.3 :1122) → D; outer_tt_cores (O, backwards "down orthogonal", §4.3 :1125) → U; L, R stay
- re-laid out: y expected: the store identifiers change length (up_tucker_* → down_tucker_*, down_tt_* → up_tt_*); K10b
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
        - tucker_cores       = (U0,U1, U2, U3), up orthogonal
        - left_tt_cores     = (L0, L1, L2),     left orthogonal
        - right_tt_cores    = (R1, R2, R3),     right orthogonal
        - outer_tt_cores    = (O0, O1, O2, O3), down orthogonal
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~


### `t3toolbox/uniform_manifold.py`

#### UNMA-0478

- file: `t3toolbox/uniform_manifold.py`; old span: 478–479; kind: index equation; fenced: y
- letters and widths: letters: L P_i U_i
- roles: ``U_i^T dU_i = 0`` → D_i (and its variation's letter by D-12); ``(P_i^L)^T dG_i^L = 0`` → L (D-7)
- re-laid out: n expected unless a label grows; K10 in place
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
        coordinate ones; not enforced at construction): ``U_i^T dU_i = 0`` (all i) and
        ``(P_i^L)^T dG_i^L = 0`` (i = 0..d-2) -- conditions (47)-(48), Appendix A.3 of Alger et al. (2026).
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~


### `tests/test_frame_variations_format.py`

#### T-TFVF-0025

- file: `tests/test_frame_variations_format.py`; old span: 25–27; kind: shape listing; fenced: y
- letters and widths: letters: D_i H_i L_i R_i U_i V_i; widths: nD_i nU_i rL_(i+1) rL_i rR_(i+1) rR_i
- roles: the test helper's shape listing (R1 review MAJOR-1): U_i (the Tucker store) → D_i (nU_i → nD_i); this docstring's D_i (the TT-shaped store) → U_i (nD_i → nU_i); L_i, R_i middle legs nU_i → nD_i; V_i rows nD_i → nU_i; H_i middle leg nU_i → nD_i (D-8); rL/rR stay
- re-laid out: n expected: letters swap at equal lengths (K10 in place)
- new == old possible: no: it carries a changing letter, width or store word
- new span: (R4/R5)
- same-reason: (R4/R5: required only when the new text equals the old)

old text:

~~~
        U_i = (nU_i, N_i)           L_i = (rL_i, nU_i, rL_(i+1))
        R_i = (rR_i, nU_i, rR_(i+1))  D_i = (rL_i, nD_i, rR_(i+1))
        V_i = (nD_i, N_i)           H_i = (rL_i, nU_i, rR_(i+1))
~~~

new text (R4/R5, verbatim the file at the new span):

~~~
~~~

