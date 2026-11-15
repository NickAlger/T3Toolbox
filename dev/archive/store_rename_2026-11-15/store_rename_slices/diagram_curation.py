"""R1 hand curation of the §4A diagram/equation inventory (read by ``diagram_scanner.py``).

Every entry has its reason. Line numbers are of ``v2026.4.0`` (``docs/glossary.md``: of ``88fbbedb``).
"""

# ---------------------------------------------------------------------------------------------------------
# Hand blocks: (file, first, last, col_start, col_end, kind, fenced). A hand block replaces every automatic
# block it overlaps.
# ---------------------------------------------------------------------------------------------------------
CURATION_BLOCKS = [
    # frame_variations_format.py:56-66: two rows of THREE side-by-side diagrams each. One block per diagram (plan
    # §4A a: "never several grouped, so a half-done group cannot pass K12"): each panel owns its column range
    # (8-31, 32-55, 56-end; the inter-panel spaces go with the panel on their left). The R1 implementer's split.
    ('t3toolbox/frame_variations_format.py', 58, 61, 8, 32, 'ascii diagram', True),
    ('t3toolbox/frame_variations_format.py', 58, 61, 32, 56, 'ascii diagram', True),
    ('t3toolbox/frame_variations_format.py', 58, 61, 56, None, 'ascii diagram', True),
    ('t3toolbox/frame_variations_format.py', 63, 66, 8, 32, 'ascii diagram', True),
    ('t3toolbox/frame_variations_format.py', 63, 66, 32, 56, 'ascii diagram', True),
    ('t3toolbox/frame_variations_format.py', 63, 66, 56, None, 'ascii diagram', True),
    # the orthogonality listing is a relation block, not a plain shape listing (sample edit 6)
    ('t3toolbox/frame_variations_format.py', 117, 120, None, None, 'index equation', True),
    # companion doctest of the fv_to_t3 diagrams (plan §4A a: inventoried, NOT fenced; its example letters are
    # letter_sites rows renamed in R2 and checked by K15; the reviewer checks they agree with FVF-1264/1271)
    ('t3toolbox/frame_variations_format.py', 1302, 1322, None, None, 'companion doctest', False),
    # probing.py seeds 640-651 / 1215-1226: the uniform supercore shape comments (backwards adjectives, §4.3)
    # and the frame-order equation are their own blocks, beside the field-comment relations 640-643 / 1215-1218
    ('t3toolbox/backend/probing.py', 646, 649, 26, None, 'field-comment relation', True),
    ('t3toolbox/backend/probing.py', 651, 651, 11, None, 'index equation', True),
    ('t3toolbox/backend/probing.py', 1221, 1224, 26, None, 'field-comment relation', True),
    ('t3toolbox/backend/probing.py', 1226, 1226, 11, None, 'index equation', True),
    # utv_operations.py seed 314-315: the doubled-rank block structure, [U | dU] = [up | down], [Q | P] = [R | L]
    ('t3toolbox/backend/utv_operations.py', 314, 315, None, None, 'index equation', True),
    # fv_operations.py: the four-family orthogonality-check listing (the twin of FVF-0354)
    ('t3toolbox/backend/fv_operations.py', 201, 202, None, None, 'shape listing', True),
    # module-header shape listings of the frame / variation data tuples (comments)
    ('t3toolbox/backend/padded_frames.py', 41, 41, None, None, 'shape listing', True),
    ('t3toolbox/backend/padded_frames.py', 43, 43, None, None, 'shape listing', True),
    ('t3toolbox/backend/uniform_padded_frames.py', 47, 48, None, None, 'shape listing', True),
    ('t3toolbox/backend/uniform_padded_frames.py', 50, 50, None, None, 'shape listing', True),
    # user_guide.rst seed 124-142: one displayed sum v = ... + ... over two rows is ONE diagram (sample batch 3,
    # edit 8, approved: "R1 splits it into 125-133 and 139-142")
    ('docs/user_guide.rst', 125, 133, None, None, 'ascii diagram', True),
    # markdown tables of frame letters / widths (frame_variations.md's symbol table is a seed)
    ('docs/frame_variations.md', 23, 30, None, None, 'markdown table', True),
    ('docs/frame_variations.md', 80, 85, None, None, 'markdown table', True),
    ('docs/batching_and_stacking.md', 40, 46, None, None, 'markdown table', True),
    ('docs/weighting.md', 69, 74, None, None, 'markdown table', True),
]

# side-by-side panels: (file, first, col_start) -> id suffix
CURATION_PANELS = {
    ('t3toolbox/frame_variations_format.py', 58, 8): 'a', ('t3toolbox/frame_variations_format.py', 58, 32): 'b',
    ('t3toolbox/frame_variations_format.py', 58, 56): 'c', ('t3toolbox/frame_variations_format.py', 63, 8): 'a',
    ('t3toolbox/frame_variations_format.py', 63, 32): 'b', ('t3toolbox/frame_variations_format.py', 63, 56): 'c',
}

# ---------------------------------------------------------------------------------------------------------
# Automatic blocks dismissed, with a reason for each of their lines
# ---------------------------------------------------------------------------------------------------------
_DROP = {
    ('docs/contributor/curvature_internals.md', 168, 168):
        'INLINE_NO_STORE: `Lᵀ K_N L = 0`, L is the lift (§3.1), no store letter',
    ('docs/contributor/curvature_internals.md', 181, 182):
        'DOC_CODE: a code line in a doc code snippet (`U_g = (Q * sign(diag(R))).T`); its letter is a §3.4 text row (R5), not a diagram or equation',
    ('docs/contributor/scan_body_principles.md', 224, 231):
        'DOC_CODE_EXAMPLE: a code exemplar in a doc fence; its shape comments are scripted by A3 (tier A runs over the docs) like the code it mirrors',
    ('docs/contributor/signature_style.md', 56, 62):
        'DOC_CODE_EXAMPLE: the signature exemplar in a doc fence; its shape comments are scripted by A3 (tier A runs over the docs) like the code it mirrors',
    ('docs/reading_signatures.md', 12, 18):
        'DOC_CODE_EXAMPLE: the signature exemplar in a doc fence; its shape comments are scripted by A3 (tier A runs over the docs) like the code it mirrors',
    ('docs/entries_apply_probe.md', 76, 76):
        'INLINE_NO_STORE: a dense-check einsum with no store letter or width',
    ('docs/t3svd_minimal_ranks.md', 16, 19):
        'WIDTH_RL_RR_ONLY: only the unchanged generic rL/rR (D-3) and the T3\'s own n, r, N; no store letter: nothing to rewrite',
    ('docs/t3svd_minimal_ranks.md', 35, 36):
        'WIDTH_RL_RR_ONLY: only the unchanged generic rL/rR (D-3) and the T3\'s own n, r, N; no store letter: nothing to rewrite',
    ('docs/t3svd_minimal_ranks.md', 62, 64):
        'WIDTH_RL_RR_ONLY: only the unchanged generic rL/rR (D-3) and the T3\'s own n, r, N; no store letter: nothing to rewrite',
    ('t3toolbox/backend/ranks.py', 83, 83):
        'WIDTH_RL_RR_ONLY: only the unchanged generic rL/rR (D-3) and the T3\'s own n, N; no store letter: nothing to rewrite',
    ('t3toolbox/backend/optimizers.py', 214, 214):
        'INLINE_NO_STORE: the Hessian formula; its L is the lift (§3.1 lists optimizers.py:214 as staying), no store letter',
    ('t3toolbox/backend/tv_operations.py', 231, 232):
        'PROSE_DASHES: "--" as a dash in prose; its `[U_g; Udot]` is a §1.2a/§3.4 text row, not a diagram',
    ('t3toolbox/shared_geometry.py', 428, 429):
        'PROSE_DASHES: "--" as a dash in a comment; not a diagram',
    ('tests/backend/test_uniform_fitting.py', 201, 202):
        'PROSE_DASHES: "--" as a dash in a docstring; not a diagram',
    ('tests/backend/test_curvature.py', 1766, 1766):
        'INLINE_NO_STORE: `Pi p_vert != 0`, Pi is the gauge projector and L the lift; no store letter',
    ('tests/test_uniform_manifold.py', 109, 109):
        'HAND_ROW: §1.5 item 3 enumerates `test_uniform_manifold.py:108` (with this continuation) as an R4a hand row with its full replacement text; a rank-mask parenthetical in docstring prose, not a displayed equation',
}
CURATION_DROP = [(f, a) for (f, a, b) in _DROP]
CURATION_NOTES = {(f, k): reason for (f, a, b), reason in _DROP.items() for k in range(a, b + 1)}
CURATION_NOTES.update({
    ('t3toolbox/frame_variations_format.py', 1348): 'DIAGRAM_NO_STORE: the input T3 (B/G) of t3_orthogonal_representations; B stays (§3.3)',
    ('t3toolbox/uniform_frame_variations_format.py', 1099): 'DIAGRAM_NO_STORE: the input T3 (B/G) of ut3_orthogonal_representations; B stays (§3.3)',
    ('t3toolbox/tucker_tensor_train.py', 64): 'DIAGRAM_NO_STORE: the TuckerTensorTrain B/G diagram (plan seed tucker_tensor_train.py:61-72, "to be dismissed"); n, r, N, B, G only',
})

# ---------------------------------------------------------------------------------------------------------
# The seed list (§4A a), entry by entry
# ---------------------------------------------------------------------------------------------------------
SEED_RESOLUTION = [
    ('`frame_variations_format.py:56–66`', 'FVF-0058a/b/c (58–61, the TT-variation row) and FVF-0063a/b/c (63–66, the Tucker-variation row): six side-by-side diagrams, one column-ranged block per diagram (columns 8–31, 32–55, 56–end), so a half-done row cannot pass K12'),
    ('`:70–78`', '<<t3toolbox/frame_variations_format.py:71>> (71–74, the frame listing) and <<t3toolbox/frame_variations_format.py:77>> (77–78, the variation listing): two listings, two blocks'),
    ('`:86–96`', '<<t3toolbox/frame_variations_format.py:88>> (88–96; 86–87 are the lead-in sentence and blank)'),
    ('`:98–108`', '<<t3toolbox/frame_variations_format.py:100>> (100–108)'),
    ('`:116–124`', '<<t3toolbox/frame_variations_format.py:117>> (117–120, the orthogonality relations) and <<t3toolbox/frame_variations_format.py:123>> (123–124, the gauge conditions); 116 and 122 are lead-in sentences'),
    ('`:156–159`', '<<t3toolbox/frame_variations_format.py:156>> (columns 48–end: the field comments; the field NAMEs stay outside the fence)'),
    ('`:219–224`', '<<t3toolbox/frame_variations_format.py:217>> (217–220) and <<t3toolbox/frame_variations_format.py:222>> (222–225): the two hole diagrams of `T3Frame.variation_shapes`'),
    ('`:353–357`', '<<t3toolbox/frame_variations_format.py:354>> (354–357)'),
    ('`:843–853`', '<<t3toolbox/frame_variations_format.py:845>> (845–848) and <<t3toolbox/frame_variations_format.py:850>> (850–853): the two hole diagrams of `T3Variations` fitting'),
    ('`:1262–1274`', '<<t3toolbox/frame_variations_format.py:1264>> (1264–1267) and <<t3toolbox/frame_variations_format.py:1271>> (1271–1274): the two `fv_to_t3` diagrams'),
    ('`:1305–1322` / companion `:1302–1322`', '<<t3toolbox/frame_variations_format.py:1302>> (1302–1322), companion doctest, NOT fenced'),
    ('`:1346–1365`', '<<t3toolbox/frame_variations_format.py:1355>> (1355–1358) and <<t3toolbox/frame_variations_format.py:1362>> (1362–1365); the input-T3 diagram 1348–1351 (B/G only) is dismissed'),
    ('`:1370–1374`', '<<t3toolbox/frame_variations_format.py:1371>> (1371–1374); the x listing 1368–1369 (B/G) and the variation listing 1376–1377 (V/H) carry no store letter'),
    ('`uniform_frame_variations_format.py:110–113`', '<<t3toolbox/uniform_frame_variations_format.py:110>> (columns 49–end)'),
    ('`uniform_frame_variations_format.py:1097–1125`', '<<t3toolbox/uniform_frame_variations_format.py:1106>> (1106–1109), <<t3toolbox/uniform_frame_variations_format.py:1113>> (1113–1116), <<t3toolbox/uniform_frame_variations_format.py:1122>> (1122–1125); the input diagram 1099–1102 (B/G) dismissed'),
    ('`manifold.py:685`', '<<t3toolbox/manifold.py:685>> (685–686)'),
    ('`uniform_manifold.py:478`', '<<t3toolbox/uniform_manifold.py:478>> (478–479)'),
    ('`probing.py:640–651`', '<<t3toolbox/backend/probing.py:640>> (640–643, cols 40–end), <<t3toolbox/backend/probing.py:646>> (646–649, cols 26–end), <<t3toolbox/backend/probing.py:651>> (651, cols 11–end)'),
    ('`probing.py:1215–1226`', '<<t3toolbox/backend/probing.py:1215>> (1215–1218, cols 40–end), <<t3toolbox/backend/probing.py:1221>> (1221–1224, cols 26–end), <<t3toolbox/backend/probing.py:1226>> (1226, cols 11–end)'),
    ('`ufv_operations.py:81–84`', '<<t3toolbox/backend/ufv_operations.py:83>> (83; 81 is the lead-in, 82/84 blank)'),
    ('`utv_operations.py:314–315`', '<<t3toolbox/backend/utv_operations.py:314>> (314–315)'),
    ('`ranks.py:644–647`', '<<t3toolbox/backend/ranks.py:644>> (644–647)'),
    ('`tucker_tensor_train.py:61–72`', 'dismissed: the B/G diagram of a T3 (n, r, N, B, G only; no store letter or width), as the plan says'),
    ('companion doctests', '<<t3toolbox/frame_variations_format.py:1302>> is the only one: no other docstring holds both a store diagram and example lines with frame letters (the scan of every doctest example line with a core letter found only this one and linalg\'s SVD U)'),
    ('`frame_variations.md:13–17` and its symbol table', '<<docs/frame_variations.md:14>> (14–16; 13 and 17 are the code-fence markers) and <<docs/frame_variations.md:23>> (23–30, the symbol table)'),
    ('`user_guide.rst:124–142`', '<<docs/user_guide.rst:125>> (125–133, one displayed sum over two rows) and <<docs/user_guide.rst:139>> (139–142, the frame-core list), as sample batch 3 edit 8'),
    ('`padded_frame_cone.tex` Table 1', '<<docs/padded_frame_cone.tex:102>> (102–116, the table float)'),
    ('the exact-Hessian note\'s two tikz figures and its equation ranges', '<<docs/exact_hessian_two_hole.tex:151>> (151–200) and <<docs/exact_hessian_two_hole.tex:338>> (338–379), the two figures; every letter-bearing equation line of the note is a block of its own'),
    ('the four notes\' frame equations', 'every letter-bearing equation line of the six notes is a TEX-* block (environments whole, text-mode runs by paragraph)'),
]

# ---------------------------------------------------------------------------------------------------------
# Notes per block (roles of the letters; re-layout expectation; why new may equal old)
# ---------------------------------------------------------------------------------------------------------
_OWN = 'B and G are a T3\'s own cores (§3.3: a B already written stays); no store letter or width'
BLOCK_NOTES = {
    'FVF-0058': {'roles': 'U0–U2 the frame\'s Tucker store (→ D); L, R left/right (stay); (H) the TT variation'},
    'FVF-0063': {'roles': 'D0–D2 the TT-shaped store (this docstring\'s own D, → U); U the Tucker store (→ D); (V) the Tucker variation'},
    'FVF-0071': {'roles': 'up_tucker_cores (U, nUi) → down_tucker_cores (D, nDi); down_tt_cores (D, nDi) → up_tt_cores (U, nUi); left/right write a generic `ni` (D-8 makes their middle leg nD: the rewriter decides whether to keep `ni`); re-laid out'},
    'FVF-0077': {'roles': 'tucker_variations rows nDi → nUi; tt_variations middle leg nUi → nDi (D-8)'},
    'FVF-0088': {'roles': 'U0, U1, Ud the Tucker store (→ D0, D1, Dd; `Ud` is the known typo for U(d-1), glossary E17); vertical widths nU → nD; rL, rR, N, (H1) stay (sample edit 5)'},
    'FVF-0100': {'roles': 'D1 the TT-shaped store (→ U1); nO1 → nU1; nU → nD; U0, Ud → D0, Dd; (V1) stays (sample edit 5)'},
    'FVF-0117': {'roles': 'U_ia (Tucker store) → D_ia; O_aib (TT-shaped store) → U_aib; L, R stay; adjectives (orthogonal → down-orthogonal, outer-orthogonal → up-orthogonal) and the missing colon after right-orthogonal (§4A d); relations re-derived (sample edit 6)', 'relayout': 'y: the store names grow and the adjectives change (K10b; sample edit 6)'},
    'FVF-0123': {'roles': 'gauge U_ia V_ja = 0 → D_ia V_ja = 0; L_abi H_abj stays'},
    'FVF-0156': {'roles': 'B_xo (Tucker store) → D_xo; R_ixj (TT store, clashes with right) → U_ixj; P → L, Q → R; widths nUi ↔ nDi; the model is sample edit 1\'s approved AFTER', 'relayout': 'n: in place after §4.10\'s realignment of the field names (K10 on the comment span; sample edit 1)'},
    'FVF-0217': {'roles': 'U0–U3 the Tucker store (→ D); ( ) the hole; L, R stay'},
    'FVF-0222': {'roles': 'O2 the TT-shaped store (→ U2); U0, U1, U3 → D; ( ) the hole'},
    'FVF-0354': {'roles': '"up_tucker U_i" → down_tucker D_i; "down/outer D_i" → up U_i; einsum subscripts byte-identical (§1.5 item 6), only the operand letters change'},
    'FVF-0784': {'roles': 'tucker_variations rows nDi → nUi; tt_variations middle leg nUi → nDi'},
    'FVF-0845': {'roles': 'as FVF-0217'},
    'FVF-0850': {'roles': 'as FVF-0222'},
    'FVF-1264': {'roles': 'U0–U3 the Tucker store (→ D); (H1) the TT variation; L, R stay'},
    'FVF-1271': {'roles': 'D2 the TT-shaped store (→ U2); U → D; (V2) the Tucker variation'},
    'FVF-1302': {'roles': 'example letters: U0–U2 the Tucker store (→ D0–D2), D0–D2 the TT-shaped store (→ U0–U2) — letter_sites rows (R2, K15); prose "Tucker (up) cores", "up cores", "down core D1" are tier E rows; expected outputs unchanged', 'relayout': 'n/a: doctest example lines (realigned by §4.10 if a name grows)', 'same': 'no'},
    'FVF-1355': {'roles': 'U0–U3 the Tucker store (→ D); H1 the TT variation; L, R stay'},
    'FVF-1362': {'roles': 'D2 the TT-shaped store (→ U2); U → D; V2 the Tucker variation'},
    'FVF-1371': {'roles': 'tucker_cores (U, "up orthogonal": backwards, §4.3 :1371) → D, down orthogonal; down_tt_cores (O, "down orthogonal": backwards, §4.3 :1372) → up_tt_cores (U), up orthogonal; L, R stay'},
    'UFVF-0110': {'roles': 'as FVF-0156, supercore form: B_dxo → D_dxo, R_dixj → U_dixj, P → L, Q → R, widths nU ↔ nD', 'relayout': 'n: in place after §4.10\'s realignment of the field names (K10 on the comment span)'},
    'UFVF-1106': {'roles': 'U0–U3 the Tucker store (→ D); H1 the TT variation'},
    'UFVF-1113': {'roles': 'O2 the TT-shaped store (→ U2); U → D; V2 the Tucker variation'},
    'UFVF-1122': {'roles': 'tucker_cores (U, backwards "up orthogonal", §4.3 :1122) → D; outer_tt_cores (O, backwards "down orthogonal", §4.3 :1125) → U; L, R stay'},
    'P-0640': {'roles': 'tuple-element comments: up_tucker_cores U_xo → down_tucker_cores D_xo; down_tt_cores O_ixj → up_tt_cores U_ixj; P → L, Q → R; nU → nD, nO → nU', 'relayout': 'y: the store labels change length (K10b on the comment span)'},
    'P-0646': {'roles': 'uniform supercore labels (up_tucker_supercore → down_tucker_supercore, down_tt_supercore → up_tt_supercore); widths nU → nD, nO → nU; adjectives "up orthogonal elements" / "down orthogonal elements" are backwards today (§4.3 probing.py:646–647)', 'relayout': 'y: labels change length (K10b on the comment span)'},
    'P-0651': {'roles': '`frame order = T3Frame.data = (up, down, left, right) = (U, O, P, Q)` → `(down, up, left, right) = (D, U, L, R)` (tier D\'s patterns, applied by hand inside the fence)', 'relayout': 'n expected (same lengths)'},
    'P-1215': {'roles': 'as P-0640', 'relayout': 'y: the store labels change length (K10b on the comment span)'},
    'P-1221': {'roles': 'as P-0646 (§4.3 probing.py:1221–1222)', 'relayout': 'y: labels change length (K10b on the comment span)'},
    'P-1226': {'roles': 'as P-0651', 'relayout': 'n expected (same lengths)'},
    'A-0465': {'roles': 'O_k the TT-shaped store (→ U_k); "over the down mode nO" → up mode nU; (rL, nU, rR) → (rL, nD, rR); dU-tilde_k (§1.2a: `_apply_transpose_adjoint` takes D forms) → dD-tilde_k over (nU, N)'},
    'UFO-0083': {'roles': '`nU = up.shape[-2]  nD = down.shape[-2] ... N = up.shape[-1]`: the width names follow their stores (nD = down.shape[-2] for the Tucker supercore, nU = up.shape[-2], N = down.shape[-1]); the rewriter keeps or reorders by physical order'},
    'UTO-0314': {'roles': '[U | dU] = [up | down] → [D | dD] = [down | up]; [Q | P] = [R | L] → [R | L] (T4S\'s [Q | P]) (§1.5 item 3\'s rows for utv_operations.py:315–322)'},
    'UTO-0538': {'roles': '``U^T dU = 0`` (the frame\'s Tucker store and its variation) → ``D^T dD = 0`` (D-12); "up-core" → down core; ``(P^L)^T dG^L = 0`` → L (D-7)'},
    'FO-0201': {'roles': '"up_tucker U_i, outer/down D_i" → down_tucker D_i, up U_i (this docstring\'s D is the TT-shaped store, inv 2 C2); L_i, R_i stay'},
    'PF-0041': {'roles': '`(up, down, left, right)` → `(down, up, left, right)` (tier D); widths swap: C+(nUi, Ni) → C+(nDi, Ni), C+(rLi, nDi, rR(i+1)) → nUi, left/right nUi → nDi'},
    'PF-0043': {'roles': 'variations: C+(nDi, Ni) → C+(nUi, Ni); C+(rLi, nUi, rR(i+1)) → nDi'},
    'UPF-0047': {'roles': 'the store words up/down (free prose, tier E) and widths swap together: "up (d,)+C+(nU, N), down (d,)+C+(rL, nD, rR)" → "down (d,)+C+(nD, N), up (d,)+C+(rL, nU, rR)"; left/right middle legs nU → nD; the mask line likewise; "block layout: nU = nD" stays true either way'},
    'UPF-0050': {'roles': 'V (d,)+C+(nD, N) → nU; H (d,)+C+(rL, nU, rR) → nD'},
    'R-0644': {'roles': 'store-word index variables: up_i (the Tucker store\'s rank) → down_i, down_i (the TT-shaped store\'s) → up_i; left_i, right_i stay; the recurrence\'s step order stays'},
    'M-0685': {'roles': '`einsum(..., U_i, V_i) = 0` → D_i; "(Tucker variations ⟂ U)" → ⟂ D; L_i, H_i stay'},
    'UM-0478': {'roles': '``U_i^T dU_i = 0`` → D_i (and its variation\'s letter by D-12); ``(P_i^L)^T dG_i^L = 0`` → L (D-7)'},
    'TO-0108': {'roles': '``U_i V_i^T = 0`` → D_i; L_i, H_i stay'},
    'T3O-0044': {'same': 'yes expected: ' + _OWN},
    'TTT-2765': {'same': 'yes expected: ' + _OWN},
    'TTT-2837': {'same': 'yes expected: ' + _OWN},
    'TTT-2908': {'same': 'yes expected: ' + _OWN},
    'TTT-2979': {'same': 'yes expected: ' + _OWN + ' (the backwards adjective at :2978 is a separate §4.3 row)'},
    'DOC-G-0106': {'same': 'yes expected: `B Bᵀ = I` is a T3\'s own Tucker core (B stays); the line\'s prose is the glossary\'s, read at 88fbbedb'},
    'DOC-G-0273': {'roles': '`Ui Viᵀ = 0` → `Di Viᵀ = 0` (the frame\'s Tucker store)'},
    'DOC-FV-0014': {'roles': 'U0–U2 the Tucker store (→ D); (H) the TT variation; L, R stay'},
    'DOC-FV-0023': {'roles': 'the symbol table: up-Tucker Uᵢ/U/nU → down-Tucker Dᵢ/D/nD; down-TT Oᵢ/O/nD → up-TT Uᵢ/U/nU; `L`/`P`, `R`/`Q` → L, R (D-7); V\'s width nD → nU, H\'s nU → nD (D-8); the T4S letters go to the correspondence line (§3)', 'relayout': 'n/a: a markdown table (not column-aligned); K10/K10b do not apply, K12 does'},
    'DOC-FV-0057': {'roles': '`Uᵢᵀ Vᵢ = 0` → `Dᵢᵀ Vᵢ = 0` (an inline gauge relation in prose)'},
    'DOC-FV-0080': {'roles': '`Uᵢ` (up / Tucker) → `Dᵢ` (down / Tucker); `Oᵢ` (down / outer) → `Uᵢ` (up); L, R stay; the rank caps use the T3\'s nᵢ, rᵢ (stay)', 'relayout': 'n/a: a markdown table; K12 applies'},
    'DOC-BAS-0040': {'roles': 'frame Tucker core `Uᵢ` → `Dᵢ`; frame TT core `Pᵢ` → `Lᵢ` (D-7); variation core `δUᵢ` → `δDᵢ` (D-12; or `Vᵢ`, the rewriter\'s choice); shapes use the T3\'s nᵢ, rᵢ (stay)', 'relayout': 'n/a: a markdown table; K12 applies'},
    'DOC-W-0069': {'roles': 'the weight families and the legs they act on: `up` | H\'s Tucker (nU) leg → `down` | H\'s Tucker (nD) leg; `down` | V\'s complement (nD) leg → `up` | V\'s complement (nU) leg; rows in physical order (down, up, left, right)', 'relayout': 'n/a: a markdown table; K12 applies'},
    'DOC-UG-0125': {'roles': 'U0–U2 the Tucker store (→ D, both rows); O0–O2 the TT-shaped store (→ U); H, V the variations; leading TABs kept (K10 on raw text; sample batch 3 edit 8)'},
    'DOC-UG-0139': {'roles': 'reordered to (down, up, left, right) with down-orthogonal / up-orthogonal; `:142`\'s "down-orthogonal" on the TT-shaped core was backwards (sample batch 3 edit 8)', 'relayout': 'y: reordered and the names/adjectives grow (K10b)'},
    'T-TFVF-0025': {'roles': 'the test helper\'s shape listing (R1 review MAJOR-1): U_i (the Tucker store) → D_i (nU_i → '
                             'nD_i); this docstring\'s D_i (the TT-shaped store) → U_i (nD_i → nU_i); L_i, R_i middle legs '
                             'nU_i → nD_i; V_i rows nD_i → nU_i; H_i middle leg nU_i → nD_i (D-8); rL/rR stay',
                    'relayout': 'n expected: letters swap at equal lengths (K10 in place)'},
    'DOC-UG-0215': {'roles': 'frame Tucker core `U_i` → `D_i`; variation core `dU_i` → `dD_i` (D-12) or `V_i`; shapes use the T3\'s n_i (stay)'},
}

_TEX_ROLES = {
    'docs/exact_hessian_two_hole.tex': 'T4S letters, per §4.8: the frame\'s U (Tucker store) → D, O → U, P/Q → L/R; θ\'s raw own cores U_i (:103–105 and on) → B_i; BU_i, \\dot{BU}_i (:781–832) → \\hat B_i; DΦ, D²Φ, D_ζP → \\mathrm d; \\Dsh/\\Ush → \\Sh/\\Sh^\\T; n^D_i → n^U_i; the retraction R_X/R_x (:288, 974, 976) → \\mathcal R; PB20\'s R_k^{-1} (:382) stays, marked "(PB20\'s notation)"; \\Lift differs by font; a plain R is the right store',
    'docs/symmetric_probe_derivatives.tex': 'per §4.8: frame U/O/P/Q → D/U/L/R; D^t (:111) → \\mathrm d^t (its P^t is the direction and stays); the T3\'s own \\mathbf U_i → \\mathbf B_i; the plain B of :158–169 (a bilinear map) stays; the corewise sentence :556–559 per §1.2a',
    'docs/shared_t3_math.tex': 'per §4.8: the T3\'s own factors → B_i, B_g (the §2 point algorithms, the corewise section :440–450); in the tangent sections (:318–420, :469–495 with rem:embedding as one range) the frame\'s down store → D_i, D_g, \\dot D; the block B_i := U_i W_i (:279–292) → K_i; :306\'s B_i (the factor) stays; the tied coordinate B (:358, 551–566) → Z; n_{D,i} → n_{U,i}; O_g (a complement basis) stays; R(\\T,\\xi) (:474) stays; Q_i at :363 stays (session ruling)',
    'docs/ttm_t3m_ht_note.tex': 'per §4.8: the Tucker factors ("leaf frames") U_i → B_i; operand labels A, B, C (superscripts) → X, Y, Z; the HT transfer tensors B_t (:217–240) → T_t; the macros \\A, \\B, \\C → \\X, \\Y, \\Z; \\C keeps one meaning (eq:t3)',
    'docs/padded_frame_cone.tex': 'per §6/D-5: Table 1, the frame letters (U Tucker store → D, O → U), widths n_{U,i} → n_{D,i}, n_{D,i} → n_{U,i}; D products/messages → Ψ; U_{x→t}, U′ → F_{x→t}, F′; R(\\theta) (:217–219) is a residual function and stays; the matrix-case proof\'s SVD U may stay',
    'docs/pad_safe_svd.tex': 'SVD / QR factors (U Σ Vᵀ, Q R) and the padding projector: §1.5 item 2, they stay; the note\'s only planned change is prose ("the down step" → "the TT up-orthogonalization", ×2, §6)',
}

PREAMBLE = """# Diagram / equation inventory (store rename, R1; plan §4A)

Built by `dev/store_rename_slices/diagram_scanner.py` with the hand curation in `diagram_curation.py`; every
candidate line and its disposition is in `diagram_candidates.tsv`, the fence list the scripts consume is
`fences.tsv`. Old spans are of `v2026.4.0` (`docs/glossary.md`: of `88fbbedb`, absent at the tag).

**Totals:** %(n_cands)d candidate lines: %(n_in)d inside blocks, %(n_dis)d dismissed with a reason. **%(n_blocks)d blocks**
(%(groups)s; %(kinds)s). **K16: %(n_k16)d LaTeX width lines.**

How to use it (R4a/R4b/R5): each block is rewritten by hand, as a whole; fill `new span` and `new text` (verbatim),
and for a block whose new text equals its old text record the reason. K10 checks in-place blocks, K10b re-laid-out
ones, K11 that no script touched a fenced span, K12 that every block is filled
(`dev/store_rename_diagram_check.py`). Fenced `n` (the companion doctest) is inventoried for the reviewer but
renamed by R2's NAME pass (letter_sites rows). Kinds: ascii diagram, index equation, shape listing,
field-comment relation, markdown table (an added kind: symbol/shape tables in the docs; not column-aligned, so
K10/K10b do not apply), companion doctest, LaTeX equation line (one LaTeX display environment, or one paragraph run
of consecutive letter-bearing lines), LaTeX table or figure.
"""

BLOCK_NOTES_BY_FILE = {f: {'roles': r} for f, r in _TEX_ROLES.items()}


_PREFIX = {
    'FVF': 't3toolbox/frame_variations_format.py', 'UFVF': 't3toolbox/uniform_frame_variations_format.py',
    'P': 't3toolbox/backend/probing.py', 'A': 't3toolbox/backend/apply.py', 'UFO': 't3toolbox/backend/ufv_operations.py',
    'UTO': 't3toolbox/backend/utv_operations.py', 'FO': 't3toolbox/backend/fv_operations.py',
    'PF': 't3toolbox/backend/padded_frames.py', 'UPF': 't3toolbox/backend/uniform_padded_frames.py',
    'R': 't3toolbox/backend/ranks.py', 'M': 't3toolbox/manifold.py', 'UM': 't3toolbox/uniform_manifold.py',
    'TO': 't3toolbox/backend/tv_operations.py', 'T3O': 't3toolbox/backend/t3_orthogonalization.py',
    'TTT': 't3toolbox/tucker_tensor_train.py', 'DOC-G': 'docs/glossary.md', 'DOC-FV': 'docs/frame_variations.md',
    'DOC-BAS': 'docs/batching_and_stacking.md', 'DOC-W': 'docs/weighting.md', 'DOC-UG': 'docs/user_guide.rst',
    'T-TFVF': 'tests/test_frame_variations_format.py',
}
# keyed by (file, first line), so the ids the scanner assigns are free to change
BLOCK_NOTES = {(_PREFIX[k.rsplit('-', 1)[0]], int(k.rsplit('-', 1)[1])): v for k, v in BLOCK_NOTES.items()}
