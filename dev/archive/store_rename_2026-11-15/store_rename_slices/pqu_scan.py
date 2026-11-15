"""R1, plan §3.4: the O/P/Q/U-family inventory and the letter-site list.

Usage::

    python dev/store_rename_slices/pqu_scan.py [--scratch DIR]

Writes ``dev/store_rename_pqu_inventory.md``, ``dev/store_rename_letter_sites.md``,
``dev/store_rename_slices/pqu_keeps.tsv`` and ``dev/store_rename_slices/pqu_text_keeps.tsv``.

* **NAME scan** (code and doctest examples), driven through ``store_rename_apply.Pass1`` so file/line/col positions
  equal the hit table's: every NAME token in ``CANDIDATES`` (the reviewed include list of tokens with a ``U``, ``O``,
  ``P`` or ``Q`` letter segment; every other token holding one of those capitals is listed in the inventory's
  "excluded tokens" appendix with its class). A token the dry run already maps is marked with its rule. Every other
  occurrence must match an entry of ``CLASSIFY`` (ordered, first match wins) or the run fails.
* **Text scan** (comments, docstring prose, docs, the six notes): the §3.4 text families (``TEXT_PATTERNS``),
  classified by ``classify_text``.
* **Slices.** ``R2`` (applied by the script from ``letter_sites.md``): rows backed by an inventory row (an inv 1 §7
  or App. A CORE-LETTER line holding the token, or an inv 2 §8 CORE-LETTER location), sites the plan's text
  enumerates with a disposition, frame cores ``P``/``Q`` -> ``L``/``R`` (§1.2 tier C's last row: "every §3.4 hit
  classified as a frame core"), and completions of a binding a tier-C row renames on only some of its lines.
  ``R2-proposed`` (loaded only after the reviewer agrees, §3.4 "R1 gate"): every other rename the classification
  gives, i.e. a ``B`` or ``D`` row (or its simultaneous ``O -> U`` partner) beyond tier C's lists. ``ASK``: not
  settled by the plan; not applied.
"""
import argparse
import collections
import csv
import io
import os
import re
import subprocess
import sys
import tokenize
import typing as typ

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, 'dev'))
sys.path.insert(0, HERE)
import store_rename_apply as sra          # noqa: E402
import r1_reconcile as rr                 # noqa: E402

OUT_INV = os.path.join(REPO, 'dev', 'store_rename_pqu_inventory.md')
OUT_SITES = os.path.join(REPO, 'dev', 'store_rename_letter_sites.md')
OUT_KEEPS = os.path.join(HERE, 'pqu_keeps.tsv')
OUT_TEXT_KEEPS = os.path.join(HERE, 'pqu_text_keeps.tsv')

# ===================================================================================================================
# The reviewed include list: NAME tokens with a U, O, P or Q letter segment (R1, from a full token census)
# ===================================================================================================================
CANDIDATES_U = set('''
BU BU1 BUd DU DUs GU KU PU PhiU U U0 U1 U2 UB UF UF0 UF1 UF_np UFb UL UM_good UM_k UM_t UR UU UU_good UV UVS UW
U_Cix U_M U_T U_a_x U_a_x_b U_i_a_x U_i_j_x U_i_o U_i_x U_i_x_j U_ia_x U_mk U_new U_nk U_o_x U_r U_rows U_u U_x_i
Ua Ub Uc Ud Udot Udot_g Udot_pm Ug Ui Uio Uj Um Uo Uox Ur Ur0 Us Usc Ut Utilde Utilde_full Uu Uw Ux Uxi Uy Uy_rev
Uyi VUt WU XiU XiUs _UCLS _UKIND dU dU0 dU_sum dU_t dU_tilde dU_tildes dU_v gU gUk gYU iU kU left_UU mU mU0 mU1
moves_U new_U new_Uxo o_U oU old_U pU pinU right_UU rowU selU svd_U wU
'''.split())
CANDIDATES_O = set('''
O O0 O1 OO O_P O_ab_x Oaxb Oc Od Odxi On Ou PhiO P_O XiO bO dOn dmO mO moves_O mvO new_O pinO real_O rowO seeds_O
selO OP SO
'''.split())
CANDIDATES_PQ = set('''
P PP PQ P_ P_L P_R P_cores P_cores_dot P_dense P_dot P_eps P_i P_old P_ours P_r P_real Pa Pc Pd PhiP PhiPs Pp
Pp_dense Pq Pr Pr_new Pr_nm Pri Ps PtA_Cab Pxi Q Q0_i_o Q1 Q_Cnx Q_dot Q_nk Q_nx Q_x_o Qc Qd Qp_nx Qt Qxi R_P L_P
H_P HP N_P VP dPxi rev_Q
'''.split())
CANDIDATES = CANDIDATES_U | CANDIDATES_O | CANDIDATES_PQ
# D-family tokens, scanned only inside a scope where a U -> D rename needs its simultaneous partner
D_PARTNERS = set('D D0 D1 D2 D3 DD'.split())


def excluded_class(tok: str) -> str:
    """Why a token holding U/O/P/Q is not a letter-segment candidate (the reviewed exclusion classes)."""
    if tok in sra.A2_SWAP or tok in sra.A1_RENAMED or tok in sra.A3_EXPLICIT or sra.A3_RE.match(tok):
        return 'tier A (store word / width), not a letter'
    if re.match(r'^(?:Test|Uniform|UT3|UFrame|UVariations|_Uniform|_Underdetermined)', tok):
        return 'CamelCase class / test-class name (U = uniform or an English word)'
    if re.match(r'^_?[A-Z][a-z]+[A-Z]', tok) or re.match(r'^_?(?:Union|Optional|Problem|Protocol|Path|Point|'
                                                         r'Parameter|PartitionSpec|StringIO|Omega)$', tok):
        return 'CamelCase word (typing / class / English)'
    if tok in ('NO', 'NP', '_EPS64'):
        return 'size/number constant (NO = output size, NP = number of probes, _EPS64 = machine eps)'
    if tok in ('setUp', 'setUpClass', 'skipUnless'):
        return 'unittest method name'
    if re.match(r'^_?[A-Z0-9]+(?:_[A-Za-z0-9]+)*$', tok) and any(len(s) >= 3 and s.isupper() and s.isalpha()
                                                                 for s in tok.strip('_').split('_')):
        return 'ALL-CAPS constant with an English word'
    if tok.startswith('test_'):
        return 'test method name'
    return 'UNREVIEWED'


# ===================================================================================================================
# The classification (ordered; first match wins). One entry: (file, function, tokens, lines, cls, images, basis, why)
#   function: the innermost def/class name holding the line ('*' any); tokens: set or '*'; lines: None or set
#   cls: LR (frame core P/Q -> L/R) | STORE (a frame's down store -> D, up store -> U) | DFORM | B | BFORM |
#        H (tv_to_t3's misnamed TT variation, R4a hand) | KEEP:<§1.5 item or reason> | ASK
#   basis: inv (R2 when an occurrence of the entry's tokens in that function sits on an inventory CL line, else
#        R2-proposed) | plan:<cite> (R2) | done:<cite> (R2: completes a tier-C row) | scan (R2-proposed)
# ===================================================================================================================
B_ = 't3toolbox/backend/'
F_ = 't3toolbox/'
T_ = 'tests/'
TB_ = 'tests/backend/'
ALL = '*'


def S(s: str) -> typ.FrozenSet[str]:
    return frozenset(s.split())


SWAP_UD = {'U': 'D', 'O': 'U'}
# The R1 reviewer agreed to every R2-proposed row (dev/store_rename_slices/R1_review.md, "The 80 R2-proposed rows":
# "Agreed, all 80 ... Flip them to R2"), per §3.4's "only with the reviewer's agreement". Set to None to list them as
# R2-proposed again.
REVIEWER_AGREED = 'R1 review 2026-11-11'
CLASSIFY = [
    # ------------------------------------------------------------------ library: kept letters
    (B_ + 'linalg.py', ALL, ALL, None, 'KEEP:2,3', None, '', 'SVD/QR/eig factors (U, Um, U_*, Uw, Ur, Ur0, U0, '
     'U_T, U_nk, U_mk, Q, Qp_nx, Q_nx, Q1, Q_nk, Q_Cnx, U_Cix) and the pinned completion P, Ps, PQ, PU, PtA_Cab, '
     'P_real (§1.5 items 2, 3)'),
    ('examples/', ALL, S('N_P'), None, 'KEEP:other', None, '', 'N_P is a count (number of probes), no letter'),
    (B_ + 'fv_operations.py', '_haar_rows', S('Q'), None, 'KEEP:2', None, '', 'QR factor'),
    (B_ + 'padded_frames.py', ALL, S('PhiPs PhiP P VP H_P P_old R_P L_P'), None, 'KEEP:3', None, '',
     'the completion P and its products (§1.5 item 3: only a store suffix changes, P_O->P_U, O_P->U_P)'),
    (B_ + 'uniform_padded_frames.py', ALL, S('P P_L P_R PhiP VP H_P R_P L_P'), None, 'KEEP:3', None, '',
     'the completion P and its products (§1.5 item 3)'),
    (B_ + 'uniform_padded_frames.py', ALL, S('Us'), None, 'KEEP:2', None, '', 'SVD factor (§1.5 item 2 names Us)'),
    (B_ + 'sharing.py', ALL, S('svd_U U_M'), None, 'KEEP:2', None, '', 'SVD factors of the shared frame data'),
    (B_ + 't3_linalg.py', ALL, S('Uw'), None, 'KEEP:11', None, '', 'SVD factor (§1.5 item 11)'),
    (B_ + 't3_linalg.py', '_t3m_swap_pair', S('U'), None, 'KEEP:11', None, '', 'SVD factor (§1.5 item 11)'),
    (B_ + 't3_linalg.py', ALL, S('Qt'), None, 'KEEP:2', None, '', 'SVD left factor named Q (Qt, st, Vt2 = svd)'),
    (B_ + 't3_linalg.py', ALL, S('P'), None, 'KEEP:other', None, '', 'P = nA * nB, a product dimension'),
    (B_ + 't3_orthogonalization.py', ALL, S('Uox U_i_a_x U_a_x U_a_x_b'), None, 'KEEP:11', None, '',
     'SVD factors (§1.5 item 11)'),
    (B_ + 't3_orthogonalization.py', ALL, S('Q0_i_o Q_x_o'), None, 'KEEP:2', None, '', 'QR factors'),
    (B_ + 't3_svd.py', ALL, S('U'), None, 'KEEP:2', None, '', 'SVD factor (dense reference SVDs)'),
    (B_ + 't3_svd.py', ALL, S('P_i'), None, 'KEEP:other', None, '',
     'P_i = W_block diag(s): the factor with B_i = P_i Vt (its own comment); no store'),
    (B_ + 'tt_orthogonalization.py', ALL, S('U'), None, 'KEEP:2', None, '', 'SVD factor'),
    (B_ + 'ut3_orthogonalization.py', ALL, S('U_o_x'), None, 'KEEP:11', None, '', 'SVD factor (§1.5 item 11)'),
    (B_ + 'ut3_svd.py', ALL, S('U'), None, 'KEEP:2', None, '', 'SVD factor'),
    (B_ + 'tv_operations.py', 'tv_to_dense', S('P'), None, 'KEEP:other', None, '',
     'P = the dense base point t3_to_dense((up_tucker_cores, left_tt_cores)); not a frame core'),
    (B_ + 'ufv_operations.py', ALL, S('P'), None, 'KEEP:3', None, '', 'a pseudo-inverse (tucker_pinvs/bond_pinvs)'),
    (F_ + 'fitting.py', ALL, S('Pp'), None, 'KEEP:other', None, '', 'Pp = the projected direction (Pi p)'),
    (F_ + 'tucker_tensor_train.py', 'share', S('Q'), None, 'KEEP:2', None, '', 'QR factor (doctest)'),
    (F_ + 'uniform_frame_variations_format.py', ALL, S('UF'), None, 'KEEP:12', None, '', 'U = uniform (a UT3 frame)'),
    # ------------------------------------------------------------------ library: renames
    (B_ + 't3_orthogonalization.py', '_up_orthogonalize_tt_cores_step', S('O_ab_x Oaxb'), None, 'STORE',
     {'O_ab_x': 'U_ab_x', 'Oaxb': 'Uaxb'}, 'plan:§3.4 (O family: both kernels produce the up-orthogonal core)',
     'the up-orthogonal TT core the frame takes as its up store'),
    (B_ + 'ut3_orthogonalization.py', 'up_orthogonalize_tt_supercores', S('O_ab_x'), None, 'STORE',
     {'O_ab_x': 'U_ab_x'}, 'plan:§3.4 (O family)', 'the up-orthogonal TT supercore (frame up store)'),
    (B_ + 'tv_operations.py', 'tv_to_t3', S('dU'), None, 'H', {'dU': 'H'}, 'plan:§3.3 / tier C exception',
     'dU = tt_variations[i] is a TT variation: H, an R4a hand row (not R2)'),
    (B_ + 'sharing.py', 'fv_share_tucker_variations', S('O U'), {619, 620}, 'STORE', SWAP_UD,
     'scan', 'doctest raw variation shapes over zip(frame_d[1], frame_d[0]): up store O -> U, down store U -> D '
     '(plan §3.4 lists sharing.py:620 for classification)'),
    (B_ + 'apply.py', 't3_apply_corewise_hessian', S('KU'), None, 'BFORM', {'KU': 'KB'}, 'scan',
     'the corewise Hessian applied to dirs: the Tucker block of theta\'s own cores (D-12: Hessian blocks take B)'),
    (B_ + 'entries.py', 't3_entries_corewise_hessian', S('KU'), None, 'BFORM', {'KU': 'KB'}, 'scan',
     'corewise Hessian Tucker block (D-12)'),
    (B_ + 'probing.py', 't3_probe_corewise_hessian', S('KU'), None, 'BFORM', {'KU': 'KB'}, 'scan',
     'corewise Hessian Tucker block (D-12)'),
    (B_ + 'sampling_derivatives.py', ALL, S('KU'), None, 'BFORM', {'KU': 'KB'}, 'scan',
     'corewise derivative-Hessian Tucker block (D-12)'),
    (B_ + 'ut3_sampling.py', ALL, S('KU'), None, 'BFORM', {'KU': 'KB'}, 'scan', 'corewise Hessian Tucker block'),
    (B_ + 'ut3_linalg.py', ALL, S('KU U'), None, 'BFORM', {'KU': 'KB', 'U': 'B'}, 'scan',
     'corewise inner-product Hessian Tucker block, and the comprehension variable over its ragged blocks '
     '(plan §3.4 lists ut3_linalg.py:417)'),
    (F_ + 'uniform_tucker_tensor_train.py', ALL, S('kU'), None, 'BFORM', {'kU': 'kB'}, 'scan',
     'the keep-W corewise gradient stack of x\'s own Tucker cores, beside the plan\'s gU -> gB (566-567)'),
    (F_ + 'frame_variations_format.py', 'fv_to_t3', S('U0 U1 U2 D0 D1 D2'), None, 'STORE',
     {'U0': 'D0', 'U1': 'D1', 'U2': 'D2', 'D0': 'U0', 'D1': 'U1', 'D2': 'U2'}, 'inv',
     'companion doctest of the fv_to_t3 diagram (1262-1274): Tucker frame cores U* -> D*, the TT store D* -> U*'),
    (F_ + 'frame_variations_format.py', 'shape', S('U'), {167}, 'STORE', {'U': 'D'}, 'scan',
     'comprehension over self.up_tucker_cores (A1: down_tucker_cores): the down store'),
    (F_ + 'frame_variations_format.py', 'up_ranks', S('U'), {171}, 'STORE', {'U': 'D'}, 'scan',
     'comprehension over the down store (sample edit 2 flagged it ask; plan §3.4 lists :171)'),
    (F_ + 'frame_variations_format.py', 'is_orthogonal', S('U'), {383}, 'STORE', {'U': 'D'}, 'scan',
     'doctest comprehension over good.up_tucker_cores: the down store'),
    (F_ + 'frame_variations_format.py', 'shape', S('U'), {821}, 'VAR', {'U': 'V'}, 'plan:session ruling 2026-11-11 (3)',
     '(was ASK) comprehension over self.tucker_variations: a Tucker variation, whose house letter is V; '
     '"U" would read as the up store after the rename (§0 item 11)'),
    (F_ + 'frame_variations_format.py', 't3svd_orthogonal_representations', S('U Ux'), {1609}, 'STORE',
     {'U': 'D', 'Ux': 'Bx'}, 'scan',
     'doctest: U over frame.up_tucker_cores (down store -> D); Ux over xs.tucker_cores (x\'s own -> Bx)'),
    (F_ + 'uniform_frame_variations_format.py', '_frame_of', S('U O'), None, 'STORE', SWAP_UD, 'inv',
     'U, O, L, R = frame_data: the frame stores'),
    # ------------------------------------------------------------------ tests: kept letters
    (TB_ + '_padded_frame_oracles.py', ALL, S('Uc Uy'), None, 'KEEP:2', None, '',
     'SVD factors (plan names _padded_frame_oracles.py:54 Uc and Uy)'),
    (TB_ + '_padded_frame_oracles.py', ALL, S('Q P_old'), None, 'KEEP:2', None, '',
     'orthonormal bases (complement / SVD U factor) and the old basis; not frame cores'),
    (TB_ + 'test_curvature.py', ALL, S('U_M svd_U'), None, 'KEEP:2', None, '', 'SVD factors'),
    (TB_ + 'test_curvature.py', ALL, S('P P_eps'), {215, 216, 217, 218, 219, 220, 353, 355, 442, 443, 444, 446, 450,
                                                    451}, 'KEEP:3', None, '',
     'a pseudo-inverse (§1.5 item 3 names test_curvature.py:215-220, 353)'),
    (TB_ + 'test_curvature.py', ALL, S('P HP'), None, 'KEEP:3', None, '', 'P = a tangent direction, HP its Hessian '
     'image (a direction, not a frame core)'),
    (TB_ + 'test_linalg.py', ALL, ALL, None, 'KEEP:2,3', None, '', 'SVD/QR factors and the pinned completion P'),
    (TB_ + 'test_optimizers.py', ALL, S('Pp'), None, 'KEEP:other', None, '', 'Pp = the projected direction'),
    (TB_ + 'test_padded_frame.py', ALL, S('P Q'), None, 'KEEP:2,3', None, '', 'the completion P and QR Q'),
    (TB_ + 'test_padded_start.py', ALL, S('Q'), None, 'KEEP:2', None, '', 'QR / SVD bases'),
    (TB_ + 'test_structural_guards.py', ALL, S('U'), None, 'KEEP:11', None, '', 'SVD factor (§1.5 item 11)'),
    (TB_ + 'test_t3_orthogonalization.py', ALL, S('O'), None, 'KEEP:13', None, '', 'the oracle module alias O'),
    (TB_ + 'test_uniform_curvature.py', ALL, S('_UKIND _UCLS'), None, 'KEEP:12', None, '', 'U = uniform'),
    (TB_ + 'test_uniform_curvature.py', '_dirty_companion', S('U svd_U'), None, 'KEEP:2', None, '', 'SVD factor'),
    (TB_ + 'test_uniform_curvature.py', ALL, S('P U'), {316, 317, 318}, 'KEEP:3,12', None, '',
     'P a ragged pseudo-inverse, U its uniform twin (U = uniform)'),
    (TB_ + 'test_uniform_curvature.py', ALL, S('P HP P_r PP'), None, 'KEEP:3', None, '', 'a tangent direction P'),
    (TB_ + 'test_uniform_fitting.py', ALL, S('O'), None, 'KEEP:13', None, '', 'an order count O'),
    (TB_ + 'test_zero_reference.py', ALL, S('UF0'), None, 'KEEP:12', None, '', 'U = uniform (a uniform frame)'),
    (T_ + 'oracle_sweeps/o1_ragged.py', ALL, S('Ud P Pd'), None, 'KEEP:other', None, '',
     'Ud = "u dense" (plan); P the projected tangent, Pd its dense'),
    (T_ + 'oracle_sweeps/o1_uniform.py', ALL, S('U'), None, 'KEEP:12', None, '',
     'U a uniform T3 (§1.5 item 12 names o1_uniform.py:193)'),
    (T_ + 'oracle_sweeps/o1_uniform.py', ALL, S('Ud P Pd'), None, 'KEEP:other', None, '',
     'Ud = "u dense" (plan §1.2a); P the projected tangent, Pd its dense'),
    (T_ + 'oracle_sweeps/sweep_exact.py', ALL, S('O'), None, 'KEEP:13', None, '', 'the oracle module alias O'),
    (T_ + 'oracle_sweeps/sweep_exact.py', ALL, S('Uj Q'), None, 'KEEP:2', None, '', 'SVD factor and its basis'),
    (T_ + 'oracle_sweeps/sweep_models.py', ALL, S('O'), None, 'KEEP:13', None, '', 'the oracle module alias O'),
    (T_ + 'oracle_sweeps/sweep_models.py', ALL, S('Pp Pq'), None, 'KEEP:other', None, '', 'projected directions'),
    (T_ + 'oracle_sweeps/sweep_optimizers.py', ALL, S('O'), None, 'KEEP:13', None, '', 'the oracle module alias O'),
    (T_ + 'test_aligned_padded_frame_frontend.py', ALL, S('UF UF1 UFb'), None, 'KEEP:12', None, '', 'U = uniform'),
    (T_ + 'test_contractions_sharding.py', ALL, S('P'), None, 'KEEP:other', None, '', 'PartitionSpec as P'),
    (T_ + 'test_dispatch.py', ALL, S('U'), {149, 150, 151}, 'KEEP:11', None, '', 'SVD factor (§1.5 item 11)'),
    (T_ + 'test_dispatch.py', ALL, S('P Q PQ P_'), None, 'KEEP:2,3', None, '',
     'the pinned completion P and QR Q (§1.5 item 2 names test_dispatch.py:163, 2095)'),
    (T_ + 'test_dispatch.py', ALL, S('UB UV UW UVS UF UF_np UF1 Usc'), None, 'KEEP:12', None, '',
     'U = uniform (§1.5 item 12 names test_dispatch.py:1705-1708)'),
    (T_ + 'test_fitting.py', ALL, S('Pp Pp_dense'), None, 'KEEP:other', None, '', 'the projected direction'),
    (T_ + 'test_manifold.py', ALL, S('Pr Pri Pr_nm Pr_new'), None, 'KEEP:other', None, '', 'dense projectors'),
    (T_ + 'test_sharing.py', ALL, S('P_ours P_dense P'), None, 'KEEP:other', None, '', 'projectors'),
    (T_ + 'test_sharing.py', ALL, S('Q'), None, 'KEEP:2', None, '', 'QR factor'),
    (T_ + 'test_sharing.py', ALL, S('svd_U'), None, 'KEEP:2', None, '', 'SVD factor (§1.5 item 11 names :478)'),
    (T_ + 'test_sharing.py', ALL, S('GU'), None, 'KEEP:12', None, '', 'the shared uniform geometry (U = uniform)'),
    (T_ + 'test_t3m.py', ALL, S('P Pa Pc'), None, 'KEEP:other', None, '', 'P = the product T3 A.t3m(B)'),
    (T_ + 'test_tucker_tensor_train.py', ALL, S('OP'), None, 'KEEP:other', None, '', 'subTest keyword "operation"'),
    (T_ + 'test_uniform_frame_variations_format.py', ALL, S('UB UV'), None, 'KEEP:12', None, '',
     'U = uniform (§1.5 item 12 names :398)'),
    (T_ + 'test_uniform_probing.py', ALL, S('O'), None, 'KEEP:13', None, '',
     'an order count O (§1.5 item 13 names test_uniform_probing.py:213-219; 826, 898 are the same)'),
    # ------------------------------------------------------------------ tests: renames
    (TB_ + '_curvature_oracles.py', 'adjoint_sweep_tangent_trs', S('P Pd'), None, 'LR', {'P': 'L', 'Pd': 'Ld'},
     'plan:§3.4 sites known now (_curvature_oracles.py\'s P, Pd)',
     'the dotted twin of twins._adj_sweep, whose core P is the left frame core (tier C maps the twin P -> L)'),
    (TB_ + '_curvature_oracles.py', 'tt_tangent_assembly_factored_trs', S('P Pd Q Qd'), None, 'KEEP:14',
     None, '',
     'session ruling 2026-11-11 (4): kept, with DNT rows. P = M xs and Q = N xs + M deta~ are factored products with index layout '
     '(t, W, C, a, i), not frame cores (rL, n, rR); §3.4\'s "sites known now" over-generalized this file'),
    (TB_ + 'test_curvature.py', '_raw_like_frame_cores', S('U'), None, 'STORE', {'U': 'D'}, 'inv',
     'U, _, L, _ = frame: theta = (D, L) by the frame stores it is built from (§3.3)'),
    (TB_ + 'test_curvature.py', 'test_factors_match_the_construction_and_the_dense_spectra', S('U O'), None,
     'STORE', SWAP_UD, 'inv', 'U, O, L, R = frame'),
    (TB_ + 'test_curvature.py', 'test_factors_match_the_construction_and_the_dense_spectra', S('VUt'), None,
     'STORE', {'VUt': 'VDt'}, 'scan', 'VUt = V_i U_i^T with U the frame\'s down store (S_i^T = V_i D_i^T)'),
    (TB_ + 'test_curvature.py', 'test_lift_identity', S('U'), None, 'STORE', {'U': 'D'},
     'plan:§1.2a ("its U over frame[0] is the frame\'s store, -> D")', 'U over frame[0]'),
    (TB_ + 'test_curvature.py', '_directions', S('U'), {483}, 'B', {'U': 'B'}, 'done:tier C B row test_curvature.py:482',
     'the comprehension\'s U is the binding :482 renames; without it R2 raises NameError'),
    (TB_ + 'test_curvature.py', 'test_probe_pieces_against_finite_differences_of_the_gn_kernels', S('U'), {1020}, 'B',
     {'U': 'B'}, 'done:tier C B row test_curvature.py:999', 'U bound at :999 (U -> B); its use at :1020'),
    (TB_ + 'test_curvature.py', 'test_apply_tt_assembly_is_the_three_term_product_rule', S('U'), {1059}, 'B',
     {'U': 'B'}, 'done:tier C B row test_curvature.py:1058', 'U bound at :1058 (U -> B); its use at :1059'),
    (TB_ + 'test_curvature.py', 'state', S('Uc'), None, 'B', {'Uc': 'Bc'}, 'scan',
     'Uc, Gc = cores: theta\'s own raw cores in the nested corewise state'),
    (TB_ + 'test_curvature.py', ALL, S('oU'), None, 'BFORM', {'oU': 'oB'}, 'scan',
     'the corewise (derivative) Hessian\'s Tucker block (D-12)'),
    (TB_ + 'test_curvature.py', 'test_normal_curvature_annihilates_vertical_directions test_gauge_laws '
     'test_normal_curvature_annihilates_a_lifted_vertical_direction '
     '_check '
     'test_the_shared_retraction_is_second_order_here', S('U O'), None, 'STORE', SWAP_UD, 'inv',
     'U, O, L, R(r) = frame: the frame stores'),
    (TB_ + 'test_padded_start.py', 'test_the_exact_model_at_a_rotated_frame', S('U'), None, 'B', {'U': 'B'}, 'scan',
     'q.T @ U over x[0]: a T3\'s own Tucker cores (plan §3.4 lists test_padded_start.py:502)'),
    (TB_ + 'test_ranks.py', '_actual', S('P Q'), None, 'LR', {'P': 'L', 'Q': 'R'}, 'frame-core',
     'P over fr.left_tt_cores, Q over fr.right_tt_cores (beside the tier-C D rows 445-446)'),
    (TB_ + 'test_t3_orthogonalization.py', ALL, S('Oc'), None, 'STORE', {'Oc': 'Uc'}, 'plan:session ruling 2026-11-11 (7)',
     '(was ASK) V, Oc = O.t3_up_orthogonalize_tt_cores(x): the same output as the kernel\'s Oaxb (plan: -> U), but here held '
     'as the point\'s own TT cores ("a point\'s own TT core -> G"); recommend Uc, matching the kernel'),
    (TB_ + 'test_tv_operations.py', 'test_projection_properties', S('U O'), None, 'STORE', SWAP_UD, 'inv',
     'U, O, L, R = fd'),
    (TB_ + 'test_tv_operations.py', '_tv_gauge_residual_b6ad78b', S('U'), None, 'STORE', {'U': 'D'}, 'scan',
     'U over up_tucker_cores (A1: down_tucker_cores): the down store (plan §3.4 lists :151)'),
    (TB_ + 'test_uniform_curvature.py', '_frame_dims', S('U O'), None, 'STORE', SWAP_UD, 'inv',
     'U over f[0], O over f[1]: the frame stores (plan §3.4 lists :99-100)'),
    (TB_ + 'test_uniform_curvature.py', '_random_theta_sc', S('U'), None, 'STORE', {'U': 'D'}, 'scan',
     'U = uframe[0]: the base point (D, L) by the frame stores (§3.3; plan §3.4 lists :249)'),
    (TB_ + 'test_uniform_fitting.py', '_expected_gauge_indicator', S('P Q'), None, 'LR', {'P': 'L', 'Q': 'R'},
     'plan:session ruling 2026-11-11 (7)', '(was ASK) P/Q over the left/right frame cores (beside the tier-C D rows 358-359); the image R is also '
     'bound in the nested fam (R = np.asarray(...)), a local of a nested def holding no renamed image, so the '
     'freeness check finds no clash (its nested-def exception, implementer call 8, reviewer-agreed); L/R by the '
     'frame-core rule (approved by the session)'),
    (T_ + 'test_dispatch.py', 'test_jit_probe_derivatives', S('U'), None, 'STORE', {'U': 'D'}, 'scan',
     'the eta-times-core operand of a probing kernel string (Cio): the frame\'s down core (plan §3.4 lists :337)'),
    (T_ + 'test_dispatch.py', 'test_jit_probe_derivatives', S('Qc'), None, 'LR', {'Qc': 'Rc'}, 'frame-core',
     'the right-core operand (Caib) of a sampling kernel string'),
    (T_ + 'test_frame_variations_format.py', '_random_frame_variations _good_frame_cores '
     'test_t3frame_validate_raises test_t3frame_validate_accepts_good test_fv_to_t3 '
     'test_orthogonal_representations_frame_orthogonality', S('U D'), None, 'STORE', {'U': 'D', 'D': 'U'}, 'inv',
     'U, D, L, R with D the old TT store: a simultaneous swap'),
    (T_ + 'test_frame_variations_format.py', 'test_stack_unstack', S('U'), None, 'STORE', {'U': 'D'}, 'scan',
     'comprehension over frame.up_tucker_cores'),
    (T_ + 'test_manifold.py', 'test_to_dense_matches_explicit_sum', S('U0 U1 U2 D0 D1 D2'), None, 'STORE',
     {'U0': 'D0', 'U1': 'D1', 'U2': 'D2', 'D0': 'U0', 'D1': 'U1', 'D2': 'U2'}, 'inv',
     'U* from up_tucker_cores, D* from down_tt_cores: a simultaneous swap'),
    (T_ + 'test_sharing.py', '_dense_tied_projector', S('UU OO'), None, 'STORE', {'UU': 'DD', 'OO': 'UU'},
     'plan:tier C last row ("test_sharing.py (its frame UU, OO at 576-598)")', 'the frame stores'),
    (T_ + 'test_sharing.py', ALL, S('OO SO'), None, 'STORE', {'OO': 'UU', 'SO': 'SU'}, 'scan',
     'OO = frame_d[1]: the up store; SO = S OO[ii] (the S-absorbed up core, the comment at :496 "S_i-absorbed O_i")'),
    (T_ + 'test_uniform_frame_variations_format.py', 'test_frame_on_non_minimal_points', S('P Q'), None, 'LR',
     {'P': 'L', 'Q': 'R'}, 'frame-core', 'P/Q over the left/right frame cores (beside the tier-C D row 864)'),
    (T_ + 'test_uniform_frame_variations_format.py', 'test_frame_gauge_equivalent_to_ragged', S('U_r U_u'), {960}, 'STORE',
     {'U_r': 'D_r', 'U_u': 'D_u'}, 'done:tier C row (958-959) U_r->D_r, U_u->D_u',
     'the bindings 958-959 rename; their use at :960 (inv 1 CL row); without it R2 raises NameError'),
    (T_ + 'test_uniform_probing.py', '_ragged_frame_ranks', S('P Q'), None, 'LR', {'P': 'L', 'Q': 'R'}, 'frame-core',
     'P/Q over the left/right frame cores (beside the tier-C D row 742)'),
]
for _e in CLASSIFY:
    assert len(_e) == 8, _e


def _func_match(spec, func_last: str) -> bool:
    if spec == ALL:
        return True
    return func_last in spec.split()


def classify_name(path: str, func_last: str, tok: str, line: int):
    for e in CLASSIFY:
        f, fn, toks, lines, cls, imgs, basis, why = e
        if not (path == f or (f.endswith('/') and path.startswith(f))):
            continue
        if not _func_match(fn, func_last):
            continue
        if toks != ALL and tok not in toks:
            continue
        if lines is not None and line not in lines:
            continue
        return e
    return None


# ===================================================================================================================
# The NAME scan
# ===================================================================================================================

class Rec(sra.Pass1):
    def __init__(self, *a):
        super().__init__(*a)
        self.names: typ.List[typ.Tuple[str, int, int, str, typ.Optional[tuple]]] = []

    def name_lookup(self, path, line, col, tok, rules):
        r = super().name_lookup(path, line, col, tok, rules)
        self.names.append((path, line, col, tok, r))
        return r


def string_spans(src: str) -> typ.List[typ.Tuple[typ.Tuple[int, int], typ.Tuple[int, int]]]:
    return [(t.start, t.end) for t in tokenize.generate_tokens(io.StringIO(src).readline)
            if t.type == tokenize.STRING]


def inside(spans, line, col) -> bool:
    return any(a <= (line, col) < b for a, b in spans)


def _def_spans(src: str):
    """Every FunctionDef/Lambda span (first line incl. decorators, end line), and the (line, name) of every
    parameter of a def or lambda nested inside another def (the freeness check's exception)."""
    import ast
    tree = ast.parse(src)
    spans, params = [], set()

    def walk(node, depth):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)):
                first = min([ch.lineno] + [d.lineno for d in ch.decorator_list])
                spans.append((first, ch.end_lineno))
                if depth > 0:
                    for a in ch.args.args + ch.args.kwonlyargs + ch.args.posonlyargs:
                        params.add((a.lineno, a.arg))
                walk(ch, depth + 1)
            elif isinstance(ch, ast.Lambda) and depth > 0:
                for a in ch.args.args:
                    params.add((a.lineno, a.arg))
                walk(ch, depth)
            else:
                walk(ch, depth)
    walk(tree, 0)
    return spans, params


def innermost(scopes: sra.FileScopes, line: int) -> typ.Tuple[int, int, str]:
    cs = [s for s in scopes.spans if s[0] <= line <= s[1]]
    return min(cs, key=lambda s: s[1] - s[0]) if cs else (1, 10 ** 9, '<module>')


def scan_names():
    p = Rec(sra.Entries(), sra.load_fences(), [])      # NO letter sites: the scan sees the tier-A-C map only
    rows = []
    sources = {}
    for path in sra.python_paths():
        src = open(os.path.join(REPO, path), encoding='utf-8').read()
        sources[path] = src
        sc = sra.file_scopes(src)
        ss = string_spans(src)
        defs, nested_params = _def_spans(src)
        n0 = len(p.names)
        p.python_file(path, src)
        for (pa, l, c, t, r) in p.names[n0:]:
            inn = innermost(sc, l)
            outer = [d for d in defs if d[0] <= l <= d[1]]
            outer = max(outer, key=lambda d: d[1] - d[0]) if outer else (inn[0], inn[1])
            rows.append(dict(file=pa, line=l, col=c, tok=t, rule=r, func=inn[2], span=(inn[0], inn[1]),
                             outer=(outer[0], outer[1]), nested_param=(l, t) in nested_params,
                             qual=sra.qualname_at(sc, l), doctest=inside(ss, l, c)))
    return rows, sources


# ===================================================================================================================
# Inventory backing (inv 1 §7 + App. A CORE-LETTER lines; inv 2 §8 CORE-LETTER locations)
# ===================================================================================================================

def inventory_lines() -> typ.Dict[typ.Tuple[str, int], typ.List[tuple]]:
    out = collections.defaultdict(list)
    for f, l, cat, cur, prop, note in rr.inv1_rows():
        tags = set(re.split(r'[,\s]+', cat))
        if 'CL' in tags or 'CLL' in tags or cat == 'APP-A':
            out[(f, l)].append((cat, cur, prop, note))
    return out


def inv2_core_letter() -> typ.List[tuple]:
    return [r for r in rr.inv2_identifier_rows() if r[2] == 'CORE-LETTER']


# ===================================================================================================================
# The text scan (comments, docstring prose, docs, notes)
# ===================================================================================================================

TEXT_PATTERNS = [
    ('O-core', re.compile(r'\bO(?:\d|_[a-z{]|ᵢ|i\b|\(d-1\))|\(mu\^\(\w+\) O\)|\b(?:mu\S*|nu) O\b|\bO (?:nu|sigma|tau)\b'
                          r'|``O``|`O`|\\mathbf\s*\{?O\b|<\s*O\s*,|,\s*O\s*>|,\s*O\s*\)|\(\s*O\s*,|\bO\[|'
                          r'\bO\.shape\b|\bfor\s+O\s+in\b|[=,(]\s*O\s*$|\bO = ')),
    ('PQ-core', re.compile(r'\b[PQ]_(?:i\b|last\b|k\b|\{)|\((?:\s*[A-Z]\w?\s*,)+\s*[PQ]\s*(?:,\s*[A-Z]\w?\s*)*\)|'
                           r'\(\s*[PQ]\s*(?:,\s*[A-Z]\w?\s*)+\)|\b[PQ] = |\b[PQ]d\b|\[Q \| P\]|\b[PQ][- ]block\b|'
                           r'\b[PQ]-part\b|\\mathbf\s*\{?[PQ]\b|\b[PQ],\s*[PQ]\b')),
    ('U-family', re.compile(r'Udot\w*|\bdU\w*|\bgU\w*|\bgYU\b|\bUd\b|\bUg\b|\bU_g\b|U̇|δU|\\dot\s*\{?\s*U|'
                            r'\\delta\s*\{?\s*U|\bBU\w*')),
]
NOTE_U = re.compile(r'(?<![A-Za-z\\])U(?![A-Za-z])')     # every U letter in the notes (not \Ush, not \U... macros)
NOTES = ['docs/exact_hessian_two_hole.tex', 'docs/padded_frame_cone.tex', 'docs/pad_safe_svd.tex',
         'docs/shared_t3_math.tex', 'docs/symmetric_probe_derivatives.tex', 'docs/ttm_t3m_ht_note.tex']
R4B_FILES = {'t3toolbox/backend/probing.py', 't3toolbox/backend/sampling_derivatives.py', 't3toolbox/backend/apply.py',
             't3toolbox/backend/entries.py', 't3toolbox/backend/utv_sampling.py',
             't3toolbox/backend/ut3_sampling.py', 'tests/_jet_trs_twins.py', 'tests/test_jet_recurrence.py',
             'tests/backend/_curvature_oracles.py', 'tests/backend/test_curvature.py',
             'tests/backend/test_uniform_curvature.py', 'tests/test_probe_derivatives.py'}


def text_segments_py(path: str, src: str):
    """(line, col, text, ctx) for every comment and every non-example docstring/string line."""
    for t in tokenize.generate_tokens(io.StringIO(src).readline):
        if t.type == tokenize.COMMENT:
            yield t.start[0], t.start[1], t.string, 'comment'
        elif t.type == tokenize.STRING:
            lines = t.string.split('\n')
            for k, ln in enumerate(lines):
                fl = t.start[0] + k
                c0 = t.start[1] if k == 0 else 0
                s = ln.lstrip()
                if s.startswith('>>>') or s.startswith('... '):
                    i = ln.find('#')
                    if i >= 0:
                        yield fl, c0 + i, ln[i:], 'doctest-comment'
                    continue
                yield fl, c0, ln, 'string'


T4S_LINE = re.compile(r'T4S|Alger et al|arXiv|Section 6\.3|Appendix A|Algorithm \d|eq(?:s|uation)?\.? ?\(\d|'
                      r'Ghattas|Koch, O|Steinlechner|Psenka', re.I)


def classify_text(path: str, line: int, fam: str, tok: str, text: str, fenced: str):
    """(disposition, slice, keep_reason) for one text hit."""
    if re.search(r'paper ↔ code|paper <-> code|Notation map|T4S writes|T4S\'s (?:letters|tuple)|which T4S writes',
                 text):
        return ('T4S correspondence line (§3): the paper\'s letters stay as the correspondence; the code side '
                'takes the new letters', 'R5 hand' if path.startswith(('docs/', 'CLAUDE', 'README')) else 'R4a hand',
                '')
    if fenced:
        return ('§4A block %s: hand rewrite of the whole block' % fenced, 'R4/R5 block', '')
    if path in NOTES:
        sl = 'R5 hand'
    elif path.startswith('docs/') or path in ('README.md', 'CLAUDE.md'):
        sl = 'R5 hand'
    elif path in R4B_FILES:
        sl = 'R4b hand'
    else:
        sl = 'R4a hand'
    # kept meanings first
    if fam == 'O-core':
        k0 = text.find(tok)
        after = text[k0:k0 + 8] if k0 >= 0 else ''
        if re.match(r'O\(', after) and not re.match(r'O\(d?xi', after) and not re.search(r'\bmu O\(', text):
            return ('kept: big-O', 'kept', 'big-O (§1.5 item 13)')
        if re.search(r'Koch, O|Ghattas, O|, O\. ', text):
            return ('kept: author initial', 'kept', 'author initial (§1.5 item 13)')
        if 'O_g' in tok:
            return ('kept: O_g, a complement basis (K8 carve-out)', 'kept', 'O_g complement basis (§4.8)')
        k = text.find(tok)
        nxt = text[k + len(tok):k + len(tok) + 1] if k >= 0 else ''
        if tok in ('O1', 'O2') and nxt not in ('_', 'ᵢ') and re.search(r'oracle|sweep|review|\bO-?[12]\b', text,
                                                                       re.I):
            return ('kept: oracle/review code', 'kept', 'oracle/review code O1/O2 (K8 carve-out)')
        if T4S_LINE.search(text) and re.search(r'T4S (?:writes|uses|letters)|U, O, P, Q', text):
            return ('T4S correspondence/citation: keep the paper letter where cited; else frame up core -> U', sl, '')
        return ('frame up (TT-shaped) core O -> U', sl, '')
    if fam == 'PQ-core':
        if re.search(r'\(X, ?P\)', tok) or re.match(r'P_\{T', tok):
            return ('kept: the derivative-direction pair (X, P) / a projector P_{T}', 'kept',
                    'derivative direction P or a tangent projector (§1.5 item 3)')
        if re.search(r'P-block|A-block|\[A-block', text) and not re.search(r'\[Q', text):
            return ('kept: the padded frame\'s completion P-block', 'kept', 'completion P-block (§1.5 item 3)')
        if re.search(r'\bP\^t\b|perturbation|direction P\b|pinv|pseudo-inverse|P\^\+|P⁺|completion|QR|qr\(', text):
            return ('kept: completion / direction / pseudo-inverse / QR', 'kept',
                    'completion P, derivative direction P, pseudo-inverse or QR Q (§1.5 items 2, 3)')
        if re.search(r'\[Q \| P\]|[QP][- ]block|[PQ]-part', text):
            return ('the doubled TT bond\'s blocks: [R | L] (§1.5 item 3, R4a hand row)', sl, '')
        return ('frame left/right core P/Q -> L/R (D-7)', sl, '')
    # U family
    if path in NOTES:
        return ('note letter: per §4.8 (own core B_i / frame store D_i, its directions \\dot B / \\delta D)', sl, '')
    return ('U family: B forms for a point\'s own Tucker core\'s direction/gradient, D forms for a frame down core\'s '
            'variation (D-12; §1.2a\'s text list)', sl, '')


def scan_text(fences):
    hits = []
    fence_at = collections.defaultdict(list)
    for f in fences:
        if f.fenced:
            for ln in range(f.first, f.last + 1):
                fence_at[(f.file, ln)].append(f)

    def fenced_id(path, line, col, c1):
        for f in fence_at.get((path, line), ()):
            a = f.col_start or 0
            b = f.col_end if f.col_end is not None else 10 ** 9
            if col < b and c1 > a:
                return f.id
        return ''

    for path in sra.python_paths():
        src = open(os.path.join(REPO, path), encoding='utf-8').read()
        for line, col, text, ctx in text_segments_py(path, src):
            for fam, rx in TEXT_PATTERNS:
                for m in rx.finditer(text):
                    hits.append((path, line, col + m.start(), fam, m.group(0).strip(), text.strip(), ctx,
                                 fenced_id(path, line, col + m.start(), col + m.end())))
    docs = sra.doc_paths() + NOTES
    for path in docs:
        for i, ln in enumerate(open(os.path.join(REPO, path), encoding='utf-8').read().split('\n')):
            fams = list(TEXT_PATTERNS)
            if path in NOTES:
                fams = fams + [('U-note', NOTE_U)]
            for fam, rx in fams:
                for m in rx.finditer(ln):
                    hits.append((path, i + 1, m.start(), fam, m.group(0).strip(), ln.strip(), 'doc',
                                 fenced_id(path, i + 1, m.start(), m.end())))
    # one hit per (file, line, col)
    seen, out = set(), []
    for h in hits:
        if (h[0], h[1], h[2]) in seen:
            continue
        seen.add((h[0], h[1], h[2]))
        out.append(h)
    return out


# ===================================================================================================================
# Main
# ===================================================================================================================

def dry_run_hits(scratch: str) -> typ.Dict[typ.Tuple[str, int, int], dict]:
    out = os.path.join(scratch, 'r1out')
    subprocess.run([sys.executable, os.path.join(REPO, 'dev', 'store_rename_apply.py'), '--dry-run', '--out', out,
                    '--scratch', os.path.join(scratch, 'dry')], check=True, stdout=subprocess.DEVNULL)
    with open(os.path.join(out, 'R1_hit_table.tsv')) as fh:
        return {(r['file'], int(r['line']), int(r['col'])): r for r in csv.DictReader(fh, delimiter='\t')
                if not r['rule'].startswith('letter_sites')}      # tiers A-C only (this script writes the sites)


_BINDS_CACHE: typ.Dict[str, object] = {}


def _nested_def_binds(path: str, span: typ.Tuple[int, int], name: str) -> bool:
    """The def spanning ``span`` binds ``name`` itself (a parameter, or a local / generator target): the
    implementer's call 8, the same exception as ``r1_freeness.py`` (reviewer-agreed, R1 review)."""
    import ast
    if path not in _BINDS_CACHE:
        _BINDS_CACHE[path] = ast.parse(open(os.path.join(REPO, path), encoding='utf-8').read())
    for n in ast.walk(_BINDS_CACHE[path]):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) and n.lineno <= span[0] \
                and n.end_lineno >= span[1] and (n.lineno, n.end_lineno) == tuple(span):
            a = n.args
            bound = {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs} | {x.arg for x in (a.vararg, a.kwarg) if x}
            bound |= {m.id for m in ast.walk(n) if isinstance(m, ast.Name) and isinstance(m.ctx, ast.Store)}
            return name in bound
    return False


def freeness(rows, renames) -> typ.List[str]:
    """Every image of a rename unit must be unbound and unread in its innermost function, except where the image
    is itself renamed away there (a simultaneous swap)."""
    errs = []
    by_func = collections.defaultdict(list)
    for r in rows:
        by_func[(r['file'], r['span'])].append(r)
    units = collections.defaultdict(dict)
    for r, new in renames:
        units[(r['file'], r['span'])][r['tok']] = new
    by_outer = collections.defaultdict(list)
    for r in rows:
        by_outer[(r['file'], r['outer'])].append(r)
    outer_of = {(r['file'], r['span']): r['outer'] for r in rows}
    for key, m in units.items():
        renamed_away = set(m)
        okey = (key[0], outer_of[key])
        for new in set(m.values()):
            image_spans = {k[1] for k, mm in units.items() if k[0] == key[0] and new in mm.values()}
            clash = [t for t in by_outer[okey] if t['tok'] == new and t['tok'] not in renamed_away
                     and not t['rule'] and not t['nested_param'] and t['span'] != key[1]
                     and not (t['span'] != okey[1] and t['span'] not in image_spans
                              and _nested_def_binds(key[0], t['span'], new))]   # call 8 (r1_freeness.py)
            if clash:
                errs.append('%s (outer def %d-%d of %d-%d): image %r bound/read in an enclosing or sibling scope at '
                            '%s' % (key[0], okey[1][0], okey[1][1], key[1][0], key[1][1], new,
                                    sorted({t['line'] for t in clash})))
    for key, m in units.items():
        toks = by_func[key]
        renamed_away = set(m)
        for new in set(m.values()):
            clash = [t for t in toks if t['tok'] == new and t['tok'] not in renamed_away and not t['rule']]
            mapped_into = [t for t in toks if t['rule'] and t['rule'][0] == new and t['tok'] not in m]
            if clash or mapped_into:
                errs.append('%s %s:%d-%d: image %r already %s' % (
                    key[0], toks[0]['func'], key[1][0], key[1][1], new,
                    'bound/read at %s' % sorted({t['line'] for t in clash}) if clash else
                    'the image of a tier-C rename at %s' % sorted({t['line'] for t in mapped_into})))
    return errs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--scratch', default=os.path.join(os.environ.get('TMPDIR', '/tmp'), 'pqu_scan'))
    a = ap.parse_args()
    os.makedirs(a.scratch, exist_ok=True)
    rows, sources = scan_names()
    inv_lines = inventory_lines()
    fences = sra.load_fences()

    # ---------------- classify NAME hits
    cand = [r for r in rows if r['tok'] in CANDIDATES]
    excluded = collections.Counter()
    for r in rows:
        if r['tok'] not in CANDIDATES and r['tok'] not in D_PARTNERS and re.search(r'[UOPQ]', r['tok']):
            excluded[(r['tok'], excluded_class(r['tok']))] += 1
    unreviewed = [k for k in excluded if k[1] == 'UNREVIEWED']
    classified, unclassified = [], []
    for r in cand:
        if r['rule']:
            classified.append((r, 'MAPPED', None))
            continue
        e = classify_name(r['file'], r['func'], r['tok'], r['line'])
        (classified if e else unclassified).append((r, e and e[4], e) if e else r)
    # D partners inside swap entries
    for r in rows:
        if r['tok'] in D_PARTNERS and not r['rule']:
            e = classify_name(r['file'], r['func'], r['tok'], r['line'])
            if e and e[5] and r['tok'] in e[5]:
                classified.append((r, e[4], e))
    if unclassified:
        for r in unclassified[:50]:
            print('UNCLASSIFIED', r['file'], r['line'], r['func'], r['tok'])
        raise SystemExit('%d unclassified NAME occurrences' % len(unclassified))
    if unreviewed:
        raise SystemExit('unreviewed exclusions: %s' % unreviewed)

    # ---------------- slices
    def backed(entry, r) -> bool:
        toks = set(entry[5] or ()) | (entry[2] if entry[2] != ALL else set())
        for rr_ in rows:
            if rr_['file'] == r['file'] and rr_['span'] == r['span'] and rr_['tok'] in toks \
                    and (rr_['file'], rr_['line']) in inv_lines:
                return True
        return False

    out_rows = []      # (row, cls, new, slice, basis, why)
    for r, cls, e in classified:
        if cls == 'MAPPED':
            out_rows.append((r, 'MAPPED', r['rule'][0], 'R2 (tier %s: %s)' % (r['rule'][1], r['rule'][2]), '', ''))
            continue
        f, fn, toks, lines, cls, imgs, basis, why = e
        new = (imgs or {}).get(r['tok'])
        if cls.startswith('KEEP'):
            sl = 'kept'
        elif cls == 'ASK':
            sl = 'ASK'
        elif cls == 'H':
            sl = 'R4a hand'
        elif basis.startswith(('plan:', 'done:')) or basis == 'frame-core':
            sl = 'R2'
        elif basis == 'inv':
            sl = 'R2' if backed(e, r) else 'R2-proposed'
        else:
            sl = 'R2-proposed'
        if sl == 'R2-proposed' and REVIEWER_AGREED:
            sl, basis, why = 'R2', 'agreed:' + REVIEWER_AGREED, why + ' [R2-proposed; agreed by the reviewer, %s]' % \
                REVIEWER_AGREED
        out_rows.append((r, cls, new, sl, basis, why))

    renames = [(r, new) for r, cls, new, sl, b, w in out_rows if sl in ('R2', 'R2-proposed') and cls != 'MAPPED']
    free_errs = freeness(rows, renames + [(r, r['rule'][0]) for r in rows if r['rule'] and r['rule'][1].startswith('C')
                                          and False])
    # ---------------- text scan
    text_hits = scan_text(fences)
    text_rows = []
    for (path, line, col, fam, tok, text, ctx, fid) in text_hits:
        disp, sl, keep = classify_text(path, line, fam, tok, text, fid)
        text_rows.append((path, line, col, fam, tok, text, ctx, fid, disp, sl, keep))

    # ---------------- dry run join (sanity: MAPPED == hit table)
    hits = dry_run_hits(a.scratch)
    mismatch = [r for r, cls, new, sl, b, w in out_rows if cls == 'MAPPED'
                and (r['file'], r['line'], r['col']) not in hits]

    write_inventory(out_rows, text_rows, excluded, free_errs, mismatch, inv_lines)
    write_sites(out_rows, rows, inv_lines, hits, text_rows)
    write_keeps(out_rows, text_rows)
    print('NAME candidates %d (mapped %d); text hits %d; freeness errors %d; mapped-not-in-hit-table %d'
          % (len(out_rows), sum(1 for o in out_rows if o[1] == 'MAPPED'), len(text_rows), len(free_errs),
             len(mismatch)))
    for e in free_errs:
        print('FREENESS', e)


# ===================================================================================================================
# Writers
# ===================================================================================================================

def _cnt(c: collections.Counter) -> str:
    return ', '.join('%s %d' % kv for kv in sorted(c.items()))


# The R1 reviewer's V/H re-check (R1_review.md, "Re-check against the maintainer's V/H confirmation"): the maintainer's
# ruling "Tucker variations are V and TT variations are H. The D U L R are for the frames." puts these plan rows in
# question. They are tier C / §1.2a rows (not letter_sites rows). The session ruled (2026-11-11): APPLIED in the
# script (tier C-VH): V1 dU -> V; V2 dU, dG -> V, H and the keys p['dG'] -> p['H']; V3 dU, dG -> V, H; V4 dU_v, dG_v
# -> V_v, H_v (the transpose outputs dU/dU0 keep dD/dD0); V5 is an R4a text row (V_i = <U_i, ...>, H_i = <D_i, ...>).
VH_ASK = (
    ('V1', 't3toolbox/backend/sampling_derivatives.py', '_tangent_z_jets_step', '863, 868', 'dU -> dD (§1.2a)',
     'dU is the tangent\'s Tucker variation (var_tucker_cores): by the ruling V (free there); :859\'s data comment and '
     ':865\'s prose follow in R4b'),
    ('V2', 'tests/test_jet_recurrence.py', '_problem', '57, 62, 67 (the dict(dU=...) keyword included)',
     'dU -> dD (§1.2a)', 'dU, dG = v.variations.data: a tangent\'s variations: by the ruling V; partner dG -> H is '
     'outside the map, and the string key p[\'dG\'] (:104-113) would need a scoped string row'),
    ('V3', 'tests/test_probe_derivatives.py', '_var', '499, 501', 'dU -> dD (§1.2a)',
     'a random tangent Tucker variation: by the ruling V; partner dG -> H'),
    ('V4', 'tests/test_probe_derivatives.py', 'test_tangent_transpose_adjoint_identity, '
     'test_tangent_transpose_K_stacked, test_apply_entries_transpose_adjoint_identity, test_adjoint_identities',
     '159/169, 273/283, 370/386, 533/556', 'dU_v -> dD_v (§1.2a)',
     'dU_v, dG_v = v.variations.data: the input tangent\'s variations: by the ruling V; partner dG_v -> H. The same '
     'functions\' dU/dU0 are transpose outputs and correctly stay dD/dD0'),
    ('V5', 't3toolbox/backend/tv_operations.py', '(text, §3.3 row)', '400-401',
     '`dU_i = <O_i, core_env_i>` -> `dD_i = <U_i, core_env_i>` (§3.3)',
     'the two variations: by the ruling V_i = <U_i, core_env_i> and H_i = <D_i, core_env_i> (R4a text)'),
)
VH_ASK_SECTION = ['', '## The V/H re-check (the R1 reviewer): rows the session ruled, 2026-11-11', '',
                  'Applied by `dev/store_rename_apply.py` as tier `C-VH` (V1-V4; the partners `dG`/`dG_v` -> `H`/`H_v` '
                  'where they hold TT variation cores; `test_jet_recurrence`\'s keys `p[\'dG\']` -> `p[\'H\']` as a '
                  'scoped string row); V5 is an R4a text row reading `V_i = <U_i, core_env_i>`, `H_i = <D_i, '
                  'core_env_i>`. The rows as the reviewer listed them:', '',
                  '| # | file | function | lines | the plan\'s row | reason |', '|---|---|---|---|---|---|'] + [
    '| %s | `%s` | `%s` | %s | %s | %s |' % r for r in VH_ASK]


def write_inventory(out_rows, text_rows, excluded, free_errs, mismatch, inv_lines):
    L = []
    P = L.append
    P('# The O/P/Q/U-family inventory (plan §3.4; R1)')
    P('')
    P('Generated by `python dev/store_rename_slices/pqu_scan.py` from the tree at `v2026.4.0` (`t3toolbox/`, '
      '`tests/`, `examples/` equal the tag). The classification table is `CLASSIFY` in that script (ordered, '
      'reviewed entry by entry; every unmapped candidate must match one or the run fails). Disposition per §3.4: '
      'frame core `P`/`Q` -> `L`/`R` (LR); a frame\'s down store -> `D`, its up store (`O`/`U`) -> `U` (STORE); '
      'a variation/direction of a frame down core -> `D` forms (DFORM); a T3\'s own Tucker core -> `B` (B) and its '
      'direction, gradient or Hessian block -> `B` forms (BFORM); `KEEP:<§1.5 item>`; `ASK` (not settled by the '
      'plan). MAPPED = the dry run (`dev/store_rename_apply.py`, tiers A-C) already renames the occurrence.')
    P('')
    P('**Slices.** `R2`: applied in R2 from `dev/store_rename_letter_sites.md` (inventory-backed, plan-enumerated, a '
      'frame core `P`/`Q`, or completing a tier-C binding). `R2-proposed`: a `B`/`D` row (or its simultaneous `O -> '
      'U` partner) beyond tier C\'s lists — on the letter-site list but NOT loaded until the reviewer agrees (§3.4 '
      '"R1 gate"). `R4a/R4b/R5 hand`: text. `kept`: listed in `pqu_keeps.tsv` / `pqu_text_keeps.tsv` for '
      '`dnt.txt` and K8\'s keep list.')
    P('')
    c_cls = collections.Counter(o[1] for o in out_rows)
    c_sl = collections.Counter(o[3] if not o[3].startswith('R2 (tier') else 'R2 (tier C)' for o in out_rows)
    c_tf = collections.Counter(t[3] for t in text_rows)
    c_ts = collections.Counter(t[9] for t in text_rows)
    P('## Counts')
    P('')
    P('- NAME occurrences scanned: **%d**; by class: %s' % (len(out_rows), _cnt(c_cls)))
    P('- NAME occurrences by slice: %s' % _cnt(c_sl))
    P('- text hits: **%d**; by family: %s; by slice: %s' % (len(text_rows), _cnt(c_tf), _cnt(c_ts)))
    P('- freeness errors among this inventory\'s renames: **%d**%s' % (
        len(free_errs), '' if not free_errs else ' (listed below)'))
    P('- MAPPED occurrences missing from the dry-run hit table (must be 0): **%d**' % len(mismatch))
    P('')
    if free_errs:
        P('### Freeness errors')
        P('')
        for e in free_errs:
            P('- ' + e)
        P('')
    P('## The plan\'s "Sites known now" (§3.4), and where each landed')
    P('')
    known = [
        ('geometry.py:122, 139 (`P_last`)', 'text (comment/docstring), R4a hand: frame left core -> `L_last` (see '
         'the text table)'),
        ('uniform_manifold.py:479 (`(P_i^L)^T dG_i^L = 0`)', 'text, R4a: `(L_i^L)^T dG_i^L = 0`; also a §4A block '
         'candidate (`uniform_manifold.py:478`)'),
        ('`_curvature_oracles.py`\'s `P`, `Pd`, `Q`, `Qd` and `Q = O = P = G`',
         '`adjoint_sweep_tangent_trs` P, Pd -> L, Ld (R2); `tt_tangent_assembly_factored_trs` P, Pd, Q, Qd: ASK '
         '(products, not cores); `Q = O = P = G` is text (:397, R4b, §1.2a corewise rows)'),
        ('the exact-Hessian note\'s :117–132, 305, 343–359', 'text rows of `docs/exact_hessian_two_hole.tex`, R5 '
         'hand per §4.8'),
        ('`backend/fitting.py:351, 374, 380, 389`', 'text, R4a: §1.2a\'s text list (B forms at 374, 380, 389; D '
         'forms at 351)'),
        ('`ut3_sampling.py:114, 231`, `entries.py:152, 290`', 'text, R4b: §1.2a\'s text list'),
        ('`frame_variations_format.py:167, 171, 383, 821, 1609`', '167/171/383 STORE U -> D (R2-proposed; reviewer-agreed: R2); 821 '
         'ASK (a Tucker variation; recommend `V`); 1609 U -> D, Ux -> Bx (R2-proposed; reviewer-agreed: R2)'),
        ('`sharing.py:620, 789`', '620 STORE (O -> U, U -> D; R2-proposed; reviewer-agreed: R2); 789 MAPPED by §1.2a (`U -> b`)'),
        ('`manifold.py:1569`', 'MAPPED (§1.2a COREWISE doctest)'),
        ('`ut3_linalg.py:417`', 'BFORM `U -> B` with `KU -> KB` (R2-proposed; reviewer-agreed: R2)'),
        ('`test_padded_start.py:502`', 'B `U -> B` (R2-proposed; reviewer-agreed: R2)'),
        ('`_curvature_oracles.py:73, 241, 398, 524`', 'MAPPED (tier C B rows)'),
        ('`test_curvature.py:482, 999, 1058`', 'MAPPED (tier C B rows); their uses at 483, 1020, 1059 are R2 '
         'completions here'),
        ('`test_uniform_curvature.py:99–100, 249`', '99–100 STORE (R2, inv 1 rows); 249 STORE U -> D '
         '(R2-proposed; reviewer-agreed: R2)'),
        ('`test_tv_operations.py:151`', 'STORE U -> D (R2-proposed; reviewer-agreed: R2)'),
        ('`test_dispatch.py:337`', 'STORE U -> D (R2-proposed; reviewer-agreed: R2)'),
        ('`test_manifold.py:683`', 'text (a comment/docstring letter), see the text table'),
        ('`test_from_sweep_reuse.py`\'s `dU`', 'MAPPED by §1.2a (`dU -> dD` in `_bare_to_dense` and '
         '`test_transposes_per_element`, 8 hits) once the plan\'s short path is read as '
         '`tests/backend/test_from_sweep_reuse.py`'),
        ('`t3_orthogonalization.py` `O_ab_x`, `Oaxb`; `ut3_orthogonalization.py` `O_ab_x`', 'STORE -> `U_ab_x`, '
         '`Uaxb` (R2, plan-enumerated); the signature/return comments :106, :189 by hand (R4a)'),
        ('`shared_t3_math.tex:358`\'s `O_g`', 'kept (text keep, K8 carve-out)'),
    ]
    P('| site | landed |')
    P('|---|---|')
    for s, w in known:
        P('| %s | %s |' % (s, w))
    P('')
    P('## NAME tokens, by file and function')
    P('')
    P('One row per (function, token, class, image, slice) with its lines. `doctest` marks occurrences inside a '
      'docstring example.')
    P('')
    grp = collections.OrderedDict()
    for r, cls, new, sl, basis, why in sorted(out_rows, key=lambda o: (o[0]['file'], o[0]['span'][0], o[0]['tok'],
                                                                        o[0]['line'])):
        k = (r['file'], r['qual'].rsplit('.', 0)[0] if False else _func_label(r), r['tok'], cls, new, sl)
        g = grp.setdefault(k, dict(lines=[], doctest=False, why=why, basis=basis))
        g['lines'].append(r['line'])
        g['doctest'] |= r['doctest']
    cur = None
    for (f, fn, tok, cls, new, sl), g in grp.items():
        if f != cur:
            P('')
            P('### `%s`' % f)
            P('')
            P('| function | token | class | image | slice | lines | basis / reason |')
            P('|---|---|---|---|---|---|---|')
            cur = f
        P('| `%s`%s | `%s` | %s | %s | %s | %s | %s |' % (
            fn, ' (doctest)' if g['doctest'] else '', tok, cls, '`%s`' % new if new else '-', sl,
            _lines(g['lines']), _md(('%s; %s' % (g['basis'], g['why'])).strip('; '))))
    P('')
    P('## Text occurrences (comments, docstring prose, docs, notes)')
    P('')
    P('One row per hit. A `§4A block` hit is rewritten with its block, by hand; the reviewer classifies each by '
      '§3.4 against the line (B vs D by §1.2a\'s object test). "T4S correspondence" lines keep the paper\'s letters '
      'only on the lines §3 names.')
    P('')
    cur = None
    for (path, line, col, fam, tok, text, ctx, fid, disp, sl, keep) in sorted(text_rows):
        if path != cur:
            P('')
            P('### `%s` (text)' % path)
            P('')
            P('| line | family | token | slice | disposition | text |')
            P('|---|---|---|---|---|---|')
            cur = path
        P('| %d | %s | `%s` | %s | %s | %s |' % (line, fam, tok.replace('`', "'").replace('|', '\\|'), sl, _md(disp),
                                              _md(text[:150])))
    P('')
    P('## Appendix: tokens holding U/O/P/Q excluded from the scan (reviewed classes)')
    P('')
    P('| token | class | n |')
    P('|---|---|---|')
    for (tok, cl), n in sorted(excluded.items()):
        P('| `%s` | %s | %d |' % (tok, cl, n))
    L += VH_ASK_SECTION
    with open(OUT_INV, 'w') as fh:
        fh.write('\n'.join(L) + '\n')


def _func_label(r) -> str:
    return r['qual'] if r['qual'] else '<module>'


def _lines(ls) -> str:
    u = sorted(set(ls))
    return ', '.join(map(str, u[:40])) + (' …' if len(u) > 40 else '')


def _md(s: str) -> str:
    return s.replace('|', '\\|').replace('\n', ' ')


def write_sites(out_rows, rows, inv_lines, hits, text_rows):
    L = []
    P = L.append
    P('# Letter sites (plan §1.2 tier C\'s last row; R1)')
    P('')
    P('Generated by `python dev/store_rename_slices/pqu_scan.py`. The machine table between the `SITES` markers is '
      'read by `dev/store_rename_apply.py` (`load_letter_sites`), which applies ONLY the rows whose slice is `R2`: '
      'each renames the NAME token `old` at (`file`, `line`[, `col`]) to `new`. `R2-proposed` rows are listed for '
      'the reviewer (§3.4: a `B` or `D` row beyond tier C\'s lists joins only with the reviewer\'s agreement; to '
      'accept one, change its slice to `R2`). `col` is the 0-based character column (as in the hit table), or `-` '
      'when the token occurs once on the line. A simultaneous swap in one function (`U, D, L, R` with `D` the old TT '
      'store) is listed completely, every occurrence of both members.')
    P('')
    P('<!-- SITES -->')
    P('| id | file | function | line | col | old | new | slice | source | reason |')
    P('|---|---|---|---|---|---|---|---|---|---|')
    site_rows = [(r, cls, new, sl, basis, why) for r, cls, new, sl, basis, why in out_rows
                 if sl in ('R2', 'R2-proposed') and cls != 'MAPPED']
    site_rows.sort(key=lambda o: (o[0]['file'], o[0]['line'], o[0]['col']))
    per_line = collections.Counter((o[0]['file'], o[0]['line'], o[0]['tok']) for o in site_rows)
    for k, (r, cls, new, sl, basis, why) in enumerate(site_rows):
        col = r['col'] if per_line[(r['file'], r['line'], r['tok'])] > 1 or _token_count_on_line(r, rows) > 1 else '-'
        src = 'inv1' if (r['file'], r['line']) in inv_lines else basis.split(':')[0] or 'scan'
        func = ('doctest:' if r['doctest'] else '') + (r['qual'] or '<module>')
        P('| LS%04d | `%s` | `%s` | %d | %s | `%s` | `%s` | %s | %s | %s |' % (
            k + 1, r['file'], func, r['line'], col, r['tok'], new, sl, src, _md(cls + ': ' + why)[:200]))
    P('<!-- /SITES -->')
    P('')
    P('## Accounting: every inv 1 §7 / App. A CORE-LETTER row and every inv 2 §8 CORE-LETTER entry')
    P('')
    P('`tier C` = the dry run renames NAMEs on the line (the rules named); `sites` = rows above; `text` = a prose '
      'letter (R4a/R4b/R5 hand, tier D/E, the text table of `dev/store_rename_pqu_inventory.md`); `§4A` = inside an '
      'inventoried block (hand); `kept` = a token classified kept. A row can have several.')
    P('')
    P('### inv 1 §7 + App. A (frontend, tests, examples)')
    P('')
    P('| file:line | cat | dispositions | current (abridged) |')
    P('|---|---|---|---|')
    hits_by_line = collections.defaultdict(list)
    for (f, l, c), h in hits.items():
        hits_by_line[(f, l)].append(h)
    sites_by_line = collections.defaultdict(list)
    for k, (r, cls, new, sl, basis, why) in enumerate(site_rows):
        sites_by_line[(r['file'], r['line'])].append('LS%04d %s' % (k + 1, sl))
    keeps_by_line = collections.defaultdict(set)
    for r, cls, new, sl, basis, why in out_rows:
        if sl == 'kept' or sl == 'ASK' or sl == 'R4a hand':
            keeps_by_line[(r['file'], r['line'])].add('%s `%s` (%s)' % (sl, r['tok'], cls))
    text_by_line = collections.defaultdict(set)
    fence_by_line = {}
    for t in text_rows:
        text_by_line[(t[0], t[1])].add(t[9])
    for fz in sra.load_fences():
        if fz.fenced:
            for ln in range(fz.first, fz.last + 1):
                fence_by_line.setdefault((fz.file, ln), fz.id)
    n_unacc = 0
    rows_by_line = collections.defaultdict(list)
    for x in rows:
        rows_by_line[(x['file'], x['line'])].append(x)
    for (f, l), recs in sorted(inv_lines.items()):
        disp = []
        hs = [h for h in hits_by_line.get((f, l), ()) if h['ctx'].endswith('NAME')]
        if hs:
            disp.append('tier C: ' + '; '.join(sorted({'%s %s->%s' % (h['rule'], h['old'], h['new']) for h in hs})))
        if sites_by_line.get((f, l)):
            disp.append('sites: ' + ', '.join(sites_by_line[(f, l)]))
        if keeps_by_line.get((f, l)):
            disp.append('; '.join(sorted(keeps_by_line[(f, l)])))
        if (f, l) in fence_by_line:
            disp.append('§4A block %s (hand)' % fence_by_line[(f, l)])
        if text_by_line.get((f, l)):
            disp.append('text: ' + ', '.join(sorted(text_by_line[(f, l)])))
        if not disp:
            code_letters = sorted({x['tok'] for x in rows_by_line.get((f, l), ())
                                   if x['tok'] in CANDIDATES | D_PARTNERS and not x['rule']})
            if code_letters:
                disp.append('**UNACCOUNTED NAME letters %s**' % code_letters)
                n_unacc += 1
            else:
                disp.append('prose letter: R4a/R4b/R5 hand (tier D/E row)')
        cat = recs[0][0]
        P('| `%s:%d` | %s | %s | %s |' % (f, l, cat, _md('; '.join(disp))[:400], _md(recs[0][1])[:110]))
    P('')
    P('Unaccounted inv 1 rows with NAME letters: **%d**.' % n_unacc)
    P('')
    P('### inv 2 §8 CORE-LETTER entries (backend NAME tokens)')
    P('')
    P('| identifier | inv 2 image | n | dry run | disposition |')
    P('|---|---|---|---|---|')
    for ident, newp, cat, n, occ in inv2_core_letter():
        mine = [h for (f, l, c), h in hits.items() if f.startswith('t3toolbox/backend/') and h['ctx'] == 'NAME'
                and h['old'] == ident]
        locs = collections.Counter((os.path.basename(h['file'])[:-3], int(h['line'])) for h in mine)
        missing = collections.Counter(occ) - locs
        rules = sorted({h['rule'] for h in mine})
        if ident in rr.KEPT_BY_PLAN and not mine:
            d = rr.KEPT_BY_PLAN[ident]
        elif not missing:
            d = 'tier C: ' + '; '.join(rules)
        else:
            d = 'NOT all mapped: %s' % dict(missing)
        P('| `%s` | `%s` | %d | %d hits, images %s | %s |' % (ident, newp, n, len(mine),
                                                            sorted({h['new'] for h in mine}), _md(d)))
    L += VH_ASK_SECTION
    with open(OUT_SITES, 'w') as fh:
        fh.write('\n'.join(L) + '\n')


def _token_count_on_line(r, rows) -> int:
    return sum(1 for x in rows if x['file'] == r['file'] and x['line'] == r['line'] and x['tok'] == r['tok'])


# Kept letter names whose meaning lies outside every §1.5 item (a projector, a product T3, a dimension, a count,
# a PartitionSpec alias, a subTest keyword): not DNT names. No token-map entry spells them (the hit table never
# touches them) and K15's closed-map check fails any change to them, so they are left off pqu_keeps.tsv rather
# than inventing a §1.5 item. The one exception gets item 9 with its plan citation:
OTHER_AS_ITEM = {'Ud': ('9', 'Ud = "u dense": the plan keeps it (tier C "letters kept", §1.2a\'s exception for '
                         'o1_ragged.py and o1_uniform.py; K8\'s carve-out)')}


def _outer_def_name(path: str, line: int) -> str:
    """The OUTERMOST def holding ``line`` (methods count; classes do not), as r1_build_lists.outer_defs."""
    if path not in _OUTER_CACHE:
        import ast
        tree = ast.parse(open(os.path.join(REPO, path), encoding='utf-8').read())
        out = []

        def visit(node, inside):
            for ch in ast.iter_child_nodes(node):
                if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if not inside:
                        out.append((min([ch.lineno] + [d.lineno for d in ch.decorator_list]), ch.end_lineno,
                                    ch.name))
                    visit(ch, True)
                else:
                    visit(ch, inside)
        visit(tree, False)
        _OUTER_CACHE[path] = out
    o = [d for d in _OUTER_CACHE[path] if d[0] <= line <= d[1]]
    return o[0][2] if o else ''


_OUTER_CACHE: typ.Dict[str, list] = {}


def write_keeps(out_rows, text_rows):
    """pqu_keeps.tsv: scope '*' (the whole file) only when EVERY occurrence of the token in the file is kept;
    otherwise one row per outermost def holding a kept occurrence (r1_build_lists matches file::outermost def)."""
    status = collections.defaultdict(lambda: [0, 0])          # (tok, file) -> [kept, total]
    for r, cls, new, sl, basis, why in out_rows:
        st = status[(r['tok'], r['file'])]
        st[1] += 1
        st[0] += sl == 'kept'
    seen, dropped = set(), []
    with open(OUT_KEEPS, 'w') as fh:
        fh.write('identifier\tfile\tfunction\tdnt_item\treason\n')
        for r, cls, new, sl, basis, why in sorted(out_rows, key=lambda o: (o[0]['file'], o[0]['tok'], o[0]['line'])):
            if sl != 'kept':
                continue
            item = cls.split(':', 1)[1]
            if item == 'other':
                if r['tok'] not in OTHER_AS_ITEM:
                    dropped.append((r['tok'], r['file']))
                    continue
                item, why = OTHER_AS_ITEM[r['tok']]
            kept, total = status[(r['tok'], r['file'])]
            if kept == total:
                func = '*'
            else:
                func = _outer_def_name(r['file'], r['line'])
                if not func:
                    raise SystemExit('a kept module-level %s in %s beside a renamed one: scope it by hand'
                                     % (r['tok'], r['file']))
            key = (r['tok'], r['file'], func)
            if key in seen:
                continue
            seen.add(key)
            fh.write('%s\t%s\t%s\t%s\t%s\n' % (r['tok'], r['file'], func, item, why))
    print('pqu_keeps: dropped (outside every §1.5 item): %s' % sorted(set(dropped)))
    with open(OUT_TEXT_KEEPS, 'w') as fh:
        fh.write('file\tline\ttoken\treason\n')
        for (path, line, col, fam, tok, text, ctx, fid, disp, sl, keep) in sorted(text_rows):
            if sl == 'kept':
                fh.write('%s\t%d\t%s\t%s\n' % (path, line, tok, keep))


if __name__ == '__main__':
    main()
