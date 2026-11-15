"""The store rename's scripted pass (plan ``dev/store_rename_plan_2026-11-10.md`` §1.3, D-10): tiers A-C.

One simultaneous mapping, implemented with sentinels:

* **PASS 1** rewrites every old token in scope, in place by position (tokenize; never untokenize), into a sentinel
  ``ZQX%04d_XQZ`` that holds no old or new text. NAME tokens (code and doctest examples) take tiers A, B1, B3 and C;
  COMMENT and STRING content (prose, docstring prose and expected output) take the A tiers ``\\b``-bounded; STRING
  tokens equal to ``'up'``/``'down'`` take B2; B4-B6 are scoped exact string literals. Docs (``.md``/``.rst``,
  ``README.md``, ``CLAUDE.md``) take the A tiers only. LaTeX, ``CHANGELOG.md`` and ``docs/release_notes.md`` are
  never touched. Text inside a §4A fence (``dev/store_rename_slices/fences.tsv``) is skipped by every text rule.
* **PASS 2** rewrites only sentinels, ``ZQX(\\d+)_XQZ -> ENTRIES[k][1]``, asserting each key's sentinel count equals
  its hit-table count.

Usage (R1 runs only the dry run; nothing in the tree is written)::

    python dev/store_rename_apply.py --dry-run [--out dev/store_rename_slices] [--scratch DIR]

writes ``R1_hit_table.tsv`` (every (file, line, column, old -> new) the rename would make), ``R1_sentinels.tsv``
(per key: sentinels written by pass 1 and resolved by pass 2), and ``R1_apply_report.md`` (counts per tier per
file, bijection, scope reports), and leaves the pass-1 and pass-2 trees under ``--scratch``.

``--apply`` (R2 only, never in R1) would write pass 2 into the working tree; it refuses to run unless
``--i-am-r2`` is also given.

The script asserts Python 3.11: on 3.12+ tokenize splits f-strings into FSTRING_MIDDLE + NAME, so
``up_ranks={self.up_ranks}`` would get its NAME renamed and keep its label (§1.3).
"""
import argparse
import ast
import bisect
import collections
import dataclasses
import doctest
import hashlib
import io
import os
import re
import shutil
import sys
import tokenize
import typing as typ

assert sys.version_info[:2] == (3, 11), 'store_rename_apply.py needs Python 3.11 (f-string tokens, §1.3)'

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLICES = os.path.join(REPO, 'dev', 'store_rename_slices')
FENCES_TSV = os.path.join(SLICES, 'fences.tsv')
LETTER_SITES_MD = os.path.join(REPO, 'dev', 'store_rename_letter_sites.md')

SENTINEL = 'ZQX%04d_XQZ'
SENTINEL_RE = re.compile(r'ZQX(\d+)_XQZ')


# ===================================================================================================================
# The token map (§1.2). Every entry below is copied from the plan's table; the comments cite the row.
# ===================================================================================================================

# --- A1 renamed: NAME tokens everywhere, \b-bounded in text, never in LaTeX/CHANGELOG -------------------------------
A1_RENAMED: typ.Dict[str, str] = {
    'up_tucker_cores':            'down_tucker_cores',
    'down_tt_cores':              'up_tt_cores',
    'up_tucker_cores_dot':        'down_tucker_cores_dot',
    'down_tt_cores_dot':          'up_tt_cores_dot',
    'up_tucker_supercore':        'down_tucker_supercore',
    'down_tt_supercore':          'up_tt_supercore',
    'masked_up_tucker_supercore': 'masked_down_tucker_supercore',
    'masked_down_tt_supercore':   'masked_up_tt_supercore',
}

# --- A1 aliases, one-way only: backend/fv_conversions.py 181-185, 220, 253 ------------------------------------------
A1_ALIASES: typ.Dict[str, str] = {
    'up_orthogonalize_tucker_cores': 'down_orthogonalize_tucker_cores',
    'down_orthogonalize_tt_cores':   'up_orthogonalize_tt_cores',
}
A1_ALIASES_FILE = 't3toolbox/backend/fv_conversions.py'
A1_ALIASES_LINES = ((181, 185), (220, 220), (253, 253))

# --- A1 labels: text only (comments, docstrings, docs) -------------------------------------------------------------
A1_LABELS: typ.Dict[str, str] = {
    'up_tucker':         'down_tucker',
    'down_tt':           'up_tt',
    'up_tucker_ranks':   'down_ranks',
    'down_tucker_ranks': 'up_ranks',
    'down_tt_ranks':     'up_ranks',
}

# --- A2 swaps (a <-> b), everywhere: each stem with up/UP and its twin with down/DOWN --------------------------------
A2_UP_STEMS: typ.Tuple[str, ...] = (
    'up_ranks', 'up_mask', 'variations_up_mask', 'var_up_mask', 'up_weights', 'up_weight_supercore', 'up_sc',
    'up_cores', 'up_supercore', 'up_inds', 'up_r', 'v_up_r', 'up_out', 'up0', 'up3', 'up4', 'vup', 'mup',
    'up_shapes', 'up_stack_shapes', 'up_slices', 'up_masks', 'up_widths', 'up_mask_frame', 'up_mask_v', '_up',
    '_up_mask', '_up_edge', '_up_message', '_u_up_messages', '_UP_R', 'bad_up', 'masked_up', 'point_up',
    'variations_up',
    'up_more', 'up_shared',                                   # session ruling 2026-11-11 (1): fv_operations.py:368, 378
)
A2_TWINS: typ.Dict[str, str] = {
    # written out (not generated) so a reviewer reads every pair; checked against A2_UP_STEMS below
    'up_ranks': 'down_ranks', 'up_mask': 'down_mask', 'variations_up_mask': 'variations_down_mask',
    'var_up_mask': 'var_down_mask', 'up_weights': 'down_weights', 'up_weight_supercore': 'down_weight_supercore',
    'up_sc': 'down_sc', 'up_cores': 'down_cores', 'up_supercore': 'down_supercore', 'up_inds': 'down_inds',
    'up_r': 'down_r', 'v_up_r': 'v_down_r', 'up_out': 'down_out', 'up0': 'down0', 'up3': 'down3', 'up4': 'down4',
    'vup': 'vdown', 'mup': 'mdown', 'up_shapes': 'down_shapes', 'up_stack_shapes': 'down_stack_shapes',
    'up_slices': 'down_slices', 'up_masks': 'down_masks', 'up_widths': 'down_widths',
    'up_mask_frame': 'down_mask_frame', 'up_mask_v': 'down_mask_v', '_up': '_down', '_up_mask': '_down_mask',
    '_up_edge': '_down_edge', '_up_message': '_down_message', '_u_up_messages': '_u_down_messages',
    '_UP_R': '_DOWN_R', 'bad_up': 'bad_down', 'masked_up': 'masked_down', 'point_up': 'point_down',
    'variations_up': 'variations_down',
    'up_more': 'down_more', 'up_shared': 'down_shared',
}
assert tuple(A2_TWINS) == A2_UP_STEMS
A2_SWAP: typ.Dict[str, str] = {**A2_TWINS, **{b: a for a, b in A2_TWINS.items()}}
A2_ONE_WAY: typ.Dict[str, str] = {
    'test_unequal_old_up_masks_in_a_group': 'test_unequal_old_down_masks_in_a_group',
}

# --- A3 widths: the family nU <-> nD (suffix kept), nO -> nU; the explicit forms ------------------------------------
A3_RE = re.compile(r'(?<![\\\w])(nU|nD|nO)(?=\d|[ijkdn]\b|_|\(|\b)')
A3_FAMILY: typ.Dict[str, str] = {'nU': 'nD', 'nD': 'nU', 'nO': 'nU'}
A3_EXPLICIT: typ.Dict[str, str] = {
    '_NU': '_ND', '_ND': '_NU', 'nnU': 'nnD', 'nnD': 'nnU', 'nDs': 'nUs', 'sum_nD': 'sum_nU',
}

# --- B1 bare NAME up <-> down, except KEEP_BARE (re-derived over doctest lines in R1; see the apply report) -------
B1_SWAP: typ.Dict[str, str] = {'up': 'down', 'down': 'up'}
KEEP_BARE: typ.Tuple[typ.Tuple[str, int, int], ...] = (
    ('tests/test_fitting.py', 1148, 1152),                  # a tangent
    ('tests/backend/_curvature_oracles.py', 529, 536),      # the xi lambda
    ('t3toolbox/fitting.py', 290, 291),                     # a doctest tangent
)

# --- B2 exact literals 'up' <-> 'down', everywhere -----------------------------------------------------------------
B2_SWAP: typ.Dict[str, str] = {'up': 'down', 'down': 'up'}


# ===================================================================================================================
# Scoped rules: A3 scoped swaps and the A3 exception, B3-B6, tier C (§1.2 and §1.2a)
# ===================================================================================================================

@dataclasses.dataclass(frozen=True)
class Rule:
    file:    str                                            # repo-relative path
    mapping: typ.Tuple[typ.Tuple[str, str], ...]            # (old, new) pairs, simultaneous
    tier:    str                                            # 'A3s', 'A3x', 'B3', 'B4', 'B5', 'B6', 'C', 'C-B', 'C-D', ...
    note:    str                                            # the plan row it encodes
    lines:   typ.Optional[typ.Tuple[typ.Tuple[int, int], ...]] = None   # 1-based inclusive ranges; None = no line limit
    funcs:   typ.Optional[typ.Tuple[str, ...]] = None       # enclosing def names; None = no function limit
    on:      str = 'name'                                   # 'name' NAME tokens; 'name+text' also comments/strings
                                                            # (\b-bounded); 'text' comments/strings only; 'str' exact
                                                            # string literal tokens (content compared without quotes)

    @property
    def map(self) -> typ.Dict[str, str]:
        return dict(self.mapping)


def _m(s: str) -> typ.Tuple[typ.Tuple[str, str], ...]:
    """'a->b, c<->d' -> (('a','b'), ('c','d'), ('d','c'))."""
    out = []
    for part in s.split(','):
        part = part.strip()
        if not part:
            continue
        if '<->' in part:
            a, b = (p.strip() for p in part.split('<->'))
            out += [(a, b), (b, a)]
        else:
            a, b = (p.strip() for p in part.split('->'))
            out.append((a, b))
    olds = [a for a, _ in out]
    assert len(olds) == len(set(olds)), s
    return tuple(out)


_PF_ROW = ('U->D, O->U, Ui->Di, U0->D0, U1->D1, U_new->D_new, oU->oD, o_U->o_D, PhiU->PhiD, XiU->XiD, XiUs->XiDs, '
           'PhiO->PhiU, XiO->XiU, bO->bU, O_P->U_P, moves_U->moves_D, moves_O->moves_U, '
           'D->Psi, D_L->Psi_L, D_R->Psi_R, DL->PsiL, DR->PsiR, DU->PsiD, DUs->PsiDs')
_UPF_EXTRA = ('O0->U0, O1->U1, old_U->old_D, new_U->new_D, new_O->new_U, seeds_O->seeds_U, real_O->real_U, P_O->P_U, '
              'rowU->rowD, rowO->rowU, pinU->pinD, pinO->pinU, selU->selD, selO->selU, mvO->mvU')
_A3_SCOPED = 'kU<->kD, mU<->mD, mU0<->mD0, mU1<->mD1, pU<->pD, iU<->iD, wU<->wD, WU<->WD'

B = 't3toolbox/backend/'
F = 't3toolbox/'
T = 'tests/'
TB = 'tests/backend/'

SCOPED: typ.Tuple[Rule, ...] = (
    # ---- A3 scoped swaps (NAME and text in the three uniform padded-frame files; comments only in two more) -------
    Rule(B + 'uniform_padded_frames.py', _m(_A3_SCOPED), 'A3s', 'A3 scoped swaps', on='name+text'),
    Rule(TB + '_uniform_padded_frame_testing.py', _m(_A3_SCOPED), 'A3s', 'A3 scoped swaps', on='name+text'),
    Rule(TB + 'test_uniform_padded_frame.py', _m(_A3_SCOPED), 'A3s', 'A3 scoped swaps', on='name+text'),
    Rule(B + 'ufv_masking.py', _m('WU<->WD'), 'A3s', 'A3 scoped swaps, comments only (:45-46)', on='text'),
    Rule(B + 'padded_frames.py', _m('kU<->kD'), 'A3s', 'A3 scoped swaps, comments only (kU/kD and forms)',
         on='text-kform'),
    # ---- A3 exception: a UT3 point's Tucker rank nU -> n (and nO = nU becomes nU = n) -----------------------------
    Rule(F + 'optimizers.py', _m('nU->n, nO->nU'), 'A3x', 'A3 exception, optimizers.py:176-180',
         lines=((176, 180),), on='name+text'),
    Rule(T + 'test_chunk_size_estimator.py', _m('nU->n, nO->nU'), 'A3x', 'A3 exception, :274-276',
         lines=((274, 276),), on='name+text'),
    Rule(B + 'sampling_derivatives.py', _m('nU->n, nO->nU'), 'A3x', 'A3 exception, chunk estimators',
         lines=((1819, 1824), (1870, 1912)), on='name+text'),
    # ---- §4.11 typos, fixed in R2 by the script ("after A1 they would read as right"): comments only ------------
    Rule(B + 'fv_conversions.py', _m('down_tucker_supercore->up_tt_supercore, right_tucker_supercore->right_tt_supercore'),
         'T-typo', '§4.11 typos (fv_conversions.py:58, 60, 157, 159)',
         lines=((58, 58), (60, 60), (157, 157), (159, 159)), on='text'),
    # ---- B3: NAME dn -> up in the enclosing function/doctest of the listed lines (resolved at load time) -----------
    Rule(F + 'uniform_frame_variations_format.py', _m('dn->up'), 'B3', 'B3 dn->up', lines=((1029, 1029),),
         on='name-enclosing'),
    Rule(T + 'test_uniform_frame_variations_format.py', _m('dn->up'), 'B3', 'B3 dn->up',
         lines=((250, 250), (292, 292), (323, 323)), on='name-enclosing'),
    Rule(T + 'test_dispatch.py', _m('dn->up'), 'B3', 'B3 dn->up', lines=((1680, 1680), (2840, 2840)),
         on='name-enclosing'),
    Rule(T + 'test_weighted.py', _m('dn->up'), 'B3', 'B3 dn->up', lines=((211, 211),), on='name-enclosing'),
    # ---- B4-B6: scoped exact string literals ------------------------------------------------------------------------
    Rule(T + 'test_jet_recurrence.py', _m("U->D, O->U, P->L, Q->R"), 'B4', 'B4 the dict keys p[...]', on='str'),
    Rule(T + 'test_tucker_tensor_train.py', _m('U->B'), 'B5', 'B5 the T3 Tucker kind',
         lines=((2236, 2236), (2255, 2255), (2283, 2283), (2301, 2301), (2370, 2370), (2391, 2391)), on='str'),
    Rule(TB + '_curvature_oracles.py', _m('dU->dB'), 'B6', "B6 st['dU']",
         lines=((423, 423), (464, 464), (543, 543)), on='str'),
    Rule(TB + 'test_curvature.py', _m('dU->dB'), 'B6', "B6 st['dU']",
         lines=((740, 740), (751, 751), (756, 756), (1328, 1328), (1335, 1335), (1338, 1338)), on='str'),

    # ---- Tier C, per-file letter maps --------------------------------------------------------------------------------
    Rule(B + 'padded_frames.py', _m(_PF_ROW), 'C', 'C padded_frames.py (+ D-5)'),
    Rule(B + 'uniform_padded_frames.py', _m(_PF_ROW + ', ' + _UPF_EXTRA), 'C', 'C uniform_padded_frames.py'),
    Rule(B + 'fv_operations.py', _m('UU<->DD, U->D, D->U'), 'C', 'C fv_operations.py'),
    Rule(B + 'fv_conversions.py', _m('D->U, left_UU->left_DD, right_UU->right_DD'), 'C', 'C fv_conversions.py'),
    Rule(B + 'ufv_conversions.py', _m('U->D, um<->dm, uc->c_down, dc->c_up'), 'C', 'C ufv_conversions.py'),
    Rule(B + 'ufv_operations.py', _m('um<->dm'), 'C', 'C ufv_operations.py'),
    Rule(B + 'ufv_masking.py', _m('UM_k->DM_k, UM_t<->DM_t'), 'C', 'C ufv_masking.py ufv_apply_frame_masks',
         funcs=('ufv_apply_frame_masks',)),
    Rule(B + 'ufv_masking.py', _m('DM_k->UM_k, UM_t->DM_t'), 'C', 'C ufv_masking.py ufv_apply_variations_masks',
         funcs=('ufv_apply_variations_masks',)),
    Rule(B + 'tv_operations.py', _m('U->D, O->U'), 'C', 'C tv_operations.py'),
    Rule(B + 'utv_operations.py', _m('U->D, O->U, BU->BD, BU1->BD1'), 'C', 'C utv_operations.py'),
    Rule(B + 't3_linalg.py', _m('U->D, O->U, BU->BD'), 'C', 'C t3_linalg.py (660-760)', lines=((660, 760),)),
    Rule(B + 'geometry.py', _m('U->D, O->U'), 'C', 'C geometry.py (860-892)', lines=((860, 892),)),
    Rule(F + 'frame_variations_format.py', _m('UU<->DD, U<->D'), 'C', 'C frame_variations_format.py (483-573)',
         lines=((483, 573),)),
    Rule(F + 'uniform_frame_variations_format.py', _m('uc->c_down, dc->c_up, UU_good<->DD_good, UM_good<->DM_good'),
         'C', 'C uniform_frame_variations_format.py'),
    Rule(B + 'probing.py', _m('U->D, O->U, P->L, Q->R, Ud->Dd, Pd->Ld, rev_Q->rev_R'), 'C', 'C probing.py'),
    Rule(B + 'sampling_derivatives.py',
         _m('U->D, O->U, P->L, Q->R, P_dot->L_dot, Q_dot->R_dot, P_cores->L_cores, P_cores_dot->L_cores_dot, '
            'Pxi->Lxi, dPxi->dLxi, Qxi->Rxi, rev_Q->rev_R, Ud->Dd, Od->Ud, mO->mU, On->Un, dmO->dmU, dOn->dUn, '
            'Odxi->Udxi'), 'C', 'C sampling_derivatives.py'),
    Rule(B + 'apply.py', _m('O->U, Q->R'), 'C', 'C apply.py'),
    Rule(B + 'entries.py', _m('D->n_cores'), 'C', 'C entries.py (361-363): the core count', lines=((361, 363),)),
    # ---- sigma-hat locals (D-1), function-scoped by the plan's lines ---------------------------------------------
    Rule(B + 'sampling_derivatives.py', _m('sh->sig_hat'), 'C-sig', 'C sigma-hat sh',
         lines=((2159, 2159), (2167, 2167), (2201, 2204), (3188, 3220))),
    Rule(B + 'sampling_derivatives.py', _m('shd->sig_hat_d'), 'C-sig', 'C sigma-hat shd', lines=((3188, 3220),)),
    Rule(B + 'apply.py', _m('sh->sig_hat, shd->sig_hat_d'), 'C-sig', 'C sigma-hat (apply.py)',
         lines=((493, 501), (539, 542))),
    Rule(T + '_jet_trs_twins.py', _m('sh->sig_hat'), 'C-sig', 'C sigma-hat (twin)', lines=((213, 215),)),
    # ---- T4S test twins ---------------------------------------------------------------------------------------------
    Rule(T + '_jet_trs_twins.py', _m('O->U, P->L, Q->R, P_cores->L_cores, rev_Q->rev_R'), 'C', 'C T4S twin'),
    Rule(T + 'test_jet_recurrence.py', _m('U->D, O->U, P->L, Q->R'), 'C', 'C test_jet_recurrence _problem',
         funcs=('_problem',), lines=((49, 68),)),
    Rule(T + 'test_jet_recurrence.py', _m('O->Uc, P->Lc, Q->Rc'), 'C', 'C test_uniform_equals_ragged (c suffix)',
         funcs=('test_uniform_equals_ragged',), lines=((163, 200),)),
    # ---- tests that mirror a mapped module --------------------------------------------------------------------------
    Rule(TB + '_uniform_padded_frame_testing.py', _m('U->D, O->U, Uu->Du, Ou->Uu'), 'C', 'C mirror'),
    Rule(TB + 'test_uniform_padded_frame.py', _m('U->D, O->U, Uu->Du, Ou->Uu'), 'C', 'C mirror'),
    Rule(TB + 'test_padded_start.py', _m('U->D, O->U, Uu->Du, Ou->Uu'), 'C', 'C mirror (290-307)',
         lines=((290, 307),)),
    Rule(TB + 'test_padded_frame.py', _m('U0->D0, O0->U0, U->D, O->U'), 'C', 'C test_padded_frame mirror',
         lines=((111, 111), (135, 139), (205, 216), (619, 629))),
    Rule(T + 'test_uniform_frame_variations_format.py', _m('U_r->D_r, U_u->D_u'), 'C', 'C (958-959)',
         lines=((958, 959),)),
    Rule(T + 'test_dispatch.py', _m('U_3->D_3'), 'C', 'C test_dispatch U_3'),
    # ---- the shape ruling's held locals (2026-11-11) ---------------------------------------------------------------
    Rule(B + 'tv_operations.py', _m('dB->V, dV->H'), 'C-VH',
         'shape ruling: held locals (dB ruled; dV, a TT variation, PENDING agreement)',
         funcs=('tv_orthogonal_gauge_projection',)),
    Rule(B + 'tv_operations.py', _m('dB->V, dB_parallel->V_parallel, dG->H, dG1->H1'), 'C-VH',
         'shape ruling: held locals (dG1, a TT variation, PENDING agreement)', funcs=('tv_oblique_gauge_projection',)),
    Rule(B + 't3_linalg.py', _m('dB->V, dG->H'), 'C-VH', 'shape ruling: held local, C+(nOi,Ni) -> V',
         funcs=('_inner_product_variations_step',)),
    Rule(B + 'utv_operations.py', _m('dB->V, dG->H'), 'C-VH', 'shape ruling: held local, (nD, N) old -> V',
         funcs=('utv_project_ut3_onto_tangent_space',)),
    # ---- session rulings of 2026-11-11 on the R1 ASK items (to be recorded by the writer in plan draft 21) --------
    Rule(T + 'test_uniform_frame_variations_format.py', _m('um<->dm'), 'C', 'C session ruling (2): _lost_directions',
         funcs=('_lost_directions',)),
    Rule(TB + 'test_curvature.py', _m('sh->sig_hat, shd->sig_hat_d'), 'C-sig',
         'C sigma-hat (test_curvature, session ruling (5))',
         funcs=('test_apply_tt_assembly_is_the_three_term_product_rule',), lines=((1063, 1066),)),
    # ---- dense oracles (function-scoped) ------------------------------------------------------------------------------
    Rule(TB + '_padded_frame_oracles.py',
         _m('D->dense, oldD->old_dense, curD->cur_dense, U->D, Uo->Do, Uc->Dc, E->U, Eo->Uo, Ec->Uc, '
            'startD->start_dense, Da->dense_a, Db->dense_b'), 'C', 'C dense oracles',
         funcs=('dense_frames', 'dense_pass', 'dense_loop', 'tangent_basis', 'dense_project', 'distances')),
    Rule(TB + 'test_padded_start.py', _m('D->dense, U->D, E->U'), 'C', 'C dense oracles (566-580)',
         funcs=('dense_tied_project',), lines=((566, 580),)),
    Rule(TB + 'test_padded_start.py', _m('startD->start_dense, D->dense'), 'C', 'C dense words (478-481)',
         lines=((478, 481),)),

    # ---- B rows (D-6: a T3's own Tucker cores), function-scoped ----------------------------------------------------
    Rule(B + 't3_linalg.py', _m('Ux->Bx, Uy->By, Uxi->Bxi, Uyi->Byi, Utilde_full->Btilde_full'), 'C-B',
         'C B rows t3m_inplace_fused', funcs=('t3m_inplace_fused',)),
    Rule(B + 't3_linalg.py', _m('UL->BL, UR->BR'), 'C-B', 'C B rows _t3m_swap_pair', funcs=('_t3m_swap_pair',)),
    Rule(B + 't3_linalg.py', _m('Ua->Ba, Ub->Bb, Utilde_full->Btilde_full, Utilde->Btilde'), 'C-B',
         'C B rows _t3m_contract_pair', funcs=('_t3m_contract_pair',)),
    Rule(B + 't3_linalg.py', _m('Ux->Bx, Uy->By, Uy_rev->By_rev'), 'C-B', 'C B rows t3m_swap', funcs=('t3m_swap',)),
    Rule(B + 't3_linalg.py', _m('BU->zB, BUd->zBd'), 'C-B', 'C B rows inner-product sweep',
         funcs=('t3_precompute_inner_product_sweep', 't3_inner_product_corewise_hessian_from_sweep')),
    Rule(B + 't3_operations.py', _m('U->B'), 'C-B', 'C B rows t3_operations.py:89', lines=((89, 89),)),
    Rule(B + 'geometry.py', _m('U->B'), 'C-B', 'C B rows geometry.py:395-399, 416-418',
         lines=((395, 399), (416, 418))),
    Rule(B + 't3_orthogonalization.py', _m('Uio->Bio'), 'C-B', 'C B rows', funcs=('_up_orthogonalize_tt_cores_step',)),
    Rule(B + 't3_orthogonalization.py', _m('new_Uxo->new_Bxo'), 'C-B', 'C B rows',
         funcs=('_down_orthogonalize_tucker_cores_step',)),
    Rule(B + 't3_orthogonalization.py', _m('U_i_o->B_i_o'), 'C-B', 'C B rows', funcs=('t3_down_svd_tucker_core',)),
    Rule(F + 'tucker_tensor_train.py', _m('gU->gB, u->b'), 'C-B', 'C B rows T3 doctests',
         lines=((3870, 3871), (4072, 4073))),
    Rule(F + 'uniform_tucker_tensor_train.py', _m('gU->gB'), 'C-B', 'C B rows UT3 doctests',
         lines=((566, 567), (631, 632))),
    Rule(T + 'test_tucker_tensor_train.py', _m('dU->dB, gU->gB, gUk->gBk, du->db, u->b'), 'C-B',
         'C B rows corewise-transpose tests (u->b over tk)', lines=((2228, 2410), (3030, 3080))),
    Rule(T + 'test_weighted.py', _m('Uc->Bc'), 'C-B', 'C B rows hadamard_cores', funcs=('hadamard_cores',)),
    Rule(T + 'test_weighted.py', _m('Ux->Bx'), 'C-B', 'C B rows', lines=((401, 402), (412, 414))),
    Rule(T + 'test_sharing.py', _m('U_rows->B_rows'), 'C-B', 'C B rows (from y[0])', lines=((245, 248),)),
    Rule(T + 'test_sharing.py', _m('U->B'), 'C-B', 'C B rows (over x_data[0])', lines=((694, 694),)),
    Rule(T + 'test_dispatch.py', _m('U->B'), 'C-B', 'C B rows (theta[0], core_pair[0])',
         lines=((524, 524), (568, 568), (611, 611), (683, 683), (2944, 2944))),
    Rule(T + 'oracle_sweeps/sweep_models.py', _m('U->B'), 'C-B', 'C B rows', lines=((49, 53),)),
    Rule(TB + 'test_curvature.py', _m('BU->zB'), 'C-B', 'C B rows', lines=((594, 594), (600, 600))),
    Rule(TB + 'test_curvature.py', _m('U->B'), 'C-B', "C B rows (theta's raw cores)",
         lines=((482, 482), (999, 999), (1042, 1042), (1058, 1058))),
    Rule(TB + '_curvature_oracles.py', _m('U->B, gU->gB, gYU->gYB'), 'C-B', 'C B rows blocksum (+1.2a)',
         funcs=('blocksum',)),
    Rule(TB + '_curvature_oracles.py', _m('U->B, B->M, dU->dB, u->b'), 'C-B',
         'C B rows vertical_direction (gauge B -> M; 1.2a dU->dB, u->b)', funcs=('vertical_direction',)),
    Rule(TB + '_curvature_oracles.py', _m('U->B, Ud->Bd'), 'C-B', 'C B rows (+1.2a Ud->Bd)',
         funcs=('probe_derivatives_two_hole_states_trs', 'apply_derivatives_two_hole_states_trs')),
    Rule(TB + 'test_optimizers.py', _m('gU->gB, u->b, du->db, dU->dB_dir'), 'C-B', 'C B rows test_derivative_kinds',
         funcs=('test_derivative_kinds',), lines=((298, 306),)),

    # ---- D rows from the draft-15 scan (a frame's store named U/O) ----------------------------------------------------
    Rule(T + 'test_weighted.py', _m('U->D'), 'C-D', 'C D rows', lines=((401, 402), (412, 414))),
    Rule(T + 'test_weighted.py', _m('U->D, Ur->Dr'), 'C-D', 'C D rows', lines=((433, 434),)),
    Rule(T + 'test_sharing.py', _m('U_rows->D_rows'), 'C-D', 'C D rows (598-604)', lines=((598, 604),)),
    Rule(T + 'test_sharing.py', _m('O->U, U->D'), 'C-D', 'C D rows zip(frame[1], frame[0])',
         lines=((628, 628), (647, 648), (687, 688), (715, 716), (746, 747), (759, 760))),
    Rule(T + 'test_sharing.py', _m('U->D'), 'C-D', 'C D rows', lines=((629, 630),)),
    Rule(TB + 'test_ranks.py', _m('U->D, O->U'), 'C-D', 'C D rows', lines=((445, 446),)),
    Rule(TB + 'test_uniform_fitting.py', _m('U->D, O->U'), 'C-D', 'C D rows', lines=((358, 359),)),
    Rule(T + 'test_aligned_padded_frame_frontend.py', _m('U->D'), 'C-D', 'C D rows', lines=((273, 273),)),
    Rule(T + 'test_dispatch.py', _m('U->D'), 'C-D', 'C D rows (theta = (D, L) from fd)',
         lines=((488, 488), (1066, 1066))),
    Rule(T + 'test_uniform_probing.py', _m('U->D, O->U'), 'C-D', 'C D rows', lines=((742, 742),)),
    Rule(T + 'test_uniform_frame_variations_format.py', _m('U->D, O->U'), 'C-D', 'C D rows', lines=((864, 864),)),

    # ---- §1.2a: the U family (D-12), function by function -------------------------------------------------------------
    # B forms
    Rule(B + 'apply.py', _m('dU->dB'), 'C-1.2a-B', '1.2a', funcs=('_apply_corewise_hessian_adjoint',)),
    Rule(B + 'probing.py', _m('dU->dB'), 'C-1.2a-B', '1.2a',
         funcs=('assemble_tucker_tangent_variations', 't3_probe_corewise_hessian_from_sweep')),
    Rule(B + 'sampling_derivatives.py', _m('dU->dB'), 'C-1.2a-B', '1.2a',
         funcs=('_assemble_tucker_tangent_variation_jets_dense', 't3_probe_corewise_derivatives_hessian_from_sweep',
                '_apply_tangent_assemble_step', '_apply_derivatives_hessian_from_jets')),
    Rule(B + 'sharing.py', _m('Udot->Bdot, Udot_pm->Bdot_pm, U->b'), 'C-1.2a-B', '1.2a sharing doctest :788-802',
         lines=((788, 802),)),
    Rule(B + 'tv_operations.py', _m('Udot->Bdot, Udot_g->Bdot_g'), 'C-1.2a-B', '1.2a', funcs=('tv_corewise_lift',)),
    Rule(B + 'tv_operations.py', _m('Udot->Bdot'), 'C-1.2a-B', '1.2a', funcs=('tv_corewise_lift_transpose',)),
    Rule(B + 'tv_operations.py', _m('U->b'), 'C-1.2a-B', '1.2a doctest :547 [b.shape for b in theta_hat[0]]',
         lines=((547, 547),)),
    Rule(B + 'utv_operations.py', _m('Udot->Bdot'), 'C-1.2a-B', '1.2a', funcs=('utv_corewise_lift',)),
    Rule(B + 'utv_operations.py', _m('Ud->Bd'), 'C-1.2a-B', '1.2a', funcs=('utv_corewise_lift_transpose',)),
    Rule(B + 'utv_operations.py', _m('dU->dB'), 'C-1.2a-B', '1.2a', funcs=('utv_corewise_retract',)),
    Rule(F + 'manifold.py', _m('U->B, dU->dB, u->b, du->db'), 'C-1.2a-B', '1.2a COREWISE doctest :1569-1571',
         lines=((1569, 1572),)),
    Rule(TB + '_curvature_oracles.py', _m('Ud->Bd, Ug->Bg'), 'C-1.2a-B', '1.2a', funcs=('tied_lift',)),
    Rule(TB + '_curvature_oracles.py', _m('Ud->Bd'), 'C-1.2a-B', '1.2a', funcs=('deta_tilde_tangent_jets_trs',)),
    Rule(TB + '_curvature_oracles.py', _m('dU->dB'), 'C-1.2a-B', '1.2a',
         funcs=('apply_derivatives_tangent_assembly_trs',)),
    Rule(TB + 'test_curvature.py', _m('Ud->Bd'), 'C-1.2a-B', '1.2a',
         funcs=('test_lift_identity', 'test_zero_point_lifts_to_zero_except_the_last_tt_slot',
                'test_probe_pieces_against_finite_differences_of_the_gn_kernels')),
    Rule(T + 'test_probe_derivatives.py', _m('gU->gB, dU->dB, u->b'), 'C-1.2a-B', '1.2a',
         funcs=('test_corewise_derivatives_finite_difference',)),
    # D forms
    # ---- the maintainer's shape ruling (2026-11-11): an array's shape decides its letter; (nU, N) is V-family and
    #      (rL, nD, rR) H-family whatever computed it, so the transpose outputs and their TT partners take V/H forms
    Rule(B + 'apply.py', _m('dU_tildes->V_tildes, dG_tildes->H_tildes'), 'C-VH', 'shape ruling: transpose outputs',
         funcs=('_apply_transpose_adjoint',)),
    Rule(B + 'sharing.py', _m('Udot->Ddot'), 'C-1.2a-D', '1.2a tied ambient direction',
         funcs=('fv_share_tucker_variations', '_tied_filtered_solve_transpose')),
    Rule(B + 'probing.py', _m('dU_tilde->V_tilde'), 'C-VH', 'shape ruling: transpose outputs',
         funcs=('_assemble_tucker_variations_core',)),
    Rule(B + 'probing.py', _m('dU_tildes->V_tildes, dG_tildes->H_tildes'), 'C-VH', 'shape ruling: transpose outputs',
         funcs=('assemble_tucker_variations', 'tv_probe_transpose_from_sweep', 'tv_probe_transpose')),
    Rule(B + 'probing.py', _m('dG_tilde->H_tilde, dG_tildes->H_tildes'), 'C-VH',
         'shape ruling: the TT assembly twins of the Tucker transposes', funcs=('_assemble_tt_variations_core',
                                                                               'assemble_tt_variations')),
    Rule(B + 'probing.py', _m('B->D, dB->V'), 'C-VH', 'shape ruling: held local (_assemble_tangent_z_step)',
         funcs=('_assemble_tangent_z_step',)),
    # probing._deta_tilde_tangent_step's Ud->Dd is the file row (Ud->Dd)
    # V1 (session ruling 2026-11-11, the reviewer's V/H re-check): dU holds the tangent's Tucker variation -> V
    Rule(B + 'sampling_derivatives.py', _m('dU->V'), 'C-VH', 'V/H ruling V1: _tangent_z_jets_step',
         funcs=('_tangent_z_jets_step',)),
    Rule(B + 'sampling_derivatives.py', _m('dU_tildes->V_tildes, dG_tildes->H_tildes'), 'C-VH',
         'shape ruling: transpose outputs',
         funcs=('_assemble_tucker_variation_jets_dense', 'tv_probe_transpose_derivatives_from_sweep',
                '_apply_derivatives_transpose_from_jets')),
    Rule(B + 'sampling_derivatives.py', _m('dG_tildes->H_tildes'), 'C-VH',
         'shape ruling: the TT assembly twin of the Tucker transposes', funcs=('_assemble_tt_variation_jets_dense',)),
    # the arrays dU_t, dG_t -> V_t, H_t (ruled); the contraction-spec strings dU, dG -> gV, gH (the twin's ruled
    # gU -> gV form; PENDING the reviewer's agreement)
    Rule(B + 'sampling_derivatives.py', _m('dU->gV, dU_t->V_t, dG->gH, dG_t->H_t'), 'C-VH',
         'shape ruling: transpose outputs (spec names pending agreement)', funcs=('_apply_transpose_assemble_core',)),
    # sampling_derivatives._deta_tilde_tangent_jets_step's Ud->Dd is the file row (Ud->Dd)
    Rule(B + 'utv_sampling.py', _m('dU_tilde->V_tilde, dG_tilde->H_tilde'), 'C-VH',
         'shape ruling: transpose outputs (the seven transposes + remasked)', funcs=('<<utv_sampling transposes>>',)),
    Rule(F + 'manifold.py', _m('dU_tildes->V_tildes, dG_tildes->H_tildes'), 'C-VH', 'shape ruling: transpose outputs',
         funcs=('probe_transpose',)),
    Rule(F + 'manifold.py', _m('dU->V, dG->H'), 'C-VH', 'shape ruling: transpose outputs',
         funcs=('apply_transpose', 'entries_transpose', 'probe_derivatives_transpose', 'apply_derivatives_transpose',
                'entries_derivatives_transpose')),
    Rule(F + 'manifold.py', _m('dU_sum->V_sum'), 'C-VH', 'shape ruling: doctest :816-818', lines=((816, 818),)),
    Rule(T + '_jet_trs_twins.py', _m('gU->gV, dU->V, gG->gH, dG->H'), 'C-VH',
         'shape ruling: the twin (gU->gV, dU->V ruled; gG->gH, dG->H as their partners, gG pending agreement)',
         funcs=('apply_derivatives_transpose_trs',)),
    # _bare_to_dense already binds V (the UT3Variations): its parameters take tkv, ttv (the library's names for the
    # uniform variation supercores; a clash spelling, PENDING the reviewer's agreement)
    Rule(TB + 'test_from_sweep_reuse.py', _m('dU->tkv, dG->ttv'), 'C-VH', 'shape ruling (clash spelling, pending)',
         funcs=('_bare_to_dense',)),
    Rule(TB + 'test_from_sweep_reuse.py', _m('dU->V, dG->H'), 'C-VH', 'shape ruling: transpose outputs',
         funcs=('test_transposes_per_element',)),
    # V2: dU, dG = v.variations.data -> V, H (the dict keywords included); the keys p['dG'] follow (string row below)
    Rule(T + 'test_jet_recurrence.py', _m('dU->V, dG->H'), 'C-VH', 'V/H ruling V2: _problem (dict keywords incl.)',
         funcs=('_problem',)),
    Rule(T + 'test_jet_recurrence.py', _m('dU->V, dG->H'), 'B4', "V/H ruling V2: the keys p['dU'], p['dG']", on='str'),
    Rule(T + 'test_manifold.py', _m('dU->V, dG->H'), 'C-VH', 'shape ruling: transpose outputs',
         funcs=('test_tangent_apply_transpose_kstack',)),
    Rule(T + 'test_probe_derivatives.py', _m('dU->V, dU0->V0, dG->H, dG0->H0'), 'C-VH', 'shape ruling: transpose outputs',
         funcs=('test_tangent_transpose_adjoint_identity', 'test_tangent_transpose_K_stacked',
                'test_apply_entries_transpose_adjoint_identity', 'test_adjoint_identities')),
    # V4: the input tangent's variations dU_v, dG_v -> V_v, H_v; V3: _var's random variations dU, dG -> V, H
    Rule(T + 'test_probe_derivatives.py', _m('dU_v->V_v, dG_v->H_v'), 'C-VH', 'V/H ruling V4: the input tangent',
         funcs=('test_tangent_transpose_adjoint_identity', 'test_tangent_transpose_K_stacked',
                'test_apply_entries_transpose_adjoint_identity', 'test_adjoint_identities')),
    Rule(T + 'test_probe_derivatives.py', _m('dU->V, dG->H'), 'C-VH', 'V/H ruling V3: _var', funcs=('_var',)),
    Rule(T + 'test_sharing.py', _m('Udot->Ddot'), 'C-1.2a-D', '1.2a tied ambient direction',
         funcs=('test_degenerate_point_min_norm',)),
)


# ===================================================================================================================
# Scopes: function spans, docstrings, doctest examples
# ===================================================================================================================

@dataclasses.dataclass
class FileScopes:
    funcs:    typ.Dict[str, typ.List[typ.Tuple[int, int]]]  # def name -> [(first line incl. decorators, end line)]
    spans:    typ.List[typ.Tuple[int, int, str]]            # every def/class span, innermost search
    doc_lines: typ.Set[typ.Tuple[int, int]]                 # (line, col) of docstring STRING tokens


def file_scopes(src: str) -> FileScopes:
    tree = ast.parse(src)
    funcs: typ.Dict[str, typ.List[typ.Tuple[int, int]]] = collections.defaultdict(list)
    spans = []
    docs = set()
    nodes = [tree] + list(ast.walk(tree))
    for n in nodes:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            first = min([n.lineno] + [d.lineno for d in n.decorator_list])
            funcs[n.name].append((first, n.end_lineno))
            spans.append((first, n.end_lineno, n.name))
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.body:
            s = n.body[0]
            if isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str):
                docs.add((s.value.lineno, s.value.col_offset))
    return FileScopes(dict(funcs), spans, docs)


def qualname_at(scopes: FileScopes, line: int) -> str:
    c = sorted([s for s in scopes.spans if s[0] <= line <= s[1]], key=lambda s: -(s[1] - s[0]))
    return '.'.join(s[2] for s in c) or '<module>'


# ===================================================================================================================
# Hits
# ===================================================================================================================

@dataclasses.dataclass(frozen=True)
class Hit:
    file: str        # repo-relative
    line: int        # 1-based
    col:  int        # 0-based, characters
    ctx:  str        # NAME | COMMENT | STRING | DOCPROSE | DOCTEST-NAME | DOCTEST-COMMENT | DOCTEST-STRING | STRLIT | DOC
    tier: str
    old:  str
    new:  str
    rule: str        # the plan row / note
    scope: str       # enclosing qualname


class Entries:
    """ENTRIES: every (old, new) pair, indexed k = 0, 1, ... in registration order."""

    def __init__(self):
        self.pairs: typ.List[typ.Tuple[str, str]] = []
        self.index: typ.Dict[typ.Tuple[str, str], int] = {}

    def key(self, old: str, new: str) -> int:
        if (old, new) not in self.index:
            self.index[(old, new)] = len(self.pairs)
            self.pairs.append((old, new))
        return self.index[(old, new)]


# ===================================================================================================================
# Fences (§4A g)
# ===================================================================================================================

@dataclasses.dataclass(frozen=True)
class Fence:
    id:        str
    file:      str
    first:     int
    last:      int
    col_start: typ.Optional[int]
    col_end:   typ.Optional[int]
    fenced:    bool
    kind:      str
    sha1:      str


def load_fences(path: str = FENCES_TSV) -> typ.List[Fence]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as fh:
        header = fh.readline().rstrip('\n').split('\t')
        assert header[:9] == ['id', 'file', 'first_line', 'last_line', 'col_start', 'col_end', 'fenced', 'kind',
                              'anchor_sha1'], header
        for row in fh:
            if not row.strip():
                continue
            c = row.rstrip('\n').split('\t')
            out.append(Fence(c[0], c[1], int(c[2]), int(c[3]), None if c[4] == '-' else int(c[4]),
                             None if c[5] == '-' else int(c[5]), c[6] == 'y', c[7], c[8]))
    return out


def fence_intervals(fences: typ.Sequence[Fence], file: str) -> typ.Dict[int, typ.List[typ.Tuple[int, float]]]:
    """line -> [(col_start, col_end)) fenced on that line."""
    out: typ.Dict[int, typ.List[typ.Tuple[int, float]]] = collections.defaultdict(list)
    for f in fences:
        if f.file != file or not f.fenced:
            continue
        for ln in range(f.first, f.last + 1):
            out[ln].append((f.col_start or 0, float('inf') if f.col_end is None else f.col_end))
    return out


def _in_fence(fi: typ.Dict[int, typ.List[typ.Tuple[int, float]]], line: int, c0: int, c1: int) -> bool:
    return any(c0 < b and c1 > a for a, b in fi.get(line, ()))


# ===================================================================================================================
# Letter sites (§1.2 tier C's last row; R1 builds dev/store_rename_letter_sites.md)
# ===================================================================================================================

@dataclasses.dataclass(frozen=True)
class LetterSite:
    id:    str
    file:  str
    func:  str
    line:  int
    col:   typ.Optional[int]
    old:   str
    new:   str
    slice: str


def load_letter_sites(path: str = LETTER_SITES_MD) -> typ.List[LetterSite]:
    """The rows of the machine table between ``<!-- SITES -->`` and ``<!-- /SITES -->`` whose slice is R2."""
    if not os.path.exists(path):
        return []
    text = open(path).read()
    m = re.search(r'<!-- SITES -->(.*?)<!-- /SITES -->', text, re.S)
    if not m:
        return []
    out = []
    for row in m.group(1).splitlines():
        if not row.startswith('| ') or row.startswith('| id ') or set(row) <= set('|- '):
            continue
        c = [x.strip().strip('`') for x in row.strip().strip('|').split('|')]
        if len(c) < 8:
            continue
        if c[7] != 'R2':                  # R2-proposed rows wait for the reviewer's agreement (§3.4); never applied
            continue
        out.append(LetterSite(c[0], c[1], c[2], int(c[3]), None if c[4] in ('', '-') else int(c[4]), c[5], c[6],
                              c[7]))
    return out


# ===================================================================================================================
# The text path: the A tiers, \b-bounded (comments, docstring prose, expected output, string content, docs)
# ===================================================================================================================

def _word_alt(words: typ.Iterable[str]) -> str:
    return '|'.join(re.escape(w) for w in sorted(set(words), key=lambda w: (-len(w), w)))


_GLOBAL_TEXT_WORDS: typ.Dict[str, typ.Tuple[str, str]] = {}      # old -> (new, tier)
for _o, _n in A1_RENAMED.items():
    _GLOBAL_TEXT_WORDS[_o] = (_n, 'A1')
for _o, _n in A1_LABELS.items():
    _GLOBAL_TEXT_WORDS[_o] = (_n, 'A1-label')
for _o, _n in A2_SWAP.items():
    _GLOBAL_TEXT_WORDS[_o] = (_n, 'A2')
for _o, _n in A2_ONE_WAY.items():
    _GLOBAL_TEXT_WORDS[_o] = (_n, 'A2')
for _o, _n in A3_EXPLICIT.items():
    _GLOBAL_TEXT_WORDS[_o] = (_n, 'A3-explicit')

_GLOBAL_TEXT_RE = re.compile(r'(?P<w>\b(?:%s)\b)|(?P<a3>%s)' % (_word_alt(_GLOBAL_TEXT_WORDS), A3_RE.pattern))
_ALIAS_TEXT_RE = re.compile(r'\b(?:%s)\b' % _word_alt(A1_ALIASES))
_KFORM_RE = re.compile(r'(?<![\\\w])(k\d*)(U|D)(?=[ijkdn]?\b|\d|_)')   # padded_frames.py comments: kU/kD and forms


def text_matches(text: str, line_no: int, file: str, scoped_text: typ.Sequence[Rule]
                 ) -> typ.List[typ.Tuple[int, int, str, str, str, str]]:
    """Every (col_start, col_end, old, new, tier, rule) of the text path on one line of text (non-overlapping,
    leftmost first, scoped rules taking precedence at a position)."""
    cands = []
    for r in scoped_text:
        if r.on == 'text-kform':
            for m in _KFORM_RE.finditer(text):
                new = m.group(1) + ('D' if m.group(2) == 'U' else 'U')
                cands.append((m.start(), m.start() + len(m.group(1)) + 1, m.group(0), new, r.tier, r.note, 0))
            continue
        pat = re.compile(r'(?<![\\\w])(?:%s)(?![\w])' % _word_alt(r.map))
        for m in pat.finditer(text):
            cands.append((m.start(), m.end(), m.group(0), r.map[m.group(0)], r.tier, r.note, 0))
    if file == A1_ALIASES_FILE and any(a <= line_no <= b for a, b in A1_ALIASES_LINES):
        for m in _ALIAS_TEXT_RE.finditer(text):
            cands.append((m.start(), m.end(), m.group(0), A1_ALIASES[m.group(0)], 'A1-alias', 'A1 aliases', 1))
    for m in _GLOBAL_TEXT_RE.finditer(text):
        if m.group('w'):
            new, tier = _GLOBAL_TEXT_WORDS[m.group('w')]
            cands.append((m.start(), m.end(), m.group(0), new, tier, tier, 2))
        else:
            old = m.group('a3')
            cands.append((m.start(), m.end(), old, A3_FAMILY[old], 'A3', 'A3 family', 2))
    cands.sort(key=lambda c: (c[0], c[6], -(c[1] - c[0])))
    out, end = [], -1
    for c in cands:
        if c[0] >= end:
            out.append(c[:6])
            end = c[1]
    return out


def map_text_a_tiers(text: str) -> str:
    """The A-tier text map, applied directly (simultaneous: one regex pass). For the battery's text comparison
    (§5.1: "Texts are equal after the A-tier map")."""
    lines = text.split('\n')
    out = []
    for ln in lines:
        pieces, last = [], 0
        for c0, c1, _old, new, _tier, _rule in text_matches(ln, 0, '', ()):
            pieces += [ln[last:c0], new]
            last = c1
        pieces.append(ln[last:])
        out.append(''.join(pieces))
    return '\n'.join(out)


# ===================================================================================================================
# PASS 1
# ===================================================================================================================

STOREWORD_NAME_RE = re.compile(r'(?i)(?<![a-z])(?:up|down)(?![a-z])')   # a store-word segment in a NAME
WIDTH_NAME_RE = re.compile(r'n[UDO](?![a-z])')                           # width letters in a NAME


class Pass1:
    """Collects hits for one file and writes its sentinelized text."""

    def __init__(self, entries: Entries, fences: typ.Sequence[Fence], sites: typ.Sequence[LetterSite]):
        self.entries = entries
        self.fences = fences
        self.sites = sites
        self.reports: typ.Dict[str, typ.List[str]] = collections.defaultdict(list)
        self.rule_hits: typ.Counter = collections.Counter()
        self.site_hits: typ.Counter = collections.Counter()
        self.unmapped: typ.Dict[str, typ.List[typ.Tuple[str, int]]] = collections.defaultdict(list)
        self.record_names: typ.Optional[typ.List[typ.Tuple[str, int, int, str, str]]] = None   # set to [] to record

    # ---- rule scoping ---------------------------------------------------------------------------------------------
    def _rules_for(self, path: str, scopes: typ.Optional[FileScopes]) -> typ.List[typ.Tuple[Rule, typ.Callable]]:
        out = []
        for r in SCOPED:
            if r.file != path:
                continue
            spans: typ.Optional[typ.List[typ.Tuple[int, int]]] = None
            if r.funcs is not None:
                assert scopes is not None
                spans = []
                for fn in r.funcs:
                    if fn == '<<utv_sampling transposes>>':
                        names = [k for k in scopes.funcs if 'transpose' in k or k == '_remasked_transpose_output']
                        self.reports['scope'].append('%s: utv_sampling transposes resolved to %s' % (path, names))
                        for k in names:
                            spans += scopes.funcs[k]
                        continue
                    if fn not in scopes.funcs:
                        self.reports['scope-error'].append('%s: rule %r names function %r, not found'
                                                           % (path, r.note, fn))
                        continue
                    if len(scopes.funcs[fn]) > 1:
                        self.reports['scope'].append('%s: function %r defined %d times; rule %r covers all'
                                                     % (path, fn, len(scopes.funcs[fn]), r.note))
                    spans += scopes.funcs[fn]
            if r.on == 'name-enclosing':
                assert scopes is not None and r.lines is not None
                spans = []
                for a, _ in r.lines:
                    e = self._enclosing_unit(scopes, a)
                    spans.append(e)
                    self.reports['scope'].append('%s: B3 line %d -> enclosing unit lines %d-%d (%s)'
                                                 % (path, a, e[0], e[1], qualname_at(scopes, a)))

            def in_scope(line: int, r=r, spans=spans) -> bool:
                if r.on != 'name-enclosing' and r.lines is not None \
                        and not any(a <= line <= b for a, b in r.lines):
                    return False
                if spans is not None and not any(a <= line <= b for a, b in spans):
                    return False
                return True
            out.append((r, in_scope))
        return out

    def _enclosing_unit(self, scopes: FileScopes, line: int) -> typ.Tuple[int, int]:
        """The innermost def holding ``line``; or, inside a docstring, that docstring's span."""
        for (dl, dc), (s0, s1) in self._doc_spans.items():
            if s0 <= line <= s1:
                return (s0, s1)
        c = [s for s in scopes.spans if s[0] <= line <= s[1]]
        if not c:
            return (1, 10 ** 9)
        s = min(c, key=lambda s: s[1] - s[0])
        return (s[0], s[1])

    # ---- the per-token lookups ------------------------------------------------------------------------------------
    def name_lookup(self, path: str, line: int, col: int, tok: str, rules) -> typ.Optional[typ.Tuple[str, str, str]]:
        """(new, tier, rule) for a NAME token, or None."""
        site = [s for s in self.sites if s.file == path and s.line == line and s.old == tok
                and (s.col is None or s.col == col)]
        if site:
            assert len({(s.new) for s in site}) == 1, site
            self.site_hits[site[0].id] += 1
            return site[0].new, 'C-site', 'letter_sites ' + site[0].id
        found = []
        for r, in_scope in rules:
            if r.on not in ('name', 'name+text', 'name-enclosing'):
                continue
            if tok in r.map and in_scope(line):
                found.append(r)
        if found:
            specific = [r for r in found if r.lines is not None or r.funcs is not None]
            pick = specific or found
            news = {r.map[tok] for r in pick}
            if len(news) > 1:
                self.reports['conflict'].append('%s:%d:%d %s -> %s (rules %s)'
                                                % (path, line, col, tok, news, [r.note for r in pick]))
            r = pick[0]
            return r.map[tok], r.tier, r.note
        if tok in A1_RENAMED:
            return A1_RENAMED[tok], 'A1', 'A1'
        if tok in A1_ALIASES:
            if path == A1_ALIASES_FILE and any(a <= line <= b for a, b in A1_ALIASES_LINES):
                return A1_ALIASES[tok], 'A1-alias', 'A1 aliases'
            self.reports['alias-outside'].append('%s:%d %s (A1 alias outside its scope: kept)' % (path, line, tok))
            return None
        if tok in A2_SWAP:
            return A2_SWAP[tok], 'A2', 'A2'
        if tok in A2_ONE_WAY:
            return A2_ONE_WAY[tok], 'A2', 'A2 one-way'
        if tok in A3_EXPLICIT:
            return A3_EXPLICIT[tok], 'A3-explicit', 'A3 explicit'
        m = A3_RE.match(tok)
        if m:
            return A3_FAMILY[m.group(1)] + tok[2:], 'A3', 'A3 family'
        if tok in B1_SWAP:
            if any(path == f and a <= line <= b for f, a, b in KEEP_BARE):
                self.reports['keep-bare'].append('%s:%d %s (KEEP_BARE)' % (path, line, tok))
                return None
            return B1_SWAP[tok], 'B1', 'B1'
        return None

    def string_literal_lookup(self, path: str, line: int, content: str, quote_ok: bool, rules
                              ) -> typ.Optional[typ.Tuple[str, str, str]]:
        """(new content, tier, rule) for an exact string literal, or None."""
        if not quote_ok:
            return None
        for r, in_scope in rules:
            if r.on == 'str' and content in r.map and in_scope(line):
                return r.map[content], r.tier, r.note
        if content in B2_SWAP:
            return B2_SWAP[content], 'B2', 'B2'
        return None

    # ---- the file passes ------------------------------------------------------------------------------------------
    def python_file(self, path: str, src: str) -> typ.Tuple[str, typ.List[Hit]]:
        scopes = file_scopes(src)
        line_starts = [0]
        for ln in src.splitlines(keepends=True):
            line_starts.append(line_starts[-1] + len(ln))
        self._doc_spans = {}
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
        for t in toks:
            if t.type == tokenize.STRING and (t.start[0], t.start[1]) in scopes.doc_lines:
                self._doc_spans[(t.start[0], t.start[1])] = (t.start[0], t.end[0])
        rules = self._rules_for(path, scopes)
        text_rules_all = [(r, s) for r, s in rules if r.on in ('name+text', 'text', 'text-kform')]
        fi = fence_intervals(self.fences, path)
        hits: typ.List[Hit] = []

        def add(line, col, ctx, old, new, tier, rule):
            hits.append(Hit(path, line, col, ctx, tier, old, new, rule, qualname_at(scopes, line)))

        def text_line(line, col0, text, ctx):
            scoped = [r for r, s in text_rules_all if s(line)]
            for c0, c1, old, new, tier, rule in text_matches(text, line, path, scoped):
                if _in_fence(fi, line, col0 + c0, col0 + c1):
                    self.reports['fenced-skip'].append('%s:%d:%d %s (fenced)' % (path, line, col0 + c0, old))
                    continue
                add(line, col0 + c0, ctx, old, new, tier, rule)

        def code_tokens(tok_iter, line_map, ctx_prefix):
            """Process python tokens; line_map(row, col) -> (file line, file col)."""
            for t in tok_iter:
                if t.type not in (tokenize.NAME, tokenize.COMMENT, tokenize.STRING):
                    continue
                fl, fc = line_map(t.start[0], t.start[1])
                if t.type == tokenize.NAME:
                    if self.record_names is not None:
                        self.record_names.append((path, fl, fc, t.string, ctx_prefix + 'NAME'))
                    r = self.name_lookup(path, fl, fc, t.string, rules)
                    if r:
                        add(fl, fc, ctx_prefix + 'NAME', t.string, r[0], r[1], r[2])
                    elif STOREWORD_NAME_RE.search(t.string) or WIDTH_NAME_RE.search(t.string):
                        self.unmapped[t.string].append((path, fl))
                elif t.type == tokenize.COMMENT:
                    text_line(fl, fc, t.string, ctx_prefix + 'COMMENT')
                elif t.type == tokenize.STRING:
                    string_token(t, line_map, ctx_prefix)

        def string_token(t, line_map, ctx_prefix):
            s = t.string
            m = re.match(r'([A-Za-z]*)("""|\'\'\'|"|\')', s)
            prefix, q = m.group(1), m.group(2)
            body = s[len(prefix) + len(q): len(s) - len(q)]
            fl0, fc0 = line_map(t.start[0], t.start[1])
            body_col0 = fc0 + len(prefix) + len(q)
            if 'f' not in prefix.lower():
                r = self.string_literal_lookup(path, fl0, body, len(q) == 1 and '\n' not in body, rules)
                if r:
                    if _in_fence(fi, fl0, body_col0, body_col0 + len(body)):
                        self.reports['fenced-skip'].append('%s:%d literal %r (fenced)' % (path, fl0, body))
                    else:
                        add(fl0, body_col0, ctx_prefix + 'STRLIT', body, r[0], r[1], r[2])
                    return
            else:
                self._report_fstring(path, fl0, body, rules)
            is_doc = ctx_prefix == '' and (t.start[0], t.start[1]) in scopes.doc_lines
            example_spans = []                              # (start, end) offsets in body of example SOURCE text
            if '>>>' in body:
                for em in doctest.DocTestParser._EXAMPLE_RE.finditer(body):
                    example_spans.append((em.start('source'), em.end('source'), em))
                if example_spans and not is_doc:
                    self.reports['doctest-in-nondoc'].append('%s:%d string with >>> examples (not a docstring)'
                                                             % (path, fl0))
            # body offsets -> file (line, col)
            body_line_starts = [0]
            for ln in body.splitlines(keepends=True):
                body_line_starts.append(body_line_starts[-1] + len(ln))

            def body_pos(off):
                i = bisect.bisect_right(body_line_starts, off) - 1
                col = off - body_line_starts[i]
                return fl0 + i, (body_col0 + col if i == 0 else col)
            # text path over the body minus example sources
            mask = [False] * len(body)
            for a, b, _ in example_spans:
                for k in range(a, b):
                    mask[k] = True
            ctx_text = (ctx_prefix + ('DOCPROSE' if is_doc else 'STRING'))
            off = 0
            for ln in body.splitlines(keepends=True):
                fl, fc = body_pos(off)
                seg = ln.rstrip('\n')
                # blank out example-source characters so they never match the text path
                seg_masked = ''.join(' ' if mask[off + k] else ch for k, ch in enumerate(seg))
                text_line(fl, fc, seg_masked, ctx_text)
                off += len(ln)
            # code path over the example sources
            for a, b, em in example_spans:
                src_lines = body[a:b].splitlines(keepends=True)
                indent = len(em.group('indent'))
                stripped, origin = [], []
                o = a
                for sl in src_lines:
                    k = indent + (4 if len(sl.rstrip('\n')) > indent + 3 else 3)
                    stripped.append(sl[k:])
                    origin.append(body_pos(o + k))
                    o += len(sl)
                code = ''.join(stripped)
                if not code.endswith('\n'):
                    code += '\n'

                def lm(row, col, origin=origin):
                    fl, fc = origin[row - 1]
                    return fl, fc + col
                try:
                    code_tokens(tokenize.generate_tokens(io.StringIO(code).readline), lm, 'DOCTEST-')
                except (tokenize.TokenError, IndentationError, SyntaxError) as exc:
                    self.reports['doctest-tokenize-error'].append('%s:%d %s' % (path, origin[0][0], exc))

        code_tokens(iter(toks), lambda r, c: (r, c), '')
        hits = self._dedupe(hits)
        return self._write(src, line_starts, hits), hits

    def _report_fstring(self, path, line, body, rules):
        for expr in re.findall(r'\{([^{}]*)\}', body):
            for name in re.findall(r'[A-Za-z_]\w*', expr):
                if name in B1_SWAP or any(name in r.map for r, s in rules if r.on in ('name', 'name+text')
                                          and s(line)):
                    self.reports['fstring'].append('%s:%d f-string expression {%s}: NAME %r takes no NAME rule '
                                                   '(text path only)' % (path, line, expr, name))

    def doc_file(self, path: str, text: str) -> typ.Tuple[str, typ.List[Hit]]:
        fi = fence_intervals(self.fences, path)
        hits = []
        line_starts = [0]
        for i, ln in enumerate(text.splitlines(keepends=True)):
            line_starts.append(line_starts[-1] + len(ln))
            for c0, c1, old, new, tier, rule in text_matches(ln.rstrip('\n'), i + 1, path, ()):
                if _in_fence(fi, i + 1, c0, c1):
                    self.reports['fenced-skip'].append('%s:%d:%d %s (fenced)' % (path, i + 1, c0, old))
                    continue
                hits.append(Hit(path, i + 1, c0, 'DOC', tier, old, new, rule, '<doc>'))
        return self._write(text, line_starts, hits), hits

    @staticmethod
    def _dedupe(hits: typ.List[Hit]) -> typ.List[Hit]:
        seen, out = set(), []
        for h in hits:
            if (h.line, h.col) in seen:
                raise AssertionError('two hits at one position: %s' % (h,))
            seen.add((h.line, h.col))
            out.append(h)
        return out

    def _write(self, src: str, line_starts: typ.List[int], hits: typ.List[Hit]) -> str:
        edits = sorted(((line_starts[h.line - 1] + h.col, h) for h in hits), key=lambda e: e[0])
        out, last = [], 0
        for off, h in edits:
            assert src[off:off + len(h.old)] == h.old, (h, src[off:off + len(h.old) + 10])
            assert off >= last, ('overlapping edits', h)
            out += [src[last:off], SENTINEL % self.entries.key(h.old, h.new)]
            last = off + len(h.old)
        out.append(src[last:])
        return ''.join(out)


# ===================================================================================================================
# PASS 2, bijection, files
# ===================================================================================================================

def resolve(text: str, entries: Entries) -> str:
    """PASS 2: every sentinel -> its entry's new token."""
    return SENTINEL_RE.sub(lambda m: entries.pairs[int(m.group(1))][1], text)


def python_paths() -> typ.List[str]:
    out = []
    for top in ('t3toolbox', 'tests', 'examples'):
        for d, _, fs in os.walk(os.path.join(REPO, top)):
            if '__pycache__' in d:
                continue
            out += [os.path.relpath(os.path.join(d, f), REPO) for f in fs if f.endswith('.py')]
    return sorted(out)


DOC_EXCLUDED = ('docs/release_notes.md',)       # records (§5.3 carve-outs); CHANGELOG.md is not in the doc set


def doc_paths() -> typ.List[str]:
    out = []
    for f in sorted(os.listdir(os.path.join(REPO, 'docs'))):
        if f.endswith(('.md', '.rst')):
            out.append('docs/' + f)
    for f in sorted(os.listdir(os.path.join(REPO, 'docs', 'contributor'))):
        if not f.startswith('.') and os.path.isfile(os.path.join(REPO, 'docs', 'contributor', f)):
            out.append('docs/contributor/' + f)
    out += ['README.md', 'CLAUDE.md']
    return [p for p in out if p not in DOC_EXCLUDED]


def _pair_counts(text: str) -> typ.Counter:
    c: typ.Counter = collections.Counter()
    for w in A2_SWAP:
        c[w] = len(re.findall(r'\b%s\b' % re.escape(w), text))
    for m in A3_RE.finditer(text):
        c[m.group(1) if m.group(1) != 'nO' else 'nU'] += 1
    for w in A3_EXPLICIT:
        c[w] = len(re.findall(r'(?<![\\\w])%s(?![\w])' % re.escape(w), text))
    return c


def bijection_errors(before: str, after: str, path: str, hits: typ.Sequence[Hit],
                     fences: typ.Sequence[Fence]) -> typ.List[str]:
    """Per file and swap pair (a, b): count_after(a) == count_before(b) and count_after(b) == count_before(a), counted
    outside fences, with nO counted into nU. One-way and scoped hits that make or consume a pair member (nO -> nU,
    nU -> n, nDs -> nUs, A3 scoped) are reported as explained deviations, not errors."""
    def unfenced(text):
        fi = fence_intervals(fences, path)
        lines = text.split('\n')
        return '\n'.join(''.join(' ' if _in_fence(fi, i + 1, k, k + 1) else ch for k, ch in enumerate(ln))
                         if (i + 1) in fi else ln for i, ln in enumerate(lines))
    cb, ca = _pair_counts(unfenced(before)), _pair_counts(unfenced(after))
    made, used = collections.Counter(), collections.Counter()   # one-way/scoped hits that make or consume a member
    for h in hits:
        if h.tier in ('A2', 'A3', 'A3-explicit') and h.rule != 'A2 one-way' and not (h.old == 'nO'):
            continue
        made[h.new if h.new != 'nO' else 'nU'] += 1
        used[h.old if h.old != 'nO' else 'nU'] += 1
    out = []
    pairs = [(a, b) for a, b in A2_TWINS.items()] + [('nU', 'nD'), ('_NU', '_ND'), ('nnU', 'nnD')]
    for a, b in pairs:
        if ca[a] != cb[b] or ca[b] != cb[a]:
            expl = (ca[a] - cb[b] == made[a] - used[b] and ca[b] - cb[a] == made[b] - used[a])
            out.append('%s: %s/%s before %d/%d after %d/%d -- %s' % (
                path, a, b, cb[a], cb[b], ca[a], ca[b],
                'EXPLAINED by one-way/scoped hits (made %s %d, %s %d; consumed %s %d, %s %d)'
                % (a, made[a], b, made[b], a, used[a], b, used[b]) if expl else 'ERROR'))
    return out


# ===================================================================================================================
# The dry run
# ===================================================================================================================

def dry_run(out_dir: str, scratch: str) -> int:
    fences = load_fences()
    sites = load_letter_sites()
    entries = Entries()
    p1 = Pass1(entries, fences, sites)
    all_hits: typ.List[Hit] = []
    sentinel_written: typ.Counter = collections.Counter()
    pass1_dir, pass2_dir = os.path.join(scratch, 'pass1'), os.path.join(scratch, 'pass2')
    for d in (pass1_dir, pass2_dir):
        if os.path.exists(d):
            shutil.rmtree(d)
    bij: typ.List[str] = []
    files = [(p, 'py') for p in python_paths()] + [(p, 'doc') for p in doc_paths()]
    for path, kind in files:
        src = open(os.path.join(REPO, path), encoding='utf-8').read()
        if kind == 'py':
            s1, hits = p1.python_file(path, src)
        else:
            s1, hits = p1.doc_file(path, src)
        for m in SENTINEL_RE.finditer(s1):
            sentinel_written[int(m.group(1))] += 1
        # sentinels must not survive in old text (no pass-1 regex can match inside one)
        s2 = resolve(s1, entries)
        assert not SENTINEL_RE.search(s2), path
        for d, s in ((pass1_dir, s1), (pass2_dir, s2)):
            os.makedirs(os.path.dirname(os.path.join(d, path)), exist_ok=True)
            with open(os.path.join(d, path), 'w', encoding='utf-8') as fh:
                fh.write(s)
        bij += bijection_errors(src, s2, path, hits, fences)
        all_hits += hits
    # the LaTeX notes are never touched (tier A never applies in LaTeX): copied verbatim so the pass trees are
    # complete for the gates (K11 locates every fenced span, the notes' included)
    for f in sorted(os.listdir(os.path.join(REPO, 'docs'))):
        if f.endswith('.tex'):
            for d in (pass1_dir, pass2_dir):
                os.makedirs(os.path.join(d, 'docs'), exist_ok=True)
                shutil.copyfile(os.path.join(REPO, 'docs', f), os.path.join(d, 'docs', f))
    # sentinel report: pass-1 count per key == hit-table count per key
    hit_count = collections.Counter(entries.key(h.old, h.new) for h in all_hits)
    mismatch = [k for k in set(hit_count) | set(sentinel_written) if hit_count[k] != sentinel_written[k]]
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'R1_hit_table.tsv'), 'w') as fh:
        fh.write('file\tline\tcol\tctx\ttier\told\tnew\trule\tscope\n')
        for h in sorted(all_hits, key=lambda h: (h.file, h.line, h.col)):
            fh.write('\t'.join(map(str, (h.file, h.line, h.col, h.ctx, h.tier, h.old, h.new, h.rule, h.scope))) + '\n')
    with open(os.path.join(out_dir, 'R1_sentinels.tsv'), 'w') as fh:
        fh.write('key\tsentinel\told\tnew\thits\tsentinels_pass1\n')
        for k, (o, n) in enumerate(entries.pairs):
            fh.write('%d\t%s\t%s\t%s\t%d\t%d\n' % (k, SENTINEL % k, o, n, hit_count[k], sentinel_written[k]))
    write_report(os.path.join(out_dir, 'R1_apply_report.md'), all_hits, entries, p1, fences, sites, bij, mismatch,
                 scratch)
    print('hits %d, keys %d, files changed %d, sentinel mismatches %d, bijection errors %d (explained deviations '
          '%d), conflicts %d, scope errors %d' % (
              len(all_hits), len(entries.pairs), len({h.file for h in all_hits}), len(mismatch),
              sum(b.endswith('ERROR') for b in bij), sum(not b.endswith('ERROR') for b in bij),
              len(p1.reports['conflict']), len(p1.reports['scope-error'])))
    return 0


def write_report(path, hits, entries, p1, fences, sites, bij, mismatch, scratch):
    by_tier_file: typ.Dict[str, typ.Counter] = collections.defaultdict(collections.Counter)
    for h in hits:
        by_tier_file[h.file][h.tier] += 1
    tiers = sorted({h.tier for h in hits})
    L = ['# R1 dry run of `dev/store_rename_apply.py`', '',
         'Generated by `python dev/store_rename_apply.py --dry-run`; nothing in the tree was written. Pass-1 and '
         'pass-2 trees: `%s`.' % scratch, '',
         '- hits: **%d** in **%d** files; ENTRIES keys: **%d**' % (len(hits), len(by_tier_file), len(entries.pairs)),
         '- sentinel keys whose pass-1 count differs from the hit table: **%d**; sentinels left after pass 2: **0** '
         '(asserted per file)' % len(mismatch),
         '- fences loaded: **%d** (%d fenced); letter-site rows loaded (slice R2): **%d**'
         % (len(fences), sum(f.fenced for f in fences), len(sites)),
         '- bijection: **%d** errors; **%d** deviations explained by one-way/scoped hits (listed below)'
         % (sum(b.endswith('ERROR') for b in bij), sum(not b.endswith('ERROR') for b in bij)), '',
         '## Hits per tier per file', '',
         '| file | ' + ' | '.join(tiers) + ' | total |', '|---|' + '---|' * (len(tiers) + 1)]
    for f in sorted(by_tier_file):
        c = by_tier_file[f]
        L.append('| `%s` | ' % f + ' | '.join(str(c[t]) if c[t] else '' for t in tiers) + ' | %d |' % sum(c.values()))
    tot = collections.Counter(h.tier for h in hits)
    L.append('| **total** | ' + ' | '.join('**%d**' % tot[t] for t in tiers) + ' | **%d** |' % len(hits))
    L += ['', '## Hits per rule (scoped rules)', '', '| rule | tier | file | hits | tokens |', '|---|---|---|---|---|']
    per_rule = collections.defaultdict(collections.Counter)
    for h in hits:
        per_rule[(h.rule, h.tier, h.file)]['%s->%s' % (h.old, h.new)] += 1
    for r in SCOPED:
        key = (r.note, r.tier, r.file)
        c = per_rule.get(key, collections.Counter())
        L.append('| %s | %s | `%s` | %d | %s |' % (r.note, r.tier, r.file, sum(c.values()),
                                                  ', '.join('`%s` %d' % kv for kv in sorted(c.items())) or '**none**'))
    L += ['', '## Fresh tokenizer scan: NAME tokens with a store-word segment or width letters that NO rule maps',
          '', 'Every such identifier, with its occurrences; the disposition column is `dev/store_rename_slices/'
          'R1_unmapped_dispositions.tsv` (R1 curated).', '', '| identifier | n | files (lines) |', '|---|---|---|']
    for tok in sorted(p1.unmapped):
        occ = p1.unmapped[tok]
        byf = collections.defaultdict(list)
        for f, ln in occ:
            byf[f].append(ln)
        L.append('| `%s` | %d | %s |' % (tok, len(occ), '; '.join('%s %s' % (f.split('/')[-1], ','.join(
            map(str, sorted(set(v))[:8])) + (',…' if len(set(v)) > 8 else '')) for f, v in sorted(byf.items()))))
    for k, v in p1.reports.items():
        L += ['', '## Report: %s (%d)' % (k, len(v)), '']
        L += ['- ' + x for x in (v if k != 'fenced-skip' else v)]
    if bij:
        L += ['', '## Bijection', ''] + ['- ' + b for b in bij]
    with open(path, 'w') as fh:
        fh.write('\n'.join(L) + '\n')


def main(argv: typ.Optional[typ.Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--i-am-r2', action='store_true')
    ap.add_argument('--out', default=SLICES)
    ap.add_argument('--scratch', default=os.path.join(os.environ.get('TMPDIR', '/tmp'), 'store_rename_dry_run'))
    a = ap.parse_args(argv)
    if a.apply:
        if not a.i_am_r2:
            raise SystemExit('--apply writes the tree: R2 only (pass --i-am-r2). R1 runs --dry-run.')
        return apply(a.out, a.scratch)
    return dry_run(a.out, a.scratch)


def apply(out_dir: str, scratch: str) -> int:
    """R2: both passes into ``scratch`` (as the dry run), the single reviewable pass-1 diff
    (``<out_dir>/R2_pass1.diff``: every file's sentinelized text against HEAD), then pass 2 copied over the tree, and
    the zero-sentinel gate (K0) on every written file. Refuses unless every target file equals HEAD."""
    import difflib
    import subprocess
    paths = python_paths() + doc_paths()
    dirty = subprocess.run(['git', '-C', REPO, 'diff', '--name-only', 'HEAD', '--'] + paths, capture_output=True,
                           text=True, check=True).stdout.split()
    if dirty:
        raise SystemExit('refusing: target files differ from HEAD: %s' % dirty[:10])
    dry_run(out_dir, scratch)
    pass1_dir, pass2_dir = os.path.join(scratch, 'pass1'), os.path.join(scratch, 'pass2')
    diff, written, placed, removed = [], [], 0, 0
    for path in paths:
        src = open(os.path.join(REPO, path), encoding='utf-8').read()
        s1 = open(os.path.join(pass1_dir, path), encoding='utf-8').read()
        if s1 == src:
            continue
        s2 = open(os.path.join(pass2_dir, path), encoding='utf-8').read()
        placed += len(SENTINEL_RE.findall(s1))
        diff += difflib.unified_diff(src.splitlines(keepends=True), s1.splitlines(keepends=True),
                                     'a/' + path, 'b/' + path, n=0)
        assert not SENTINEL_RE.search(s2), path
        with open(os.path.join(REPO, path), 'w', encoding='utf-8') as fh:
            fh.write(s2)
        removed += len(SENTINEL_RE.findall(s1)) - len(SENTINEL_RE.findall(open(os.path.join(REPO, path),
                                                                                 encoding='utf-8').read()))
        written.append(path)
    with open(os.path.join(out_dir, 'R2_pass1.diff'), 'w', encoding='utf-8') as fh:
        fh.writelines(diff)
    left = [p for p in paths if SENTINEL_RE.search(open(os.path.join(REPO, p), encoding='utf-8').read())]
    print('applied: %d files written; sentinels placed in pass 1 %d, removed by pass 2 %d; sentinels left in the '
          'tree %d (K0)' % (len(written), placed, removed, len(left)))
    return 1 if left else 0


if __name__ == '__main__':
    sys.exit(main())
