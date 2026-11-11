"""R1 gate: reconcile the dry-run hit table with inv 2 §8 (key by key) and inv 1 §7 (line by line).

Usage::

    python dev/store_rename_slices/r1_reconcile.py <dir holding R1_hit_table.tsv> > dev/store_rename_slices/R1_reconciliation.md

Every difference gets an explanation: automatic where a plan section decides it (cited), by hand otherwise
(``MANUAL`` below, each entry a reviewed judgement).
"""
import collections
import csv
import os
import re
import sys
import typing as typ

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INV1 = os.path.join(REPO, 'dev', 'store_rename_inventory_1_frontend_tests.md')
INV2 = os.path.join(REPO, 'dev', 'store_rename_inventory_2_backend.md')

# inv 2 §8 identifiers the plan does not map, with the plan section that says so
KEPT_BY_PLAN: typ.Dict[str, str] = {}
for _w in ('rL', 'rR', 'rL0', 'rL1', 'rR0', 'rR1', 'rR_d', 'rLi', 'rRi'):
    KEPT_BY_PLAN[_w] = 'kept: rL/rR untouched (D-3); inv 2 proposed word forms, withdrawn by D-8'
for _w in ('kL', 'kR', 'mL', 'mR', 'mL0', 'mR0', 'mL1', 'mR1', 'pL', 'pR', 'iL', 'iR', 'WL', 'WR'):
    KEPT_BY_PLAN[_w] = 'kept: the L/R members of the A3 scoped swaps are unchanged (§1.2 A3 scoped); inv 2 word form withdrawn'


# inv 2 §8 differences explained by a later ruling (the plan section that decides each)
MANUAL_DIFF: typ.Dict[str, str] = {
    'BU': 'EXPLAINED: t3_linalg 837-995 are the inner-product sweep functions, whose `BU` (z\'s Tucker cores against '
          'theta\'s) tier C\'s B rows map `BU->zB` (D-6, draft 15); inv 2 predates D-6',
    'D': 'EXPLAINED: inv 2 has two `D` rows (fv_* `D->U`, padded frames `D->E`), merged here; the plan maps the padded '
         'frames\' `D->Psi` (D-5) and adds `entries.py:361-363` `D->n_cores` (tier C), which inv 2 left as prose-only',
    'U': 'EXPLAINED: geometry 395-418 and t3_operations 89 are a point\'s own Tucker cores, tier C\'s B rows `U->B` '
         '(D-6, draft 15); inv 2 predates D-6',
    'Ud': 'EXPLAINED: utv_operations 740-748 is `utv_corewise_lift_transpose`, §1.2a `Ud->Bd` (D-12); inv 2 kept the U '
          'family by design (M4, C1)',
}
EXTRA_WHY: typ.Dict[str, str] = {
    'P': 'D-7 (frame P -> L); inv 2 kept P/Q by design (M4)', 'Q': 'D-7 (Q -> R)', 'P_cores': 'D-7',
    'P_cores_dot': 'D-7', 'P_dot': 'D-7', 'Pd': 'D-7', 'Pxi': 'D-7', 'dPxi': 'D-7', 'Q_dot': 'D-7', 'Qxi': 'D-7',
    'rev_Q': 'D-7', 'BUd': 'tier C B rows (D-6)', 'UL': 'tier C B rows (D-6)', 'UR': 'tier C B rows (D-6)',
    'U_i_o': 'tier C B rows (D-6)', 'Ua': 'tier C B rows (D-6)', 'Ub': 'tier C B rows (D-6)', 'Uio': 'tier C B rows',
    'Utilde': 'tier C B rows', 'Utilde_full': 'tier C B rows', 'Ux': 'tier C B rows', 'Uxi': 'tier C B rows',
    'Uy': 'tier C B rows', 'Uy_rev': 'tier C B rows', 'Uyi': 'tier C B rows', 'new_Uxo': 'tier C B rows',
    'DM_k': 'tier C ufv_masking rows (sample batch 2 §1)', 'DM_t': 'tier C ufv_masking rows',
    'UM_k': 'tier C ufv_masking rows', 'UM_t': 'tier C ufv_masking rows',
    'Udot': '§1.2a (B forms; the tied ambient direction D forms)', 'Udot_g': '§1.2a', 'dU': '§1.2a (D-12)',
    'dU_t': '§1.2a', 'dU_tilde': '§1.2a', 'dU_tildes': '§1.2a', 'sh': 'D-1 sigma-hat row', 'shd': 'D-1 sigma-hat row',
    'O_ab_x': '§3.4 sites known now (letter_sites)', 'Oaxb': '§3.4 sites known now (letter_sites)',
    # the maintainer's shape ruling (2026-11-11): an array's shape decides its letter
    'B': 'shape ruling: _assemble_tangent_z_step\'s frame core B -> D (a held local)',
    'dB': 'shape ruling: held locals holding a Tucker variation, (nU, N) new -> V',
    'dB_parallel': 'shape ruling: held local -> V_parallel',
    'dG': 'shape ruling: TT-variation-shaped partners -> H; the spec string in _apply_transpose_assemble_core -> gH '
          '(pending the reviewer)',
    'dG1': 'shape ruling: a TT variation -> H1 (pending the reviewer)',
    'dG_t': 'shape ruling: transpose output partner -> H_t', 'dG_tilde': 'shape ruling: -> H_tilde',
    'dG_tildes': 'shape ruling: -> H_tildes', 'dV': 'shape ruling: a TT variation -> H (pending the reviewer)',
}


def load_hits(d: str) -> typ.List[dict]:
    with open(os.path.join(d, 'R1_hit_table.tsv')) as fh:
        return list(csv.DictReader(fh, delimiter='\t'))


def inv2_identifier_rows() -> typ.List[typ.Tuple[str, str, str, int, typ.List[typ.Tuple[str, int]]]]:
    text = open(INV2).read()
    sec = text[text.index('## 8. Identifier map'):text.index('## 9. Per-file line inventory')]
    rows = []
    for ln in sec.splitlines():
        m = re.match(r'\| `([^`]+)` \| (`[^`]*`|→ same) \| ([A-Z\-]+) \| (\d+) \| (.*) \|$', ln)
        if not m:
            continue
        ident, new, cat, n, locs = m.groups()
        occ = []
        for part in locs.split(';'):
            part = part.strip()
            if not part:
                continue
            mod, nums = part.split(' ', 1)
            occ += [(mod, int(x)) for x in nums.split(',')]
        rows.append((ident, new.strip('`'), cat, int(n), occ))
    return rows


def inv1_rows() -> typ.List[typ.Tuple[str, int, str, str, str, str]]:
    """(file, line, cat, current, proposed, note) for every main-table row (§7) and App. A row."""
    text = open(INV1).read()
    out = []
    main = text[text.index('## 7. Main table'):text.index('## Appendix A.')]
    cur = None
    for ln in main.splitlines():
        m = re.match(r'### `([^`]+)`', ln)
        if m:
            cur = m.group(1)
            continue
        m = re.match(r'\| (\d+) \| ([^|]*) \| (.*) \|$', ln)
        if m and cur:
            cells = _split_cells(ln)
            if len(cells) >= 5:
                out.append((cur, int(cells[0]), cells[1], cells[2], cells[3], cells[4]))
    appa = text[text.index('## Appendix A.'):text.index('## Appendix B.')]
    for ln in appa.splitlines():
        cells = _split_cells(ln)
        if len(cells) >= 5 and cells[0].startswith('`') and cells[1].isdigit():
            out.append((cells[0].strip('`'), int(cells[1]), 'APP-A', cells[2], cells[3], cells[4]))
    return out


def _split_cells(ln: str) -> typ.List[str]:
    """Split a markdown table row on unescaped pipes."""
    if not ln.startswith('|'):
        return []
    cells, cur, i = [], '', 1
    while i < len(ln):
        ch = ln[i]
        if ch == '\\' and i + 1 < len(ln) and ln[i + 1] == '|':
            cur += '|'
            i += 2
            continue
        if ch == '|':
            cells.append(cur.strip())
            cur = ''
        else:
            cur += ch
        i += 1
    return cells


def main(d: str) -> None:
    hits = load_hits(d)
    P = print
    P('# R1 reconciliation: the dry-run hit table vs inv 2 §8, inv 1 §7 (+ App. A)')
    P('')
    P('Generated by `python dev/store_rename_slices/r1_reconcile.py`. A MATCH needs no explanation; every other row '
      'carries one.')
    P('')
    # ---------------------------------------------------------------- inv 2 §8, key by key (backend NAME tokens)
    be = collections.defaultdict(collections.Counter)            # ident -> Counter((module, line))
    be_new = collections.defaultdict(set)
    for h in hits:
        if h['file'].startswith('t3toolbox/backend/') and h['ctx'] == 'NAME':
            mod = os.path.basename(h['file'])[:-3]
            be[h['old']][(mod, int(h['line']))] += 1
            be_new[h['old']].add(h['new'])
    raw_rows = inv2_identifier_rows()
    merged = collections.OrderedDict()
    for ident, new, cat, n, occ in raw_rows:
        if ident in merged:
            o = merged[ident]
            merged[ident] = (ident, o[1] + ' | ' + new, o[2], o[3] + n, o[4] + occ)
        else:
            merged[ident] = (ident, new, cat, n, occ)
    rows = list(merged.values())
    seen = set()
    P('## inv 2 §8 (backend NAME tokens), key by key')
    P('')
    P('| identifier | inv 2 (cat, n, proposal) | dry run (n, image) | verdict |')
    P('|---|---|---|---|')
    nmatch = 0
    for ident, new, cat, n, occ in rows:
        key = (ident, new, cat)
        seen.add(ident)
        inv = collections.Counter(occ)
        mine = be.get(ident, collections.Counter())
        if new == '→ same' or cat in ('ADJECTIVE', 'UNRELATED'):
            verdict = 'MATCH (kept by both)' if not mine else 'DIFF: inv 2 keeps, the dry run maps it'
        elif ident in KEPT_BY_PLAN:
            verdict = ('MATCH-KEPT: ' + KEPT_BY_PLAN[ident]) if not mine else 'DIFF: kept by plan but mapped'
        elif inv == mine:
            verdict = 'MATCH (same %d locations)' % n
        else:
            only_inv = inv - mine
            only_me = mine - inv
            verdict = 'DIFF: only inv 2 %s; only dry run %s -- %s' % (
                _fmt(only_inv), _fmt(only_me), MANUAL_DIFF.get(ident, 'UNEXPLAINED'))
        if verdict.startswith('MATCH'):
            nmatch += 1
        P('| `%s` | %s, %d, `%s` | %d, %s | %s |' % (ident, cat, n, new, sum(mine.values()),
                                                   ', '.join('`%s`' % x for x in sorted(be_new.get(ident, ()))) or '-',
                                                   verdict))
    extra = sorted(set(be) - seen)
    P('')
    P('%d of %d inv 2 §8 keys match. Backend NAME identifiers the dry run maps that inv 2 §8 has no row for '
      '(%d):' % (nmatch, len(rows), len(extra)))
    P('')
    P('| identifier | n | image | where | why |')
    P('|---|---|---|---|---|')
    for ident in extra:
        rs = [h for h in hits if h['file'].startswith('t3toolbox/backend/') and h['ctx'] == 'NAME'
              and h['old'] == ident]
        why = EXTRA_WHY.get(ident) or ('letter_sites (R1 §3.4 classification)' if all(h['tier'] == 'C-site'
                                                                                      for h in rs) else 'UNEXPLAINED')
        P('| `%s` | %d | %s | %s | %s |' % (ident, len(rs), ', '.join(sorted({'`%s`' % h['new'] for h in rs})),
                                          _where(rs), why))
    # ---------------------------------------------------------------- inv 1 §7, line by line
    P('')
    P('## inv 1 §7 + App. A (frontend, tests, examples), line by line')
    P('')
    hit_lines = collections.defaultdict(list)
    for h in hits:
        hit_lines[(h['file'], int(h['line']))].append(h)
    rows1 = inv1_rows()
    row_lines = {(f, l) for f, l, *_ in rows1}
    unhit = collections.defaultdict(list)
    for f, l, cat, cur, prop, note in rows1:
        changes = prop not in ('(keep)', '') and not prop.startswith('(keep')
        if (f, l) in hit_lines:
            continue
        if not changes and cat != 'APP-A':
            continue
        unhit[_classify_unhit(f, l, cat, cur, prop, note)].append((f, l, cat, cur[:110]))
    P('### inv 1 rows with a change (or App. A rows) where the dry run makes no change, by reason')
    P('')
    for reason in sorted(unhit):
        P('#### %s (%d)' % (reason, len(unhit[reason])))
        P('')
        for f, l, cat, cur in unhit[reason]:
            P('- `%s:%d` [%s] `%s`' % (f, l, cat, cur.replace('`', "'")))
        P('')
    extra1 = sorted((k for k in hit_lines if not k[0].startswith(('t3toolbox/backend/', 'docs/'))
                     and k[0] not in ('README.md', 'CLAUDE.md') and k not in row_lines))
    P('### Dry-run hit lines (frontend, tests, examples) with no inv 1 row (%d)' % len(extra1))
    P('')
    byreason = collections.defaultdict(list)
    for k in extra1:
        hs = hit_lines[k]
        tiers = sorted({h['tier'] for h in hs})
        byreason[', '.join(tiers)].append((k, hs))
    for reason in sorted(byreason):
        P('#### tiers %s (%d lines)' % (reason, len(byreason[reason])))
        P('')
        for (f, l), hs in byreason[reason]:
            P('- `%s:%d` %s' % (f, l, ', '.join('`%s→%s`' % (h['old'], h['new']) for h in hs)))
        P('')


def _classify_unhit(f, l, cat, cur, prop, note) -> str:
    """Why R2's script leaves an inv 1 change row alone (the plan section that owns it)."""
    tags = set(re.split(r'[,\s]+', cat))
    if cat == 'APP-A':
        return 'App. A (optional letters) — classified by §3.4 / letter_sites, prose letters R4'
    if 'LA' in tags:
        return 'LA: the update letter API — R3 (§1.3, §4.5)'
    if re.search(r'BACKWARDS|backwards', note):
        return 'BACKWARDS adjective — tier E hand row, R4a (§4.3)'
    if 'ADJ' in tags:
        return 'ADJ — tier E hand row, R4a'
    if 'CL' in tags:
        return 'CL in prose or a fenced block — tier D/E, §4A or letter_sites (R4a/R4b/R5)'
    if 'WD' in tags:
        return 'WD — inside a fenced block (hand, §4A) or a word-form proposal superseded by D-8'
    if 'SW' in tags or 'AK' in tags:
        return 'SW/AK in free prose — tier E (R4a), K9; or inside a fence'
    return 'other: ' + cat


def _fmt(c: collections.Counter) -> str:
    if not c:
        return '{}'
    byf = collections.defaultdict(list)
    for (m, l), k in sorted(c.items()):
        byf[m] += [str(l)] * k
    return '{' + '; '.join('%s %s' % (m, ','.join(v)) for m, v in byf.items()) + '}'


def _where(rs) -> str:
    byf = collections.defaultdict(list)
    for h in rs:
        byf[os.path.basename(h['file'])].append(h['line'])
    return '; '.join('%s %s' % (f, ','.join(sorted(set(v), key=int))) for f, v in sorted(byf.items()))


if __name__ == '__main__':
    main(sys.argv[1])
