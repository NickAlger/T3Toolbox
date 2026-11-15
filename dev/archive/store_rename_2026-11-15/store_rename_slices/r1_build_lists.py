"""R1: build ``dev/store_rename_dnt.txt`` (§1.5, gate K14) and ``dev/store_rename_keeps.txt`` (K9).

Usage::

    python dev/store_rename_slices/r1_build_lists.py <dir holding R1_hit_table.tsv>

DNT (``dev/store_rename_dnt.txt``), one row per (identifier, scope), tab-separated:
``identifier  scope  item  note`` where scope is ``*`` (every file), ``<file>``, ``<file>::<outermost def>`` or
``<file>:<first>-<last>``. Built from §1.5's items 1-13: explicit lists from the plan, scans of the tree (Greek-stem
names, ``rL``/``rR`` forms, orthogonalization/SVD/unfolding names, the ``dataclasses`` alias, public names) and the
§3.4 inventory's keeps (``dev/store_rename_slices/pqu_keeps.tsv``). The builder PROVES disjointness from the token
map: no hit of the dry run renames a listed identifier inside its scope (it raises otherwise). Images of the map
that spell a listed name (``B``, ``b``, ``n``, ...) are not counted by K14 at their mapped positions.

KEEPS (``dev/store_rename_keeps.txt``), one row per old-tree line, tab-separated:
``file  line  kind  source  text`` with kind ``KEEP`` (K9 (a), and the converse: every store word on the line
survives at its aligned position) or ``EXEMPT`` (a tier-E line whose words change by design: §3.3's ``-ward``
rewordings from the authoritative grep, §4.2's store-and-arrow sites, §4.3's backwards adjectives), plus
``PATTERN`` rows for inv 3 §9's unlisted English classes in the docs.
"""
import ast
import collections
import csv
import importlib.util
import os
import re
import sys
import typing as typ

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_spec = importlib.util.spec_from_file_location('store_rename_apply', os.path.join(REPO, 'dev', 'store_rename_apply.py'))
sra = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sra)
INV1 = os.path.join(REPO, 'dev', 'store_rename_inventory_1_frontend_tests.md')
INV2 = os.path.join(REPO, 'dev', 'store_rename_inventory_2_backend.md')
INV3 = os.path.join(REPO, 'dev', 'store_rename_inventory_3_docs.md')
PQU_KEEPS = os.path.join(REPO, 'dev', 'store_rename_slices', 'pqu_keeps.tsv')

GREEK_RE = re.compile(r'^d?(?:xi|eta|mu|nu|sigma|tau|zeta|theta|omega|rho|kappa|Phi|Xi|Psi)(?:s|_\w+)?$')
ORTH_RE = re.compile(r'orthogonaliz|(?:^|_)svd(?:_|$)|matricization|^_up_res$')
RLR_RE = re.compile(r'^r?r[LR](?:[0-9A-Za-z_()]*)$')


def outer_defs(src: str) -> typ.List[typ.Tuple[int, int, str]]:
    tree = ast.parse(src)
    out = []

    def visit(node, inside):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not inside:
                    first = min([ch.lineno] + [d.lineno for d in ch.decorator_list])
                    out.append((first, ch.end_lineno, ch.name))
                visit(ch, True)
            else:
                visit(ch, inside)
    visit(tree, False)
    return out


def public_names(path: str, src: str) -> typ.Set[str]:
    """Module-level public defs/classes/assignments; public methods, properties and class-level fields."""
    tree = ast.parse(src)
    out = set()
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and not n.name.startswith('_'):
            out.add(n.name)
            if isinstance(n, ast.ClassDef):
                for m in n.body:
                    if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and not m.name.startswith('_'):
                        out.add(m.name)
                    if isinstance(m, ast.AnnAssign) and isinstance(m.target, ast.Name) \
                            and not m.target.id.startswith('_'):
                        out.add(m.target.id)
        if isinstance(n, (ast.Assign, ast.AnnAssign)):
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            for t in targets:
                if isinstance(t, ast.Name) and not t.id.startswith('_') and t.id != '__all__':
                    out.add(t.id)
    return out


def main(hit_dir: str) -> None:
    hits = list(csv.DictReader(open(os.path.join(hit_dir, 'R1_hit_table.tsv')), delimiter='\t'))
    renamed_olds = collections.defaultdict(list)            # old -> [(file, line)]
    for h in hits:
        if h['ctx'].endswith('NAME'):
            renamed_olds[h['old']].append((h['file'], int(h['line'])))
    # every NAME token of the old tree (code + doctests), with its file and line
    entries = sra.Entries()
    p1 = sra.Pass1(entries, sra.load_fences(), sra.load_letter_sites())
    p1.record_names = []
    srcs, outer = {}, {}
    for path in sra.python_paths():
        src = open(os.path.join(REPO, path), encoding='utf-8').read()
        srcs[path] = src
        outer[path] = outer_defs(src)
        p1.python_file(path, src)
    names_by_file = collections.defaultdict(collections.Counter)
    for path, line, col, tok, ctx in p1.record_names:
        names_by_file[path][tok] += 1
    all_names = collections.Counter()
    for c in names_by_file.values():
        all_names.update(c)

    rows: typ.List[typ.Tuple[str, str, str, str]] = []

    def add(ident, scope, item, note):
        rows.append((ident, scope, item, note))

    def files_with(ident):
        return sorted(f for f, c in names_by_file.items() if c[ident])

    def func_scope(path, line):
        o = [s for s in outer[path] if s[0] <= line <= s[1]]
        return '%s::%s' % (path, o[0][2]) if o else '%s:%d-%d' % (path, line, line)

    # ---- item 1: Greek-letter names and their derived forms (store-letter suffix forms are tier C, excluded)
    for ident in sorted(all_names):
        if GREEK_RE.match(ident) and ident not in renamed_olds:
            add(ident, '*', '1', 'Greek-letter name')
    # ---- item 2 (part): decomposition factors that are never a store letter anywhere
    for ident in ('S', 'Vt', 'svd_U', 'svd_Vt', 'U_M', 'Us', 'U_a_x_b', 'U_i_a_x', 'U_o_x', 'U_nk', 'U_T', 'U_a_x'):
        if all_names[ident] and ident not in renamed_olds:
            add(ident, '*', '2', 'decomposition factor')
    # ---- item 3 (part): the derivative direction pp (P, the completion, is scoped: pqu_keeps)
    for ident in ('pp',):
        add(ident, '*', '3', 'derivative direction')
    # ---- item 4: the RNG helpers R
    for path, lines in (('tests/test_jet_recurrence.py', (166, 208, 228)),
                        ('tests/test_probe_derivatives.py', (488, 497))):
        for ln in lines:
            add('R', func_scope(path, ln), '4', 'RNG helper (line %d)' % ln)
    # ---- item 5: the other letters
    for ident in ('G', 'V', 'H', 'K', 'C', 'W', 'N', 'n', 'r', 'd'):
        add(ident, '*', '5', 'house letter')
    b_renamed = collections.defaultdict(set)                  # file -> lines where a NAME B is renamed
    for f, ln in renamed_olds.get('B', ()):
        b_renamed[f].add(ln)
    for path in sorted(names_by_file):
        if not names_by_file[path]['B']:
            continue
        if path not in b_renamed:
            add('B', path, '5', "a T3's own Tucker cores")
            continue
        for s0, s1, fn in outer[path]:                     # per def, skipping the defs that rename B
            if 'B' in _names_in(p1.record_names, path, s0, s1) and not any(s0 <= l <= s1 for l in b_renamed[path]):
                add('B', '%s::%s' % (path, fn), '5', "a T3's own Tucker cores (B is renamed elsewhere in this file: "
                                                      "vertical_direction's gauge B -> M; _assemble_tangent_z_step's "
                                                      "frame core B -> D, the shape ruling)")
    for ident in sorted(all_names):
        if RLR_RE.match(ident) and ident not in renamed_olds:
            add(ident, '*', '5', 'rL/rR and their forms (D-3)')
    for path in sorted(srcs):
        if re.search(r'^import dataclasses as dc$', srcs[path], re.M):
            add('dc', path, '5', 'the dataclasses alias')
    # ---- item 7: orthogonalization / SVD / unfolding names (the fv_conversions aliases excepted)
    for ident in sorted(all_names):
        if ORTH_RE.search(ident):
            if ident in sra.A1_ALIASES:
                for f in files_with(ident):
                    if f != sra.A1_ALIASES_FILE:
                        add(ident, f, '7', 'orthogonalization name (alias renamed only in fv_conversions)')
            elif ident not in renamed_olds:
                add(ident, '*', '7', 'orthogonalization/SVD/unfolding name')
    # ---- item 8: the jet-shift functions and c_up
    for ident in ('_shift_up', '_shift_down', '_shift_up_pair', '_shift_down_pair'):
        add(ident, '*', '8', 'jet-shift function (Nick, 2026-11-11)')
    add('c_up', 'tests/backend/test_curvature.py', '8', 'a shifted jet')
    # ---- item 9: KEEP_BARE locals, inv 2 §8's keep identifiers
    for f, a, b in sra.KEEP_BARE:
        add('up', '%s:%d-%d' % (f, a, b), '9', 'KEEP_BARE (B1 exclusion)')
    inv2 = open(INV2).read()
    sec = inv2[inv2.index('## 8. Identifier map'):inv2.index('## 9. Per-file line inventory')]
    for m in re.finditer(r'^\| `([^`]+)` \| → same \|', sec, re.M):
        if m.group(1) not in renamed_olds:
            add(m.group(1), '*', '9', 'inv 2 §8 keep (ADJECTIVE/UNRELATED)')
    # ---- item 11: SVD-factor U names beside the B rows (explicit)
    for path, fn, idents in (('t3toolbox/backend/t3_linalg.py', '_t3m_swap_pair', ('U',)),
                             ('t3toolbox/backend/t3_linalg.py', 't3m_inplace_fused', ('Uw',)),
                             ('t3toolbox/backend/t3_linalg.py', '_t3m_contract_pair', ('Uw',)),
                             ('t3toolbox/backend/t3_orthogonalization.py', None, ('Uox', 'U_i_a_x', 'U_a_x', 'U_a_x_b')),
                             ('t3toolbox/backend/ut3_orthogonalization.py', None, ('U_o_x',)),
                             ('tests/test_sharing.py', None, ('svd_U',))):
        for ident in idents:
            if names_by_file[path][ident]:
                add(ident, path + ('::' + fn if fn else ''), '11', 'SVD factor beside the B rows')
    for path, ln in (('tests/test_dispatch.py', 149), ('tests/backend/test_structural_guards.py', 46)):
        add('U', func_scope(path, ln), '11', 'SVD factor (U, ss, Vt = ...)')
    # ---- item 12: U/u meaning uniform
    for ident in sorted(all_names):
        if (ident.startswith(('UT3', 'ut3_', 'ufv_', 'utv_')) or (ident.startswith('_u_') and ident not in
                                                                     sra.A2_SWAP)) and ident not in renamed_olds:
            add(ident, '*', '12', 'U/u = uniform')
    for path, idents in (('tests/test_dispatch.py', ('UB', 'UV', 'UW', 'UF', 'UFb', 'Usc')),
                         ('tests/test_aligned_padded_frame_frontend.py', ('UB', 'UV', 'UW', 'UF', 'UFb', 'Usc')),
                         ('tests/test_uniform_frame_variations_format.py', ('UB', 'UV', 'UW', 'UF', 'UFb', 'Usc'))):
        for ident in idents:
            if names_by_file[path][ident]:
                add(ident, path, '12', 'U = uniform')
    add('U', func_scope('tests/oracle_sweeps/o1_uniform.py', 193), '12', 'a uniform T3')
    # ---- item 13: O with other meanings
    for path in ('tests/backend/test_t3_orthogonalization.py', 'tests/oracle_sweeps/sweep_models.py',
                 'tests/oracle_sweeps/sweep_exact.py', 'tests/oracle_sweeps/sweep_optimizers.py'):
        add('O', path, '13', 'the oracle module alias O')
    add('O', func_scope('tests/test_uniform_probing.py', 213), '13', 'an order count')
    # ---- item 2 (part): QR's R where Q is listed (R is also the right store: scoped per function; R1 review MINOR-1a)
    for path, ln in (('t3toolbox/backend/linalg.py', 562), ('t3toolbox/backend/fv_operations.py', 284),
                     ('tests/backend/test_padded_frame.py', 99)):
        add('R', func_scope(path, ln), '14', 'QR factor R (line %d)' % ln)
    # ---- item 1 override (D-1, §3.2): the shape locals `sh` and the shifted pairs `psh`, `pshd` stay; scoped so no
    #      row covers a renamed sigma-hat `sh` (R1 review MINOR-1b): per outermost def, or per line where a def also
    #      holds a renamed one
    renamed_pos = {(h['file'], int(h['line']), int(h['col'])): h['old'] for h in hits if h['ctx'].endswith('NAME')}
    for ident in ('sh', 'psh', 'pshd'):
        occ = collections.defaultdict(set)
        for path, line, col, tok, ctx in p1.record_names:
            if tok == ident and (path, line, col) not in renamed_pos:
                o = [s_ for s_ in outer[path] if s_[0] <= line <= s_[1]]
                occ[(path, o[0][2] if o else None, o[0][:2] if o else None)].add(line)
        for (path, fn, span), lines in sorted(occ.items(), key=lambda kv: (kv[0][0], min(kv[1]))):
            # line-scoped only where the same identifier is also renamed inside this def (sigma-hat sh)
            mixed = span is not None and any(f == path and span[0] <= l <= span[1] and o == ident
                                             for (f, l, c), o in renamed_pos.items())
            if fn is None or mixed:
                for l in sorted(lines):
                    add(ident, '%s:%d-%d' % (path, l, l), '14', 'a shape local / shifted pair, not sigma-hat (§3.2)')
            else:
                add(ident, '%s::%s' % (path, fn), '14', 'a shape local / shifted pair, not sigma-hat (§3.2)')
    # ---- item 14: the finite-difference `shift = lambda h: ...` locals (a move of the point, not the jet shift)
    for path, ln in (('t3toolbox/backend/apply.py', 421), ('t3toolbox/backend/entries.py', 333),
                     ('t3toolbox/backend/t3_linalg.py', 911), ('t3toolbox/backend/probing.py', 1805),
                     ('t3toolbox/backend/sampling_derivatives.py', 3080), ('t3toolbox/backend/sampling_derivatives.py', 3353),
                     ('t3toolbox/backend/sampling_derivatives.py', 3438), ('tests/backend/test_curvature.py', 583),
                     ('tests/backend/test_curvature.py', 706), ('tests/backend/test_curvature.py', 865),
                     ('tests/backend/test_curvature.py', 1168), ('tests/backend/test_curvature.py', 1496)):
        add('shift', func_scope(path, ln), '14', 'a finite-difference move of the point (line %d), not the jet shift' % ln)
    # ---- items 2, 3, 13 (scoped letters): the §3.4 inventory's keeps
    n_pqu = 0
    if os.path.exists(PQU_KEEPS):
        for r in csv.DictReader(open(PQU_KEEPS), delimiter='\t'):
            scope = r['file'] if r['function'] in ('*', '', '-') else '%s::%s' % (r['file'], r['function'])
            add(r['identifier'], scope, r['dnt_item'], 'pqu: ' + r['reason'])
            n_pqu += 1
    # ---- item 10: every public name not in the token map
    for path in sorted(p for p in srcs if p.startswith('t3toolbox/')):
        for ident in sorted(public_names(path, srcs[path])):
            if ident not in renamed_olds and ident not in sra.A2_SWAP and ident not in sra.A1_RENAMED:
                add(ident, '*', '10', 'public name (%s)' % path)

    # ---- dedupe; prove disjointness from the dry run
    seen, uniq = set(), []
    for r in rows:
        if (r[0], r[1]) not in seen:
            seen.add((r[0], r[1]))
            uniq.append(r)
    violations = []
    for ident, scope, item, note in uniq:
        for f, ln in renamed_olds.get(ident, ()):
            if _in(scope, f, ln, outer):
                violations.append('%s %s (item %s) renamed at %s:%d' % (ident, scope, item, f, ln))
    out = os.path.join(REPO, 'dev', 'store_rename_dnt.txt')
    with open(out, 'w') as fh:
        fh.write('# store rename DO-NOT-TOUCH list (plan §1.5, gate K14). Built by '
                 'dev/store_rename_slices/r1_build_lists.py at v2026.4.0.\n')
        fh.write('# Columns: identifier, scope (* | file | file::outermost def | file:first-last), §1.5 item, note.\n')
        fh.write('# Item 6 (contract/einsum string arguments) is checked by K14 directly; item 10 lists the public\n')
        fh.write('# names of t3toolbox/ not in the token map. Disjoint from the dry-run hit table: %s.\n'
                 % ('PROVEN (0 violations)' if not violations else '%d VIOLATIONS' % len(violations)))
        fh.write('identifier\tscope\titem\tnote\n')
        for r in sorted(uniq, key=lambda r: (int(r[2]) if r[2].isdigit() else 99, r[2], r[0], r[1])):
            fh.write('\t'.join(r) + '\n')
    print('dnt rows %d (pqu keeps %d); disjointness violations %d' % (len(uniq), n_pqu, len(violations)))
    for v in violations[:40]:
        print('  VIOLATION', v)
    build_keeps()


def _names_in(record, path, a, b):
    return {tok for p, line, col, tok, ctx in record if p == path and a <= line <= b and not ctx.startswith('DOCTEST')}


def _in(scope, f, ln, outer) -> bool:
    if scope == '*':
        return True
    if '::' in scope:
        p, fn = scope.split('::')
        return p == f and any(a <= ln <= b and n == fn for a, b, n in outer.get(f, ()))
    m = re.match(r'(.*):(\d+)-(\d+)$', scope)
    if m:
        return m.group(1) == f and int(m.group(2)) <= ln <= int(m.group(3))
    return scope == f


# ===================================================================================================================
# keeps.txt (K9)
# ===================================================================================================================

STORE_WORD_RE = re.compile(r'\b(?:up|down|Up|Down|UP|DOWN)\b|\b(?:up|down)_[a-z]\b')
S33_RE = re.compile(r'(var|adjoint|Adjoint)-(up|down)\b|\b(up|down) (jets|eta|sweep|sweeps|lift)\b|\bup-index\b')
S42_SITES = (('docs/contributor/padded_frame_internals.md', 27, 31), ('docs/contributor/padded_frame_internals.md', 42, 43),
             ('docs/padded_frame_cone.tex', 403, 404), ('t3toolbox/backend/uniform_padded_frames.py', 563, 563),
             ('t3toolbox/backend/uniform_padded_frames.py', 592, 592), ('tests/backend/test_padded_frame.py', 562, 562),
             ('tests/test_contractions_interpreter.py', 363, 363))
S43_SITES = (('t3toolbox/frame_variations_format.py', (1371, 1372)),
             ('t3toolbox/uniform_frame_variations_format.py', (1122, 1125)),
             ('t3toolbox/tucker_tensor_train.py', (615, 2956, 2978, 3002, 3108)),
             ('t3toolbox/backend/probing.py', (646, 647, 1221, 1222)),
             ('t3toolbox/backend/t3_orthogonalization.py', (147, 171)),
             ('t3toolbox/backend/ut3_orthogonalization.py', (189,)), ('t3toolbox/backend/geometry.py', (348,)),
             ('t3toolbox/backend/t3_svd.py', (312, 345)), ('t3toolbox/backend/fv_conversions.py', (181, 182, 184, 185,
                                                                                                219, 251)),
             ('tests/test_frame_variations_format.py', (447,)), ('tests/test_tucker_tensor_train.py', (3202,)),
             ('docs/user_guide.rst', (142,)))
INV3_S9_PATTERNS = (
    ('English "up to"/"up front"/"warm-up"/"blow up"/"picked up"/"set up"', r'\bup (?:to|front)\b|\bwarm-up\b|\bblow[- ]up\b|\bpicked up\b|\bset up\b'),
    ('English "down to"/"downstream"/"write down"/"down-weight"', r'\bdown to\b|\bdownstream\b|\bwrite down\b|\bdown-weight'),
    ('jet-order shifts', r'_shift_up|_shift_down|down-shift|\\\\Dsh'),
)


S33_EXPLICIT = (('t3toolbox/backend/apply.py', 215, 'down ``eta``'),)   # the plan's rows the grep cannot see


def write_s33(grep_rows: typ.Dict[typ.Tuple[str, int], str]) -> None:
    """dev/store_rename_slices/R1_s33_ward_sites.tsv: §3.3's authoritative list = the grep + the plan's explicit rows."""
    out = os.path.join(REPO, 'dev', 'store_rename_slices', 'R1_s33_ward_sites.tsv')
    rows = []
    for (f, l), _ in sorted(grep_rows.items()):
        text = open(os.path.join(REPO, f), encoding='utf-8').read().split('\n')[l - 1]
        rows.append((f, l, ';'.join(m.group(0) for m in S33_RE.finditer(text)), 'grep', text.strip()))
    for f, l, mt in S33_EXPLICIT:
        text = open(os.path.join(REPO, f), encoding='utf-8').read().split('\n')[l - 1]
        rows.append((f, l, mt, 'plan (explicit; the grep misses it)', text.strip()))
    with open(out, 'w') as fh:
        fh.write('file\tline\tmatch\tsource\ttext\n')
        for r in sorted(rows):
            fh.write('%s\t%d\t%s\t%s\t%s\n' % (r[0], r[1], r[2], r[3], r[4][:200]))
    print('s33 sites: %d (grep %d + explicit %d)' % (len(rows), len(grep_rows), len(S33_EXPLICIT)))


def build_keeps() -> None:
    rows = []
    # inv 2 §9: every keep row (file stem -> backend path)
    inv2 = open(INV2).read()
    sec = inv2[inv2.index('## 9. Per-file line inventory'):inv2.index('## 10. Cross-references')]
    cur = None
    for ln in sec.splitlines():
        m = re.match(r'### `([^`]+)`', ln)
        if m:
            cur = 't3toolbox/backend/' + m.group(1)
            continue
        m = re.match(r'\| (\d+) \| (\w[\w+]*) \| ([^|]*) \| keep \| ?(.*?) ?\|$', ln)
        if m and cur:
            rows.append([cur, int(m.group(1)), 'KEEP', 'inv2 §9 %s %s' % (m.group(2), m.group(3).strip())])
    # inv 1 App. B-D: per-file line lists
    inv1 = open(INV1).read()
    for app in ('B', 'C', 'D'):
        a = inv1.index('## Appendix %s.' % app)
        nxt = inv1.find('## Appendix', a + 10)
        part = inv1[a:nxt if nxt > 0 else len(inv1)]
        for m in re.finditer(r'^- `([^`]+)`: ([\d, ]+)$', part, re.M):
            for x in m.group(2).split(','):
                rows.append([m.group(1), int(x), 'KEEP', 'inv1 App. %s' % app])
    # inv 3: rows proposing keep (docs and notes)
    inv3 = open(INV3).read()
    for m in re.finditer(r'^\| `?([\w/.\-]+\.(?:md|rst|tex))`?:(\d+)(?:-(\d+))?[^|]*\|[^|]*\| ?keep', inv3, re.M):
        a = int(m.group(2))
        b = int(m.group(3)) if m.group(3) else a
        for x in range(a, b + 1):
            f = m.group(1)
            if not os.path.exists(os.path.join(REPO, f)) and os.path.exists(os.path.join(REPO, 'docs', f)):
                f = 'docs/' + f                    # inv 3 writes docs paths relative to docs/ (R1 review MINOR-3)
            rows.append([f, x, 'KEEP', 'inv3 keep row'])
    # EXEMPT: §3.3 (the authoritative grep), §4.2, §4.3
    exempt = {}
    for top in ('t3toolbox', 'tests', 'examples'):
        for d, _, fs in os.walk(os.path.join(REPO, top)):
            for f in fs:
                if f.endswith('.py'):
                    p = os.path.relpath(os.path.join(d, f), REPO)
                    for i, ln in enumerate(open(os.path.join(REPO, p), encoding='utf-8'), 1):
                        if S33_RE.search(ln):
                            exempt[(p, i)] = '§3.3 -ward rewording (grep)'
    write_s33(exempt)
    exempt[('t3toolbox/backend/apply.py', 215)] = '§3.3 -ward rewording (the plan\'s explicit row; the grep misses ``eta``)'
    for f, a, b in S42_SITES:
        for x in range(a, b + 1):
            exempt[(f, x)] = '§4.2 store-and-arrow site'
    for f, ls in S43_SITES:
        for x in ls:
            exempt[(f, x)] = '§4.3 backwards adjective'
    keyed = collections.OrderedDict()
    for f, l, kind, src in rows:
        keyed.setdefault((f, l), [f, l, kind, src])
    for (f, l), why in exempt.items():
        if (f, l) in keyed:
            keyed[(f, l)][2] = 'EXEMPT'
            keyed[(f, l)][3] += '; ' + why
        else:
            keyed[(f, l)] = [f, l, 'EXEMPT', why]
    missing = sorted({f for (f, l) in keyed if not os.path.exists(os.path.join(REPO, f))})
    assert not missing, 'keep rows on paths that do not exist: %s' % missing
    out = os.path.join(REPO, 'dev', 'store_rename_keeps.txt')
    n = collections.Counter()
    with open(out, 'w') as fh:
        fh.write('# store rename K9 keep list (plan §5.3 K9), built by dev/store_rename_slices/r1_build_lists.py at '
                 'v2026.4.0.\n')
        fh.write('# KEEP: (a) a bare store word here is a conscious keep, and (converse) survives at its aligned '
                 'position.\n# EXEMPT: a tier-E line whose words change by design (§3.3 -ward rewordings, the '
                 'authoritative grep;\n# §4.2 store-and-arrow sites; §4.3 backwards adjectives): neither (a) nor the '
                 'converse applies.\n# PATTERN: inv 3 §9 classes in the docs, matched by regex (no line rows exist).\n')
        fh.write('file\tline\tkind\tsource\ttext\n')
        for (f, l), (f_, l_, kind, src) in sorted(keyed.items()):
            p = os.path.join(REPO, f)
            text = ''
            if os.path.exists(p):
                lines = open(p, encoding='utf-8').read().split('\n')
                text = lines[l - 1].strip() if l <= len(lines) else '<beyond EOF>'
            n[kind] += 1
            fh.write('%s\t%d\t%s\t%s\t%s\n' % (f, l, kind, src, text.replace('\t', ' ')))
        for name, pat in INV3_S9_PATTERNS:
            n['PATTERN'] += 1
            fh.write('docs/*\t-\tPATTERN\tinv3 §9: %s\t%s\n' % (name, pat))
    print('keeps: %s' % dict(n))


if __name__ == '__main__':
    main(sys.argv[1])
