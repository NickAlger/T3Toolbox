"""M4: the transposes' argument ``residual -> c`` (Q2), as a located rename with an edit log.

    python rename.py ROOT forward LOG.tsv         # rename under ROOT/t3toolbox and ROOT/tests in place; write the log
    python rename.py ROOT inverse LOG.tsv         # replay the log backwards (the round trip)

Not a test (no `test_` name): a one-off of slice M4, kept as its record (M4_HANDOVER.md). Every edit is located, by the
AST or inside a located docstring, never by a bare search of the tree:

  A1  the argument ``residual`` of the 42 functions (TARGETS: a ``def`` of that name with that argument), and every
      ``Name`` node ``residual`` in its body (no nested scope of these functions rebinds it: checked)
  A2  a call's keyword ``residual=`` (``ast.keyword``), in t3toolbox/ and tests/, when the callee is one of the 42: by
      its name (the attribute or the function); a ``getattr(obj, name)`` whose name string (a literal, a ``%`` or a
      ``+`` of literals) contains ``_transpose`` or ``_corewise_hessian``; or one of the reviewed table calls
      (CALL_SITES: ``_TRANSPOSE[kind]``, ``getattr(x, ref_method[kind])`` in test_fitting.py, ``op`` over three
      ``UT3Tangent`` transposes in test_uniform_probing.py). Any other ``residual=`` keyword is reported and left alone
  D1  in the docstring of each of the 42: ``residual=`` on a doctest line (``>>>`` / ``...``) -> ``c=``
  D2  in the docstring of each of the 42: the inline literal ````residual```` -> ````c````
  E   the string literals ``'residual'`` of the signature table in tests/test_sampling_functions.py (by line)

The log has one row per edit: file, line, column in the old text, column in the new text, old, new. The inverse
replays it backwards at the new columns, so a name that already existed (``t3_apply_ambient_transpose``'s local ``c``)
needs no rule of its own: HEAD -> forward -> inverse gives HEAD back byte for byte. The forward run prints every
pre-existing ``c`` in the scope of the 42, for review by eye.
"""
import ast
import csv
import pathlib
import re
import sys

TARGETS = {
    't3toolbox/backend/apply.py': {'t3_apply_ambient_transpose', 'tv_apply_transpose', 't3_apply_corewise_transpose',
                                   't3_apply_corewise_hessian'},
    't3toolbox/backend/entries.py': {'t3_entries_ambient_transpose', 'tv_entries_transpose',
                                     't3_entries_corewise_transpose', 't3_entries_corewise_hessian'},
    't3toolbox/backend/probing.py': {'tv_probe_transpose', 't3_probe_ambient_transpose', 't3_probe_corewise_transpose',
                                     't3_probe_corewise_hessian'},
    't3toolbox/backend/ut3_sampling.py': {'ut3_apply_corewise_transpose', 'ut3_entries_corewise_transpose',
                                          'ut3_probe_corewise_transpose', 'ut3_probe_corewise_hessian',
                                          'ut3_apply_corewise_hessian', 'ut3_entries_corewise_hessian'},
    't3toolbox/backend/utv_sampling.py': {'utv_probe_transpose', 'utv_apply_transpose', 'utv_entries_transpose'},
    't3toolbox/manifold.py': {'probe_transpose', 'apply_transpose', 'entries_transpose'},
    't3toolbox/uniform_manifold.py': {'probe_transpose', 'apply_transpose', 'entries_transpose'},
    't3toolbox/tucker_tensor_train.py': {'probe_ambient_transpose', 'probe_corewise_transpose', 'apply_ambient_transpose',
                                         'entries_ambient_transpose', 'apply_corewise_transpose',
                                         'entries_corewise_transpose', 'probe_corewise_hessian',
                                         'apply_corewise_hessian', 'entries_corewise_hessian'},
    't3toolbox/uniform_tucker_tensor_train.py': {'apply_corewise_transpose', 'entries_corewise_transpose',
                                                 'probe_corewise_transpose', 'probe_corewise_hessian',
                                                 'apply_corewise_hessian', 'entries_corewise_hessian'},
}
CALLEES = set().union(*TARGETS.values())
TABLE_CALLEES = {('tests/test_fitting.py', '_TRANSPOSE')}          # a dict of the 42, called as _TRANSPOSE[kind](...)
CALL_SITES = {('tests/test_fitting.py', 374),                        # getattr(x, ref_method[kind]): *_corewise_transpose
              ('tests/test_uniform_probing.py', 607),                # op in (UT3Tangent.{probe,apply,entries}_transpose)
              ('tests/test_uniform_probing.py', 608)}
E_LINES = {'tests/test_sampling_functions.py'}                     # the signature table's 'residual' literals
DOCTEST = re.compile(r'^\s*(>>>|\.\.\.)')


def _literal_text(node):
    """The literal fragments of a name expression: a string, ``'%s_x' % kind``, ``kind + '_x'``, an f-string."""
    return ' '.join(n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str))


def _callee(func):
    if isinstance(func, ast.Call) and isinstance(func.func, ast.Name) and func.func.id == 'getattr' and len(func.args) == 2:
        text = _literal_text(func.args[1])
        if '_transpose' in text or '_corewise_hessian' in text:
            return ('getattr', text)
        return None
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Subscript) and isinstance(func.value, ast.Name):
        return ('table', func.value.id)
    return None


def edits_of(rel, text):
    tree = ast.parse(text)
    lines = text.split('\n')
    out, report = [], []
    names = TARGETS.get(rel, set())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in names:
            arguments = node.args.args + node.args.kwonlyargs
            if not any(a.arg == 'residual' for a in arguments):
                continue
            for a in arguments:                                                     # A1, the argument
                if a.arg == 'residual':
                    out.append((a.lineno, a.col_offset, 'residual', 'c', 'A1'))
            for inner in ast.walk(node):
                if inner is not node and isinstance(inner, (ast.FunctionDef, ast.Lambda)):
                    bound = {x.arg for x in inner.args.args + inner.args.kwonlyargs}
                    if 'residual' in bound:
                        raise SystemExit('%s: a nested scope of %s rebinds residual' % (rel, node.name))
                if isinstance(inner, ast.Name) and inner.id == 'residual':          # A1, the uses
                    out.append((inner.lineno, inner.col_offset, 'residual', 'c', 'A1'))
                if isinstance(inner, ast.Name) and inner.id == 'c' or isinstance(inner, ast.arg) and inner.arg == 'c':
                    report.append('pre-existing c: %s:%d in %s' % (rel, inner.lineno, node.name))
            doc = node.body[0]
            if isinstance(doc, ast.Expr) and isinstance(doc.value, ast.Constant) and isinstance(doc.value.value, str):
                for ln in range(doc.lineno, doc.end_lineno + 1):
                    line = lines[ln - 1]
                    if DOCTEST.match(line):                                         # D1
                        for m in re.finditer(r'\bresidual=', line):
                            out.append((ln, m.start(), 'residual=', 'c=', 'D1'))
                    for m in re.finditer(r'``residual``', line):                    # D2
                        out.append((ln, m.start(), '``residual``', '``c``', 'D2'))
    for node in ast.walk(tree):                                                     # A2
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == 'residual':
                    who = _callee(node.func)
                    ok = (who in CALLEES or (isinstance(who, tuple) and who[0] == 'getattr')
                          or (isinstance(who, tuple) and (rel, who[1]) in TABLE_CALLEES) or (rel, kw.lineno) in CALL_SITES)
                    if ok:
                        out.append((kw.lineno, kw.col_offset, 'residual', 'c', 'A2'))
                    else:
                        report.append('left alone: residual= to %r at %s:%d' % (who, rel, kw.lineno))
    if rel in E_LINES:                                                              # E
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value == 'residual':
                out.append((node.lineno, node.col_offset, "'residual'", "'c'", 'E'))
    for ln, col, old, new, rule in out:
        got = lines[ln - 1][col:col + len(old)]
        if got != old:
            raise SystemExit('%s:%d:%d (%s) expected %r, found %r' % (rel, ln, col, rule, old, got))
    return sorted(set(out)), report


def forward(root, log_path):
    rows, reports = [], []
    files = sorted(p for d in ('t3toolbox', 'tests') for p in (root / d).rglob('*.py'))
    for path in files:
        rel = str(path.relative_to(root))
        text = path.read_text()
        edits, report = edits_of(rel, text)
        reports += report
        if not edits:
            continue
        lines = text.split('\n')
        by_line = {}
        for ln, col, old, new, rule in edits:
            by_line.setdefault(ln, []).append((col, old, new, rule))
        for ln, items in by_line.items():
            items.sort()
            line, shift, pieces, last = lines[ln - 1], 0, [], 0
            for col, old, new, rule in items:
                pieces.append(line[last:col] + new)
                rows.append((rel, ln, col, col + shift, old, new, rule))
                shift += len(new) - len(old)
                last = col + len(old)
            lines[ln - 1] = ''.join(pieces) + line[last:]
        path.write_text('\n'.join(lines))
    with open(log_path, 'w', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(('file', 'line', 'old_col', 'new_col', 'old', 'new', 'rule'))
        w.writerows(rows)
    counts = {}
    for row in rows:
        counts[row[-1]] = counts.get(row[-1], 0) + 1
    print('edits:', len(rows), dict(sorted(counts.items())), 'files:', len({r[0] for r in rows}))
    for line in reports:
        print(line)


def inverse(root, log_path):
    with open(log_path) as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    by_file = {}
    for r in rows:
        by_file.setdefault(r['file'], []).append(r)
    for rel, items in by_file.items():
        path = root / rel
        lines = path.read_text().split('\n')
        for r in sorted(items, key=lambda r: (int(r['line']), int(r['new_col'])), reverse=True):
            ln, col, old, new = int(r['line']), int(r['new_col']), r['old'], r['new']
            line = lines[ln - 1]
            if line[col:col + len(new)] != new:
                raise SystemExit('%s:%d:%d expected %r, found %r' % (rel, ln, col, new, line[col:col + len(new)]))
            lines[ln - 1] = line[:col] + old + line[col + len(new):]
        path.write_text('\n'.join(lines))
    print('reverted:', len(rows))


if __name__ == '__main__':
    root, direction, log_path = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    {'forward': forward, 'inverse': inverse}[direction](root, log_path)
