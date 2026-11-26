#!/usr/bin/env python3
"""K1's audits by construction (K1_DESIGN_NOTE.md section 7.4), standard library only; never imports the library.

    python3 diff_audit.py BASE [--part 0] [--out FILE]      # BASE: the git revision the part starts from
    python3 diff_audit.py BASE --part 1 [--reasons TSV] [--files PATH ...] --out FILE

It compares every changed file (tests, examples, the library's docstrings' doctests, the pages' doctests) at BASE with
the working tree, parsed with `ast`, and writes:

(a) the lines: every changed line classified (a construction, a moved method call, a keyword, ...), and every changed
    line holding an `assert`, a tolerance (`rtol`, `atol`, `places`, `delta`) or a numeric literal listed by name;
(b) part 0, the kind and the order: every old construction of `LeastSquaresOracle` or of a subclass of it (in the same
    file) bound to its `(kind, order)`; every new `LeastSquaresBatch(` bound to its `(kind, order)`; in each scope
    (the function, a docstring, a page) each new pair must equal an old oracle pair of the same scope, else of the
    enclosing class's `setUp` / body, else of the module; a new pair made of the scope's own parameters is resolved one
    level up, at each call site of the helper in the file; the counts of batch constructions per scope agree. Listed:
    what cannot be paired, removed oracles whose scope keeps no paired batch, scopes with two distinct old pairs, and
    every subclass construction, old and new argument lists side by side.
(c) parts 1 to 5, the weights: every `blocks_to_weight(k, o, b, y)` removed from a scope needs a `block_weight=` of
    value `b` there (one plain assignment deep, either side); every old `weight=` keyword is paired with the new
    keyword of the same callee in the scope, by order, and classified (W-BLOCK, W-FULL, W-GONE); every new
    `full_weight=`, and every new `block_weight=` whose value is not a `blocks_to_weight` table's of the scope, needs a
    line in `full_weight_reasons.tsv` (file, scope, callee, keyword, expression, kind, reason; tab-separated). A reason
    line that matches no site is listed STALE. `audit_c_planted.py` plants a right and three wrong conversions.

Exit status 1 if (b) finds a pair that differs from every old pair in reach, or a scope whose batch counts differ; or if (c)
finds an unpaired removal or a new weight with no reason line.
Expressions are compared as `ast.unparse` text.
"""
import argparse
import ast
import collections
import difflib
import doctest
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
ORACLE, BATCH = 'LeastSquaresOracle', 'LeastSquaresBatch'
WEIGHT_KEYWORDS = ('weight', 'block_weight', 'full_weight')
REASONS = os.path.join(HERE, 'full_weight_reasons.tsv')


def git_show(rev, path):
    try:
        return subprocess.run(['git', '-C', REPO, 'show', '%s:%s' % (rev, path)], check=True, capture_output=True,
                              text=True).stdout
    except subprocess.CalledProcessError:
        return None


def changed_files(base):
    out = subprocess.run(['git', '-C', REPO, 'diff', '--name-only', base, '--', 't3toolbox', 'tests', 'examples',
                          'docs'], check=True, capture_output=True, text=True).stdout.split()
    return [p for p in out if p.endswith(('.py', '.md', '.rst'))]


# ---------------------------------------------------------------------------------------------------------------------
# Scopes: function bodies (qualified names), docstrings' doctests, pages' doctests
# ---------------------------------------------------------------------------------------------------------------------
class Scope:
    def __init__(self, name, params=(), cls=None):
        self.name, self.params, self.cls = name, tuple(params), cls
        self.oracles = []            # (class name, args text list, kw dict text, (kind, order) or None, line)
        self.batches = []            # ((kind, order) text, line, args)
        self.calls = []              # (callee name, positional texts, keyword texts, line)
        self.oracle_reads = []       # (text, line)
        self.weights = []            # (callee text, keyword, value text, first positional text or None, line)
        self.btw = []                # (third argument text, the name it is assigned to or None, line)
        self.assigns = {}            # name -> value text (the last plain assignment `name = value` in the scope)


def _callee(node):
    f = node.func
    return f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else None)


def _subclasses(tree):
    names = {ORACLE}
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name not in names:
                for b in node.bases:
                    base = b.attr if isinstance(b, ast.Attribute) else (b.id if isinstance(b, ast.Name) else None)
                    if base in names:
                        names.add(node.name)
                        changed = True
    return names


def _pair(node, first='kind', second='order'):
    args = list(node.args)
    kw = {k.arg: k.value for k in node.keywords if k.arg}
    a = args[0] if len(args) > 0 else kw.get(first)
    b = args[1] if len(args) > 1 else kw.get(second)
    if a is None or b is None:
        return None
    return ast.unparse(a), ast.unparse(b)


class _Walker(ast.NodeVisitor):
    def __init__(self, scopes, prefix, oracle_classes, line0=0):
        self.scopes, self.prefix, self.oracle_classes, self.line0 = scopes, prefix, oracle_classes, line0
        self.stack = [self._scope(prefix + '<module>')]
        self.cls = [None]
        self._target = {}

    def _scope(self, name, params=(), cls=None):
        if name not in self.scopes:
            self.scopes[name] = Scope(name, params, cls)
        return self.scopes[name]

    def visit_ClassDef(self, node):
        self.cls.append(node.name)
        self.stack.append(self._scope('%s%s.<body>' % (self.prefix, node.name), cls=node.name))
        self.generic_visit(node)
        self.stack.pop()
        self.cls.pop()

    def visit_FunctionDef(self, node):
        q = '%s%s%s' % (self.prefix, (self.cls[-1] + '.') if self.cls[-1] else '', node.name)
        params = [a.arg for a in node.args.args + node.args.kwonlyargs]
        self.stack.append(self._scope(q, params, self.cls[-1]))
        self.cls.append(self.cls[-1])
        self.generic_visit(node)
        self.cls.pop()
        self.stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node):
        s = self.stack[-1]
        name = _callee(node)
        line = self.line0 + node.lineno
        if name in self.oracle_classes:
            s.oracles.append((name, [ast.unparse(a) for a in node.args],
                              {k.arg: ast.unparse(k.value) for k in node.keywords}, _pair(node), line))
        if name == BATCH:
            s.batches.append((_pair(node), line, [ast.unparse(a) for a in node.args]))
        if name:
            s.calls.append((name, [ast.unparse(a) for a in node.args],
                            {k.arg: ast.unparse(k.value) for k in node.keywords if k.arg}, line))
        callee = name or ast.unparse(node.func)                       # getattr(sdef, name)(...) is a callee too
        first = ast.unparse(node.args[0]) if node.args else None
        for k in node.keywords:
            if k.arg in WEIGHT_KEYWORDS and callee != 'subTest':           # a subTest label is not a weight
                s.weights.append((callee, k.arg, ast.unparse(k.value), first, line))
        if name == 'blocks_to_weight':
            b = node.args[2] if len(node.args) > 2 else next((k.value for k in node.keywords if k.arg == 'blocks'),
                                                             None)
            s.btw.append((ast.unparse(b) if b is not None else '?', self._target.get(id(node)), line))
        self.generic_visit(node)

    def visit_Assign(self, node):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            self.stack[-1].assigns[node.targets[0].id] = ast.unparse(node.value)
            if isinstance(node.value, ast.Call):
                self._target[id(node.value)] = node.targets[0].id
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if node.attr in ('kind', 'order') and isinstance(node.ctx, ast.Load):
            base = ast.unparse(node.value)
            if 'oracle' in base.lower():
                self.stack[-1].oracle_reads.append(('%s.%s' % (base, node.attr), self.line0 + node.lineno))
        self.generic_visit(node)


def _docstring_owners(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)):
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) \
                    and isinstance(node.body[0].value.value, str):
                yield getattr(node, 'name', '<module>'), node.body[0].lineno, node.body[0].value.value


def _doctest_source(text):
    try:
        return '\n'.join(ex.source for ex in doctest.DocTestParser().get_examples(text))
    except ValueError:
        return ''


def scopes_of(path, text):
    scopes = {}
    if text is None:
        return scopes
    if path.endswith('.py'):
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return scopes
        classes = _subclasses(tree)
        if path.startswith(('tests/', 'examples/')):
            _Walker(scopes, '', classes).visit(tree)
        for owner, line, doc in _docstring_owners(tree):
            src = _doctest_source(doc)
            if src:
                try:
                    _Walker(scopes, 'doc:%s:' % owner, classes, line).visit(ast.parse(src))
                except SyntaxError:
                    pass
    else:
        src = _doctest_source(text)
        if src:
            try:
                _Walker(scopes, 'page:', {ORACLE}).visit(ast.parse(src))
            except SyntaxError:
                pass
    return scopes


# ---------------------------------------------------------------------------------------------------------------------
# (b) the kind and the order
# ---------------------------------------------------------------------------------------------------------------------
def _fallbacks(scopes, scope):
    '''The scopes whose old oracle pairs are in reach of `scope`: itself, the class's setUp and body, the module.'''
    out = [scope.name]
    prefix = scope.name.split(':')[0] + ':' if scope.name.startswith(('doc:', 'page:')) else ''
    if scope.cls:
        out += ['%s%s.setUp' % (prefix, scope.cls), '%s%s.<body>' % (prefix, scope.cls)]
    out.append(prefix + '<module>' if not prefix.startswith('doc:') else scope.name)
    return [n for n in out if n in scopes]


def _old_pairs(old, names):
    pairs = []
    for n in names:
        pairs += [o[3] for o in old[n].oracles if o[3] is not None]
    return pairs


def audit_b(path, old, new, report):
    bad = 0
    for name in sorted(set(old) | set(new)):
        o, n = old.get(name), new.get(name)
        ob = len(o.batches) if o else 0
        nb = len(n.batches) if n else 0
        if ob != nb and o is not None and n is not None:
            report.append('COUNT  %s %s: %d batch constructions before, %d after' % (path, name, ob, nb))
            bad += 1
        elif ob != nb:
            report.append('SCOPE  %s %s: a scope %s, with %d batch constructions (read by hand: a new test, or a '
                          'docstring that moved)' % (path, name, 'only after' if o is None else 'only before',
                                                     nb if o is None else ob))
    for name, n in sorted(new.items()):
        o = old.get(name)
        reach = _fallbacks(old, o) if o else []
        cands = sorted(set(_old_pairs(old, reach)))
        for pair, line, _ in n.batches:
            if pair is None:
                report.append('UNREAD %s:%d %s: a batch whose kind and order cannot be read' % (path, line, name))
                continue
            if pair in cands:
                tag = 'AMBIG ' if len(cands) > 1 else 'PAIRED'
                report.append('%s %s:%d %s: %s%s' % (tag, path, line, name, pair,
                                                      ('  (old pairs in reach: %s)' % cands) if len(cands) > 1 else ''))
                continue
            if all(p in n.params for p in pair):          # a helper: resolve at its call sites in the file
                short = name.split('.')[-1]
                sites = [(s, c) for s in new.values() for c in s.calls if c[0] == short]
                if not sites:
                    report.append('HELPER %s:%d %s: %s from its parameters; no call site in the file' %
                                  (path, line, name, pair))
                    continue
                for s, (callee, pos, kw, cline) in sites:
                    params = [p for p in n.params if p != 'self']
                    bind = dict(zip(params, pos))
                    bind.update(kw)
                    sub = tuple(bind.get(p, '?') for p in pair)
                    so = old.get(s.name)
                    cands2 = sorted(set(_old_pairs(old, _fallbacks(old, so)))) if so else []
                    if sub in cands2:
                        report.append('PAIRED %s:%d %s -> call %s:%d %s: %s' % (path, line, name, path, cline,
                                                                             s.name, sub))
                    elif not cands2:
                        report.append('HELPER %s:%d %s -> call %s:%d %s: %s (no old oracle in reach of the call: '
                                      'read by hand)' % (path, line, name, path, cline, s.name, sub))
                    else:
                        report.append('DIFFER %s:%d %s -> call %s:%d %s: %s not in %s' % (path, line, name, path,
                                                                                         cline, s.name, sub, cands2))
                        bad += 1
                continue
            if not cands:
                report.append('NOORCL %s:%d %s: %s, no old oracle in reach (read by hand)' % (path, line, name, pair))
            else:
                report.append('DIFFER %s:%d %s: %s not in the old pairs %s' % (path, line, name, pair, cands))
                bad += 1
    for name, o in sorted(old.items()):
        for cls, args, kw, pair, line in o.oracles:
            n = new.get(name)
            kept = n is not None and any(b[0] == pair for b in n.batches)
            if not kept:
                report.append('ORPHAN %s:%d %s: %s(%s) removed; no batch of its pair in this scope'
                              % (path, line, name, cls, ', '.join(args + ['%s=%s' % kv for kv in kw.items()])))
            if cls != ORACLE:
                newc = [c for c in (n.calls if n else []) if c[0] == cls]
                report.append('SUBCLS %s:%d %s: old %s(%s) | new %s' % (
                    path, line, name, cls, ', '.join(args + ['%s=%s' % kv for kv in kw.items()]),
                    '; '.join('%s(%s)' % (cls, ', '.join(c[1] + ['%s=%s' % kv for kv in c[2].items()]))
                              for c in newc) or '(none)'))
        if len(set(p for *_, p, _l in [(x[0], x[1], x[2], x[3], x[4]) for x in o.oracles] if p)) > 1:
            report.append('TWOPRS %s %s: two distinct old pairs %s' % (path, name, sorted(set(x[3] for x in o.oracles
                                                                                               if x[3]))))
        for text, line in o.oracle_reads:
            n = new.get(name)
            report.append('OREAD  %s:%d %s: old %s | new reads in scope: %s' % (
                path, line, name, text, [t for t, _ in (n.oracle_reads if n else [])]))
    return bad


# ---------------------------------------------------------------------------------------------------------------------
# (c) the weights (parts 1 to 5)
# ---------------------------------------------------------------------------------------------------------------------
def read_reasons(path=REASONS):
    """The reason lines: file, scope, callee, keyword, expression, kind, reason (tab-separated; '#' comments)."""
    out = {}
    if not os.path.exists(path):
        return out
    for raw in open(path):
        if not raw.strip() or raw.startswith('#'):
            continue
        cols = raw.rstrip('\n').split('\t')
        if len(cols) < 7:
            raise SystemExit('full_weight_reasons.tsv: a line with %d columns, 7 wanted: %r' % (len(cols), raw))
        out[tuple(cols[:5])] = (cols[5], cols[6])
    return out


def _resolve(scope, text):
    """One plain assignment deep: a name assigned in the scope stands for its value."""
    return scope.assigns.get(text, text) if scope is not None else text


def _minus(a, b):
    """The multiset a - b, as a list in a's order."""
    left = collections.Counter(b)
    out = []
    for x in a:
        if left[x]:
            left[x] -= 1
        else:
            out.append(x)
    return out


def audit_c(path, old, new, reasons, used, report):
    bad = 0
    for name in sorted(set(old) | set(new)):
        o, n = old.get(name), new.get(name)
        o_btw = [b for b, _t, _l in (o.btw if o else [])]
        n_btw = [b for b, _t, _l in (n.btw if n else [])]
        o_w = [(c, k, v, f) for c, k, v, f, _l in (o.weights if o else [])]
        n_w = [(c, k, v, f, l) for c, k, v, f, l in (n.weights if n else [])]
        new_block = [(c, v, f, l) for c, k, v, f, l in n_w if k == 'block_weight']
        n_block_vals = {_resolve(n, v) for c, v, f, l in new_block} | {v for c, v, f, l in new_block}
        # every blocks_to_weight(k, o, b, y) removed: a block_weight= of value b in the scope
        for b in _minus(o_btw, n_btw):
            if b in n_block_vals or _resolve(o, b) in n_block_vals:
                report.append('BTW-OK %s %s: blocks_to_weight(.., %s, ..) removed; block_weight=%s in the scope'
                              % (path, name, b, b))
            else:
                report.append('BTW-UNPAIRED %s %s: blocks_to_weight(.., %s, ..) removed, and no block_weight= of that '
                              'value in the scope' % (path, name, b))
                bad += 1
        # every old weight= keyword: the new call of the same callee, by order
        by_callee_old, by_callee_new = collections.defaultdict(list), collections.defaultdict(list)
        for c, k, v, f in o_w:
            if k == 'weight':
                by_callee_old[c].append(v)
        for c, k, v, f, l in n_w:
            by_callee_new[c].append((k, v, l))
        for c, values in sorted(by_callee_old.items()):
            news = list(by_callee_new.get(c, []))
            kept = [v for k, v, l in news if k == 'weight']
            for v in _minus(values, kept):
                moved = [x for x in news if x[0] != 'weight']
                if moved:
                    k2, v2, l2 = moved.pop(0)
                    news.remove((k2, v2, l2))
                    report.append('W-%s %s:%d %s: %s(weight=%s) -> %s=%s' % ('BLOCK' if k2 == 'block_weight' else
                                                                              'FULL ', path, l2, name, c, v, k2, v2))
                else:
                    report.append('W-GONE  %s %s: %s(weight=%s): no weight keyword on that callee now (read: the '
                                  'subject moved?)' % (path, name, c, v))
        # every new full_weight= and every block_weight= not made from a blocks_to_weight table: a reason line
        old_keys = collections.Counter((c, k, v) for c, k, v, f in o_w)
        btw_vals = set(o_btw) | set(n_btw) | {_resolve(o, b) for b in o_btw} | {_resolve(n, b) for b in n_btw}
        for c, k, v, f, l in n_w:
            if k == 'weight':
                continue
            if old_keys[(c, k, v)]:
                old_keys[(c, k, v)] -= 1
                continue                                            # not new in this part
            key = (path, name, c, k, v)
            if k == 'block_weight' and (v in btw_vals or _resolve(n, v) in btw_vals):
                report.append('B-PAIRED %s:%d %s: %s(block_weight=%s), a blocks_to_weight table of the scope'
                              % (path, l, name, c, v))
                continue
            if key in reasons:
                used.add(key)
                kind, why = reasons[key]
                report.append('%s-REASON %s:%d %s: %s(%s=%s) [%s] %s' % ('F' if k == 'full_weight' else 'B', path,
                                                                        l, name, c, k, v, kind, why))
            else:
                report.append('%s-NOREASON %s:%d %s: %s(%s=%s), first argument %s' % (
                    'F' if k == 'full_weight' else 'B', path, l, name, c, k, v, f))
                bad += 1
    return bad


# ---------------------------------------------------------------------------------------------------------------------
# (a) the lines
# ---------------------------------------------------------------------------------------------------------------------
CLASSES = [
    ('construction', re.compile(r'LeastSquares(Oracle|Batch)\(|Oracle\(\)|_Oracle\(|Oracle\(')),
    ('method', re.compile(r'\.(take|check|check_batch|n_measurements|data_metric)\b|flat_draw|pack_batch|'
                          r'prepare_batch|_frontend_draw|_require_draw|make_newton_display')),
    ('kind_order', re.compile(r'\.(kind|order)\b')),
]
FLAG = re.compile(r'assert|rtol|atol|places=|delta=|allclose|isclose')


def audit_a(path, before, after, report):
    a, b = (before or '').splitlines(), (after or '').splitlines()
    counts = collections.Counter()
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == 'equal':
            continue
        for side, lines, start in (('-', a[i1:i2], i1), ('+', b[j1:j2], j1)):
            for k, line in enumerate(lines):
                cls = next((c for c, rx in CLASSES if rx.search(line)), 'other')
                counts[cls] += 1
                if FLAG.search(line) or cls == 'other':
                    report.append('LINE   %s %s%d [%s] %s' % (path, side, start + k + 1, cls, line.strip()[:150]))
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('base')
    ap.add_argument('--part', default='0')
    ap.add_argument('--out')
    ap.add_argument('--reasons', default=REASONS)
    ap.add_argument('--scope-map', nargs='*', default=[], help='OldClass=NewClass: a class renamed by the part')
    ap.add_argument('--files', nargs='*', help='restrict to these paths (default: every changed file)')
    args = ap.parse_args()
    report, bad, totals = [], 0, collections.Counter()
    reasons, used = read_reasons(args.reasons), set()
    for path in (args.files or changed_files(args.base)):
        before = git_show(args.base, path)
        after = open(os.path.join(REPO, path)).read() if os.path.exists(os.path.join(REPO, path)) else None
        totals.update(audit_a(path, before, after, report))
        if args.part == '0':
            bad += audit_b(path, scopes_of(path, before), scopes_of(path, after), report)
        else:
            old_scopes = scopes_of(path, before)
            for rename in args.scope_map:                   # a renamed class: compare its scopes under the new name
                old, new = rename.split('=')
                for name in [n for n in old_scopes if n.split('.')[0] == old]:
                    old_scopes[new + name[len(old):]] = old_scopes.pop(name)
                    report.append('SCOPE-MAP %s %s -> %s' % (path, name, new + name[len(old):]))
            bad += audit_c(path, old_scopes, scopes_of(path, after), reasons, used, report)
    if args.part != '0':
        for key in sorted(set(reasons) - used):
            report.append('STALE-REASON %s: a reason line that matches no new site' % ('\t'.join(key),))
    kinds = collections.Counter(line.split()[0] for line in report)
    head = ['K1 diff audit, part %s, base %s' % (args.part, args.base),
            'changed lines by class: %s' % dict(totals),
            'audit entries: %s' % dict(kinds),
            'RESULT: %s' % ('PASS' if not bad else 'FAIL (%d)' % bad), '']
    first = ('DIFFER', 'BTW-UNPAIRED', 'F-NOREASON', 'B-NOREASON')
    text = '\n'.join(head + sorted(report, key=lambda l: (l.split()[0] not in first, l.split()[0], l)))
    if args.out:
        open(args.out, 'w').write(text + '\n')
    print('\n'.join(head))
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
