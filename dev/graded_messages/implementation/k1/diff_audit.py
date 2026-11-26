#!/usr/bin/env python3
"""K1's audits by construction (K1_DESIGN_NOTE.md section 7.4), standard library only; never imports the library.

    python3 diff_audit.py BASE [--part 0] [--out FILE]      # BASE: the git revision the part starts from

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

Exit status 1 if (b) finds a pair that differs from every old pair in reach, or a scope whose batch counts differ.
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
    args = ap.parse_args()
    report, bad, totals = [], 0, collections.Counter()
    for path in changed_files(args.base):
        before = git_show(args.base, path)
        after = open(os.path.join(REPO, path)).read() if os.path.exists(os.path.join(REPO, path)) else None
        totals.update(audit_a(path, before, after, report))
        if args.part == '0':
            bad += audit_b(path, scopes_of(path, before), scopes_of(path, after), report)
    kinds = collections.Counter(line.split()[0] for line in report)
    head = ['K1 diff audit, part %s, base %s' % (args.part, args.base),
            'changed lines by class: %s' % dict(totals),
            'audit entries: %s' % dict(kinds),
            'RESULT: %s' % ('PASS' if not bad else 'FAIL (%d)' % bad), '']
    text = '\n'.join(head + sorted(report, key=lambda l: (l.split()[0] != 'DIFFER', l.split()[0], l)))
    if args.out:
        open(args.out, 'w').write(text + '\n')
    print('\n'.join(head))
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
