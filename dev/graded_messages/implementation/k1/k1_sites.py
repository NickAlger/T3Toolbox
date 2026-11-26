#!/usr/bin/env python3
"""K1's site scan: every place that reads or passes a weight, a batch field, the oracle's kind or order, or calls a
function whose signature K1 changes. Static (`ast` + `doctest`), standard library only; it never imports the library.

    python3 dev/graded_messages/implementation/k1/k1_sites.py                 # the summary (markdown) on stdout
    python3 dev/graded_messages/implementation/k1/k1_sites.py --sites         # also every site, file:line
    python3 dev/graded_messages/implementation/k1/k1_sites.py --json F        # the raw sites as JSON

WHERE. Five places, each counted apart: the library's code (`t3toolbox/**.py`), the library's doctests (the `>>>`
examples of every docstring in `t3toolbox/`), the tests (`tests/**.py`, code; their docstrings hold no doctests),
the examples (`examples/*.py`), and the pages' doctests (`docs/**.md`, `docs/**.rst`: the `>>>` examples, which CI
runs; prose and fenced non-doctest code are counted apart as "page text" by a plain identifier search).

WHAT. A *call* of a target name: `name(...)` or `<anything>.name(...)`, the qualifier recorded (so `jets.data_metric`
and `batch.data_metric` are told apart by the reader). A *keyword* `weight=` in any call, with its callee. An
*attribute read* `X.kind` / `X.order` where `X`'s source text contains `oracle` (the oracle's statics), and
`X.weight` anywhere (the batch's field, or a frontend object's). Old-form optimizer calls (a string as the second
positional argument, or `kind=`) are tagged, since the translation keeps them working.

Within a test file each site is attributed to the enclosing `def test_*` (or to the helper / module level), so that
the number of tests touched can be read off.
"""
import argparse
import ast
import collections
import doctest
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))

CALL_TARGETS = (
    # K1 removes or renames
    'blocks_to_weight', 'data_metric', 'jet_to_dual_jet', 'dual_jet_to_jet', 'derivative_weight_to_taylor_weight',
    'prepare_derivative_data', 'check_weight', 'check_data', 'pack_data', '_pack_data_space', 'pack_batch',
    # the batch and the oracle, and the methods that move
    'LeastSquaresBatch', 'LeastSquaresOracle', 'check_batch', 'n_measurements', 'take',
    'flat_draw', '_require_draw', 'make_newton_display', 'prepare_batch', '_is_packed', '_frontend_draw',
    # the engine (its weight keyword)
    'sampling_variables', 'computed_names',
    # the private helpers of backend/fitting.py that K1 rewrites
    '_y_metric_y', '_y_block_metric_y', '_y_times_metric_y', '_arrays_of', '_take_data_space', '_take_W',
)
OPTIMIZERS = ('newton_cg', 'mc_sgd', 'adam', 'gradient_descent')


_LINES = {}


def _src(node, text):
    '''The source of a node (``ast.get_source_segment`` splits the text at every call: quadratic on long files).'''
    lines = _LINES.get(id(text))
    if lines is None or lines[0] is not text:
        lines = _LINES[id(text)] = (text, text.splitlines(True))
    lines = lines[1]
    try:
        l0, l1, c0, c1 = node.lineno - 1, node.end_lineno - 1, node.col_offset, node.end_col_offset
    except AttributeError:
        return ''
    if l0 == l1:
        return lines[l0].encode()[c0:c1].decode(errors='replace')
    first = lines[l0].encode()[c0:].decode(errors='replace')
    last = lines[l1].encode()[:c1].decode(errors='replace')
    return first + ''.join(lines[l0 + 1:l1]) + last


class _Visitor(ast.NodeVisitor):
    def __init__(self, text, where, path, line0=0):
        self.text, self.where, self.path, self.line0 = text, where, path, line0
        self.sites = []
        self.scope = ['<module>']

    def _add(self, what, node, detail):
        self.sites.append(dict(where=self.where, path=self.path, line=self.line0 + node.lineno, what=what,
                               detail=detail, scope=self.scope[-1]))

    def visit_FunctionDef(self, node):
        self.scope.append(node.name if node.name.startswith('test') or len(self.scope) == 1 else self.scope[-1])
        self.generic_visit(node)
        self.scope.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        self.generic_visit(node)

    def visit_Call(self, node):
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else (func.id if isinstance(func, ast.Name) else None)
        qual = _src(func.value, self.text) if isinstance(func, ast.Attribute) else ''
        if name in CALL_TARGETS:
            kws = [k.arg for k in node.keywords]
            self._add('call', node, dict(name=name, qual=qual, n_pos=len(node.args), keywords=kws))
        if name in OPTIMIZERS:
            old = ((len(node.args) > 1 and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str))
                   or any(k.arg == 'kind' for k in node.keywords))
            if old:
                self._add('old_form', node, dict(name=name, qual=qual,
                                                 weighted=any(k.arg == 'weight' for k in node.keywords)))
        for k in node.keywords:
            if k.arg == 'weight':
                self._add('kw_weight', node, dict(callee=name, qual=qual))
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if isinstance(node.ctx, ast.Load):
            base = _src(node.value, self.text)
            if node.attr in ('kind', 'order') and 'oracle' in base.lower():
                self._add('oracle_attr', node, dict(attr=node.attr, base=base))
            if node.attr == 'weight':
                self._add('weight_attr', node, dict(base=base))
        self.generic_visit(node)


def _scan_code(text, where, path, line0=0):
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    v = _Visitor(text, where, path, line0)
    v.visit(tree)
    return v.sites


def _doctest_blocks(text):
    '''The >>> examples of a text, joined into one source per run of examples (line offsets kept by padding).'''
    parser = doctest.DocTestParser()
    try:
        examples = parser.get_examples(text)
    except ValueError:
        return []
    out = []
    for ex in examples:
        out.append((ex.lineno, ex.source))
    return out


def _scan_doctests(text, where, path, line0=0):
    sites = []
    for lineno, source in _doctest_blocks(text):
        for s in _scan_code(source, where, path, line0 + lineno):
            s['line'] -= 1
            sites.append(s)
    return sites


def _docstrings(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                yield body[0].lineno, body[0].value.value


def scan():
    sites = []
    rel = lambda p: os.path.relpath(p, REPO)
    for path in sorted(glob.glob(os.path.join(REPO, 't3toolbox', '**', '*.py'), recursive=True)):
        text = open(path).read()
        sites += _scan_code(text, 'library', rel(path))
        tree = ast.parse(text)
        for lineno, doc in _docstrings(tree):
            sites += _scan_doctests(doc, 'library doctest', rel(path), lineno)
    for path in sorted(glob.glob(os.path.join(REPO, 'tests', '**', '*.py'), recursive=True)):
        sites += _scan_code(open(path).read(), 'tests', rel(path))
    for path in sorted(glob.glob(os.path.join(REPO, 'examples', '*.py'))):
        sites += _scan_code(open(path).read(), 'examples', rel(path))
    pages = sorted(glob.glob(os.path.join(REPO, 'docs', '*.md')) + glob.glob(os.path.join(REPO, 'docs', '*.rst'))
                   + glob.glob(os.path.join(REPO, 'docs', 'contributor', '*.md')))
    for path in pages:
        sites += _scan_doctests(open(path).read(), 'page doctest', rel(path))
    return sites


PAGE_WORDS = ('blocks_to_weight', 'data_metric', 'jet_to_dual_jet', 'dual_jet_to_jet',
              'derivative_weight_to_taylor_weight', 'prepare_derivative_data', 'check_weight', 'pack_data',
              'check_batch', 'n_measurements', 'LeastSquaresBatch', 'LeastSquaresOracle', 'weight=')


def page_text():
    '''Prose and non-doctest code of the pages: a plain search, line by line, excluding >>> and ... lines.'''
    counts = collections.Counter()
    files = collections.defaultdict(collections.Counter)
    pages = sorted(glob.glob(os.path.join(REPO, 'docs', '*.md')) + glob.glob(os.path.join(REPO, 'docs', '*.rst'))
                   + glob.glob(os.path.join(REPO, 'docs', 'contributor', '*.md')))
    for path in pages:
        for line in open(path):
            s = line.strip()
            if s.startswith('>>>') or s.startswith('...'):
                continue
            for w in PAGE_WORDS:
                if (w.endswith('=') and w in line) or re.search(r'\b%s\b' % re.escape(w), line):
                    counts[w] += 1
                    files[w][os.path.relpath(path, REPO)] += 1
    return counts, files


def summary(sites):
    out = []
    WHERE = ('library', 'library doctest', 'tests', 'examples', 'page doctest')
    out.append('## Calls of the targets (count; files) by place\n')
    out.append('| name | ' + ' | '.join(WHERE) + ' |')
    out.append('|---|' + '---|' * len(WHERE))
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for s in sites:
        if s['what'] == 'call':
            by[s['detail']['name']][s['where']].append(s)
    for name in CALL_TARGETS:
        if name not in by:
            continue
        cells = []
        for w in WHERE:
            ss = by[name][w]
            cells.append('%d (%d)' % (len(ss), len({s['path'] for s in ss})) if ss else '')
        out.append('| `%s` | ' % name + ' | '.join(cells) + ' |')
    out.append('\n## `weight=` keyword, by callee and place\n')
    kw = collections.Counter((s['detail']['callee'], s['where']) for s in sites if s['what'] == 'kw_weight')
    out.append('| callee | ' + ' | '.join(WHERE) + ' |')
    out.append('|---|' + '---|' * len(WHERE))
    for callee in sorted({c for c, _ in kw}, key=str):
        out.append('| `%s` | ' % callee + ' | '.join(str(kw[callee, w]) if kw[callee, w] else '' for w in WHERE)
                   + ' |')
    total = collections.Counter(s['where'] for s in sites if s['what'] == 'kw_weight')
    files = collections.defaultdict(set)
    for s in sites:
        if s['what'] == 'kw_weight':
            files[s['where']].add(s['path'])
    out.append('| **all** | ' + ' | '.join('%d (%d files)' % (total[w], len(files[w])) if total[w] else ''
                                          for w in WHERE) + ' |')
    out.append('\n## Old-form optimizer calls (the translation keeps them working)\n')
    old = collections.Counter((s['where'], s['detail']['weighted']) for s in sites if s['what'] == 'old_form')
    for (w, weighted), n in sorted(old.items()):
        out.append('- %s, %s: %d' % (w, 'with weight=' if weighted else 'no weight', n))
    out.append('\n## Attribute reads\n')
    attr = collections.Counter((s['what'], s['detail'].get('attr', 'weight'), s['where']) for s in sites
                               if s['what'] in ('oracle_attr', 'weight_attr'))
    for (what, a, w), n in sorted(attr.items()):
        out.append('- `.%s` (%s), %s: %d' % (a, 'on an oracle' if what == 'oracle_attr' else 'any object', w, n))
    out.append('\n## Tests touched (a `def test_*` containing a site), per file\n')
    tests = collections.defaultdict(set)
    for s in sites:
        if s['where'] == 'tests' and s['scope'].startswith('test') and (
                s['what'] in ('kw_weight', 'oracle_attr', 'weight_attr', 'old_form')
                or (s['what'] == 'call' and s['detail']['name'] in (
                    'blocks_to_weight', 'data_metric', 'jet_to_dual_jet', 'dual_jet_to_jet',
                    'derivative_weight_to_taylor_weight', 'prepare_derivative_data', 'check_weight', 'pack_data',
                    'pack_batch', 'LeastSquaresBatch', 'LeastSquaresOracle', 'check_batch', 'n_measurements',
                    'take', 'flat_draw'))):
            tests[s['path']].add(s['scope'])
    for path in sorted(tests):
        out.append('- `%s`: %d' % (path, len(tests[path])))
    out.append('- **all**: %d test functions (names, not parametrized nodes) in %d files'
               % (sum(len(v) for v in tests.values()), len(tests)))
    counts, pfiles = page_text()
    out.append('\n## Page text (prose and fenced non-doctest lines; a plain search)\n')
    for w in PAGE_WORDS:
        if counts[w]:
            out.append('- `%s`: %d lines in %d pages' % (w, counts[w], len(pfiles[w])))
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sites', action='store_true')
    ap.add_argument('--json')
    args = ap.parse_args()
    sites = scan()
    print(summary(sites))
    if args.sites:
        print('\n## Every site\n')
        for s in sorted(sites, key=lambda s: (s['where'], s['path'], s['line'])):
            print('- %s `%s:%d` %s %s [%s]' % (s['where'], s['path'], s['line'], s['what'],
                                               json.dumps(s['detail']), s['scope']))
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(sites, f, indent=0)


if __name__ == '__main__':
    main()
