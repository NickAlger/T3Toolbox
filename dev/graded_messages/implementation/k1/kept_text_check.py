#!/usr/bin/env python3
"""K1's kept-text check: in every library module a part changes, each top-level statement and each method of a class
is compared, text whole (signature, shape comments and their columns, diagrams, docstring, body), between BASE and the
working tree. Prints, per module, the statements whose text changed (with whether the header, the def line through its
colon, changed), the statements added and removed; the unchanged ones are counted. Every changed one must be a change
the part's note or handover names; an untouched function's signature comments and columns are then shown unchanged.

    python3 kept_text_check.py BASE [--diff NAME]     # --diff: print the unified diff of one statement
"""
import ast
import difflib
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))


def statements(source):
    lines = source.split('\n')
    out = {}
    counts = {}

    def add(node, prefix=''):
        start = min([node.lineno] + [d.lineno for d in getattr(node, 'decorator_list', [])])
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            key = prefix + node.name
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            key = prefix + ','.join(sorted(n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name)))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            key = prefix + 'import:' + ast.unparse(node)
        elif isinstance(node, ast.Expr) and isinstance(getattr(node, 'value', None), ast.Constant) \
                and isinstance(node.value.value, str):
            key = prefix + '<docstring>'
        else:
            counts[prefix + type(node).__name__] = counts.get(prefix + type(node).__name__, 0) + 1
            key = prefix + '%s#%d' % (type(node).__name__, counts[prefix + type(node).__name__])
        text = '\n'.join(lines[start - 1:node.end_lineno])
        head = text
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            head = '\n'.join(lines[start - 1:node.body[0].lineno - 1])
        out[key] = (text, head)
        if isinstance(node, ast.ClassDef):
            for sub in node.body:
                add(sub, prefix + node.name + '.')
    for node in ast.parse(source).body:
        add(node)
    return out


def main():
    base = sys.argv[1]
    want = sys.argv[sys.argv.index('--diff') + 1] if '--diff' in sys.argv else None
    files = subprocess.run(['git', '-C', REPO, 'diff', '--name-only', base, '--', 't3toolbox'], capture_output=True,
                           text=True, check=True).stdout.split()
    for path in files:
        old = subprocess.run(['git', '-C', REPO, 'show', '%s:%s' % (base, path)], capture_output=True, text=True).stdout
        new = open(os.path.join(REPO, path)).read()
        a, b = statements(old), statements(new)
        changed = [k for k in a if k in b and a[k][0] != b[k][0]]
        print('== %s: %d unchanged, %d changed, %d added, %d removed'
              % (path, sum(1 for k in a if k in b and a[k][0] == b[k][0]), len(changed),
                 len(set(b) - set(a)), len(set(a) - set(b))))
        for k in changed:
            print('   changed %-50s header %s' % (k, 'CHANGED' if a[k][1] != b[k][1] else 'same'))
            if want == k:
                print('\n'.join(difflib.unified_diff(a[k][0].split('\n'), b[k][0].split('\n'), lineterm='')))
        for k in sorted(set(b) - set(a)):
            print('   added   %s' % k)
        for k in sorted(set(a) - set(b)):
            print('   removed %s' % k)


if __name__ == '__main__':
    main()
