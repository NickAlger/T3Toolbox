#!/usr/bin/env python3
"""The kept-comments check of K0: every top-level statement of the seven modules that K0 keeps, its text whole
(signature, shape comments and their columns, ASCII diagrams, docstrings, body), compared between b5125f76 and the
working tree, line by line. Prints each statement whose text changed, with a unified diff; every one must be a
rewrite of `k0/prose_edits.md`. Also lists the top-level statements present now that were not before (none expected).

    python3 dev/graded_messages/implementation/k0/kept_text_check.py
"""
import ast
import difflib
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
MODULES = ('probing', 'apply', 'entries', 'ut3_sampling', 'utv_sampling', 'fitting', 'uniform_fitting', 'common')


def statements(source):
    """name -> text of each top-level statement (with its decorators), keyed by its defined names or its kind."""
    lines = source.split('\n')
    out = {}
    for node in ast.parse(source).body:
        start = min([node.lineno] + [d.lineno for d in getattr(node, 'decorator_list', [])])
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            key = node.name
        elif isinstance(node, ast.Assign):
            key = ','.join(sorted(n.id for t in node.targets for n in ast.walk(t) if isinstance(n, ast.Name)))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        elif isinstance(node, ast.Expr) and node is ast.parse(source).body[0]:
            key = '<module docstring>'
        else:
            key = '%s@%s' % (type(node).__name__, ast.unparse(node)[:40])
        out[key] = '\n'.join(lines[start - 1:node.end_lineno])
    return out


def main():
    changed = 0
    for m in MODULES:
        path = 't3toolbox/backend/%s.py' % m
        before = subprocess.run(['git', '-C', REPO, 'show', 'b5125f76:' + path], capture_output=True, text=True,
                                check=True).stdout
        after = open(os.path.join(REPO, path), encoding='utf-8').read()
        old, new = statements(before), statements(after)
        for key in new:
            if key not in old:
                print('NEW in %s: %s' % (path, key))
            elif old[key] != new[key]:
                changed += 1
                print('CHANGED %s: %s' % (path, key))
                for line in difflib.unified_diff(old[key].split('\n'), new[key].split('\n'), lineterm='', n=0):
                    print('    ' + line)
        print('%s: %d kept statements, %d unchanged' % (path, len(new), sum(1 for k in new if old.get(k) == new[k])))
    print('%d kept statements changed' % changed)


if __name__ == '__main__':
    main()
