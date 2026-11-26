"""K0's editing helper: remove top-level or class-level statements of a Python file by name, with their decorators
and the blank lines before them (comments directly above a removed statement are kept unless `with_comments`).

    python3 edit_tools.py FILE NAME [NAME ...]     NAME: 'func', 'Class', 'Class.method', or 'Class.attr'

Prints what it removed (lines). The file must still parse afterwards.
"""
import ast
import sys


def spans(tree, names):
    out = []
    for node in tree.body:
        targets = []
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            targets = [node.name]
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            ts = node.targets if isinstance(node, ast.Assign) else [node.target]
            targets = [n.id for t in ts for n in ast.walk(t) if isinstance(n, ast.Name)]
        if any(t in names for t in targets):
            out.append(node)
        if isinstance(node, ast.ClassDef):
            for sub in node.body:
                sub_t = []
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    sub_t = [sub.name]
                elif isinstance(sub, (ast.Assign, ast.AnnAssign)):
                    ts = sub.targets if isinstance(sub, ast.Assign) else [sub.target]
                    sub_t = [n.id for t in ts for n in ast.walk(t) if isinstance(n, ast.Name)]
                if any('%s.%s' % (node.name, t) in names for t in sub_t):
                    out.append(sub)
    return out


def remove(path, names, with_comments=False):
    src = open(path).read().split('\n')
    tree = ast.parse('\n'.join(src))
    nodes = spans(tree, set(names))
    found = set()
    drop = set()
    for node in nodes:
        start = min([node.lineno] + [d.lineno for d in getattr(node, 'decorator_list', [])])
        end = node.end_lineno
        k = start - 1
        while with_comments and k >= 1 and src[k - 1].strip().startswith('#'):
            k -= 1
            start = k + 1
        k = start - 1
        while k >= 1 and src[k - 1].strip() == '':          # the blank lines before it
            k -= 1
        drop |= set(range(k + 1, end + 1))
        print('%s: removed %d-%d (%s)' % (path, k + 1, end, getattr(node, 'name', 'assignment')))
    kept = [line for i, line in enumerate(src, 1) if i not in drop]
    text = '\n'.join(kept)
    ast.parse(text)
    open(path, 'w').write(text)


if __name__ == '__main__':
    args = sys.argv[1:]
    wc = '--with-comments' in args
    args = [a for a in args if a != '--with-comments']
    remove(args[0], args[1:], with_comments=wc)
