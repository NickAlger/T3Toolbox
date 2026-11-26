"""Compare two trees unit by unit, by exact text (comments, columns and diagrams included).

    python unitdiff.py OLDROOT NEWROOT [--diff] [--comments] [relpath ...]

Without relpaths: every .py under t3toolbox/, tests/, examples/ of either root.

A *unit* is a module docstring, a top-level or class-level statement (a def or class header with its docstring, an
assignment, an import, any other statement), or a method. Its text runs from its first decorator to its last line,
so every comment and blank line inside a function is part of it. A class is split: its header (decorators, the
class line, its docstring) is one unit and each of its body statements another, so that an edited method does not
mark the whole class.

Comment blocks *between* units (maximal runs of comment-only lines outside every unit, at module or class level) are
a separate ledger: each block of OLD is reported kept (the same lines appear in NEW in the same order, contiguous)
or gone, with the unit that follows it in OLD; each block of NEW not in OLD is reported new.

Report per file: identical / changed / deleted / added units. --diff prints a unified diff of each changed unit.
"""
import ast
import difflib
import os
import sys


def py_files(root):
    out = []
    for top in ('t3toolbox', 'tests', 'examples'):
        for d, _, files in os.walk(os.path.join(root, top)):
            if '__pycache__' in d:
                continue
            out += [os.path.relpath(os.path.join(d, f), root) for f in files if f.endswith('.py')]
    return out


def first_line(node):
    lines = [node.lineno] + [dec.lineno for dec in getattr(node, 'decorator_list', [])]
    return min(lines)


def key_of(node, prefix, seen):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        k = node.name
    elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        k = node.targets[0].id
    elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        k = node.target.id
    elif isinstance(node, (ast.Import, ast.ImportFrom)):
        k = 'import:' + ast.unparse(node)
    elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
        k = '<docstring>'
    else:
        k = 'stmt:' + ast.unparse(node).splitlines()[0][:70]
    k = prefix + k
    n = seen.get(k, 0)
    seen[k] = n + 1
    return k if n == 0 else '%s#%d' % (k, n)


def units(src):
    """{key: (start, end, text)} with 1-based inclusive line numbers; and the set of lines units cover."""
    lines = src.splitlines()
    tree = ast.parse(src)
    out, covered, seen = {}, set(), {}

    def add(key, a, b):
        out[key] = (a, b, '\n'.join(lines[a - 1:b]))
        covered.update(range(a, b + 1))

    def walk(body, prefix):
        for node in body:
            k = key_of(node, prefix, seen)
            a = first_line(node)
            if isinstance(node, ast.ClassDef):
                body0 = node.body
                hdr_end = body0[0].lineno - 1
                if (body0 and isinstance(body0[0], ast.Expr) and isinstance(body0[0].value, ast.Constant)
                        and isinstance(body0[0].value.value, str)):
                    hdr_end = body0[0].end_lineno
                    rest = body0[1:]
                else:
                    rest = body0
                add(k + ' <header>', a, hdr_end)
                walk(rest, node.name + '.')
            else:
                add(k, a, node.end_lineno)
    walk(tree.body, '')
    return out, covered, lines


def comment_blocks(lines, covered, unit_spans):
    """Maximal runs of comment-only lines outside every unit: [(start, end, [lines], next_unit_key)]."""
    blocks, run = [], []
    for i, line in enumerate(lines, 1):
        if i not in covered and line.strip().startswith('#'):
            run.append(i)
        else:
            if run:
                blocks.append(run)
            run = []
    if run:
        blocks.append(run)
    out = []
    for run in blocks:
        a, b = run[0], run[-1]
        nxt = min(((s, k) for k, (s, e, _) in unit_spans.items() if s > b), default=(None, '<end of file>'))[1]
        out.append((a, b, [lines[i - 1] for i in run], nxt))
    return out


def contains(haystack, needle):
    n = len(needle)
    return any(haystack[i:i + n] == needle for i in range(len(haystack) - n + 1))


def compare(old_root, new_root, rel, show_diff, show_comments):
    po, pn = os.path.join(old_root, rel), os.path.join(new_root, rel)
    if not os.path.exists(pn):
        print('FILE DELETED  %s' % rel)
        return
    if not os.path.exists(po):
        print('FILE ADDED    %s' % rel)
        return
    so, sn = open(po).read(), open(pn).read()
    if so == sn:
        return
    uo, co, lo = units(so)
    un, cn, ln = units(sn)
    same = [k for k in uo if k in un and uo[k][2] == un[k][2]]
    changed = [k for k in uo if k in un and uo[k][2] != un[k][2]]
    deleted = [k for k in uo if k not in un]
    added = [k for k in un if k not in uo]
    print('\n=== %s: %d identical, %d changed, %d deleted, %d added' % (rel, len(same), len(changed), len(deleted),
                                                                       len(added)))
    for k in changed:
        print('  CHANGED  %-70s old %d-%d  new %d-%d' % (k, uo[k][0], uo[k][1], un[k][0], un[k][1]))
        if show_diff:
            for line in difflib.unified_diff(uo[k][2].splitlines(), un[k][2].splitlines(), 'old', 'new', n=1,
                                             lineterm=''):
                print('      ' + line)
    for k in deleted:
        print('  DELETED  %-70s old %d-%d' % (k, uo[k][0], uo[k][1]))
    for k in added:
        print('  ADDED    %-70s new %d-%d' % (k, un[k][0], un[k][1]))
    bo = comment_blocks(lo, co, uo)
    bn = comment_blocks(ln, cn, un)
    new_lines = ln
    for a, b, text, nxt in bo:
        kept = contains(new_lines, text)
        if show_comments or not kept:
            print('  COMMENT %-4s old %d-%d (%d lines), before %s' % ('kept' if kept else 'GONE', a, b, len(text), nxt))
            if not kept and show_comments:
                for t in text:
                    print('      | ' + t)
    for a, b, text, nxt in bn:
        if not contains(lo, text):
            print('  COMMENT NEW  new %d-%d (%d lines), before %s' % (a, b, len(text), nxt))
            for t in text:
                print('      + ' + t)


def main(argv):
    show_diff = '--diff' in argv
    show_comments = '--comments' in argv
    args = [a for a in argv if not a.startswith('--')]
    old_root, new_root, rels = args[0], args[1], args[2:]
    if not rels:
        rels = sorted(set(py_files(old_root)) | set(py_files(new_root)))
    for rel in rels:
        compare(old_root, new_root, rel, show_diff, show_comments)


if __name__ == '__main__':
    main(sys.argv[1:])
