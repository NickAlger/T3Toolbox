"""The mechanical part of K0 applied to a copy of a library: the frozen names cut, nothing else.

    python sim.py SRCROOT OUTROOT [NAMES_JSON]

Copies SRCROOT/t3toolbox to OUTROOT/t3toolbox; removes backend/sampling_derivatives.py; in each other module of
NAMES_JSON removes every top-level statement that binds a frozen name (a def or class from its first decorator, an
assignment), every import line that imports sampling_derivatives or from-imports a frozen name of its module, and
every frozen name's line in __all__. Docstrings, comments and the other imports are left as they are.

At review time: `unitdiff.py OUTROOT BUILTROOT` then lists exactly what the build changed beyond the cut (the
prose, the comment blocks, the unused imports); each item must be one of the design note's.
"""
import ast
import json
import os
import shutil
import sys

DEFAULT_NAMES = '/home/nick/repos/T3Toolbox/dev/graded_messages/implementation/k0/k0_names.json'


def binds(node):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}
    if isinstance(node, ast.Assign):
        return {t.id for t in node.targets if isinstance(t, ast.Name)}
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return {node.target.id}
    return set()


def main(src_root, out_root, names_path):
    deleted = json.load(open(names_path))['deleted']
    dst = os.path.join(out_root, 't3toolbox')
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(os.path.join(src_root, 't3toolbox'), dst, ignore=shutil.ignore_patterns('__pycache__'))
    sd = 't3toolbox.backend.sampling_derivatives'
    os.remove(os.path.join(out_root, sd.replace('.', '/') + '.py'))
    for mod, names in deleted.items():
        if mod == sd:
            continue
        names = set(names)
        path = os.path.join(out_root, mod.replace('.', '/') + '.py')
        lines = open(path).read().split('\n')
        tree = ast.parse('\n'.join(lines))
        drop, found = set(), set()
        for node in tree.body:
            start = min([node.lineno] + [d.lineno for d in getattr(node, 'decorator_list', [])])
            b = binds(node)
            if b & names:
                assert b <= names, (mod, b)            # never a statement binding a kept name beside a deleted one
                drop |= set(range(start, node.end_lineno + 1))
                found |= b
            elif isinstance(node, ast.Import) and any(a.name == sd for a in node.names):
                drop |= set(range(start, node.end_lineno + 1))
            elif isinstance(node, ast.ImportFrom) and (node.module == sd or any(
                    a.name in deleted.get(node.module or '', ()) for a in node.names)):
                assert all(a.name in deleted.get(node.module, ()) for a in node.names) or node.module == sd, node
                drop |= set(range(start, node.end_lineno + 1))
            elif b == {'__all__'}:
                for k in range(node.lineno, node.end_lineno + 1):
                    item = lines[k - 1].strip().rstrip(',').strip('\'"')
                    if item in names:
                        drop.add(k)
        missing = names - found
        assert not missing, (mod, sorted(missing))      # every frozen name is a top-level binding of its module
        kept = [line for k, line in enumerate(lines, 1) if k not in drop]
        ast.parse('\n'.join(kept))
        open(path, 'w').write('\n'.join(kept))
        print('%-36s %4d names, %5d -> %5d lines' % (mod, len(names), len(lines), len(kept)))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else DEFAULT_NAMES)
