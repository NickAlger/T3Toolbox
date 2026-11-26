#!/usr/bin/env python3
"""A rough K0 library tree for measurements, built in a scratch directory: the deleted top-level statements of
the seven modules (k0_inventory.py's closure) cut out by line range, `sampling_derivatives.py` removed, the import
lines that import a deleted module or name cut, and every deleted name dropped from the `__all__` lists. It is NOT
K0's tree (no docstring, comment or test is edited, unused imports stay); it is for the static scans whose result
depends on what library code exists (the subscript counts of tests/test_contractions_interpreter.py, the import
graph), and for importing the library with and without jax before K0 is built.

    python3 dev/graded_messages/implementation/k0/simulate.py OUTDIR      # writes OUTDIR/t3toolbox
"""
import ast
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import k0_inventory as K   # noqa: E402


def main(outdir):
    modules, _ = K.load_library()
    deleted, _, _ = K.closure(modules)
    dst = os.path.join(outdir, 't3toolbox')
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(os.path.join(K.REPO, 't3toolbox'), dst, ignore=shutil.ignore_patterns('__pycache__'))
    os.remove(os.path.join(dst, 'backend', 'sampling_derivatives.py'))
    for m in K.SEVEN:
        if m == K.SD:
            continue
        info = modules[m]
        lines = K.read(info['path']).split('\n')
        drop = set()
        for u in info['units']:
            if u.deleted:
                drop |= set(range(u.start, u.end + 1))
            elif u.is_import:
                node = u.node
                names = [a.name for a in node.names]
                module = getattr(node, 'module', None) or ''
                if any(n.endswith('sampling_derivatives') for n in names) or module.endswith('sampling_derivatives'):
                    drop |= set(range(u.start, u.end + 1))
                elif isinstance(node, ast.ImportFrom) and module in deleted and \
                        any(a.name in deleted[module] for a in node.names):
                    drop |= set(range(u.start, u.end + 1))
            elif '__all__' in u.defines:
                for k in range(u.start, u.end + 1):
                    stripped = lines[k - 1].strip().rstrip(',').strip("'\"")
                    if stripped in deleted[m]:
                        drop.add(k)
        kept = [line for k, line in enumerate(lines, 1) if k not in drop]
        with open(os.path.join(outdir, info['path']), 'w') as f:
            f.write('\n'.join(kept))
        ast.parse('\n'.join(kept))           # still Python
    print('wrote', dst)


if __name__ == '__main__':
    main(sys.argv[1])
