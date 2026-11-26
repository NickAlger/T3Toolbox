"""Import every module of the t3toolbox found on PYTHONPATH, with jax or with jax unimportable.

    PYTHONPATH=ROOT python -B imports.py ROOT [nojax]

Asserts that the package imported is ROOT's (a stale install or the wrong PYTHONPATH would otherwise pass), lists
every module by walking the package's files (not pkgutil's import side effects), imports each, and prints the
failures. With nojax, `import jax` raises ImportError (sys.modules entries set to None) and the check that jax is
really absent is printed.
"""
import importlib
import os
import sys
import traceback

root = os.path.abspath(sys.argv[1])
nojax = len(sys.argv) > 2 and sys.argv[2] == 'nojax'
if nojax:
    for m in ('jax', 'jax.numpy', 'jaxlib', 'jax.tree_util'):
        sys.modules[m] = None
import t3toolbox  # noqa: E402

pkg = os.path.dirname(os.path.abspath(t3toolbox.__file__))
assert pkg == os.path.join(root, 't3toolbox'), (pkg, root)
mods = []
for d, _, files in os.walk(pkg):
    if '__pycache__' in d:
        continue
    for f in sorted(files):
        if f.endswith('.py'):
            rel = os.path.relpath(os.path.join(d, f), root)[:-3].replace(os.sep, '.')
            mods.append(rel[:-len('.__init__')] if rel.endswith('.__init__') else rel)
bad = []
for m in sorted(mods):
    try:
        importlib.import_module(m)
    except Exception:
        bad.append((m, traceback.format_exc().strip().splitlines()[-1][:150]))
jax_state = 'jax hidden' if nojax else 'jax %s' % ('present' if 'jax' in sys.modules and sys.modules['jax'] else
                                                  'not imported')
if nojax:
    try:
        import jax  # noqa: F401
        jax_state = 'JAX STILL IMPORTABLE'
    except ImportError:
        pass
print('%s: %d modules from %s, %d failed' % (jax_state, len(mods), pkg, len(bad)))
for m, e in bad:
    print('  FAILED %s: %s' % (m, e))
