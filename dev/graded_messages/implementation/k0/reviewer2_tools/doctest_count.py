"""Count the docstrings with examples in every module of the t3toolbox on PYTHONPATH (what --doctest-modules collects).

    PYTHONPATH=ROOT python -B doctest_count.py ROOT [--list]
"""
import doctest, importlib, os, sys
root = os.path.abspath(sys.argv[1]); pkg = os.path.join(root, 't3toolbox')
import t3toolbox
assert os.path.dirname(os.path.abspath(t3toolbox.__file__)) == pkg
names = []
for d, _, fs in os.walk(pkg):
    if '__pycache__' in d: continue
    for f in fs:
        if f.endswith('.py'):
            m = os.path.relpath(os.path.join(d, f), root)[:-3].replace(os.sep, '.')
            names.append(m[:-9] if m.endswith('.__init__') else m)
total, rows = 0, []
for m in sorted(names):
    mod = importlib.import_module(m)
    tests = [t for t in doctest.DocTestFinder(exclude_empty=True).find(mod, m) if t.examples]
    total += len(tests); rows += [(t.name, len(t.examples)) for t in tests]
print('docstrings with examples: %d (examples %d)' % (total, sum(n for _, n in rows)))
if '--list' in sys.argv:
    for name, n in rows: print('  %s %d' % (name, n))
