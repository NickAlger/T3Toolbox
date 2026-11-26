"""Every :py: role of a page resolved against a tree: import the module, getattr the rest.

    PYTHONPATH=ROOT python -B roles.py PAGE [PAGE ...]

Sphinx -W does not catch a dangling Python cross-reference here (docs/conf.py is not nitpicky), so this is the
check. Prints each target that does not resolve.
"""
import importlib
import re
import sys

ROLE = re.compile(r':py:(func|class|mod|data|meth|attr|obj|exc|const):`~?([\w.]+)`')


def resolves(target):
    parts = target.split('.')
    for k in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module('.'.join(parts[:k]))
        except Exception:
            continue
        try:
            for p in parts[k:]:
                obj = getattr(obj, p)
            return True
        except AttributeError:
            return False
    return False


for page in sys.argv[1:]:
    text = open(page).read()
    found = ROLE.findall(text)
    bad = sorted({t for _, t in found if t.startswith('t3toolbox') and not resolves(t)})
    print('%s: %d roles, %d unresolved' % (page, len(found), len(bad)))
    for t in bad:
        print('  ' + t)
