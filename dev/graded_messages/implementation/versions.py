"""Print the versions the gate runs on, one line (used by run_gate.sh; written to LOGDIR/versions.txt)."""
import importlib.metadata
import sys

parts = ['python ' + sys.version.split()[0]]
for name in ('numpy', 'jax', 'scipy', 'optax', 'pytest', 'pytest-xdist', 'pytest-subtests', 'sphinx',
             'sphinx-autoapi', 'pydata-sphinx-theme', 'myst-parser'):
    try:
        parts.append('%s %s' % (name, importlib.metadata.version(name)))
    except importlib.metadata.PackageNotFoundError:
        parts.append('%s absent' % name)
print(', '.join(parts))
