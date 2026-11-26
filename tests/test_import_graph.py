# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/T3Toolbox
"""The import graph of the package's module files (T18 of the aligned-padded-frame plan).

The graph is built from the source (``ast``), on module FILES, not packages: ``from t3toolbox import corewise`` is
an edge to ``t3toolbox/corewise.py``, and Python's implicit import of the parent packages (``t3toolbox/__init__.py``
for every ``t3toolbox.*`` import) is not an edge -- it would make every module depend on the whole frontend. An
edge to a package ``__init__`` appears only when a module imports the package itself, or a name defined there.
Every ``import`` statement counts, function-level ones included: a lazy import is still a dependency.

Six facts are pinned:

- the backend's import graph (everything reachable from ``t3toolbox/backend/``) is acyclic;
- ``backend/optimizers.py`` imports only ``common``, ``regularization`` and ``corewise`` -- the optimizer layer
  stays T3-free, reaching every T3-specific operation through the geometry;
- the padded-frame modules import only the modules their plan names (downward; no new cycle);
- no backend module imports ``t3toolbox.safety``: there is no safe mode in the backend, which gives checkers and not
  guarantees, and the frontend enforces (graded messages, P3);
- ``backend/jets.py`` and ``backend/data_preparation.py`` import only what they are built on;
- the old sampling kernels' module, ``backend/sampling_derivatives.py``, is gone (deleted in K0; the engine,
  ``backend/sampling_variables.py``, computes every sampling variable).
"""
import ast
import importlib.util
import pathlib
import unittest

import t3toolbox

PACKAGE_ROOT = pathlib.Path(t3toolbox.__file__).resolve().parent


def module_files():
    """Dotted module name -> source file, for every module file of the package (a package is its __init__)."""
    out = {}
    for path in sorted(PACKAGE_ROOT.rglob('*.py')):
        parts = ('t3toolbox',) + path.relative_to(PACKAGE_ROOT).with_suffix('').parts
        if parts[-1] == '__init__':
            parts = parts[:-1]
        out['.'.join(parts)] = path
    return out


MODULES = module_files()


def resolve(dotted):
    """The module file a dotted name lives in: its longest prefix that is a module of the package (None outside)."""
    parts = dotted.split('.')
    for k in range(len(parts), 0, -1):
        name = '.'.join(parts[:k])
        if name in MODULES:
            return name
    return None


def imports_of(module):
    """The package modules ``module`` imports (every import statement in its source)."""
    tree = ast.parse(MODULES[module].read_text(), filename=str(MODULES[module]))
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise AssertionError('%s: relative import (the package uses absolute imports only)' % module)
            targets = [node.module + ('' if alias.name == '*' else '.' + alias.name) for alias in node.names]
        else:
            continue
        out.update(r for r in map(resolve, targets) if r is not None)
    out.discard(module)
    return out


GRAPH = {module: imports_of(module) for module in MODULES}


def find_cycle(graph, roots):
    """A cycle reachable from ``roots`` as a list of modules (first == last), or None."""
    state = {}                      # 1 = on the current path, 2 = done

    def visit(u, path):
        state[u] = 1
        path.append(u)
        for v in sorted(graph[u]):
            if state.get(v) == 1:
                return path[path.index(v):] + [v]
            if v not in state:
                cycle = visit(v, path)
                if cycle:
                    return cycle
        path.pop()
        state[u] = 2
        return None

    for root in sorted(roots):
        if root not in state:
            cycle = visit(root, [])
            if cycle:
                return cycle
    return None


class TestImportGraph(unittest.TestCase):

    def test_the_graph_is_on_module_files(self):
        """``from t3toolbox import safety`` resolves to the module file, not to the package's __init__ (the package's
        own ``__init__`` is the one module that imports this way since P16 part 3 dropped the display's
        ``from t3toolbox import corewise``); a dotted import resolves to its module."""
        self.assertIn('t3toolbox.safety', GRAPH['t3toolbox'])
        self.assertNotIn('t3toolbox', GRAPH['t3toolbox'])
        self.assertIn('t3toolbox.backend.optimizers', GRAPH['t3toolbox.backend.optimizer_display'])
        self.assertNotIn('t3toolbox', GRAPH['t3toolbox.backend.optimizer_display'])

    def test_the_backend_graph_is_acyclic(self):
        backend = [m for m in MODULES if m.startswith('t3toolbox.backend.')]
        self.assertGreater(len(backend), 30)
        self.assertIsNone(find_cycle(GRAPH, backend))

    def test_the_cycle_finder_finds_a_cycle(self):
        self.assertEqual(find_cycle({'a': {'b'}, 'b': {'c'}, 'c': {'a'}}, ['a']), ['a', 'b', 'c', 'a'])
        self.assertIsNone(find_cycle({'a': {'b', 'c'}, 'b': {'c'}, 'c': set()}, ['a']))

    def test_the_optimizer_layer_is_t3_free(self):
        self.assertEqual(GRAPH['t3toolbox.backend.optimizers'],
                         {'t3toolbox.backend.common', 't3toolbox.backend.regularization', 't3toolbox.corewise'})

    def test_the_padded_frame_modules_import_downward(self):
        allowed = {
            't3toolbox.backend.padded_frames': {'common', 'contractions', 'linalg', 'ranks', 'sharing',
                                                'fv_operations'},
            't3toolbox.backend.uniform_padded_frames': {'common', 'contractions', 'linalg', 'ranks', 'sharing',
                                                        'ufv_masking', 'fv_operations', 'padded_frames'},
        }
        for module, names in allowed.items():
            if module not in MODULES:                    # the uniform module lands in a later slice
                continue
            with self.subTest(module=module):
                self.assertLessEqual(GRAPH[module], {'t3toolbox.backend.' + n for n in names})

    def test_no_backend_module_imports_safety(self):
        """There is no safe mode in the backend: no module under ``t3toolbox/backend/`` imports ``t3toolbox.safety``,
        in either import form or inside a function, directly or through another backend module. (The frontend
        modules do, and call the backend's checkers.)"""
        backend = sorted(m for m in MODULES if m.startswith('t3toolbox.backend'))
        self.assertGreater(len(backend), 30)
        self.assertIn('t3toolbox.safety', MODULES)
        for module in backend:
            with self.subTest(module=module):
                self.assertNotIn('t3toolbox.safety', GRAPH[module])
                reached, frontier = set(), {module}                     # everything the module reaches
                while frontier:
                    reached |= frontier
                    frontier = set().union(*(GRAPH[m] for m in frontier)) - reached
                self.assertNotIn('t3toolbox.safety', reached)
        self.assertIn('t3toolbox.safety', GRAPH['t3toolbox.fitting'])   # the finder sees the import where it is

    def test_the_import_finder_sees_safety_in_both_import_forms(self):
        """The two spellings a backend module could use resolve to the safety module (so the test above cannot
        be walked around by the other form)."""
        self.assertEqual(resolve('t3toolbox.safety'), 't3toolbox.safety')                  # import t3toolbox.safety, and
        #                                                           from t3toolbox import safety (module + '.' + name)
        self.assertEqual(resolve('t3toolbox.safety.checks_active'), 't3toolbox.safety')    # from t3toolbox.safety import ...
        self.assertIn('t3toolbox.safety', GRAPH['t3toolbox.manifold'])                     # a module that uses it today

    def test_the_jet_and_preparation_modules_import_downward(self):
        # jets reads the jet shape with contractions' one reader (P4d); contractions does not import jets
        self.assertEqual(GRAPH['t3toolbox.backend.jets'], {'t3toolbox.backend.common', 't3toolbox.backend.contractions'})
        self.assertNotIn('t3toolbox.backend.jets', GRAPH['t3toolbox.backend.contractions'])
        self.assertEqual(GRAPH['t3toolbox.backend.data_preparation'],
                         {'t3toolbox.backend.common', 't3toolbox.backend.jets', 't3toolbox.backend.ut3_operations'})

    def test_the_old_sampling_kernels_are_gone(self):
        self.assertNotIn('t3toolbox.backend.sampling_derivatives', MODULES)
        self.assertIsNone(importlib.util.find_spec('t3toolbox.backend.sampling_derivatives'))


if __name__ == '__main__':
    unittest.main()
