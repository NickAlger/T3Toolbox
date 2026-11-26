"""Reviewer 4: is the dense reference independent of the engine and of jets (K1 part 1)? And which tests reach it?

    PYTHONPATH=<tree> python -B ref_independence.py --api old|new [--tests FILE ...]

(1) Static, on <tree>/t3toolbox/backend/sampling_definitions.py (ast, code only, docstrings excluded): every import;
    every attribute read off an imported module; every bare name that is a known engine/jets function. Allowed: the
    three dense functions (probing.dense_probe, apply.dense_apply, entries.dense_entries) and the star import of
    common. Anything else is printed as SUSPECT.
(2) Dynamic: a profile hook records every Python function entered while the 37 public functions run with both
    weights (probe at (2,), apply at (2, 1), entries), and prints every function outside sampling_definitions with
    its module. FORBIDDEN modules (jets, the engine, its steps, the fitting layer, the uniform layer) fail the run.
(3) The tests (--tests): for each file, the module-local call graph (functions and methods, by name), the helpers
    that reach the reference (call sdef.* or getattr(sdef, ...), directly or through local helpers), and for each test
    method: the reference calls, the reference-reaching helpers, and the library calls besides. A test that uses a
    reference-reaching helper is listed for reading by hand: is that helper on the expected side of a comparison
    with the reference?
"""
import argparse
import ast
import collections
import os
import sys

import numpy as np

FORBIDDEN = ('jets', 'sampling_variables', 'sampling_steps', 'fitting', 'uniform_fitting', 'data_preparation',
             'optimizers', 'ut3_operations', 'ut3_sampling', 'utv_sampling', 'geometry', 'curvature')
ALLOWED_ATTRS = {('probing', 'dense_probe'), ('apply', 'dense_apply'), ('entries', 'dense_entries')}
ENGINE_NAMES = {'sampling_variables', 'data_metric', 'blocks_to_weight', '_metric', '_omega_squared', 'check_data',
                'check_weight', 'jet_to_dual_jet', 'probe_data_metric', 'apply_data_metric', 'entries_data_metric',
                '_check_pair', '_lowest_precision_dtype', 'pack_if_ragged', '_pack_data_space'}


def code_only(tree):
    """Drop docstrings, so a name in prose is not read as a use."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module, ast.AsyncFunctionDef)):
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(getattr(node.body[0], 'value', None),
                                                                                ast.Constant):
                node.body = node.body[1:] or [ast.Pass()]
    return tree


def static(path):
    tree = code_only(ast.parse(open(path).read()))
    modules = {}
    print('(1) static: %s' % path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                modules[a.asname or a.name.split('.')[0]] = a.name
                print('    import %s as %s' % (a.name, a.asname or a.name))
        elif isinstance(node, ast.ImportFrom):
            print('    from %s import %s' % (node.module, ', '.join(a.name for a in node.names)))
            if node.module and node.module.split('.')[-1] in FORBIDDEN:
                print('    SUSPECT: a from-import of a forbidden module')
    bad = 0
    seen = collections.Counter()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in modules:
            key = (node.value.id, node.attr)
            if node.value.id in ('np', 'math', 'typ', 'dc', 'jax', 'itertools'):
                continue
            seen[key] += 1
        if isinstance(node, ast.Name) and node.id in ENGINE_NAMES:
            print('    SUSPECT: the name %s (line %d)' % (node.id, node.lineno))
            bad += 1
    for key, n in sorted(seen.items()):
        tag = 'ok     ' if key in ALLOWED_ATTRS else 'SUSPECT'
        bad += key not in ALLOWED_ATTRS
        print('    %s %s.%s x%d' % (tag, key[0], key[1], n))
    return bad


def dynamic(api):
    import t3toolbox.backend.sampling_definitions as sdef
    pkg = os.path.dirname(os.path.dirname(os.path.abspath(sdef.__file__)))     # .../t3toolbox
    rng = np.random.default_rng(3)
    d, N, nD, nU, rL, rR = 3, (3, 4, 3), (2, 3, 2), (3, 2, 2), (2, 2, 3, 1), (1, 3, 2, 2)
    D = [rng.standard_normal((nD[i], N[i])) for i in range(d)]
    U = [rng.standard_normal((rL[i], nU[i], rR[i + 1])) for i in range(d)]
    L = [rng.standard_normal((rL[i], nD[i], rL[i + 1])) for i in range(d)]
    R = [rng.standard_normal((rR[i], nD[i], rR[i + 1])) for i in range(d)]
    frame = (D, U, L, R)
    V = [rng.standard_normal((nU[i], N[i])) for i in range(d)]
    H = [rng.standard_normal((rL[i], nD[i], rR[i + 1])) for i in range(d)]
    cd = ([rng.standard_normal(B.shape) for B in D], [rng.standard_normal(G.shape) for G in L])
    W = (2,)
    entered = collections.Counter()

    def hook(frame_, event, arg):
        if event == 'call':
            co = frame_.f_code
            if co.co_filename.startswith(pkg + os.sep):          # the library only
                entered[(os.path.relpath(co.co_filename, pkg), co.co_name)] += 1
    for kind, order in (('probe', (2,)), ('apply', (2, 1)), ('entries', ())):
        m, F = len(order), tuple(n + 1 for n in order)
        if kind == 'entries':
            smp = np.stack([rng.integers(0, n, size=W) for n in N])
            zsh = [W]
        else:
            smp = [rng.standard_normal(F + W + (n,)) for n in N]
            zsh = [F + W + (n,) for n in N] if kind == 'probe' else [F + W]
        y = [rng.standard_normal(s) for s in zsh]
        full = [0.5 + rng.random(s) for s in zsh]
        b = 0.5 + rng.random((d,) + F if kind == 'probe' else F)
        if kind != 'probe':
            y, full = y[0], full[0]
        if api == 'new':
            wargs = dict(block_weight=b, full_weight=full)
        else:
            wargs = dict(weight=full)
        sys.setprofile(hook)
        try:
            for name in sdef.__all__:
                getattr(sdef, name)(kind, frame, smp, None, order, variations=(V, H), core_directions=cd, y=y,
                                    **wargs)
        finally:
            sys.setprofile(None)
    print('(2) dynamic: functions entered outside sampling_definitions.py')
    bad = 0
    for (mod, fn), n in sorted(entered.items()):
        if mod == os.path.join('backend', 'sampling_definitions.py'):
            continue
        base = os.path.basename(mod)[:-3]
        tag = 'FORBIDDEN' if base in FORBIDDEN else 'ok       '
        bad += base in FORBIDDEN
        print('    %s %s.%s x%d' % (tag, mod, fn, n))
    return bad


class Graph(ast.NodeVisitor):
    def __init__(self):
        self.defs = {}                   # qualified name -> node
        self.calls = collections.defaultdict(set)
        self.ref = set()
        self.lib = collections.defaultdict(set)
        self.stack = []

    def visit_ClassDef(self, node):
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node):
        q = '.'.join(self.stack + [node.name])
        self.defs[q] = node
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def current(self):
        return '.'.join(self.stack)

    def visit_Call(self, node):
        cur = self.current()
        f = node.func
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
            if f.value.id == 'sdef':
                self.ref.add(cur)
            elif f.value.id in ('self', 'cls', 'P', 'problem'):
                self.calls[cur].add(f.attr)
            else:
                self.lib[cur].add('%s.%s' % (f.value.id, f.attr))
        elif isinstance(f, ast.Name):
            if f.id == 'getattr' and node.args and isinstance(node.args[0], ast.Name) and node.args[0].id == 'sdef':
                self.ref.add(cur)
            else:
                self.calls[cur].add(f.id)
        self.generic_visit(node)


def tests(paths):
    print('(3) the tests')
    for path in paths:
        g = Graph()
        g.visit(ast.parse(open(path).read()))
        short = {q.split('.')[-1]: q for q in g.defs}
        reach = set(g.ref)
        changed = True
        while changed:
            changed = False
            for q in g.defs:
                if q in reach:
                    continue
                if any(short.get(c) in reach for c in g.calls.get(q, ())) or \
                        any(c.startswith(q + '.') for c in reach):
                    reach.add(q)
                    changed = True
        helpers = sorted(q for q in reach if not q.split('.')[-1].startswith('test'))
        print('  %s: helpers reaching the reference: %s' % (path, ', '.join(helpers) or 'none'))
        for q in sorted(g.defs):
            if not q.split('.')[-1].startswith('test'):
                continue
            via = sorted(c for c in g.calls.get(q, ()) if short.get(c) in reach and short.get(c) != q)
            direct = q in g.ref
            if direct or via:
                lib = sorted(x for x in g.lib.get(q, ()) if not x.startswith(('np.', 'self.')))
                flag = 'READ ' if via else '     '
                print('    %s %s: direct=%s via=%s lib=%s' % (flag, q, direct, via, lib))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--api', choices=('old', 'new'), required=True)
    ap.add_argument('--tests', nargs='*', default=[])
    args = ap.parse_args()
    import t3toolbox.backend.sampling_definitions as sdef
    bad = static(sdef.__file__)
    bad += dynamic(args.api)
    tests(args.tests)
    print('RESULT: %s' % ('independent' if not bad else '%d suspect or forbidden entries' % bad))
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
