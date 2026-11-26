#!/usr/bin/env python3
"""K0's inventory: what the deletion of the old sampling kernels removes, and every place that refers to it.

    python3 dev/graded_messages/implementation/k0/k0_inventory.py            # the report (markdown) on stdout
    python3 dev/graded_messages/implementation/k0/k0_inventory.py --json F   # also the raw results as JSON
    python3 dev/graded_messages/implementation/k0/k0_inventory.py --check    # the gate: exit 1 if any code reference
                                                                             #   to a deleted name is left

Static (`ast` + `doctest`), standard library only; it never imports the library. Run it on the tree before K0 for
the plan, and on K0's tree for the gate (`--check`): there the deleted modules and names are gone, so the names are
read from `k0_names.json`, which the first run writes (`--freeze`) from the tree before K0.

WHAT IS DELETED (PACKING_PLAN.md section 7, K0; PART1_PLAN.md section 2.4 and section 10, re-applied to today's
`__all__`):
  - `backend/sampling_derivatives.py`, whole (every top-level statement);
  - the public names of the five sampling modules that the rule names: probing every `compute_*`, `assemble_*`,
    `tv_precompute_probe_frame_sweep`, every `*_from_sweep`; apply and entries `tv_precompute_*_frame_sweep` and every
    `*_from_sweep`; ut3_sampling every name containing `derivatives` or ending in `_from_sweep`; utv_sampling every
    name but the six `utv_{probe,apply,entries}[_transpose]`;
  - `backend/fitting.py`: every public name but `LeastSquaresOracle` and `LeastSquaresBatch` (the old kinds, their
    singletons and aliases, the four `*sumsq_over_*` reductions);
  - `backend/uniform_fitting.py`: every public name but `uniform_minimal`, `pack_sample`, `pack_data`, `pack_batch`;
  - and, by closure, every other top-level statement of those seven modules (private or public, a def, a class, an
    assignment, a module-level `if`) all of whose references in the library come from deleted statements. A
    statement with no reference at all is reported apart ("dead before K0"), with its references from the tests.

WHAT A REFERENCE IS. Every module-level statement is read whole (a module-level `if`, a decorator, a registration
call, a default value), not only the top-level definitions. In code: a `Name` bound in the module (or imported from
the defining module with `from M import n [as a]`), an attribute of a name that is the defining module (any import
alias, and dotted chains `t3toolbox.backend.probing.x`), a string constant that is exactly the name or contains it as
a whole identifier (`getattr`, `mock.patch.object(sd, 'x')`, a script in a string). In doctests: the same, in the
`>>>` examples of every docstring and every page. In prose (docstrings, comments, pages): the name as a whole
identifier, reported as text, never as a code reference. `__all__` entries go with their names.
A `Name` is resolved to the module's top-level binding without regard to local shadowing (a local variable that
shadows a deleted top-level name would be counted as a reference: none is in the report's list, which is checked by
hand).

TESTS. A test is a `def test_*` (method or function). It refers to the deleted code DIRECTLY (its body), THROUGH A
HELPER (a function or method of the same file, or of a helper module of the tests, `import _fits as F`, whose body
refers to it, transitively), or AT IMPORT (the file's module level imports a deleted module or name: the whole file
fails at collection).
"""
import argparse
import ast
import collections
import doctest
import fnmatch
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
NAMES_FILE = os.path.join(HERE, 'k0_names.json')
PKG = 't3toolbox'
SD = 't3toolbox.backend.sampling_derivatives'
SAMPLING = ('probing', 'apply', 'entries', 'ut3_sampling', 'utv_sampling')
FITTING = ('fitting', 'uniform_fitting')
SEVEN = tuple('t3toolbox.backend.' + m for m in SAMPLING + FITTING) + (SD,)
IDENT = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')
EXCLUDED_PAGES = ('docs/release_notes.md',)
# Dead before K0 (no reference anywhere but dev/ records): the scan bodies of the pre-engine t3_apply and t3_entries,
# whose callers P-slices replaced. K0's decision: they go with the old kernels (K0_DESIGN_NOTE.md, Decisions).
DEAD = {'apply': {'_t3_apply_step'}, 'entries': {'_t3_entries_step'}}          # names the removed names on purpose (as CHANGELOG.md does)


# ------------------------------------------------------------------------------------------------ files

def rel(path):
    return os.path.relpath(path, REPO)


def walk(root, suffixes):
    out = []
    for dirpath, dirnames, filenames in os.walk(os.path.join(REPO, root)):
        dirnames[:] = sorted(d for d in dirnames if d not in ('__pycache__', '_build', 'autoapi'))
        for f in sorted(filenames):
            if f.endswith(suffixes):
                out.append(rel(os.path.join(dirpath, f)))
    return out


def module_of(path):
    parts = path[:-3].split('/')
    if parts[-1] == '__init__':
        parts = parts[:-1]
    return '.'.join(parts)


def read(path):
    with open(os.path.join(REPO, path), encoding='utf-8') as f:
        return f.read()


# ------------------------------------------------------------------------------------------------ names of a file

class Bindings(object):
    """What the imports of a piece of code bind: local name -> module (dotted), local name -> (module, name)."""

    def __init__(self, modules=None, names=None):
        self.modules = dict(modules or {})
        self.names = dict(names or {})

    def read(self, tree, known_modules):
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.asname:
                        self.modules[a.asname] = a.name
                    else:
                        self.modules.setdefault(a.name.split('.')[0], a.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                for a in node.names:
                    full = node.module + '.' + a.name
                    if full in known_modules:
                        self.modules[a.asname or a.name] = full
                    else:
                        self.names[a.asname or a.name] = (node.module, a.name)
        return self

    def copy(self):
        return Bindings(self.modules, self.names)


def dotted(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return list(reversed(parts))
    return None


def docstring_nodes(tree):
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                    and isinstance(first.value.value, str):
                out.add(id(first.value))
    return out


def code_refs(tree, bindings, own_module, own_names, known_modules, docstrings=frozenset(), all_strings=frozenset()):
    """Every (module, name, line, how) the code of `tree` refers to. `own_names`: the top-level names of
    `own_module` (a bare Name of one of them refers to it). Strings: `how='string'`."""
    out = []
    attr_value_ids = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            chain = dotted(node)
            if chain is None:
                continue
            for k in range(len(chain) - 1, 0, -1):          # the longest prefix that is a module
                head = chain[0]
                if head in bindings.modules:
                    module = '.'.join([bindings.modules[head]] + chain[1:k])
                else:
                    module = '.'.join(chain[:k])
                if module in known_modules:
                    out.append((module, chain[k], node.lineno, 'attr'))
                    break
            # mark inner attribute nodes so that a.b.c counts once
            inner = node.value
            while isinstance(inner, ast.Attribute):
                attr_value_ids.add(id(inner))
                inner = inner.value
    out = [o for o in out]
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if node.id in bindings.names:
                module, name = bindings.names[node.id]
                out.append((module, name, node.lineno, 'name'))
            elif node.id in own_names:
                out.append((own_module, node.id, node.lineno, 'name'))
        elif isinstance(node, ast.ImportFrom) and node.module:
            for a in node.names:
                if node.module + '.' + a.name not in known_modules:
                    out.append((node.module, a.name, node.lineno, 'import'))
                else:
                    out.append((node.module + '.' + a.name, '<module>', node.lineno, 'import-module'))
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name in known_modules:
                    out.append((a.name, '<module>', node.lineno, 'import-module'))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            how = 'all' if id(node) in all_strings else 'string'
            for line_no, line in enumerate(node.value.split('\n')):
                for m in IDENT.finditer(line):
                    out.append((None, m.group(), node.lineno + line_no, how))   # resolved against the deleted set later
    # dedupe attribute chains reported twice (an Attribute inside an Attribute)
    return out


def doctest_refs(text, first_line, bindings, own_module, own_names, known_modules):
    """Code references in the `>>>` examples of a docstring or a page. Each example sees the imports of the examples
    before it (and of its module)."""
    out = []
    try:
        examples = doctest.DocTestParser().get_examples(text)
    except ValueError:
        return out, ['%d: doctest examples not parsable' % first_line]
    local = bindings.copy()
    notes = []
    for ex in examples:
        try:
            tree = ast.parse(ex.source)
        except SyntaxError:
            notes.append('%d: an example that is not Python' % (first_line + ex.lineno))
            continue
        local.read(tree, known_modules)
        for module, name, line, how in code_refs(tree, local, own_module, own_names, known_modules):
            out.append((module, name, first_line + ex.lineno + line - 1, 'doctest-' + how))
    return out, notes


# ------------------------------------------------------------------------------------------------ the library

class Unit(object):
    """A top-level statement of a library module."""

    def __init__(self, module, node, path):
        self.module, self.node, self.path = module, node, path
        self.defines = set()
        self.kind = type(node).__name__
        self._collect(node)
        self.start, self.end = node.lineno, node.end_lineno
        if getattr(node, 'decorator_list', None):
            self.start = min(d.lineno for d in node.decorator_list)
        self.refs = []          # (module, name, line, how)
        self.doc_refs = []      # the same, in its doctests
        self.prose = []         # (name, line, context) in its docstrings and comments
        self.deleted = False
        self.why = ''

    def _collect(self, node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            self.defines.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name):
                        self.defines.add(n.id)
        elif isinstance(node, (ast.If, ast.Try, ast.For, ast.While, ast.With)):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.stmt):
                    self._collect(child)
            for h in getattr(node, 'handlers', []):
                for child in h.body:
                    self._collect(child)
            for child in getattr(node, 'orelse', []) + getattr(node, 'finalbody', []):
                self._collect(child)

    @property
    def label(self):
        names = sorted(self.defines)
        if self.kind in ('Import', 'ImportFrom'):
            return 'import line %d' % self.start
        if names == ['__all__']:
            return '__all__'
        if names:
            return ', '.join(names)
        return '%s at line %d' % (self.kind, self.start)

    @property
    def is_import(self):
        return self.kind in ('Import', 'ImportFrom')


def comments_by_line(source):
    import io
    import tokenize
    out = {}
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type == tokenize.COMMENT:
            out[tok.start[0]] = tok.string
    return out


def load_library():
    files = walk(PKG, ('.py',))
    known = {module_of(p) for p in files}
    modules = {}
    for path in files:
        module = module_of(path)
        source = read(path)
        tree = ast.parse(source)
        bindings = Bindings().read(tree, known)
        units = [Unit(module, node, path) for node in tree.body]
        own = set()
        for u in units:
            if not u.is_import:
                own |= u.defines
        docs = docstring_nodes(tree)
        all_strings = set()
        for u in units:
            if '__all__' in u.defines:
                all_strings |= {id(c) for c in ast.walk(u.node) if isinstance(c, ast.Constant)}
        comments = comments_by_line(source)
        for u in units:
            u.refs = code_refs(u.node, bindings, module, own, known, docs, all_strings)
            for node in ast.walk(u.node):
                if id(node) in docs:
                    refs, _ = doctest_refs(node.value, node.lineno, bindings, module, own, known)
                    u.doc_refs += refs
                    for k, line in enumerate(node.value.split('\n')):
                        if line.lstrip().startswith(('>>>', '... ')):
                            continue
                        for m in IDENT.finditer(line):
                            u.prose.append((m.group(), node.lineno + k, 'docstring'))
            for line in range(u.start, u.end + 1):
                if line in comments:
                    for m in IDENT.finditer(comments[line]):
                        u.prose.append((m.group(), line, 'comment'))
        # comments and docstring between units (module-level comment blocks): attach to a pseudo unit
        covered = set()
        for u in units:
            covered |= set(range(u.start, u.end + 1))
        loose = Unit(module, ast.Pass(lineno=0, end_lineno=0), path)
        loose.kind, loose.defines = 'loose comments', set()
        for line, text in comments.items():
            if line not in covered:
                for m in IDENT.finditer(text):
                    loose.prose.append((m.group(), line, 'comment'))
        modules[module] = dict(path=path, units=units, loose=loose, bindings=bindings, own=own, tree=tree,
                               all=_all_of(tree))
    return modules, known


def _all_of(tree):
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', None) == '__all__':
            return list(ast.literal_eval(node.value))
    return []


def seeds(modules):
    """The deleted public names of the plan's rule, per module."""
    out = {}
    for m in SEVEN:
        if m not in modules:
            continue
        short = m.rsplit('.', 1)[1]
        names = modules[m]['all']
        if m == SD:
            out[m] = set(modules[m]['own']) - {'__all__'}
        elif short == 'probing':
            out[m] = {n for n in names if n.startswith(('compute_', 'assemble_')) or n.endswith('_from_sweep')
                      or n == 'tv_precompute_probe_frame_sweep'}
        elif short in ('apply', 'entries'):
            out[m] = {n for n in names if n.endswith('_from_sweep') or fnmatch.fnmatch(n, 'tv_precompute_*_frame_sweep')}
            out[m] |= DEAD.get(short, set())
        elif short == 'ut3_sampling':
            out[m] = {n for n in names if 'derivatives' in n or n.endswith('_from_sweep')}
        elif short == 'utv_sampling':
            keep = {'utv_%s%s' % (k, t) for k in ('probe', 'apply', 'entries') for t in ('', '_transpose')}
            out[m] = set(names) - keep
        elif short == 'fitting':
            out[m] = set(names) - {'LeastSquaresOracle', 'LeastSquaresBatch'}
        elif short == 'uniform_fitting':
            out[m] = set(names) - {'uniform_minimal', 'pack_sample', 'pack_data', 'pack_batch'}
    return out


def resolve_strings(refs, deleted_names):
    """Strings were recorded with module None: keep those that are a deleted name of some module."""
    out = []
    for module, name, line, how in refs:
        if module is None:
            for m, names in deleted_names.items():
                if name in names:
                    out.append((m, name, line, how))
        else:
            out.append((module, name, line, how))
    return out


def closure(modules):
    """Mark the deleted units; return (deleted names per module, the public names swept by the closure, the units
    that no library code refers to)."""
    seed = seeds(modules)
    deleted = {m: set(s) for m, s in seed.items()}
    for m in seed:
        for u in modules[m]['units']:
            if m == SD or (u.defines and u.defines <= seed[m] and not u.is_import):
                u.deleted, u.why = True, 'whole module' if m == SD else 'the plan\'s rule'
    # who refers to whom (code, no strings of __all__)
    def referrers(module, names):
        out = []
        for mm, info in modules.items():
            for u in info['units']:
                if u.is_import or '__all__' in u.defines:
                    continue
                for (rm, rn, _, how) in u.refs:
                    if how == 'all':
                        continue
                    if rn in names and (rm == module or (rm is None and mm == module)):
                        if not (mm == module and u.defines & names and rn in u.defines):
                            out.append(u)
                            break
        return out

    swept_public, unreferenced = [], []
    changed = True
    while changed:
        changed = False
        for m in SEVEN:
            if m not in modules or m == SD:
                continue
            for u in modules[m]['units']:
                if u.deleted or u.is_import or not u.defines or '__all__' in u.defines:
                    continue
                if u.defines & {'__all__'}:
                    continue
                kept_public = set(modules[m]['all']) - deleted[m]
                if u.defines & kept_public:
                    continue
                refs = referrers(m, u.defines)
                if not refs:
                    continue
                if all(r.deleted for r in refs):
                    u.deleted, u.why = True, 'orphaned: referred to only by ' + ', '.join(
                        sorted({'%s.%s' % (r.module.rsplit('.', 1)[1], r.label) for r in refs}))
                    deleted[m] |= u.defines
                    if any(not n.startswith('_') for n in u.defines) and not u.defines <= seed[m]:
                        swept_public.append((m, u))
                    changed = True
    for m in SEVEN:
        if m not in modules or m == SD:
            continue
        for u in modules[m]['units']:
            if not u.deleted and not u.is_import and u.defines and '__all__' not in u.defines \
                    and not (u.defines & set(modules[m]['all'])) and not referrers(m, u.defines):
                unreferenced.append((m, u))
    return deleted, swept_public, unreferenced


# ------------------------------------------------------------------------------------------------ tests and others

def python_refs(path, known, deleted_names, test_modules):
    """Code references of a non-library Python file (tests, examples), per enclosing top-level unit and test."""
    source = read(path)
    tree = ast.parse(source)
    bindings = Bindings().read(tree, known | set(test_modules))
    docs = docstring_nodes(tree)
    refs = resolve_strings(code_refs(tree, bindings, None, set(), known, docs), deleted_names)
    hits = [r for r in refs if r[0] in deleted_names and (r[1] in deleted_names[r[0]] or r[1] == '<module>')]
    hits = [r for r in hits if not (r[1] == '<module>' and r[0] != SD)]
    return tree, bindings, hits


def page_refs(path, known, deleted_names):
    text = read(path)
    refs, _ = doctest_refs(text, 1, Bindings(), None, set(), known)
    refs = resolve_strings(refs, deleted_names)
    code = [r for r in refs if r[0] in deleted_names and (r[1] in deleted_names[r[0]]
                                                         or (r[1] == '<module>' and r[0] == SD))]
    all_names = set().union(*deleted_names.values()) | {'sampling_derivatives'}
    prose = []
    in_fence, fence_has_prompt, fence_lines = False, False, []
    lines = text.split('\n')
    for k, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith('```'):
            if in_fence and not fence_has_prompt:
                for j in fence_lines:
                    for m in IDENT.finditer(lines[j]):
                        if m.group() in all_names:
                            prose.append((m.group(), j + 1, 'fence'))
            in_fence, fence_has_prompt, fence_lines = not in_fence, False, []
            continue
        if in_fence:
            fence_lines.append(k)
            if stripped.startswith('>>>'):
                fence_has_prompt = True
            continue
        if stripped.startswith(('>>>', '... ')):
            continue
        for m in IDENT.finditer(line):
            if m.group() in all_names:
                kind = 'role' if re.search(r':py:\w+:`[~.]?[\w.]*%s`' % re.escape(m.group()), line) else 'prose'
                prose.append((m.group(), k + 1, kind))
    return code, prose


def _assigned(node):
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    return [n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name)]


def test_units(tree):
    """(qualified name, node, class) for every function and method, and every module-level and class-level
    assignment (a table of functions read by a test is a path to the code it names); tests are the functions named
    test_*."""
    out = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append((node.name, node, None))
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            out += [(name, node, None) for name in _assigned(node)]
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out.append(('%s.%s' % (node.name, sub.name), sub, node.name))
                elif isinstance(sub, (ast.Assign, ast.AnnAssign)):
                    out += [('%s.%s' % (node.name, name), sub, node.name) for name in _assigned(sub)]
    return out


def _completions(expr):
    """The names a computed attribute name can take when its variable part is a kind (or a kind with a suffix)."""
    kinds = ('probe', 'apply', 'entries')
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Mod) and isinstance(expr.left, ast.Constant):
        return {expr.left.value % k for k in kinds if expr.left.value.count('%') == 1}
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
        if isinstance(expr.right, ast.Constant):
            return {k + expr.right.value for k in kinds}
        if isinstance(expr.left, ast.Constant):
            return {expr.left.value + k for k in kinds}
    return None


def dynamic_names(tree, deleted):
    """`getattr(x, <not a literal>)`: a name built at run time, which no static scan resolves. Returned: those whose
    completions by a kind give a deleted name, and the count of the others (not resolvable: reviewed by hand)."""
    hits, other = [], 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ('getattr', 'hasattr') \
                and len(node.args) >= 2 and not isinstance(node.args[1], ast.Constant):
            names = _completions(node.args[1])
            if names and names & deleted:
                hits.append((node.lineno, ast.unparse(node)[:140], sorted(names & deleted)))
            elif names is None:
                other += 1
    return hits, other


def analyse_tests(known, deleted_names):
    files = walk('tests', ('.py',))
    helpers = {os.path.basename(p)[:-3]: p for p in files if os.path.basename(p).startswith('_')}
    parsed = {}
    for path in files:
        tree, bindings, hits = python_refs(path, known, deleted_names, helpers)
        parsed[path] = (tree, bindings, hits)

    # direct refs per function (by line range); module-level refs
    def lines_of(node):
        start = min([node.lineno] + [d.lineno for d in getattr(node, 'decorator_list', [])])
        return start, node.end_lineno

    funcs = {}            # (path, qualname) -> dict(node, cls, direct=[hits], calls=set())
    module_level = {}     # path -> hits outside any function
    for path, (tree, bindings, hits) in parsed.items():
        units = test_units(tree)
        spans = [(lines_of(n), q) for q, n, c in units]
        module_level[path] = []
        for h in hits:
            owners = [q for (a, b), q in spans if a <= h[2] <= b]
            if owners:
                for q in owners:                  # every assigned name of a statement owns its hits
                    if spans[[qq for _, qq in spans].index(q)][0] == spans[[qq for _, qq in spans].index(owners[-1])][0]:
                        funcs.setdefault((path, q), []).append(h)
            else:
                module_level[path].append(h)
    # call graph within a file and into helper modules
    info = {}
    for path, (tree, bindings, hits) in parsed.items():
        names_in_file = {}
        for q, node, cls in test_units(tree):
            names_in_file.setdefault(q.split('.')[-1], []).append(q)
        for q, node, cls in test_units(tree):
            calls = set()
            for n in ast.walk(node):
                if isinstance(n, ast.Name) and n.id in names_in_file:
                    calls |= {(path, t) for t in names_in_file[n.id] if '.' not in t}
                elif isinstance(n, ast.Name) and n.id in bindings.names:
                    mod, name = bindings.names[n.id]
                    if mod in helpers:
                        calls.add((helpers[mod], name))
                elif isinstance(n, ast.Attribute):
                    if isinstance(n.value, ast.Name) and n.value.id in ('self', 'cls') and cls is not None:
                        calls |= {(path, t) for t in names_in_file.get(n.attr, []) if t.startswith(cls + '.')
                                  or '.' not in t}
                        calls |= {(path, t) for t in names_in_file.get(n.attr, []) if '.' in t}
                    elif isinstance(n.value, ast.Name) and n.value.id in bindings.modules:
                        mod = bindings.modules[n.value.id]
                        if mod in helpers:
                            calls.add((helpers[mod], n.attr))
                    elif isinstance(n.value, ast.Name) and n.value.id in names_in_file:
                        pass
            # a class used by name (a helper class) -> its methods
            info[(path, q)] = dict(node=node, cls=cls, calls=calls, direct=funcs.get((path, q), []))
        # helper classes referenced by name: map class name -> methods
    # module-level helper functions of helper files are 'path, name' entries already (no dot)
    # transitive closure
    reach = {}

    def reaches(key, stack=()):
        if key in reach:
            return reach[key]
        if key in stack:
            return set()
        entry = info.get(key)
        if entry is None:
            # a helper module's top-level name that is a class: its methods
            path, name = key
            out = set()
            for (p, q), e in info.items():
                if p == path and q.startswith(name + '.'):
                    out |= reaches((p, q), stack + (key,))
            ml = module_level.get(path, []) if key[1] == '<module-level>' else []
            reach[key] = out
            return out
        out = {(key, h) for h in entry['direct']}
        for c in entry['calls']:
            if c != key:
                out |= reaches(c, stack + (key,))
        reach[key] = out
        return out

    tests = collections.OrderedDict()
    for path, (tree, bindings, hits) in parsed.items():
        at_import = [h for h in module_level[path] if h[3] in ('import', 'import-module')]
        # a module-level import of a helper module that itself fails at import
        for name, mod in bindings.modules.items():
            if mod in helpers and any(h[3] in ('import', 'import-module') for h in module_level.get(helpers[mod], [])):
                at_import.append((helpers[mod], '<helper fails at import>', 0, 'helper-import'))
        for q, node, cls in test_units(tree):
            if not q.split('.')[-1].startswith('test'):
                continue
            r = reaches((path, q))
            direct = [h for (k, h) in r if k == (path, q)]
            through = sorted({'%s:%s' % (k[0], k[1]) for (k, h) in r if k != (path, q)})
            tests[(path, q)] = dict(direct=direct, through=through, at_import=at_import, line=node.lineno)
    return parsed, module_level, tests, helpers


# ------------------------------------------------------------------------------------------------ report

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--json')
    p.add_argument('--freeze', action='store_true', help='write k0_names.json from this tree (before K0 only)')
    p.add_argument('--check', action='store_true', help='the gate: no code reference to a deleted name')
    args = p.parse_args()

    modules, known = load_library()
    if args.check:
        with open(NAMES_FILE) as f:
            deleted_names = {m: set(v) for m, v in json.load(f)['deleted'].items()}
        known = known | set(deleted_names)
        return gate(modules, known, deleted_names)

    deleted_names, swept_public, unreferenced = closure(modules)
    if args.freeze:
        with open(NAMES_FILE, 'w') as f:
            json.dump({'deleted': {m: sorted(v) for m, v in sorted(deleted_names.items())}}, f, indent=1)
            f.write('\n')
    out = []
    w = out.append
    w('# K0 inventory\n')
    w('Tree: `git rev-parse HEAD` = %s (library, tests, docs and examples as at that commit).\n'
      % os.popen('git -C %s rev-parse --short HEAD' % REPO).read().strip())

    # 1. deleted, per module
    w('## 1. What is deleted, per module\n')
    for m in SEVEN:
        info = modules[m]
        units = info['units']
        dele = [u for u in units if u.deleted]
        if m == SD:
            w('### `%s`: whole, %d top-level statements, %d lines, `__all__` of %d\n'
              % (m, len(units), len(read(info['path']).split('\n')), len(info['all'])))
            continue
        rule = [u for u in dele if u.why == "the plan's rule"]
        orph = [u for u in dele if u.why.startswith('orphaned')]
        w('### `%s`\n' % m)
        w('- by the rule (and the DEAD list) (%d names): %s' % (len(set().union(*[u.defines for u in rule]) if rule else ()),
                                                  ', '.join('`%s`' % u.label for u in rule)))
        w('- by the closure (%d statements):' % len(orph))
        for u in orph:
            w('  - `%s` (line %d): %s' % (u.label, u.start, u.why))
        kept = [u for u in units if not u.deleted and not u.is_import]
        w('- kept: %s' % ', '.join('`%s`' % u.label for u in kept))
        w('- `__all__` after: %s' % ', '.join('`%s`' % n for n in info['all'] if n not in deleted_names[m]))
        w('')
    w('### Public names swept by the closure (not named by the rule)\n')
    w('\n'.join('- `%s.%s`' % (m, u.label) for m, u in swept_public) or 'None.')
    w('\n### Top-level statements of the seven modules that no library code refers to (dead before K0)\n')
    w('\n'.join('- `%s`: `%s` (line %d)' % (m, u.label, u.start) for m, u in unreferenced) or 'None.')

    # 2. kept library code that refers to deleted names
    w('\n## 2. Kept library code that refers to a deleted name\n')
    w('### Code (imports, calls, attributes, strings)\n')
    rows = []
    for m, info in sorted(modules.items()):
        for u in info['units']:
            if u.deleted:
                continue
            for (rm, rn, line, how) in resolve_strings(u.refs, deleted_names):
                if how == 'all':
                    continue
                if rm == SD and rn == '<module>':
                    rows.append('- `%s:%d` (`%s`): imports `%s`' % (info['path'], line, u.label, rm))
                elif rm in deleted_names and rn in deleted_names[rm]:
                    rows.append('- `%s:%d` (`%s`): `%s.%s` (%s)' % (info['path'], line, u.label,
                                                                   rm.rsplit('.', 1)[1], rn, how))
    w('\n'.join(rows) or 'None.')
    w('\n### Doctests of kept code\n')
    rows = []
    for m, info in sorted(modules.items()):
        for u in info['units']:
            if u.deleted:
                continue
            for (rm, rn, line, how) in resolve_strings(u.doc_refs, deleted_names):
                if (rm == SD and rn == '<module>') or (rm in deleted_names and rn in deleted_names[rm]):
                    rows.append('- `%s:%d` (`%s`): `%s.%s` (%s)' % (info['path'], line, u.label,
                                                                   rm.rsplit('.', 1)[1], rn, how))
    w('\n'.join(sorted(set(rows))) or 'None.')
    w('\n### Prose of kept code (docstrings and comments; text, not references)\n')
    all_names = set().union(*deleted_names.values())
    generic = {'APPLY', 'ENTRIES', 'PROBE'}
    rows = collections.defaultdict(list)
    for m, info in sorted(modules.items()):
        for u in info['units'] + [info['loose']]:
            if u.deleted:
                continue
            for name, line, ctx in u.prose:
                if (name in all_names and name not in generic) or name == 'sampling_derivatives':
                    rows[info['path']].append('%d %s (%s)' % (line, name, ctx))
    for path in sorted(rows):
        w('- `%s`: %d: %s' % (path, len(rows[path]), '; '.join(rows[path])))
    w('\n### Imports of the seven modules that no kept code uses after K0 (used before, by deleted code only)\n')
    w('A name counts as used when a `Name` node of kept code (its doctests excluded: they import for themselves) '
      'reads it; a string annotation is not a use.\n')
    rows, already = [], []
    for m in SEVEN:
        if m == SD or m not in modules:
            continue
        info = modules[m]

        def used_by(units):
            used = set()
            for u in units:
                for node in ast.walk(u.node):
                    if isinstance(node, ast.Name):
                        used.add(node.id)
            return used
        used_after = used_by([u for u in info['units'] if not u.deleted and not u.is_import])
        used_before = used_by([u for u in info['units'] if not u.is_import])
        for u in info['units']:
            if not u.is_import:
                continue
            for a in u.node.names:
                local = a.asname or a.name.split('.')[0]
                if a.name == '*' or local in used_after:
                    continue
                (rows if local in used_before else already).append(
                    '- `%s:%d`: `%s`' % (info['path'], u.start, local))
    w('\n'.join(rows) or 'None.')
    w('\nUnused already before K0 (not K0\'s; listed for the record):\n')
    w('\n'.join(already) or 'None.')

    # 3. tests
    w('\n## 3. Tests\n')
    parsed, module_level, tests, helpers = analyse_tests(known, deleted_names)
    by_file = collections.OrderedDict()
    for (path, q), t in tests.items():
        by_file.setdefault(path, []).append((q, t))
    totals = collections.Counter()
    for path, items in by_file.items():
        n = len(items)
        imp = bool(items and items[0][1]['at_import'])
        d = [q for q, t in items if t['direct']]
        th = [q for q, t in items if not t['direct'] and t['through']]
        if not (imp or d or th or module_level.get(path)):
            continue
        totals['files'] += 1
        w('### `%s`: %d tests; at import: %s; direct %d; through a helper %d\n'
          % (path, n, 'YES' if imp else 'no', len(d), len(th)))
        if module_level.get(path):
            w('- module level: ' + '; '.join(sorted({'%d %s.%s (%s)' % (h[2], h[0].rsplit('.', 1)[1], h[1], h[3])
                                                    for h in module_level[path]})))
        for q, t in items:
            if t['direct'] or t['through']:
                names = sorted({'%s.%s' % (h[0].rsplit('.', 1)[1], h[1]) for h in t['direct']})
                w('- `%s` (line %d): %s%s' % (q, t['line'], ', '.join('`%s`' % x for x in names) or '-',
                                              ('; through ' + ', '.join(t['through'])) if t['through'] else ''))
        w('')
    w('### Names built at run time in the tests (`getattr` with a computed name; reviewed by hand)\n')
    rows, others = [], 0
    every = set().union(*deleted_names.values())
    for path in sorted(parsed):
        hits, other = dynamic_names(parsed[path][0], every)
        others += other
        for line, text, names in hits:
            rows.append('- `%s:%d`: `%s` -> %s' % (path, line, text, ', '.join('`%s`' % n for n in names)))
    w('\n'.join(rows) or 'None.')
    w('\n%d other computed names (a variable, not a kind pattern): read by hand; none reaches a deleted name '
      '(the list: `--dynamic`).' % others)
    w('')
    # files with no tests of their own: helper modules, the sweeps
    w('### Files of the tests with no tests of their own (helpers, the oracle sweeps)\n')
    with_tests = {path for (path, q) in tests}
    for path in sorted(parsed):
        hits = parsed[path][2]
        if hits and path not in with_tests:
            w('- `%s`: %s' % (path, '; '.join(sorted({'%d %s.%s (%s)' % (h[2], h[0].rsplit('.', 1)[1], h[1], h[3])
                                                       for h in hits}))))

    # 4. examples and pages
    w('\n## 4. Examples\n')
    rows = []
    for path in walk('examples', ('.py',)):
        _, _, hits = python_refs(path, known, deleted_names, {})
        rows += ['- `%s:%d`: `%s.%s`' % (path, h[2], h[0].rsplit('.', 1)[1], h[1]) for h in hits]
    w('\n'.join(rows) or 'None.')
    w('\n## 5. Pages (`docs/*.md`, `docs/contributor/*.md`, `docs/*.rst`; not `%s`)\n' % ', '.join(EXCLUDED_PAGES))
    pages = [p for p in walk('docs', ('.md', '.rst')) if p.count('/') <= 2 and p not in EXCLUDED_PAGES]
    code_rows, prose_rows = [], collections.defaultdict(list)
    for path in pages:
        code, prose = page_refs(path, known, deleted_names)
        code_rows += ['- `%s:%d`: `%s.%s` (%s)' % (path, h[2], h[0].rsplit('.', 1)[1], h[1], h[3]) for h in code]
        for name, line, kind in prose:
            if name not in generic:
                prose_rows[path].append('%d %s (%s)' % (line, name, kind))
    w('### In doctest examples (run by the gate)\n')
    w('\n'.join(code_rows) or 'None.')
    w('\n### In prose, fenced code without prompts, and Sphinx roles (P19\'s, unless a role or a toctree fails `-W`)\n')
    for path in sorted(prose_rows):
        w('- `%s`: %d: %s' % (path, len(prose_rows[path]), '; '.join(prose_rows[path])))
    print('\n'.join(out))
    if args.json:
        with open(args.json, 'w') as f:
            json.dump({'deleted': {m: sorted(v) for m, v in deleted_names.items()},
                       'tests': {'%s::%s' % k: dict(direct=sorted({'%s.%s' % (h[0], h[1]) for h in v['direct']}),
                                                     through=v['through'], at_import=bool(v['at_import']))
                                 for k, v in tests.items()}}, f, indent=1)
    return 0


# Strings that name a deleted name on purpose in K0's tree: (path pattern, name pattern, reason). Everything else a
# string names is a failure of the gate, as a code reference is.
STRING_ALLOWLIST = (
    ('tests/oracle_sweeps/o1_*.py', '*_derivatives',
     'labels of the O1 sweep\'s checks (the review harness, verbatim copies); the oracle is the dense one'),
    ('tests/test_jets.py', 'check_perturbation_*', 'asserts the name is absent from jets and t3toolbox.fitting'),
    ('examples/*.py', 'PROBE', 'the English word in a printed heading'),
)


def _allowed(path, name):
    return any(fnmatch.fnmatch(path, pp) and fnmatch.fnmatch(name, np_) for pp, np_, _ in STRING_ALLOWLIST)


def gate(modules, known, deleted_names):
    """On K0's tree: every code or doctest reference to a deleted name in the library, the tests and the examples
    (an import, a name, an attribute, a doctest line, a string not in STRING_ALLOWLIST, a computed `getattr` or
    `'..%s..' % kind` string that completes to one), and every doctest reference in the pages; and the file of the
    deleted module. Exit 1 if there is any."""
    found, allowed = [], []
    every = set().union(*deleted_names.values())

    def add(path, line, rm, rn, how):
        if how in ('string', 'all') and _allowed(path, rn):
            allowed.append((path, line, rm, rn, how))
        else:
            found.append((path, line, rm, rn, how))

    for m, info in modules.items():
        for u in info['units']:
            for (rm, rn, line, how) in resolve_strings(u.refs + u.doc_refs, deleted_names):
                if (rm == SD) or (rm in deleted_names and rn in deleted_names[rm]):
                    add(info['path'], line, rm, rn, how)
    for root in ('t3toolbox', 'tests', 'examples'):
        for path in walk(root, ('.py',)):
            tree, _, hits = python_refs(path, known, deleted_names, {})
            if root != 't3toolbox':
                for h in hits:
                    add(path, h[2], h[0], h[1], h[3])
            computed, _ = dynamic_names(tree, every)
            for line, text, names in computed:
                found.append((path, line, '<computed>', ','.join(names), 'getattr'))
            for node in ast.walk(tree):          # '..%s..' % kind, wherever the string is used
                if isinstance(node, ast.Constant) and isinstance(node.value, str) and '%s' in node.value \
                        and len(node.value) < 80:
                    for kind in ('probe', 'apply', 'entries'):
                        for w in IDENT.findall(node.value.replace('%s', kind)):
                            if w in every and not _allowed(path, w):
                                found.append((path, node.lineno, '<format>', w, 'format'))
    for path in [p for p in walk('docs', ('.md', '.rst')) if p.count('/') <= 2]:
        code, _ = page_refs(path, known, deleted_names)
        found += [(path, h[2], h[0], h[1], h[3]) for h in code]
    if os.path.exists(os.path.join(REPO, 't3toolbox/backend/sampling_derivatives.py')):
        found.append(('t3toolbox/backend/sampling_derivatives.py', 0, SD, '<module>', 'file exists'))
    for row in sorted(set(found)):
        print('%s:%d: %s.%s (%s)' % row)
    print('%d references to deleted names; %d allowed strings (STRING_ALLOWLIST)' % (len(set(found)),
                                                                                      len(set(allowed))))
    return 1 if found else 0


if __name__ == '__main__':
    sys.exit(main())
