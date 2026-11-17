#!/usr/bin/env python3
"""Inventory of the references to the names that Part 1 of the graded-messages refactor changes or removes.

    python3 dev/graded_messages/implementation/inventory.py                    # the summary tables (markdown)
    python3 dev/graded_messages/implementation/inventory.py --lines --name tv_probe --kind call
    python3 dev/graded_messages/implementation/inventory.py --lines --group method --file tests/test_dispatch.py
    python3 dev/graded_messages/implementation/inventory.py --count --name dense_probe --kind call --by file
    python3 dev/graded_messages/implementation/inventory.py --freeze           # P0 only: (re)write inventory_names.json

It is static: it reads the files and never imports the library. It needs only the standard library.

WHAT IS SEARCHED (PART1_PLAN.md section 5.3, slice P0): `t3toolbox/` (the package `__init__.py` included),
`tests/`, `docs/`, `examples/` and `CLAUDE.md`, and the root `README.md` (added in the review of P0: it holds the
old call forms and no slice named it); the files `*.py`, `*.md`, `*.rst`, `*.tex`, `*.txt`. Not searched:
`CHANGELOG.md` and `docs/release_notes.md`, which name the removed names on purpose and which P20's final search
leaves out (their counts are in a table of their own at the end of the summary); `dev/` (the prototypes are
records, O-26); `docs/_build`, `docs/autoapi`, binary files, the consumer T3Polynomial.

THE NAMES are frozen in `inventory_names.json` beside this script (written at P0 by `--freeze`, from the tree as it
was then), so the inventory keeps meaning the same thing after a slice has removed a module:
  sampling   the 166 public names of the six sampling modules (`backend/probing`, `apply`, `entries`,
             `ut3_sampling`, `utv_sampling`, `sampling_derivatives`): their `__all__` lists. `dense_probe` is one.
  method     the 51 frontend sampling methods: those of `TuckerTensorTrain`, `T3Tangent`,
             `UniformTuckerTensorTrain` and `UT3Tangent` named `probe*`, `apply*` or `entries*`
             (`apply_masks` is not one). 21 distinct names, since the classes share them.
  factory    the six model factories of `t3toolbox/fitting.py` (`*_model`).
  other      `Problem`, `least_squares_problem`, `_canonical_weight`, `_check_ragged_sample`, `chunk_size`,
             `sum_over_probes`, `ztildes`.

WHAT IT CAN SEE. Every occurrence of a name as a whole identifier, anywhere in a searched file, and each one is
put in exactly one context and given one kind:
  context `code`      Python source outside strings and comments. Kinds, from the syntax tree:
                        def      a `def` or `class` of that name (a local helper of the same name counts here)
                        param    a parameter of a `def`
                        import   `from m import NAME [as X]`
                        import-module   the name inside a module path (`from t3toolbox.backend.apply import ...`,
                                 `import t3toolbox.backend.entries as e`): the module, not the function
                        call     `NAME(...)` or `anything.NAME(...)`
                        kwarg    `f(NAME=...)`
                        ref      any other use: a table of functions, an argument, `obj.NAME` without a call
                        assign   a local variable or attribute of that name being set
  context `doctest`   the `>>>` examples of a docstring or of a documentation page, parsed the same way (same kinds,
                      and `str` for a name inside a string literal of an example, such as `kind='probe_derivatives'`).
  context `string`    a string that is not a docstring: `getattr(x, 'NAME')`, labels, a script in a string given to a
                      subprocess. Kind `all` for an entry of an `__all__` list; otherwise guessed from the text
                      (`call` if followed by `(`, `kwarg` if followed by `=`, else `ref`).
  context `comment`, `docstring` (the prose of a docstring), `page` (the prose of a .md/.rst/.tex/.txt file),
  context `codeblock` (a fenced code block of a page without `>>>`): kind guessed from the text as for strings.
  kind `glob`         in prose: the name written through a brace pattern, `t3_{apply,entries,probe}_corewise_transpose`.
Imports are followed: after `from m import t3_probe as tp`, every `tp` is counted as `t3_probe` (marked `alias`).
(Not for `apply` and `entries`, which are also backend modules: `from t3toolbox.backend import apply as bapply`
imports the module; it is kind `import-module`, and `bapply` is not counted as the method `apply`.)
For each occurrence in code or a doctest the script also says what the name is bound to, as far as one file tells:
  `lib:<module>`   imported from that module, or an attribute of a name that is that module (`probing.dense_probe`,
                   through any import alias), or of one of the four frontend classes (`cls:T3Tangent`)
  `local`          defined by a `def` in the same file (a test's or an example's own helper of the same name)
  `obj`            an attribute of some object (a method call `x.probe(ww)`): the type of `x` is not known
  `-`              not determined
A doctest is assumed to see the names its own module defines or imports.

WHAT IT CANNOT SEE.
  - The type of an object. `x.probe(ww)` is counted under the method name `probe` whatever `x` is, so the short
    names `probe`, `apply`, `entries` include every other attribute of that name. For those three, and for
    `Problem`, the counts in prose are English words and module names more often than references.
  - A name that is built at run time: `getattr(x, kind + '_derivatives')`, `f't3_{kind}_transpose'`, a loop over
    `('probe', 'apply')` that forms method names. (A literal name inside a string is seen.)
  - A name written through a star or a slash: `*_from_sweep`, `compute_*`, `probe/apply/entries_transpose`. The
    star patterns that match at least one name are listed (`--globs`), in backticks or not, without being counted
    for any name. Markdown emphasis (`*word*`, `**word**`) is not taken for a pattern.
  - A function reached through a variable (`op = t.probe_transpose; op(...)`): the reference `t.probe_transpose` is
    seen (kind `ref`), the call `op(...)` is not attributed.
  - Rebinding and scopes: imports are read for the whole file, not per function. `from m import *` is not
    followed (the binding is then `-`). `tests/oracle_sweeps/o1_ragged.py` and `o1_uniform.py` take their helpers
    from `o1_common` this way. Forty-seven library modules have `from t3toolbox.backend.common import *`, which
    does not matter here: `common` defines none of the names.
  - That a string is a kind name. The old kind strings (`'probe'`, `'apply_derivatives'`, ...) are spelled like
    the methods, so they are counted under the method names, in the `string` context (and `str` in doctests).
  - A string that holds code is not parsed; its kinds are guessed from the text.
"""
import argparse
import ast
import collections
import doctest
import fnmatch
import io
import json
import os
import re
import sys
import tokenize

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
NAMES_FILE = os.path.join(HERE, 'inventory_names.json')

ROOTS = ('t3toolbox', 'tests', 'docs', 'examples', 'CLAUDE.md', 'README.md')
# These name the removed names on purpose (the history of the releases), and P20's final search leaves them out
# (PART1_PLAN.md section 10). They are not in the inventory; the summary gives their counts in a table of its own.
EXCLUDED = ('CHANGELOG.md', 'docs/release_notes.md')
SUFFIXES = ('.py', '.md', '.rst', '.tex', '.txt')
SKIP_DIRS = ('_build', 'autoapi', '__pycache__')

SAMPLING_MODULES = ('probing', 'apply', 'entries', 'ut3_sampling', 'utv_sampling', 'sampling_derivatives')
FRONTEND_CLASSES = (('tucker_tensor_train', 'TuckerTensorTrain'), ('manifold', 'T3Tangent'),
                    ('uniform_tucker_tensor_train', 'UniformTuckerTensorTrain'), ('uniform_manifold', 'UT3Tangent'))
OTHER_NAMES = ('Problem', 'least_squares_problem', '_canonical_weight', '_check_ragged_sample', 'chunk_size',
               'sum_over_probes', 'ztildes')
NOISY = ('probe', 'apply', 'entries', 'Problem')        # English words / module names: prose counts mean little
MODULE_NAMES = ('apply', 'entries')                      # also backend modules: `import apply as bapply` is no method

IDENT = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')
BRACE = re.compile(r'[A-Za-z0-9_]*(?:\{[A-Za-z0-9_,]*\}[A-Za-z0-9_]*)+')
STAR = re.compile(r'(?<![A-Za-z0-9_*])(?:[A-Za-z0-9_]+\*|\*[A-Za-z0-9_]+)[A-Za-z0-9_*]*')

Occurrence = collections.namedtuple('Occurrence', 'name file line context kind binding alias text')


# ----------------------------------------------------------------------------------------- the names

def freeze_names():
    """Read the names from the tree as it is now and write inventory_names.json (P0 only)."""
    def top(path):
        with open(os.path.join(REPO, path)) as f:
            return ast.parse(f.read())

    sampling = {}
    for module in SAMPLING_MODULES:
        for node in top('t3toolbox/backend/%s.py' % module).body:
            if isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', None) == '__all__':
                sampling[module] = list(ast.literal_eval(node.value))
    methods = {}
    for module, cls in FRONTEND_CLASSES:
        for node in top('t3toolbox/%s.py' % module).body:
            if isinstance(node, ast.ClassDef) and node.name == cls:
                methods[cls] = [n.name for n in node.body if isinstance(n, ast.FunctionDef)
                                and re.match(r'(probe|apply|entries)(_|$)', n.name) and n.name != 'apply_masks']
    factories = [n.name for n in top('t3toolbox/fitting.py').body
                 if isinstance(n, ast.FunctionDef) and n.name.endswith('_model') and not n.name.startswith('_')]
    names = {'sampling': sampling, 'method': methods, 'factory': factories, 'other': list(OTHER_NAMES)}
    with open(NAMES_FILE, 'w') as f:
        json.dump(names, f, indent=1)
        f.write('\n')
    return names


def load_names():
    with open(NAMES_FILE) as f:
        names = json.load(f)
    group_of, detail = {}, {}                                 # name -> group; name -> where it is defined
    for module, lst in names['sampling'].items():
        for n in lst:
            group_of[n] = 'sampling'
            detail.setdefault(n, []).append(module)
    for cls, lst in names['method'].items():
        for n in lst:
            group_of.setdefault(n, 'method')
            detail.setdefault(n, []).append(cls)
    for n in names['factory']:
        group_of[n] = 'factory'
    for n in names['other']:
        group_of[n] = 'other'
    return names, group_of, detail


# ----------------------------------------------------------------------------------------- python syntax

def _dotted(node):
    """`a.b.c` as a string for a chain of Name/Attribute nodes, else None."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return '.'.join(reversed(parts))
    return None


class FileNames(object):
    """What one file's imports and defs say about its names (read for the whole file, no scopes)."""

    def __init__(self):
        self.modules = {}        # local name -> dotted module it is     (`import a.b as x`, `from a import b`)
        self.imported = {}       # local name -> (module, original name) (`from m import n as x`)
        self.defs = set()        # names defined by def/class anywhere in the file

    def read(self, tree):
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.asname:
                        self.modules[a.asname] = a.name
                    else:
                        self.modules[a.name.split('.')[0]] = a.name.split('.')[0]
            elif isinstance(node, ast.ImportFrom):
                for a in node.names:
                    module = node.module or '.'
                    self.imported[a.asname or a.name] = (module, a.name)
                    self.modules.setdefault(a.asname or a.name, module + '.' + a.name)   # may be a submodule
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.defs.add(node.name)

    def copy(self):
        other = FileNames()
        other.modules, other.imported, other.defs = dict(self.modules), dict(self.imported), set(self.defs)
        return other

    def binding_of_name(self, ident):
        if ident in self.imported:
            return 'lib:' + self.imported[ident][0]
        if ident in self.defs:
            return 'local'
        return '-'

    def binding_of_receiver(self, value):
        dotted = _dotted(value)
        if dotted is None:
            return 'obj'
        head, _, rest = dotted.partition('.')
        last = dotted.rsplit('.', 1)[-1]
        if last in [c for _, c in FRONTEND_CLASSES]:
            return 'cls:' + last
        if head in self.modules:
            full = self.modules[head] + ('.' + rest if rest else '')
            return 'lib:' + full
        return 'obj'


class _CharColumns(dict):
    """A dict keyed by (line, column) that turns the syntax tree's columns, which count UTF-8 bytes, into columns
    that count characters, as the tokenizer and the regular expressions do. Other keys pass through."""

    def __init__(self, source):
        dict.__init__(self)
        self._lines = source.split('\n')

    def _key(self, key):
        if len(key) == 2 and isinstance(key[0], int):
            line = self._lines[key[0] - 1] if 0 < key[0] <= len(self._lines) else ''
            if not line.isascii():
                return (key[0], len(line.encode('utf-8')[:key[1]].decode('utf-8', errors='ignore')))
        return key

    def __setitem__(self, key, value):
        dict.__setitem__(self, self._key(key), value)

    def setdefault(self, key, default=None):
        return dict.setdefault(self, self._key(key), default)


def classify_tree(tree, names, targets, source):
    """Map (line, col) of every identifier the tree gives a role to -> (kind, binding, original name or None).

    Lines are 1-based within the parsed `source`, columns 0-based, in characters. `original name` is set when the
    identifier is an import alias of a target name (then the text at that position is the alias, not the name).
    """
    alias_of = {local: orig for local, (_, orig) in names.imported.items()
                if orig in targets and local != orig and orig not in MODULE_NAMES}
    out = _CharColumns(source)
    call_funcs = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            call_funcs.add(id(node.func))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out[('def', node.lineno, node.name)] = ('def', 'local', None)          # resolved by line + name below
        elif isinstance(node, ast.arg):
            out[(node.lineno, node.col_offset)] = ('param', '-', None)
        elif isinstance(node, ast.keyword) and node.arg is not None:
            out[(node.lineno, node.col_offset)] = ('kwarg', '-', None)
        elif isinstance(node, ast.Name):
            kind = 'call' if id(node) in call_funcs else ('assign' if isinstance(node.ctx, ast.Store) else 'ref')
            out[(node.lineno, node.col_offset)] = (kind, names.binding_of_name(node.id), alias_of.get(node.id))
        elif isinstance(node, ast.Attribute):
            kind = 'call' if id(node) in call_funcs else ('assign' if isinstance(node.ctx, ast.Store) else 'ref')
            out[(node.end_lineno, node.end_col_offset - len(node.attr))] = (
                kind, names.binding_of_receiver(node.value), None)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            for line in range(node.lineno, node.end_lineno + 1):                   # a string literal's extent
                out.setdefault(('str-span', line), []).append((       # bytes; a line with a string and a
                    node.col_offset if line == node.lineno else 0,          # non-ASCII character before it may
                    node.end_col_offset if line == node.end_lineno else 10 ** 9))   # be off by a few columns
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for line in range(node.lineno, node.end_lineno + 1):
                out[('import-line', line)] = True
            if isinstance(node, ast.ImportFrom):
                for a in node.names:                                # `from ...backend import apply`: the module
                    is_module = a.name in MODULE_NAMES and (node.module or '').split('.')[-1] in ('backend', 't3toolbox')
                    out[(a.lineno, a.col_offset)] = ('import-module' if is_module else 'import',
                                                     'lib:' + (node.module or '.'), None)
    return out


def guess_kind(line, end_col):
    """The kind of a name outside parsed code, from the text that follows it."""
    rest = line[end_col:].lstrip()
    if rest.startswith('('):
        return 'call'
    if rest.startswith('=') and not rest.startswith('=='):
        return 'kwarg'
    return 'ref'


def lookup(roles, line, col, ident, text_line):
    """(kind, binding, original) of the identifier at (line, col), given the roles of classify_tree."""
    if (line, col) in roles:
        return roles[(line, col)]
    if ('def', line, ident) in roles and re.search(r'\b(def|class)\s+%s\b' % re.escape(ident), text_line):
        return roles[('def', line, ident)]
    if ('import-line', line) in roles:
        return ('import-module', '-', None)
    if any(a <= col < b for a, b in roles.get(('str-span', line), ())):
        return ('str', '-', None)
    return None


# ----------------------------------------------------------------------------------------- scanning

class Scanner(object):
    def __init__(self, targets):
        self.targets = targets
        self.found = []
        self.star_globs = []
        self.notes = []

    def add(self, name, path, line, context, kind, binding='-', alias=False, text=''):
        self.found.append(Occurrence(name, path, line, context, kind, binding, alias, text.strip()))

    # -- prose: whole identifiers, brace patterns, star patterns
    def prose(self, path, first_line, text, context):
        for k, line in enumerate(text.split('\n')):
            braces = [m.span() for m in BRACE.finditer(line)]       # the pieces of a brace pattern are not names
            for m in IDENT.finditer(line):
                if m.group() in self.targets and not any(a <= m.start() < b for a, b in braces):
                    self.add(m.group(), path, first_line + k, context, guess_kind(line, m.end()), text=line)
            for m in BRACE.finditer(line):
                for expanded in _expand_braces(m.group()):
                    if expanded in self.targets and expanded != m.group():
                        self.add(expanded, path, first_line + k, context, 'glob', text=line)
            for m in STAR.finditer(line):
                pattern = m.group()
                if pattern.startswith('**') or re.fullmatch(r'\*[A-Za-z0-9_]+\*', pattern):
                    continue                                        # markdown emphasis, not a pattern
                if re.search(r'[A-Za-z0-9]{3}', pattern) and '_' in pattern:
                    hits = [n for n in self.targets if fnmatch.fnmatchcase(n, pattern)]
                    if hits:
                        self.star_globs.append((pattern, path, first_line + k, len(hits)))

    # -- a text that may hold doctest examples (a docstring's source text, or a whole page)
    def with_doctests(self, path, first_line, text, prose_context, names):
        lines = text.split('\n')
        in_example = {}                                         # 0-based line -> roles dict or None (unparsed)
        try:
            examples = doctest.DocTestParser().get_examples(text)
        except ValueError:                                      # inconsistent indentation: fall back to the prompts
            examples = None
            for k, line in enumerate(lines):
                if re.match(r'\s*(>>>|\.\.\.)( |$)', line):
                    in_example[k] = (None, 0)
            self.notes.append('%s:%d: doctest examples not parsable as a whole; kinds guessed from the text'
                              % (path, first_line))
        if examples:
            local = names.copy()
            trees = []
            for ex in examples:
                try:
                    tree = ast.parse(ex.source)
                    local.read(tree)
                except SyntaxError:
                    tree = None
                trees.append(tree)
            for ex, tree in zip(examples, trees):
                n_lines = ex.source.rstrip('\n').count('\n') + 1
                roles = classify_tree(tree, local, self.targets, ex.source) if tree is not None else None
                for j in range(n_lines):
                    in_example[ex.lineno + j] = (roles, j + 1, ex.indent + 4)
        prose_lines = []
        for k, line in enumerate(lines):
            if k not in in_example:
                prose_lines.append(line)
                continue
            prose_lines.append('')
            entry = in_example[k]
            roles = entry[0]
            for m in IDENT.finditer(line):
                ident = m.group()
                role = None
                if roles is not None:
                    role = lookup(roles, entry[1], m.start() - entry[2], ident, line)
                if role is not None and role[2] is not None:                    # an import alias of a target
                    self.add(role[2], path, first_line + k, 'doctest', role[0], role[1], True, line)
                elif ident in self.targets:
                    if role is None:
                        role = (guess_kind(line, m.end()), '-', None)
                    self.add(ident, path, first_line + k, 'doctest', role[0], role[1], False, line)
        self.prose(path, first_line, '\n'.join(prose_lines), prose_context)

    # -- a Python file
    def python_file(self, path, source):
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            self.notes.append('%s: not parsable as Python (%s); scanned as text' % (path, e))
            self.prose(path, 1, source, 'page')
            return
        names = FileNames()
        names.read(tree)
        roles = classify_tree(tree, names, self.targets, source)
        docstrings, all_entries = set(), set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
                first = node.body[0]
                if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                        and isinstance(first.value.value, str):
                    docstrings.add((first.value.lineno, first.value.col_offset))
            if isinstance(node, (ast.Assign, ast.AugAssign)):
                target = node.targets[0] if isinstance(node, ast.Assign) else node.target
                if getattr(target, 'id', None) == '__all__':
                    for c in ast.walk(node.value):
                        if isinstance(c, ast.Constant) and isinstance(c.value, str):
                            all_entries.add((c.lineno, c.col_offset))
        src_lines = source.split('\n')
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            row, col = tok.start
            if tok.type == tokenize.NAME:
                role = lookup(roles, row, col, tok.string, src_lines[row - 1])
                if role is not None and role[2] is not None:
                    self.add(role[2], path, row, 'code', role[0], role[1], True, src_lines[row - 1])
                elif tok.string in self.targets:
                    if role is None:
                        role = ('ref', '-', None)
                    self.add(tok.string, path, row, 'code', role[0], role[1], False, src_lines[row - 1])
            elif tok.type == tokenize.COMMENT:
                self.prose(path, row, tok.string, 'comment')
            elif tok.type == tokenize.STRING:
                if (row, col) in docstrings:
                    self.with_doctests(path, row, tok.string, 'docstring', names)
                elif (row, col) in all_entries:
                    for m in IDENT.finditer(tok.string):
                        if m.group() in self.targets:
                            self.add(m.group(), path, row, 'string', 'all', text=src_lines[row - 1])
                else:
                    self._string(path, row, tok.string)
            elif tok.type == getattr(tokenize, 'FSTRING_MIDDLE', -1):
                self._string(path, row, tok.string)

    def _string(self, path, first_line, text):
        for k, line in enumerate(text.split('\n')):
            for m in IDENT.finditer(line):
                if m.group() in self.targets:
                    self.add(m.group(), path, first_line + k, 'string', guess_kind(line, m.end()), text=line)

    # -- a page: doctest examples, fenced code blocks without prompts, prose
    def text_file(self, path, source):
        lines = source.split('\n')
        fenced, inside, has_prompt, start = set(), False, False, 0
        for k, line in enumerate(lines):
            if line.lstrip().startswith('```'):
                if inside and not has_prompt:
                    fenced.update(range(start + 1, k))
                inside, has_prompt, start = (not inside), False, k
            elif inside and line.lstrip().startswith('>>>'):
                has_prompt = True
        masked = ['' if k in fenced else line for k, line in enumerate(lines)]
        self.with_doctests(path, 1, '\n'.join(masked), 'page', FileNames())
        for k in sorted(fenced):
            for m in IDENT.finditer(lines[k]):
                if m.group() in self.targets:
                    self.add(m.group(), path, k + 1, 'codeblock', guess_kind(lines[k], m.end()), text=lines[k])


def _expand_braces(pattern):
    m = re.search(r'\{([^{}]*)\}', pattern)
    if not m:
        return [pattern]
    out = []
    for option in m.group(1).split(','):
        out.extend(_expand_braces(pattern[:m.start()] + option + pattern[m.end():]))
    return out


def searched_files():
    for root in ROOTS:
        full = os.path.join(REPO, root)
        if os.path.isfile(full):
            yield root
            continue
        for dirpath, dirnames, filenames in os.walk(full):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
            for name in sorted(filenames):
                rel = os.path.relpath(os.path.join(dirpath, name), REPO)
                if name.endswith(SUFFIXES) and rel not in EXCLUDED:
                    yield rel


def scan(targets, paths=None):
    scanner = Scanner(targets)
    n_files = 0
    for path in (searched_files() if paths is None else paths):
        with open(os.path.join(REPO, path), encoding='utf-8', errors='replace') as f:
            source = f.read()
        n_files += 1
        if path.endswith('.py'):
            scanner.python_file(path, source)
        else:
            scanner.text_file(path, source)
    return scanner, n_files


# ----------------------------------------------------------------------------------------- reports

COLUMNS = (                                              # the columns of the summary tables
    ('def', lambda o: o.context == 'code' and o.kind in ('def',)),
    ('import', lambda o: o.context == 'code' and o.kind in ('import', 'import-module')),
    ('call', lambda o: o.context == 'code' and o.kind == 'call'),
    ('kw', lambda o: o.context == 'code' and o.kind in ('kwarg', 'param')),
    ('ref', lambda o: o.context == 'code' and o.kind in ('ref', 'assign')),
    ('dt-call', lambda o: o.context == 'doctest' and o.kind == 'call'),
    ('dt-other', lambda o: o.context == 'doctest' and o.kind != 'call'),
    ('string', lambda o: o.context == 'string'),
    ('text', lambda o: o.context in ('comment', 'docstring', 'page', 'codeblock')),
)


def _row(occurrences):
    cells = [sum(1 for o in occurrences if test(o)) for _, test in COLUMNS]
    return cells + [len(occurrences)]


def _table(header, rows):
    out = ['| ' + ' | '.join(header) + ' |', '| ' + ' | '.join(['---'] * len(header)) + ' |']
    out += ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows]
    return '\n'.join(out)


def area_of(path):
    if path.startswith('t3toolbox/backend/'):
        return 't3toolbox/backend'
    return path.split('/')[0]


def summary(scanner, n_files, names, group_of, detail):
    found = scanner.found
    by_name = collections.defaultdict(list)
    for o in found:
        by_name[o.name].append(o)
    head = [c for c, _ in COLUMNS] + ['total']
    out = []
    out.append('Files searched: %d. Occurrences: %d, of %d names (%d of the frozen names occur nowhere).' % (
        n_files, len(found), len(by_name), len(group_of) - len(by_name)))
    out.append('')
    out.append('Columns: `def` definitions; `import`; `call` call sites in code; `kw` keyword arguments and '
               'parameters; `ref` other uses in code (tables, arguments, assignments); `dt-call`, `dt-other` the '
               'same inside doctest examples (docstrings and pages; `dt-other` includes names inside the string literals '
               'of an example); `string` in strings that are not docstrings '
               '(`__all__` entries, `getattr` names, labels, scripts in strings); `text` comments, docstring '
               'prose, page prose, fenced code without prompts, brace patterns.')

    out.append('\n### By group and area\n')
    areas = ['t3toolbox/backend', 't3toolbox', 'tests', 'docs', 'examples', 'CLAUDE.md', 'README.md']
    rows = []
    for group in ('sampling', 'method', 'factory', 'other'):
        for area in areas:
            sel = [o for o in found if group_of[o.name] == group and area_of(o.file) == area]
            if sel:
                rows.append([group, area] + _row(sel))
        rows.append(['**' + group + '**', '**all**'] + _row([o for o in found if group_of[o.name] == group]))
    out.append(_table(['group', 'area'] + head, rows))

    for group, title in (('sampling', 'The 166 sampling names'), ('method', 'The frontend sampling methods'),
                         ('factory', 'The six model factories'), ('other', 'The other names')):
        out.append('\n### %s, by name\n' % title)
        if group == 'method':
            out.append('One row per distinct name; the classes that define it are in the second column '
                       '(T = `TuckerTensorTrain`, G = `T3Tangent`, UT = `UniformTuckerTensorTrain`, '
                       'UG = `UT3Tangent`). The receiver of a method call is not typed, so a row counts every '
                       'attribute of that name; for `probe`, `apply`, `entries` the `import` and `text` columns '
                       'are mostly the backend modules of the same name and English words (marked \\*).\n')
        short = {'TuckerTensorTrain': 'T', 'T3Tangent': 'G', 'UniformTuckerTensorTrain': 'UT', 'UT3Tangent': 'UG'}
        rows = []
        ordered = [n for n in group_of if group_of[n] == group]
        if group == 'sampling':
            ordered.sort(key=lambda n: (SAMPLING_MODULES.index(detail[n][0]), names['sampling'][detail[n][0]].index(n)))
        for n in ordered:
            where = ', '.join(short.get(w, w) for w in detail.get(n, []))
            rows.append(['`%s`%s' % (n, ' (\\*)' if n in NOISY else ''), where] + _row(by_name.get(n, [])))
        out.append(_table(['name', 'module' if group == 'sampling' else 'defined in'] + head, rows))

    out.append('\n### By file\n')
    out.append('Every file with at least one occurrence. `s`, `m`, `f`, `o`: occurrences of the sampling names, '
               'the method names, the factories, the other names; then the same columns as above over all four.\n')
    by_file = collections.defaultdict(list)
    for o in found:
        by_file[o.file].append(o)
    rows = []
    for path in sorted(by_file, key=lambda p: (areas.index(area_of(p)), p)):
        sel = by_file[path]
        groups = [sum(1 for o in sel if group_of[o.name] == g) for g in ('sampling', 'method', 'factory', 'other')]
        rows.append(['`%s`' % path] + groups + _row(sel))
    out.append(_table(['file', 's', 'm', 'f', 'o'] + head, rows))

    out.append('\n### Names reached through an import alias\n')
    aliased = collections.Counter((o.name, o.file) for o in found if o.alias)
    out.append(_table(['name', 'file', 'occurrences'], [['`%s`' % n, '`%s`' % f, c]
                                                        for (n, f), c in sorted(aliased.items())])
               if aliased else 'None: no file imports one of the names under another name.')

    out.append('\n### Star patterns in prose that match at least one name\n')
    out.append('Not counted for any name above. Pattern, the number of names it matches, and how often it is '
               'written, by area.\n')
    stars = collections.defaultdict(collections.Counter)
    hits = {}
    for pattern, path, _, n_hits in scanner.star_globs:
        stars[pattern][area_of(path)] += 1
        hits[pattern] = n_hits
    rows = [['`%s`' % p, hits[p], sum(c.values()), ', '.join('%s %d' % kv for kv in sorted(c.items()))]
            for p, c in sorted(stars.items(), key=lambda kv: -sum(kv[1].values()))]
    out.append(_table(['pattern', 'names matched', 'written', 'where'], rows))

    out.append('\n### Not in the inventory: the two files that name the removed names on purpose\n')
    out.append('`CHANGELOG.md` and `docs/release_notes.md` record the history of the releases, so they keep the old '
               'names, and the final search of P20 leaves them out (PART1_PLAN.md section 10). They are not counted '
               'anywhere above. Their occurrences, for the record:\n')
    excluded, _ = scan(scanner.targets, EXCLUDED)
    rows = []
    for path in EXCLUDED:
        for group in ('sampling', 'method', 'factory', 'other'):
            sel = [o for o in excluded.found if o.file == path and group_of[o.name] == group]
            rows.append(['`%s`' % path, group, len(set(o.name for o in sel)), len(set(o.line for o in sel)), len(sel)])
        sel = [o for o in excluded.found if o.file == path]
        rows.append(['`%s`' % path, '**all**', len(set(o.name for o in sel)), len(set(o.line for o in sel)), len(sel)])
    out.append(_table(['file', 'group', 'distinct names', 'lines', 'occurrences'], rows))

    if scanner.notes:
        out.append('\n### Notes from the scan\n')
        out += ['- ' + n for n in scanner.notes]
    return '\n'.join(out)


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--freeze', action='store_true', help='(re)write inventory_names.json from the tree (P0 only)')
    p.add_argument('--lines', action='store_true', help='list the occurrences, one per line')
    p.add_argument('--count', action='store_true', help='count the occurrences that pass the filters')
    p.add_argument('--globs', action='store_true', help='list the star patterns with their places')
    p.add_argument('--by', default='name', help='for --count: a comma list of name,file,context,kind,binding,area')
    p.add_argument('--name', action='append', help='filter: this name (repeatable; fnmatch pattern allowed)')
    p.add_argument('--group', action='append', help='filter: sampling | method | factory | other')
    p.add_argument('--module', action='append', help='filter: sampling names of this module (e.g. probing)')
    p.add_argument('--kind', action='append', help='filter: def param import import-module call kwarg ref assign str all glob')
    p.add_argument('--context', action='append', help='filter: code doctest string comment docstring page codeblock')
    p.add_argument('--file', action='append', help='filter: path, fnmatch pattern (e.g. "tests/*")')
    p.add_argument('--binding', action='append', help='filter: fnmatch pattern on the binding (e.g. "lib:*", local, obj)')
    args = p.parse_args()

    if args.freeze:
        names = freeze_names()
        print('wrote %s: %d sampling names, %d methods (%d distinct), %d factories, %d other' % (
            NAMES_FILE, sum(len(v) for v in names['sampling'].values()), sum(len(v) for v in names['method'].values()),
            len(set(n for v in names['method'].values() for n in v)), len(names['factory']), len(names['other'])))
        return
    names, group_of, detail = load_names()
    scanner, n_files = scan(set(group_of))

    def passes(o):
        def any_match(value, patterns):
            return patterns is None or any(fnmatch.fnmatchcase(value, pat) for pat in patterns)
        return (any_match(o.name, args.name) and any_match(group_of[o.name], args.group)
                and any_match(o.kind, args.kind) and any_match(o.context, args.context)
                and any_match(o.file, args.file) and any_match(o.binding, args.binding)
                and (args.module is None or (group_of[o.name] == 'sampling'
                                             and any(m in detail[o.name] for m in args.module))))

    if args.lines:
        for o in scanner.found:
            if passes(o):
                print('%s:%d: %-8s %-6s %-22s %s%s | %s' % (o.file, o.line, o.context, o.kind, o.binding, o.name,
                                                           ' (alias)' if o.alias else '', o.text[:110]))
    elif args.count:
        keys = args.by.split(',')
        counter = collections.Counter()
        for o in scanner.found:
            if passes(o):
                counter[tuple(area_of(o.file) if k == 'area' else getattr(o, k) for k in keys)] += 1
        for key, c in sorted(counter.items()):
            print('%6d  %s' % (c, '  '.join(key)))
        print('%6d  total' % sum(counter.values()))
    elif args.globs:
        for pattern, path, line, n_hits in scanner.star_globs:
            print('%s:%d: %s (matches %d names)' % (path, line, pattern, n_hits))
    else:
        print(summary(scanner, n_files, names, group_of, detail))


if __name__ == '__main__':
    sys.exit(main())
