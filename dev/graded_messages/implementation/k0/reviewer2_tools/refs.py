"""Every occurrence of a name K0 deletes, in a tree, classified. Shares no code with k0_inventory.py.

    python refs.py ROOT [NAMES_JSON] [--docs] [--quiet-prose]

NAMES_JSON defaults to the frozen k0_names.json ({'deleted': {module: [names]}}); the module
t3toolbox.backend.sampling_derivatives itself is a name too ('sampling_derivatives').

Files: every .py under ROOT/t3toolbox, ROOT/tests, ROOT/examples; with --docs also the ```python fences of
ROOT/docs/**/*.md and ROOT/docs/*.rst (their >>> lines are page doctests, the rest of a page is prose).

Each occurrence is classified by the tokenizer: CODE (a NAME token), DOCTEST (inside a docstring, on a >>> or ...
line), STRING (another string literal, or a non-doctest docstring line: DOCSTRING), COMMENT. A CODE hit is then
resolved by ast:
  RESOLVED      an attribute of an alias of the owning module, a from-import from it, or a bare name in the owning
                module itself: a reference to the deleted object. Must be 0 after K0.
  OTHER-MODULE  an attribute of an alias of another module (a kept name of the same spelling there).
  LOCAL         a bare name bound in this file other than by importing it from the owner (a test-local helper of
                the same spelling, a parameter, a keyword argument's name).
  UNRESOLVED    a bare name with no binding in this file, or an attribute of something not known to be a module:
                read by hand.
Run-time names: every getattr/hasattr/setattr whose name argument is not a literal, and every string template (a
%-format, a .format or an f-string) whose pattern, with each placeholder read as [A-Za-z0-9_]*, matches a deleted
name in full (TEMPLATE hits).
"""
import ast
import collections
import io
import json
import os
import re
import sys
import tokenize

DEFAULT_NAMES = '/home/nick/repos/T3Toolbox/dev/graded_messages/implementation/k0/k0_names.json'
IDENT = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')
SD_MODULE = 't3toolbox.backend.sampling_derivatives'


def load_names(path):
    deleted = json.load(open(path))['deleted']
    owners = collections.defaultdict(set)
    for mod, names in deleted.items():
        for n in names:
            owners[n].add(mod)
    owners['sampling_derivatives'].add(SD_MODULE)          # the module, as a name
    return owners


def files(root, docs):
    out = []
    for top in ('t3toolbox', 'tests', 'examples'):
        for d, _, fs in os.walk(os.path.join(root, top)):
            if '__pycache__' not in d:
                out += [os.path.join(d, f) for f in fs if f.endswith('.py')]
    pages = []
    if docs:
        for d, _, fs in os.walk(os.path.join(root, 'docs')):
            if '_build' in d or 'autoapi' in d:
                continue
            pages += [os.path.join(d, f) for f in fs if f.endswith(('.md', '.rst'))]
    return sorted(out), sorted(pages)


def module_of(root, path):
    rel = os.path.relpath(path, root)
    return rel[:-3].replace(os.sep, '.') if rel.startswith('t3toolbox') else None


def docstring_spans(tree):
    spans = []
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            b = n.body
            if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant) \
                    and isinstance(b[0].value.value, str):
                spans.append((b[0].lineno, b[0].end_lineno))
    return spans


def aliases(tree):
    """alias -> module, for `import a.b.c as x`, `from a.b import c [as x]` (c a module of the package), and the
    dotted chain of a bare `import a.b.c`."""
    out = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.asname:
                    out[a.asname] = a.name
                else:
                    out[a.name] = a.name                 # used as a dotted chain
        elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            for a in n.names:
                out[a.asname or a.name] = n.module + '.' + a.name   # a module, if it is one
    return out


def dotted(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return '.'.join(reversed(parts))
    return None


def bound_names(tree):
    """Names bound in the file other than by a from-import: defs, classes, args, assignment targets, loop and
    with targets, except handlers, comprehension targets, plain imports' aliases."""
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, ast.arg):
            out.add(n.arg)
        elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            out.add(n.id)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            out.add(n.name)
        elif isinstance(n, ast.Import):
            for a in n.names:
                out.add(a.asname or a.name.split('.')[0])
    return out


def template_pattern(node):
    """A regex for a string template, or None."""
    if isinstance(node, ast.JoinedStr):
        parts = []
        for v in node.values:
            parts.append(re.escape(v.value) if isinstance(v, ast.Constant) else '[A-Za-z0-9_]*')
        return ''.join(parts)
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        s = node.value
        if re.search(r'%(\([^)]*\))?[sdr]|\{[^{}]*\}', s):
            pieces = re.split(r'%(?:\([^)]*\))?[sdr]|\{[^{}]*\}', s)
            return '[A-Za-z0-9_]*'.join(re.escape(p) for p in pieces)
    return None


def scan_py(root, path, owners, hits, dyn):
    src = open(path).read()
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        hits['PARSE-ERROR'].append((path, e.lineno, str(e)))
        return
    mod = module_of(root, path)
    lines = src.splitlines()
    dspans = docstring_spans(tree)
    in_doc = lambda ln: any(a <= ln <= b for a, b in dspans)
    al = aliases(tree)
    bound = bound_names(tree)
    from_imported = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module:
            for a in n.names:
                from_imported[a.asname or a.name] = (n.module, a.name, n.lineno)
    # code nodes by (line, col)
    attr_at, name_at = {}, {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute):
            attr_at.setdefault(n.end_lineno, []).append(n)
        elif isinstance(n, ast.Name):
            name_at[(n.lineno, n.col_offset)] = n
    # tokens
    toks = tokenize.generate_tokens(io.StringIO(src).readline)
    for tok in toks:
        ttype, text, (ln, col), _, _ = tok
        if ttype == tokenize.NAME and text in owners:
            cls = resolve(text, ln, col, mod, owners, al, bound, from_imported, attr_at, name_at)
            hits[cls].append((path, ln, text, lines[ln - 1].strip()[:110]))
        elif ttype == tokenize.STRING or ttype == getattr(tokenize, 'FSTRING_MIDDLE', -1):
            for k, piece in enumerate(text.split('\n')):
                for ident in IDENT.findall(piece):
                    if ident in owners:
                        line_no = ln + k
                        src_line = lines[line_no - 1].strip()
                        if in_doc(line_no):
                            cls = 'DOCTEST' if src_line.startswith(('>>>', '...')) else 'DOCSTRING'
                        else:
                            cls = 'STRING'
                        hits[cls].append((path, line_no, ident, src_line[:110]))
        elif ttype == tokenize.COMMENT:
            for ident in IDENT.findall(text):
                if ident in owners:
                    hits['COMMENT'].append((path, ln, ident, text.strip()[:110]))
    # run-time names
    owner_mods = {m for ms in owners.values() for m in ms}
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ('getattr', 'hasattr',
                                                                                         'setattr', 'delattr'):
            if len(n.args) >= 2 and not (isinstance(n.args[1], ast.Constant)):
                base = dotted(n.args[0])
                on_owner = base is not None and (al.get(base) in owner_mods or base in owner_mods)
                silent = n.func.id == 'hasattr' or (n.func.id == 'getattr' and len(n.args) == 3)
                key = 'DYNAMIC-ATTR' if (on_owner or silent) else 'dynamic-attr (other objects; counted only)'
                dyn[key].append((path, n.lineno, ast.unparse(n)[:120]))
            elif len(n.args) >= 2 and n.args[1].value in owners:
                dyn['LITERAL-ATTR'].append((path, n.lineno, ast.unparse(n)[:120]))
        pat = template_pattern(n) if isinstance(n, (ast.JoinedStr, ast.Constant)) else None
        if pat:
            tight = re.compile(pat.replace('[A-Za-z0-9_]*', '(probe|apply|entries)'))
            m = sorted(name for name in owners if tight.fullmatch(name))
            if m:
                dyn['TEMPLATE-KIND'].append((path, n.lineno, '%r -> %s' % (pat[:60], m[:6])))
            elif any(re.fullmatch(pat, name) for name in owners):
                dyn['template-broad (counted only)'].append((path, n.lineno, pat[:60]))


def resolve(name, ln, col, mod, owners, al, bound, from_imported, attr_at, name_at):
    own = owners[name]
    for a in attr_at.get(ln, []):
        if a.attr == name and a.end_col_offset == col + len(name):
            base = dotted(a.value)
            if base is None:
                return 'UNRESOLVED'
            if base in al:
                target = al[base]
            elif base.startswith('t3toolbox'):
                target = base
            else:
                return 'UNRESOLVED'
            if target in own or target == SD_MODULE:
                return 'RESOLVED'
            return 'OTHER-MODULE'
    if (ln, col) in name_at:
        if mod in own:
            return 'RESOLVED'
        if name in from_imported:
            m, orig, _ = from_imported[name]
            return 'RESOLVED' if m in own else 'OTHER-MODULE'
        if name in bound:
            return 'LOCAL'
        return 'UNRESOLVED'
    # a NAME token that is neither an Attribute's attr nor a Name node: an import line, a def/class name, an arg, a
    # keyword argument, an import alias
    if name in from_imported and from_imported[name][2] == ln:
        return 'RESOLVED' if from_imported[name][0] in own else 'OTHER-MODULE'
    if name == 'sampling_derivatives':
        return 'RESOLVED'                           # `import ...sampling_derivatives` or `from ... import it`
    if mod in own:
        return 'RESOLVED'                           # its own definition, still present
    return 'LOCAL'                                  # a def/arg/keyword of the same spelling elsewhere


def scan_page(path, owners, hits):
    in_fence = False
    for ln, line in enumerate(open(path, errors='replace'), 1):
        s = line.strip()
        if s.startswith('```'):
            in_fence = not in_fence if not s[3:].strip() or not in_fence else in_fence
            if s.startswith('```python') or s.startswith('```{code') :
                in_fence = True
            continue
        idents = [i for i in IDENT.findall(line) if i in owners]
        if not idents:
            continue
        if s.startswith(('>>>', '...')) and (in_fence or path.endswith('.rst')):
            cls = 'PAGE-DOCTEST'
        elif in_fence:
            cls = 'PAGE-FENCE'
        else:
            cls = 'PAGE-PROSE'
        for i in idents:
            hits[cls].append((path, ln, i, s[:110]))


ORDER = ('PARSE-ERROR', 'RESOLVED', 'UNRESOLVED', 'OTHER-MODULE', 'LOCAL', 'DOCTEST', 'STRING', 'DOCSTRING',
         'COMMENT', 'PAGE-DOCTEST', 'PAGE-FENCE', 'PAGE-PROSE')


def main(argv):
    flags = {a for a in argv if a.startswith('--')}
    args = [a for a in argv if not a.startswith('--')]
    root = os.path.abspath(args[0])
    owners = load_names(args[1] if len(args) > 1 else DEFAULT_NAMES)
    pys, pages = files(root, '--docs' in flags)
    hits, dyn = collections.defaultdict(list), collections.defaultdict(list)
    for p in pys:
        scan_py(root, p, owners, hits, dyn)
    for p in pages:
        scan_page(p, owners, hits)
    sd_file = os.path.join(root, 't3toolbox/backend/sampling_derivatives.py')
    print('root %s: %d .py files, %d pages; %d names; sampling_derivatives.py %s' % (
        root, len(pys), len(pages), len(owners), 'PRESENT' if os.path.exists(sd_file) else 'absent'))
    print('counts:', {k: len(hits[k]) for k in ORDER if hits[k]}, {k: len(v) for k, v in dyn.items()})
    quiet = {'DOCSTRING', 'COMMENT', 'PAGE-PROSE'} if '--quiet-prose' in flags else set()
    for k in ORDER:
        if hits[k] and k not in quiet:
            print('\n== %s (%d)' % (k, len(hits[k])))
            for path, ln, name, line in hits[k]:
                print('  %s:%d  %s  |  %s' % (os.path.relpath(path, root), ln, name, line))
    for k, v in dyn.items():
        print('\n== %s (%d)' % (k, len(v)))
        if 'counted only' in k:
            continue
        for path, ln, text in v:
            print('  %s:%d  %s' % (os.path.relpath(path, root), ln, text))


if __name__ == '__main__':
    main(sys.argv[1:])
