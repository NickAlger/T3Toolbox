"""AST scan over every module in t3toolbox/: per function report
  (a) parameters never referenced in the body,
  (b) parameters assigned before first read (shadowed),
  (c) **kwargs accepted but never forwarded / read,
  (d) wrapper functions (single delegate call) whose accepted keyword params are not forwarded.
"""
import ast, os, sys

ROOT = '/home/nick/repos/T3Toolbox/t3toolbox'


def iter_modules():
    for dp, dn, fn in os.walk(ROOT):
        for f in sorted(fn):
            if f.endswith('.py'):
                yield os.path.join(dp, f)


class NameUse(ast.NodeVisitor):
    """Collect, in source order, (name, kind) events for Name loads/stores, ignoring nested defs."""
    def __init__(self):
        self.events = []
    def visit_Name(self, node):
        self.events.append((node.id, 'store' if isinstance(node.ctx, (ast.Store,)) else 'load', node.lineno))
    def visit_Assign(self, node):
        self.visit(node.value)
        for t in node.targets:
            self.visit(t)
    def visit_AnnAssign(self, node):
        if node.value is not None:
            self.visit(node.value)
        self.visit(node.target)
    def visit_AugAssign(self, node):
        self.visit(node.value)
        # aug-assign reads the target first
        self.events.append((node.target.id if isinstance(node.target, ast.Name) else '?', 'load', node.lineno))
        self.visit(node.target)
    def visit_For(self, node):
        self.visit(node.iter)
        self.visit(node.target)
        for s in node.body + node.orelse:
            self.visit(s)
    def visit_NamedExpr(self, node):
        self.visit(node.value)
        self.visit(node.target)
    def visit_comprehension(self, node):
        self.visit(node.iter)
        self.visit(node.target)
        for i in node.ifs:
            self.visit(i)
    def visit_FunctionDef(self, node):
        # nested function: its body may read closure vars -> treat its free names as loads
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                self.events.append((n.id, 'load', n.lineno))
    visit_AsyncFunctionDef = visit_FunctionDef
    def visit_Lambda(self, node):
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                self.events.append((n.id, 'load', n.lineno))


def params_of(fn):
    a = fn.args
    names = [p.arg for p in a.posonlyargs + a.args + a.kwonlyargs]
    vararg = a.vararg.arg if a.vararg else None
    kwarg = a.kwarg.arg if a.kwarg else None
    return names, vararg, kwarg


def analyze(fn, path, qual):
    names, vararg, kwarg = params_of(fn)
    # skip abstract bodies (docstring + raise)
    nondoc = [s for s in fn.body if not (isinstance(s, ast.Expr) and isinstance(getattr(s, 'value', None), ast.Constant))]
    if len(nondoc) == 1 and isinstance(nondoc[0], (ast.Raise, ast.Pass)):
        return []
    if len(nondoc) == 1 and isinstance(nondoc[0], ast.Expr) and isinstance(nondoc[0].value, ast.Constant) and nondoc[0].value.value is Ellipsis:
        return []
    nu = NameUse()
    for stmt in fn.body:
        nu.visit(stmt)
    events = nu.events
    out = []
    first = {}
    for name, kind, ln in events:
        first.setdefault(name, (kind, ln))
    loads = {n for n, k, _ in events if k == 'load'}
    for p in names:
        if p in ('self', 'cls'):
            continue
        if p.startswith('_'):
            continue
        if p not in loads:
            out.append(f'UNUSED   {path}:{fn.lineno} {qual}({p})')
        elif first.get(p, ('load', 0))[0] == 'store':
            out.append(f'SHADOWED {path}:{fn.lineno} {qual}({p}) first assigned at line {first[p][1]}')
    if kwarg and kwarg not in loads:
        out.append(f'KWARGS-DROPPED {path}:{fn.lineno} {qual}(**{kwarg})')
    if vararg and vararg not in loads:
        out.append(f'VARARG-DROPPED {path}:{fn.lineno} {qual}(*{vararg})')
    # wrapper check: body is (docstring +) a single return of a call, or single expr call
    body = [s for s in fn.body if not (isinstance(s, ast.Expr) and isinstance(getattr(s, 'value', None), ast.Constant))]
    if len(body) == 1 and isinstance(body[0], ast.Return) and isinstance(body[0].value, ast.Call):
        call = body[0].value
        passed = set()
        for a_ in call.args:
            for n in ast.walk(a_):
                if isinstance(n, ast.Name):
                    passed.add(n.id)
        for k in call.keywords:
            for n in ast.walk(k.value):
                if isinstance(n, ast.Name):
                    passed.add(n.id)
        missing = [p for p in names if p not in ('self', 'cls') and p not in passed]
        if missing:
            out.append(f'WRAPPER-DROP {path}:{fn.lineno} {qual} does not forward {missing}')
    return out


def main():
    total = 0
    for path in iter_modules():
        src = open(path).read()
        tree = ast.parse(src)
        rel = os.path.relpath(path, '/home/nick/repos/T3Toolbox')
        stack = []
        def walk(node, prefix):
            for ch in ast.iter_child_nodes(node):
                if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    q = prefix + ch.name
                    for line in analyze(ch, rel, q):
                        print(line)
                    walk(ch, q + '.')
                elif isinstance(ch, ast.ClassDef):
                    walk(ch, prefix + ch.name + '.')
                else:
                    walk(ch, prefix)
        walk(tree, '')

main()
