"""Diff keyword defaults (a) between a frontend method and the backend function it delegates to
(any call inside the body whose callee name matches a backend def), and (b) between ragged/uniform
twin pairs by name prefix (t3_/ut3_, fv_/ufv_, tv_/utv_, X/UniformX, UT3*/T3*)."""
import ast, os, collections

ROOT = '/home/nick/repos/T3Toolbox/t3toolbox'


def defaults_of(fn):
    a = fn.args
    pos = a.posonlyargs + a.args
    d = {}
    for p, dv in zip(pos[len(pos) - len(a.defaults):], a.defaults):
        d[p.arg] = ast.unparse(dv)
    for p, dv in zip(a.kwonlyargs, a.kw_defaults):
        if dv is not None:
            d[p.arg] = ast.unparse(dv)
    params = [p.arg for p in pos + a.kwonlyargs if p.arg not in ('self', 'cls')]
    return params, d


funcs = {}   # qualname -> (rel, lineno, params, defaults, fn)
by_short = collections.defaultdict(list)
for dp, dn, fns in os.walk(ROOT):
    for f in sorted(fns):
        if not f.endswith('.py'):
            continue
        path = os.path.join(dp, f)
        rel = os.path.relpath(path, '/home/nick/repos/T3Toolbox')
        tree = ast.parse(open(path).read())
        def walk(node, prefix):
            for ch in ast.iter_child_nodes(node):
                if isinstance(ch, ast.FunctionDef):
                    q = rel + '::' + prefix + ch.name
                    params, d = defaults_of(ch)
                    funcs[q] = (rel, ch.lineno, params, d, ch)
                    by_short[ch.name].append(q)
                    by_short[prefix + ch.name].append(q)
                elif isinstance(ch, ast.ClassDef):
                    walk(ch, prefix + ch.name + '.')
                else:
                    walk(ch, prefix)
        walk(tree, '')

print('=== (a) frontend method -> backend callee default diffs')
for q, (rel, ln, params, d, fn) in funcs.items():
    if rel.startswith('t3toolbox/backend/'):
        continue
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            callee = None
            if isinstance(node.func, ast.Attribute):
                callee = node.func.attr
            elif isinstance(node.func, ast.Name):
                callee = node.func.id
            if callee is None:
                continue
            targets = [t for t in by_short.get(callee, []) if funcs[t][0].startswith('t3toolbox/backend/')]
            if len(targets) != 1:
                continue
            t = targets[0]
            trel, tln, tparams, td, _ = funcs[t]
            # which frontend params are forwarded by keyword under the same name, or positionally?
            passed_kw = {k.arg: ast.unparse(k.value) for k in node.keywords if k.arg}
            for p, dv in d.items():
                if p in td and td[p] != dv:
                    # only report if the frontend param is actually passed to this callee
                    if p in passed_kw or any(isinstance(a_, ast.Name) and a_.id == p for a_ in node.args):
                        print(f'{rel}:{ln} {q.split("::")[1]}({p}={dv})  ->  {trel}:{tln} {callee}({p}={td[p]})')
            # frontend params with defaults that the callee accepts but the frontend does NOT pass at all
            for p, dv in d.items():
                if p in tparams and p not in passed_kw and not any(isinstance(a_, ast.Name) and a_.id == p for a_ in node.args):
                    # check whether p is used anywhere in the body at all
                    used = any(isinstance(n, ast.Name) and n.id == p for n in ast.walk(fn))
                    if not used:
                        print(f'NOT-FORWARDED {rel}:{ln} {q.split("::")[1]}({p}) accepted by {callee} but never used')

print()
print('=== (b) ragged/uniform twin default diffs')
def twin_name(n):
    for a, b in (('t3_', 'ut3_'), ('fv_', 'ufv_'), ('tv_', 'utv_'), ('t3svd', 'ut3svd')):
        if n.startswith(a):
            return b + n[len(a):]
    return None

seen = set()
for q, (rel, ln, params, d, fn) in funcs.items():
    short = q.split('::')[1]
    cls = ''
    if '.' in short:
        cls, meth = short.rsplit('.', 1)
    else:
        meth = short
    cands = []
    if cls:
        for uc in ('Uniform' + cls, cls.replace('T3', 'UT3'), cls.replace('Tucker', 'UniformTucker')):
            cands += by_short.get(uc + '.' + meth, [])
    else:
        tn = twin_name(meth)
        if tn:
            cands += by_short.get(tn, [])
    for t in cands:
        if t == q or (q, t) in seen:
            continue
        seen.add((q, t))
        trel, tln, tparams, td, _ = funcs[t]
        diffs = [(p, d[p], td[p]) for p in d if p in td and td[p] != d[p]]
        only_r = [p for p in d if p not in tparams]
        only_u = [p for p in td if p not in params]
        if diffs or only_r or only_u:
            print(f'{rel}:{ln} {short}  <->  {trel}:{tln} {t.split("::")[1]}')
            for p, a, b in diffs:
                print(f'    DEFAULT {p}: ragged={a} uniform={b}')
            if only_r:
                print(f'    ragged-only defaulted params: {only_r}')
            if only_u:
                print(f'    uniform-only defaulted params: {only_u}')
