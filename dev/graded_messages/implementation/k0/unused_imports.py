"""Imports of a file that no code of it uses now but that HEAD's version used: python unused_imports.py FILE ..."""
import ast, subprocess, sys
def unused(src):
    tree = ast.parse(src)
    names = {}
    for n in tree.body:
        if isinstance(n, ast.Import):
            for a in n.names: names[a.asname or a.name.split('.')[0]] = n.lineno
        elif isinstance(n, ast.ImportFrom):
            for a in n.names: names[a.asname or a.name] = n.lineno
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    return {k: v for k, v in names.items() if k not in used and k != '*'}
for path in sys.argv[1:]:
    now = unused(open(path).read())
    try:
        old = unused(subprocess.run(['git', 'show', 'HEAD:' + path], capture_output=True, text=True, check=True).stdout)
    except subprocess.CalledProcessError:
        old = {}
    new = sorted(set(now) - set(old))
    if new: print(path, 'newly unused:', new)
