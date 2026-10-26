"""Read-only plan check (v5): (1) the grouped `contract` strings v5 assigns to kernels v4 wrote as '...' einsums
(lift, lift transpose, center factors, T3-T3 tangent zipper + two-hole assembly) -- each checked against np.einsum on
the generic multi-axis point W=(2,3), K=(2,2), C=(2,3), and the supplement it needs (the HISTORICAL value);
(2) the backend import graph at 920d776 plus v5's planned new edges is acyclic."""
import sys, re, ast, pathlib, itertools
sys.path.insert(0, __import__("os").getcwd())
import numpy as np
from t3toolbox.backend.contractions import contract
src = pathlib.Path('tests/test_contractions_interpreter.py').read_text()
H = ast.literal_eval(src[src.index('HISTORICAL = {') + len('HISTORICAL = '): src.index('}', src.index('HISTORICAL = {')) + 1])
shapes = {'W': (2, 3), 'K': (2, 2), 'C': (2, 3)}
rng = np.random.default_rng(0)
STRINGS = {
    # lift / lift transpose (C-only factors; K+C variations)
    'Cux,KCxn->KCun': 'lift Udot = (S^T)^+ V',
    'KCaib,Cbc->KCaic': 'lift Gdot = H x3 T^+',
    'Cux,KCun->KCxn': 'lift^T on the Tucker slot',
    'KCaic,Cbc->KCaib': 'lift^T on a TT slot',
    # frame factors (C only)
    'Caib,Cbc->Caic': 'center H = L T',
    'Caxb,Caub->Cxu': 'S^T = <O, H>',
    # T3-T3 tangent zipper step (carry over Z rank i / theta rank j) and two-hole pieces
    'KCij,Ciak,Cjal->KCkl': 'dotted zipper: carry term',
    'Cij,Ciak,KCjal->KCkl': 'dotted zipper: direction-core term',
    'Czn,KCun->KCzu': 'BUd = B_Z Udot^T',
    'Cazb,KCzu->KCaub': 'G2d = G_Z x2 BUd',
    'KCazb,Czu->KCaub': 'dG term envd x BU',
    # factored dotted TT assembly (c35c / c35d): pair intermediates + the product rule
    'trs,rWCa,sWCi->tWCai': 'assembly pair P (sigma group)',
    'trs,rWKCa,sWCi->tWKCai': 'assembly dotted pair, K on mu',
    'trs,rWCa,sWKCi->tWKCai': 'assembly dotted pair, K on xi',
    'tWKCai,tWCb->KCaib': 'assembly final, K on the pair',
    'tWCai,tWKCb->KCaib': 'assembly final, K on the edge',
    'trs,tWCa,rWCi->sWCai': 'assembly pair Q (tau group)',
    'trs,tWKCa,rWCi->sWKCai': 'assembly dotted pair Q, K on tau~',
    'trs,tWCa,rWKCi->sWKCai': 'assembly dotted pair Q, K on xi',
    'trs,rWCa,tWCi->sWCai': 'assembly pair B (deta group)',
    'trs,rWKCa,tWCi->sWKCai': 'assembly dotted pair B, K on mu',
    'trs,rWCa,tWKCi->sWKCai': 'assembly dotted pair B, K on deta~',
    'sWKCai,sWCb->KCaib': 'assembly final (tau/deta), K on pair',
    'sWCai,sWKCb->KCaib': 'assembly final (tau/deta), K on nu',
    'Cij,KCiak,Cjal->KCkl': 'dotted zipper: G2-dot term',
}
def ops_for(s):
    out = []
    for term in s.split('->')[0].split(','):
        shp = ()
        for ch in term:
            shp += shapes[ch] if ch.isupper() else ((3 if ch in 'trs' else 2),)
        out.append(rng.standard_normal(shp))
    return out
def ref(s, ops):                  # expand group letters into explicit axes, then np.einsum
    letters = iter('ABDEFGHIJLMNOPQRSTUVXYZ'.lower()[::-1] + 'ABDEFGHIJLMNOPQRSTUVXYZ')
    used = set(ch for ch in s if ch.islower())
    pool = [c for c in 'ghjmpqvwyzefcdbaijklmnorstux' if c not in used]
    gmap = {g: ''.join(pool.pop() for _ in shapes[g]) for g in 'WKC'}
    return np.einsum(''.join(gmap.get(ch, ch) for ch in s), *ops)
for s, what in STRINGS.items():
    if what is None:
        continue
    ops = ops_for(s)
    need = None
    try:
        out = contract(s, *ops); need = ''
    except ValueError:
        for g in 'CWK':
            if g in s:
                try:
                    out = contract(s, *ops, **{'len_' + g: len(shapes[g])}); need = g; break
                except ValueError:
                    pass
    ok = need is not None and np.allclose(out, ref(s, ops))
    print(f"  {s:26s} {what:34s} HISTORICAL? {'yes' if s in H else 'NEW'}; supplement {need!r}; == einsum {ok}")

# (2) the import DAG
B = pathlib.Path('t3toolbox/backend')
mods = {p.stem for p in B.glob('*.py') if p.stem != '__init__'}
edges = {m: set() for m in mods}
for m in mods:
    for line in (B / f'{m}.py').read_text().splitlines():
        if line.lstrip().startswith(('>>>', '...', '#')):
            continue   # doctest / comment lines are not module imports
        for hit in re.findall(r'(?:import|from) t3toolbox\.backend\.([a-z_0-9]+)', line) + \
                   re.findall(r'from t3toolbox\.backend import ([a-z_0-9]+)', line):
            if hit in mods and hit != m:
                edges[m].add(hit)
PLANNED = [('fv_operations', 'linalg'), ('sharing', 'fv_operations'), ('tv_operations', 'fv_operations'),
           ('tv_operations', 't3_linalg'), ('t3_linalg', 'fv_conversions'), ('geometry', 't3_linalg'),
           ('geometry', 'ut3_linalg'), ('ufv_operations', 'linalg'), ('ut3_linalg', 'ufv_conversions'),
           ('ut3_sampling', 'ut3_masking'), ('uniform_fitting', 'ufv_masking'),
           ('ufv_operations', 'fv_operations'), ('ufv_operations', 'ufv_conversions'), ('utv_sampling', 'ufv_masking'),
           ('sharing', 'linalg'), ('geometry', 'linalg'), ('uniform_fitting', 'ufv_operations')]
for a, b in PLANNED:
    edges[a].add(b)
def cycle():
    color = {m: 0 for m in mods}; stack = []
    def dfs(u):
        color[u] = 1; stack.append(u)
        for v in edges[u]:
            if color[v] == 1:
                return stack[stack.index(v):] + [v]
            if color[v] == 0:
                c = dfs(v)
                if c: return c
        color[u] = 2; stack.pop(); return None
    for m in sorted(mods):
        if color[m] == 0:
            c = dfs(m)
            if c: return c
    return None
c = cycle()
print('backend import graph + %d planned edges: %s' % (len(PLANNED), 'ACYCLIC' if c is None else 'CYCLE ' + ' -> '.join(c)))
print('optimizers.py imports (must stay T3-free):', sorted(edges['optimizers']))
