"""Read-only plan check (v2): every grouped string the plan's kernels use (c5, c6, c13, c14), whether it is
already a HISTORICAL row, and -- if new -- the supplement the TestSupplementContract would demand, decided
on the generic multi-axis point W=(2,3), K=(2,2), C=(2,3) (the test's own shapes)."""
import sys, re, ast, pathlib; sys.path.insert(0, __import__("os").getcwd())
import numpy as np
from t3toolbox.backend.contractions import contract
src = pathlib.Path('tests/test_contractions_interpreter.py').read_text()
H = ast.literal_eval(src[src.index('HISTORICAL = {') + len('HISTORICAL = '): src.index('}', src.index('HISTORICAL = {')) + 1])
used = set()
for f in ['c5_apply_twohole.py', 'c6_probe_t3t3_twohole.py', 'c13_apply_jets_twohole.py', 'c14_probe_derivs_twohole.py']:
    t = pathlib.Path(__file__).with_name(f).read_text()
    used |= set(re.findall(r"'([a-zA-Z]+(?:,[a-zA-Z]+)+->[a-zA-Z]+)'", t))
used = {s for s in used if any(ch.isupper() for ch in s)}
shapes = {'W': (2, 3), 'K': (2, 2), 'C': (2, 3)}
rng = np.random.default_rng(0)
def ops_for(s):
    ins = s.split('->')[0].split(',')
    out = []
    for term in ins:
        shp = ()
        for ch in term:
            shp += shapes[ch] if ch.isupper() else ((3 if ch in 'trs' else 2),)
        out.append(rng.standard_normal(shp))
    return out
new = []
for s in sorted(used):
    if s in H:
        continue
    ops = ops_for(s)
    try:
        contract(s, *ops); req = ''
    except ValueError:
        ok = [g for g in 'WKC' if g in s and _ok(s, ops, g)] if False else []
        for g in 'CWK':
            if g not in s: continue
            try:
                contract(s, *ops, **{'len_' + g: len(shapes[g])}); ok.append(g)
            except ValueError:
                pass
        req = '/'.join(ok) if ok else 'NONE-SUFFICES'
    new.append((s, req))
print('strings used: %d ; already HISTORICAL: %d ; new: %d' % (len(used), len(used) - len(new), len(new)))
for s, req in new:
    print('  %-34s single-letter supplements that suffice: %r' % (s, req))
