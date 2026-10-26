"""Read-only plan check: for each NEW grouped-subscript string the plan introduces, does contract()
infer the group split unaided, and if not which len_* supplement does it demand?  W=(3,), K=(2,), C=(4,)."""
import sys; sys.path.insert(0, __import__("os").getcwd())
import re, numpy as np
from t3toolbox.backend.contractions import contract
NEW = [
    # plain apply/entries two-hole (dG assembly, K on mu-dot / xi-dot)
    'WKCa,WCi,WCb->KCaib', 'WCa,WKCi,WCb->KCaib',
    # probe two-hole: deta-tilde-dot and the eta-dot Tucker assembly term
    'WCo,KCio->WKCi', 'WCo,WKCa->KCao',
    # jets: sigma-hat-dot (adjoint wiring), apply-deriv dG assembly dots
    'trs,tWCa,KCaib,rWCi->sWKCb', 'trs,tWCa,Caib,rWKCi->sWKCb',
    'trs,rWKCa,sWCi,tWCb->KCaib', 'trs,rWCa,sWKCi,tWCb->KCaib',
    'trs,rWKCa,Caib,tWCb->sWKCi', 'trs,rWCa,KCaib,tWCb->sWKCi', 'trs,rWCa,Caib,tWKCb->sWKCi',
    'uWCa,uWo->KCao',
    # uniform d-prefixed twins of the plain two-hole assemblies
    'dWKCa,dWCi,dWCb->dKCaib', 'dWCa,dWKCi,dWCb->dKCaib', 'dWCo,dKCio->dWKCi', 'dWCo,dWKCa->dKCao',
]
SIZES = {'W': (3,), 'K': (2,), 'C': (4,)}
rng = np.random.default_rng(0)
def build(subs):
    ins, out = subs.split('->')
    ops = []
    for term in ins.split(','):
        shp = ()
        for ch in term:
            shp += SIZES[ch] if ch.isupper() else (({'t': 3, 'r': 3, 's': 3, 'u': 2}.get(ch, 2)),)
        ops.append(rng.standard_normal(shp))
    return ops
for s in NEW:
    if s == 'uWCa,uWo->KCao':
        continue
    ops = build(s)
    need = []
    try:
        contract(s, *ops)
        need = ''
    except Exception as e:
        msg = str(e)
        for letter in 'WKC':
            try:
                contract(s, *ops, **{'len_' + letter: 1})
                need = letter
                break
            except Exception:
                pass
        if need == []:
            need = 'UNRESOLVED: ' + msg[:120]
    print('%-32s supplement: %r' % (s, need))
