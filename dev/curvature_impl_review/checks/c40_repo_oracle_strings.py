"""Read-only plan check (v6, M5): every new grouped string the plan lists, run through the REPO's own contraction-oracle
machinery (tests/test_contractions_interpreter.py: build_operands, loop_oracle, all_lens, groups_of, SHAPE_MATRIX, and
the TestSupplementContract logic) -- with AXIS_SIZE extended by the plan's proposed S1 edit {l, n, x, z} -- plus their
uniform d-prefixed twins. Also re-runs the existing HISTORICAL table under the extended AXIS_SIZE (must be unaffected)."""
import sys, re
sys.path.insert(0, __import__("os").getcwd()); sys.path.insert(0, "tests")
import numpy as np
import test_contractions_interpreter as T
from t3toolbox.backend.contractions import contract

EXT = {'l': 4, 'n': 6, 'x': 7, 'z': 5}          # the proposed AXIS_SIZE additions (distinct from co-occurring letters)
missing_before = set()
NEW = {   # string -> the supplement the plan records
    # v6 non-jet strings
    'Cux,KCxn->KCun': '', 'KCaib,Cbc->KCaic': '', 'Cux,KCun->KCxn': '', 'KCaic,Cbc->KCaib': '',
    'Caib,Cbc->Caic': '', 'Caxb,Caub->Cxu': '',
    'KCij,Ciak,Cjal->KCkl': '', 'Cij,KCiak,Cjal->KCkl': '', 'Cij,Ciak,KCjal->KCkl': '',
    'Czn,KCun->KCzu': '', 'Cazb,KCzu->KCaub': '', 'KCazb,Czu->KCaub': '',
    # factored TT assembly
    'trs,rWCa,sWCi->tWCai': '', 'trs,rWKCa,sWCi->tWKCai': 'C', 'trs,rWCa,sWKCi->tWKCai': 'C',
    'tWKCai,tWCb->KCaib': 'C', 'tWCai,tWKCb->KCaib': 'C',
    'trs,tWCa,rWCi->sWCai': '', 'trs,tWKCa,rWCi->sWKCai': 'C', 'trs,tWCa,rWKCi->sWKCai': 'C',
    'trs,rWCa,tWCi->sWCai': '', 'trs,rWKCa,tWCi->sWKCai': 'C', 'trs,rWCa,tWKCi->sWKCai': 'C',
    'sWKCai,sWCb->KCaib': 'C', 'sWCai,sWKCb->KCaib': 'C',
    # v4's jet rows (kept): the plan's D list
    'WKCa,WCi,WCb->KCaib': 'W', 'WCa,WKCi,WCb->KCaib': 'W',
    'WCo,KCio->WKCi': 'C', 'tWCo,KCio->tWKCi': 'C',
    'WCo,WKCa->KCao': 'W', 'tWKCa,tWCo->KCao': 'W',
    'trs,tWCa,KCaib,rWCi->sWKCb': 'C', 'trs,rWCa,KCaib,tWCi->sWKCb': 'C',
    'trs,rWCa,KCaib,tWCb->sWKCi': 'C', 'trs,tWCa,KCaib,sWCb->rWKCi': 'C',
    'trs,rWKCa,sWCi,tWCb->KCaib': 'W', 'trs,rWCa,sWKCi,tWCb->KCaib': 'W',
    'trs,tWCa,rWKCi,sWCb->KCaib': 'W', 'trs,tWCa,rWCi,sWKCb->KCaib': 'W',
    'trs,rWKCa,tWCi,sWCb->KCaib': 'W', 'trs,rWCa,tWCi,sWKCb->KCaib': 'W',
    'trs,tWCa,Caib,rWKCi->sWKCb': '', 'trs,rWKCa,Caib,tWCi->sWKCb': '',
    'trs,rWKCa,Caib,tWCb->sWKCi': '', 'trs,tWCa,Caib,sWKCb->rWKCi': '',
}
for s in NEW:
    missing_before |= {ch for ch in s if ch.islower() and ch not in T.AXIS_SIZE}
print('letters missing from the repo AXIS_SIZE today:', sorted(missing_before),
      '| strings that would KeyError today:', sum(any(ch.islower() and ch not in T.AXIS_SIZE for ch in s) for s in NEW))
T.AXIS_SIZE.update(EXT)

def dtwin(s):     # the uniform d-prefixed twin: a leading core axis 'd' on every operand and the output
    lhs, out = s.split('->')
    return ','.join('d' + t for t in lhs.split(',')) + '->d' + out

def check(subs, req):
    rng = np.random.default_rng(0)
    # TestVocabularyOracle body
    for W, K, C in T.SHAPE_MATRIX:
        shapes = {'W': W, 'K': K, 'C': C}
        for g in T.groups_of(subs):
            shapes.setdefault(g, (2,))
        ops = T.build_operands(subs, shapes, rng)
        ref = T.loop_oracle(subs, ops, shapes)
        got = contract(subs, *ops, **T.all_lens(subs, shapes))
        assert got.shape == ref.shape and np.allclose(got, ref, rtol=1e-10, atol=1e-12), (subs, W, K, C)
        lens = {'len_' + req: len(shapes[req])} if req else {}
        got2 = contract(subs, *ops, **lens)
        assert np.array_equal(np.asarray(got), np.asarray(got2)), (subs, 'inference path', W, K, C)
    # TestSupplementContract body
    shapes = {'W': (2, 3), 'K': (2, 2), 'C': (2, 3)}
    ops = T.build_operands(subs, shapes, rng)
    if req:
        try:
            contract(subs, *ops); return 'NOT DEMANDED'
        except ValueError as e:
            if 'do not determine' not in str(e): return 'WRONG ERROR ' + str(e)[:60]
        contract(subs, *ops, **{'len_' + req: len(shapes[req])})
    else:
        contract(subs, *ops)
    return 'ok'

bad = []
n = 0
for subs, req in NEW.items():
    for s in (subs, dtwin(subs)):
        n += 1
        try:
            r = check(s, req)
        except Exception as e:
            r = f'{type(e).__name__}: {str(e)[:80]}'
        if r != 'ok':
            bad.append((s, req, r))
print(f'{n} strings (ragged + d-prefixed twins) through the repo oracle + supplement contract with the extended AXIS_SIZE: '
      f'{n - len(bad)} pass')
for b in bad:
    print('   FAIL', b)
# the existing HISTORICAL table is unaffected by the extension (only new letters added)
nh = 0
for subs, req in T.HISTORICAL.items():
    r = check(subs, req); nh += 1
    assert r == 'ok', (subs, r)
print(f'existing HISTORICAL rows re-run under the extended AXIS_SIZE: {nh} pass')
