"""Is  <forward message, reverse message>  the same number on every edge? (grade t, manifold frame, jets)
Usage: python3 pairing_check.py [<repo>]   (the repo is not needed; accepted for a uniform command line)"""
import pathlib, sys, types
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
gm = types.ModuleType('gm'); exec(compile((HERE / 'graded_messages_prototype.py').read_text(), 'gm', 'exec'), gm.__dict__)
rng = np.random.default_rng(2); rn = rng.standard_normal
Ns, nD, nU = (6, 5, 7, 4), (3, 4, 2, 3), (2, 3, 3, 2)
rL, rR = (1, 3, 2, 4, 1), (1, 2, 4, 3, 1)
d, W, P = len(Ns), 5, 4
D = [rn((nD[i], Ns[i])) for i in range(d)]; U = [rn((rL[i], nU[i], rR[i + 1])) for i in range(d)]
L = [rn((rL[i], nD[i], rL[i + 1])) for i in range(d)]; R = [rn((rR[i], nD[i], rR[i + 1])) for i in range(d)]
V = [rn((nU[i], Ns[i])) for i in range(d)]; H = [rn((rL[i], nD[i], rR[i + 1])) for i in range(d)]
ww = [rn((W, N)) for N in Ns]; pp = [rn((W, N)) for N in Ns]
bnd = gm.ones_boundary(P, W)
leaves = [{'': gm.jet(P, ww[i], pp[i])} for i in range(d)]
f = gm.run(*gm.manifold((D, U, L, R), (V, H)), leaves, bnd, bnd, P)
ref = f['Phi']['t']
rel = lambda a: np.linalg.norm(a - ref) / np.linalg.norm(ref)
print('grade t, all jet orders, relative difference from tv_apply-derivatives value:')
print('  bonds 0..d   <mu, nu>  :', ' '.join('%.0e' % rel(gm.gmul('a,a->', [f['M'][i], f['N'][i]], P, ('t',))['t']) for i in range(d + 1)))
print('  mode edges   <xi, eta> :', ' '.join('%.0e' % rel(gm.gmul('i,i->', [f['X'][i], f['E'][i]], P, ('t',))['t']) for i in range(d)))
print('  leaves       <w, z>    :', ' '.join('%.0e' % rel(gm.gmul('o,o->', [leaves[i], f['Z'][i]], P, ('t',))['t']) for i in range(d)))
print('components per bond (space dims):')
for i in range(d + 1):
    print('  bond %d: mu %s | nu %s' % (i, {g: a.shape[-1] for g, a in f['M'][i].items()}, {g: a.shape[-1] for g, a in f['N'][i].items()}))
