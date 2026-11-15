import pathlib, sys, types
import numpy as np
gm = types.ModuleType('gm'); exec(compile(pathlib.Path('/home/claude/graded_messages_handoff/scripts/graded_messages_prototype.py').read_text(), 'gm', 'exec'), gm.__dict__)
rng = np.random.default_rng(2); rn = rng.standard_normal
Ns, nD = (6, 5, 7, 4), (3, 4, 2, 3)
r = (2, 3, 2, 4, 3)          # tails > 1 as well
d, W, P = len(Ns), 5, 4
Bc = [rn((nD[i], Ns[i])) for i in range(d)]; Gc = [rn((r[i], nD[i], r[i + 1])) for i in range(d)]
Bd = [rn(b.shape) for b in Bc]; Gd = [rn(g.shape) for g in Gc]
ww = [rn((W, N)) for N in Ns]; pp = [rn((W, N)) for N in Ns]
zt = [rn((P, W, N)) for N in Ns]
bL = gm.ones_boundary(P, W); bL = {'': gm.jet(P, np.ones((W, r[0])))}; bR = {'': gm.jet(P, np.ones((W, r[-1])))}
resid = [{'': gm.jet(P, ww[i], pp[i]), 'z': gm.dual_jet(P, zt[i])} for i in range(d)]
f = gm.run(*gm.corewise((Bc, Gc), (Bd, Gd)), resid, bL, bR, P)
for g in ('', 't', 'z', 'tz'):
    vals = [gm.gmul('a,a->', [f['M'][i], f['N'][i]], P, (g,))[g] for i in range(d + 1)]
    vals += [gm.gmul('i,i->', [f['X'][i], f['E'][i]], P, (g,))[g] for i in range(d)]
    vals += [gm.gmul('o,o->', [resid[i], f['Z'][i]], P, (g,))[g] for i in range(d)]
    ref = vals[0]
    print('corewise frame, grade %-3r: %d edges, max rel spread over edges %.1e' % (g, len(vals), max(np.linalg.norm(v - ref) / np.linalg.norm(ref) for v in vals)))
# manifold frame: plain pairing on interior bonds
rL, rR = (1, 3, 2, 4, 1), (1, 2, 4, 3, 1); nU = (2, 3, 3, 2)
D = [rn((nD[i], Ns[i])) for i in range(d)]; U = [rn((rL[i], nU[i], rR[i + 1])) for i in range(d)]
L = [rn((rL[i], nD[i], rL[i + 1])) for i in range(d)]; R = [rn((rR[i], nD[i], rR[i + 1])) for i in range(d)]
bnd = gm.ones_boundary(P, W)
m = gm.run(*gm.manifold((D, U, L, R)), resid, bnd, bnd, P)
for g in ('', 'z'):
    for i in range(d + 1):
        try:
            v = gm.gmul('a,a->', [m['M'][i], m['N'][i]], P, (g,))
            print('manifold frame, grade %-3r bond %d: %s' % (g, i, 'defined' if g in v else 'no terms'))
        except Exception as e:
            print('manifold frame, grade %-3r bond %d: %s' % (g, i, str(e)[:70]))
