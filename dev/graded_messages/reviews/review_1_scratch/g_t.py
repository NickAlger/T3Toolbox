# does the prototype silently produce a grade-t core gradient in a manifold frame when shapes coincide?
import pathlib, types
import numpy as np
gm = types.ModuleType('gm'); exec(compile(pathlib.Path('/home/claude/graded_messages_handoff/scripts/graded_messages_prototype.py').read_text(), 'gm', 'exec'), gm.__dict__)
rng = np.random.default_rng(0); rn = rng.standard_normal
for label, rL, rR, nD, nU in (('rL != rR', (1, 3, 2, 4, 1), (1, 2, 4, 3, 1), (3, 4, 2, 3), (2, 3, 3, 2)),
                              ('rL == rR, nU == nD', (1, 3, 3, 3, 1), (1, 3, 3, 3, 1), (3, 3, 3, 3), (3, 3, 3, 3))):
    Ns = (6, 5, 7, 4); d, W, P = 4, 5, 1
    D = [rn((nD[i], Ns[i])) for i in range(d)]; U = [rn((rL[i], nU[i], rR[i + 1])) for i in range(d)]
    L = [rn((rL[i], nD[i], rL[i + 1])) for i in range(d)]; R = [rn((rR[i], nD[i], rR[i + 1])) for i in range(d)]
    V = [rn((nU[i], Ns[i])) for i in range(d)]; H = [rn((rL[i], nD[i], rR[i + 1])) for i in range(d)]
    w = [{'': gm.jet(P, rn((W, N)))} for N in Ns]; bnd = gm.ones_boundary(P, W)
    try:
        out = gm.run(*gm.manifold((D, U, L, R), (V, H)), w, bnd, bnd, P, grad=('t',))
        print(label, '-> grade-t TT gradient returned without error, shapes', [g['t'].shape[2:] for g in out['dG']])
    except Exception as e:
        print(label, '->', type(e).__name__, str(e)[:80])
