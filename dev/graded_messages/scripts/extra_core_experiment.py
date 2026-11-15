"""What do the 'extra' frame cores L[d-1] and R[0] actually touch? Replace them by (a) other random values,
(b) NaN, and see which outputs change / get poisoned. Usage: python3 extra_core_experiment.py <repo>"""
import pathlib, sys, types
import numpy as np
repo = sys.argv[1]
sys.path.append(repo)
HERE = pathlib.Path(__file__).resolve().parent
gm = types.ModuleType('gm'); exec(compile((HERE / 'graded_messages_prototype.py').read_text(), 'gm', 'exec'), gm.__dict__)
import t3toolbox.backend.probing as probing
import t3toolbox.backend.apply as apply
import t3toolbox.backend.sampling_derivatives as sd

rng = np.random.default_rng(0); rn = rng.standard_normal
Ns, nD, nU = (6, 5, 7, 4), (3, 4, 2, 3), (2, 3, 3, 2)
rL, rR = (1, 3, 2, 4, 1), (1, 2, 4, 3, 1)
d, W, order = len(Ns), 5, 2
P = order + 1
D = [rn((nD[i], Ns[i])) for i in range(d)]
U = [rn((rL[i], nU[i], rR[i + 1])) for i in range(d)]
L = [rn((rL[i], nD[i], rL[i + 1])) for i in range(d)]
R = [rn((rR[i], nD[i], rR[i + 1])) for i in range(d)]
V = [rn((nU[i], Ns[i])) for i in range(d)]
H = [rn((rL[i], nD[i], rR[i + 1])) for i in range(d)]
ww = [rn((W, N)) for N in Ns]; pp = [rn((W, N)) for N in Ns]
zt = [rn((P, W, N)) for N in Ns]; c = rn((P, W))

def flat(x):
    if isinstance(x, np.ndarray): return [x]
    return [a for y in x for a in flat(y)]

def library(L, R):
    f = (D, U, L, R); v = (V, H)
    return {
        'tv_probe':                       probing.tv_probe(ww, v, f),
        'tv_apply':                       apply.tv_apply(ww, v, f),
        'tv_probe_transpose':             probing.tv_probe_transpose([z[0] for z in zt], ww, f, sum_over_probes=True),
        'tv_apply_transpose':             apply.tv_apply_transpose(c[0], ww, f, sum_over_probes=True),
        'tv_probe_derivatives':           sd.tv_probe_derivatives(ww, pp, v, f, order),
        'tv_apply_derivatives':           sd.tv_apply_derivatives(ww, pp, v, f, order),
        'tv_probe_derivatives_transpose': sd.tv_probe_derivatives_transpose(zt, ww, pp, f, order, True, chunk_size=None),
        'tv_apply_derivatives_transpose': sd.tv_apply_derivatives_transpose(c, ww, pp, f, order, True),
    }

def prototype(L, R):
    f = (D, U, L, R)
    bnd = gm.ones_boundary(P, W)
    plain = [{'': gm.jet(P, ww[i], pp[i])} for i in range(d)]
    resid = [{'': gm.jet(P, ww[i], pp[i]), 'z': gm.dual_jet(P, zt[i])} for i in range(d)]
    seeded = {'': bnd[''], 'z': gm.dual_jet(P, c)[..., None]}
    fwd = gm.run(*gm.manifold(f, (V, H)), plain, bnd, bnd, P)
    tp = gm.run(*gm.manifold(f), resid, bnd, bnd, P)
    ta = gm.run(*gm.manifold(f), plain, bnd, seeded, P)
    top = lambda xs: [x['z'][-1].sum(axis=0) for x in xs]
    return {
        'J probe   (z at grade t)':        [z['t'] for z in fwd['Z']],
        'J apply   (Phi at grade t)':      fwd['Phi']['t'],
        'J^T probe (dB, dG at grade z)':   top(tp['dB']) + top(tp['dG']),
        'J^T apply (dB, dG at grade z)':   top(ta['dB']) + top(ta['dG']),
        'plain value (Phi at grade "")':   fwd['Phi'][''],
    }

def variants():
    L2 = list(L); R2 = list(R); L2[-1] = rn(L[-1].shape); R2[0] = rn(R[0].shape)
    L3 = list(L); R3 = list(R); L3[-1] = np.full(L[-1].shape, np.nan); R3[0] = np.full(R[0].shape, np.nan)
    return (L2, R2), (L3, R3)

for name, fn in (('LIBRARY', library), ('PROTOTYPE', prototype)):
    with np.errstate(all='ignore'):
        base = fn(L, R); (La, Ra), (Lb, Rb) = variants(); rand = fn(La, Ra); nan = fn(Lb, Rb)
    print(name)
    print('  %-34s %-28s %s' % ('output', 'extras -> other random values', 'extras -> NaN'))
    for k in base:
        diff = max(np.abs(a - b).max() for a, b in zip(flat(base[k]), flat(rand[k])))
        poisoned = any(np.isnan(a).any() for a in flat(nan[k]))
        print('  %-34s %-28s %s' % (k, 'unchanged' if diff == 0 else 'CHANGED (max diff %.1e)' % diff,
                                    'NaN in output' if poisoned else 'clean'))
