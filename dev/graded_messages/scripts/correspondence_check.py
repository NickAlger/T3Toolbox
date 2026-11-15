"""Check, array by array, that each current edge-variable jet equals the proposed one under the two rules
   forward grades:  new[k] = old[k] / k!          residual grades:  new[k] = (p-k)! * old[p-k]
Usage: python3 correspondence_check.py <repo>"""
import pathlib, sys, types, math
import numpy as np
repo = sys.argv[1]
sys.path.append(repo)
HERE = pathlib.Path(__file__).resolve().parent
gm = types.ModuleType('gm'); exec(compile((HERE / 'graded_messages_prototype.py').read_text(), 'gm', 'exec'), gm.__dict__)
import t3toolbox.backend.sampling_derivatives as sd
from t3toolbox.backend.probing import compute_xi

rng = np.random.default_rng(1); rn = rng.standard_normal
Ns, nD, nU = (6, 5, 7, 4), (3, 4, 2, 3), (2, 3, 3, 2)
rL, rR = (1, 3, 2, 4, 1), (1, 2, 4, 3, 1)
d, W, order = len(Ns), 5, 3
P = order + 1
D = [rn((nD[i], Ns[i])) for i in range(d)]
U = [rn((rL[i], nU[i], rR[i + 1])) for i in range(d)]
L = [rn((rL[i], nD[i], rL[i + 1])) for i in range(d)]
R = [rn((rR[i], nD[i], rR[i + 1])) for i in range(d)]
V = [rn((nU[i], Ns[i])) for i in range(d)]
H = [rn((rL[i], nD[i], rR[i + 1])) for i in range(d)]
frame = (D, U, L, R)
ww = [rn((W, N)) for N in Ns]; pp = [rn((W, N)) for N in Ns]
zt = [rn((P, W, N)) for N in Ns]; c = rn((P, W))

# ---- current (library) edge-variable jets
xi_j, mu_j, nu_j, eta_j = sd.tv_precompute_probe_frame_sweep_jets(frame, ww, pp, order)
dxi_j = sd.build_input_jets(compute_xi(V, ww), compute_xi(V, pp))
sig_j = sd.compute_sigma_jets(H, R, U, xi_j, dxi_j, mu_j)
tau_j = sd.compute_tau_jets(H, L, U, xi_j, dxi_j, nu_j)
deta_j = sd.compute_deta_jets(H, L, R, mu_j, nu_j, sig_j, tau_j)
detat = sd.compute_deta_tilde_jets(D, zt)
taut = sd.compute_tau_tilde_jets(L, xi_j, detat, mu_j)
sigt = sd.compute_sigma_tilde_jets(R, xi_j, detat, nu_j)
dxit = sd.compute_dxi_tilde_jets(U, mu_j, nu_j, sigt, taut)
sigh = sd.compute_sigma_hat_jets(R, xi_j, c)

# ---- proposed (prototype) graded Taylor jets
bnd = gm.ones_boundary(P, W)
plain = [{'': gm.jet(P, ww[i], pp[i])} for i in range(d)]
resid = [{'': gm.jet(P, ww[i], pp[i]), 'z': gm.dual_jet(P, zt[i])} for i in range(d)]
seeded = {'': bnd[''], 'z': gm.dual_jet(P, c)[..., None]}
f = gm.run(*gm.manifold(frame, (V, H)), plain, bnd, bnd, P)      # grades '', 't'
t = gm.run(*gm.manifold(frame), resid, bnd, bnd, P)              # grade 'z', residual on leaves
a = gm.run(*gm.manifold(frame), plain, bnd, seeded, P)           # grade 'z', residual at the closure

fwd = lambda new: new * np.array([math.factorial(k) for k in range(new.shape[0])]).reshape((-1,) + (1,) * (new.ndim - 1))
def res(new):                                                    # invert  new[k] = (p-k)! old[p-k]
    rev = new[::-1]
    return rev / np.array([math.factorial(k) for k in range(new.shape[0])]).reshape((-1,) + (1,) * (new.ndim - 1))

def check(label, olds, news, conv, rows=None):
    err = 0.0
    for o, n in zip(olds, news):
        n = conv(n)
        if rows is not None: n = n[:rows]
        o = np.asarray(o)[:n.shape[0]]
        err = max(err, np.linalg.norm(o - n) / np.linalg.norm(o))
    print('  %-46s rel err %.1e' % (label, err))

r = range(d)
check('xi_jets (2 rows)   <-> xi',            xi_j,   [f['X'][i][''] for i in r], fwd, 2)
check('mu_jets            <-> mu',            mu_j,   [f['M'][i][''] for i in r], fwd)
check('nu_jets            <-> nu',            nu_j,   [f['N'][i + 1][''] for i in r], fwd)
check('eta_jets           <-> eta',           eta_j,  [f['E'][i][''] for i in r], fwd)
check('dxi_jets (2 rows)  <-> xi_t',          dxi_j,  [f['X'][i]['t'] for i in r], fwd, 2)
check('sigma_jets         <-> mu_t',          sig_j[1:],  [f['M'][i]['t'] for i in range(1, d)], fwd)
check('tau_jets           <-> nu_t',          tau_j[:-1], [f['N'][i + 1]['t'] for i in range(d - 1)], fwd)
check('deta_jets          <-> eta_t',         deta_j, [f['E'][i]['t'] for i in r], fwd)
check('deta_tildes        <-> xi_z',          detat,  [t['X'][i]['z'] for i in r], res)
check('tau_tildes         <-> mu_z',          taut[1:],   [t['M'][i]['z'] for i in range(1, d)], res)
check('sigma_tildes       <-> nu_z',          sigt[:-1],  [t['N'][i + 1]['z'] for i in range(d - 1)], res)
check('dxi_tildes (2 rows)<-> eta_z (top 2)', dxit,   [t['E'][i]['z'] for i in r], res, 2)
check('sigma_hats         <-> nu_z (closure)', sigh,  [a['N'][i + 1]['z'] for i in r], res)
print('  boundary slots: sigma_jets[0] max |.| = %.1e, tau_jets[-1] max |.| = %.1e, tau_tildes[0] = %.1e, sigma_tildes[-1] = %.1e'
      % (np.abs(sig_j[0]).max(), np.abs(tau_j[-1]).max(), np.abs(taut[0]).max(), np.abs(sigt[-1]).max()))
print('  prototype has no component there: %s %s %s %s' % ('t' in f['M'][0], 't' in f['N'][d], 'z' in t['M'][0], 'z' in t['N'][d]))
