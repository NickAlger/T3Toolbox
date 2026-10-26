"""Read-only plan check (v3, M4 fix): in the REAL uniform layout (UniformTuckerTensorTrain.from_t3, boundary bonds
padded), mask BOTH theta and theta_hat on entry with ut3_apply_masks(point masks) -- K-stacked theta_hat against
C-only masks, via the '...' broadcast -- then run the probe_derivatives two-hole on the per-core slices.
Expect: garbage anywhere in theta_hat's padding (incl. boundary-bond slots) -> real block exact (vs ragged);
mutation (skip the theta_hat mask) -> real block wrong."""
import sys, os; sys.path.insert(0, __import__("os").getcwd()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.backend.ut3_masking as um
from c14_probe_derivs_twohole import two_hole
np.random.seed(7); rng = np.random.default_rng(7)
shape, tr, ttr, order, Wn, K = (4, 5, 6), (2, 3, 2), (1, 2, 2, 1), 2, 5, (2,)
x = t3.TuckerTensorTrain.randn(shape, tr, ttr)
ux = ut3.UniformTuckerTensorTrain.from_t3(x, n=4, r=3, N=7)        # force padding on every edge, boundaries included
tk_sc, tt_sc, sh, masks = ux.tucker_supercore, ux.tt_supercore, ux.shape, ux.masks.data if hasattr(ux.masks, 'data') else ux.masks
print('tt edge mask row 0 / row d:', masks[1][0].astype(int), masks[1][-1].astype(int), ' supercores', tk_sc.shape, tt_sc.shape)
U, G = x.data
Ud = tuple(rng.standard_normal(K + u.shape) for u in U); Gd = tuple(rng.standard_normal(K + g.shape) for g in G)
ww = [rng.standard_normal((Wn, N)) for N in shape]; pp = [rng.standard_normal((Wn, N)) for N in shape]
zt = [rng.standard_normal((order + 1, Wn, N)) for N in shape]
ref = two_hole(U, G, Ud, Gd, ww, pp, zt, order, K, ())
d, n, N, r = 3, 4, 7, 3
def embed_dir(garbage):
    tk = 1e3 * rng.standard_normal((d,) + K + (n, N)) if garbage else np.zeros((d,) + K + (n, N))
    tt = 1e3 * rng.standard_normal((d,) + K + (r, n, r)) if garbage else np.zeros((d,) + K + (r, n, r))
    for i in range(d):
        tk[(i,) + (slice(None),) * len(K) + tuple(slice(0, s) for s in Ud[i].shape[len(K):])] = Ud[i]
        tt[(i,) + (slice(None),) * len(K) + tuple(slice(0, s) for s in Gd[i].shape[len(K):])] = Gd[i]
    return tk, tt
pad = lambda a: np.concatenate([a, np.zeros(a.shape[:-1] + (N - a.shape[-1],))], axis=-1)
wwp, ppp, ztp = [pad(w) for w in ww], [pad(p) for p in pp], [pad(z) for z in zt]
mtk, mtt = um.ut3_apply_masks((tk_sc, tt_sc, sh, masks))                     # theta masked (the existing wrapper's step)
th_cores = (tuple(mtk[i] for i in range(d)), tuple(mtt[i] for i in range(d)))
for label, mask_dir in (('theta_hat MASKED on entry', True), ('MUTATION: theta_hat NOT masked', False)):
    tkd, ttd = embed_dir(garbage=True)
    if mask_dir:
        tkd, ttd = um.ut3_apply_masks((tkd, ttd, sh, masks))               # K+C supercore vs C-only masks: '...' broadcast
    dU, dG = two_hole(th_cores[0], th_cores[1], tuple(tkd[i] for i in range(d)), tuple(ttd[i] for i in range(d)),
                      wwp, ppp, ztp, order, K, ())
    err = max(np.linalg.norm(a[(slice(None),) * len(K) + tuple(slice(0, s) for s in b.shape[len(K):])] - b) / np.linalg.norm(b)
              for a, b in zip(dU + dG, ref[0] + ref[1]))
    print('%-34s garbage-padded theta_hat (incl. boundary slots): real-block rel err %.1e' % (label, err))
