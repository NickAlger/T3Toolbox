"""Read-only plan check (v3, m26 pushback): nu_c = T nu (F16) is a MANIFOLD-frame identity -- T is the geometry's
bond factor (zipper of the orthogonal L against R). At a COREWISE frame (U,G,G,G) the stored nu already runs through
G, and applying the 'T' built from that frame is wrong. So a geometry-agnostic kind cannot use the transform without a
geometry->kind channel; it re-sweeps nu through G instead (c25: ~2% of the two-hole)."""
import sys; sys.path.insert(0, __import__("os").getcwd())
import numpy as np
import t3toolbox.backend.fv_conversions as fvc, t3toolbox.backend.sampling_derivatives as sd
import t3toolbox.backend.t3_constructors as t3k, t3toolbox.backend.tt_operations as tto
np.random.seed(5); rng = np.random.default_rng(5)
shape, tr, ttr, order, W = (4, 5, 6, 4), (2, 3, 3, 2), (1, 2, 3, 2, 1), 2, 5
x = t3k.t3_corewise_randn(shape, tr, ttr)
ww = [rng.standard_normal((W, N)) for N in shape]; pp = [rng.standard_normal((W, N)) for N in shape]
mf, _ = fvc.t3_orthogonal_representations(x)
for label, fr in (('manifold frame (U,O,L,R)', mf), ('corewise frame (U,G,G,G)', fvc.t3_corewise_frame(x))):
    xi, mu, nu, eta = sd.tv_precompute_probe_frame_sweep_jets(fr, ww, pp, order)
    nu_through_P = sd.compute_nu_jets(fr[2], xi, sd.binomial_combine_tensor(order))   # the corewise nu at THIS frame's (U, P)
    T = tto.tt_zipper_right_to_left(fr[2], fr[3])[1:]
    err = max(np.linalg.norm(np.einsum('...ab,t...b->t...a', T[i], nu[i]) - nu_through_P[i]) / np.linalg.norm(nu_through_P[i]) for i in range(3))
    print('%-26s |T nu - nu_c| / |nu_c| = %.1e' % (label, err))
