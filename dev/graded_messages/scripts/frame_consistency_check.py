"""Are library frames consistent ((D, L) and (D, R) the same tensor)? Does a sample-based check detect an
inconsistent one? Is the final carry discarded by compute_mu the apply value of the base point?
Usage: python3 frame_consistency_check.py <repo>"""
import sys
import numpy as np
sys.path.append(sys.argv[1])
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.frame_variations_format as fvf
import t3toolbox.backend.apply as apply
import t3toolbox.backend.probing as probing

np.random.seed(0); rng = np.random.default_rng(0)
shape = (10, 11, 12, 9)
x = t3.TuckerTensorTrain.randn(shape, (5, 6, 4, 5), (1, 3, 4, 3, 1))
frame = fvf.T3Frame.from_t3(x)
D, U, L, R = frame.data
k = 8
ww = [rng.standard_normal((k, N)) for N in shape]

def gap(D, L, R):                      # the two plain terminals: apply through the L-form vs through the R-form
    a, b = apply.t3_apply((D, L), ww), apply.t3_apply((D, R), ww)
    return np.linalg.norm(a - b) / np.linalg.norm(a)

truth = apply.t3_apply(x.data, ww)
print('library frame (from_t3):')
print('  (D, L) vs original T3 on %d random rank-1 samples: %.1e' % (k, np.linalg.norm(apply.t3_apply((D, L), ww) - truth) / np.linalg.norm(truth)))
print('  (D, R) vs original T3:                             %.1e' % (np.linalg.norm(apply.t3_apply((D, R), ww) - truth) / np.linalg.norm(truth)))
print('  L-form vs R-form gap:                              %.1e' % gap(D, L, R))

R_bad = list(R); R_bad[0] = R[0] * 1.001                         # a 0.1% inconsistency in the first right core
L_bad = list(L); L_bad[-1] = L[-1] + 1e-6 * rng.standard_normal(L[-1].shape)
print('hand-built inconsistent frames:')
print('  R[0] scaled by 1.001      -> gap %.1e' % gap(D, L, R_bad))
print('  L[-1] perturbed by 1e-6   -> gap %.1e' % gap(D, L_bad, R))

# what the existing machinery already computes: mu's final carry (discarded by compute_mu) is the L-form value
xis = probing.compute_xi(D, ww)
mus = probing.compute_mu(L, xis)
last = np.einsum('wa,aib,wi->wb', mus[-1], L[-1], xis[-1]).sum(-1)
print('discarded final carry of compute_mu vs apply of the base point: %.1e' % (np.linalg.norm(last - truth) / np.linalg.norm(truth)))
