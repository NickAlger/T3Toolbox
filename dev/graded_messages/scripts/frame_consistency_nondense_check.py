"""Frame consistency without densifying: norm of the difference of the two canonical forms, built from the
library's own T3 subtraction and norm, against the existing dense residual.
Usage: python3 frame_consistency_nondense_check.py <repo>"""
import sys, time
import numpy as np
sys.path.append(sys.argv[1])
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.frame_variations_format as fvf
np.random.seed(0)
shape = (10, 11, 12, 9)
x = t3.TuckerTensorTrain.randn(shape, (5, 6, 4, 5), (1, 3, 4, 3, 1))
frame = fvf.T3Frame.from_t3(x)
D, U, L, R = frame.data
def cheap(D, L, R):                       # norm of the difference T3, via the library's own subtraction and norm
    a, b = t3.TuckerTensorTrain(D, L), t3.TuckerTensorTrain(D, R)
    return float((a - b).norm() / b.norm())
print('existing dense residual (fv_frame_consistency_residual):', float(np.asarray(__import__('t3toolbox.backend.fv_operations', fromlist=['x']).fv_frame_consistency_residual(frame.data))))
print('non-dense residual, consistent frame:                   %.1e' % cheap(D, L, R))
for eps in (1e-3, 1e-6, 1e-9, 1e-12):
    Rb = list(R); Rb[0] = R[0] * (1 + eps)
    bad = fvf.T3Frame(D, U, L, tuple(Rb))
    print('R[0] scaled by 1+%.0e:  dense %.2e   non-dense %.2e' % (eps, float(np.asarray(__import__('t3toolbox.backend.fv_operations', fromlist=['x']).fv_frame_consistency_residual(bad.data))), cheap(D, L, Rb)))
