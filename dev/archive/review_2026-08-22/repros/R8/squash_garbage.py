"""R8: does ut3_squash_tails (and the frontend ops that call it: squash_tails, +, -, sum_stack) mix
garbage padding into the real boundary-bond slot?  Prong 3 (garbage-padded input == clean input)."""
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
from t3toolbox.backend.common import prefix_mask

np.random.seed(0)

def corrupt(ux, scale=10.0):
    """Write finite garbage into every padded slot of both supercores (masks untouched)."""
    tkm, ttm = ux.masks.data
    d, N, n, r = ux.d, ux.N, ux.n, ux.r
    stack = ux.stack_shape
    shape_mask = prefix_mask(ux.shape, N).reshape((d,) + (1,) * len(stack) + (1, N))
    tk_real = tkm[..., :, None] & shape_mask                       # (d,)+stack+(n,N)
    tt_real = ttm[:-1][..., :, None, None] & tkm[..., None, :, None] & ttm[1:][..., None, None, :]
    g_tk = scale * np.random.randn(*ux.tucker_supercore.shape) * (~tk_real)
    g_tt = scale * np.random.randn(*ux.tt_supercore.shape) * (~tt_real)
    return ut3.UniformTuckerTensorTrain(ux.tucker_supercore + g_tk, ux.tt_supercore + g_tt, ux.shape, ux.masks)

def relerr(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))

x = t3.TuckerTensorTrain.randn((4, 5, 3), (2, 3, 2), (1, 2, 3, 1))
y = t3.TuckerTensorTrain.randn((4, 5, 3), (3, 2, 2), (1, 3, 2, 1))
# pad ABOVE the real ranks so the boundary bond r0=1 sits in a padded r=4 axis
ux = ut3.UniformTuckerTensorTrain.from_t3(x, n=4, r=4)
uy = ut3.UniformTuckerTensorTrain.from_t3(y, n=4, r=4)
uxg, uyg = corrupt(ux), corrupt(uy)

print('sanity: to_dense is garbage-robust          :', relerr(uxg.to_dense(), ux.to_dense()))
print('squash_tails garbage vs clean (rel err)     :', relerr(uxg.squash_tails().to_dense(), ux.to_dense()))
print('x + y   garbage vs clean (rel err)          :', relerr((uxg + uyg).to_dense(), (x + y).to_dense()))
print('x - y   garbage vs clean (rel err)          :', relerr((uxg - uyg).to_dense(), (x - y).to_dense()))

# stacked: sum_stack
xs = t3.TuckerTensorTrain.randn((4, 5, 3), (2, 3, 2), (1, 2, 3, 1), stack_shape=(2,))
us = ut3.UniformTuckerTensorTrain.from_t3(xs, n=4, r=4)
usg = corrupt(us)
ref = xs.to_dense().sum(axis=0)
print('sum_stack garbage vs clean (rel err)        :', relerr(usg.sum_stack().to_dense(), ref))
print('sum_stack clean     vs ragged (rel err)     :', relerr(us.sum_stack().to_dense(), ref))

# Does a LIBRARY op itself leave garbage in the boundary-bond padding?  left_orthogonalize is SVD-based:
# the zero-sigma completion columns can land in padded rows.
lo = ux.left_orthogonalize_tt_cores()
tkm, ttm = lo.masks.data
G0 = lo.tt_supercore[0]                        # (r, n, r): left leg is bond 0 (mask [T,F,F,F])
pad_rows = G0[~ttm[0]]                         # rows of the padded left-leg slots
print('left_orthogonalize: |G0 padded-left-leg rows| =', float(np.abs(pad_rows).max()),
      ' bond0 mask =', ttm[0].astype(int).tolist())
print('left_orth then squash: rel err vs x       :', relerr(lo.squash_tails().to_dense(), x.to_dense()))
print('left_orth(x) + uy    : rel err vs x+y      :', relerr((lo + uy).to_dense(), (x + y).to_dense()))
ro = ux.right_orthogonalize_tt_cores()
Gf = ro.tt_supercore[-1]
print('right_orthogonalize: |Gf padded-right-leg cols| =', float(np.abs(Gf[..., ~ro.masks.tt_edge_mask[-1]]).max()))
print('right_orth(x) + uy   : rel err vs x+y      :', relerr((ro + uy).to_dense(), (x + y).to_dense()))
# and the down-orth Tucker SVD completion: garbage in Tucker padded rows is harmless for squash (not summed)
do = ux.down_orthogonalize_tucker_cores()
print('down_orth: |padded tucker rows| =', float(np.abs(do.tucker_supercore[~do.masks.tucker_edge_mask]).max()))
