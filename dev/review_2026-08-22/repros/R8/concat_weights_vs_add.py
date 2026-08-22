"""R8: docs/weighting.md:56 and ut3_concatenate_weights' docstring pair `W_A.concatenate(W_B)` with `A + B`.
But the frontend `+` squashes the boundary bonds (rank 1+1 -> 1) while the weight concat does not."""
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
np.random.seed(0)
x = t3.TuckerTensorTrain.randn((4, 5, 3), (2, 3, 2), (1, 2, 3, 1)).t3svd()[0]
y = t3.TuckerTensorTrain.randn((4, 5, 3), (2, 2, 2), (1, 2, 2, 1)).t3svd()[0]
Wx, Wy = t3.T3Weights.from_t3svd(x), t3.T3Weights.from_t3svd(y)
s = x + y
Wc = Wx.concatenate(Wy)
print('ragged: (x+y).tt_ranks =', s.tt_ranks, '| concat weight tt lengths =', [w.shape[-1] for w in Wc.data[1]])
try:
    t3.t3_absorb_weights(s, Wc); print('ragged absorb(x+y, Wx.concat(Wy)): ok')
except Exception as e:
    print('ragged absorb(x+y, Wx.concat(Wy)) ->', type(e).__name__, str(e)[:120])
ux, uy = ut3.UniformTuckerTensorTrain.from_t3(x), ut3.UniformTuckerTensorTrain.from_t3(y)
UWx = ut3.UT3Weights.from_t3weights(Wx, n=ux.n, r=ux.r); UWy = ut3.UT3Weights.from_t3weights(Wy, n=uy.n, r=uy.r)
us = ux + uy; UWc = UWx.concatenate(UWy)
print('uniform: (x+y) tt bond-0 mask', us.masks.tt_edge_mask[0].astype(int).tolist(), '| concat weight bond-0 mask', UWc.masks.tt_edge_mask[0].astype(int).tolist())
try:
    ut3.ut3_absorb_weights(us, UWc); print('uniform absorb(x+y, Wx.concat(Wy)): ok')
except Exception as e:
    print('uniform absorb(x+y, Wx.concat(Wy)) ->', type(e).__name__, str(e).splitlines()[0][:120])
