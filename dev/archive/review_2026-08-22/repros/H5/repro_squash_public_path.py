"""How a user reaches garbage-padded boundary bonds through the PUBLIC API, then hits the non-masking
ut3_squash_tails via `+`, `-`, `sum_stack`, `squash_tails`.
Path A: UNIFORM_COREWISE.randn(frame) fills the WHOLE padded variation supercore (docstring: 'filled
        completely'), and UNIFORM_COREWISE.retract adds it raw into the point -> garbage padding.
Path B: a backend user constructs UniformTuckerTensorTrain from raw supercores (contract: padding is
        don't-care).
"""
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.uniform_manifold as ut3m
import t3toolbox.backend.ut3_operations as ut3_ops
import t3toolbox.backend.ut3_masking as ut3_masking

np.random.seed(0)
x = t3.TuckerTensorTrain.randn((5, 7, 6), (2, 3, 2), (1, 2, 3, 1))
ux = ut3.UniformTuckerTensorTrain.from_t3(x, r=4)          # padded bonds r=4 > real boundary rank 1
frame = ut3m.UNIFORM_COREWISE.frame(ux)
v = ut3m.UNIFORM_COREWISE.randn(frame)                      # raw randn on the full padded supercores
tk, tt = v.variations.supercores
print('corewise randn variation: tt padding nonzero?', bool(np.any(tt[0][1:] != 0)), ' (bond-0 real rank 1, padded r=%d)' % ux.r)
y = ut3m.UNIFORM_COREWISE.retract(v)                        # a UniformTuckerTensorTrain with garbage padding
print('retracted point tt[0] left-bond padding nonzero?', bool(np.any(y.tt_supercore[0][1:] != 0)))
D = y.to_dense()
print('y.to_dense()  vs  y.apply_masks().to_dense()  relerr:', np.linalg.norm(D - y.apply_masks().to_dense()) / np.linalg.norm(D))
# Now the four ops that route through ut3_squash_tails:
z = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn((5, 7, 6), (2, 2, 2), (1, 2, 2, 1)), r=4)
def rel(a, b): return float(np.linalg.norm(a - b) / np.linalg.norm(b))
print('(y + z).to_dense()        vs y.to_dense()+z.to_dense():   relerr = %.3e' % rel((y + z).to_dense(), D + z.to_dense()))
print('(y - z).to_dense()        vs y.to_dense()-z.to_dense():   relerr = %.3e' % rel((y - z).to_dense(), D - z.to_dense()))
print('(y + y).to_dense()        vs 2*y.to_dense():              relerr = %.3e' % rel((y + y).to_dense(), 2 * D))
print('y.squash_tails().to_dense() vs y.to_dense():             relerr = %.3e' % rel(y.squash_tails().to_dense(), D))
ys = ut3.UniformTuckerTensorTrain.stack([y, y])
print('stack([y,y]).sum_stack()  vs 2*y.to_dense():              relerr = %.3e' % rel(ys.sum_stack().to_dense(), 2 * D))
print('y.norm() vs dense norm  (masks on entry -> fine):         relerr = %.3e' % rel(float(y.norm()), np.linalg.norm(D)))
# Root cause: backend ut3_squash_tails sums the RAW boundary bond (no mask):
tk2, tt2, shape, masks = ut3_ops.ut3_squash_tails(y.data)
print('backend ut3_squash_tails: new G0[0] == sum over ALL padded left-bond slots (garbage included)?',
      bool(np.allclose(tt2[0][0], y.tt_supercore[0].sum(axis=0))))
mtk, mtt = ut3_masking.ut3_apply_masks(y.data)
print('                          == sum over MASKED left-bond slots (the correct value)?',
      bool(np.allclose(tt2[0][0], mtt[0].sum(axis=0))))
# the fix in one line: squash the MASKED supercore
fixed = ut3_ops.ut3_squash_tails((mtk, mtt, shape, masks))
print('squash(masked data).to_dense() == y.to_dense()?', bool(np.allclose(ut3.UniformTuckerTensorTrain(*fixed[:3], ut3.UT3Masks(*fixed[3])).to_dense(), D)))
