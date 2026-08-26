import numpy as np, sys
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.uniform_frame_variations_format as ufvf
np.random.seed(0)
x0 = t3.TuckerTensorTrain.randn((5, 6, 7), (2, 2, 2), (1, 2, 2, 1))
x = x0.resize((5, 6, 7), (3, 3, 3), (1, 3, 3, 1))
ux = ut3.UniformTuckerTensorTrain.from_t3(x)
uf = ufvf.UT3Frame.from_ut3(ux)
res = float(np.max(np.asarray(uf.orthogonality_residual)))
down = np.asarray(uf.data[1]); masks = uf.data[5]
dm = masks[1]; rm = masks[3]
# last down core: (rL, nD, rR) with real rR = rm[-1]; mass of each mode slot in padded rR columns
O = down[-1]; pad_rR = ~np.asarray(rm[-1], dtype=bool)
mass_pad = [float(np.linalg.norm(O[:, a, :][:, pad_rR])) for a in range(O.shape[1])]
print('%s: uniform frame residual %.1e | last down core: per mode slot, |entries in PADDED rR slots| = %s (mask rank %d)'
      % (sys.argv[1], res, np.round(mass_pad, 3), int(np.asarray(dm[-1]).sum())))
