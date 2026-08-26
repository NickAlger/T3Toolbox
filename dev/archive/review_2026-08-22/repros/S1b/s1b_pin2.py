import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.uniform_frame_variations_format as ufvf
import t3toolbox.backend.ufv_masking as ufm
np.random.seed(0)
x = t3.TuckerTensorTrain.randn((5, 6, 7), (2, 2, 2), (1, 2, 2, 1)).resize((5, 6, 7), (3, 3, 3), (1, 3, 3, 1))
uf = ufvf.UT3Frame.from_ut3(ut3.UniformTuckerTensorTrain.from_t3(x))
um_, dm, lm, rm = uf.data[5]
mup, mdown, mleft, mright = ufm.ufv_apply_frame_masks(uf.data)
print('masks: up', um_.sum(-1), 'down', dm.sum(-1), 'left', lm.sum(-1), 'right', rm.sum(-1))
for i in range(3):
    G = np.einsum('io,jo->ij', mup[i], mup[i]);  print('up   %d masked gram diag' % i, np.round(np.diag(G), 3))
    G = np.einsum('iaj,ibj->ab', mdown[i], mdown[i]); print('down %d masked gram diag' % i, np.round(np.diag(G), 3))
for i in range(2):
    G = np.einsum('iaj,iak->jk', mleft[i], mleft[i]); print('left %d masked gram diag (outgoing bond)' % i, np.round(np.diag(G), 3))
for i in range(1, 3):
    G = np.einsum('iaj,kaj->ik', mright[i], mright[i]); print('right %d masked gram diag (incoming bond)' % i, np.round(np.diag(G), 3))
# where is the content of the offending core? compare masked vs unmasked norms per slot
raw_down = np.asarray(uf.data[1])
for i in range(3):
    print('down %d: |raw| per mode-slot' % i, np.round(np.linalg.norm(raw_down[i], axis=(0, 2)), 3), ' |masked| per mode-slot', np.round(np.linalg.norm(np.asarray(mdown[i]), axis=(0, 2)), 3))
