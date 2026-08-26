"""Default of sum_over_probes differs between tv_*_transpose_from_sweep (False) and utv_* twins (True)."""
import numpy as np, t3toolbox as t3t
import t3toolbox.backend.apply as bapply, t3toolbox.backend.probing as bprob, t3toolbox.backend.utv_sampling as utvs
import t3toolbox.manifold as t3m, t3toolbox.uniform_manifold as ut3m, t3toolbox.uniform_tucker_tensor_train as ut3
np.random.seed(0)
shape = (5, 6, 7)
x = t3t.TuckerTensorTrain.randn(shape, (2, 3, 2), (1, 2, 2, 1))
frame = t3m.MANIFOLD.frame(x)
ww = [np.random.randn(4, N) for N in shape]
c = np.random.randn(4)
sweep = bapply.tv_precompute_apply_frame_sweep(frame.data, ww)
dU, dG = bapply.tv_apply_transpose_from_sweep(c, ww, frame.data, sweep)
print('ragged  tv_apply_transpose_from_sweep (defaults): dU[0].shape =', dU[0].shape)
ux = ut3.UniformTuckerTensorTrain.from_t3(x)
uframe = ut3m.UNIFORM_MANIFOLD.frame(ux)
usweep = utvs.utv_precompute_apply_frame_sweep(uframe.data, ww)
udU, udG = utvs.utv_apply_transpose_from_sweep(c, usweep)
print('uniform utv_apply_transpose_from_sweep (defaults): dU.shape    =', udU.shape, ' (d leads)')
# probe
zt = [np.random.randn(4, N) for N in shape]
psweep = bprob.tv_precompute_probe_frame_sweep(frame.data, ww)
pU, pG = bprob.tv_probe_transpose_from_sweep(zt, ww, frame.data, psweep)
print('ragged  tv_probe_transpose_from_sweep (defaults): dU[0].shape =', pU[0].shape)
upsweep = utvs.utv_precompute_probe_frame_sweep(uframe.data, ww)
upU, upG = utvs.utv_probe_transpose_from_sweep(zt, upsweep)
print('uniform utv_probe_transpose_from_sweep (defaults): dU.shape    =', upU.shape)
import inspect
for f in (bapply.tv_apply_transpose_from_sweep, utvs.utv_apply_transpose_from_sweep,
          bprob.tv_probe_transpose_from_sweep, utvs.utv_probe_transpose_from_sweep):
    print(f.__name__, 'sum_over_probes default =', inspect.signature(f).parameters['sum_over_probes'].default)
