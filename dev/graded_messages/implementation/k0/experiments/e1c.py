"""E1c (M4 of the review): the padding check itself fails without the masks -- the engine's, and for the tangent faces their own re-mask too.
E1b: with every engine mask removed (entry table emptied, exit mask the identity), garbage in the point's /
frame's padding changes the output of each plain transpose face (so the hygiene test's garbage clause is load-bearing)."""
import unittest.mock as mock
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.uniform_manifold as ut3m
import t3toolbox.backend.sampling_variables as sv
import t3toolbox.backend.ut3_sampling as ut3s
import t3toolbox.backend.utv_sampling as utvs
import t3toolbox.backend.ut3_masking as ut3mask
import t3toolbox.backend.ufv_masking as ufvm

np.random.seed(0)
shape = (5, 6, 4)
x = t3.TuckerTensorTrain.randn(shape, (2, 3, 2), (1, 2, 2, 1), stack_shape=(2,))
xu = ut3.UniformTuckerTensorTrain.from_t3(x, N=7, n=4, r=4)
W = (2,)
ww = [np.random.randn(*(W + (n,))) for n in shape]
index = np.stack([np.random.randint(0, n, size=W) for n in shape])
S = W + (2,)
z = [np.random.randn(*(S + (n,))) for n in shape]
c = np.random.randn(*S)
frame = ut3m.UNIFORM_MANIFOLD.frame(xu).data
def dirty_ut3(d):
    ind = ut3mask.ut3_apply_masks((np.ones_like(d[0]), np.ones_like(d[1]), d[2], d[3]))
    return (d[0] + 1e6 * (1 - ind[0]), d[1] + 1e6 * (1 - ind[1]), d[2], d[3])
def dirty_frame(f):
    ones = ufvm.ufv_apply_frame_masks((np.ones_like(f[0]), np.ones_like(f[1]), np.ones_like(f[2]), np.ones_like(f[3]), f[4], f[5]))
    return tuple(a + 1e6 * (1 - m) for a, m in zip(f[:4], ones)) + (f[4], f[5])
nomask = dict()
for kind, sample, res in (('probe', ww, z), ('apply', ww, c), ('entries', index, c)):
    for label, op, obj, dobj in (('ut3', getattr(ut3s, 'ut3_%s_corewise_transpose' % kind), xu.data, dirty_ut3(xu.data)),
                                 ('utv', getattr(utvs, 'utv_%s_transpose' % kind), frame, dirty_frame(frame))):
        clean = op(obj, sample, res)
        with_masks = op(dobj, sample, res)
        with mock.patch.object(sv, '_UNIFORM_MASKS', {}), mock.patch.object(sv, '_mask_variable', lambda n, v, s: v), mock.patch.object(utvs.ufv_masking, 'ufv_apply_variations_masks', lambda data: tuple(data[:2])):
            unmasked = op(dobj, sample, res)
        same = all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(clean[:2], with_masks[:2]))
        diff = max(float(np.abs(np.asarray(a) - np.asarray(b)).max()) for a, b in zip(clean[:2], unmasked[:2]))
        ind_out = [np.asarray(a) != 0 for a in clean[:2]]
        pad = max(float(np.abs(np.asarray(u)[np.asarray(c) == 0]).max(initial=0.0)) for u, c in zip(unmasked[:2], clean[:2]))
        print(label, kind, "padding (where the clean output is zero) without the engine masks: %.2e" % pad)
