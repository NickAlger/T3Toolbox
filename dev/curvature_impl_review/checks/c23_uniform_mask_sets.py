"""Read-only plan check (v3, n13 + M4): (a) the uniform frame's (up, frame_left) masks equal the point's plain
(tucker, tt) masks -- minimal, varying-rank C, and non-minimal points -- so masking theta_hat with the base point's
own core masks (frame_data[5] -> (um, lm)) is the same set the kinds hold; (b) ut3_apply_masks broadcasts a K+C
direction supercore against C-only masks ('...' rides K for free)."""
import sys; sys.path.insert(0, __import__("os").getcwd())
import numpy as np
import t3toolbox.tucker_tensor_train as t3, t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.backend.ufv_conversions as ufc, t3toolbox.backend.ut3_masking as um
np.random.seed(1)
cases = (('minimal', [t3.TuckerTensorTrain.randn((5, 6, 7), (2, 3, 2), (1, 2, 2, 1))]),
         ('varying-rank C=(2,)', [t3.TuckerTensorTrain.randn((5, 6, 7), (2, 3, 2), (1, 2, 2, 1)),
                                  t3.TuckerTensorTrain.randn((5, 6, 7), (1, 2, 2), (1, 1, 2, 1))]),
         ('NON-minimal Tucker slack', [t3.TuckerTensorTrain.randn((6, 6, 6), (4, 4, 4), (1, 2, 2, 1))]),
         ('NON-minimal TT slack', [t3.TuckerTensorTrain.randn((6, 6, 6), (4, 4, 4), (1, 9, 9, 1))]))
for label, xs in cases:
    uxs = [ut3.UniformTuckerTensorTrain.from_t3(x, n=4, r=3, N=7) for x in xs]
    ux = uxs[0] if len(uxs) == 1 else ut3.UniformTuckerTensorTrain.stack(uxs)
    if 'TT slack' in label:
        ux = ut3.UniformTuckerTensorTrain.from_t3(xs[0])
    data = ux.data
    fr, _ = ufc.ut3_orthogonal_representations(data)
    upm, dnm, lfm, rtm = fr[5]
    tkm, ttm = data[3]
    print('%-26s frame up == point tucker mask: %s ; frame left == point tt mask: %s ; frame masks SUBSET of point masks: %s' % (
        label, np.array_equal(upm, tkm), np.array_equal(lfm, ttm), bool(np.all(~upm | tkm) and np.all(~lfm | ttm))))
x = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn((5, 6, 7), (2, 3, 2), (1, 2, 2, 1), stack_shape=(2,)), n=4, r=3, N=7)
tk, tt, sh, ms = x.data
tkd = np.random.randn(*((tk.shape[0], 3) + tk.shape[1:])); ttd = np.random.randn(*((tt.shape[0], 3) + tt.shape[1:]))
mtk, mtt = um.ut3_apply_masks((tkd, ttd, sh, ms))
ref = um.ut3_apply_masks((tkd[:, 1], ttd[:, 1], sh, ms))
print('K+C direction supercore vs C-only masks:', mtk.shape == tkd.shape and np.allclose(mtk[:, 1], ref[0]) and np.allclose(mtt[:, 1], ref[1]))
