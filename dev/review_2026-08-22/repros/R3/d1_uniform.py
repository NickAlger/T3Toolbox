"""R3: d=1 on the uniform SVD layer -- which entry points fail?"""
import traceback
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.uniform_frame_variations_format as ufvf
import t3toolbox.backend.ut3_svd as busvd

np.random.seed(0)
x = t3.TuckerTensorTrain.randn((6,), (3,), (1, 1))
print('ragged d=1 t3svd ranks:', x.t3svd()[0].ranks, '; rank_adjustment_sweep:', x.rank_adjustment_sweep().ranks)
u = ut3.UniformTuckerTensorTrain.from_t3(x)
for name, f in [
    ('u.t3svd()', lambda: u.t3svd()),
    ('u.t3svd(max_tucker_ranks=2)', lambda: u.t3svd(max_tucker_ranks=2)),
    ('u.t3svd(assume_orthogonal=True)', lambda: u.t3svd(assume_orthogonal=True)),
    ('u.rank_adjustment_sweep()', lambda: u.rank_adjustment_sweep()),
    ("u.rank_adjustment_sweep('left_to_right')", lambda: u.rank_adjustment_sweep('left_to_right')),
    ('UT3Frame.from_ut3(u)', lambda: ufvf.UT3Frame.from_ut3(u)),
    ('u.to_dense()', lambda: u.to_dense()),
    ('u.is_left_orthogonal()', lambda: u.is_left_orthogonal()),
]:
    try:
        r = f()
        print('OK  ', name)
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)[-1]
        print('FAIL', name, '->', type(e).__name__, str(e)[:110].replace('\n', ' '), '@', tb.filename.split('/')[-1], tb.lineno)

# the same at d=1 with a stack and padding
xs = t3.TuckerTensorTrain.randn((6,), (3,), (1, 1), stack_shape=(2,))
us = ut3.UniformTuckerTensorTrain.from_t3(xs, n=4, r=2)
for name, f in [('stacked/padded u.t3svd()', lambda: us.t3svd()), ('stacked/padded sweep', lambda: us.rank_adjustment_sweep())]:
    try:
        f(); print('OK  ', name)
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)[-1]
        print('FAIL', name, '->', type(e).__name__, str(e)[:110].replace('\n', ' '), '@', tb.filename.split('/')[-1], tb.lineno)
