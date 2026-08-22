"""d = 1 (single-mode) uniform T3: which ops crash? (ragged supports d=1 throughout)."""
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.uniform_frame_variations_format as ubv
import t3toolbox.backend.tt_operations as tt_ops
np.random.seed(0)
x = t3.TuckerTensorTrain.randn((5,), (3,), (1, 1))
y = t3.TuckerTensorTrain.randn((5,), (2,), (1, 1))
ux = ut3.UniformTuckerTensorTrain.from_t3(x); uy = ut3.UniformTuckerTensorTrain.from_t3(y)
print('backend _tt_squash_tails_uniform on a d=1 supercore: in shape', ux.tt_supercore.shape, '-> out shape', tt_ops.tt_squash_tails(ux.tt_supercore).shape, '(d doubled!)')
ops = {
    'to_dense': lambda: np.allclose(ux.to_dense(), x.to_dense()),
    'squash_tails': lambda: ux.squash_tails(),
    'x + y': lambda: np.allclose((ux + uy).to_dense(), (x + y).to_dense()),
    'x - y': lambda: (ux - uy),
    'norm()': lambda: abs(float(ux.norm()) - float(x.norm())) < 1e-9,
    'norm(use_orthogonalization=False)': lambda: abs(float(ux.norm(use_orthogonalization=False)) - float(x.norm())) < 1e-9,
    'inner(y)': lambda: abs(float(ux.inner(uy)) - float(x.inner(y))) < 1e-9,
    't3svd()': lambda: ux.t3svd(),
    'rank_adjustment_sweep': lambda: ux.rank_adjustment_sweep(),
    'left_orthogonalize_tt_cores': lambda: ux.left_orthogonalize_tt_cores(),
    'right_orthogonalize_tt_cores': lambda: ux.right_orthogonalize_tt_cores(),
    'down_orthogonalize_tucker_cores': lambda: ux.down_orthogonalize_tucker_cores(),
    'up_orthogonalize_tt_cores': lambda: ux.up_orthogonalize_tt_cores(),
    'is_left_orthogonal': lambda: ux.is_left_orthogonal(),
    'entries': lambda: np.allclose(ux.entries(np.array([2])), x.to_dense()[2]),
    'apply': lambda: ux.apply([np.random.randn(5)]),
    'probe': lambda: ux.probe([np.random.randn(5)]),
    'sum()': lambda: abs(float(ux.sum()) - float(x.to_dense().sum())) < 1e-9,
    'ut3_orthogonal_representations': lambda: ubv.ut3_orthogonal_representations(ux),
    'stack/sum_stack': lambda: ut3.UniformTuckerTensorTrain.stack([ux, ux]).sum_stack(),
    'ragged x.squash_tails (d=1)': lambda: x.squash_tails() if hasattr(x, 'squash_tails') else 'n/a',
    'ragged x+y (d=1)': lambda: (x + y).to_dense().shape,
    'ragged x.t3svd (d=1)': lambda: x.t3svd()[0],
    'ragged x.norm (d=1)': lambda: x.norm(),
}
for name, fn in ops.items():
    try:
        r = fn()
        print('%-38s OK   %s' % (name, r if isinstance(r, (bool, str, tuple, float)) else type(r).__name__))
    except Exception as e:
        print('%-38s CRASH %s: %s' % (name, type(e).__name__, str(e)[:140]))
