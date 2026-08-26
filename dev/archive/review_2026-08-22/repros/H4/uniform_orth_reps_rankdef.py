"""ut3_orthogonal_representations on a rank-deficient (x + x) uniform train: is the frame orthogonal?"""
import numpy as np
import t3toolbox as tb
from t3toolbox import TuckerTensorTrain as T3, T3Tangent, MANIFOLD
from t3toolbox.frame_variations_format import t3_orthogonal_representations
from t3toolbox.uniform_tucker_tensor_train import UniformTuckerTensorTrain as UT3
from t3toolbox.uniform_frame_variations_format import ut3_orthogonal_representations
from t3toolbox.uniform_manifold import UT3Tangent, UNIFORM_MANIFOLD as UM
np.random.seed(0)
for struct, stack in [(((4, 6), (2, 3), (1, 2, 1)), (2,)), (((4, 6), (1, 1), (1, 1, 1)), ()), (((4, 6), (2, 3), (1, 2, 1)), ()), (((4, 5, 6), (2, 2, 2), (1, 2, 2, 1)), ())]:
    x = T3.randn(*struct, stack_shape=stack)
    ux = UT3.from_t3(x)
    y = x + x
    uy = ux + ux
    fr, vr = t3_orthogonal_representations(y)
    fu, vu = ut3_orthogonal_representations(uy)
    print('=== struct', struct, 'stack', stack)
    print('  ragged  frame residual %.2e  ranks up=%s left=%s' % (float(np.max(fr.orthogonality_residual)), fr.up_ranks, fr.left_ranks))
    print('  uniform frame residual %.2e  ranks up=%s left=%s right=%s down=%s' % (float(np.max(fu.orthogonality_residual)), fu.up_ranks, fu.left_ranks, fu.right_ranks, fu.down_ranks))
    print('  uniform frame is_orthogonal:', fu.is_orthogonal(), ' is_consistent:', fu.is_consistent() if hasattr(fu, 'is_consistent') else 'n/a')
    print('  uniform point from frame matches y.to_dense():', np.allclose(UT3Tangent(fu, vu).to_ut3(include_shift=True).to_dense(), y.to_dense()))
    print('  uniform frame.to_dense vs ragged frame.to_dense (base point):', np.allclose(fu.to_dense(), fr.to_dense()))
    # where does the residual come from?
    import t3toolbox.backend.ufv_operations as ufv
    try:
        print('  masks: up', fu.masks.up_mask.astype(int).tolist() if fu.masks.up_mask.ndim <= 2 else fu.masks.up_mask.shape,
              'down', fu.masks.down_mask.astype(int).tolist() if fu.masks.down_mask.ndim <= 2 else fu.masks.down_mask.shape,
              'left', fu.masks.frame_left_mask.astype(int).tolist() if fu.masks.frame_left_mask.ndim <= 2 else fu.masks.frame_left_mask.shape,
              'right', fu.masks.frame_right_mask.astype(int).tolist() if fu.masks.frame_right_mask.ndim <= 2 else fu.masks.frame_right_mask.shape)
    except Exception as e:
        print('  mask print failed', e)
    # per-family residual
    U = fu.up_tucker_supercore if hasattr(fu, 'up_tucker_supercore') else None
    try:
        t = UM.project(UT3Tangent(fu, vu)); print('  UM.project ok')
    except Exception as e:
        print('  UM.project RAISED:', str(e).splitlines()[0][:100])
