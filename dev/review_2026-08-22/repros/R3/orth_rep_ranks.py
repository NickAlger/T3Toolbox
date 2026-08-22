"""R3: compute_orthogonal_representation_ranks vs the ranks the actual sweep produces
(ragged T3Frame.from_t3), and whether the uniform frame (UT3Frame.from_ut3), whose masks are built
from that prediction, still represents the same tensor / same tangent map on NON-minimal inputs."""
import numpy as np, itertools
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.frame_variations_format as fvf
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.uniform_frame_variations_format as ufvf
import t3toolbox.backend.ranks as ranks
import t3toolbox.backend.fv_conversions as fvc

np.random.seed(0)

def actual_frame_ranks(x):
    fr = fvf.T3Frame.from_t3(x)
    up = tuple(U.shape[-2] for U in fr.up_tucker_cores)
    down = tuple(O.shape[-2] for O in fr.down_tt_cores)
    left = tuple(P.shape[-3] for P in fr.left_tt_cores) + (fr.left_tt_cores[-1].shape[-1],)
    right = tuple(Q.shape[-3] for Q in fr.right_tt_cores) + (fr.right_tt_cores[-1].shape[-1],)
    return up, down, left, right

bad = []
n_cases = 0
cases = [
    ((5, 5, 5), (2, 2, 2), (1, 4, 4, 1)),          # hand case from the analysis
    ((5, 6, 7), (3, 6, 2), (1, 2, 2, 1)),          # the docs' orphan example after t3svd(max_tt=2)
    ((4, 6, 5, 7), (3, 5, 4, 6), (1, 3, 9, 6, 1)),
]
for trial in range(300):
    d = np.random.randint(1, 5)
    shape = tuple(np.random.randint(2, 7, size=d))
    tk = tuple(np.random.randint(1, 7, size=d))
    tt = (1,) + tuple(np.random.randint(1, 9, size=d - 1)) + (1,)
    cases.append((shape, tk, tt))

for shape, tk, tt in cases:
    x = t3.TuckerTensorTrain.randn(shape, tk, tt)
    pred = ranks.compute_orthogonal_representation_ranks(shape, tk, tt)
    act = actual_frame_ranks(x)
    n_cases += 1
    if pred != act:
        bad.append((shape, tk, tt, pred, act))
print('compute_orthogonal_representation_ranks vs actual T3Frame ranks: %d/%d mismatches' % (len(bad), n_cases))
for b in bad[:6]:
    shape, tk, tt, pred, act = b
    print('  structure', shape, tk, tt)
    print('    predicted (up,down,left,right) =', pred)
    print('    actual    (up,down,left,right) =', act)

# ---- does the mismatch matter for the UNIFORM frame? (masks are built from the prediction) ----
print()
print('uniform frame on a non-minimal point:')
for shape, tk, tt in [((5, 5, 5), (2, 2, 2), (1, 4, 4, 1)), ((4, 6, 5, 7), (3, 5, 4, 6), (1, 3, 9, 6, 1))]:
    x = t3.TuckerTensorTrain.randn(shape, tk, tt)
    pred = ranks.compute_orthogonal_representation_ranks(shape, tk, tt)
    act = actual_frame_ranks(x)
    ux = ut3.UniformTuckerTensorTrain.from_t3(x)
    uf = ufvf.UT3Frame.from_ut3(ux)
    um = uf.data[5]
    mask_ranks = tuple(tuple(int(v) for v in m.sum(axis=-1)) for m in um)
    rf = fvf.T3Frame.from_t3(x)
    print('  structure', shape, tk, tt, 'minimal:', x.has_minimal_ranks)
    print('    predicted', pred)
    print('    actual   ', act)
    print('    uniform frame mask ranks', mask_ranks)
    # does the uniform frame still represent the same tensor?
    try:
        rf_u = uf.to_t3frame()
        xu = rf_u.to_t3()
        print('    uniform frame -> ragged frame -> to_t3 == x ?', bool(np.allclose(xu.to_dense(), x.to_dense())))
        print('    ragged frame -> to_t3 == x ?', bool(np.allclose(rf.to_t3().to_dense(), x.to_dense())))
        print('    orthogonality residual of converted frame:', float(np.max(rf_u.orthogonality_residual)) if hasattr(rf_u, 'orthogonality_residual') else 'n/a')
    except Exception as e:
        print('    to_t3frame failed:', type(e).__name__, e)
