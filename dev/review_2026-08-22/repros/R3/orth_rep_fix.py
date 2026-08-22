"""R3: (1) a corrected recurrence for compute_orthogonal_representation_ranks, checked against the actual
sweep on the same 300 random structures; (2) the consequence on the public uniform path: UT3Frame.from_ut3
on a non-minimal (but perfectly valid) point is NOT orthogonal, while T3Frame.from_t3 of the same point is."""
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.frame_variations_format as fvf
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.uniform_frame_variations_format as ufvf
import t3toolbox.backend.ranks as ranks
import t3toolbox.uniform_manifold as um
import t3toolbox.manifold as mf

def corrected(shape, tk, tt):
    d = len(shape)
    up = [min(tk[i], shape[i]) for i in range(d)]
    left = list(tt); left[0] = 1
    for i in range(d - 1):                       # left sweep on the ORIGINAL bonds (after the Tucker down-orth)
        left[i + 1] = min(left[i] * up[i], tt[i + 1])
    left[d] = 1
    right = list(left); right[d] = 1             # right sweep over the LEFT-orthogonal chain
    for i in range(d - 1, 0, -1):
        right[i] = min(left[i], up[i] * right[i + 1])
    right[0] = 1
    down = [min(up[i], left[i] * right[i + 1]) for i in range(d)]
    return tuple(up), tuple(down), tuple(left), tuple(right)

def actual_frame_ranks(x):
    fr = fvf.T3Frame.from_t3(x)
    up = tuple(U.shape[-2] for U in fr.up_tucker_cores)
    down = tuple(O.shape[-2] for O in fr.down_tt_cores)
    left = tuple(P.shape[-3] for P in fr.left_tt_cores) + (fr.left_tt_cores[-1].shape[-1],)
    right = tuple(Q.shape[-3] for Q in fr.right_tt_cores) + (fr.right_tt_cores[-1].shape[-1],)
    return up, down, left, right

np.random.seed(0)
cases = [((5, 5, 5), (2, 2, 2), (1, 4, 4, 1))]
for trial in range(300):
    d = np.random.randint(1, 5)
    shape = tuple(int(v) for v in np.random.randint(2, 7, size=d))
    tk = tuple(int(v) for v in np.random.randint(1, 7, size=d))
    tt = (1,) + tuple(int(v) for v in np.random.randint(1, 9, size=d - 1)) + (1,)
    cases.append((shape, tk, tt))
bad_old = bad_new = 0
for shape, tk, tt in cases:
    x = t3.TuckerTensorTrain.randn(shape, tk, tt)
    act = actual_frame_ranks(x)
    if ranks.compute_orthogonal_representation_ranks(shape, tk, tt) != act: bad_old += 1
    if corrected(shape, tk, tt) != act: bad_new += 1
print('cases %d: current function mismatches %d ; corrected recurrence mismatches %d' % (len(cases), bad_old, bad_new))

# (2) public-path consequence
print()
for shape, tk, tt in [((5, 5, 5), (2, 2, 2), (1, 4, 4, 1)), ((13, 14, 15, 16), (4, 5, 6, 7), (1, 4, 99, 7, 1))]:
    x = t3.TuckerTensorTrain.randn(shape, tk, tt)
    rf = fvf.T3Frame.from_t3(x)
    u = ut3.UniformTuckerTensorTrain.from_t3(x)
    uf = ufvf.UT3Frame.from_ut3(u)
    print('structure', shape, tk, tt, 'has_minimal_ranks:', x.has_minimal_ranks)
    print('   ragged  T3Frame.from_t3  is_orthogonal:', bool(rf.is_orthogonal()), ' residual %.2e' % float(np.max(np.asarray(rf.orthogonality_residual))))
    print('   uniform UT3Frame.from_ut3 is_orthogonal:', bool(np.all(np.asarray(uf.is_orthogonal()))), ' residual %.2e' % float(np.max(np.asarray(uf.orthogonality_residual))))
    print('   uniform frame mask ranks (up,down,left,right):', tuple(tuple(int(v) for v in m.sum(-1)) for m in uf.data[5]))
    print('   ragged frame ranks                         :', actual_frame_ranks(x))
    # manifold-level consequence: project an ambient direction and compare with ragged
    try:
        geom = um.UNIFORM_MANIFOLD
        ufr = geom.frame(u)
        print('   UNIFORM_MANIFOLD.frame(u) is_orthogonal:', bool(np.all(np.asarray(ufr.is_orthogonal()))))
    except Exception as e:
        print('   UNIFORM_MANIFOLD.frame(u):', type(e).__name__, str(e).splitlines()[0][:120])
