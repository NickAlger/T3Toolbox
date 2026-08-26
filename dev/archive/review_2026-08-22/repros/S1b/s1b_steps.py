"""S1b: where does the uniform sweep lose the orthonormal completion on a numerically rank-deficient train?
Run the ragged and the uniform pipeline step by step on the same zero-padded resize() input."""
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.backend.t3_orthogonalization as ro
import t3toolbox.backend.ut3_orthogonalization as uo
import t3toolbox.backend.ut3_masking as um
import t3toolbox.backend.tt_orthogonalization as orth
import t3toolbox.backend.tt_operations as tto
np.random.seed(0)
x = t3.TuckerTensorTrain.randn((6, 7, 8), (2, 2, 2), (1, 2, 2, 1)).resize((6, 7, 8), (3, 3, 3), (1, 3, 3, 1))
ux = ut3.UniformTuckerTensorTrain.from_t3(x)
print('ranks', x.tucker_ranks, x.tt_ranks, '| uniform dims n=%d r=%d N=%d' % (ux.n, ux.r, ux.N))

def gram_dev(U):   # rows orthonormal?  U: (n, N)
    return float(np.max(np.abs(U @ U.T - np.eye(U.shape[0]))))
def left_dev(G):   # (rL, n, rR): left unfolding columns orthonormal?
    M = G.reshape(-1, G.shape[-1]); return float(np.max(np.abs(M.T @ M - np.eye(M.shape[1]))))

# ---- step 1: Tucker down-orthogonalization
rU, rG = ro.t3_down_orthogonalize_tucker_cores(x.data)
mtk, mtt = um.ut3_apply_masks(ux.data)
uU, uG = uo.down_orthogonalize_tucker_supercores(mtk, mtt)
print('step1 Tucker down-orth: ragged U gram dev', [round(gram_dev(np.asarray(U)), 2) for U in rU],
      '| uniform U gram dev', [round(gram_dev(np.asarray(uU[i])), 2) for i in range(3)])
print('   uniform U zero rows?', [int(np.sum(np.all(np.abs(uU[i]) < 1e-14, axis=-1))) for i in range(3)],
      ' ragged U zero rows?', [int(np.sum(np.all(np.abs(np.asarray(U)) < 1e-14, axis=-1))) for U in rU])
# ---- step 2: left sweep
rL = orth.tt_left_orthogonalize(rG)
uL = orth.tt_left_orthogonalize(uG)
print('step2 left sweep: ragged left dev', [round(left_dev(np.asarray(G)), 2) for G in rL[:-1]],
      '| uniform left dev', [round(left_dev(np.asarray(uL[i])), 2) for i in range(2)])
print('   uniform L zero bond-slices?', [int(np.sum(np.all(np.abs(uL[i]).reshape(-1, uL.shape[-1]) < 1e-14, axis=0))) for i in range(3)])
# ---- step 3: right sweep over the left chain
rR, rH = orth.tt_right_orthogonalize(rL, return_variation_cores=True)
uR, uH = orth.tt_right_orthogonalize(uL, return_variation_cores=True)
def right_dev(G):
    M = G.reshape(G.shape[0], -1); return float(np.max(np.abs(M @ M.T - np.eye(M.shape[0]))))
print('step3 right sweep: ragged right dev', [round(right_dev(np.asarray(G)), 2) for G in rR[1:]],
      '| uniform right dev', [round(right_dev(np.asarray(uR[i])), 2) for i in range(1, 3)])
# ---- step 4: down step (TT up-orthogonalization -> down cores)
rV, rO = ro.t3_up_orthogonalize_tt_cores((rU, rH))
uV, uO = uo.up_orthogonalize_tt_supercores(uU, uH)
def down_dev(O):  # (rL, nD, rR): mode index orthonormal over the bonds
    M = np.moveaxis(O, 1, 0).reshape(O.shape[1], -1); return float(np.max(np.abs(M @ M.T - np.eye(M.shape[0]))))
print('step4 down step: ragged O dev', [round(down_dev(np.asarray(O)), 2) for O in rO],
      '| uniform O dev', [round(down_dev(np.asarray(uO[i])), 2) for i in range(3)])
print('   ragged O shapes', [tuple(O.shape) for O in rO], ' uniform O shape', uO.shape)
