"""R3: rank_adjustment_sweep 'compose both directions for guaranteed minimal ranks' -- does it hold when a
Tucker rank exceeds its mode size (n_i > N_i, a legal structure: the minimal_ranks doctest uses one)?
Unshared and shared; then re-run the shared minimal-vs-composed comparison with n_i <= N_i."""
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.backend.ranks as ranks
import t3toolbox.backend.sharing as sh
import t3toolbox.safety as safety

np.random.seed(0)
x = t3.TuckerTensorTrain.randn((13, 14, 15, 16), (4, 99, 6, 7), (1, 4, 9, 7, 1))   # the minimal_ranks doctest structure
y = x.rank_adjustment_sweep('right_to_left').rank_adjustment_sweep('left_to_right')
z = x.rank_adjustment_sweep('left_to_right').rank_adjustment_sweep('right_to_left')
print('unshared, n_1 = 99 > N_1 = 14:')
print('   minimal_ranks          :', x.minimal_ranks)
print('   R->L then L->R ranks   :', y.ranks, 'has_minimal_ranks:', y.has_minimal_ranks, 'lossless:', np.allclose(y.to_dense(), x.to_dense()))
print('   L->R then R->L ranks   :', z.ranks, 'has_minimal_ranks:', z.has_minimal_ranks)
print('   t3svd() ranks          :', x.t3svd()[0].ranks)
# does the same hold with a TT bond above its mode-size-driven bound? (n <= N everywhere)
x2 = t3.TuckerTensorTrain.randn((13, 14, 15, 16), (4, 5, 6, 7), (1, 4, 99, 7, 1))
y2 = x2.rank_adjustment_sweep('right_to_left').rank_adjustment_sweep('left_to_right')
print('unshared, r_2 = 99, all n <= N: minimal', x2.minimal_ranks, 'composed', y2.ranks, y2.has_minimal_ranks)

# random unshared sweep with n <= N: composed both directions reaches minimal?
rng = np.random.default_rng(3)
bad = 0; tot = 0
for trial in range(150):
    d = int(rng.integers(1, 5))
    shape = tuple(int(v) for v in rng.integers(2, 7, size=d))
    tk = tuple(int(rng.integers(1, shape[i] + 1)) for i in range(d))
    tt = (1,) + tuple(int(v) for v in rng.integers(1, 9, size=d - 1)) + (1,)
    x = t3.TuckerTensorTrain.randn(shape, tk, tt)
    y = x.rank_adjustment_sweep('right_to_left').rank_adjustment_sweep('left_to_right')
    tot += 1
    if y.ranks != x.minimal_ranks or not np.allclose(y.to_dense(), x.to_dense()):
        bad += 1
        if bad <= 3: print('   unshared mismatch', shape, tk, tt, 'minimal', x.minimal_ranks, 'composed', y.ranks)
print('unshared random (n <= N): composed != minimal in %d/%d' % (bad, tot))

# shared, n <= N
def random_tied(d, rng):
    while True:
        labels = tuple(int(v) for v in rng.integers(0, max(1, d - 1), size=d))
        groups = sh._groups_from_labels(labels, d)
        if any(len(g) > 1 for g in groups): break
    shape = [0] * d; tk = [0] * d
    for g in groups:
        N = int(rng.integers(2, 7)); n = int(rng.integers(1, N + 1))
        for i in g: shape[i] = N; tk[i] = n
    tt = (1,) + tuple(int(v) for v in rng.integers(1, 9, size=d - 1)) + (1,)
    x = t3.TuckerTensorTrain.randn(tuple(shape), tuple(tk), tt)
    tkc = list(x.data[0])
    for g in groups:
        for i in g[1:]: tkc[i] = tkc[g[0]]
    return t3.TuckerTensorTrain(tuple(tkc), x.data[1]), labels
bad = 0; tot = 0
for trial in range(150):
    d = int(rng.integers(2, 5))
    x, labels = random_tied(d, rng)
    with safety.unsafe():
        y = x.rank_adjustment_sweep('right_to_left', sharing=labels).rank_adjustment_sweep('left_to_right', sharing=labels)
    mn = ranks.compute_minimal_ranks(x.shape, x.tucker_ranks, x.tt_ranks, sharing=labels)
    tot += 1
    if (tuple(mn[0]), tuple(mn[1])) != y.ranks or not np.allclose(y.to_dense(), x.to_dense()):
        bad += 1
        if bad <= 3: print('   shared mismatch', x.shape, x.ranks, labels, 'minimal', mn, 'composed', y.ranks)
print('shared random (n <= N): composed != shared-minimal in %d/%d' % (bad, tot))

# shared with n > N at a GROUP mode
x = t3.TuckerTensorTrain.randn((3, 3, 5), (5, 5, 2), (1, 4, 4, 1))
tkc = list(x.data[0]); tkc[1] = tkc[0]
x = t3.TuckerTensorTrain(tuple(tkc), x.data[1])
with safety.unsafe():
    y = x.rank_adjustment_sweep('right_to_left', sharing=(0, 0, 1)).rank_adjustment_sweep('left_to_right', sharing=(0, 0, 1))
    ys, _, _ = x.t3svd(sharing=(0, 0, 1))
print('shared, group n = 5 > N = 3: minimal', ranks.compute_minimal_ranks(x.shape, x.tucker_ranks, x.tt_ranks, sharing=(0, 0, 1)),
      'composed sweeps', y.ranks, 't3svd(sharing)', ys.ranks)
