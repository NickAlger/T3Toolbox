"""R3: shared rank bookkeeping vs what the grouped sweeps actually produce.
 (1) compute_raw_sweep_ranks(sharing) vs t3svd(sharing, caps) output ranks (random structures/caps)
 (2) compute_minimal_ranks(sharing) vs rank_adjustment_sweep(sharing) composed both directions (lossless)
 (3) single L->R grouped sweep on a RIGHT-orthogonal input vs compute_minimal_ranks(sharing): the
     assumption behind uniform _reduce_left_to_right (masks = minimal) -> lossy if they differ
 (4) uniform rank_adjustment_sweep('right_to_left', sharing) on a ut3svd(sharing) output: lossless?
 (5) dense ground truth: group-concatenated matricization rank vs compute_minimal_ranks(sharing)."""
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.backend.ranks as ranks
import t3toolbox.backend.t3_svd as bsvd
import t3toolbox.backend.ut3_svd as busvd
import t3toolbox.backend.sharing as sharing_mod
import t3toolbox.safety as safety

np.random.seed(1)

def random_tied(d, rng):
    # random sharing spec with at least one real group; equal sizes + equal tucker ranks within groups
    while True:
        labels = tuple(rng.integers(0, max(1, d - 1), size=d))
        groups = sharing_mod._groups_from_labels(labels, d)
        if any(len(g) > 1 for g in groups):
            break
    shape = [0] * d; tk = [0] * d
    for g in groups:
        N = int(rng.integers(2, 7)); n = int(rng.integers(1, 7))
        for i in g:
            shape[i] = N; tk[i] = n
    tt = (1,) + tuple(int(v) for v in rng.integers(1, 9, size=d - 1)) + (1,)
    x = t3.TuckerTensorTrain.randn(tuple(shape), tuple(tk), tt)
    tkc, ttc = x.data
    tkc = list(tkc)
    for g in groups:
        for i in g[1:]:
            tkc[i] = tkc[g[0]]
    return t3.TuckerTensorTrain(tuple(tkc), ttc), labels, groups

rng = np.random.default_rng(2)
mism_raw = []; mism_min = []; mism_single = []; lossy_uniform = []; mism_dense = []
n = 0
for trial in range(200):
    d = int(rng.integers(2, 5))
    x, labels, groups = random_tied(d, rng)
    shape, tk, tt = x.shape, x.tucker_ranks, x.tt_ranks
    n += 1
    # (1) raw sweep ranks under random caps
    cap_tk = [int(rng.integers(1, 8)) for _ in range(d)]
    for g in groups:
        for i in g[1:]:
            cap_tk[i] = cap_tk[g[0]]
    cap_tt = (1,) + tuple(int(v) for v in rng.integers(1, 9, size=d - 1)) + (1,)
    with safety.unsafe():
        y, _, _ = x.t3svd(max_tucker_ranks=tuple(cap_tk), max_tt_ranks=cap_tt, sharing=labels)
    pred = ranks.compute_raw_sweep_ranks(shape, tk, tt, tuple(min(a, b) for a, b in zip(tk, cap_tk)),
                                         tuple(min(a, b) for a, b in zip(tt, cap_tt)), sharing=labels)
    if (tuple(pred[0]), tuple(pred[1])) != (y.tucker_ranks, y.tt_ranks):
        mism_raw.append((shape, tk, tt, labels, cap_tk, cap_tt, pred, (y.tucker_ranks, y.tt_ranks)))
    # (2) minimal ranks vs composed lossless sweeps
    with safety.unsafe():
        z = x.rank_adjustment_sweep('right_to_left', sharing=labels).rank_adjustment_sweep('left_to_right', sharing=labels)
    mn = ranks.compute_minimal_ranks(shape, tk, tt, sharing=labels)
    if (tuple(mn[0]), tuple(mn[1])) != (z.tucker_ranks, z.tt_ranks) or not np.allclose(z.to_dense(), x.to_dense()):
        mism_min.append((shape, tk, tt, labels, mn, (z.tucker_ranks, z.tt_ranks)))
    # (3) single L->R grouped sweep on a right-orthogonal input
    xr = x.down_orthogonalize_tucker_cores().right_orthogonalize_tt_cores()
    with safety.unsafe():
        w = xr.rank_adjustment_sweep('left_to_right', sharing=labels)
    if (tuple(mn[0]), tuple(mn[1])) != (w.tucker_ranks, w.tt_ranks):
        mism_single.append((shape, tk, tt, labels, mn, (w.tucker_ranks, w.tt_ranks)))
    # (5) dense ground truth of the group ranks
    T = np.asarray(x.to_dense())
    dense_tk = []
    for g in groups:
        mats = [np.moveaxis(T, i, 0).reshape(T.shape[i], -1) for i in g]
        dense_tk.append(np.linalg.matrix_rank(np.concatenate(mats, axis=1), tol=1e-9))
    dense_tt = [1]
    for k in range(1, d):
        dense_tt.append(np.linalg.matrix_rank(T.reshape(int(np.prod(shape[:k])), -1), tol=1e-9))
    dense_tt.append(1)
    exp_tk = [0] * d
    for gi, g in enumerate(groups):
        for i in g:
            exp_tk[i] = int(dense_tk[gi])
    if (tuple(exp_tk), tuple(int(v) for v in dense_tt)) != (tuple(mn[0]), tuple(mn[1])):
        mism_dense.append((shape, tk, tt, labels, mn, (tuple(exp_tk), tuple(dense_tt))))

print('cases:', n)
print('(1) compute_raw_sweep_ranks(sharing) != t3svd(sharing, caps) ranks: %d' % len(mism_raw))
for m in mism_raw[:5]:
    print('   ', m)
print('(2) compute_minimal_ranks(sharing) != composed sweeps (or lossy): %d' % len(mism_min))
for m in mism_min[:5]:
    print('   ', m)
print('(3) single L->R grouped sweep on right-orth input != minimal: %d' % len(mism_single))
for m in mism_single[:5]:
    print('   ', m)
print('(5) dense group/TT edge-cut ranks != compute_minimal_ranks(sharing): %d' % len(mism_dense))
for m in mism_dense[:5]:
    print('   ', m)

# (4) uniform: ut3svd(sharing) output (left-orth) -> rank_adjustment_sweep('right_to_left', sharing)
print()
print('(4) uniform shared rank_adjustment_sweep on ut3svd(sharing) outputs (documented path):')
rng = np.random.default_rng(5)
lossy = 0; tot = 0; rank_mism = 0
for trial in range(60):
    d = int(rng.integers(2, 5))
    x, labels, groups = random_tied(d, rng)
    with safety.unsafe():
        y, _, _ = x.t3svd(sharing=labels)             # lossless grouped svd -> left-orth, tied
        u = ut3.UniformTuckerTensorTrain.from_t3(y)
        v = u.rank_adjustment_sweep('right_to_left', sharing=labels)
        yr = y.rank_adjustment_sweep('right_to_left', sharing=labels)
    tot += 1
    ok = np.allclose(np.asarray(v.to_dense()), np.asarray(y.to_dense()), atol=1e-8)
    if not ok:
        lossy += 1
        print('   LOSSY:', x.shape, x.tucker_ranks, x.tt_ranks, labels, 'y ranks', y.tucker_ranks, y.tt_ranks,
              'ragged R->L ranks', yr.tucker_ranks, yr.tt_ranks,
              'uniform mask ranks', tuple(int(s) for s in v.data[3][0].sum(-1)), tuple(int(s) for s in v.data[3][1].sum(-1)),
              'rel err %.2e' % (np.linalg.norm(np.asarray(v.to_dense()) - np.asarray(y.to_dense())) / np.linalg.norm(np.asarray(y.to_dense()))))
    if (tuple(int(s) for s in v.data[3][0].sum(-1)), tuple(int(s) for s in v.data[3][1].sum(-1))) != (yr.tucker_ranks, yr.tt_ranks):
        rank_mism += 1
print('   lossy: %d/%d ; mask ranks != ragged R->L ranks: %d' % (lossy, tot, rank_mism))
