"""R3: sharing-partition normalization edges: labels vs groups, all-singletons, one group for all modes,
non-contiguous groups through the R->L reversal (ragged + uniform), groups_to_labels round trip."""
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.backend.sharing as sh
import t3toolbox.backend.t3_svd as bsvd
import t3toolbox.backend.ranks as ranks
import t3toolbox.safety as safety

print('validate_sharing with a GROUPS tuple where labels are expected:')
for spec, shape in [(((0, 1), (2,)), (5, 5, 5)), (((0,), (1,)), (5, 5)), (((0, 1),), (5, 5)), (((0, 2), (1, 3)), (5, 5, 5, 5))]:
    try:
        print('   ', spec, '->', sh.validate_sharing(spec, shape))
    except Exception as e:
        print('   ', spec, '-> raises', type(e).__name__, str(e)[:80])
print('groups_to_labels round trip on non-contiguous:', sh.groups_to_labels(((0, 2), (1, 3))),
      sh.validate_sharing(sh.groups_to_labels(((0, 2), (1, 3))), (4, 4, 4, 4)))
print('canonical_groups all-singletons ->', sh.canonical_groups((0, 1, 2), (4, 5, 6)), '; None ->', sh.canonical_groups(None, (4,)))
print('_reversed_groups(((0,2),(1,),(3,)), 4) =', bsvd._reversed_groups(((0, 2), (1,), (3,)), 4))

np.random.seed(0)
def tied(shape, tk, tt, labels):
    x = t3.TuckerTensorTrain.randn(shape, tk, tt)
    groups = sh.validate_sharing(labels, shape)
    tkc = list(x.data[0])
    for g in groups:
        for i in g[1:]:
            tkc[i] = tkc[g[0]]
    return t3.TuckerTensorTrain(tuple(tkc), x.data[1])

cases = [
    ('non-contiguous (0,2)', (5, 6, 5, 4), (3, 4, 3, 2), (1, 3, 4, 2, 1), ('a', 'b', 'a', 'c')),
    ('non-contiguous (0,3),(1,2)', (5, 4, 4, 5), (3, 2, 2, 3), (1, 3, 5, 2, 1), (0, 1, 1, 0)),
    ('single group all modes', (5, 5, 5), (4, 4, 4), (1, 3, 3, 1), (7, 7, 7)),
    ('all singletons', (5, 6, 7), (4, 4, 4), (1, 3, 3, 1), (0, 1, 2)),
    ('non-contiguous d=4 caps', (6, 5, 6, 5), (5, 4, 5, 4), (1, 6, 6, 4, 1), (0, 1, 0, 1)),
]
for name, shape, tk, tt, labels in cases:
    x = tied(shape, tk, tt, labels)
    with safety.unsafe():
        y, sk, st = x.t3svd(sharing=labels)
        yl = y.rank_adjustment_sweep('right_to_left', sharing=labels)
        yr = x.rank_adjustment_sweep('right_to_left', sharing=labels).rank_adjustment_sweep('left_to_right', sharing=labels)
        u = ut3.UniformTuckerTensorTrain.from_t3(y)
        ur = u.rank_adjustment_sweep('right_to_left', sharing=labels)
    groups = sh.validate_sharing(labels, shape)
    tied_ok = all(y.data[0][i] is y.data[0][g[0]] for g in groups for i in g) and all(yl.data[0][i] is yl.data[0][g[0]] for g in groups for i in g)
    spec_ok = all(np.array_equal(sk[i], sk[g[0]]) for g in groups for i in g)
    mn = ranks.compute_minimal_ranks(shape, tk, tt, sharing=labels)
    print('%-28s lossless t3svd:%s  R->L lossless:%s tied:%s spectra-equal:%s  R->L right-orth:%s  ranks after R->L %s %s  minimal %s %s  composed %s %s  uniform R->L lossless:%s masks %s %s' % (
        name, np.allclose(y.to_dense(), x.to_dense()), np.allclose(yl.to_dense(), x.to_dense()), tied_ok, spec_ok,
        yl.is_right_orthogonal(), yl.tucker_ranks, yl.tt_ranks, mn[0], mn[1], yr.tucker_ranks, yr.tt_ranks,
        np.allclose(np.asarray(ur.to_dense()), y.to_dense()),
        tuple(int(v) for v in ur.data[3][0].sum(-1)), tuple(int(v) for v in ur.data[3][1].sum(-1))))
