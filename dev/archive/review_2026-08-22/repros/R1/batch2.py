import numpy as np, os, itertools
import t3toolbox.tucker_tensor_train as t3
from t3toolbox.tucker_tensor_train import TuckerTensorTrain as T3, T3Weights, t3_absorb_weights, t3_weighted_norm, t3_weighted_inner
import t3toolbox.corewise as cw
import t3toolbox.backend.sharing as sh
import t3toolbox.safety as safety
np.random.seed(0)
def sec(n): print('\n### '+n)
def tryit(label, f):
    try: print(f'  [{label}] OK ->', f())
    except Exception as e: print(f'  [{label}] RAISED {type(e).__name__}: {str(e)[:160]}')

sec('M sharing at asymmetric shapes (d=4, non-adjacent group, distinct ranks)')
# shape (5,7,5,6), sharing (0,1,0,2): modes 0 and 2 share (size 5), tucker ranks must be equal within group
def tied_point(stack=()):
    x = T3.randn((5,7,5,6), (3,4,3,2), (1,3,4,2,1), stack_shape=stack)
    return T3(*sh.t3_tie_tucker_factors(x.data, (0,1,0,2)))
x = tied_point()
print('  tied?', x.has_shared_tucker_factors((0,1,0,2)))
y, sst, ssT = x.t3svd(sharing=(0,1,0,2))
print('  t3svd lossless err', np.linalg.norm(y.to_dense()-x.to_dense())/np.linalg.norm(x.to_dense()), 'ranks', y.ranks, 'tied', y.data[0][0] is y.data[0][2], 'svals equal', np.array_equal(sst[0], sst[2]), 'left-orth', y.is_left_orthogonal())
yt, _, _ = x.t3svd(sharing=(0,1,0,2), max_tucker_ranks=(2,4,2,2))
print('  truncated grouped ranks', yt.ranks, 'tied', yt.data[0][0] is yt.data[0][2], 'relerr', np.linalg.norm(yt.to_dense()-x.to_dense())/np.linalg.norm(x.to_dense()))
# rank adjustment with sharing
z2, _, _ = x.t3svd(sharing=(0,1,0,2), max_tt_ranks=(1,1,4,2,1))
z3 = z2.rank_adjustment_sweep('right_to_left', sharing=(0,1,0,2))
print('  ras: ranks', z2.ranks, '->', z3.ranks, 'tied', z3.data[0][0] is z3.data[0][2], 'lossless', np.allclose(z3.to_dense(), z2.to_dense()), 'minimal(sharing)', T3.get_minimal_ranks(z3.shape, *z3.ranks, sharing=(0,1,0,2)) == z3.ranks)
# share from untied
xu = T3.randn((5,7,5,6), (3,4,3,2), (1,3,4,2,1))
xsh = xu.share((0,1,0,2))
print('  share lossless-ish? ranks', xsh.ranks, 'tied', xsh.data[0][0] is xsh.data[0][2])
xsh2 = xu.share((0,1,0,2), max_tucker_ranks=(3,4,3,2), max_tt_ranks=(1,3,4,2,1))
print('  share capped ranks', xsh2.ranks, 'relerr', np.linalg.norm(xsh2.to_dense()-xu.to_dense())/np.linalg.norm(xu.to_dense()))
# resize with sharing
xr = x.resize(x.shape, (4,5,4,3), (1,4,5,3,1), sharing=(0,1,0,2))
print('  resize sharing: ranks', xr.ranks, 'tied', xr.data[0][0] is xr.data[0][2], 'same tensor', np.allclose(xr.to_dense(), x.to_dense()))
tryit('resize sharing on untied (safe)', lambda: xu.resize(xu.shape, (4,5,4,3), (1,4,5,3,1), sharing=(0,1,0,2)).ranks)
# shrink with sharing
xr2 = x.resize(x.shape, (2,4,2,2), (1,3,4,2,1), sharing=(0,1,0,2))
print('  resize shrink sharing ranks', xr2.ranks, 'tied', xr2.data[0][0] is xr2.data[0][2])
# continuation ranks with sharing
tryit('continuation_ranks sharing', lambda: x.continuation_ranks(sharing=(0,1,0,2)))
tryit('continuation_ranks sharing max_grow=1 tau=1', lambda: x.continuation_ranks(sharing=(0,1,0,2), tau=1.0, max_grow=1))
# stacked t3svd with sharing + max ranks
xs = tied_point(stack=(2,))
tryit('stacked t3svd sharing max ranks', lambda: (lambda r: (r[0].ranks, r[0].stack_shape, bool(np.allclose(r[0].to_dense(), xs.to_dense()))))(xs.t3svd(sharing=(0,1,0,2))))
tryit('stacked t3svd sharing truncated', lambda: (lambda r: (r[0].ranks, r[0].stack_shape, r[0].data[0][0] is r[0].data[0][2]))(xs.t3svd(sharing=(0,1,0,2), max_tucker_ranks=(2,4,2,2))))
tryit('stacked rank_adjustment_sweep sharing', lambda: xs.t3svd(sharing=(0,1,0,2), max_tt_ranks=(1,1,4,2,1))[0].rank_adjustment_sweep('right_to_left', sharing=(0,1,0,2)).ranks)
tryit('stacked resize sharing', lambda: xs.resize(xs.shape, (4,5,4,3), (1,4,5,3,1), sharing=(0,1,0,2)).structure)
tryit('stacked has_shared_tucker_factors', lambda: xs.has_shared_tucker_factors((0,1,0,2)))
# bad sharing specs
tryit('t3svd sharing wrong len', lambda: x.t3svd(sharing=(0,1,0)))
tryit('t3svd sharing unequal sizes', lambda: x.t3svd(sharing=(0,0,1,2)))
tryit('t3svd max_tucker_ranks unequal within group', lambda: x.t3svd(sharing=(0,1,0,2), max_tucker_ranks=(2,4,3,2)))
tryit('t3svd untied in unsafe mode', lambda: (lambda: (safety.unsafe().__enter__(), xu.t3svd(sharing=(0,1,0,2))[0].ranks)[1])())

sec('O/P T3Weights')
x = T3.randn((5,6,7), (2,3,2), (1,2,3,1)).t3svd()[0]
x = T3.randn((5,6,7), (2,3,2), (1,2,2,1))
print('  minimal', x.has_minimal_ranks)
W = T3Weights.from_t3svd(x)
print('  consistent', W.is_consistent_with(x), W.tucker_ranks, W.tt_ranks, [w.shape for w in W.tt_weights])
xw = t3_absorb_weights(x, W)
print('  weighted_norm vs absorb.norm', float(t3_weighted_norm(x, W)), float(xw.norm()))
y = T3.randn((5,6,7), (3,2,4), (1,3,2,1)); Wy = T3Weights.from_t3svd(y)
print('  weighted_inner vs absorb.inner', float(t3_weighted_inner(x, W, y, Wy)), float(xw.inner(t3_absorb_weights(y, Wy))))
# reverse consistency
a = t3_absorb_weights(x.reverse(), W.reverse()); b = t3_absorb_weights(x, W).reverse()
print('  reverse consistency err', np.linalg.norm(a.to_dense()-b.to_dense()))
# stack/unstack
xs = T3.randn((5,6,7), (2,3,2), (1,2,2,1), stack_shape=(2,3))
Ws = T3Weights(tuple(np.random.rand(2,3,n) for n in xs.tucker_ranks), tuple(np.random.rand(2,3,r) for r in xs.tt_ranks))
un = Ws.unstack(); print('  unstack tree', len(un), len(un[0]), type(un[0][0]).__name__, un[1][2].stack_shape)
Wr = T3Weights.stack(un); print('  restack err', cw.corewise_norm(cw.corewise_sub(Wr.data, Ws.data)), Wr.stack_shape)
# stacked weighted norm vs per-element
wn = t3_weighted_norm(xs, Ws); print('  stacked weighted_norm shape', wn.shape, 'vs absorb', np.max(np.abs(wn - t3_absorb_weights(xs, Ws).norm())))
# reciprocal/sqrt/concatenate/kronecker semantics
x2 = T3.randn((5,6,7), (3,2,4), (1,3,2,1)); W2 = T3Weights.from_t3svd(x2)
Wc = W.concatenate(W2); print('  concat ranks', Wc.tucker_ranks, Wc.tt_ranks, 'consistent with x+x2', Wc.is_consistent_with(x + x2))
print('  absorb(x+x2, Wc) == absorb(x,W)+absorb(x2,W2)?', np.allclose(t3_absorb_weights(x+x2, Wc).to_dense(), (t3_absorb_weights(x,W)+t3_absorb_weights(x2,W2)).to_dense()))
Wk = W.kronecker(W2); print('  kron ranks', Wk.tucker_ranks, Wk.tt_ranks, 'consistent with x*x2', Wk.is_consistent_with(x * x2))
print('  absorb(x*x2, Wk) == absorb(x,W)*absorb(x2,W2)?', np.allclose(t3_absorb_weights(x*x2, Wk).to_dense(), (t3_absorb_weights(x,W)*t3_absorb_weights(x2,W2)).to_dense()))
print('  sqrt.sqrt vs W^(1/2)?', np.allclose(W.sqrt().tucker_weights[0]**2, W.tucker_weights[0]))
tryit('T3Weights bad len', lambda: T3Weights((np.ones(2),np.ones(3)), (np.ones(1),np.ones(2))))
tryit('is_consistent_with wrong', lambda: W.is_consistent_with(x2))
tryit('absorb inconsistent (no check?)', lambda: t3_absorb_weights(x, W2).ranks)
# jax
xj = x.to_jax(); Wj = T3Weights(tuple(map(__import__('jax').numpy.asarray, W.tucker_weights)), tuple(map(__import__('jax').numpy.asarray, W.tt_weights)))
tryit('jax sqrt', lambda: type(Wj.sqrt().tucker_weights[0]).__name__)
tryit('jax reciprocal', lambda: type(Wj.reciprocal().tucker_weights[0]).__name__)
tryit('jax weighted norm', lambda: float(t3_weighted_norm(xj, Wj)) - float(t3_weighted_norm(x, W)))

sec('K orthogonalization at asymmetric shapes, tails!=1, stacks, d=4')
for stack in [(), (2,), (2,3)]:
    x = T3.randn((5,7,4,6), (3,4,2,3), (2,3,4,2,3), stack_shape=stack)
    X = x.to_dense(squash_tails=False)
    for ii in range(4):
        xr = x.orthogonalize_relative_to_tucker_core(ii)
        assert np.allclose(xr.to_dense(squash_tails=False), X), ('tucker', ii, stack)
        # complement of B_ii orthogonal: contract everything except B_ii -> env with (..., n_ii, rest)
        tk, tt = xr.data
        ttc = list(xr.to_tensor_train()); 
        # build dense with mode ii left as n_ii (not N_ii)
        cores = [tt[ii] if j==ii else np.einsum('...aib,...ik->...akb', tt[j], tk[j]) for j in range(4)]
        env = cores[0]
        for c in cores[1:]:
            env = np.einsum('...a,...b->...ab', env.reshape(env.shape[:len(stack)] + (-1, env.shape[-1])), c.reshape(c.shape[:len(stack)]+(c.shape[-3], -1))) if False else None
        # simpler: use gram via einsum on full dense with mode ii index n
        D = np.einsum('...aib,...ik->...akb', tt[0], tk[0]) if ii!=0 else tt[0]
        for j in range(1,4):
            cj = tt[j] if j==ii else np.einsum('...aib,...ik->...akb', tt[j], tk[j])
            D = np.einsum('...xb,...byc->...xyc', D.reshape(D.shape[:len(stack)]+(-1, D.shape[-1])), cj)
            D = D.reshape(D.shape[:len(stack)] + (-1, D.shape[-1]))
        # D: stack + (r0*prod(modes with n_ii at ii), rd)
        # gram over everything except n_ii requires reshaping; do it by moving mode ii explicitly:
        shp = [x.tt_ranks[0]] + [xr.tucker_ranks[j] if j==ii else x.shape[j] for j in range(4)] + [x.tt_ranks[-1]]
        D = D.reshape(tuple(stack) + tuple(shp))
        Dm = np.moveaxis(D, len(stack)+1+ii, -1)
        Dm = Dm.reshape(tuple(stack)+(-1, Dm.shape[-1]))
        G = np.einsum('...ai,...aj->...ij', Dm, Dm)
        err = np.max(np.abs(G - np.eye(G.shape[-1])))
        assert err < 1e-10, ('tucker complement', ii, stack, err)
        xr = x.orthogonalize_relative_to_tt_core(ii)
        assert np.allclose(xr.to_dense(squash_tails=False), X), ('tt', ii, stack)
        tk, tt = xr.data
        # left part left-orthogonal, right part right-orthogonal, B_ii up-orthogonal (B B^T = I)
        B = tk[ii]; assert np.max(np.abs(np.einsum('...ik,...jk->...ij', B, B) - np.eye(B.shape[-2]))) < 1e-10, ('B', ii, stack)
        if ii > 0:
            L = np.einsum('...aib,...ik->...akb', tt[0], tk[0])
            for j in range(1, ii):
                L = np.einsum('...xb,...byc->...xyc', L.reshape(L.shape[:len(stack)]+(-1, L.shape[-1])), np.einsum('...aib,...ik->...akb', tt[j], tk[j]))
            L = L.reshape(L.shape[:len(stack)] + (-1, L.shape[-1]))
            assert np.max(np.abs(np.einsum('...ai,...aj->...ij', L, L) - np.eye(L.shape[-1]))) < 1e-10, ('L', ii, stack)
        if ii < 3:
            R = np.einsum('...aib,...ik->...akb', tt[ii+1], tk[ii+1])
            for j in range(ii+2, 4):
                R = np.einsum('...xb,...byc->...xyc', R.reshape(R.shape[:len(stack)]+(-1, R.shape[-1])), np.einsum('...aib,...ik->...akb', tt[j], tk[j]))
                R = R.reshape(R.shape[:len(stack)] + (-1, R.shape[-1]))
            R = R.reshape(R.shape[:len(stack)] + (R.shape[len(stack)], -1))
            assert np.max(np.abs(np.einsum('...ia,...ja->...ij', R, R) - np.eye(R.shape[-2]))) < 1e-10, ('R', ii, stack)
    xu = x.up_orthogonalize_tt_cores()
    assert np.allclose(xu.to_dense(squash_tails=False), X)
    for G in xu.tt_cores:
        assert np.max(np.abs(np.einsum('...iaj,...ibj->...ab', G, G) - np.eye(G.shape[-2]))) < 1e-10
    xd = x.down_orthogonalize_tucker_cores(); assert np.allclose(xd.to_dense(squash_tails=False), X)
    xl = x.left_orthogonalize_tt_cores(); assert np.allclose(xl.to_dense(squash_tails=False), X)
    xl2, var = x.left_orthogonalize_tt_cores(return_variation_cores=True); print('  var cores shapes', [v.shape for v in var])
    xrr = x.right_orthogonalize_tt_cores(); assert np.allclose(xrr.to_dense(squash_tails=False), X)
    print('  stack', stack, 'all orthogonalization checks pass')

sec('has_numerically_minimal_ranks on minimal stacked')
xs = T3.randn((5,6,7), (2,3,2), (1,2,2,1), stack_shape=(2,))
print('  minimal', xs.has_minimal_ranks)
tryit('xs.has_numerically_minimal_ranks()', lambda: xs.has_numerically_minimal_ranks())

sec('sum with negative / all axes')
xs = T3.randn((5,6,7), (2,3,2), (1,2,2,1), stack_shape=(2,))
tryit('sum(axis=-1)', lambda: xs.sum(axis=-1).structure)
tryit('sum(axis=(0,1,2))', lambda: xs.sum(axis=(0,1,2)).shape)
tryit('sum(axis=[0,2])', lambda: xs.sum(axis=[0,2]).structure)
tryit('sum_stack(axis=-1)', lambda: xs.sum_stack(axis=-1).structure)
tryit('sum_stack(axis=5)', lambda: xs.sum_stack(axis=5).structure)

sec('jax paths on misc methods')
import jax, jax.numpy as jnp
xj = T3.randn((5,6,7), (2,3,2), (2,2,2,3), stack_shape=(2,), use_jax=True)
for name, f in [('copy', lambda: xj.copy().contains_jax), ('reverse', lambda: xj.reverse().contains_jax), ('squash_tails', lambda: xj.squash_tails().contains_jax),
                ('resize', lambda: xj.resize((6,7,8),(3,3,3),(2,3,3,3)).contains_jax), ('unstack', lambda: xj.unstack()[0].contains_jax),
                ('stack', lambda: T3.stack(xj.unstack()).contains_jax), ('segment', lambda: xj.segment(0,2).contains_jax), ('concatenate', lambda: T3.concatenate([xj.segment(0,2), xj.segment(2,3)]).contains_jax),
                ('sum_stack', lambda: xj.sum_stack().contains_jax), ('sum_stack_corewise', lambda: xj.sum_stack_corewise().contains_jax), ('sum', lambda: xj.sum(axis=0).contains_jax),
                ('x+s', lambda: (xj + 2.0).contains_jax), ('x*x', lambda: (xj*xj).contains_jax), ('x+x', lambda: (xj+xj).contains_jax), ('inner', lambda: type(xj.inner(xj)).__name__), ('norm', lambda: type(xj.norm()).__name__),
                ('t3svd', lambda: xj.t3svd(max_tt_ranks=2)[0].contains_jax), ('down_orth', lambda: xj.down_orthogonalize_tucker_cores().contains_jax), ('rel tucker', lambda: xj.orthogonalize_relative_to_tucker_core(1).contains_jax),
                ('from_tt', lambda: T3.from_tensor_train(xj.to_tensor_train()).contains_jax), ('to_vector/from_vector', lambda: T3.from_vector(xj.to_vector(), *xj.structure).contains_jax),
                ('jit resize', lambda: jax.jit(lambda t: t.resize((6,7,8),(3,3,3),(2,3,3,3)))(xj).structure),
                ('jit reverse', lambda: jax.jit(lambda t: t.reverse())(xj).structure),
                ('jit sum_stack', lambda: jax.jit(lambda t: t.sum_stack())(xj).structure),
                ('jit segment', lambda: jax.jit(lambda t: t.segment(0,2))(xj).structure),
                ('jit squash_tails', lambda: jax.jit(lambda t: t.squash_tails())(xj).structure),
                ('jit from_canonical', lambda: jax.jit(lambda F: T3.from_canonical(F))([jnp.ones((2,3,5)), jnp.ones((2,3,6))]).structure),
                ('jit from_tensor_train', lambda: jax.jit(lambda t: T3.from_tensor_train(t.to_tensor_train()))(xj).structure),
                ('jit t3m inplace', lambda: jax.jit(lambda a: a.t3m(a, max_tucker_ranks=3, max_tt_ranks=3))(xj).structure),
                ('jit t3m swap', lambda: jax.jit(lambda a: a.t3m(a, method='swap', max_tucker_ranks=3, max_tt_ranks=3))(xj).structure),
                ('jit rank_adjustment_sweep', lambda: jax.jit(lambda a: a.rank_adjustment_sweep())(xj).structure),
                ('jit sum(axis=1)', lambda: jax.jit(lambda a: a.sum(axis=1))(xj).structure),
                ('jit x+s', lambda: jax.jit(lambda a: a + 1.5)(xj).structure),
                ]:
    tryit(name, f)
