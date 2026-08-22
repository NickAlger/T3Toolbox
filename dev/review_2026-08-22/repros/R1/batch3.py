import numpy as np
from t3toolbox.tucker_tensor_train import TuckerTensorTrain as T3
np.random.seed(0)
def tryit(label, f):
    try: print(f'  [{label}] OK ->', f())
    except Exception as e: print(f'  [{label}] RAISED {type(e).__name__}: {str(e)[:160]}')

def chain(cores, ns):  # cores: list of (...,a,i,b); returns (..., a, prod, b)
    R = cores[0]; R = R.reshape(R.shape[:ns] + (R.shape[-3], -1, R.shape[-1]))
    for c in cores[1:]:
        R = np.einsum('...axb,...byc->...axyc', R, c)
        R = R.reshape(R.shape[:ns] + (R.shape[ns], -1, R.shape[-1]))
    return R

print('### K orthogonalization at asymmetric shapes, tails!=1, stacks, d=4')
for stack in [(), (2,), (2,3)]:
    ns = len(stack)
    x = T3.randn((5,7,4,6), (3,4,2,3), (2,3,4,2,3), stack_shape=stack)
    X = x.to_dense(squash_tails=False)
    for ii in range(4):
        xr = x.orthogonalize_relative_to_tucker_core(ii)
        assert np.allclose(xr.to_dense(squash_tails=False), X), ('tucker same tensor', ii, stack)
        tk, tt = xr.data
        cores = [tt[j] if j == ii else np.einsum('...aib,...ik->...akb', tt[j], tk[j]) for j in range(4)]
        D = chain(cores, ns)   # (..., r0, prod, rd) with mode ii carrying n_ii
        shp = [x.tt_ranks[0]] + [xr.tucker_ranks[j] if j == ii else x.shape[j] for j in range(4)] + [x.tt_ranks[-1]]
        D = D.reshape(tuple(stack) + tuple(shp))
        Dm = np.moveaxis(D, ns + 1 + ii, -1).reshape(tuple(stack) + (-1, shp[1 + ii]))
        G = np.einsum('...ai,...aj->...ij', Dm, Dm)
        err = np.max(np.abs(G - np.eye(G.shape[-1]))); assert err < 1e-10, ('tucker complement', ii, stack, err)
        xr = x.orthogonalize_relative_to_tt_core(ii)
        assert np.allclose(xr.to_dense(squash_tails=False), X), ('tt same tensor', ii, stack)
        tk, tt = xr.data
        B = tk[ii]; assert np.max(np.abs(np.einsum('...ik,...jk->...ij', B, B) - np.eye(B.shape[-2]))) < 1e-10, ('B up-orth', ii, stack)
        if ii > 0:
            L = chain([np.einsum('...aib,...ik->...akb', tt[j], tk[j]) for j in range(ii)], ns)
            L = L.reshape(L.shape[:ns] + (-1, L.shape[-1]))
            assert np.max(np.abs(np.einsum('...ai,...aj->...ij', L, L) - np.eye(L.shape[-1]))) < 1e-10, ('L', ii, stack)
        if ii < 3:
            R = chain([np.einsum('...aib,...ik->...akb', tt[j], tk[j]) for j in range(ii+1, 4)], ns)
            R = R.reshape(R.shape[:ns] + (R.shape[ns], -1))
            assert np.max(np.abs(np.einsum('...ia,...ja->...ij', R, R) - np.eye(R.shape[-2]))) < 1e-10, ('R', ii, stack)
    xu = x.up_orthogonalize_tt_cores(); assert np.allclose(xu.to_dense(squash_tails=False), X)
    for G in xu.tt_cores:
        assert np.max(np.abs(np.einsum('...iaj,...ibj->...ab', G, G) - np.eye(G.shape[-2]))) < 1e-10
    xd = x.down_orthogonalize_tucker_cores(); assert np.allclose(xd.to_dense(squash_tails=False), X)
    for B in xd.tucker_cores:
        assert np.max(np.abs(np.einsum('...ik,...jk->...ij', B, B) - np.eye(B.shape[-2]))) < 1e-10
    xl = x.left_orthogonalize_tt_cores(); assert np.allclose(xl.to_dense(squash_tails=False), X)
    for G in xl.tt_cores[:-1]:
        assert np.max(np.abs(np.einsum('...iaj,...iak->...jk', G, G) - np.eye(G.shape[-1]))) < 1e-10
    xl2, var = x.left_orthogonalize_tt_cores(return_variation_cores=True)
    xrr = x.right_orthogonalize_tt_cores(); assert np.allclose(xrr.to_dense(squash_tails=False), X)
    for G in xrr.tt_cores[1:]:
        assert np.max(np.abs(np.einsum('...iaj,...kaj->...ik', G, G) - np.eye(G.shape[-3]))) < 1e-10
    print('  stack', stack, 'all orthogonalization checks pass; var core shapes', [v.shape for v in var])

print('### has_numerically_minimal_ranks on minimal stacked')
xs = T3.randn((5,6,7), (2,3,2), (1,2,2,1), stack_shape=(2,))
print('  minimal', xs.has_minimal_ranks)
tryit('xs.has_numerically_minimal_ranks()', lambda: xs.has_numerically_minimal_ranks())

print('### sum with negative / all axes')
tryit('sum(axis=-1)', lambda: xs.sum(axis=-1).structure)
tryit('sum(axis=(0,1,2))', lambda: xs.sum(axis=(0,1,2)).shape)
tryit('sum(axis=[0,2])', lambda: xs.sum(axis=[0,2]).structure)
tryit('sum_stack(axis=-1)', lambda: xs.sum_stack(axis=-1).structure)
tryit('sum_stack(axis=5)', lambda: xs.sum_stack(axis=5).structure)
tryit('sum_stack on unstacked', lambda: T3.randn((5,6,7),(2,3,2),(1,2,2,1)).sum_stack().structure)

print('### jax paths on misc methods')
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
                ('jit copy', lambda: jax.jit(lambda a: a.copy())(xj).structure),
                ('jit unstack/stack', lambda: jax.jit(lambda a: T3.stack(a.unstack()))(xj).structure),
                ('jit to_dense', lambda: jax.jit(lambda a: a.to_dense())(xj).shape),
                ]:
    tryit(name, f)
