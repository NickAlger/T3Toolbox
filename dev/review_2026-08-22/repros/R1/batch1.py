import numpy as np, traceback, os
import t3toolbox.tucker_tensor_train as t3
from t3toolbox.tucker_tensor_train import TuckerTensorTrain as T3, T3Weights
import t3toolbox.corewise as cw
np.random.seed(0)

def sec(name): print('\n### ' + name)
def tryit(label, f):
    try:
        r = f(); print(f'  [{label}] OK ->', r)
    except Exception as e:
        print(f'  [{label}] RAISED {type(e).__name__}: {str(e)[:200]}')

x = T3.randn((5,6,7), (2,3,2), (1,2,3,1))
xs = T3.randn((5,6,7), (2,3,2), (1,2,3,1), stack_shape=(2,))

sec('A entries wrong-length list error path')
tryit('entries([1,2])', lambda: x.entries([1,2]))
tryit('entries(np.array([1,2]))', lambda: x.entries(np.array([1,2])))

sec('B scalar * T3 (__rmul__)')
tryit('x * 2.0', lambda: (x*2.0).structure)
tryit('2.0 * x', lambda: (2.0*x).structure)
tryit('np.float64(2) * x', lambda: (np.float64(2.0)*x))
tryit('-x', lambda: (-x).structure)
tryit('2.0 + x', lambda: (2.0 + x))
tryit('x / 2.0', lambda: (x / 2.0))
import t3toolbox.uniform_tucker_tensor_train as u
ux = u.UniformTuckerTensorTrain.from_t3(x)
tryit('2.0 * uniform', lambda: type(2.0*ux).__name__)

sec('C T3 * ndarray wrong shape -> assert')
tryit('x * randn(5,6)', lambda: x * np.random.randn(5,6))
tryit('xs * randn(5,6,7) (stacked, unstacked array)', lambda: xs * np.random.randn(5,6,7))

sec('D T3 + T3 different stack shape')
tryit('x + xs', lambda: x + xs)
tryit('x.inner(xs)', lambda: x.inner(xs))

sec('E eq/hash')
y = T3(tuple(B.copy() for B in x.tucker_cores), tuple(G.copy() for G in x.tt_cores))
tryit('x == y (equal copies)', lambda: x == y)
tryit('x == x', lambda: x == x)
tryit('hash(x)', lambda: hash(x))
tryit('x != y', lambda: x != y)

sec('F share on stacked with rtol / max ranks')
tryit('xs.share((0,0,1)) on shape (5,6,7) -> unequal sizes', lambda: xs.share((0,0,1)))
xs2 = T3.randn((6,6,5), (2,3,2), (1,2,3,1), stack_shape=(2,))
tryit('xs2.share((0,0,1)) stacked no tol', lambda: xs2.share((0,0,1)).structure)
tryit('xs2.share((0,0,1), rtol=1e-3) stacked', lambda: xs2.share((0,0,1), rtol=1e-3).structure)
tryit('xs2.share((0,0,1), max_tucker_ranks=2) stacked', lambda: xs2.share((0,0,1), max_tucker_ranks=2).structure)

sec('G reverse tt_ranks claim')
xt = T3.randn((5,6,7), (2,3,2), (2,2,3,4))
print('  reverse().tt_ranks =', xt.reverse().tt_ranks, ' docstring claims (1, r(d-1),...,r1, 1)')

sec('I save/load with stacks and jax')
fn = os.path.join(os.getcwd(), 'rt.npz')
xs.save(fn); xl = T3.load(fn)
print('  stacked roundtrip err', cw.corewise_norm(cw.corewise_sub(xs.data, xl.data)), xl.stack_shape)
xj = xs.to_jax()
tryit('save jax-backed', lambda: xj.save(fn))
xlj = T3.load(fn, use_jax=True)
print('  load use_jax ->', xlj.contains_jax, type(xlj.tucker_cores[0]).__name__)
tryit('load via open file', lambda: T3.load(open(fn,'rb')).structure)

sec('Q from_tensor_train / to_tensor_train with stacks and r0!=1')
tt = [np.random.randn(2, 3,5,2), np.random.randn(2, 2,6,3), np.random.randn(2, 3,7,4)]
xf = T3.from_tensor_train(tt)
print('  structure', xf.structure)
d1 = np.einsum('sakb,sbjc,sckd->sakjd' if False else 'saib,sbjc,sckd->saijkd', *tt)
print('  dense err (squash_tails=False)', np.linalg.norm(xf.to_dense(squash_tails=False) - d1))
back = xf.to_tensor_train()
print('  to_tensor_train shapes', [g.shape for g in back], 'err', max(np.linalg.norm(a-b) for a,b in zip(back, tt)))
# to_tensor_train on stacked with r0!=1
xb = T3.randn((5,6,7), (2,3,2), (2,2,3,4), stack_shape=(3,))
ttb = xb.to_tensor_train()
print('  stacked r0!=1 to_tensor_train shapes', [g.shape for g in ttb],
      'err', np.linalg.norm(np.einsum('saib,sbjc,sckd->saijkd', *ttb) - xb.to_dense(squash_tails=False)))

sec('R segment/concatenate')
tryit('segment(2,1)', lambda: x.segment(2,1).structure)
tryit('segment(1,1)', lambda: x.segment(1,1).structure)
tryit('segment(-2,3)', lambda: x.segment(-2,3).structure)
tryit('segment(0,3) stacked', lambda: xs.segment(0,3).structure)
tryit('concatenate mismatched seam', lambda: T3.concatenate([x.segment(0,1), x.segment(2,3)]).structure)

sec('S from_canonical inconsistent')
tryit('from_canonical ranks mismatch', lambda: T3.from_canonical([np.random.randn(3,5), np.random.randn(2,6)]))
tryit('from_canonical stacks mismatch', lambda: T3.from_canonical([np.random.randn(2,3,5), np.random.randn(3,6)]))

sec('V has_numerically_minimal_ranks on stacked')
tryit('xs.has_numerically_minimal_ranks()', lambda: xs.has_numerically_minimal_ranks())

sec('W is_left_orthogonal d=1 etc')
x1 = T3.randn((5,), (3,), (1,1))
tryit('x1.is_left_orthogonal', lambda: x1.is_left_orthogonal())
tryit('x1.t3svd', lambda: x1.t3svd()[0].structure)
tryit('x1.t3svd()[0].is_left_orthogonal', lambda: x1.t3svd()[0].is_left_orthogonal())
tryit('x1.rank_adjustment_sweep', lambda: x1.rank_adjustment_sweep().structure)
tryit('x1.continuation_ranks', lambda: x1.continuation_ranks())
tryit('x1.probe', lambda: [z.shape for z in x1.probe([np.random.randn(5)])])
tryit('x1.apply', lambda: x1.apply([np.random.randn(5)]))
tryit('x1.entries', lambda: x1.entries([2]))
tryit('x1.reverse', lambda: x1.reverse().structure)
tryit('x1.orthogonalize_relative_to_tucker_core(0)', lambda: x1.orthogonalize_relative_to_tucker_core(0).structure)
tryit('x1.orthogonalize_relative_to_tt_core(0)', lambda: x1.orthogonalize_relative_to_tt_core(0).structure)
tryit('x1 * x1', lambda: (x1*x1).structure)
tryit('x1.t3m(x1, max_tt_ranks=1)', lambda: x1.t3m(x1, max_tucker_ranks=2).structure)
tryit('x1.sum()', lambda: x1.sum())
tryit('x1.norm()', lambda: float(x1.norm()) - np.linalg.norm(x1.to_dense()))

sec('Y t3m swap with stacks')
ys = T3.randn((5,6,7), (2,2,3), (1,3,2,1), stack_shape=(2,))
for m in ['form_then_round','inplace_fused','swap']:
    tryit(f't3m {m} stacked max ranks', lambda m=m: (lambda z: (z.structure, float(np.max(np.abs(z.to_dense() - xs.to_dense()*ys.to_dense())))))(xs.t3m(ys, method=m, max_tucker_ranks=4, max_tt_ranks=6)))
