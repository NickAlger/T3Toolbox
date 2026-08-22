"""R3: user-visible consequence of the wrongly-masked uniform frame of a valid NON-minimal point:
safe mode -> obscure 'frame not orthogonal' error; unsafe/jit -> silently wrong tangent projection vs ragged."""
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.uniform_manifold as um
import t3toolbox.manifold as mf
import t3toolbox.safety as safety
import t3toolbox.backend.linalg as linalg

np.random.seed(0)
x = t3.TuckerTensorTrain.randn((5, 5, 5), (2, 2, 2), (1, 4, 4, 1))    # valid, non-minimal (r_1 = 4 > rL*n = 2)
g = t3.TuckerTensorTrain.randn((5, 5, 5), (3, 3, 3), (1, 3, 3, 1))     # an 'ambient gradient' in T3 form
u = ut3.UniformTuckerTensorTrain.from_t3(x); ug = ut3.UniformTuckerTensorTrain.from_t3(g)
fr = mf.MANIFOLD.frame(x); ufr = um.UNIFORM_MANIFOLD.frame(u)
print('ragged frame orthogonal:', bool(fr.is_orthogonal()), '| uniform frame orthogonal:', bool(np.all(np.asarray(ufr.is_orthogonal()))))
try:
    um.UNIFORM_MANIFOLD.project_ambient(ufr, ug)
    print('SAFE mode uniform project_ambient: no error')
except Exception as e:
    print('SAFE mode uniform project_ambient raises:', type(e).__name__, '--', str(e).splitlines()[0][:150])
with safety.unsafe():
    tu = um.UNIFORM_MANIFOLD.project_ambient(ufr, ug)
    tr = mf.MANIFOLD.project_ambient(fr, g)
    du = np.asarray(tu.to_dense()); dr = np.asarray(tr.to_dense())
    print('UNSAFE (= what jit does): ||uniform proj - ragged proj|| / ||ragged proj|| = %.3e' % (np.linalg.norm(du - dr) / np.linalg.norm(dr)))
    # minimal point, same pipeline: agreement as a control
    xm = x.rank_adjustment_sweep('right_to_left').rank_adjustment_sweep('left_to_right')
    umn = ut3.UniformTuckerTensorTrain.from_t3(xm)
    tu2 = um.UNIFORM_MANIFOLD.project_ambient(um.UNIFORM_MANIFOLD.frame(umn), ug)
    tr2 = mf.MANIFOLD.project_ambient(mf.MANIFOLD.frame(xm), g)
    print('control (minimal point): rel diff = %.3e' % (np.linalg.norm(np.asarray(tu2.to_dense()) - np.asarray(tr2.to_dense())) / np.linalg.norm(np.asarray(tr2.to_dense()))))
A = np.random.randn(8, 9)
U, ss, Vt = linalg.truncated_svd(A, min_rank=5, max_rank=3)
print('truncated_svd(min_rank=5, max_rank=3) kept rank:', ss.shape[-1])
