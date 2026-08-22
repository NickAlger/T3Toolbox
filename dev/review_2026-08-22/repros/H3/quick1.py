import numpy as np, t3toolbox.tucker_tensor_train as t3
np.random.seed(0)
for C in [(), (3,), (2,3)]:
    x = t3.TuckerTensorTrain.randn((5,6,7),(2,3,4),(1,2,3,1), stack_shape=C)
    print(C, 'svd:', x.t3svd()[0].is_left_orthogonal(), 'down+left:', x.down_orthogonalize_tucker_cores().left_orthogonalize_tt_cores().is_left_orthogonal(),
          'right:', x.down_orthogonalize_tucker_cores().right_orthogonalize_tt_cores().is_right_orthogonal())
    c = np.asarray(np.random.randn())
    ww = tuple(np.random.randn(N) for N in (5,6,7))
    if C == ():
        print('scalar c apply_ambient_transpose ok:', [f.shape for f in t3.TuckerTensorTrain.apply_ambient_transpose(c, ww)])
        print('scalar c apply_corewise_transpose ok:', [f.shape for f in x.apply_corewise_transpose(c, ww)[0]])
