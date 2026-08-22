import numpy as np, t3toolbox.tucker_tensor_train as t3
np.random.seed(0)
x = t3.TuckerTensorTrain.randn((5,6,7),(2,3,4),(1,2,3,1))
xs, st, stt = x.t3svd()
print('x ranks', x.ranks, 'svd ranks', xs.ranks, 'sval lens', [len(s) for s in st], [len(s) for s in stt])
print('has_minimal', x.has_minimal_ranks, x.minimal_ranks)
W = t3.T3Weights.from_t3svd(x)
print('consistent with svd result:', W.is_consistent_with(xs))
