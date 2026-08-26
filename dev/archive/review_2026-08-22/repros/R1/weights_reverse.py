import numpy as np
from t3toolbox.tucker_tensor_train import TuckerTensorTrain as T3, T3Weights, t3_absorb_weights
np.random.seed(0)
x = T3.randn((5,6,7), (2,3,2), (1,2,2,1))
W = T3Weights(tuple(np.random.rand(n)+0.5 for n in x.tucker_ranks), tuple(np.random.rand(r)+0.5 for r in x.tt_ranks))
(B0,B1,B2),(G0,G1,G2) = x.data
(u0,u1,u2),(t0,t1,t2,t3) = W.data
# hand-built fully weighted dense tensor: diag weights on every edge
ref = np.einsum('p,pxq,q,qyr,r,rzs,s,x,xi,y,yj,z,zk->ijk', t0,G0,t1,G1,t2,G2,t3, u0,B0,u1,B1,u2,B2)
a = t3_absorb_weights(x, W).to_dense()
print('absorb(x,W) vs hand:', np.linalg.norm(a-ref))
b = t3_absorb_weights(x.reverse(), W.reverse()).to_dense()
print('absorb(x.rev, W.rev) vs hand reversed:', np.linalg.norm(b - ref.transpose(2,1,0)))
c = t3_absorb_weights(x, W).reverse().to_dense()
print('absorb(x,W).rev vs hand reversed:', np.linalg.norm(c - ref.transpose(2,1,0)))
print('x.reverse() dense vs x dense transposed:', np.linalg.norm(x.reverse().to_dense() - x.to_dense().transpose(2,1,0)))
xr = x.reverse(); print('x.reverse structure', xr.structure, 'W.reverse ranks', W.reverse().tucker_ranks, W.reverse().tt_ranks)
