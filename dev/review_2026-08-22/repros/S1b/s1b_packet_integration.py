"""Integration check: the packet's pad_safe_svd on the REAL S1b case-B unfoldings.

Case B = zero-padded resize warm start (2,2,2)/(1,2,2,1) -> (3,3,3)/(1,3,3,1) on shape (5,6,7):
structurally minimal masks, numerically deficient -- the continuation case that loses tangent
directions today.  For each uniform-sweep site kind we take the exact matrix the sweep SVDs,
build row/col pad masks from the library's masks, and compare plain SVD vs pad_safe_svd:
  - does the masked real block of U keep full mask rank (no lost directions)?
  - is the real block orthonormal?
  - does U @ (S*V.T) reconstruct the padded unfolding (remainder push-through exact)?
"""
import sys, numpy as np
sys.path.insert(0, '/home/nick/repos/T3Toolbox')
sys.path.insert(0, '/tmp/claude-1000/-home-nick-repos-T3Toolbox/7a6ed361-8c79-489c-87ff-713bf71ecb11/scratchpad/s1b_packet/packet')
from pad_safe_svd import pad_safe_svd
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
np.random.seed(0)

x0 = t3.TuckerTensorTrain.randn((5, 6, 7), (2, 2, 2), (1, 2, 2, 1))
xB = x0.resize((5, 6, 7), (3, 3, 3), (1, 3, 3, 1))          # the failing warm start
u  = ut3.UniformTuckerTensorTrain.from_t3(xB)
tk, tt = np.asarray(u.tucker_supercore), np.asarray(u.tt_supercore)   # (d,n,N), (d,r,n,r)
tkm, ttm = u.masks.data                                                # (d,n) bool, (d+1,r) bool
d, n, N = tk.shape; r = tt.shape[-1]
shape = u.shape

def check(tag, M_pad, row_pad, col_pad):
    m = int((~col_pad).sum())
    # -- plain SVD (what the sweep does today)
    U0, S0, _ = np.linalg.svd(M_pad, full_matrices=False)
    real0 = U0[~row_pad][:, :m]
    # -- pad-safe
    U1, S1, V1 = pad_safe_svd(M_pad, row_pad, col_pad)
    real1 = U1[~row_pad][:, :m]
    rec = np.linalg.norm(U1 * S1 @ V1.T - M_pad)
    lost0 = m - np.linalg.matrix_rank(real0, tol=1e-10)
    lost1 = m - np.linalg.matrix_rank(real1, tol=1e-10)
    orth1 = np.linalg.norm(real1.T @ real1 - np.eye(m))
    bit = bool(np.all(U1[row_pad][:, :m] == 0.0))
    sig_true = np.linalg.svd(M_pad[np.ix_(~row_pad, ~col_pad)], compute_uv=False)
    sig_ok = np.allclose(S1[:m], sig_true[:m], atol=1e-12)
    print(f"  {tag:34s} lost: plain={lost0} pad_safe={lost1} | real-block orth {orth1:.1e} | "
          f"pads bitwise0 {bit} | sigma==unpadded {sig_ok} | recon {rec:.1e}")
    return lost0, lost1

tot0 = tot1 = 0
print("Tucker down-orth site (rows = mode index, SUFFIX pads; cols = tucker rank mask):")
for i in range(d):
    M_pad = tk[i].T                                   # (N, n)
    row_pad = np.arange(N) >= shape[i]
    col_pad = ~tkm[i]
    a, b = check(f"mode {i}: ({shape[i]}+{N-shape[i]} rows) x ({int(tkm[i].sum())} of {n})", M_pad, row_pad, col_pad)
    tot0 += a; tot1 += b

print("TT up-orth site (rows = (a,b) Kronecker of the two tt masks -- INTERIOR pads; cols = tucker rank mask):")
for i in range(d):
    H = np.swapaxes(tt[i], -1, -2).reshape(r * r, n)  # rows a*r+b
    row_pad = ~np.kron(ttm[i], ttm[i + 1])
    col_pad = ~tkm[i]
    interior = "interior" if (~row_pad).nonzero()[0].max() > (~row_pad).sum() - 1 else "suffix  "
    a, b = check(f"mode {i}: rows real {int((~row_pad).sum())}/{r*r} ({interior})", H, row_pad, col_pad)
    tot0 += a; tot1 += b

print(f"\nTOTAL lost directions across sites: plain SVD = {tot0}, pad_safe_svd = {tot1}")
assert tot1 == 0
