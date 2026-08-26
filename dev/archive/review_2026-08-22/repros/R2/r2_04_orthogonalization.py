"""t3_right_orthogonalize (no caller/test), relative-to-core orthogonalizations, single-core SVD steps, up-orthogonalize."""
import numpy as np, t3toolbox as t3
from t3toolbox.backend import t3_orthogonalization as O, t3_conversions as C
np.random.seed(0)
def dense(d): return C.t3_to_dense(d)
def tucker_res(tk): return max(np.abs(np.einsum('...io,...jo->...ij', B, B) - np.eye(B.shape[-2])).max() for B in tk)
def left_res(G): return np.abs(np.einsum('...aib,...aic->...bc', G, G) - np.eye(G.shape[-1])).max()
def right_res(G): return np.abs(np.einsum('...aib,...cib->...ac', G, G) - np.eye(G.shape[-3])).max()
def up_res(G): return np.abs(np.einsum('...aib,...ajb->...ij', G, G) - np.eye(G.shape[-2])).max()
bad = []
for (shape, tr, ttr) in [((4,), (3,), (1, 1)), ((4, 5), (3, 2), (1, 3, 1)), ((4, 5, 6, 3), (3, 2, 4, 2), (1, 2, 3, 2, 1)), ((3, 5, 6), (4, 2, 3), (2, 2, 3, 2))]:
    for ss in [(), (2,), (2, 3)]:
        x = t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=ss).data
        X = dense(x); d = len(shape)
        # right orthogonalize
        y = O.t3_right_orthogonalize(x)
        e = np.linalg.norm(dense(y) - X) / np.linalg.norm(X)
        rr = O.t3_orthogonality_residual(y, 'right').max(); rl = O.t3_orthogonality_residual(y, 'left').max()
        ok = e < 1e-12 and rr < 1e-12
        if not ok: bad.append(('right', shape, ss, e, rr))
        print('right_orth  shape=%-14s stack=%-6s dense relerr=%.1e right-res=%.1e (left-res=%.1e) %s' % (shape, ss, e, rr, rl, '' if ok else 'FAIL'))
        # relative to tucker core ii / tt core ii
        for ii in range(d):
            tk, tt = O.t3_orthogonalize_relative_to_tucker_core(x, ii)
            e = np.linalg.norm(dense((tk, tt)) - X) / np.linalg.norm(X)
            res = max([tucker_res([B]) for j, B in enumerate(tk) if j != ii] + [left_res(G) for G in tt[:ii]] + [right_res(G) for G in tt[ii+1:]] + [up_res(tt[ii])])
            if e > 1e-12 or res > 1e-12: bad.append(('rel_tucker', shape, ss, ii, e, res))
            tk, tt = O.t3_orthogonalize_relative_to_tt_core(x, ii)
            e = np.linalg.norm(dense((tk, tt)) - X) / np.linalg.norm(X)
            res = max([tucker_res(tk)] + [left_res(G) for G in tt[:ii]] + [right_res(G) for G in tt[ii+1:]])
            if e > 1e-12 or res > 1e-12: bad.append(('rel_tt', shape, ss, ii, e, res))
            # single core steps
            for name, f in [('down_svd_tucker', O.t3_down_svd_tucker_core), ('left_svd_tt', O.t3_left_svd_tt_core), ('right_svd_tt', O.t3_right_svd_tt_core), ('down_svd_tt', O.t3_down_svd_tt_core), ('up_svd_tt', O.t3_up_svd_tt_core)]:
                try:
                    nx, sv = f(x, ii)
                    e = np.linalg.norm(dense(nx) - X) / np.linalg.norm(X)
                    if e > 1e-12: bad.append((name, shape, ss, ii, e))
                except Exception as ex:
                    bad.append((name, shape, ss, ii, 'RAISED', type(ex).__name__, str(ex)[:80]))
        # up-orthogonalize tt cores: (variations, outer) should represent the same tensor, outer cores up-orthonormal
        V, Oc = O.t3_up_orthogonalize_tt_cores(x)
        e = np.linalg.norm(dense((V, Oc)) - X) / np.linalg.norm(X); res = max(up_res(G) for G in Oc)
        if e > 1e-12 or res > 1e-12: bad.append(('up_orth_tt', shape, ss, e, res))
print('FAILURES:', bad if bad else 'none')
