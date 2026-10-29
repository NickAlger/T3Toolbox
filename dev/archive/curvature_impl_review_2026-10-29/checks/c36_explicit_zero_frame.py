"""Read-only plan check (v5, I-18a): an EXPLICIT canonical zero-point frame (identity-prefix U, L[:-1], R[1:], O;
L[-1] = R[0] = 0), built from (shape, tucker ranks, tt ranks) with no LAPACK call. (1) valid: orthogonality residual
0, base point = 0, gauge projection idempotent; (2) its zero reference vs the LAPACK frame geom.frame(zeros) gives;
(3) packed into the uniform layout (t3frame_to_ut3frame) the uniform kind + geometry give the same reference;
(4) shared (0,0,1): tied by construction, same reference ragged vs uniform."""
import sys; sys.path.insert(0, __import__("os").getcwd())
import numpy as np
import t3toolbox.tucker_tensor_train as t3, t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.backend.optimizers as bopt, t3toolbox.backend.geometry as bgeo, t3toolbox.backend.fitting as bfit
import t3toolbox.backend.uniform_fitting as uf, t3toolbox.backend.ufv_conversions as ufc
import t3toolbox.backend.fv_operations as fvo, t3toolbox.backend.fv_conversions as fvc, t3toolbox.backend.tv_operations as tvo
import t3toolbox.backend.t3_conversions as t3c

def prefix(rows, cols):                     # rows x cols with ones on the leading diagonal
    E = np.zeros((rows, cols)); k = min(rows, cols); E[np.arange(k), np.arange(k)] = 1.0; return E

def zero_frame_like(frame):
    """The canonical zero-point frame with `frame`'s STRUCTURE (its core shapes): identity-prefix U, O, L[:-1], R[1:];
    L[-1] = R[0] = 0 (the base-point remainder of the zero tensor). Shapes only -- no LAPACK completion."""
    U0, O0, L0, R0 = frame; d = len(U0)
    U = tuple(prefix(*u.shape) for u in U0)
    L = tuple(prefix(l.shape[0] * l.shape[1], l.shape[2]).reshape(l.shape) if i < d - 1 else np.zeros(l.shape) for i, l in enumerate(L0))
    R = tuple(prefix(r.shape[0], r.shape[1] * r.shape[2]).reshape(r.shape) if i > 0 else np.zeros(r.shape) for i, r in enumerate(R0))
    O = tuple(np.moveaxis(prefix(o.shape[1], o.shape[0] * o.shape[2]).reshape(o.shape[1], o.shape[0], o.shape[2]), 0, 1) for o in O0)
    return (U, O, L, R)

def zero_point_frame(shape, tk, tt):
    d = len(shape)
    U = tuple(prefix(n, N) for n, N in zip(tk, shape))                                       # (n, N), U U^T = I
    L = tuple(prefix(tt[i] * tk[i], tt[i + 1]).reshape(tt[i], tk[i], tt[i + 1]) if i < d - 1
              else np.zeros((tt[i], tk[i], tt[i + 1])) for i in range(d))                    # left-orth, last = 0
    R = tuple(prefix(tt[i], tk[i] * tt[i + 1]).reshape(tt[i], tk[i], tt[i + 1]) if i > 0
              else np.zeros((tt[i], tk[i], tt[i + 1])) for i in range(d))                    # right-orth, first = 0
    O = tuple(np.moveaxis(prefix(tk[i], tt[i] * tt[i + 1]).reshape(tk[i], tt[i], tt[i + 1]), 0, 1) for i in range(d))
    return (U, O, L, R)

rng = np.random.default_rng(0); np.random.seed(0)
for shape, tk, tt, sharing in (((5, 6, 7), (2, 3, 2), (1, 2, 2, 1), None), ((6, 6, 6, 6), (3, 3, 3, 3), (1, 3, 4, 3, 1), None),
                               ((6, 6, 5), (3, 3, 2), (1, 3, 2, 1), (0, 0, 1)), ((6,), (3,), (1, 1), None)):
    W = 40
    ww = [rng.standard_normal((W, N)) for N in shape]
    data = np.asarray(t3.TuckerTensorTrain.randn(shape, tk, tt).apply(ww))
    zero = t3.TuckerTensorTrain.zeros(shape, tk, tt)
    geom = bgeo.ManifoldGeometryOps() if sharing is None else bgeo.ManifoldGeometryOps().with_sharing(sharing, shape)
    prob = bopt.least_squares_problem(geom, bfit.APPLY, ww, data)
    F = zero_frame_like(prob.local_model(zero.data).frame)   # the structure of the frame the model would use
    orth = float(fvo.fv_frame_orthogonality_residual(F))
    base = float(np.linalg.norm(t3c.t3_to_dense((F[0], F[2]))))
    aux = geom.precompute(F)
    g = geom.project(F, prob.kind.transpose(-data, ww, F, prob.kind.precompute(F, ww)), aux=aux)
    gg = geom.project(F, g, aux=aux)
    idem = max(float(np.linalg.norm(a - b)) for a, b in zip(g[0] + g[1], gg[0] + gg[1]))
    ref_explicit = float(geom.inner(g, g)) ** 0.5
    glapack = prob.local_model(zero.data).gradient; ref_lapack = float(geom.inner(glapack, glapack)) ** 0.5
    uz = ut3.UniformTuckerTensorTrain.from_t3(zero)
    ug = bgeo.UniformManifoldGeometryOps.from_point(uz.data, sharing)
    ukind = uf.uniform_sampling_kind('apply', uz.data, None); ps = uf.pack_sample('apply', ww, uz.N)
    uframe = ug.frame((uz.data[0], uz.data[1]))      # padded sizes of the uniform frame the model would use
    UF = ufc.t3frame_to_ut3frame(F, N=uframe[0].shape[-1], nU=uframe[0].shape[-2], nD=uframe[1].shape[-2],
                                 rL=uframe[2].shape[-1], rR=uframe[3].shape[-1])
    gu = ug.project(UF, ukind.transpose(-data, ps, UF, ukind.precompute(UF, ps)), aux=ug.precompute(UF))
    ref_u = float(ug.inner(gu, gu)) ** 0.5
    print(f"{shape} {tk} {tt} sharing={sharing}: orth resid {orth:.0e}, |base point| {base:.0e}, Pi idempotent {idem:.0e}; "
          f"zero ref explicit {ref_explicit:.6e} | LAPACK frame {ref_lapack:.6e} | uniform packed {ref_u:.6e} "
          f"(u vs explicit {abs(ref_u - ref_explicit) / ref_explicit:.0e})")
