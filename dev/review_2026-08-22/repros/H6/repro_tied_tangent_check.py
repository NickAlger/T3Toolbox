"""H6: does the shared-retract TIED-tangent precondition actually detect an untied tangent?"""
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.frame_variations_format as bvf
import t3toolbox.manifold as t3m
import t3toolbox.shared_geometry as sg
import t3toolbox.backend.sharing as bsh
np.random.seed(1)
M = t3m.MANIFOLD
for shape, tr, ttr, sh in [((5, 5, 4), (2, 2, 2), (1, 2, 2, 1), (0, 0, 1)), ((6, 6, 3, 3), (3, 3, 2, 2), (1, 3, 4, 2, 1), (0, 0, 1, 1)),
                           ((4, 4), (2, 2), (1, 2, 1), (0, 0)), ((5, 5, 5), (2, 2, 2), (1, 3, 3, 1), (0, 0, 0))]:
    x = t3.TuckerTensorTrain.randn(shape, tr, ttr).share(sh); S = sg.shared_manifold(sh); f = S.frame(x); sd = S.shared_frame_data(f)
    u = M.randn(f)                                   # gauged, generically untied
    V, H = u.variations.data
    print('%s tr=%s ttr=%s sh=%s  holes V=%s  down_ranks=%s' % (shape, tr, ttr, sh, f.variation_shapes[0], f.down_ranks))
    for name, Vm in [('randn', V), ('V1:=0', tuple(np.zeros_like(V[i]) if i == 1 else V[i] for i in range(len(V)))),
                     ('V0:=0', tuple(np.zeros_like(V[i]) if i == 0 else V[i] for i in range(len(V)))),
                     ('tied', S.project(u).variations.data[0])]:
        t = t3m.T3Tangent(f, bvf.T3Variations(Vm, H))
        res = float(bsh.fv_tied_variations_residual(t.variations.data, sd))
        pt = S.project(t)
        vec_change = float(np.linalg.norm(pt.to_dense() - t.to_dense()) / np.linalg.norm(t.to_dense()))
        y1 = S.retract(t) if res <= 1e-9 else None
        y2 = S.retract(pt)
        rd = float(np.linalg.norm(y1.to_dense() - y2.to_dense()) / np.linalg.norm(y2.to_dense())) if y1 is not None else float('nan')
        print('   %-6s tied-residual=%.1e  |project(t)-t|/|t| (ambient)=%.1e   retract(t) vs retract(project t) rel diff=%s'
              % (name, res, vec_change, ('%.1e' % rd) if y1 is not None else 'n/a (safe mode raises)'))
