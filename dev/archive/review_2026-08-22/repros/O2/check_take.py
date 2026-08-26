"""flat_draw / kind.take invariant: with data y = S(A), the drawn minibatch must satisfy S(A; sample_B) == data_B
for every kind, ragged and uniform (packed take), with W = (12,) and W = (3, 4)."""
import sys, numpy as np
sys.path.insert(0, '.')
import oracle as O, sweep_models as SM, sweep_optimizers as SO
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.backend.optimizers as bopt
import t3toolbox.backend.fitting as bfit
import t3toolbox.backend.geometry as bgeo
import t3toolbox.backend.uniform_fitting as uf
import t3toolbox.backend.ut3_operations as uops
shape, tk, tt = (4, 5, 6), (2, 3, 2), (1, 2, 2, 1)
for W in [(12,), (3, 4)]:
    for kind in ['apply', 'entries', 'probe', 'apply_derivatives', 'entries_derivatives', 'probe_derivatives']:
        order = 2 if kind.endswith('_derivatives') else None
        rng = np.random.default_rng(0)
        A = t3.TuckerTensorTrain.randn(shape, tk, tt)
        sample = SM.make_sample(kind, shape, W, rng)
        y = O.S(kind, A.to_dense(), sample, order)
        # ragged
        bk = {'apply': bfit.APPLY, 'entries': bfit.ENTRIES, 'probe': bfit.PROBE}.get(kind) or \
             {'apply_derivatives': bfit.apply_derivatives_kind, 'entries_derivatives': bfit.entries_derivatives_kind,
              'probe_derivatives': bfit.probe_derivatives_kind}[kind](order)
        prob = bopt.least_squares_problem(bgeo.ManifoldGeometryOps(), bk, sample, y)
        sB, dB = bopt.flat_draw(prob, 5)(np.random.default_rng(1))
        hand = O.S(kind, A.to_dense(), sB, order)
        e_r = SM.rel_list(dB, hand) if kind.startswith('probe') else SM.rel(dB, hand)
        # uniform (packed)
        uA = ut3.UniformTuckerTensorTrain.from_t3(A)
        uprob = uf.uniform_least_squares_problem('manifold', kind, uA, sample, y, order)
        usB, udB = bopt.flat_draw(uprob, 5)(np.random.default_rng(1))
        # unpack the packed minibatch sample back to ragged for the oracle
        def unpack_ww(p):  # (d,)+W'+(N,) -> list of (W', Ni)
            return [np.asarray(p[i][..., :shape[i]]) for i in range(len(shape))]
        if kind in ('apply', 'probe'):
            usB_r = unpack_ww(usB)
        elif kind == 'entries':
            usB_r = np.asarray(usB)
        elif kind in ('apply_derivatives', 'probe_derivatives'):
            usB_r = (unpack_ww(usB[0]), unpack_ww(usB[1]))
        else:
            usB_r = (np.asarray(usB[0]), unpack_ww(usB[1]))
        uhand = O.S(kind, A.to_dense(), usB_r, order)
        if kind.startswith('probe'):
            udB = np.asarray(udB)
            e_u = max(SM.rel(udB[i][..., :shape[i]], uhand[i]) for i in range(len(shape)))
        else:
            e_u = SM.rel(udB, uhand)
        print(f'W={W} {kind:19s} ragged take: {e_r:.1e}  uniform packed take: {e_u:.1e}', 'OK' if max(e_r, e_u) < 1e-12 else 'MISMATCH')
