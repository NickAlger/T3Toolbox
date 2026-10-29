"""Read-only plan check (v5, P1c fix): the critic's O2 ragged-vs-uniform pairing (check_zero_reference_o2_pairs.py,
SO.setup, 6 Newton steps, threshold 1e-7) with each layer's zero reference taken at the CANONICAL zero-point frame
(identity-prefix completion from the frame's structure; c36) instead of the LAPACK frame. Covers d = 3 and 2, manifold,
unshared and shared, every kind with its weight specs."""
import sys, os; sys.path.insert(0, __import__("os").getcwd()); sys.path.insert(0, "tests/oracle_sweeps")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.optimizers as topt
import t3toolbox.corewise as cw
import t3toolbox.backend.ufv_conversions as ufc
import sweep_optimizers as SO
import sweep_models as SM
import importlib.util
spec = importlib.util.spec_from_file_location('c36', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'c36_explicit_zero_frame.py'))
src = open(spec.origin).read(); ns = {}
exec(compile(src[:src.index('rng = np.random.default_rng(0)')], 'c36_defs', 'exec'), ns)
zero_frame_like = ns['zero_frame_like']

def canonical_ref(geom, kind, sample, y, x0, order, weight, rep):
    problem, init, _ = topt._setup(geom, kind, sample, y, x0, order, weight, None, chunk_size=None)
    zeros = cw.corewise_zeros_like(init)
    lm = problem.local_model(zeros)                        # only for the frame STRUCTURE (and its masks)
    if rep == 'ragged':
        F = zero_frame_like(lm.frame)
    else:
        rag = ufc.ut3frame_to_t3frame(lm.frame)
        Fr = zero_frame_like(rag)
        U, O, L, R = lm.frame[:4]
        F = ufc.t3frame_to_ut3frame(Fr, N=U.shape[-1], nU=U.shape[-2], nD=O.shape[-2], rL=L.shape[-1], rR=R.shape[-1])
    pk, geo = problem.kind, problem.geom
    sw = pk.precompute(F, problem.sample)
    resid = cw.corewise_scale(problem.data, -1.0) if isinstance(problem.data, (list, tuple)) else -problem.data
    g = geo.project(F, pk.transpose(resid, problem.sample, F, sw), aux=geo.precompute(F))
    return float(geo.inner(g, g)) ** 0.5

worst = {'lapack': 0.0, 'canonical': 0.0}; refdiff = 0.0; n = 0
for d in (3, 2):
    for shared in (False, True):
        for kind in ('apply', 'entries', 'probe', 'apply_derivatives', 'entries_derivatives', 'probe_derivatives'):
            order = 2 if kind.endswith('_derivatives') else None
            wspecs = ['none'] + (['mode'] if kind == 'probe' else []) + (['order'] if kind.endswith('_derivatives') else []) \
                     + (['matrix'] if kind == 'probe_derivatives' else [])
            for wspec in wspecs:
                x, sharing, sample, weight, y = SO.setup(d, shared, kind, order, wspec, 11)
                refs, losses = {}, {}
                for rep in ('ragged', 'uniform'):
                    geom = SO.geometry_for(rep, 'manifold', shared, sharing)
                    x0 = x if rep == 'ragged' else ut3.UniformTuckerTensorTrain.from_t3(x)
                    refs[rep] = canonical_ref(geom, kind, sample, y, x0, order, weight, rep)
                    _, st = topt.newton_cg(geom, kind, sample, y, x0, order=order, weight=weight, max_newton=6,
                                           g0norm_newton=refs[rep])
                    losses[rep] = np.array(st['losses'])
                m = min(len(losses['ragged']), len(losses['uniform']))
                e = SM.rel(losses['ragged'][:m], losses['uniform'][:m]); n += 1
                rd = abs(refs['ragged'] - refs['uniform']) / refs['ragged']; refdiff = max(refdiff, rd)
                worst['canonical'] = max(worst['canonical'], e)
                if e > 1e-7 or rd > 1e-12:
                    print(f"  d={d} shared={shared} {kind} {wspec}: ref rel diff {rd:.1e}, losses {e:.1e}")
print(f"{n} O2 pairs with canonical zero references: worst ref rel diff {refdiff:.1e}; worst paired newton_losses {worst['canonical']:.1e} (threshold 1e-7)")
