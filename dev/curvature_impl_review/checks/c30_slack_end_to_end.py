"""Read-only plan check (v5, I-23): a FULL exact Newton + Steihaug/TR run started from a NON-minimal point -- Tucker
slack and TT slack -- with no minimal-rank reduction (the backend Problem; the frontend's _ragged_minimal is not
applied). Prototype exact hvp = the critic's E2 construction (apply kind; lift with pinv/filter; block-sum
two-hole). Reports whether the run converges, its outer/hvp counts, and the frame vs point ranks along the way."""
import sys, os, math
sys.path.insert(0, __import__("os").getcwd())
SCR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = open(os.path.join(SCR, 'critic_checks', 'check_restart_tr2.py')).read()
ns = {}
exec(compile(src[:src.index('def scenario(')], 'check_restart_tr2_defs', 'exec'), ns)
np, t3, bopt, bgeo, bfit, fvc, run = (ns[k] for k in ('np', 't3', 'bopt', 'bgeo', 'bfit', 'fvc', 'run'))

for label, x_tk, x_tt in (('Tucker slack', (4, 4, 4), (1, 2, 2, 1)), ('TT slack', (4, 4, 4), (1, 9, 9, 1))):
    rng = np.random.default_rng(11); np.random.seed(11)
    shape, W = (6, 6, 6), 300
    target = t3.TuckerTensorTrain.randn(shape, (3, 3, 3), (1, 3, 3, 1))
    ww = [rng.standard_normal((W, N)) for N in shape]
    clean = np.asarray(target.apply(ww))
    data = clean + 0.02 * np.linalg.norm(clean) / math.sqrt(W) * rng.standard_normal(W)
    x0 = t3.TuckerTensorTrain.randn(shape, x_tk, x_tt)
    frame, _ = fvc.t3_orthogonal_representations(x0.data)
    U, O, L, R = frame
    print(f"== {label}: point tucker {x_tk} tt {x_tt}; frame up {[u.shape[0] for u in U]} down {[o.shape[1] for o in O]} "
          f"left {[L[0].shape[0]] + [l.shape[2] for l in L]} right {[R[0].shape[0]] + [r.shape[2] for r in R]}")
    problem = bopt.least_squares_problem(bgeo.ManifoldGeometryOps(), bfit.APPLY, ww, data)
    _, st = bopt.newton_cg(problem, x0.data, max_newton=80, gtol_rel=1e-8)
    h = st['history']
    print(f"  GN + Armijo (library): newton {st['newton']}, converged {h[-1]['converged']}, final rel g {h[-1]['gnorm'] / h[-1]['g0norm']:.1e}")
    for rule in ('rel1e-6', 'zero'):
        r = run(problem, ww, x0.data, 'exact', rule, max_outer=80)
        print(f"  exact + TR eps={rule:8s}: outer {r['iters']:3d} (rejected {r['rejected']:2d}), hvps {r['hvps']:4d}, "
              f"converged {r['converged']}, final rel g {r['final_rel_g']:.1e}")
