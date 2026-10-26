"""Read-only plan check (v6, M7): re-scope T25 to the FRAME claim. From c30's Tucker-slack and TT-slack starts: (a) the
first local model's frame is the slack (non-minimal) one, as t3_orthogonal_representations gives it; (b) after the first
accepted exact+TR step the point's ranks; (c) exact+TR from the slack start and from the same tensor reduced to minimal
ranks reach the same objective (and, where both converge, the same tensor up to the fit's own conditioning)."""
import sys, os, math
sys.path.insert(0, __import__("os").getcwd())
SCR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = open(os.path.join(SCR, 'check_restart_tr2.py')).read(); ns = {}
exec(compile(src[:src.index('def scenario(')], 'check_restart_tr2_defs', 'exec'), ns)
np, t3, bopt, bgeo, bfit, fvc, run = (ns[k] for k in ('np', 't3', 'bopt', 'bgeo', 'bfit', 'fvc', 'run'))
cw = ns['cw']
for label, x_tk, x_tt in (('Tucker slack', (4, 4, 4), (1, 2, 2, 1)), ('TT slack', (4, 4, 4), (1, 9, 9, 1))):
    rng = np.random.default_rng(11); np.random.seed(11)
    shape, W = (6, 6, 6), 300
    target = t3.TuckerTensorTrain.randn(shape, (3, 3, 3), (1, 3, 3, 1))
    ww = [rng.standard_normal((W, N)) for N in shape]
    clean = np.asarray(target.apply(ww))
    data = clean + 0.02 * np.linalg.norm(clean) / math.sqrt(W) * rng.standard_normal(W)
    x0 = t3.TuckerTensorTrain.randn(shape, x_tk, x_tt)
    problem = bopt.least_squares_problem(bgeo.ManifoldGeometryOps(), bfit.APPLY, ww, data)
    lm = problem.local_model(x0.data); U, O, L, R = lm.frame
    ref_frame = fvc.t3_orthogonal_representations(x0.data)[0]
    same = all(a.shape == b.shape for a, b in zip(lm.frame[0] + lm.frame[1] + lm.frame[2] + lm.frame[3],
                                                   ref_frame[0] + ref_frame[1] + ref_frame[2] + ref_frame[3]))
    p = cw.corewise_scale(lm.gradient, -1e-3)
    x1 = lm.retract(p)
    r1 = (tuple(u.shape[0] for u in x1[0]), (1,) + tuple(g.shape[2] for g in x1[1]))
    y, _, _ = x0.t3svd(); xm = y.rank_adjustment_sweep('right_to_left')
    rs = run(problem, ww, x0.data, 'exact', 'rel1e-6', max_outer=80)
    rm = run(problem, ww, xm.data, 'exact', 'rel1e-6', max_outer=80)
    fs = float(problem.objective(x0.data)) if False else None
    print(f"{label}: first frame == t3_orthogonal_representations frame structure: {same} "
          f"(down {[o.shape[1] for o in O]}, left {[L[0].shape[0]] + [l.shape[2] for l in L]}, right {[R[0].shape[0]] + [r.shape[2] for r in R]}); "
          f"ranks after one retraction: tucker {r1[0]} tt {r1[1]} (minimal of x0: tucker {xm.tucker_ranks} tt {xm.tt_ranks})")
    print(f"   exact+TR from slack: outer {rs['iters']}, converged {rs['converged']}, final rel g {rs['final_rel_g']:.1e} | "
          f"from minimal: outer {rm['iters']}, converged {rm['converged']}, final rel g {rm['final_rel_g']:.1e}")
