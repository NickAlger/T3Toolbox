"""Does the Gauss-Newton product J^T J p for probes need the ambient (W x N) intermediate at all?
With an orthogonal frame (D D^T = I) and a gauged tangent (D V^T = 0):
    xi_z  =  D (J p)  =  eta_t                       (no N-sized array)
    sum_w eta (x) (J p)  =  (sum_w eta (x) eta_t) D  +  (sum_w eta (x) eta) V      (two n x n Gram matrices)
Usage: python3 gn_fusion_check.py <repo>"""
import sys, pathlib, importlib.util
import numpy as np
repo, scripts = sys.argv[1], str(pathlib.Path(__file__).resolve().parent)
sys.path.append(repo)
spec = importlib.util.spec_from_file_location('ex', scripts + '/probe_fit_step_example.py'); ex = importlib.util.module_from_spec(spec); spec.loader.exec_module(ex)
import t3toolbox.backend.t3_constructors as t3c, t3toolbox.backend.fv_conversions as fvc, t3toolbox.backend.tv_operations as tvo
sitewise, sweep, CARRY = ex.sitewise, ex.sweep, ex.CARRY
UP, RIGHT, LEFT, DOWN, OUT, GRAD_G, GRAD_B = ex.UP, ex.RIGHT, ex.LEFT, ex.DOWN, ex.OUT, ex.GRAD_G, ex.GRAD_B

np.random.seed(0); rng = np.random.default_rng(0)
shape, tucker_ranks, tt_ranks, W = (40, 50, 45, 60), (4, 5, 4, 3), (1, 3, 4, 3, 1), 30
B, G = t3c.t3_corewise_randn(shape, tucker_ranks, tt_ranks)
w = [rng.standard_normal((W, N)) for N in shape]
frame, _ = fvc.t3_orthogonal_representations((B, G)); D, U, L, R = frame
b_L, b_R = np.ones((W, 1)), np.ones((W, 1)); zL, zR = np.zeros_like(b_L), np.zeros_like(b_R)
xi = sitewise(UP, [(D, w)]); mu = sweep(RIGHT, b_L, [(CARRY, L, xi)]); nu = sweep(LEFT, b_R, [(R, xi, CARRY)], reverse=True)
eta = sitewise(DOWN, [(mu, U, nu[1:])])
project = lambda v: tvo.tv_orthogonal_gauge_projection(frame, v)

def tangent(V, H):                                   # grade t, up to eta_t (no ambient array yet)
    xi_t = sitewise(UP, [(V, w)])
    mu_t = sweep(RIGHT, zL, [(CARRY, R, xi), (mu, H, xi), (mu, U, xi_t)])
    nu_t = sweep(LEFT, zR, [(L, xi, CARRY), (H, xi, nu[1:]), (U, xi_t, nu[1:])], reverse=True)
    return sitewise(DOWN, [(mu_t, R, nu[1:]), (mu, H, nu[1:]), (mu, L, nu_t[1:])])

def transpose_tail(xi_z):                            # grade z, from xi_z on
    mu_z = sweep(RIGHT, zL, [(CARRY, L, xi), (mu, L, xi_z)])
    nu_z = sweep(LEFT, zR, [(R, xi, CARRY), (R, xi_z, nu[1:])], reverse=True)
    eta_z = sitewise(DOWN, [(mu_z, U, nu[1:]), (mu, U, nu_z[1:])])
    dH = sitewise(GRAD_G, [(mu_z, xi, nu[1:]), (mu, xi_z, nu[1:]), (mu, xi, nu_z[1:])], sum_w=True)
    return eta_z, dH

V, H = project(([rng.standard_normal((nU.shape[1], N)) for nU, N in zip(U, shape)], [rng.standard_normal((l.shape[0], l.shape[1], r.shape[2])) for l, r in zip(L, R)]))

# reference: through the ambient space, as today
eta_t = tangent(V, H)
Jp = sitewise(OUT, [(eta_t, D), (eta, V)])                                   # W x N per mode
xi_z = sitewise(UP, [(D, Jp)])
eta_z, dH = transpose_tail(xi_z)
dV = sitewise(GRAD_B, [(eta_z, w), (eta, Jp)], sum_w=True)
ref = project((dV, dH))

# fused: never form Jp
eta_z2, dH2 = transpose_tail(eta_t)
dV2 = [np.einsum('wi,wo->io', ez, wi) + np.einsum('wi,wj->ij', e, et) @ Di + np.einsum('wi,wj->ij', e, e) @ Vi
       for ez, wi, e, et, Di, Vi in zip(eta_z2, w, eta, eta_t, D, V)]
fused = project((dV2, dH2))

flat = lambda v: np.concatenate([np.ravel(a) for part in v for a in part])
print('xi_z vs eta_t                      :', max(np.linalg.norm(a - b) / np.linalg.norm(b) for a, b in zip(xi_z, eta_t)))
print('fused J^T J p vs through-ambient   :', np.linalg.norm(flat(fused) - flat(ref)) / np.linalg.norm(flat(ref)))
print('largest array formed, through-ambient: W x N = %d x %d;  fused: W x n = %d x %d' % (W, max(shape), W, max(tucker_ranks)))
