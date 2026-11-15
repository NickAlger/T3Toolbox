"""Which per-sample edge variables can be shared between operations, and in which gauge.

Checks, for probes at order 0:
  1. Plain components swept through the point's own cores (B, G) are NOT the frame's components.
  2. Swept through the base point (D, L): xi and mu ARE the frame's; nu is not (the frame's nu goes through R).
  3. The two nu's differ by a bond-sized gauge matrix:  nu_L[i] = S_i nu_R[i].  The same holds at the residual grade.
  4. The plain probe output (hence the residual and the objective) can be read off the frame's components and the
     point's center cores, with no sweep through the point's own cores.
  5. The library's cached frame sweep is exactly these components; its exact-Hessian kernel re-sweeps nu through L.

Usage:  python3 reuse_check.py /path/to/T3Toolbox
"""
import importlib.util
import pathlib
import sys

import numpy as np

if len(sys.argv) > 1:
    sys.path.append(sys.argv[1])

HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('plan_mockup', HERE / 'plan_mockup.py')
pm = importlib.util.module_from_spec(_spec)
sys.modules['plan_mockup'] = pm
_spec.loader.exec_module(pm)
UP, RIGHT, LEFT, DOWN, OUT, CARRY = pm.UP, pm.RIGHT, pm.LEFT, pm.DOWN, pm.OUT, pm.CARRY
sitewise, sweep = pm.sitewise, pm.sweep


def rel(a, b):
    if any(p.shape != q.shape for p, q in zip(a, b)):
        return 'different shapes'
    num = np.sqrt(sum(np.sum((p - q) ** 2) for p, q in zip(a, b)))
    return '%.1e' % (num / np.sqrt(sum(np.sum(q ** 2) for q in b)))


def main():
    import t3toolbox.backend.fv_conversions as fv_conversions
    import t3toolbox.backend.probing as probing

    rng = np.random.default_rng(0)
    np.random.seed(0)
    shape, tucker_ranks, tt_ranks, W = (12, 11, 13, 10), (4, 5, 4, 3), (1, 5, 6, 4, 1), 50
    x = pm.random_point(shape, tucker_ranks, tt_ranks)
    B, G = x
    frame, (Vc, Hc) = fv_conversions.t3_orthogonal_representations(x)     # the frame, and the point's center cores
    D, U, L, R = frame
    d = len(D)
    w = [rng.standard_normal((W, N)) for N in shape]
    r = [rng.standard_normal((W, N)) for N in shape]                      # a stand-in residual
    plan = pm.Plan()
    ones, zero = np.ones((W, 1)), np.zeros((W, 1))

    def plain(tucker, left, right):
        xi = sitewise(UP, [(tucker, w)])
        mu = sweep(RIGHT, ones, [(CARRY, left, xi)], plan)
        nu = sweep(LEFT, ones, [(right, xi, CARRY)], plan, reverse=True)
        return xi, mu, nu

    xi, mu, nu = plain(D, L, R)                                           # the frame's components
    eta = sitewise(DOWN, [(mu, U, nu[1:])])
    xi_x, mu_x, nu_x = plain(B, G, G)                                     # through the point's own cores
    xi_b, mu_b, nu_b = plain(D, L, L)                                     # through the base point (D, L)

    print('1. through the point\'s own cores (B, G), against the frame\'s components')
    print('   xi %s   mu %s   nu %s' % (rel(xi_x, xi), rel(mu_x[:d], mu[:d]), rel(nu_x[1:], nu[1:])))
    print('2. through the base point (D, L), against the frame\'s components')
    print('   xi %s   mu %s   nu %s' % (rel(xi_b, xi), rel(mu_b[:d], mu[:d]), rel(nu_b[1:], nu[1:])))

    # the gauge matrices S_i on bonds 0..d:  L_i ... L_{d-1} = S_i R_i ... R_{d-1}
    S = [None] * (d + 1)
    S[d] = np.ones((1, 1))
    for i in reversed(range(d)):
        S[i] = np.einsum('aib,bc,eic->ae', L[i], S[i + 1], R[i])
    print('3. nu through L against S nu through R (bonds 1..d-1):          %s'
          % rel(nu_b[1:d], [n @ s.T for n, s in zip(nu[1:d], S[1:d])]))

    xi_z = sitewise(UP, [(D, r)])                                         # residual grade
    nu_z = sweep(LEFT, zero, [(R, xi, CARRY), (R, xi_z, nu[1:])], plan, reverse=True)
    nu_zb = sweep(LEFT, zero, [(L, xi, CARRY), (L, xi_z, nu_b[1:])], plan, reverse=True)
    print('   residual grade: nu_z through L against S nu_z through R:     %s'
          % rel(nu_zb[1:d], [n @ s.T for n, s in zip(nu_z[1:d], S[1:d])]))

    z_ref = probing.t3_probe(w, x)
    z_tucker = sitewise(OUT, [(eta, Vc)])                                 # eta and the Tucker center cores
    z_tt = sitewise(OUT, [(sitewise(DOWN, [(mu, Hc, nu[1:])]), D)])       # mu, nu and the TT center cores
    print('4. probe output from the frame\'s eta and the Tucker center cores: %s' % rel(z_tucker, z_ref))
    print('   probe output from the frame\'s mu, nu and the TT center cores:  %s' % rel(z_tt, z_ref))

    lib = probing.tv_precompute_probe_frame_sweep(frame, w)               # (xis, mus, nus, etas)
    print('5. the library\'s cached probe frame sweep against these components')
    print('   xi %s   mu %s   nu %s   eta %s' % (rel(lib[0], xi), rel(lib[1], mu[:d]), rel(lib[2], nu[1:]),
                                                   rel(lib[3], eta)))
    print('   the exact-Hessian kernel\'s re-swept nu, compute_nu(L, xi), against nu through L: %s'
          % rel(probing.compute_nu(L, lib[0]), nu_b[1:]))


if __name__ == '__main__':
    main()
