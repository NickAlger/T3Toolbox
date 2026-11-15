"""Target decision 2.4: a tilde jet is a Lagrange multiplier, and that is where the reversal comes from.

On the manifold frame the tilde sweeps are the adjoint of the LINEAR program  (V, H) -> Jp.  Its variables are the
dot variables.  Claim, with jets of order n and a functional phi on Jp given in forward order
(F = sum_i sum_t phi_i[t] Jp_i[t]):

    the tilde of a letter is the multiplier of the DOT of its partner on the same edge,
        xi_tilde <-> eta_dot      eta_tilde <-> xi_dot      mu_tilde <-> nu_dot      nu_tilde <-> mu_dot
    and the pairing is the top coefficient of the product:
        dF = sum over the sample and network indices of  sum_t tilde[n - t] * delta[t]
    when the partner's dot variable is perturbed by a jet delta and everything downstream is recomputed.

The network index is paired plainly (the two variables point opposite ways on the edge).  The order index is
paired reversed (both are stored as jets; the multiplier is a dual jet).

The program is the explicit one of jet_steps_check.py.  numpy only.   Usage:  python3 -I tilde_multiplier_check.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jet_steps_check as J


def main():
    rng = np.random.default_rng(0)
    rn = rng.standard_normal
    Ns, nD, nU = (6, 5, 7, 4), (3, 4, 2, 3), (2, 3, 3, 2)
    rL, rR = (1, 3, 2, 4, 1), (1, 2, 4, 3, 1)
    d, W, order = len(Ns), 5, 3
    P = order + 1
    D = [rn((nD[i], Ns[i])) for i in range(d)]
    U = [rn((rL[i], nU[i], rR[i + 1])) for i in range(d)]
    L = [rn((rL[i], nD[i], rL[i + 1])) for i in range(d)]
    R = [rn((rR[i], nD[i], rR[i + 1])) for i in range(d)]
    V = [rn((nU[i], Ns[i])) for i in range(d)]
    H = [rn((rL[i], nD[i], rR[i + 1])) for i in range(d)]
    w = [J.input_jet(P, rn((W, N)), rn((W, N))) for N in Ns]
    phi = [rn((P, W, N)) for N in Ns]                       # a functional on Jp, forward order

    xi = J.each_site(J.xi_step, d, D, w)
    mu = J.left_to_right(J.mu_step, J.ones_start(P, W, 1), d, L, xi)
    nu = J.right_to_left(J.nu_step, J.ones_start(P, W, 1), d, R, xi)
    eta = J.each_site(J.eta_step, d, mu, U, nu[1:])
    zeros = J.zeros_start(P, W, 1)

    def F(bump=None):                                       # bump = (name, position, delta)
        add = lambda name, k, a: a + bump[2] if bump and bump[:2] == (name, k) else a
        xi_dot = [add('xi_dot', i, J.xi_dot_step(V[i], w[i])) for i in range(d)]
        mu_dot = [zeros]
        for i in range(d):
            mu_dot.append(add('mu_dot', i + 1, J.mu_dot_step(mu_dot[i], mu[i], xi[i], xi_dot[i], R[i], U[i], H[i])))
        nu_dot = [None] * d + [zeros]
        for i in reversed(range(d)):
            nu_dot[i] = add('nu_dot', i, J.nu_dot_step(nu_dot[i + 1], nu[i + 1], xi[i], xi_dot[i], L[i], U[i], H[i]))
        eta_dot = [add('eta_dot', i, J.eta_dot_step(mu_dot[i], mu[i], nu_dot[i + 1], nu[i + 1], L[i], R[i], H[i]))
                   for i in range(d)]
        Jp = [J.Jp_step(eta_dot[i], eta[i], D[i], V[i]) for i in range(d)]
        return sum(float(np.sum(phi[i] * Jp[i])) for i in range(d)), dict(xi_dot=xi_dot, mu_dot=mu_dot, nu_dot=nu_dot, eta_dot=eta_dot)

    F0, dots = F()
    seed = [f[::-1] for f in phi]                           # the functional, turned round on the order leg
    xi_tilde = J.each_site(J.xi_tilde_step, d, D, seed)
    mu_tilde = J.left_to_right(J.mu_tilde_step, zeros, d, mu, xi, xi_tilde, L)
    nu_tilde = J.right_to_left(J.nu_tilde_step, zeros, d, nu[1:], xi, xi_tilde, R)
    eta_tilde = J.each_site(J.eta_tilde_step, d, mu_tilde, mu, nu_tilde[1:], nu[1:], U)

    top_pairing = lambda tilde, delta: float(np.sum(tilde[::-1] * delta))       # sum_t tilde[n-t] delta[t]
    slot_pairing = lambda tilde, delta: float(np.sum(tilde * delta))            # sum_t tilde[t] delta[t]
    print('order n = %d.  dF when the dot variable is perturbed, against the pairing with the tilde variable' % order)
    print('   %-10s %-10s %-22s %-22s' % ('perturbed', 'multiplier', 'top coefficient', 'slot by slot (wrong)'))
    for dot_name, tilde_name, tilde, positions in (('eta_dot', 'xi_tilde', xi_tilde, range(d)),
                                                   ('xi_dot', 'eta_tilde', eta_tilde, range(d)),
                                                   ('nu_dot', 'mu_tilde', mu_tilde, range(1, d)),
                                                   ('mu_dot', 'nu_tilde', nu_tilde, range(1, d))):
        worst_top, worst_slot = 0.0, 0.0
        for k in positions:
            delta = rn(dots[dot_name][k].shape)
            dF = F((dot_name, k, delta))[0] - F0
            worst_top = max(worst_top, abs(dF - top_pairing(tilde[k], delta)) / abs(dF))
            worst_slot = max(worst_slot, abs(dF - slot_pairing(tilde[k], delta)) / abs(dF))
        print('   %-10s %-10s %-22s %-22s' % (dot_name, tilde_name, 'relative error %.0e' % worst_top,
                                             'relative error %.0e' % worst_slot))


if __name__ == '__main__':
    main()
