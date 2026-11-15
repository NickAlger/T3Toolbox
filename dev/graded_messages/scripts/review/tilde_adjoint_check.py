"""In what sense are the residual-grade ("tilde") variables adjoints?  Two checks on a corewise frame.

F = sum_i <s_i, z_i> with the forward constraints  xi = UP, mu = RIGHT, nu = LEFT, eta = DOWN, z = OUT.

  1. Multiplier sense: dF/d(variable), with the variable perturbed as an independent quantity and everything
     downstream of it recomputed, equals the tilde variable of its PARTNER on the same edge:
         dF/d mu = nu_tilde,   dF/d nu = mu_tilde,   dF/d xi = eta_tilde,   dF/d eta = xi_tilde.
  2. Forward sense: each tilde variable is the derivative of the variable it is named after, with respect to
     the sample vectors, in the direction of the seed s.

Usage:  python3 -I tilde_adjoint_check.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unbundled_examples import (up, right, left, down, out, each_site, rightward, leftward,
                                xi_site, mu_step, nu_step, eta_site, z_site,
                                xi_z_site, mu_z_step, nu_z_step, eta_z_site)

rng = np.random.default_rng(1)
rn = rng.standard_normal
Ns, n, r = (6, 5, 7, 4), (3, 4, 2, 3), (1, 3, 2, 4, 1)
d, W = len(Ns), 5
B = [rn((n[i], Ns[i])) for i in range(d)]
G = [rn((r[i], n[i], r[i + 1])) for i in range(d)]
w = [rn((W, N)) for N in Ns]
s = [rn((W, N)) for N in Ns]
one, zero = np.ones((W, 1)), np.zeros((W, 1))


def forward(w, bump=None):
    """All plain variables.  bump = (name, site, delta) adds delta to one variable as it is produced."""
    add = lambda name, i, a: a + bump[2] if bump and bump[:2] == (name, i) else a
    xi = [add('xi', i, xi_site(B[i], w[i])) for i in range(d)]
    mu = [one]
    for i in range(d):
        mu.append(add('mu', i + 1, mu_step(mu[i], G[i], xi[i])))
    nu = [None] * d + [one]
    for i in reversed(range(d)):
        nu[i] = add('nu', i, nu_step(nu[i + 1], G[i], xi[i]))
    eta = [add('eta', i, eta_site(mu[i], G[i], nu[i + 1])) for i in range(d)]
    z = [z_site(eta[i], B[i]) for i in range(d)]
    return xi, mu, nu, eta, sum(float(np.sum(s[i] * z[i])) for i in range(d))


xi, mu, nu, eta, F = forward(w)
xi_tilde = each_site(xi_z_site, d, B, s)
mu_tilde = rightward(mu_z_step, zero, d, mu, xi, xi_tilde, G)
nu_tilde = leftward(nu_z_step, zero, d, nu[1:], xi, xi_tilde, G)
eta_tilde = each_site(eta_z_site, d, mu_tilde, mu, nu_tilde[1:], nu[1:], G)

print('1. multiplier sense: dF/d(variable) against the partner tilde variable (F is multilinear, so exact)')
for name, i, like, partner, label in (('mu', 2, mu[2], nu_tilde[2], 'nu_tilde'), ('nu', 2, nu[2], mu_tilde[2], 'mu_tilde'),
                                      ('xi', 1, xi[1], eta_tilde[1], 'eta_tilde'), ('eta', 1, eta[1], xi_tilde[1], 'xi_tilde')):
    delta = rn(like.shape)
    dF = forward(w, (name, i, delta))[4] - F                      # F is affine in each single variable
    pred = float(np.sum(partner * delta))
    print('   dF/d %-3s at site/bond %d  =  <%s, delta>     rel err %.1e' % (name, i, label, abs(dF - pred) / abs(pred)))

print('2. forward sense: d(variable)/d(sample) along s against the tilde variable of the same name')
eps = 1e-6
plus = forward([a + eps * b for a, b in zip(w, s)])
minus = forward([a - eps * b for a, b in zip(w, s)])
for k, (label, tilde) in enumerate((('xi', xi_tilde), ('mu', mu_tilde), ('nu', nu_tilde), ('eta', eta_tilde))):
    fd = [(a - b) / (2 * eps) for a, b in zip(plus[k], minus[k])]
    err = max(np.linalg.norm(a - b) / np.linalg.norm(b) for a, b in zip(fd, tilde) if np.linalg.norm(b) > 0)
    print('   d %-3s/dw along s  =  %s_tilde     rel err %.1e (central difference)' % (label, label, err))
