"""Do apply and entries use the same per-site functions as probe, differing only in the leaf and the readout?

Apply and entries read the output at the right boundary; the seed is one scalar per sample and enters there.
Entries makes xi by a gather.  Everything else is the probe case's step functions.  Checked against T3Toolbox.

Usage:  python3 -I kind_readout_check.py /path/to/T3Toolbox
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unbundled_examples import (each_site, rightward as left_to_right, leftward as right_to_left,
                                xi_site as xi_step, mu_step, nu_step, eta_site as eta_step,
                                xi_t_site as xi_dot_step, mu_t_step as mu_dot_step,
                                grad_G as H_term, grad_B as V_term)


def xi_step_entries(D, index):                 # the only function entries does not share: a gather
    return D[:, index].T


def main(repo):
    sys.path.append(repo)
    import t3toolbox.backend.apply as apply
    import t3toolbox.backend.entries as entries

    rng = np.random.default_rng(0)
    rn = rng.standard_normal
    Ns, nD, nU = (6, 5, 7, 4), (3, 4, 2, 3), (2, 3, 3, 2)
    rL, rR = (1, 3, 2, 4, 1), (1, 2, 4, 3, 1)
    d, W = len(Ns), 5
    D = [rn((nD[i], Ns[i])) for i in range(d)]
    U = [rn((rL[i], nU[i], rR[i + 1])) for i in range(d)]
    L = [rn((rL[i], nD[i], rL[i + 1])) for i in range(d)]
    R = [rn((rR[i], nD[i], rR[i + 1])) for i in range(d)]
    V = [rn((nU[i], Ns[i])) for i in range(d)]
    H = [rn((rL[i], nD[i], rR[i + 1])) for i in range(d)]
    frame, variation = (D, U, L, R), (V, H)
    w = [rn((W, N)) for N in Ns]
    index = np.stack([rng.integers(0, N, size=W) for N in Ns])
    c = rn(W)                                                       # the seed: one scalar per sample
    one, zero = np.ones((W, 1)), np.zeros((W, 1))
    err = lambda mine, theirs: max(np.linalg.norm(np.asarray(a) - np.asarray(b)) / np.linalg.norm(np.asarray(b))
                                   for a, b in zip(mine, theirs))

    for kind, sample, lib in (('apply', w, apply), ('entries', index, entries)):
        if kind == 'apply':
            xi = each_site(xi_step, d, D, sample)
            xi_U_dot = each_site(xi_dot_step, d, V, sample)
            leaf = sample                                           # what the Tucker gradient block pairs with
        else:
            xi = each_site(xi_step_entries, d, D, sample)
            xi_U_dot = each_site(xi_step_entries, d, V, sample)
            leaf = [np.eye(N)[sample[i]] for i, N in enumerate(Ns)]
        mu = left_to_right(mu_step, one, d, L, xi)
        z = mu[d][:, 0]                                             # the readout: the right boundary

        mu_R_dot = left_to_right(mu_dot_step, zero, d, mu, xi, xi_U_dot, R, U, H)
        Jp = mu_R_dot[d][:, 0]

        nu_tilde = right_to_left(nu_step, c[:, None], d, R, xi)     # the plain step, started from the seed
        eta_tilde = each_site(eta_step, d, mu, U, nu_tilde[1:])
        V_tilde = each_site(V_term, d, eta_tilde, leaf)
        H_tilde = each_site(H_term, d, mu, xi, nu_tilde[1:])

        f = getattr(lib, 't3_' + kind)
        z_ref = f((D, L), sample)
        Jp_ref = getattr(lib, 'tv_' + kind)(sample, variation, frame)
        g_ref = getattr(lib, 'tv_%s_transpose' % kind)(c, sample, frame, sum_over_probes=True)
        print('%-8s z %.1e   Jp %.1e   Jt c %.1e' % (kind, err([z], [z_ref]), err([Jp], [Jp_ref]),
                                                     err(V_tilde + H_tilde, list(g_ref[0]) + list(g_ref[1]))))


if __name__ == '__main__':
    main(sys.argv[1])
