"""A check of the type catalog of the sampling variables (target decision 2.4).

Each variable is an element of  (a space of the network) (x) (a space of the order).  The catalog says:

  network factor   w, xi, mu are VECTORS; eta, nu (in either basis) and z are COVECTORS; in every grade.
                   Which space: N (mode), X^D and X^U (the two Tucker-rank families), A^L and A^R (the two bond
                   families of the manifold frame).
  order factor     plain and dot variables are JETS (R); tilde and tilde-dot variables are DUAL JETS (R*).
  blocks           V_tilde, H_tilde, and the tilde-dot blocks, are real covectors on the space of the cores.

A type is a transformation rule, so it can be tested.  Part A changes the basis of every network space by an
arbitrary invertible matrix and checks that every variable transforms as the catalog says.  That passes only if
no inner product of a network space is used anywhere in the program.  Part B shows the one place where an inner
product is used: turning a residual into a seed.  Part C does the same for the order factor, under the change of
variable s -> c s.

The program is the explicit one of jet_steps_check.py, imported from beside this file.  numpy only; nothing is
read from T3Toolbox.  The names are those of SAMPLE_VARIABLES.md (2026-11-15, 23:23).

Usage:  python3 -I type_catalog_check.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jet_steps_check as J


# ---- the program: every variable of every grade, on the manifold frame and (for Kp) at the base point ----
def program(frame, V, H, D_dot, L_dot, w, w_tilde, d, P, W):
    D, U, L, R = frame
    v = {}
    v['w'] = w
    v['xi'] = J.each_site(J.xi_step, d, D, w)
    v['mu'] = J.left_to_right(J.mu_step, J.ones_start(P, W, 1), d, L, v['xi'])
    v['nu'] = J.right_to_left(J.nu_step, J.ones_start(P, W, 1), d, R, v['xi'])
    v['eta'] = J.each_site(J.eta_step, d, v['mu'], U, v['nu'][1:])
    v['nu_L'] = J.right_to_left(J.nu_step, J.ones_start(P, W, 1), d, L, v['xi'])
    v['z'] = J.each_site(J.z_step, d, v['mu'], L, v['nu_L'][1:], D)
    zeros = J.zeros_start(P, W, 1)
    # dot, the frame's form: along the direction (V, H)
    v['xi_U_dot'] = J.each_site(J.xi_dot_step, d, V, w)
    v['mu_R_dot'] = J.left_to_right(J.mu_dot_step, zeros, d, v['mu'], v['xi'], v['xi_U_dot'], R, U, H)
    v['nu_L_dot'] = J.right_to_left(J.nu_dot_step, zeros, d, v['nu'][1:], v['xi'], v['xi_U_dot'], L, U, H)
    v['eta_D_dot'] = J.each_site(J.eta_dot_step, d, v['mu_R_dot'], v['mu'], v['nu_L_dot'][1:], v['nu'][1:], L, R, H)
    v['Jp'] = J.each_site(J.Jp_step, d, v['eta_D_dot'], v['eta'], D, V)
    # tilde
    v['w_tilde'] = w_tilde
    v['xi_tilde'] = J.each_site(J.xi_tilde_step, d, D, w_tilde)
    v['mu_tilde'] = J.left_to_right(J.mu_tilde_step, zeros, d, v['mu'], v['xi'], v['xi_tilde'], L)
    v['nu_tilde'] = J.right_to_left(J.nu_tilde_step, zeros, d, v['nu'][1:], v['xi'], v['xi_tilde'], R)
    v['eta_tilde'] = J.each_site(J.eta_tilde_step, d, v['mu_tilde'], v['mu'], v['nu_tilde'][1:], v['nu'][1:], U)
    v['V_tilde'] = J.top(J.each_site(J.V_tilde_step, d, v['eta_tilde'], v['eta'], w, w_tilde))
    v['H_tilde'] = J.top(J.each_site(J.H_tilde_step, d, v['mu_tilde'], v['mu'], v['xi_tilde'], v['xi'],
                                     v['nu_tilde'][1:], v['nu'][1:]))
    # tilde-dot, at the base point (D, L), with the raw core direction (D_dot, L_dot).  raw_nu_L_dot and
    # raw_eta_D_dot are a local spelling of this script, not notation: the raw form's nu_L_dot and eta_D_dot, which
    # differ from the frame's above because (D_dot, L_dot) is not the lift of (V, H).
    v['nu_L_tilde'] = J.right_to_left(J.nu_tilde_step, zeros, d, v['nu_L'][1:], v['xi'], v['xi_tilde'], L)
    v['xi_dot'] = J.each_site(J.xi_dot_step, d, D_dot, w)
    v['mu_dot'] = J.left_to_right(J.mu_dot_step, zeros, d, v['mu'], v['xi'], v['xi_dot'], L, L, L_dot)
    v['raw_nu_L_dot'] = J.right_to_left(J.nu_dot_step, zeros, d, v['nu_L'][1:], v['xi'], v['xi_dot'], L, L, L_dot)
    v['raw_eta_D_dot'] = J.each_site(J.eta_dot_step, d, v['mu_dot'], v['mu'], v['raw_nu_L_dot'][1:], v['nu_L'][1:],
                                     L, L, L_dot)
    v['xi_tilde_dot'] = J.each_site(J.xi_tilde_dot_step, d, D_dot, w_tilde)
    v['mu_tilde_dot'] = J.left_to_right(J.mu_tilde_dot_step, zeros, d, v['mu_dot'], v['mu_tilde'], v['mu'],
                                        v['xi_tilde_dot'], v['xi_dot'], v['xi_tilde'], v['xi'], L, L_dot)
    v['nu_L_tilde_dot'] = J.right_to_left(J.nu_tilde_dot_step, zeros, d, v['raw_nu_L_dot'][1:], v['nu_L_tilde'][1:],
                                          v['nu_L'][1:], v['xi_tilde_dot'], v['xi_dot'], v['xi_tilde'], v['xi'],
                                          L, L_dot)
    v['eta_D_tilde_dot'] = J.each_site(J.eta_tilde_dot_step, d, v['mu_tilde_dot'], v['mu_dot'], v['mu_tilde'],
                                       v['mu'], v['nu_L_tilde_dot'][1:], v['raw_nu_L_dot'][1:], v['nu_L_tilde'][1:],
                                       v['nu_L'][1:], L, L_dot)
    v['V_tilde_dot'] = J.top(J.each_site(J.V_tilde_dot_step, d, v['eta_D_tilde_dot'], v['raw_eta_D_dot'], w, w_tilde))
    v['H_tilde_dot'] = J.top(J.each_site(J.H_tilde_dot_step, d, v['mu_tilde_dot'], v['mu_dot'], v['mu_tilde'],
                                         v['mu'], v['xi_tilde_dot'], v['xi_dot'], v['xi_tilde'], v['xi'],
                                         v['nu_L_tilde_dot'][1:], v['raw_nu_L_dot'][1:], v['nu_L_tilde'][1:],
                                         v['nu_L'][1:]))
    return v


# ---- the catalog: for each variable, where it sits and how each of its network indices transforms ----------
# 'vec'  : x' = x S           (last index)          'cov' : x' = x S^-T
SITE, BOND = 'site', 'bond'
CATALOG = (                      # name, where, ((variance, space), ...) for its network indices, order factor
    ('w',               SITE, (('vec', 'N'),),                              'jet'),
    ('xi',              SITE, (('vec', 'XD'),),                             'jet'),
    ('mu',              BOND, (('vec', 'AL'),),                             'jet'),
    ('nu',              BOND, (('cov', 'AR'),),                             'jet'),
    ('eta',             SITE, (('cov', 'XU'),),                             'jet'),
    ('nu_L',            BOND, (('cov', 'AL'),),                             'jet'),
    ('z',               SITE, (('cov', 'N'),),                              'jet'),
    ('xi_U_dot',        SITE, (('vec', 'XU'),),                             'jet'),
    ('mu_R_dot',        BOND, (('vec', 'AR'),),                             'jet'),
    ('nu_L_dot',        BOND, (('cov', 'AL'),),                             'jet'),
    ('eta_D_dot',       SITE, (('cov', 'XD'),),                             'jet'),
    ('Jp',              SITE, (('cov', 'N'),),                              'jet'),
    ('xi_dot',          SITE, (('vec', 'XD'),),                             'jet'),
    ('mu_dot',          BOND, (('vec', 'AL'),),                             'jet'),
    ('raw_nu_L_dot',    BOND, (('cov', 'AL'),),                             'jet'),
    ('raw_eta_D_dot',   SITE, (('cov', 'XD'),),                             'jet'),
    ('w_tilde',         SITE, (('vec', 'N'),),                              'dual jet'),
    ('xi_tilde',        SITE, (('vec', 'XD'),),                             'dual jet'),
    ('mu_tilde',        BOND, (('vec', 'AL'),),                             'dual jet'),
    ('nu_tilde',        BOND, (('cov', 'AR'),),                             'dual jet'),
    ('eta_tilde',       SITE, (('cov', 'XU'),),                             'dual jet'),
    ('nu_L_tilde',      BOND, (('cov', 'AL'),),                             'dual jet'),
    ('xi_tilde_dot',    SITE, (('vec', 'XD'),),                             'dual jet'),
    ('mu_tilde_dot',    BOND, (('vec', 'AL'),),                             'dual jet'),
    ('nu_L_tilde_dot',  BOND, (('cov', 'AL'),),                             'dual jet'),
    ('eta_D_tilde_dot', SITE, (('cov', 'XD'),),                             'dual jet'),
    ('V_tilde',         SITE, (('cov', 'XU'), ('vec', 'N')),                'real'),      # dual of the space of V
    ('H_tilde',         SITE, (('vec', 'AL'), ('vec', 'XD'), ('cov', 'AR+')), 'real'),    # dual of the space of H
    ('V_tilde_dot',     SITE, (('cov', 'XD'), ('vec', 'N')),                'real'),      # dual of the space of D
    ('H_tilde_dot',     SITE, (('vec', 'AL'), ('vec', 'XD'), ('cov', 'AL+')), 'real'),    # dual of the space of L
)
GRADE = lambda name: ('tilde-dot' if 'tilde_dot' in name else 'tilde' if 'tilde' in name
                      else 'dot' if ('dot' in name or name == 'Jp') else 'plain')


def matrix(S, i, space):                       # the change-of-basis matrix of `space` at site or bond i
    return S[space[:-1]][i + 1] if space.endswith('+') else S[space][i]


def move(x, slots, S, i):                      # transform the trailing len(slots) indices of x
    k = len(slots)
    for j, (variance, space) in enumerate(slots):
        M = matrix(S, i, space)
        M = M if variance == 'vec' else np.linalg.inv(M).T
        x = np.moveaxis(np.tensordot(x, M, axes=([x.ndim - k + j], [0])), -1, x.ndim - k + j)
    return x


def main():
    rng = np.random.default_rng(0)
    rn = rng.standard_normal
    Ns, nD, nU = (6, 5, 7, 4), (3, 4, 2, 3), (2, 3, 3, 2)
    rL, rR = (1, 3, 2, 4, 1), (1, 2, 4, 3, 1)
    d, W, order = len(Ns), 5, 2
    P = order + 1
    D = [rn((nD[i], Ns[i])) for i in range(d)]
    U = [rn((rL[i], nU[i], rR[i + 1])) for i in range(d)]
    L = [rn((rL[i], nD[i], rL[i + 1])) for i in range(d)]
    R = [rn((rR[i], nD[i], rR[i + 1])) for i in range(d)]
    V = [rn((nU[i], Ns[i])) for i in range(d)]
    H = [rn((rL[i], nD[i], rR[i + 1])) for i in range(d)]
    D_dot, L_dot = [rn(b.shape) for b in D], [rn(g.shape) for g in L]
    ww, pp = [rn((W, N)) for N in Ns], [rn((W, N)) for N in Ns]
    w = [J.input_jet(P, ww[i], pp[i]) for i in range(d)]
    w_tilde = [rn((P, W, N)) for N in Ns]
    flat = lambda blocks: np.concatenate([np.ravel(np.asarray(b)) for b in blocks])
    err = lambda a, b: np.linalg.norm(flat(a) - flat(b)) / np.linalg.norm(flat(b))

    def invertible(n):                         # well conditioned, far from orthogonal
        if n == 1:
            return np.ones((1, 1))             # the boundary vector is ones: no change of basis at the ends
        q1, q2 = np.linalg.qr(rn((n, n)))[0], np.linalg.qr(rn((n, n)))[0]
        return q1 @ np.diag(0.5 + 1.5 * rng.random(n)) @ q2

    def change(S):                             # the frame, directions and inputs in the new bases
        core = lambda G, i, a, x, b: move(G, (('cov', a), ('cov', x), ('vec', b + '+')), S, i)
        tucker = lambda B, i, x: move(B, (('vec', x), ('cov', 'N')), S, i)
        return ([tucker(D[i], i, 'XD') for i in range(d)], [core(U[i], i, 'AL', 'XU', 'AR') for i in range(d)],
                [core(L[i], i, 'AL', 'XD', 'AL') for i in range(d)], [core(R[i], i, 'AR', 'XD', 'AR') for i in range(d)],
                [tucker(V[i], i, 'XU') for i in range(d)], [core(H[i], i, 'AL', 'XD', 'AR') for i in range(d)],
                [tucker(D_dot[i], i, 'XD') for i in range(d)], [core(L_dot[i], i, 'AL', 'XD', 'AL') for i in range(d)])

    base = program((D, U, L, R), V, H, D_dot, L_dot, w, w_tilde, d, P, W)

    # ---- Part A --------------------------------------------------------------------------------------------
    S = {'N': [invertible(N) for N in Ns], 'XD': [invertible(n) for n in nD], 'XU': [invertible(n) for n in nU],
         'AL': [invertible(r) for r in rL], 'AR': [invertible(r) for r in rR]}
    D2, U2, L2, R2, V2, H2, D_dot2, L_dot2 = change(S)
    new = program((D2, U2, L2, R2), V2, H2, D_dot2, L_dot2,
                  [move(w[i], (('vec', 'N'),), S, i) for i in range(d)],
                  [move(w_tilde[i], (('vec', 'N'),), S, i) for i in range(d)], d, P, W)
    print('Part A. Arbitrary invertible changes of basis in N, X^D, X^U, A^L, A^R (condition numbers up to %.0f).'
          % max(np.linalg.cond(m) for ms in S.values() for m in ms))
    print('        Each variable in the new bases against the catalog\'s rule applied to the old one:')
    worst = {}
    for name, where, slots, _ in CATALOG:
        predicted = [move(x, slots, S, i) for i, x in enumerate(base[name])]
        worst[GRADE(name)] = max(worst.get(GRADE(name), 0.0), err(new[name], predicted))
    for grade in ('plain', 'dot', 'tilde', 'tilde-dot'):
        names = [n for n, _, _, _ in CATALOG if GRADE(n) == grade]
        print('        %-9s (%2d variables: %s)  worst relative difference %.0e' % (grade, len(names), ', '.join(names), worst[grade]))
    pair = lambda blocks, dirs: sum(float(np.sum(a * b)) for a, b in zip(blocks, dirs))
    print('        the blocks are covectors on the cores: <V_tilde, V> + <H_tilde, H> is unchanged: %.0e;  the same for Kp: %.0e'
          % (abs(pair(new['V_tilde'], V2) + pair(new['H_tilde'], H2) - pair(base['V_tilde'], V) - pair(base['H_tilde'], H))
             / abs(pair(base['V_tilde'], V) + pair(base['H_tilde'], H)),
             abs(pair(new['V_tilde_dot'], D_dot2) + pair(new['H_tilde_dot'], L_dot2)
                 - pair(base['V_tilde_dot'], D_dot) - pair(base['H_tilde_dot'], L_dot))
             / abs(pair(base['V_tilde_dot'], D_dot) + pair(base['H_tilde_dot'], L_dot))))

    # ---- Part B --------------------------------------------------------------------------------------------
    print('Part B. The one inner product: residual -> seed.  The residual is a covector of N; the seed is a vector of N.')
    y = [rn((P, W, N)) for N in Ns]
    omega = 0.5 + rng.random((d, P))
    weigh = lambda c: [(omega[i] ** 2).reshape((P, 1, 1)) * c[i] for i in range(d)]
    misfit = lambda z, data: 0.5 * sum(float(np.sum((omega[i] ** 2).reshape((P, 1, 1)) * (z[i] - data[i]) ** 2))
                                       for i in range(d))
    r = [base['z'][i] - y[i] for i in range(d)]
    for label, SN in (('arbitrary invertible', S['N']), ('orthogonal', [np.linalg.qr(rn((N, N)))[0] for N in Ns])):
        S2 = dict(S, N=SN)
        D3, U3, L3, R3, V3, H3, D_dot3, L_dot3 = change(S2)
        v3 = program((D3, U3, L3, R3), V3, H3, D_dot3, L_dot3,
                     [move(w[i], (('vec', 'N'),), S2, i) for i in range(d)],
                     [move(w_tilde[i], (('vec', 'N'),), S2, i) for i in range(d)], d, P, W)
        y3 = [move(y[i], (('cov', 'N'),), S2, i) for i in range(d)]             # data is of the type of z
        r3 = [v3['z'][i] - y3[i] for i in range(d)]
        seed_copied = weigh(r3)                                                  # omega^2 r, copied slot for slot
        seed_typed = [move(c, (('vec', 'N'),), S2, i) for i, c in enumerate(weigh(r))]   # the old seed, as a vector
        print('        N changed by an %-21s matrix:  omega^2 r copied into the seed is off by %.0e;  the misfit changes by %.0e'
              % (label, err(seed_copied, seed_typed), abs(misfit(v3['z'], y3) - misfit(base['z'], y)) / misfit(base['z'], y)))

    # ---- Part C --------------------------------------------------------------------------------------------
    c = 1.7
    print('Part C. The order factor, under s -> c s with c = %.1f (the direction p becomes c p).' % c)
    powers = lambda sign, x: (c ** (sign * np.arange(P))).reshape((P,) + (1,) * (x.ndim - 1)) * x
    phi = [rn((P, W, N)) for N in Ns]                                            # a functional on z, in forward order
    seed = [f[::-1] for f in phi]                                                # its representing jet
    old = program((D, U, L, R), V, H, D_dot, L_dot, w, seed, d, P, W)
    phi2 = [powers(-1, f) for f in phi]                                          # the same functional after s -> c s
    new = program((D, U, L, R), V, H, D_dot, L_dot, [J.input_jet(P, ww[i], c * pp[i]) for i in range(d)],
                  [f[::-1] for f in phi2], d, P, W)
    jets = [n for n, _, _, kind in CATALOG if kind == 'jet']
    duals = [n for n, _, _, kind in CATALOG if kind == 'dual jet']
    reals = [n for n, _, _, kind in CATALOG if kind == 'real']
    print('        jets: slot t is multiplied by c^t                                   worst %.0e'
          % max(err(new[n], [powers(+1, x) for x in old[n]]) for n in jets))
    print('        a functional in forward order: slot t is multiplied by c^-t (given), so <phi, z> is unchanged: %.0e'
          % (abs(pair(phi2, new['z']) - pair(phi, old['z'])) / abs(pair(phi, old['z']))))
    print('        dual jets, as stored (representing jets): slot t is multiplied by c^-n c^t worst %.0e'
          % max(err(new[n], [c ** (-order) * powers(+1, x) for x in old[n]]) for n in duals))
    print('        dual jets, read in forward order: slot t is multiplied by c^-t            worst %.0e'
          % max(err([x[::-1] for x in new[n]], [powers(-1, x[::-1]) for x in old[n]]) for n in duals))
    print('        the blocks are real: unchanged                                           worst %.0e'
          % max(err(new[n], old[n]) for n in reals))


if __name__ == '__main__':
    main()
