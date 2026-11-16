"""The sampling variables in the notation decided on 2026-11-15 (23:23): four letters and a basis.

This is the program of SAMPLE_VARIABLES.md, written out with the names of that page, and checked two ways.

  A. THE RECURRENCES of the page (section 5), against the library's own kind
     (`backend.fitting.ProbeDerivativesKind`):  z, g, Jp, JtJp, Kp.
  B. THE SUBNETWORKS.  The page defines every variable as the contraction of the subnetwork on one side of an
     edge, with a direction put in at one site (dot), a seed put in at one input (tilde), or both, summed over
     where they are put.  Each sum is formed here term by term, with no recurrence, and compared with the
     recurrence of part A.  This is what the drawings of the "Variable diagrams" tab show.
  C. THE DIRECTION AT A CUT EDGE (section 8 of the page).
  D. THE TILDE VARIABLES ARE THE LIBRARY'S ADJOINT VARIABLES, array by array, at order 0: `xi_tilde`, `mu_tilde`,
     `nu_tilde`, `eta_tilde` against `compute_deta_tilde`, `compute_tau_tilde`, `compute_sigma_tilde`,
     `compute_dxi_tilde` of `backend/probing.py`.  There is one array per variable; it combines the residuals of
     every mode on its side.

A name is  <letter>[_<basis>][_tilde][_dot].  The letter is the side that is contracted: mu the left of a bond,
nu the right of a bond, xi the leaf below a Tucker edge, eta the trunk above it.  The basis is written only where
it is not the usual one (L for mu, R for nu, D for xi, U for eta).

The seven term functions and the drivers are imported from jet_steps_check.py; they have no names of variables in
them.  Covered: the probe kind, ragged cores, numpy, one sample axis, orders 0 and 2, stores that are random
arrays (so nothing here rests on orthogonality).  Not covered: apply and entries, the uniform layout, the K and C
stacks, a frame the library built (that is subtree_gauge_check.py, which shows that the two forms of `nu_L_dot`
and of `eta_D_dot` are one array along the lifted direction).

Usage:  python3 -I sample_variables_check.py /path/to/T3Toolbox
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jet_steps_check as J
from jet_steps_check import xi_term, z_term, mu_term, nu_term, eta_term, H_term, V_term


# ---- the subnetworks, with no recurrence ---------------------------------------------------------------
def left_side(tt, leaves, i, P, W):        # sites 0 .. i-1, contracted; the open leg is bond i
    out = J.ones_start(P, W, tt[0].shape[0])
    for j in range(i):
        out = mu_term(out, tt[j], leaves[j])
    return out

def right_side(tt, leaves, i, P, W):       # sites i .. d-1, contracted; the open leg is bond i
    out = J.ones_start(P, W, tt[-1].shape[2])
    for j in reversed(range(i, len(tt))):
        out = nu_term(tt[j], leaves[j], out)
    return out

def trunk(tt, leaves, i, P, W):            # everything but the leaf of site i; the open leg is Tucker edge i
    return eta_term(left_side(tt, leaves, i, P, W), tt[i], right_side(tt, leaves, i + 1, P, W))

def put(items, j, item):                   # a copy of a list with entry j replaced
    return items[:j] + [item] + items[j + 1:]


def main(repo):
    sys.path.append(repo)
    import t3toolbox.backend.fitting as bfit

    rng = np.random.default_rng(0)
    rn = rng.standard_normal
    Ns, nD, nU = (6, 5, 7, 4), (3, 4, 2, 3), (2, 3, 3, 2)
    rL, rR = (1, 3, 2, 4, 1), (1, 2, 4, 3, 1)
    d, W = len(Ns), 5
    D = [rn((nD[i], Ns[i])) for i in range(d)]                 # D[i]: (nD_i, N_i)
    U = [rn((rL[i], nU[i], rR[i + 1])) for i in range(d)]      # U[i]: (rL_i, nU_i, rR_(i+1))
    L = [rn((rL[i], nD[i], rL[i + 1])) for i in range(d)]      # L[i]: (rL_i, nD_i, rL_(i+1))
    R = [rn((rR[i], nD[i], rR[i + 1])) for i in range(d)]      # R[i]: (rR_i, nD_i, rR_(i+1))
    V = [rn((nU[i], Ns[i])) for i in range(d)]                 # the direction at the frame: (V, H)
    H = [rn((rL[i], nD[i], rR[i + 1])) for i in range(d)]
    D_dot, L_dot = [rn(b.shape) for b in D], [rn(g.shape) for g in L]     # a raw core direction at (D, L)
    frame = (D, U, L, R)
    ww, pp = [rn((W, N)) for N in Ns], [rn((W, N)) for N in Ns]
    flat = lambda blocks: np.concatenate([np.ravel(np.asarray(b)) for b in blocks])
    err = lambda mine, theirs: np.linalg.norm(flat(mine) - flat(theirs)) / np.linalg.norm(flat(theirs))
    top = J.top
    ok = True

    for order in (0, 2):
        P = order + 1
        one = lambda r: J.ones_start(P, W, r)
        zero = lambda r: J.zeros_start(P, W, r)
        omega = 0.5 + rng.random((d, P))
        y = [rn((P, W, N)) for N in Ns]
        kind = bfit.ProbeDerivativesKind(order=order, weight=omega, chunk_size=None)
        sample = (ww, pp)
        sweep = kind.precompute(frame, sample)
        weigh = lambda c: [(omega[i] ** 2).reshape((P, 1, 1)) * c[i] for i in range(d)]
        w = [J.input_jet(P, ww[i], pp[i]) for i in range(d)]

        # ================= A. the recurrences of the page ===============================================
        # ---- plain
        xi = [xi_term(D[i], w[i]) for i in range(d)]
        mu, nu, nu_L = [one(1)], [None] * d + [one(1)], [None] * d + [one(1)]
        for i in range(d):
            mu.append(mu_term(mu[i], L[i], xi[i]))
        for i in reversed(range(d)):
            nu[i] = nu_term(R[i], xi[i], nu[i + 1])
            nu_L[i] = nu_term(L[i], xi[i], nu_L[i + 1])
        eta = [eta_term(mu[i], U[i], nu[i + 1]) for i in range(d)]
        eta_D = [eta_term(mu[i], L[i], nu_L[i + 1]) for i in range(d)]
        z = [z_term(eta_D[i], D[i]) for i in range(d)]

        # ---- dot, the frame's form: the direction is (V, H)
        xi_U_dot = [xi_term(V[i], w[i]) for i in range(d)]
        mu_R_dot, nu_L_dot = [zero(1)], [None] * d + [zero(1)]
        for i in range(d):
            mu_R_dot.append(mu_term(mu_R_dot[i], R[i], xi[i]) + mu_term(mu[i], H[i], xi[i])
                            + mu_term(mu[i], U[i], xi_U_dot[i]))
        for i in reversed(range(d)):
            nu_L_dot[i] = (nu_term(L[i], xi[i], nu_L_dot[i + 1]) + nu_term(H[i], xi[i], nu[i + 1])
                           + nu_term(U[i], xi_U_dot[i], nu[i + 1]))
        eta_D_dot = [eta_term(mu_R_dot[i], R[i], nu[i + 1]) + eta_term(mu[i], H[i], nu[i + 1])
                     + eta_term(mu[i], L[i], nu_L_dot[i + 1]) for i in range(d)]
        Jp = [z_term(eta_D_dot[i], D[i]) + z_term(eta[i], V[i]) for i in range(d)]

        # ---- tilde, for a seed w_tilde
        def tilde_family(w_tilde):
            xi_tilde = [xi_term(D[i], w_tilde[i]) for i in range(d)]
            mu_tilde, nu_tilde, nu_L_tilde = [zero(1)], [None] * d + [zero(1)], [None] * d + [zero(1)]
            for i in range(d):
                mu_tilde.append(mu_term(mu_tilde[i], L[i], xi[i]) + mu_term(mu[i], L[i], xi_tilde[i]))
            for i in reversed(range(d)):
                nu_tilde[i] = nu_term(R[i], xi[i], nu_tilde[i + 1]) + nu_term(R[i], xi_tilde[i], nu[i + 1])
                nu_L_tilde[i] = nu_term(L[i], xi[i], nu_L_tilde[i + 1]) + nu_term(L[i], xi_tilde[i], nu_L[i + 1])
            eta_tilde = [eta_term(mu_tilde[i], U[i], nu[i + 1]) + eta_term(mu[i], U[i], nu_tilde[i + 1])
                         for i in range(d)]
            eta_D_tilde = [eta_term(mu_tilde[i], L[i], nu_L[i + 1]) + eta_term(mu[i], L[i], nu_L_tilde[i + 1])
                           for i in range(d)]
            V_tilde = top([V_term(eta_tilde[i], w[i]) + V_term(eta[i], w_tilde[i]) for i in range(d)])
            H_tilde = top([H_term(mu_tilde[i], xi[i], nu[i + 1]) + H_term(mu[i], xi_tilde[i], nu[i + 1])
                           + H_term(mu[i], xi[i], nu_tilde[i + 1]) for i in range(d)])
            return xi_tilde, mu_tilde, nu_tilde, nu_L_tilde, eta_tilde, eta_D_tilde, V_tilde, H_tilde

        r = [J.derivatives(z[i]) - y[i] for i in range(d)]
        w_tilde = [J.seed_jet(c) for c in weigh(r)]                            # the seed of g and of Kp
        xi_tilde, mu_tilde, nu_tilde, nu_L_tilde, eta_tilde, eta_D_tilde, V_tilde, H_tilde = tilde_family(w_tilde)
        JtJp = tilde_family([J.seed_jet(c) for c in weigh([J.derivatives(a) for a in Jp])])[6:]

        # ---- dot, the raw form: the direction is (D_dot, L_dot) at the base point (D, L)
        xi_dot = [xi_term(D_dot[i], w[i]) for i in range(d)]
        mu_dot, raw_nu_L_dot = [zero(1)], [None] * d + [zero(1)]
        for i in range(d):
            mu_dot.append(mu_term(mu_dot[i], L[i], xi[i]) + mu_term(mu[i], L_dot[i], xi[i])
                          + mu_term(mu[i], L[i], xi_dot[i]))
        for i in reversed(range(d)):
            raw_nu_L_dot[i] = (nu_term(L[i], xi[i], raw_nu_L_dot[i + 1]) + nu_term(L_dot[i], xi[i], nu_L[i + 1])
                               + nu_term(L[i], xi_dot[i], nu_L[i + 1]))
        raw_eta_D_dot = [eta_term(mu_dot[i], L[i], nu_L[i + 1]) + eta_term(mu[i], L_dot[i], nu_L[i + 1])
                         + eta_term(mu[i], L[i], raw_nu_L_dot[i + 1]) for i in range(d)]

        # ---- tilde-dot: both, at the base point, raw form
        xi_tilde_dot = [xi_term(D_dot[i], w_tilde[i]) for i in range(d)]
        mu_tilde_dot, nu_L_tilde_dot = [zero(1)], [None] * d + [zero(1)]
        for i in range(d):
            mu_tilde_dot.append(
                mu_term(mu_tilde_dot[i], L[i], xi[i]) + mu_term(mu_dot[i], L[i], xi_tilde[i])
                + mu_term(mu_tilde[i], L[i], xi_dot[i]) + mu_term(mu[i], L[i], xi_tilde_dot[i])
                + mu_term(mu_tilde[i], L_dot[i], xi[i]) + mu_term(mu[i], L_dot[i], xi_tilde[i]))
        for i in reversed(range(d)):
            nu_L_tilde_dot[i] = (
                nu_term(L[i], xi[i], nu_L_tilde_dot[i + 1]) + nu_term(L[i], xi_tilde[i], raw_nu_L_dot[i + 1])
                + nu_term(L[i], xi_dot[i], nu_L_tilde[i + 1]) + nu_term(L[i], xi_tilde_dot[i], nu_L[i + 1])
                + nu_term(L_dot[i], xi[i], nu_L_tilde[i + 1]) + nu_term(L_dot[i], xi_tilde[i], nu_L[i + 1]))
        eta_D_tilde_dot = [
            eta_term(mu_tilde_dot[i], L[i], nu_L[i + 1]) + eta_term(mu_dot[i], L[i], nu_L_tilde[i + 1])
            + eta_term(mu_tilde[i], L[i], raw_nu_L_dot[i + 1]) + eta_term(mu[i], L[i], nu_L_tilde_dot[i + 1])
            + eta_term(mu_tilde[i], L_dot[i], nu_L[i + 1]) + eta_term(mu[i], L_dot[i], nu_L_tilde[i + 1])
            for i in range(d)]
        V_tilde_dot = top([V_term(eta_D_tilde_dot[i], w[i]) + V_term(raw_eta_D_dot[i], w_tilde[i]) for i in range(d)])
        H_tilde_dot = top([
            H_term(mu_tilde_dot[i], xi[i], nu_L[i + 1]) + H_term(mu[i], xi_tilde_dot[i], nu_L[i + 1])
            + H_term(mu[i], xi[i], nu_L_tilde_dot[i + 1])
            + H_term(mu_dot[i], xi_tilde[i], nu_L[i + 1]) + H_term(mu_dot[i], xi[i], nu_L_tilde[i + 1])
            + H_term(mu_tilde[i], xi_dot[i], nu_L[i + 1]) + H_term(mu[i], xi_dot[i], nu_L_tilde[i + 1])
            + H_term(mu_tilde[i], xi[i], raw_nu_L_dot[i + 1]) + H_term(mu[i], xi_tilde[i], raw_nu_L_dot[i + 1])
            for i in range(d)])

        # ---- against the library
        z_ref = kind.point_forward((D, L), sample)
        r_ref = [z_ref[i] - y[i] for i in range(d)]
        g_ref = kind.transpose(r_ref, sample, frame, sweep)
        Jp_ref = kind.forward((V, H), sample, frame, sweep)
        JtJp_ref = kind.transpose(Jp_ref, sample, frame, sweep)
        Kp_ref = kind.corewise_hessian((D_dot, L_dot), r_ref, sample, frame, sweep, (D, L))
        errs = dict(z=err([J.derivatives(a) for a in z], z_ref),
                    g=err(V_tilde + H_tilde, list(g_ref[0]) + list(g_ref[1])),
                    Jp=err([J.derivatives(a) for a in Jp], Jp_ref),
                    JtJp=err(list(JtJp[0]) + list(JtJp[1]), list(JtJp_ref[0]) + list(JtJp_ref[1])),
                    Kp=err(V_tilde_dot + H_tilde_dot, list(Kp_ref[0]) + list(Kp_ref[1])))
        ok = ok and max(errs.values()) < 1e-12
        print('A. order %d, the recurrences against the library:  %s'
              % (order, '   '.join('%s %.0e' % kv for kv in errs.items())))

        # ================= B. the subnetworks, term by term =============================================
        sides = lambda f, tt, leaves, idx: [f(tt, leaves, i, P, W) for i in idx]
        bonds_l, bonds_r, sites = range(1, d + 1), range(0, d), range(d)       # where a sum is not empty
        add = lambda terms: [sum(t[k] for t in terms) for k in range(len(terms[0]))]
        checks = []

        def check(name, mine, recurrence):
            worst = max(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300) for a, b in zip(mine, recurrence))
            checks.append((name, worst))

        # the frame's form of a direction at site j: cores before it are L, cores after it are R
        frame_H = lambda j: L[:j] + [H[j]] + R[j + 1:]                          # H at site j, over D
        frame_U = lambda j: L[:j] + [U[j]] + R[j + 1:]                          # U at site j, over V
        raw_tt = lambda j: put(L, j, L_dot[j])                                   # the raw form: every other core is L

        def over_sites(f, idx, tt_of, leaves_of, which):                        # sum over the sites `which(i)`
            return [sum(f(tt_of(j), leaves_of(j), i, P, W) for j in which(i)) for i in idx]

        lt, ge, ne = (lambda i: range(i)), (lambda i: range(i, d)), (lambda i: [j for j in range(d) if j != i])

        check('mu_R_dot', add([over_sites(left_side, bonds_l, frame_H, lambda j: xi, lt),
                               over_sites(left_side, bonds_l, frame_U, lambda j: put(xi, j, xi_U_dot[j]), lt)]),
              [mu_R_dot[i] for i in bonds_l])
        check('nu_L_dot (frame form)', add([over_sites(right_side, bonds_r, frame_H, lambda j: xi, ge),
                                            over_sites(right_side, bonds_r, frame_U, lambda j: put(xi, j, xi_U_dot[j]), ge)]),
              [nu_L_dot[i] for i in bonds_r])
        check('eta_D_dot (frame form)', add([over_sites(trunk, sites, frame_H, lambda j: xi, lambda i: range(d)),
                                             over_sites(trunk, sites, frame_U, lambda j: put(xi, j, xi_U_dot[j]), ne)]),
              eta_D_dot)
        check('mu_dot', add([over_sites(left_side, bonds_l, raw_tt, lambda j: xi, lt),
                             over_sites(left_side, bonds_l, lambda j: L, lambda j: put(xi, j, xi_dot[j]), lt)]),
              [mu_dot[i] for i in bonds_l])
        check('nu_L_dot (raw form)', add([over_sites(right_side, bonds_r, raw_tt, lambda j: xi, ge),
                                          over_sites(right_side, bonds_r, lambda j: L, lambda j: put(xi, j, xi_dot[j]), ge)]),
              [raw_nu_L_dot[i] for i in bonds_r])
        check('eta_D_dot (raw form)', add([over_sites(trunk, sites, raw_tt, lambda j: xi, lambda i: range(d)),
                                           over_sites(trunk, sites, lambda j: L, lambda j: put(xi, j, xi_dot[j]), ne)]),
              raw_eta_D_dot)

        seeded = lambda k: put(xi, k, xi_tilde[k])                              # input k replaced by the seed
        check('mu_tilde', over_sites(left_side, bonds_l, lambda k: L, seeded, lt), [mu_tilde[i] for i in bonds_l])
        check('nu_tilde', over_sites(right_side, bonds_r, lambda k: R, seeded, ge), [nu_tilde[i] for i in bonds_r])
        check('nu_L_tilde', over_sites(right_side, bonds_r, lambda k: L, seeded, ge), [nu_L_tilde[i] for i in bonds_r])
        check('eta_tilde', [sum(trunk(put(L[:i] + R[i:], i, U[i]), seeded(k), i, P, W) for k in ne(i)) for i in sites],
              eta_tilde)
        check('eta_D_tilde', over_sites(trunk, sites, lambda k: L, seeded, ne), eta_D_tilde)

        # tilde-dot: a direction at site j (raw form) and the seed at input k, summed over both
        def both(f, i, js, ks):
            total = 0.0
            for j in js:
                for k in ks:
                    total = total + f(raw_tt(j), seeded(k), i, P, W)                         # L_dot at j, seed at k
                    if k == j:
                        total = total + f(L, put(xi, j, xi_tilde_dot[j]), i, P, W)          # D_dot at j under the seed
                    else:
                        total = total + f(L, put(seeded(k), j, xi_dot[j]), i, P, W)         # D_dot at j, seed at k
            return total

        check('mu_tilde_dot', [both(left_side, i, lt(i), lt(i)) for i in bonds_l], [mu_tilde_dot[i] for i in bonds_l])
        check('nu_L_tilde_dot', [both(right_side, i, ge(i), ge(i)) for i in bonds_r], [nu_L_tilde_dot[i] for i in bonds_r])

        def trunk_both(i):                                                      # the leaf of site i is not in the trunk
            total = 0.0
            for k in ne(i):
                for j in range(d):
                    total = total + trunk(raw_tt(j), seeded(k), i, P, W)
                    if j == i:
                        continue
                    if k == j:
                        total = total + trunk(L, put(xi, j, xi_tilde_dot[j]), i, P, W)
                    else:
                        total = total + trunk(L, put(seeded(k), j, xi_dot[j]), i, P, W)
            return total

        check('eta_D_tilde_dot', [trunk_both(i) for i in sites], eta_D_tilde_dot)

        worst = max(v for _, v in checks)
        ok = ok and worst < 1e-12
        print('B. order %d, %d variables as sums over where the direction and the seed are put:  %s (worst %.0e)'
              % (order, len(checks), 'equal' if worst < 1e-12 else 'DIFFERENT', worst))
        for name, v in checks:
            if v >= 1e-12:
                print('     %s  %.0e' % (name, v))

        # ================= C. the direction's network at a cut edge (section 8 of the page) =============
        # At a cut bond it is mu_R_dot paired with nu plus mu paired with nu_L_dot; at a cut Tucker edge it is
        # xi_U_dot paired with eta plus xi paired with eta_D_dot.  Every cut gives the same number per sample.
        pair = lambda a, b: J.truncated_product(lambda p, q: np.einsum('Wa,Wa->W', p, q), a, b)
        cuts = ([pair(mu_R_dot[i], nu[i]) + pair(mu[i], nu_L_dot[i]) for i in range(d + 1)]
                + [pair(xi_U_dot[i], eta[i]) + pair(xi[i], eta_D_dot[i]) for i in range(d)])
        worst = max(np.linalg.norm(c - cuts[0]) / np.linalg.norm(cuts[0]) for c in cuts)
        ok = ok and worst < 1e-12
        print('C. order %d, the direction at the %d cut edges, as one side dotted against the other plain:  %s (worst %.0e)'
              % (order, len(cuts), 'equal' if worst < 1e-12 else 'DIFFERENT', worst))

        # ================= D. the tilde variables are the library's adjoint variables, array by array ====
        # One array per variable.  Each combines the residuals of every mode on its side: the library's sweep
        # (`_tau_tilde_step`: carry . L . xi  +  mu . L . deta_tilde) is the recurrence of mu_tilde above.
        if order == 0:
            import t3toolbox.backend.probing as probing
            plain = lambda xs: [x[0] for x in xs]                              # drop the order axis (one slot)
            ztildes = plain(w_tilde)                                           # one residual-like array per mode
            mus, nus, xis = plain(mu[:d]), plain(nu[1:]), plain(xi)
            deta_tildes = probing.compute_deta_tilde(D, ztildes)
            tau_tildes = probing.compute_tau_tilde(deta_tildes, L, xis, mus)
            sigma_tildes = probing.compute_sigma_tilde(deta_tildes, R, xis, nus)
            dxi_tildes = probing.compute_dxi_tilde(sigma_tildes, tau_tildes, U, mus, nus)
            pairs = [('xi_tilde  = deta_tildes ', plain(xi_tilde), deta_tildes),
                     ('mu_tilde  = tau_tildes  ', plain(mu_tilde[:d]), tau_tildes),
                     ('nu_tilde  = sigma_tildes', plain(nu_tilde[1:]), sigma_tildes),
                     ('eta_tilde = dxi_tildes  ', plain(eta_tilde), dxi_tildes)]
            for name, mine, theirs in pairs:
                e = err(mine, theirs)
                ok = ok and e < 1e-12
                print('D. order 0, %s  %.0e   one array per site or bond, of shape %s'
                      % (name, e, ' '.join(str(tuple(a.shape)) for a in theirs)))

    print('all checks passed' if ok else 'SOME CHECK FAILED')


if __name__ == '__main__':
    main(sys.argv[1])
