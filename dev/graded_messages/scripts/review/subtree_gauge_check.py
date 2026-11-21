"""An edge variable is the full contraction of the subnetwork on one side of an edge, written in a basis of
that edge.  (The maintainer, 2026-11-15, 22:27: "consider the edge variables as being as corresponding to fully
contracted subnetworks on one side of the edge".)

A cut of the tree has four kinds of side: the left of a bond, the right of a bond, the leaf below a Tucker edge,
the trunk above it.  A manifold frame gives each cut edge two bases: the base point's own (the stores L and D)
and the frame's other one (the stores R and U).  The script checks two consequences, with the explicit step
functions of jet_steps_check.py and the library's own frame, gauge projection and lift (undamped):

  A. ONE DOT.  The dots of the frame's form, along (V, H), and the dots of the raw form, along the LIFTED raw core
     direction, are the same tangent vector.  Where the two are on the same edge they are the same array:
       the right of a bond in the L basis:      `nu_L_dot`, the frame's form  ==  the raw form
       the trunk above a Tucker edge, D basis:  `eta_D_dot`, the frame's form ==  the raw form
  B. TWINS.  Where they are on different edges, they are one subnetwork in two bases, and the lift's own
     factors are the change of basis:
       the left of a bond:   `mu_R_dot` (R basis)  ==  `mu_dot` (L basis) through the bond matrix T
       the leaf:             `xi_dot` (D basis)    ==  `xi_U_dot` (U basis) through the Tucker factor
     and the same factors, used the other way, join the plain twins:
       the right of a bond:  `nu`  (R basis)     ==  the bond factor applied to `nu_L` (L basis)
       the trunk:            `eta` (U basis)     ==  the Tucker factor applied to `eta_D` (D basis)
     Each is stated in the direction that holds on any frame the library builds.  This frame has slack at its
     last bond (4 in the L basis, 3 in the R basis), and there the first relation does NOT hold the other way
     round: the L basis has a dimension the right side annihilates, and `mu_dot` is free in it.

The names are those of SAMPLE_VARIABLES.md (2026-11-15, 23:23).  The raw form's `nu_L_dot` and `eta_D_dot` are
spelled `raw_nu_L_dot` and `raw_eta_D_dot` below: a local spelling of this script, not notation.

Covered: the probe kind, ragged cores, numpy, one sample axis, order 3, a frame the library built, no damping, no
shared factors.   Usage:  python3 -I subtree_gauge_check.py /path/to/T3Toolbox
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jet_steps_check as J


def main(repo):
    sys.path.insert(0, repo)
    import t3toolbox.tucker_tensor_train as t3
    import t3toolbox.backend.fv_conversions as fvc
    import t3toolbox.backend.fv_operations as fvo
    import t3toolbox.backend.tv_operations as tvo

    np.random.seed(0)                                                     # the library's constructors draw from it
    rng = np.random.default_rng(0)
    Ns, W, order = (6, 5, 7, 4), 5, 3
    d, P = len(Ns), order + 1
    x = t3.TuckerTensorTrain.randn(Ns, (3, 4, 2, 3), (1, 3, 2, 4, 1))
    frame, _ = fvc.t3_orthogonal_representations(x.data)
    D, U, L, R = (list(c) for c in frame)
    v = tvo.tv_orthogonal_gauge_projection(frame, fvo.fv_variations_randn(fvo.fv_variation_shapes(frame), ()))
    V, H = (list(c) for c in v)
    factors = fvo.fv_lift_factors(frame, 0.0)                             # undamped
    D_dot, L_dot = (list(c) for c in tvo.tv_corewise_lift(frame, v, factors))
    tucker_factor = [np.asarray(m) for m in factors.tucker_pinvs]         # [i]: (nDi, nUi), U basis -> D basis
    bond_factor = [np.asarray(m) for m in factors.bond_pinvs]             # [i]: (rR(i+1), rL(i+1)), R basis -> L basis
    ww, pp = [rng.standard_normal((W, N)) for N in Ns], [rng.standard_normal((W, N)) for N in Ns]
    w = [J.input_jet(P, ww[i], pp[i]) for i in range(d)]

    def same(a, b):
        worst = max(np.linalg.norm(p - q) / np.linalg.norm(q) for p, q in zip(a, b))
        return 'equal' if worst < 1e-12 else 'DIFFERENT (%.0e)' % worst

    # plain: the base point's four, and the frame's two twins
    xi = J.each_site(J.xi_step, d, D, w)
    mu = J.left_to_right(J.mu_step, J.ones_start(P, W, 1), d, L, xi)
    nu_L = J.right_to_left(J.nu_step, J.ones_start(P, W, 1), d, L, xi)
    eta_D = [J.eta_term(mu[i], L[i], nu_L[i + 1]) for i in range(d)]
    nu = J.right_to_left(J.nu_step, J.ones_start(P, W, 1), d, R, xi)
    eta = J.each_site(J.eta_step, d, mu, U, nu[1:])

    # the dots of the frame's form, along (V, H)
    xi_U_dot = J.each_site(J.xi_dot_step, d, V, w)
    mu_R_dot = J.left_to_right(J.mu_dot_step, J.zeros_start(P, W, 1), d, mu, xi, xi_U_dot, R, U, H)
    nu_L_dot = J.right_to_left(J.nu_dot_step, J.zeros_start(P, W, 1), d, nu[1:], xi, xi_U_dot, L, U, H)
    eta_D_dot = J.each_site(J.eta_dot_step, d, mu_R_dot, mu, nu_L_dot[1:], nu[1:], L, R, H)
    Jp_frame = J.each_site(J.Jp_step, d, eta_D_dot, eta, D, V)

    # the dots of the raw form, along the lifted raw core direction (D_dot, L_dot), through L
    # (raw_nu_L_dot, raw_eta_D_dot: a local spelling of this script for the raw form's nu_L_dot, eta_D_dot)
    xi_dot = J.each_site(J.xi_dot_step, d, D_dot, w)
    mu_dot = J.left_to_right(J.mu_dot_step, J.zeros_start(P, W, 1), d, mu, xi, xi_dot, L, L, L_dot)
    raw_nu_L_dot = J.right_to_left(J.nu_dot_step, J.zeros_start(P, W, 1), d, nu_L[1:], xi, xi_dot, L, L, L_dot)
    raw_eta_D_dot = J.each_site(J.eta_dot_step, d, mu_dot, mu, raw_nu_L_dot[1:], nu_L[1:], L, L, L_dot)
    Jp_base = [J.z_term(raw_eta_D_dot[i], D[i]) + J.z_term(eta_D[i], D_dot[i]) for i in range(d)]

    print('A. One dot: the frame along (V, H) against the base point along the lifted direction')
    print('   Jp                                                        %s' % same(Jp_frame, Jp_base))
    print("   right of a bond, L basis:   nu_L_dot,  frame and raw form  %s" % same(nu_L_dot[:d], raw_nu_L_dot[:d]))
    print("   trunk, D basis:             eta_D_dot, frame and raw form  %s" % same(eta_D_dot, raw_eta_D_dot))

    bonds = range(1, d)                                                   # the interior bonds 1 .. d-1
    print('B. Twins: one subnetwork in two bases, joined by the lift\'s own factors')
    T = [np.linalg.pinv(m) for m in bond_factor]                          # [i]: (rL(i+1), rR(i+1)), L basis -> R basis
    print("   left of a bond:   mu_R_dot (R)  and mu_dot (L)             %s"
          % same([mu_R_dot[j] for j in bonds], [np.einsum('tWc,cb->tWb', mu_dot[j], T[j - 1]) for j in bonds]))
    print("      the other way round, through the bond factor         %s"
          % ('equal' if same([mu_dot[j] for j in bonds], [np.einsum('tWb,bc->tWc', mu_R_dot[j], bond_factor[j - 1])
                                                         for j in bonds]) == 'equal' else 'different: the bond has slack'))
    print("   leaf:             xi_dot (D)    and xi_U_dot (U)           %s"
          % same(xi_dot, [np.einsum('ux,tWx->tWu', tucker_factor[i], xi_U_dot[i]) for i in range(d)]))
    print("   right of a bond:  nu (R)        and nu_L (L)               %s"
          % same([nu[j] for j in bonds], [np.einsum('bc,tWc->tWb', bond_factor[j - 1], nu_L[j]) for j in bonds]))
    print("   trunk:            eta (U)       and eta_D (D)              %s"
          % same(eta, [np.einsum('ux,tWu->tWx', tucker_factor[i], eta_D[i]) for i in range(d)]))
    print('   bond sizes, L basis %s, R basis %s' % ([m.shape[-1] for m in mu_dot[1:d]], [m.shape[-1] for m in mu_R_dot[1:d]]))


if __name__ == '__main__':
    main(sys.argv[1])
