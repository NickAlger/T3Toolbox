"""T3Polynomial's path in the explicit form: probe derivatives (jets) with residual weights.

Every sampling variable is its own array with a leading order axis (order + 1 Taylor coefficients).  Every
function takes exactly the arrays it reads.  The per-site functions are the ones of the order-0 example, named by
letter and grade (one, nu_tilde_dot_step, renamed on 2026-11-16); the ONLY thing jets change is the seven term
functions, where a contraction of jets becomes a truncated product over the order axis.  The variables in main()
follow the notation decided on 2026-11-15, 23:23 (SAMPLE_VARIABLES.md); a step function serves both bases (see the
comment above the dot steps).  There is no second set of functions for the tilde grade: residual jets are stored
reversed (the Taylor convention of the first phase), so the adjoint is the same product.

The library's conventions are kept at the boundary: the inputs (w, p), the data y, the weight omega[mode, order],
and the outputs z, r, Jp are derivative-normalized, exactly as `ProbeDerivativesKind` takes and returns them.

Checked against the library's own kind (`backend.fitting.ProbeDerivativesKind`):  z, f, r_block_sumsq, g, Jp,
Jp_sumsq, JtJp, Kp.  Ragged cores, numpy, one sample axis.  Not covered: the uniform layout, the K and C stacks,
jit, order scans (the truncated products here are plain loops), apply and entries with jets.

Usage:  python3 -I jet_steps_check.py /path/to/T3Toolbox
"""
import itertools
import math
import sys

import numpy as np


# ---- jets ---------------------------------------------------------------------------------------------
def truncated_product(term, *jets):
    """sum over orders r1 + r2 + ... = t of term(jet1[r1], jet2[r2], ...), for t = 0 .. order."""
    P = jets[0].shape[0]
    out = None
    for orders in itertools.product(range(P), repeat=len(jets)):
        t = sum(orders)
        if t >= P:
            continue
        val = term(*(j[r] for j, r in zip(jets, orders)))
        if out is None:
            out = np.zeros((P,) + val.shape, val.dtype)
        out[t] += val
    return out


def input_jet(P, w, p):                    # w + s p as Taylor coefficients: (w, p, 0, ...)
    out = np.zeros((P,) + w.shape)
    out[0] = w
    if P > 1:
        out[1] = p
    return out


def seed_jet(c):                           # derivative-normalized (P, W, ...) -> reversed Taylor: [P-1-t] = t! c[t]
    P = c.shape[0]
    fact = np.array([math.factorial(t) for t in range(P)]).reshape((P,) + (1,) * (c.ndim - 1))
    return (fact * c)[::-1]


def derivatives(x):                        # Taylor coefficients -> derivatives
    P = x.shape[0]
    return x * np.array([math.factorial(t) for t in range(P)]).reshape((P,) + (1,) * (x.ndim - 1))


def ones_start(P, W, r):                   # a plain sweep starts from ones at order 0, zero above
    out = np.zeros((P, W, r))
    out[0] = 1.0
    return out


def zeros_start(P, W, r):                  # every other sweep of the probe case starts from zeros
    return np.zeros((P, W, r))


# ---- the seven terms: the only contractions -------------------------------------------------------------
def xi_term(B, w):          return np.einsum('io,tWo->tWi', B, w)
def z_term(eta, B):         return np.einsum('tWi,io->tWo', eta, B)
def mu_term(mu, G, xi):     return truncated_product(lambda a, x: np.einsum('Wa,aib,Wi->Wb', a, G, x), mu, xi)
def nu_term(G, xi, nu):     return truncated_product(lambda x, b: np.einsum('aib,Wi,Wb->Wa', G, x, b), xi, nu)
def eta_term(mu, G, nu):    return truncated_product(lambda a, b: np.einsum('Wa,aib,Wb->Wi', a, G, b), mu, nu)
def H_term(mu, xi, nu):     return truncated_product(lambda a, x, b: np.einsum('Wa,Wi,Wb->aib', a, x, b), mu, xi, nu)
def V_term(eta, w):         return truncated_product(lambda e, v: np.einsum('Wi,Wo->io', e, v), eta, w)


# ---- plain ------------------------------------------------------------------------------------------------
def xi_step(D, w):                  return xi_term(D, w)
def mu_step(mu, L, xi):             return mu_term(mu, L, xi)
def nu_step(nu, R, xi):             return nu_term(R, xi, nu)
def eta_step(mu, U, nu):            return eta_term(mu, U, nu)
def z_step(mu, L, nu, D):           return z_term(eta_term(mu, L, nu), D)         # the value, through (D, L)


# ---- dot: the variable varied in the cores along p = (V, H) ---------------------------------------------
# A step function is named by letter and grade and serves both bases; the basis of what it returns follows from
# the cores it is given (nu_step given L cores returns nu_L; mu_dot_step given (R, U, H) and the frame's xi_U_dot
# returns mu_R_dot, given (L, L, L_dot) and the raw xi_dot it returns mu_dot).  Decided by the maintainer on
# 2026-11-16 at 00:23.  So the variable parameters of every step function are basis-free (letter and grade); the
# core parameters (R, U, H, G, G_dot, ...) are placeholders named after one set of cores they are called with.
def xi_dot_step(V, w):
    return xi_term(V, w)

def mu_dot_step(mu_dot, mu, xi, xi_dot, R, U, H):
    return mu_term(mu_dot, R, xi) + mu_term(mu, H, xi) + mu_term(mu, U, xi_dot)

def nu_dot_step(nu_dot, nu, xi, xi_dot, L, U, H):
    return nu_term(L, xi, nu_dot) + nu_term(H, xi, nu) + nu_term(U, xi_dot, nu)

def eta_dot_step(mu_dot, mu, nu_dot, nu, L, R, H):
    return eta_term(mu_dot, R, nu) + eta_term(mu, H, nu) + eta_term(mu, L, nu_dot)

def Jp_step(eta_dot, eta, D, V):
    return z_term(eta_dot, D) + z_term(eta, V)


# ---- tilde: the variable varied in the samples along the seed w_tilde -----------------------------------
def xi_tilde_step(D, w_tilde):
    return xi_term(D, w_tilde)

def mu_tilde_step(mu_tilde, mu, xi, xi_tilde, L):
    return mu_term(mu_tilde, L, xi) + mu_term(mu, L, xi_tilde)

def nu_tilde_step(nu_tilde, nu, xi, xi_tilde, R):
    return nu_term(R, xi, nu_tilde) + nu_term(R, xi_tilde, nu)

def eta_tilde_step(mu_tilde, mu, nu_tilde, nu, U):
    return eta_term(mu_tilde, U, nu) + eta_term(mu, U, nu_tilde)

def V_tilde_step(eta_tilde, eta, w, w_tilde):
    return V_term(eta_tilde, w) + V_term(eta, w_tilde)

def H_tilde_step(mu_tilde, mu, xi_tilde, xi, nu_tilde, nu):
    return H_term(mu_tilde, xi, nu) + H_term(mu, xi_tilde, nu) + H_term(mu, xi, nu_tilde)


# ---- tilde-dot: both, at the base point with a raw core direction ----------------------------------------
# (D, L) and (D_dot, L_dot) at the base point of a manifold frame; (B, G) and (B_dot, G_dot) on the corewise frame.
def xi_tilde_dot_step(B_dot, w_tilde):
    return xi_term(B_dot, w_tilde)

def mu_tilde_dot_step(mu_tilde_dot, mu_dot, mu_tilde, mu, xi_tilde_dot, xi_dot, xi_tilde, xi, G, G_dot):
    return (mu_term(mu_tilde_dot, G, xi) + mu_term(mu_dot, G, xi_tilde) + mu_term(mu_tilde, G, xi_dot)
            + mu_term(mu, G, xi_tilde_dot) + mu_term(mu_tilde, G_dot, xi) + mu_term(mu, G_dot, xi_tilde))

def nu_tilde_dot_step(nu_tilde_dot, nu_dot, nu_tilde, nu, xi_tilde_dot, xi_dot, xi_tilde, xi,
                      G, G_dot):
    return (nu_term(G, xi, nu_tilde_dot) + nu_term(G, xi_tilde, nu_dot) + nu_term(G, xi_dot, nu_tilde)
            + nu_term(G, xi_tilde_dot, nu) + nu_term(G_dot, xi, nu_tilde) + nu_term(G_dot, xi_tilde, nu))

def eta_tilde_dot_step(mu_tilde_dot, mu_dot, mu_tilde, mu, nu_tilde_dot, nu_dot, nu_tilde, nu,
                       G, G_dot):
    return (eta_term(mu_tilde_dot, G, nu) + eta_term(mu_dot, G, nu_tilde) + eta_term(mu_tilde, G, nu_dot)
            + eta_term(mu, G, nu_tilde_dot) + eta_term(mu_tilde, G_dot, nu) + eta_term(mu, G_dot, nu_tilde))

def V_tilde_dot_step(eta_tilde_dot, eta_dot, w, w_tilde):
    return V_term(eta_tilde_dot, w) + V_term(eta_dot, w_tilde)

def H_tilde_dot_step(mu_tilde_dot, mu_dot, mu_tilde, mu, xi_tilde_dot, xi_dot, xi_tilde, xi,
                     nu_tilde_dot, nu_dot, nu_tilde, nu):
    return (H_term(mu_tilde_dot, xi, nu) + H_term(mu, xi_tilde_dot, nu) + H_term(mu, xi, nu_tilde_dot)
            + H_term(mu_dot, xi_tilde, nu) + H_term(mu_dot, xi, nu_tilde)
            + H_term(mu_tilde, xi_dot, nu) + H_term(mu, xi_dot, nu_tilde)
            + H_term(mu_tilde, xi, nu_dot) + H_term(mu, xi_tilde, nu_dot))


# ---- drivers: mu[i] and nu[i] sit on bond i, i = 0 .. d -------------------------------------------------
def left_to_right(step, start, d, *per_site):
    out = [start]
    for i in range(d):
        out.append(step(out[i], *(a[i] for a in per_site)))
    return out

def right_to_left(step, start, d, *per_site):
    out = [None] * d + [start]
    for i in reversed(range(d)):
        out[i] = step(out[i + 1], *(a[i] for a in per_site))
    return out

def each_site(step, d, *per_site):
    return [step(*(a[i] for a in per_site)) for i in range(d)]

def top(blocks):                           # a gradient block is the top Taylor coefficient
    return [b[-1] for b in blocks]


# ---- the sampling variables of the probe-derivatives kind, written out ----------------------------------
def tilde_family(seed, w, xi, mu, nu, eta, frame, d, P, W):
    """(V_tilde, H_tilde) = Jt applied to a seed, on the manifold frame."""
    D, U, L, R = frame
    xi_tilde = each_site(xi_tilde_step, d, D, seed)
    mu_tilde = left_to_right(mu_tilde_step, zeros_start(P, W, L[0].shape[0]), d, mu, xi, xi_tilde, L)
    nu_tilde = right_to_left(nu_tilde_step, zeros_start(P, W, R[-1].shape[2]), d, nu[1:], xi, xi_tilde, R)
    eta_tilde = each_site(eta_tilde_step, d, mu_tilde, mu, nu_tilde[1:], nu[1:], U)
    V_tilde = each_site(V_tilde_step, d, eta_tilde, eta, w, seed)
    H_tilde = each_site(H_tilde_step, d, mu_tilde, mu, xi_tilde, xi, nu_tilde[1:], nu[1:])
    return top(V_tilde), top(H_tilde), (xi_tilde, mu_tilde)


def main(repo):
    sys.path.append(repo)
    import t3toolbox.backend.fitting as bfit

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
    D_dot, L_dot = [rn(b.shape) for b in D], [rn(g.shape) for g in L]     # a raw core direction at (D, L)
    frame = (D, U, L, R)
    ww, pp = [rn((W, N)) for N in Ns], [rn((W, N)) for N in Ns]
    flat = lambda blocks: np.concatenate([np.ravel(np.asarray(b)) for b in blocks])
    err = lambda mine, theirs: np.linalg.norm(flat(mine) - flat(theirs)) / np.linalg.norm(flat(theirs))

    for order in (0, 2, 4):
        P = order + 1
        omega = 0.5 + rng.random((d, P))                                   # weight[mode, order]
        y = [rn((P, W, N)) for N in Ns]                                    # data, derivative-normalized
        kind = bfit.ProbeDerivativesKind(order=order, weight=omega, chunk_size=None)
        sample, n_w = (ww, pp), 1
        sweep = kind.precompute(frame, sample)
        weigh = lambda c: [(omega[i] ** 2).reshape((P, 1, 1)) * c[i] for i in range(d)]   # omega^2, per mode and order

        # plain variables on the manifold frame, and nu_L (the right of a bond in the L basis: through L)
        w = [input_jet(P, ww[i], pp[i]) for i in range(d)]
        xi = each_site(xi_step, d, D, w)
        mu = left_to_right(mu_step, ones_start(P, W, L[0].shape[0]), d, L, xi)
        nu = right_to_left(nu_step, ones_start(P, W, R[-1].shape[2]), d, R, xi)
        eta = each_site(eta_step, d, mu, U, nu[1:])
        nu_L = right_to_left(nu_step, ones_start(P, W, L[-1].shape[2]), d, L, xi)

        # the value, the residual, the misfit and its per-mode, per-order sums
        z = [derivatives(a) for a in each_site(z_step, d, mu, L, nu_L[1:], D)]
        r = [z[i] - y[i] for i in range(d)]
        r_block_sumsq = np.array([[np.sum(r[i][t] ** 2) for t in range(P)] for i in range(d)])
        f = 0.5 * np.sum(omega ** 2 * r_block_sumsq)

        # g = Jt (omega^2 r)
        w_tilde = [seed_jet(c) for c in weigh(r)]
        V_tilde, H_tilde, (xi_tilde, mu_tilde) = tilde_family(w_tilde, w, xi, mu, nu, eta, frame, d, P, W)

        # Jp, its weighted sum of squares, and JtJp = Jt (omega^2 Jp); the dots of the frame's form, along (V, H)
        xi_U_dot = each_site(xi_dot_step, d, V, w)
        mu_R_dot = left_to_right(mu_dot_step, zeros_start(P, W, R[0].shape[0]), d, mu, xi, xi_U_dot, R, U, H)
        nu_L_dot = right_to_left(nu_dot_step, zeros_start(P, W, L[-1].shape[2]), d, nu[1:], xi, xi_U_dot, L, U, H)
        eta_D_dot = each_site(eta_dot_step, d, mu_R_dot, mu, nu_L_dot[1:], nu[1:], L, R, H)
        Jp = [derivatives(a) for a in each_site(Jp_step, d, eta_D_dot, eta, D, V)]
        Jp_sumsq = sum(np.sum((omega[i] ** 2).reshape((P, 1, 1)) * Jp[i] ** 2) for i in range(d))
        JtJp_V, JtJp_H, _ = tilde_family([seed_jet(c) for c in weigh(Jp)], w, xi, mu, nu, eta, frame, d, P, W)

        # Kp: the two-hole term at the base point (D, L), raw core direction (D_dot, L_dot), seed omega^2 r.
        # Shared with the frame: xi, mu, xi_tilde, mu_tilde.  Through L instead of R: nu_L, nu_L_tilde.
        nu_L_tilde = right_to_left(nu_tilde_step, zeros_start(P, W, L[-1].shape[2]), d, nu_L[1:], xi, xi_tilde, L)
        # The dots of the raw form, along (D_dot, L_dot).  raw_nu_L_dot and raw_eta_D_dot are a local spelling of
        # this script, not notation: the raw form's nu_L_dot and eta_D_dot, which differ here from the frame's above
        # because (D_dot, L_dot) is not the lift of (V, H).
        xi_dot = each_site(xi_dot_step, d, D_dot, w)
        mu_dot = left_to_right(mu_dot_step, zeros_start(P, W, L[0].shape[0]), d, mu, xi, xi_dot, L, L, L_dot)
        raw_nu_L_dot = right_to_left(nu_dot_step, zeros_start(P, W, L[-1].shape[2]), d, nu_L[1:], xi, xi_dot,
                                     L, L, L_dot)
        raw_eta_D_dot = each_site(eta_dot_step, d, mu_dot, mu, raw_nu_L_dot[1:], nu_L[1:], L, L, L_dot)
        xi_tilde_dot = each_site(xi_tilde_dot_step, d, D_dot, w_tilde)
        mu_tilde_dot = left_to_right(mu_tilde_dot_step, zeros_start(P, W, L[0].shape[0]), d, mu_dot, mu_tilde, mu,
                                     xi_tilde_dot, xi_dot, xi_tilde, xi, L, L_dot)
        nu_L_tilde_dot = right_to_left(nu_tilde_dot_step, zeros_start(P, W, L[-1].shape[2]), d, raw_nu_L_dot[1:],
                                       nu_L_tilde[1:], nu_L[1:], xi_tilde_dot, xi_dot, xi_tilde, xi, L, L_dot)
        eta_D_tilde_dot = each_site(eta_tilde_dot_step, d, mu_tilde_dot, mu_dot, mu_tilde, mu, nu_L_tilde_dot[1:],
                                    raw_nu_L_dot[1:], nu_L_tilde[1:], nu_L[1:], L, L_dot)
        Kp_V = top(each_site(V_tilde_dot_step, d, eta_D_tilde_dot, raw_eta_D_dot, w, w_tilde))
        Kp_H = top(each_site(H_tilde_dot_step, d, mu_tilde_dot, mu_dot, mu_tilde, mu, xi_tilde_dot, xi_dot,
                             xi_tilde, xi, nu_L_tilde_dot[1:], raw_nu_L_dot[1:], nu_L_tilde[1:], nu_L[1:]))

        # the library's own kind
        z_ref = kind.point_forward((D, L), sample)
        r_ref = [z_ref[i] - y[i] for i in range(d)]
        g_ref = kind.transpose(r_ref, sample, frame, sweep)
        Jp_ref = kind.forward((V, H), sample, frame, sweep)
        JtJp_ref = kind.transpose(Jp_ref, sample, frame, sweep)
        Kp_ref = kind.corewise_hessian((D_dot, L_dot), r_ref, sample, frame, sweep, (D, L))
        print('order %d   z %.1e   f %.1e   r_block_sumsq %.1e   g %.1e   Jp %.1e   Jp_sumsq %.1e   JtJp %.1e   Kp %.1e'
              % (order, err(z, z_ref),
                 abs(f - 0.5 * float(kind.sumsq(r_ref, n_w))) / (0.5 * float(kind.sumsq(r_ref, n_w))),
                 err([r_block_sumsq], [np.asarray(kind.block_sumsq(r_ref, n_w))]),
                 err(V_tilde + H_tilde, list(g_ref[0]) + list(g_ref[1])),
                 err(Jp, Jp_ref),
                 abs(Jp_sumsq - float(kind.sumsq(Jp_ref, n_w))) / float(kind.sumsq(Jp_ref, n_w)),
                 err(JtJp_V + JtJp_H, list(JtJp_ref[0]) + list(JtJp_ref[1])),
                 err(Kp_V + Kp_H, list(Kp_ref[0]) + list(Kp_ref[1]))))

    import inspect
    import re
    steps = [f for name, f in sorted(globals().items()) if callable(f) and name.endswith('_step')]
    body = lambda f: inspect.getsource(f).split(':', 1)[1]
    n_terms = sum(len(re.findall(r'\b(?:xi|mu|nu|eta|z|H|V)_term\(', body(f))) for f in steps)
    print('%d step functions, %d terms, 7 term functions; the same for every order' % (len(steps), n_terms))


if __name__ == '__main__':
    main(sys.argv[1])
