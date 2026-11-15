"""Review finding F2, second attempt: no graded container at all.

Every component is its own array, named by letter and grade.  Every function takes exactly the arrays
it reads, as separate arguments, and returns one array.  There is no dict, no tuple of optional slots,
no operand object, and no inspection of what is present: a reader sees what each component reads from
its argument list, and what it computes from its terms.

Covered: the probe case, value / J p / J^T r on a manifold frame, and the exact two-hole term on a
corewise frame, each checked against T3Toolbox.  Not covered: jets (each product becomes a jet product,
the terms are the same), the K and C stacks, the uniform layout.

Usage:  python3 -I unbundled_examples.py /path/to/T3Toolbox
"""
import sys

import numpy as np


# ---- the seven site rules: the only contractions, each with a literal subscript string ---------------
def up(B, w):               return np.einsum('io,Wo->Wi', B, w)              # mode leg, upward
def right(mu, G, xi):       return np.einsum('Wa,aib,Wi->Wb', mu, G, xi)     # bond, rightward
def left(G, xi, nu):        return np.einsum('aib,Wi,Wb->Wa', G, xi, nu)     # bond, leftward
def down(mu, G, nu):        return np.einsum('Wa,aib,Wb->Wi', mu, G, nu)     # mode leg, downward
def out(eta, B):            return np.einsum('Wi,io->Wo', eta, B)            # ambient leg, outward
def grad_G(mu, xi, nu):     return np.einsum('Wa,Wi,Wb->aib', mu, xi, nu)    # TT core gradient, samples summed
def grad_B(eta, w):         return np.einsum('Wi,Wo->io', eta, w)            # Tucker core gradient, samples summed


# ---- plain grade, one site.  Frame cores D, U, L, R ---------------------------------------------------
def xi_site(D, w):                  return up(D, w)
def mu_step(mu, L, xi):             return right(mu, L, xi)
def nu_step(nu, R, xi):             return left(R, xi, nu)
def eta_site(mu, U, nu):            return down(mu, U, nu)
def z_site(eta, Vc):                return out(eta, Vc)


# ---- tangent grade (J p), one site.  Direction cores V, H ---------------------------------------------
def xi_t_site(V, w):
    return up(V, w)

def mu_t_step(mu_t, mu, xi, xi_t, R, U, H):
    return right(mu_t, R, xi) + right(mu, H, xi) + right(mu, U, xi_t)

def nu_t_step(nu_t, nu, xi, xi_t, L, U, H):
    return left(L, xi, nu_t) + left(H, xi, nu) + left(U, xi_t, nu)

def eta_t_site(mu_t, mu, nu_t, nu, L, R, H):
    return down(mu_t, R, nu) + down(mu, H, nu) + down(mu, L, nu_t)

def Jp_site(eta_t, eta, D, V):
    return out(eta_t, D) + out(eta, V)


# ---- residual grade (J^T r), one site.  Seed s -------------------------------------------------------
def xi_z_site(D, s):
    return up(D, s)

def mu_z_step(mu_z, mu, xi, xi_z, L):
    return right(mu_z, L, xi) + right(mu, L, xi_z)

def nu_z_step(nu_z, nu, xi, xi_z, R):
    return left(R, xi, nu_z) + left(R, xi_z, nu)

def eta_z_site(mu_z, mu, nu_z, nu, U):
    return down(mu_z, U, nu) + down(mu, U, nu_z)

def g_B_site(eta_z, eta, w, s):
    return grad_B(eta_z, w) + grad_B(eta, s)

def g_G_site(mu_z, mu, xi_z, xi, nu_z, nu):
    return grad_G(mu_z, xi, nu) + grad_G(mu, xi_z, nu) + grad_G(mu, xi, nu_z)


# ---- both grades (the exact two-hole term), one site.  Corewise: cores B, G and directions Bd, Gd ----
def xi_tz_site(Bd, s):
    return up(Bd, s)

def mu_tz_step(mu_tz, mu_t, mu_z, mu, xi_tz, xi_t, xi_z, xi, G, Gd):
    return (right(mu_tz, G, xi) + right(mu_t, G, xi_z) + right(mu_z, G, xi_t) + right(mu, G, xi_tz)
            + right(mu_z, Gd, xi) + right(mu, Gd, xi_z))

def nu_tz_step(nu_tz, nu_t, nu_z, nu, xi_tz, xi_t, xi_z, xi, G, Gd):
    return (left(G, xi, nu_tz) + left(G, xi_z, nu_t) + left(G, xi_t, nu_z) + left(G, xi_tz, nu)
            + left(Gd, xi, nu_z) + left(Gd, xi_z, nu))

def eta_tz_site(mu_tz, mu_t, mu_z, mu, nu_tz, nu_t, nu_z, nu, G, Gd):
    return (down(mu_tz, G, nu) + down(mu_t, G, nu_z) + down(mu_z, G, nu_t) + down(mu, G, nu_tz)
            + down(mu_z, Gd, nu) + down(mu, Gd, nu_z))

def K_B_site(eta_tz, eta_t, w, s):
    return grad_B(eta_tz, w) + grad_B(eta_t, s)

def K_G_site(mu_tz, mu_t, mu_z, mu, xi_tz, xi_t, xi_z, xi, nu_tz, nu_t, nu_z, nu):
    return (grad_G(mu_tz, xi, nu) + grad_G(mu, xi_tz, nu) + grad_G(mu, xi, nu_tz)
            + grad_G(mu_t, xi_z, nu) + grad_G(mu_t, xi, nu_z)
            + grad_G(mu_z, xi_t, nu) + grad_G(mu, xi_t, nu_z)
            + grad_G(mu_z, xi, nu_t) + grad_G(mu, xi_z, nu_t))


# ---- drivers: ordinary loops over sites.  mu[i] and nu[i] sit on bond i, i = 0..d ---------------------
def rightward(step, start, d, *per_site):
    msgs = [start]
    for i in range(d):
        msgs.append(step(msgs[i], *(a[i] for a in per_site)))
    return msgs

def leftward(step, start, d, *per_site):
    msgs = [None] * d + [start]
    for i in reversed(range(d)):
        msgs[i] = step(msgs[i + 1], *(a[i] for a in per_site))
    return msgs

def each_site(site, d, *per_site):
    return [site(*(a[i] for a in per_site)) for i in range(d)]


def main(repo):
    sys.path.append(repo)
    import t3toolbox.backend.probing as probing

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
    w = [rn((W, N)) for N in Ns]
    s = [rn((W, N)) for N in Ns]                                   # a per-sample residual
    one, zero = np.ones((W, 1)), np.zeros((W, 1))
    err = lambda mine, theirs: max(np.linalg.norm(a - b) / np.linalg.norm(b) for a, b in zip(mine, theirs))

    # ---- manifold frame -------------------------------------------------------------------------
    xi = each_site(xi_site, d, D, w)
    mu = rightward(mu_step, one, d, L, xi)                          # mu[0..d]
    nu = leftward(nu_step, one, d, R, xi)                           # nu[0..d]
    eta = each_site(eta_site, d, mu, U, nu[1:])

    xi_t = each_site(xi_t_site, d, V, w)
    mu_t = rightward(mu_t_step, zero, d, mu, xi, xi_t, R, U, H)
    nu_t = leftward(nu_t_step, zero, d, nu[1:], xi, xi_t, L, U, H)
    eta_t = each_site(eta_t_site, d, mu_t, mu, nu_t[1:], nu[1:], L, R, H)
    Jp = each_site(Jp_site, d, eta_t, eta, D, V)
    print('J p        against tv_probe                    %.1e'
          % err(Jp, probing.tv_probe(w, (V, H), (D, U, L, R))))

    xi_z = each_site(xi_z_site, d, D, s)
    mu_z = rightward(mu_z_step, zero, d, mu, xi, xi_z, L)
    nu_z = leftward(nu_z_step, zero, d, nu[1:], xi, xi_z, R)
    eta_z = each_site(eta_z_site, d, mu_z, mu, nu_z[1:], nu[1:], U)
    g_B = each_site(g_B_site, d, eta_z, eta, w, s)
    g_G = each_site(g_G_site, d, mu_z, mu, xi_z, xi, nu_z[1:], nu[1:])
    theirs = probing.tv_probe_transpose(s, w, (D, U, L, R), sum_over_probes=True)
    print('J^T r      against tv_probe_transpose          %.1e'
          % err(g_B + g_G, list(theirs[0]) + list(theirs[1])))

    # ---- corewise frame: the same functions, called with (B, G, G, G) and directions (Bd, Gd) ----
    B, G = D, L
    Bd, Gd = [rn(b.shape) for b in B], [rn(g.shape) for g in G]
    xi = each_site(xi_site, d, B, w)
    mu = rightward(mu_step, one, d, G, xi)
    nu = leftward(nu_step, one, d, G, xi)
    xi_t = each_site(xi_t_site, d, Bd, w)
    mu_t = rightward(mu_t_step, zero, d, mu, xi, xi_t, G, G, Gd)
    nu_t = leftward(nu_t_step, zero, d, nu[1:], xi, xi_t, G, G, Gd)
    eta_t = each_site(eta_t_site, d, mu_t, mu, nu_t[1:], nu[1:], G, G, Gd)
    xi_z = each_site(xi_z_site, d, B, s)
    mu_z = rightward(mu_z_step, zero, d, mu, xi, xi_z, G)
    nu_z = leftward(nu_z_step, zero, d, nu[1:], xi, xi_z, G)

    xi_tz = each_site(xi_tz_site, d, Bd, s)
    mu_tz = rightward(mu_tz_step, zero, d, mu_t, mu_z, mu, xi_tz, xi_t, xi_z, xi, G, Gd)
    nu_tz = leftward(nu_tz_step, zero, d, nu_t[1:], nu_z[1:], nu[1:], xi_tz, xi_t, xi_z, xi, G, Gd)
    eta_tz = each_site(eta_tz_site, d, mu_tz, mu_t, mu_z, mu, nu_tz[1:], nu_t[1:], nu_z[1:], nu[1:], G, Gd)
    K_B = each_site(K_B_site, d, eta_tz, eta_t, w, s)
    K_G = each_site(K_G_site, d, mu_tz, mu_t, mu_z, mu, xi_tz, xi_t, xi_z, xi,
                    nu_tz[1:], nu_t[1:], nu_z[1:], nu[1:])
    theirs = probing.t3_probe_corewise_hessian(s, w, (B, G), (Bd, Gd))
    print('two-hole   against t3_probe_corewise_hessian   %.1e'
          % err(K_B + K_G, list(theirs[0]) + list(theirs[1])))

    import inspect
    import re
    comps = [f for name, f in sorted(globals().items())
             if callable(f) and name.endswith(('_site', '_step')) and name != 'each_site']
    body = lambda f: inspect.getsource(f).split(':', 1)[1]              # after the signature
    terms = sum(len(re.findall(r'\b(?:up|right|left|down|out|grad_G|grad_B)\(', body(f))) for f in comps)
    print('%d component functions, %d terms in total, 7 literal subscript strings' % (len(comps), terms))

if __name__ == '__main__':
    main(sys.argv[1])
