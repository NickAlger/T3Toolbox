"""Assessment for target decision 2.7: can order 0 be the one-coefficient case of the SAME jet functions, with no
order axis ever created and no extra contractions?

Part 1.  The library today (jax): the plain probe kind against the derivative kind at order 0.  What does order 0
         cost through the jet kernels?  Counted as primitives in the traced program, and as XLA scratch memory.
Part 2.  The explicit form in Taylor coefficients, with the terms written through the library's `contract` and an
         order GROUP that is empty at order 0.  At order 0 it is compared, call by call and as a traced program,
         with a second implementation that knows nothing about jets.  At orders 2 and 4 it is compared with the
         library's derivative kind.

Nothing is timed.  Ragged cores, one sample axis, probe kind, the operations z, Jp and g = Jt c.

Usage:  python3 -I order0_assessment.py /path/to/T3Toolbox        (needs jax)
"""
import collections
import math
import re
import sys

import numpy as np

sys.path.append(sys.argv[1])
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import t3toolbox.backend.fitting as bfit
from t3toolbox.backend.contractions import contract as _contract

LOG = []


def contract(subscripts, *operands, **group_lens):
    out = _contract(subscripts, *operands, **group_lens)
    LOG.append((tuple(tuple(o.shape) for o in operands), tuple(out.shape)))
    return out


# ---- helpers that never create an order axis at order 0 -------------------------------------------------
def lenT(order):                           # the order group has no axis at order 0, one axis otherwise
    return 0 if order == 0 else 1


def add_shifted(c, x, r):                  # c[r:] += x
    if isinstance(c, np.ndarray):
        c = c.copy()
        c[r:] += x
        return c
    return c.at[r:].add(x)


def convolve(term, a, b, order):
    """c[t] = sum_r term(a[t - r], b[r]).  `a` is a full jet; `b` is an input jet (2 coefficients) or a full
    jet.  `term` is linear in `a` with the order group riding on it, and takes ONE coefficient of `b`.
    At order 0 neither operand has an order axis and this is one call."""
    if order == 0:
        return term(a, b)
    c = term(a, b[0])
    for r in range(1, b.shape[0]):
        c = add_shifted(c, term(a[:a.shape[0] - r], b[r]), r)
    return c


def top2(term, a, b, order):               # sum over r + s = order of term(a[r], b[s]); one call at order 0
    if order == 0:
        return term(a, b)
    out = None
    for s in range(b.shape[0]):
        r = order - s
        if 0 <= r < a.shape[0]:
            v = term(a[r], b[s])
            out = v if out is None else out + v
    return out


def top3(term, a, x, b, order):            # sum over r + s + u = order; one call at order 0
    if order == 0:
        return term(a, x, b)
    out = None
    for s in range(x.shape[0]):
        for u in range(b.shape[0]):
            r = order - s - u
            if 0 <= r < a.shape[0]:
                v = term(a[r], x[s], b[u])
                out = v if out is None else out + v
    return out


def start(value, W, r, order, like):       # ones for a plain sweep, zeros otherwise; no order axis at order 0
    xnp = np if isinstance(like, np.ndarray) else jnp
    if order == 0:
        return xnp.full((W, r), value, dtype=like.dtype)
    first = xnp.full((1, W, r), value, dtype=like.dtype)
    return xnp.concatenate([first, xnp.zeros((order, W, r), dtype=like.dtype)], axis=0)


# ---- the seven terms, twice ------------------------------------------------------------------------------
class JetTerms:
    """Order-aware.  One literal string per term; `T` is the order group, empty at order 0."""
    name = 'jet-capable terms'

    @staticmethod
    def xi_term(B, w, order):       return contract('io,TWo->TWi', B, w, len_T=lenT(order))
    @staticmethod
    def z_term(eta, B, order):      return contract('TWi,io->TWo', eta, B, len_T=lenT(order))
    @staticmethod
    def mu_term(mu, G, xi, order):
        return convolve(lambda a, x: contract('TWa,aib,Wi->TWb', a, G, x), mu, xi, order)
    @staticmethod
    def nu_term(G, xi, nu, order):
        return convolve(lambda b, x: contract('aib,Wi,TWb->TWa', G, x, b), nu, xi, order)
    @staticmethod
    def eta_term(mu, G, nu, order):
        return convolve(lambda a, b: contract('TWa,aib,Wb->TWi', a, G, b), mu, nu, order)
    @staticmethod
    def H_term(mu, xi, nu, order):
        return top3(lambda a, x, b: contract('Wa,Wi,Wb->aib', a, x, b), mu, xi, nu, order)
    @staticmethod
    def V_term(eta, w, order):
        return top2(lambda e, v: contract('Wi,Wo->io', e, v), eta, w, order)


class PlainTerms:
    """Knows nothing about jets: the reference for what order 0 should cost."""
    name = 'plain-only terms'

    @staticmethod
    def xi_term(B, w, order):       return contract('io,Wo->Wi', B, w)
    @staticmethod
    def z_term(eta, B, order):      return contract('Wi,io->Wo', eta, B)
    @staticmethod
    def mu_term(mu, G, xi, order):  return contract('Wa,aib,Wi->Wb', mu, G, xi)
    @staticmethod
    def nu_term(G, xi, nu, order):  return contract('aib,Wi,Wb->Wa', G, xi, nu)
    @staticmethod
    def eta_term(mu, G, nu, order): return contract('Wa,aib,Wb->Wi', mu, G, nu)
    @staticmethod
    def H_term(mu, xi, nu, order):  return contract('Wa,Wi,Wb->aib', mu, xi, nu)
    @staticmethod
    def V_term(eta, w, order):      return contract('Wi,Wo->io', eta, w)


# ---- the probe kind, written once for every order ---------------------------------------------------------
def left_to_right(step, first, d, *per_site):
    out = [first]
    for i in range(d):
        out.append(step(out[i], *(a[i] for a in per_site)))
    return out


def right_to_left(step, last, d, *per_site):
    out = [None] * d + [last]
    for i in reversed(range(d)):
        out[i] = step(out[i + 1], *(a[i] for a in per_site))
    return out


def each_site(step, d, *per_site):
    return [step(*(a[i] for a in per_site)) for i in range(d)]


def probe(T, order, w, w_tilde, frame, variation):
    """z, Jp and g = Jt (seed) for the probe kind.  `T` supplies the seven terms.  Every step function below
    takes exactly the arrays it reads, plus the static `order`."""
    D, U, L, R = frame
    V, H = variation
    d, W = len(D), w[0].shape[-2]
    o = order
    ones = lambda r: start(1.0, W, r, o, w[0])
    zeros = lambda r: start(0.0, W, r, o, w[0])

    xi = each_site(lambda D, w: T.xi_term(D, w, o), d, D, w)
    mu = left_to_right(lambda mu, L, xi: T.mu_term(mu, L, xi, o), ones(L[0].shape[0]), d, L, xi)
    nu = right_to_left(lambda nu, R, xi: T.nu_term(R, xi, nu, o), ones(R[-1].shape[2]), d, R, xi)
    eta = each_site(lambda mu, U, nu: T.eta_term(mu, U, nu, o), d, mu, U, nu[1:])
    nu_L = right_to_left(lambda nu_L, L, xi: T.nu_term(L, xi, nu_L, o), ones(L[-1].shape[2]), d, L, xi)
    z = each_site(lambda mu, L, nu_L, D: T.z_term(T.eta_term(mu, L, nu_L, o), D, o), d, mu, L, nu_L[1:], D)

    xi_U_dot = each_site(lambda V, w: T.xi_term(V, w, o), d, V, w)
    mu_R_dot = left_to_right(
        lambda mu_R_dot, mu, xi, xi_U_dot, R, U, H:
            T.mu_term(mu_R_dot, R, xi, o) + T.mu_term(mu, H, xi, o) + T.mu_term(mu, U, xi_U_dot, o),
        zeros(R[0].shape[0]), d, mu, xi, xi_U_dot, R, U, H)
    nu_L_dot = right_to_left(
        lambda nu_L_dot, nu, xi, xi_U_dot, L, U, H:
            T.nu_term(L, xi, nu_L_dot, o) + T.nu_term(H, xi, nu, o) + T.nu_term(U, xi_U_dot, nu, o),
        zeros(L[-1].shape[2]), d, nu[1:], xi, xi_U_dot, L, U, H)
    eta_D_dot = each_site(
        lambda mu_R_dot, mu, nu_L_dot, nu, L, R, H:
            T.eta_term(mu_R_dot, R, nu, o) + T.eta_term(mu, H, nu, o) + T.eta_term(mu, L, nu_L_dot, o),
        d, mu_R_dot, mu, nu_L_dot[1:], nu[1:], L, R, H)
    Jp = each_site(lambda eta_D_dot, eta, D, V: T.z_term(eta_D_dot, D, o) + T.z_term(eta, V, o),
                   d, eta_D_dot, eta, D, V)

    xi_tilde = each_site(lambda D, s: T.xi_term(D, s, o), d, D, w_tilde)
    mu_tilde = left_to_right(
        lambda mu_tilde, mu, xi, xi_tilde, L: T.mu_term(mu_tilde, L, xi, o) + T.mu_term(mu, L, xi_tilde, o),
        zeros(L[0].shape[0]), d, mu, xi, xi_tilde, L)
    nu_tilde = right_to_left(
        lambda nu_tilde, nu, xi, xi_tilde, R: T.nu_term(R, xi, nu_tilde, o) + T.nu_term(R, xi_tilde, nu, o),
        zeros(R[-1].shape[2]), d, nu[1:], xi, xi_tilde, R)
    eta_tilde = each_site(
        lambda mu_tilde, mu, nu_tilde, nu, U: T.eta_term(mu_tilde, U, nu, o) + T.eta_term(mu, U, nu_tilde, o),
        d, mu_tilde, mu, nu_tilde[1:], nu[1:], U)
    V_tilde = each_site(lambda eta_tilde, eta, w, s: T.V_term(eta_tilde, w, o) + T.V_term(eta, s, o),
                        d, eta_tilde, eta, w, w_tilde)
    H_tilde = each_site(
        lambda mu_tilde, mu, xi_tilde, xi, nu_tilde, nu:
            T.H_term(mu_tilde, xi, nu, o) + T.H_term(mu, xi_tilde, nu, o) + T.H_term(mu, xi, nu_tilde, o),
        d, mu_tilde, mu, xi_tilde, xi, nu_tilde[1:], nu[1:])
    return z, Jp, V_tilde + H_tilde


# ---- counting -------------------------------------------------------------------------------------------------
def primitives(fn, *args):
    text = str(jax.make_jaxpr(fn)(*args))
    return collections.Counter(re.findall(r' = ([a-z_]+)[\[ ]', text)), text


def scratch_bytes(fn, *args):
    return jax.jit(fn).lower(*args).compile().memory_analysis().temp_size_in_bytes


def show(counter, keys=('dot_general', 'while', 'scan', 'concatenate', 'pad', 'dynamic_slice', 'transpose',
                        'broadcast_in_dim', 'mul', 'add')):
    return '  '.join('%s %d' % (k, counter.get(k, 0)) for k in keys if counter.get(k, 0)) + \
           '   | all %d' % sum(counter.values())


def main():
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
    ww, pp = [rn((W, N)) for N in Ns], [rn((W, N)) for N in Ns]
    J = lambda x: jax.tree_util.tree_map(jnp.asarray, x)
    flat = lambda blocks: np.concatenate([np.ravel(np.asarray(b)) for b in blocks])
    err = lambda mine, theirs: np.linalg.norm(flat(mine) - flat(theirs)) / np.linalg.norm(flat(theirs))
    fact = lambda P, nd: np.array([math.factorial(t) for t in range(P)]).reshape((P,) + (1,) * (nd - 1))

    # ------------------------------------------------------------------------------------------------------
    print('PART 1.  The library today, on jax: the plain probe kind against the derivative kind at order 0')
    plain, jet0 = bfit.ProbeKind(), bfit.ProbeDerivativesKind(order=0, chunk_size=None)
    c0 = [rn((W, N)) for N in Ns]
    c0j = [c[None] for c in c0]                                   # the derivative kind wants an order axis
    ops = (
        ('z   (point_forward)',
         lambda ww, pp, fr, va, c: plain.point_forward((fr[0], fr[2]), ww),
         lambda ww, pp, fr, va, c: jet0.point_forward((fr[0], fr[2]), (ww, pp)), c0, c0j),
        ('Jp  (precompute + forward)',
         lambda ww, pp, fr, va, c: plain.forward(va, ww, fr, plain.precompute(fr, ww)),
         lambda ww, pp, fr, va, c: jet0.forward(va, (ww, pp), fr, jet0.precompute(fr, (ww, pp))), c0, c0j),
        ('g   (precompute + transpose)',
         lambda ww, pp, fr, va, c: plain.transpose(c, ww, fr, plain.precompute(fr, ww)),
         lambda ww, pp, fr, va, c: jet0.transpose(c, (ww, pp), fr, jet0.precompute(fr, (ww, pp))), c0, c0j),
    )
    for label, f_plain, f_jet, c_plain, c_jet in ops:
        a_plain = (J(ww), J(pp), J(frame), J(variation), J(c_plain))
        a_jet = (J(ww), J(pp), J(frame), J(variation), J(c_jet))
        hp, _ = primitives(f_plain, *a_plain)
        hj, text = primitives(f_jet, *a_jet)
        same = err(jax.tree_util.tree_leaves(f_jet(*a_jet)), jax.tree_util.tree_leaves(f_plain(*a_plain)))
        print('  %-30s results agree to %.0e' % (label, same))
        print('      plain kind         %s   | XLA scratch %6d bytes' % (show(hp), scratch_bytes(f_plain, *a_plain)))
        print('      derivative kind, 0 %s   | XLA scratch %6d bytes' % (show(hj), scratch_bytes(f_jet, *a_jet)))

    # ------------------------------------------------------------------------------------------------------
    print('\nPART 2.  The explicit form in Taylor coefficients, with an order group that is empty at order 0')
    seed0 = [rn((W, N)) for N in Ns]

    print('  order 0, numpy: jet-capable terms against plain-only terms, call by call')
    logs, outs = {}, {}
    for T in (PlainTerms, JetTerms):
        LOG.clear()
        outs[T] = probe(T, 0, ww, seed0, frame, variation)
        logs[T] = list(LOG)
    print('      contractions: plain-only %d, jet-capable %d;  same operand and result shapes, in the same order: %s'
          % (len(logs[PlainTerms]), len(logs[JetTerms]), logs[PlainTerms] == logs[JetTerms]))
    print('      largest ndim of any operand or result: plain-only %d, jet-capable %d'
          % tuple(max(len(s) for call in logs[T] for s in call[0] + (call[1],)) for T in (PlainTerms, JetTerms)))
    print('      results bitwise equal: %s'
          % all(np.array_equal(a, b) for x, y in zip(outs[PlainTerms], outs[JetTerms]) for a, b in zip(x, y)))
    sweep = plain.precompute(frame, ww)
    print('      against the library plain kind:  z %.0e   Jp %.0e   g %.0e'
          % (err(outs[JetTerms][0], plain.point_forward((D, L), ww)),
             err(outs[JetTerms][1], plain.forward(variation, ww, frame, sweep)),
             err(outs[JetTerms][2], [b for part in plain.transpose(seed0, ww, frame, sweep) for b in part])))

    print('  order 0, jax: the traced programs')
    args = (J(ww), J(seed0), J(frame), J(variation))
    hp, text_p = primitives(lambda w, s, fr, va: probe(PlainTerms, 0, w, s, fr, va), *args)
    hj, text_j = primitives(lambda w, s, fr, va: probe(JetTerms, 0, w, s, fr, va), *args)
    print('      plain-only   %s' % show(hp))
    print('      jet-capable  %s' % show(hj))
    print('      the two traced programs are identical, character for character: %s' % (text_p == text_j))

    for order in (2, 4):
        P = order + 1
        kind = bfit.ProbeDerivativesKind(order=order, chunk_size=None)
        w = [np.stack([ww[i], pp[i]]) for i in range(d)]                        # the input jet: (2, W, N)
        c_lib = [rn((P, W, N)) for N in Ns]                                     # a seed in the library's convention
        w_tilde = [(fact(P, 3) * c)[::-1].copy() for c in c_lib]                # Taylor, stored reversed
        LOG.clear()
        z, Jp, g = probe(JetTerms, order, w, w_tilde, frame, variation)
        n_calls = len(LOG)
        sweep = kind.precompute(frame, (ww, pp))
        zj, Jpj, gj = probe(JetTerms, order, J(w), J(w_tilde), J(frame), J(variation))
        print('  order %d: against the library derivative kind (converting at the comparison only):  '
              'z %.0e   Jp %.0e   g %.0e   | jax against numpy %.0e   | %d contractions'
              % (order,
                 err([fact(P, 3) * a for a in z], kind.point_forward((D, L), (ww, pp))),
                 err([fact(P, 3) * a for a in Jp], kind.forward(variation, (ww, pp), frame, sweep)),
                 err(g, [b for part in kind.transpose(c_lib, (ww, pp), frame, sweep) for b in part]),
                 err(list(zj) + list(Jpj) + list(gj), list(z) + list(Jp) + list(g)), n_calls))


if __name__ == '__main__':
    main()
