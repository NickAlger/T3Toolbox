"""The maintainer's arrow picture as a type checker for every contraction (target decision 2.4).

Every edge of the network gets an arrow.  Every leg of every array, core or variable, is labelled

    (family of the edge, position, sign)      sign = '+' if the arrow ENTERS the array at that leg, '-' if it LEAVES.

A contraction over an index is well typed when the two legs are on the same edge with opposite signs.  A sum is
well typed when the two arrays have the same legs.  The arrows of the frame, as the maintainer proposed
(2026-11-15, 05:26): every arrow points away from its store and towards the hole.

    D[i]  below the Tucker edge   i: leaves, o: enters          (bottom-to-top)     ('XD', i, -), ('N', i, +)
    L[i]  left of its bonds       a: enters, i: enters, b: leaves   (left-to-right)  ('AL', i, +), ('XD', i, +), ('AL', i+1, -)
    R[i]  right of its bonds      a: leaves, i: enters, b: enters   (right-to-left)  ('AR', i, -), ('XD', i, +), ('AR', i+1, +)
    U[i]  above the Tucker edge   a: enters, i: leaves (top-to-bottom), b: enters    ('AL', i, +), ('XU', i, -), ('AR', i+1, +)
    H[i]  a hole: every arrow enters                                                 ('AL', i, +), ('XD', i, +), ('AR', i+1, +)
    V[i]  a hole: every arrow enters                                                 ('XU', i, +), ('N', i, +)

The order index is one more leg.  Its core is the ring product, which the term functions use in one orientation
only, so an operand's order leg is either 'in' (it may enter a product) or 'out' (an output read at the exterior,
in forward order).

The program is the explicit one of jet_steps_check.py.  Its seven term functions are replaced by typed ones that
check the legs, then call the original on the numbers.  The 22 step functions are not touched.  The names of the
variables are those of SAMPLE_VARIABLES.md (2026-11-15, 23:23).

Usage:  python3 -I arrow_type_check.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jet_steps_check as J
import type_catalog_check as T

FLIP = {'+': '-', '-': '+'}
COUNT = {'contractions': 0, 'sums': 0}


class ArrowError(TypeError):
    pass


class A:
    """An array with a label on each of its network legs, and on its order leg."""
    def __init__(self, x, legs, order='in'):
        self.x, self.legs, self.order = x, tuple(legs), order

    def __add__(self, other):
        if self.legs != other.legs or self.order != other.order:
            raise ArrowError('sum of unlike types: %s and %s' % (show(self), show(other)))
        COUNT['sums'] += 1
        return A(self.x + other.x, self.legs, self.order)

    def __getitem__(self, k):               # J.top takes the top coefficient: the exterior readout of a block
        return A(self.x[k], self.legs, 'real')


def show(a):
    return '(' + ', '.join('%s%d%s' % leg for leg in a.legs) + '; order %s)' % a.order


def join(name, leg1, leg2):                 # one contraction over a network index
    if leg1[:2] != leg2[:2]:
        raise ArrowError('%s: contraction across two different edges, %s%d and %s%d' % ((name,) + leg1[:2] + leg2[:2]))
    if leg1[2] == leg2[2]:
        raise ArrowError('%s: on edge %s%d both legs have sign %s: the arrow points the wrong way'
                         % (name, leg1[0], leg1[1], leg1[2]))
    COUNT['contractions'] += 1


def jets_in(name, *operands):               # the ring product accepts only order legs that point in
    for a in operands:
        if a.order != 'in':
            raise ArrowError('%s: an operand\'s order leg is %r: it cannot enter the product' % (name, a.order))


RAW = {name: getattr(J, name) for name in ('xi_term', 'z_term', 'mu_term', 'nu_term', 'eta_term', 'H_term', 'V_term')}

def xi_term(B, w):
    join('xi_term', B.legs[1], w.legs[0]); jets_in('xi_term', w)
    return A(RAW['xi_term'](B.x, w.x), (B.legs[0],))

def z_term(eta, B):
    join('z_term', eta.legs[0], B.legs[0]); jets_in('z_term', eta)
    return A(RAW['z_term'](eta.x, B.x), (B.legs[1],))

def mu_term(mu, G, xi):
    join('mu_term', mu.legs[0], G.legs[0]); join('mu_term', xi.legs[0], G.legs[1]); jets_in('mu_term', mu, xi)
    return A(RAW['mu_term'](mu.x, G.x, xi.x), (G.legs[2],))

def nu_term(G, xi, nu):
    join('nu_term', xi.legs[0], G.legs[1]); join('nu_term', nu.legs[0], G.legs[2]); jets_in('nu_term', xi, nu)
    return A(RAW['nu_term'](G.x, xi.x, nu.x), (G.legs[0],))

def eta_term(mu, G, nu):
    join('eta_term', mu.legs[0], G.legs[0]); join('eta_term', nu.legs[0], G.legs[2]); jets_in('eta_term', mu, nu)
    return A(RAW['eta_term'](mu.x, G.x, nu.x), (G.legs[1],))

def H_term(mu, xi, nu):
    jets_in('H_term', mu, xi, nu)
    return A(RAW['H_term'](mu.x, xi.x, nu.x), (mu.legs[0], xi.legs[0], nu.legs[0]))

def V_term(eta, w):
    jets_in('V_term', eta, w)
    return A(RAW['V_term'](eta.x, w.x), (eta.legs[0], w.legs[0]))

TYPED = dict(xi_term=xi_term, z_term=z_term, mu_term=mu_term, nu_term=nu_term, eta_term=eta_term,
             H_term=H_term, V_term=V_term)


def typed_program(frame, V, H, B_dot, G_dot, w, w_tilde, d, P, W):
    """type_catalog_check.program, with the start vectors labelled.  Returns every variable, typed."""
    starts = iter([                         # in the order in which program() asks for them
        ('ones', ('AL', 0, '-')),           # mu      : left to right through L, with the arrows
        ('ones', ('AR', d, '-')),           # nu      : right to left through R, with the arrows
        ('ones', ('AL', d, '+')),           # nu_L    : right to left through L, against the arrows
    ])
    zero_legs = {'mu_dot_step': ('AR', 0, '+'), 'nu_dot_step': ('AL', d, '+'), 'mu_tilde_step': ('AL', 0, '-'),
                 'nu_tilde_step': ('AR', d, '-'), 'mu_tilde_dot_step': ('AL', 0, '-'),
                 'nu_tilde_dot_step': ('AL', d, '+')}
    seen = {'nu_tilde_step': 0, 'mu_dot_step': 0, 'nu_dot_step': 0}
    real_ones, real_l2r, real_r2l = J.ones_start, J.left_to_right, J.right_to_left

    def ones_start(P_, W_, r):
        return A(real_ones(P_, W_, r), (next(starts)[1],))

    def start_for(step, right):             # the zero start of each sweep, by the step function it feeds
        name = step.__name__
        if name in seen:
            seen[name] += 1
            if seen[name] == 2:             # second use: the base-point (Kp) sweep, on the L bonds
                return {'nu_tilde_step': ('AL', d, '+'), 'mu_dot_step': ('AL', 0, '-'), 'nu_dot_step': ('AL', d, '+')}[name]
        return zero_legs[name]

    def left_to_right(step, start, d_, *per_site):
        if not isinstance(start, A):
            start = A(start, (start_for(step, False),))
        return real_l2r(step, start, d_, *per_site)

    def right_to_left(step, start, d_, *per_site):
        if not isinstance(start, A):
            start = A(start, (start_for(step, True),))
        return real_r2l(step, start, d_, *per_site)

    saved = {k: getattr(J, k) for k in list(TYPED) + ['ones_start', 'left_to_right', 'right_to_left']}
    try:
        for k, f in TYPED.items():
            setattr(J, k, f)
        J.ones_start, J.left_to_right, J.right_to_left = ones_start, left_to_right, right_to_left
        return T.program(frame, V, H, B_dot, G_dot, w, w_tilde, d, P, W)
    finally:
        for k, f in saved.items():
            setattr(J, k, f)


def main():
    rng = np.random.default_rng(0)
    rn = rng.standard_normal
    Ns, nD, nU = (6, 5, 7, 4), (3, 4, 2, 3), (2, 3, 3, 2)
    rL, rR = (1, 3, 2, 4, 1), (1, 2, 4, 3, 1)
    d, W, order = len(Ns), 5, 2
    P = order + 1
    raw = dict(D=[rn((nD[i], Ns[i])) for i in range(d)], U=[rn((rL[i], nU[i], rR[i + 1])) for i in range(d)],
               L=[rn((rL[i], nD[i], rL[i + 1])) for i in range(d)], R=[rn((rR[i], nD[i], rR[i + 1])) for i in range(d)],
               V=[rn((nU[i], Ns[i])) for i in range(d)], H=[rn((rL[i], nD[i], rR[i + 1])) for i in range(d)])
    raw['B_dot'] = [rn(b.shape) for b in raw['D']]
    raw['G_dot'] = [rn(g.shape) for g in raw['L']]
    arrows = dict(                                           # the maintainer's orientation of the frame
        D=lambda i: (('XD', i, '-'), ('N', i, '+')),
        L=lambda i: (('AL', i, '+'), ('XD', i, '+'), ('AL', i + 1, '-')),
        R=lambda i: (('AR', i, '-'), ('XD', i, '+'), ('AR', i + 1, '+')),
        U=lambda i: (('AL', i, '+'), ('XU', i, '-'), ('AR', i + 1, '+')),
        H=lambda i: (('AL', i, '+'), ('XD', i, '+'), ('AR', i + 1, '+')),
        V=lambda i: (('XU', i, '+'), ('N', i, '+')),
        B_dot=lambda i: (('XD', i, '-'), ('N', i, '+')),     # a direction of D: the type of D
        G_dot=lambda i: (('AL', i, '+'), ('XD', i, '+'), ('AL', i + 1, '-')))   # a direction of L: the type of L
    typed = {k: [A(x, arrows[k](i), 'real') for i, x in enumerate(xs)] for k, xs in raw.items()}
    ww, pp = [rn((W, N)) for N in Ns], [rn((W, N)) for N in Ns]
    w_raw = [J.input_jet(P, ww[i], pp[i]) for i in range(d)]
    w = [A(x, (('N', i, '-'),)) for i, x in enumerate(w_raw)]                    # w points in
    y = [rn((P, W, N)) for N in Ns]
    omega = 0.5 + rng.random((d, P))

    def run(seed):
        return typed_program((typed['D'], typed['U'], typed['L'], typed['R']), typed['V'], typed['H'],
                             typed['B_dot'], typed['G_dot'], w, seed, d, P, W)

    def exterior(z):                        # an output, read at the exterior in forward order
        return [A(a.x, a.legs, 'out') for a in z]

    def riesz(r):                           # the one conversion: weights (the misfit), then the two turns
        return [A(((omega[i] ** 2).reshape((P, 1, 1)) * a.x)[::-1], [(s, k, FLIP[sign]) for s, k, sign in a.legs], 'in')
                for i, a in enumerate(r)]

    # ---- the whole program, typed --------------------------------------------------------------------------
    plain = run([A(np.zeros_like(x), (('N', i, '-'),)) for i, x in enumerate(w_raw)])     # a first pass, for z
    r = [A(a.x - y[i], a.legs, 'out') for i, a in enumerate(exterior(plain['z']))]         # the residual: type of z
    COUNT.update(contractions=0, sums=0)
    v = run(riesz(r))
    print('1. The whole program type checks with the maintainer\'s arrows: %d contractions over network indices and'
          % COUNT['contractions'])
    print('   %d sums, in the 22 step functions, at order %d with d = %d.  No error.' % (COUNT['sums'], order, d))
    untyped = T.program((raw['D'], raw['U'], raw['L'], raw['R']), raw['V'], raw['H'], raw['B_dot'], raw['G_dot'],
                        w_raw, [a.x for a in riesz(r)], d, P, W)
    worst = max(np.max(np.abs(a.x - b)) for name in untyped for a, b in zip(v[name], untyped[name]))
    print('   The numbers are those of the untyped program: largest difference %.0e.' % worst)

    print('2. Which way each variable flows (its leg at site or bond 1; "-" = the arrow leaves it = it flows WITH the arrow):')
    with_arrow = [n for n in v if n != 'w' and len(v[n][1].legs) == 1 and v[n][1].legs[0][2] == '-']
    against = [n for n in v if len(v[n][1].legs) == 1 and v[n][1].legs[0][2] == '+']
    print('   with the arrows:    w, ' + ', '.join(with_arrow))
    print('   against the arrows: ' + ', '.join(against))
    neg = lambda legs: tuple((s, k, FLIP[sign]) for s, k, sign in legs)
    print('   each block fits its hole (the legs of the block are the legs of the direction, every sign flipped):')
    for block, hole in (('V_tilde', 'V'), ('H_tilde', 'H'), ('V_tilde_dot', 'B_dot'), ('H_tilde_dot', 'G_dot')):
        print('      %-12s %-32s fits %-6s %s' % (block, show(v[block][1]), hole,
                                                  all(v[block][i].legs == neg(arrows[hole](i)) for i in range(d))))

    print('3. What the checker rejects')
    D, L, R = typed['D'], typed['L'], typed['R']
    weighted_only = [A((omega[i] ** 2).reshape((P, 1, 1)) * a.x, [(s, k, FLIP[sign]) for s, k, sign in a.legs], 'out')
                     for i, a in enumerate(r)]
    cases = (
        ('the residual put back as it is:          xi_term(D, r)', lambda: xi_term(D[1], r[1])),
        ('weights applied, order not reversed:     xi_term(D, omega^2 r)', lambda: xi_term(D[1], weighted_only[1])),
        ('mu swept through an R core:              mu_term(mu, R, xi)', lambda: mu_term(v['mu'][1], R[1], v['xi'][1])),
        ('nu swept through an L core:              nu_term(L, xi, nu)', lambda: nu_term(L[1], v['xi'][1], v['nu'][2])),
        ('nu_L used where nu is wanted:            eta_term(mu, U, nu_L)', lambda: eta_term(v['mu'][1], typed['U'][1], v['nu_L'][2])),
        ('a bond variable from the wrong bond:     mu_term(mu[2], L[1], xi[1])', lambda: mu_term(v['mu'][2], L[1], v['xi'][1])),
        ('a sum of mu and mu_R_dot:                mu + mu_R_dot', lambda: v['mu'][1] + v['mu_R_dot'][1]),
    )
    for label, call in cases:
        try:
            call()
            print('   %-68s accepted' % label)
        except ArrowError as e:
            print('   %-68s REJECTED: %s' % (label, e))


if __name__ == '__main__':
    main()
