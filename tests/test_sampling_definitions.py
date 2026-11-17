# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""``backend.sampling_definitions`` (the reference module) and the three dense sampling functions.

The reference module says what each sampling variable IS, and later tests hold the engine to it, so this file
pins the reference itself. Its lines of evidence, each by a route the reference does not take:

1. **The outputs against dense tensors** (`TestOutputsAgainstDense`). ``z`` against ``t3_to_dense`` contracted
   with the inputs; ``Jp`` against the dense tangent tensor (``tv_to_dense``); ``g``, ``V_tilde``, ``H_tilde``
   and ``JtJp`` against the dense Jacobian, built here column by column from ``tv_to_dense`` and transposed
   with the PLAIN slot-by-slot inner product (the reference pairs a turned seed at the top coefficient);
   ``r``, ``w_tilde``, ``f``, ``r_block_sumsq`` and ``Jp_norm_sq`` against their formulas. The sampling of a
   dense tensor and its jets are computed here by ``np.tensordot`` and by interpolation, not by the library's
   dense functions.
2. **The edge variables against the recurrences** (`TestRecurrences`): every recurrence of
   ``SAMPLE_VARIABLES.md`` section 5, written out here with truncated products as explicit loops. The reference
   contracts each whole subnetwork at once, so this is the page's statement that the two definitions agree.
3. **The readouts** (`TestReadouts`): section 6 of the page. The outputs and blocks (dense tensor, dense
   Jacobian, second derivative) equal the terms of the edge variables (subnetworks). This is what ties the
   tilde variables to the dense Jacobian and the tilde-dot variables to the second derivative.
4. **The pairings** (`TestPairings`): section 8 of the page, at every cut edge.
5. **Jets two ways** (`TestJetsTwoWays`): every variable at an order against the interpolation, at two other
   sets of nodes, of the same variable at order 0 along ``ww + s pp`` (with the stored seed as the polynomial
   ``sum_t w_tilde[t] s^t``, which is what "stored turned" means).
6. **Kp** (`TestKp`): automatic differentiation (jax arrays) against finite differences (numpy arrays), and
   that the route is read off the arrays; against the difference quotient of ``g`` on the corewise frame with the
   seed held; against the second derivative of the dense objective.
7. **The corewise frame** (`TestCorewiseFrame`), **structure** (`TestStructure`: the names, the one
   signature, the shapes, the errors, a doctest on every function, and that the module calls nothing of the
   library but the three dense functions) and **dispatch** (`TestJax`).
8. **The dense functions** (`TestDenseFunctions`): order 0 against hand-written contractions and indexing,
   jets against interpolation of order-0 calls, the ``W`` and ``C`` stacks, the errors.

Numerical comparisons are to a tolerance, structural ones exact. numpy only, by the house convention, except
`TestJax` and the jax half of `TestKp`; the dense functions under jit are in ``test_dispatch.py``.
"""
import ast
import inspect
import itertools
import math
import unittest
import unittest.mock

import numpy as np

import t3toolbox.backend.common as common
import t3toolbox.backend.sampling_definitions as sdef
import t3toolbox.backend.probing as probing
import t3toolbox.backend.apply as apply
import t3toolbox.backend.entries as entries
import t3toolbox.backend.t3_conversions as t3_conversions
import t3toolbox.backend.tv_operations as tv_operations

KINDS = ('probe', 'apply', 'entries')
ORDERS = (0, 1, 3)
TOL = 1e-10                 # relative, float64: sums of products of a dozen O(1) arrays, and a small solve

# The names of PART1_PLAN.md section 2.3, "The names the engine accepts", by group.
NAMES = {
    'inputs and seed':  ('w', 'w_tilde'),
    'plain':            ('xi', 'mu', 'nu', 'eta', 'nu_L', 'eta_D'),
    'dot, frame form':  ('xi_U_dot', 'mu_R_dot', 'nu_L_dot', 'eta_D_dot'),
    'dot, raw form':    ('xi_dot', 'mu_dot'),
    'tilde':            ('xi_tilde', 'mu_tilde', 'nu_tilde', 'eta_tilde', 'nu_L_tilde', 'eta_D_tilde'),
    'tilde-dot':        ('xi_tilde_dot', 'mu_tilde_dot', 'nu_L_tilde_dot', 'eta_D_tilde_dot'),
    'outputs':          ('z', 'r', 'Jp', 'g', 'JtJp', 'Kp', 'V_tilde', 'H_tilde', 'V_tilde_dot', 'H_tilde_dot'),
    'sums':             ('f', 'r_block_sumsq', 'Jp_norm_sq'),
}
ALL_NAMES = tuple(name for group in NAMES.values() for name in group)
BOND_NAMES = ('mu', 'nu', 'nu_L', 'mu_R_dot', 'nu_L_dot', 'mu_dot', 'mu_tilde', 'nu_tilde', 'nu_L_tilde',
              'mu_tilde_dot', 'nu_L_tilde_dot')
SITE_NAMES = ('w', 'xi', 'eta', 'eta_D', 'xi_U_dot', 'eta_D_dot', 'xi_dot', 'xi_tilde', 'eta_tilde', 'eta_D_tilde',
              'xi_tilde_dot', 'eta_D_tilde_dot', 'V_tilde', 'H_tilde', 'V_tilde_dot', 'H_tilde_dot')
MODE_NAMES = ('z', 'r', 'Jp', 'w_tilde')                    # indexed by the mode for probe, not indexed otherwise
UNINDEXED_NAMES = ('g', 'JtJp', 'Kp', 'f', 'r_block_sumsq', 'Jp_norm_sq')
AFFINE_NAMES = ('w', 'xi', 'xi_dot', 'xi_U_dot')            # the group A: two coefficients at any order >= 1


#######################################
########    Problems to test on    ####
#######################################

class Problem:
    """Random stores (not orthogonal, not consistent with each other: no definition needs that), a sample for
    each kind, both forms of a direction, data and a weight."""

    def __init__(self, seed, shape, nD, nU, rL, rR, W):
        rng = np.random.default_rng(seed)
        rn = rng.standard_normal
        d = len(shape)
        self.d, self.shape, self.W = d, tuple(shape), tuple(W)
        self.D = [rn((nD[i], shape[i])) for i in range(d)]
        self.U = [rn((rL[i], nU[i], rR[i + 1])) for i in range(d)]
        self.L = [rn((rL[i], nD[i], rL[i + 1])) for i in range(d)]
        self.R = [rn((rR[i], nD[i], rR[i + 1])) for i in range(d)]
        self.frame = (self.D, self.U, self.L, self.R)
        self.V = [rn((nU[i], shape[i])) for i in range(d)]
        self.H = [rn((rL[i], nD[i], rR[i + 1])) for i in range(d)]
        self.D_dot = [rn(B.shape) for B in self.D]
        self.L_dot = [rn(G.shape) for G in self.L]
        self.ww = [rn(self.W + (n,)) for n in shape]
        self.pp = [rn(self.W + (n,)) for n in shape]
        self.index = np.stack([rng.integers(0, n, size=self.W) for n in shape])
        self.rng = rng

    def inputs(self, kind):
        """The input vectors at s = 0: ``ww``, or the unit vectors at ``index``."""
        if kind == 'entries':
            return [np.eye(n)[self.index[i]] for i, n in enumerate(self.shape)]
        return self.ww

    def sample(self, kind, order):
        first = self.index if kind == 'entries' else self.ww
        return first if order == 0 else (first, self.pp)

    def data(self, kind, order):
        """``(y, weight)``: of the type of ``z``, and of the shape the weight has for the kind and the order."""
        T = (order + 1,) if order else ()
        rng = np.random.default_rng(1000 + order)
        if kind == 'probe':
            return [rng.standard_normal(T + self.W + (n,)) for n in self.shape], 0.5 + rng.random((self.d,) + T)
        return rng.standard_normal(T + self.W), 0.5 + rng.random(T)

    def arguments(self, kind, order):
        y, weight = self.data(kind, order)
        return dict(order=order, variations=(self.V, self.H), core_directions=(self.D_dot, self.L_dot),
                    y=y, weight=weight)


def small():        # no tails; one sample axis; every size different from its neighbours
    return Problem(0, (4, 3, 5), (2, 3, 2), (3, 2, 3), (1, 2, 3, 1), (1, 3, 2, 1), (3,))

def tails():        # end bonds larger than one; two sample axes
    return Problem(1, (3, 4, 3), (2, 2, 3), (3, 2, 2), (2, 2, 3, 2), (3, 3, 2, 2), (2, 2))

def long():         # four sites: jets of order 3 have every coefficient nonzero
    return Problem(2, (3, 2, 3, 2), (2, 2, 2, 2), (2, 3, 2, 2), (1, 2, 2, 2, 1), (1, 2, 3, 2, 1), (2,))

def two_sites():    # no sample axis at all: W = ()
    return Problem(3, (3, 4), (2, 3), (3, 2), (1, 2, 1), (1, 3, 1), ())

def one_site():     # one site: no bond inside, every trunk is one core, every sum over other sites is empty
    return Problem(4, (4,), (3,), (2,), (2, 2), (2, 3), (2,))

PROBLEMS = (('small', small), ('tails', tails), ('long', long), ('two sites, W = ()', two_sites),
            ('one site', one_site))


############################################################
########    Jets, written out: explicit loops over orders    ####
############################################################
#
# Inside these tests every jet has a leading order axis of length order + 1, at order 0 too.  `lift` brings a
# value of the reference into that form (an affine variable is padded with zeros), `tprod` multiplies jets.

def lift(value, order):
    """A reference value (one array, or a tuple of them) with a full leading order axis."""
    if isinstance(value, (tuple, list)):
        return [lift(v, order) for v in value]
    value = np.asarray(value)
    if order == 0:
        return value[None]
    if value.shape[0] == order + 1:
        return value
    assert value.shape[0] == 2, value.shape                 # an affine variable: the group A
    return np.concatenate([value, np.zeros((order - 1,) + value.shape[1:])])


def tprod(term, *jets):
    """The truncated product: slot t is the sum, over the orders that add up to t, of ``term`` of the slots."""
    P = jets[0].shape[0]
    out = None
    for orders in itertools.product(range(P), repeat=len(jets)):
        if sum(orders) < P:
            value = term(*(jet[t] for jet, t in zip(jets, orders)))
            if out is None:
                out = np.zeros((P,) + value.shape)
            out[sum(orders)] += value
    return out


def top(term, *jets):
    """The coefficient of the highest order of the product."""
    return tprod(term, *jets)[-1]


xi_term  = lambda B, w:        np.stack([np.einsum('io,...o->...i', B, w_t) for w_t in w])
z_term   = lambda eta, B:      np.stack([np.einsum('...i,io->...o', eta_t, B) for eta_t in eta])
mu_term  = lambda mu, G, xi:   tprod(lambda a, x: np.einsum('...a,aib,...i->...b', a, G, x), mu, xi)
nu_term  = lambda G, xi, nu:   tprod(lambda x, b: np.einsum('aib,...i,...b->...a', G, x, b), xi, nu)
eta_term = lambda mu, G, nu:   tprod(lambda a, b: np.einsum('...a,aib,...b->...i', a, G, b), mu, nu)

def V_top(eta, w):              # summed over the samples, at the top coefficient
    return top(lambda e, v: np.einsum('Wi,Wo->io', flat(e), flat(v)), eta, w)

def H_top(mu, xi, nu):
    return top(lambda a, x, b: np.einsum('Wa,Wi,Wb->aib', flat(a), flat(x), flat(b)), mu, xi, nu)

def flat(x_WX):                 # the sample stack as one axis
    return x_WX.reshape((-1,) + x_WX.shape[-1:])

def ones_jet(order, W, size):   # the start of a plain sweep: ones at order 0, zero above
    out = np.zeros((order + 1,) + W + (size,))
    out[0] = 1.0
    return out


def relative_error(got, expected):
    got, expected = np.asarray(got), np.asarray(expected)
    assert got.shape == expected.shape, (got.shape, expected.shape)
    return np.linalg.norm(got - expected) / max(np.linalg.norm(expected), 1.0)


#######################################################################
########    A dense tensor sampled, by a route of the tests' own    ####
#######################################################################

def sample_dense(kind, T, vectors):
    """Order 0: a dense tensor ``T`` (no stack) contracted with vectors ``W + (Ni,)``, by ``np.tensordot`` one
    sample at a time. probe: a list of ``W + (Ni,)``; apply and entries: ``W``."""
    d = T.ndim
    W = vectors[0].shape[:-1]
    flat_vectors = [v.reshape((-1, v.shape[-1])) for v in vectors]
    n_samples = flat_vectors[0].shape[0]

    def contract_all_but(k, open_mode):
        out = T
        for j in reversed(range(d)):                        # from the last mode, so the axis numbers stay
            if j != open_mode:
                out = np.tensordot(out, flat_vectors[j][k], axes=([j], [0]))
        return out
    if kind == 'probe':
        return [np.stack([contract_all_but(k, i) for k in range(n_samples)]).reshape(W + (T.shape[i],))
                for i in range(d)]
    return np.array([contract_all_but(k, None) for k in range(n_samples)]).reshape(W)


def interpolate(value_at, nodes, n_slots):
    """Taylor coefficients ``0 .. n_slots - 1`` of a polynomial in ``s``, from its values at ``nodes``."""
    values = np.stack([np.asarray(value_at(s)) for s in nodes])
    coefficients = np.linalg.solve(np.vander(nodes, increasing=True), values.reshape((len(nodes), -1)))
    return coefficients[:n_slots].reshape((n_slots,) + values.shape[1:])


def node_sets(degree):
    """Two sets of ``degree + 1`` nodes, neither of them the reference's (Chebyshev nodes on [-1, 1])."""
    return np.linspace(-1.0, 1.0, degree + 1), np.linspace(-0.6, 1.3, degree + 1)


def sample_dense_jet(P, kind, T, order, nodes=None):
    """``T`` sampled along ``inputs + s pp`` in Taylor coefficients, of the type of ``z``: by interpolation of
    `sample_dense`. At order 0 no order axis, as the reference returns it."""
    inputs = P.inputs(kind)
    if order == 0:
        return sample_dense(kind, T, inputs)
    nodes = node_sets(max(P.d, order))[0] if nodes is None else nodes     # the degree in s is at most d
    at = lambda s: sample_dense(kind, T, [w + s * p for w, p in zip(inputs, P.pp)])
    if kind == 'probe':
        return [interpolate(lambda s: at(s)[i], nodes, order + 1) for i in range(P.d)]
    return interpolate(at, nodes, order + 1)


def modes_of(kind, value):
    """An object of the type of ``z`` as a list of arrays: the modes for probe, one array otherwise."""
    return list(value) if kind == 'probe' else [value]


def weight_of(weight, kind, i, order):
    """The weight of mode ``i`` as an array ``(order + 1,)`` (one number at order 0)."""
    w = np.asarray(weight)[i] if kind == 'probe' else np.asarray(weight)
    return w.reshape((order + 1,))


def weighted(jets, weight, kind, order):
    """``weight**2 * jet``, slot by slot, on lifted jets (a list over the modes)."""
    return [(weight_of(weight, kind, i, order) ** 2).reshape((order + 1,) + (1,) * (jet.ndim - 1)) * jet
            for i, jet in enumerate(jets)]


def cases(problems=PROBLEMS, kinds=KINDS, orders=ORDERS):
    for name, make in problems:
        for kind in kinds:
            for order in orders:
                yield dict(problem=name, kind=kind, order=order), make(), kind, order


class TestOutputsAgainstDense(unittest.TestCase):
    """The outputs against dense tensors the library's conversions build, sampled by the tests' own route."""

    def check(self, got, expected, tol=TOL):
        self.assertLessEqual(relative_error(got, expected), tol)

    def test_z_is_the_dense_tensor_contracted_with_the_inputs(self):
        for labels, P, kind, order in cases():
            with self.subTest(**labels):
                T = t3_conversions.t3_to_dense((P.D, P.L))
                expected = modes_of(kind, sample_dense_jet(P, kind, T, order))
                got = modes_of(kind, sdef.z(kind, P.frame, P.sample(kind, order), order=order))
                self.assertEqual(len(got), len(expected))
                for a, b in zip(got, expected):
                    self.check(a, b)

    def test_Jp_is_the_dense_tangent_tensor_contracted_with_the_inputs(self):
        for labels, P, kind, order in cases():
            with self.subTest(**labels):
                T_dot = tv_operations.tv_to_dense(P.frame, (P.V, P.H))
                expected = modes_of(kind, sample_dense_jet(P, kind, T_dot, order))
                got = modes_of(kind, sdef.Jp(kind, P.frame, P.sample(kind, order), order=order,
                                             variations=(P.V, P.H)))
                for a, b in zip(got, expected):
                    self.check(a, b)

    def test_r_and_the_seed_and_the_sums_are_their_formulas(self):
        for labels, P, kind, order in cases():
            with self.subTest(**labels):
                sample, args = P.sample(kind, order), P.arguments(kind, order)
                y, weight = args['y'], args['weight']
                z = lift(modes_of(kind, sdef.z(kind, P.frame, sample, order=order)), order)
                r = lift(modes_of(kind, sdef.r(kind, P.frame, sample, **args)), order)
                for r_i, z_i, y_i in zip(r, z, lift(modes_of(kind, y), order)):
                    self.check(r_i, z_i - y_i)

                # the seed: the weights, then the turn of the order axis
                seed = lift(modes_of(kind, sdef.w_tilde(kind, P.frame, sample, **args)), order)
                for seed_i, weighted_r_i in zip(seed, weighted(r, weight, kind, order)):
                    self.check(seed_i, weighted_r_i[::-1])
                if order:                                   # and the turn is not a no-op here
                    self.assertGreater(relative_error(seed[0], weighted(r, weight, kind, order)[0]), 1e-3)

                f = sdef.f(kind, P.frame, sample, **args)
                self.assertEqual(np.shape(f), ())
                self.check(f, 0.5 * sum(np.sum(wr * r_i) for wr, r_i in zip(weighted(r, weight, kind, order), r)))

                sums = sdef.r_block_sumsq(kind, P.frame, sample, **args)
                self.assertEqual(np.shape(sums), np.shape(weight))          # structural: exact
                expected = np.array([[np.sum(r_i[t] ** 2) for t in range(order + 1)] for r_i in r])
                self.check(np.reshape(sums, expected.shape), expected)

                Jp = lift(modes_of(kind, sdef.Jp(kind, P.frame, sample, **args)), order)
                Jp_norm_sq = sdef.Jp_norm_sq(kind, P.frame, sample, **args)
                self.assertEqual(np.shape(Jp_norm_sq), ())
                self.check(Jp_norm_sq, sum(np.sum(wJ * J) for wJ, J in zip(weighted(Jp, weight, kind, order), Jp)))

    def jacobian_columns(self, P, kind, order):
        """The dense Jacobian of ``(V, H) -> Jp``, one column per unit direction: a list, over the blocks
        ``V[0..d-1], H[0..d-1]``, of arrays ``block shape + (list over modes of lifted jets)``."""
        zero_V, zero_H = [0 * V for V in P.V], [0 * H for H in P.H]
        blocks = []
        for which, cores in ((0, P.V), (1, P.H)):
            for i, block in enumerate(cores):
                columns = {}
                for entry in np.ndindex(block.shape):
                    unit = np.zeros(block.shape)
                    unit[entry] = 1.0
                    V = zero_V[:i] + [unit] + zero_V[i + 1:] if which == 0 else zero_V
                    H = zero_H[:i] + [unit] + zero_H[i + 1:] if which == 1 else zero_H
                    T_dot = tv_operations.tv_to_dense(P.frame, (V, H))
                    columns[entry] = lift(modes_of(kind, sample_dense_jet(P, kind, T_dot, order)), order)
                blocks.append((block.shape, columns))
        return blocks

    def transposed(self, blocks, functional):
        """The Jacobian transposed onto a functional (a list over the modes of lifted jets, forward order),
        with the PLAIN inner product, slot by slot."""
        out = []
        for shape, columns in blocks:
            block = np.zeros(shape)
            for entry, column in columns.items():
                block[entry] = sum(np.sum(c * f) for c, f in zip(column, functional))
            out.append(block)
        return out

    def test_g_and_JtJp_are_the_dense_jacobian_transposed(self):
        for labels, P, kind, order in cases(problems=PROBLEMS[:2]):
            with self.subTest(**labels):
                sample, args = P.sample(kind, order), P.arguments(kind, order)
                weight = args['weight']
                blocks = self.jacobian_columns(P, kind, order)

                r = lift(modes_of(kind, sdef.r(kind, P.frame, sample, **args)), order)
                expected = self.transposed(blocks, weighted(r, weight, kind, order))
                V_tilde, H_tilde = sdef.g(kind, P.frame, sample, **args)
                self.assertEqual((len(V_tilde), len(H_tilde)), (P.d, P.d))
                for a, b in zip(list(V_tilde) + list(H_tilde), expected):
                    self.check(a, b)
                # the two blocks' own functions are the same arrays
                for a, b in zip(sdef.V_tilde(kind, P.frame, sample, **args), V_tilde):
                    self.assertTrue(np.array_equal(a, b))
                for a, b in zip(sdef.H_tilde(kind, P.frame, sample, **args), H_tilde):
                    self.assertTrue(np.array_equal(a, b))

                Jp = lift(modes_of(kind, sdef.Jp(kind, P.frame, sample, **args)), order)
                expected = self.transposed(blocks, weighted(Jp, weight, kind, order))
                JtJp = sdef.JtJp(kind, P.frame, sample, **args)
                for a, b in zip(list(JtJp[0]) + list(JtJp[1]), expected):
                    self.check(a, b)

    def test_a_given_seed_is_used_as_it_is(self):
        # g with the residual's seed passed in as w_tilde is g from y and the weight; and a seed of one's own is
        # paired turned: the transpose of an UN-turned functional c is g(w_tilde = c turned).
        for labels, P, kind, order in cases(problems=PROBLEMS[:1]):
            with self.subTest(**labels):
                sample, args = P.sample(kind, order), P.arguments(kind, order)
                seed = sdef.w_tilde(kind, P.frame, sample, **args)
                from_y = sdef.g(kind, P.frame, sample, **args)
                from_seed = sdef.g(kind, P.frame, sample, order=order, w_tilde=seed)
                for a, b in zip(from_y[0] + from_y[1], from_seed[0] + from_seed[1]):
                    self.assertTrue(np.array_equal(a, b))

                c = [P.rng.standard_normal(np.shape(s)) for s in modes_of(kind, seed)]      # forward order
                turned = [c_i[::-1] if order else c_i for c_i in c]
                got = sdef.g(kind, P.frame, sample, order=order, w_tilde=turned if kind == 'probe' else turned[0])
                expected = self.transposed(self.jacobian_columns(P, kind, order), lift(c, order))
                for a, b in zip(list(got[0]) + list(got[1]), expected):
                    self.check(a, b)


class Variables:
    """Every variable of one case from the reference, lifted, by name: ``v.mu[i]`` is a jet with a full order
    axis. Computed on first use."""

    def __init__(self, P, kind, order, **overrides):
        self.P, self.kind, self.order = P, kind, order
        self.args = dict(P.arguments(kind, order), **overrides)
        self.sample = P.sample(kind, order)
        self.cache = {}

    def __getattr__(self, name):
        if name not in ALL_NAMES:
            raise AttributeError(name)
        if name not in self.cache:
            self.cache[name] = lift(getattr(sdef, name)(self.kind, self.P.frame, self.sample, **self.args), self.order)
        return self.cache[name]


class TestRecurrences(unittest.TestCase):
    """SAMPLE_VARIABLES.md section 5: each variable, contracted whole by the reference, satisfies its
    recurrence. The recurrences are written out here from the page."""

    def check(self, got, expected, what):
        self.assertLessEqual(relative_error(got, expected), TOL, what)

    def test_plain(self):
        for labels, P, kind, order in cases():
            with self.subTest(**labels):
                v, d = Variables(P, kind, order), P.d
                D, U, L, R = P.frame
                for i in range(d):
                    self.check(v.xi[i], xi_term(D[i], v.w[i]), 'xi')
                    self.check(v.mu[i + 1], mu_term(v.mu[i], L[i], v.xi[i]), 'mu')
                    self.check(v.nu[i], nu_term(R[i], v.xi[i], v.nu[i + 1]), 'nu')
                    self.check(v.nu_L[i], nu_term(L[i], v.xi[i], v.nu_L[i + 1]), 'nu_L')
                    self.check(v.eta[i], eta_term(v.mu[i], U[i], v.nu[i + 1]), 'eta')
                    self.check(v.eta_D[i], eta_term(v.mu[i], L[i], v.nu_L[i + 1]), 'eta_D')
                # the starts: ones in slot 0 and zero above, sized by the end bond of the cores (tails). Exact
                # at order 0; with jets they come out of the interpolation, so to rounding.
                for start, size in ((v.mu[0], L[0].shape[0]), (v.nu[d], R[d - 1].shape[2]),
                                    (v.nu_L[d], L[d - 1].shape[2])):
                    self.assertEqual(start.shape, (order + 1,) + P.W + (size,))
                    if order == 0:
                        self.assertTrue(np.array_equal(start, ones_jet(order, P.W, size)))
                    self.check(start, ones_jet(order, P.W, size), 'a start')

    def test_dot_in_the_frame_form(self):
        for labels, P, kind, order in cases():
            with self.subTest(**labels):
                v, d = Variables(P, kind, order, core_directions=None), P.d
                D, U, L, R = P.frame
                V, H = P.V, P.H
                for i in range(d):
                    self.check(v.xi_U_dot[i], xi_term(V[i], v.w[i]), 'xi_U_dot')
                    self.check(v.mu_R_dot[i + 1],
                               mu_term(v.mu_R_dot[i], R[i], v.xi[i]) + mu_term(v.mu[i], H[i], v.xi[i])
                               + mu_term(v.mu[i], U[i], v.xi_U_dot[i]), 'mu_R_dot')
                    self.check(v.nu_L_dot[i],
                               nu_term(L[i], v.xi[i], v.nu_L_dot[i + 1]) + nu_term(H[i], v.xi[i], v.nu[i + 1])
                               + nu_term(U[i], v.xi_U_dot[i], v.nu[i + 1]), 'nu_L_dot')
                    self.check(v.eta_D_dot[i],
                               eta_term(v.mu_R_dot[i], R[i], v.nu[i + 1]) + eta_term(v.mu[i], H[i], v.nu[i + 1])
                               + eta_term(v.mu[i], L[i], v.nu_L_dot[i + 1]), 'eta_D_dot')
                self.assertTrue(np.all(v.mu_R_dot[0] == 0) and v.mu_R_dot[0].shape[-1] == R[0].shape[0])
                self.assertTrue(np.all(v.nu_L_dot[d] == 0) and v.nu_L_dot[d].shape[-1] == L[d - 1].shape[2])

    def test_dot_in_the_raw_form(self):
        for labels, P, kind, order in cases():
            with self.subTest(**labels):
                v, d = Variables(P, kind, order, variations=None), P.d
                D, U, L, R = P.frame
                D_dot, L_dot = P.D_dot, P.L_dot
                for i in range(d):
                    self.check(v.xi_dot[i], xi_term(D_dot[i], v.w[i]), 'xi_dot')
                    self.check(v.mu_dot[i + 1],
                               mu_term(v.mu_dot[i], L[i], v.xi[i]) + mu_term(v.mu[i], L_dot[i], v.xi[i])
                               + mu_term(v.mu[i], L[i], v.xi_dot[i]), 'mu_dot')
                    self.check(v.nu_L_dot[i],
                               nu_term(L[i], v.xi[i], v.nu_L_dot[i + 1]) + nu_term(L_dot[i], v.xi[i], v.nu_L[i + 1])
                               + nu_term(L[i], v.xi_dot[i], v.nu_L[i + 1]), 'nu_L_dot (raw)')
                    self.check(v.eta_D_dot[i],
                               eta_term(v.mu_dot[i], L[i], v.nu_L[i + 1]) + eta_term(v.mu[i], L_dot[i], v.nu_L[i + 1])
                               + eta_term(v.mu[i], L[i], v.nu_L_dot[i + 1]), 'eta_D_dot (raw)')
                self.assertTrue(np.all(v.mu_dot[0] == 0) and np.all(v.nu_L_dot[d] == 0))

    def seed_starts(self, v):
        """What the seed puts in: the seeded leaves, and the start of the right-to-left tilde sweeps. probe: the
        seed at the inputs, sweeps from zero. apply, entries: no seeded leaf, sweeps from the seed times ones."""
        P, d, order = v.P, v.P.d, v.order
        D, U, L, R = P.frame
        if v.kind == 'probe':
            xi_tilde = [xi_term(D[i], v.w_tilde[i]) for i in range(d)]
            xi_tilde_dot = [xi_term(P.D_dot[i], v.w_tilde[i]) for i in range(d)]
            start = lambda size: np.zeros((order + 1,) + P.W + (size,))
        else:
            xi_tilde = [np.zeros((order + 1,) + P.W + (D[i].shape[0],)) for i in range(d)]
            xi_tilde_dot = xi_tilde
            start = lambda size: v.w_tilde[..., None] * np.ones(size)
        return xi_tilde, xi_tilde_dot, start

    def test_tilde(self):
        for labels, P, kind, order in cases():
            with self.subTest(**labels):
                v, d = Variables(P, kind, order), P.d
                D, U, L, R = P.frame
                xi_tilde, _, start = self.seed_starts(v)
                for i in range(d):
                    self.check(v.xi_tilde[i], xi_tilde[i], 'xi_tilde')
                    self.check(v.mu_tilde[i + 1],
                               mu_term(v.mu_tilde[i], L[i], v.xi[i]) + mu_term(v.mu[i], L[i], v.xi_tilde[i]),
                               'mu_tilde')
                    self.check(v.nu_tilde[i],
                               nu_term(R[i], v.xi[i], v.nu_tilde[i + 1]) + nu_term(R[i], v.xi_tilde[i], v.nu[i + 1]),
                               'nu_tilde')
                    self.check(v.nu_L_tilde[i],
                               nu_term(L[i], v.xi[i], v.nu_L_tilde[i + 1])
                               + nu_term(L[i], v.xi_tilde[i], v.nu_L[i + 1]), 'nu_L_tilde')
                    self.check(v.eta_tilde[i],
                               eta_term(v.mu_tilde[i], U[i], v.nu[i + 1]) + eta_term(v.mu[i], U[i], v.nu_tilde[i + 1]),
                               'eta_tilde')
                    self.check(v.eta_D_tilde[i],
                               eta_term(v.mu_tilde[i], L[i], v.nu_L[i + 1])
                               + eta_term(v.mu[i], L[i], v.nu_L_tilde[i + 1]), 'eta_D_tilde')
                self.assertTrue(np.all(v.mu_tilde[0] == 0))
                self.check(v.nu_tilde[d], start(R[d - 1].shape[2]), 'the start of nu_tilde')
                self.check(v.nu_L_tilde[d], start(L[d - 1].shape[2]), 'the start of nu_L_tilde')
                if kind != 'probe':                         # no left side and no leaf holds the seed: exact zeros
                    for i in range(d):
                        self.assertTrue(np.all(v.xi_tilde[i] == 0) and np.all(v.mu_tilde[i + 1] == 0))

    def test_tilde_dot(self):
        for labels, P, kind, order in cases():
            with self.subTest(**labels):
                v, d = Variables(P, kind, order, variations=None), P.d      # the raw dots
                D, U, L, R = P.frame
                D_dot, L_dot = P.D_dot, P.L_dot
                _, xi_tilde_dot, _ = self.seed_starts(v)
                for i in range(d):
                    self.check(v.xi_tilde_dot[i], xi_tilde_dot[i], 'xi_tilde_dot')
                    self.check(v.mu_tilde_dot[i + 1],
                               mu_term(v.mu_tilde_dot[i], L[i], v.xi[i]) + mu_term(v.mu_dot[i], L[i], v.xi_tilde[i])
                               + mu_term(v.mu_tilde[i], L[i], v.xi_dot[i]) + mu_term(v.mu[i], L[i], v.xi_tilde_dot[i])
                               + mu_term(v.mu_tilde[i], L_dot[i], v.xi[i]) + mu_term(v.mu[i], L_dot[i], v.xi_tilde[i]),
                               'mu_tilde_dot')
                    self.check(v.nu_L_tilde_dot[i],
                               nu_term(L[i], v.xi[i], v.nu_L_tilde_dot[i + 1])
                               + nu_term(L[i], v.xi_tilde[i], v.nu_L_dot[i + 1])
                               + nu_term(L[i], v.xi_dot[i], v.nu_L_tilde[i + 1])
                               + nu_term(L[i], v.xi_tilde_dot[i], v.nu_L[i + 1])
                               + nu_term(L_dot[i], v.xi[i], v.nu_L_tilde[i + 1])
                               + nu_term(L_dot[i], v.xi_tilde[i], v.nu_L[i + 1]), 'nu_L_tilde_dot')
                    self.check(v.eta_D_tilde_dot[i],
                               eta_term(v.mu_tilde_dot[i], L[i], v.nu_L[i + 1])
                               + eta_term(v.mu_dot[i], L[i], v.nu_L_tilde[i + 1])
                               + eta_term(v.mu_tilde[i], L[i], v.nu_L_dot[i + 1])
                               + eta_term(v.mu[i], L[i], v.nu_L_tilde_dot[i + 1])
                               + eta_term(v.mu_tilde[i], L_dot[i], v.nu_L[i + 1])
                               + eta_term(v.mu[i], L_dot[i], v.nu_L_tilde[i + 1]), 'eta_D_tilde_dot')
                self.assertTrue(np.all(v.mu_tilde_dot[0] == 0) and np.all(v.nu_L_tilde_dot[d] == 0))


class TestReadouts(unittest.TestCase):
    """SAMPLE_VARIABLES.md section 6: the outputs and the blocks, which the reference takes from dense tensors,
    from the dense Jacobian and from the second derivative, are the terms of the edge variables, which it takes
    from subnetworks. This ties the tilde variables to the Jacobian and the tilde-dot variables to ``Kp``."""

    def check(self, got, expected, what):
        self.assertLessEqual(relative_error(got, expected), TOL, what)

    def test_z_and_Jp(self):
        for labels, P, kind, order in cases():
            with self.subTest(**labels):
                v, d = Variables(P, kind, order, core_directions=None), P.d
                D, U, L, R = P.frame
                if kind == 'probe':
                    for i in range(d):
                        self.check(v.z[i], z_term(v.eta_D[i], D[i]), 'z')
                        self.check(v.Jp[i], z_term(v.eta_D_dot[i], D[i]) + z_term(v.eta[i], P.V[i]), 'Jp')
                else:                                       # read at the right end: bond d, summed with the ones
                    self.check(v.z, v.mu[d].sum(axis=-1), 'z')
                    self.check(v.Jp, v.mu_R_dot[d].sum(axis=-1), 'Jp')

    def test_the_blocks_of_g(self):
        for labels, P, kind, order in cases():
            with self.subTest(**labels):
                v, d = Variables(P, kind, order), P.d
                V_tilde, H_tilde = sdef.g(kind, P.frame, v.sample, **v.args)
                for i in range(d):
                    V_block = V_top(v.eta_tilde[i], v.w[i])
                    if kind == 'probe':                     # the seed at input i itself
                        V_block = V_block + V_top(v.eta[i], v.w_tilde[i])
                    self.check(V_tilde[i], V_block, 'V_tilde')
                    self.check(H_tilde[i],
                               H_top(v.mu_tilde[i], v.xi[i], v.nu[i + 1]) + H_top(v.mu[i], v.xi_tilde[i], v.nu[i + 1])
                               + H_top(v.mu[i], v.xi[i], v.nu_tilde[i + 1]), 'H_tilde')

    def test_the_blocks_of_Kp(self):
        for labels, P, kind, order in cases():
            with self.subTest(**labels):
                v, d = Variables(P, kind, order, variations=None), P.d      # the raw dots
                V_tilde_dot, H_tilde_dot = sdef.Kp(kind, P.frame, v.sample, **v.args)
                for i in range(d):
                    V_block = V_top(v.eta_D_tilde_dot[i], v.w[i])
                    if kind == 'probe':
                        V_block = V_block + V_top(v.eta_D_dot[i], v.w_tilde[i])
                    self.check(V_tilde_dot[i], V_block, 'V_tilde_dot')
                    mu = {'': v.mu[i], 'd': v.mu_dot[i], 't': v.mu_tilde[i], 'td': v.mu_tilde_dot[i]}
                    xi = {'': v.xi[i], 'd': v.xi_dot[i], 't': v.xi_tilde[i], 'td': v.xi_tilde_dot[i]}
                    nu = {'': v.nu_L[i + 1], 'd': v.nu_L_dot[i + 1], 't': v.nu_L_tilde[i + 1],
                          'td': v.nu_L_tilde_dot[i + 1]}
                    nine = [g for g in itertools.product(('', 'd', 't', 'td'), repeat=3)
                            if ''.join(sorted(''.join(g))) == 'dt']           # one tilde and one dot in all
                    self.assertEqual(len(nine), 9)
                    self.check(H_tilde_dot[i], sum(H_top(mu[a], xi[b], nu[c]) for a, b, c in nine), 'H_tilde_dot')
                # the two blocks' own functions are the same arrays
                for a, b in zip(sdef.V_tilde_dot(kind, P.frame, v.sample, **v.args), V_tilde_dot):
                    self.assertTrue(np.array_equal(a, b))
                for a, b in zip(sdef.H_tilde_dot(kind, P.frame, v.sample, **v.args), H_tilde_dot):
                    self.assertTrue(np.array_equal(a, b))


def pair(a, b):
    """Two jets on the two sides of one edge, contracted over the edge: a jet ``T + W``."""
    return tprod(lambda p, q: np.einsum('...a,...a->...', p, q), a, b)


def top_pair(a, b):
    """The number two jets stand for together: the top coefficient of their product, summed over everything."""
    return float(np.sum(pair(a, b)[-1]))


class TestPairings(unittest.TestCase):
    """SAMPLE_VARIABLES.md section 8, at every cut edge."""

    def test_the_direction_at_every_cut_edge(self):
        # At a cut bond the direction's sampled network is mu_R_dot paired with nu plus mu paired with nu_L_dot;
        # at a cut Tucker edge, xi_U_dot paired with eta plus xi paired with eta_D_dot. Every cut gives the same
        # jet per sample: the direction's network with every mode contracted, which is Jp of apply / entries.
        for labels, P, kind, order in cases(kinds=('apply', 'entries')):
            with self.subTest(**labels):
                v, d = Variables(P, kind, order, core_directions=None), P.d
                cuts = ([pair(v.mu_R_dot[i], v.nu[i]) + pair(v.mu[i], v.nu_L_dot[i]) for i in range(d + 1)]
                        + [pair(v.xi_U_dot[i], v.eta[i]) + pair(v.xi[i], v.eta_D_dot[i]) for i in range(d)])
                self.assertEqual(len(cuts), 2 * d + 1)
                for cut in cuts:
                    self.assertLessEqual(relative_error(cut, v.Jp), TOL)

    def one_sided(self, P, keep):
        """The direction ``(V, H)`` with its cores zeroed at the sites where ``keep(j)`` is false."""
        return ([V if keep(j) else 0 * V for j, V in enumerate(P.V)],
                [H if keep(j) else 0 * H for j, H in enumerate(P.H)])

    def check(self, got, expected):
        self.assertLessEqual(abs(got - expected), TOL * max(abs(expected), 1.0))

    def test_the_multipliers_of_the_probe_kind(self):
        # The tilde of a variable is the multiplier of the dot of the other side of the same edge, in the same
        # basis: with the direction wholly on one side of the edge, the seed paired with Jp, over the modes on
        # the OTHER side, is the tilde variable paired with the dotted one.
        for labels, P, kind, order in cases(kinds=('probe',)):
            with self.subTest(**labels):
                d = P.d
                base = Variables(P, kind, order)
                seed = base.w_tilde

                def modes(direction, which):
                    Jp = Variables(P, kind, order, variations=direction).Jp
                    return sum(top_pair(seed[k], Jp[k]) for k in which)

                for i in range(d + 1):
                    right = self.one_sided(P, lambda j: j >= i)                 # mu_tilde multiplies nu_L_dot
                    v = Variables(P, kind, order, variations=right, core_directions=None)
                    self.check(top_pair(base.mu_tilde[i], v.nu_L_dot[i]), modes(right, range(i)))
                    left = self.one_sided(P, lambda j: j < i)                   # nu_tilde multiplies mu_R_dot
                    v = Variables(P, kind, order, variations=left, core_directions=None)
                    self.check(top_pair(base.nu_tilde[i], v.mu_R_dot[i]), modes(left, range(i, d)))
                for i in range(d):
                    leaf = (self.one_sided(P, lambda j: j == i)[0], [0 * H for H in P.H])   # only V[i]
                    v = Variables(P, kind, order, variations=leaf, core_directions=None)
                    self.check(top_pair(base.eta_tilde[i], v.xi_U_dot[i]),      # eta_tilde multiplies xi_U_dot
                               modes(leaf, [k for k in range(d) if k != i]))
                    trunk = (self.one_sided(P, lambda j: j != i)[0], P.H)       # everything but V[i]
                    v = Variables(P, kind, order, variations=trunk, core_directions=None)
                    self.check(top_pair(base.xi_tilde[i], v.eta_D_dot[i]), modes(trunk, [i]))   # xi_tilde: eta_D_dot

    def test_the_multipliers_of_apply_and_entries(self):
        # The seed sits at bond d, so it is on the right of every bond and in every trunk: nu_tilde multiplies
        # mu_R_dot and eta_tilde multiplies xi_U_dot, and there is no mu_tilde and no xi_tilde.
        for labels, P, kind, order in cases(kinds=('apply', 'entries')):
            with self.subTest(**labels):
                d = P.d
                base = Variables(P, kind, order)
                def paired_with_Jp(direction):
                    Jp = Variables(P, kind, order, variations=direction).Jp
                    return top_pair(base.w_tilde[..., None], Jp[..., None])
                for i in range(d + 1):
                    left = self.one_sided(P, lambda j: j < i)
                    v = Variables(P, kind, order, variations=left, core_directions=None)
                    self.check(top_pair(base.nu_tilde[i], v.mu_R_dot[i]), paired_with_Jp(left))
                    self.assertTrue(np.all(base.mu_tilde[i] == 0))
                for i in range(d):
                    leaf = (self.one_sided(P, lambda j: j == i)[0], [0 * H for H in P.H])
                    v = Variables(P, kind, order, variations=leaf, core_directions=None)
                    self.check(top_pair(base.eta_tilde[i], v.xi_U_dot[i]), paired_with_Jp(leaf))
                    self.assertTrue(np.all(base.xi_tilde[i] == 0))


def times_seed(seed, jet):
    """The seed of apply or entries (a jet ``T + W``, one number per sample) times a jet ``T + W + (size,)``, as
    polynomials in ``s``, truncated."""
    return tprod(lambda c, x: c[..., None] * x, seed, jet)


class TestApplyAndEntries(unittest.TestCase):
    """What the tilde and tilde-dot variables ARE for apply and entries, whose seed is one number per sample at
    bond ``d``: zero where the side does not reach that bond, the seed times the untilded variable where it does.
    And which of the pairings of section 8 hold for which kind. The docstrings of the reference say these."""

    def check(self, got, expected, what):
        self.assertLessEqual(relative_error(got, expected), TOL, what)

    def test_they_are_zero_or_the_seed_times_the_untilded_variable(self):
        for labels, P, kind, order in cases(kinds=('apply', 'entries')):
            with self.subTest(**labels):
                v, d = Variables(P, kind, order, variations=None), P.d      # the raw dots
                seed = v.w_tilde
                for i in range(d + 1):
                    self.assertTrue(np.all(v.mu_tilde[i] == 0) and np.all(v.mu_tilde_dot[i] == 0))
                    self.check(v.nu_tilde[i], times_seed(seed, v.nu[i]), 'nu_tilde')
                    self.check(v.nu_L_tilde[i], times_seed(seed, v.nu_L[i]), 'nu_L_tilde')
                    self.check(v.nu_L_tilde_dot[i], times_seed(seed, v.nu_L_dot[i]), 'nu_L_tilde_dot')
                for i in range(d):
                    self.assertTrue(np.all(v.xi_tilde[i] == 0) and np.all(v.xi_tilde_dot[i] == 0))
                    self.assertEqual(v.xi_tilde[i].shape, (order + 1,) + P.W + (P.D[i].shape[0],))     # T, not A
                    self.check(v.eta_tilde[i], times_seed(seed, v.eta[i]), 'eta_tilde')
                    self.check(v.eta_D_tilde[i], times_seed(seed, v.eta_D[i]), 'eta_D_tilde')
                    self.check(v.eta_D_tilde_dot[i], times_seed(seed, v.eta_D_dot[i]), 'eta_D_tilde_dot')

    def test_their_blocks_have_only_the_terms_with_the_seed_on_the_right(self):
        for labels, P, kind, order in cases(kinds=('apply', 'entries')):
            with self.subTest(**labels):
                v, d = Variables(P, kind, order, variations=None), P.d
                V_tilde, H_tilde = sdef.g(kind, P.frame, v.sample, **v.args)
                V_tilde_dot, H_tilde_dot = sdef.Kp(kind, P.frame, v.sample, **v.args)
                for i in range(d):
                    self.check(V_tilde[i], V_top(v.eta_tilde[i], v.w[i]), 'V_tilde')
                    self.check(H_tilde[i], H_top(v.mu[i], v.xi[i], v.nu_tilde[i + 1]), 'H_tilde')
                    self.check(V_tilde_dot[i], V_top(v.eta_D_tilde_dot[i], v.w[i]), 'V_tilde_dot')
                    self.check(H_tilde_dot[i],
                               H_top(v.mu_dot[i], v.xi[i], v.nu_L_tilde[i + 1])
                               + H_top(v.mu[i], v.xi_dot[i], v.nu_L_tilde[i + 1])
                               + H_top(v.mu[i], v.xi[i], v.nu_L_tilde_dot[i + 1]), 'H_tilde_dot')

    def test_two_of_the_four_pairings_are_of_the_probe_kind_only(self):
        # Section 8 pairs mu_tilde with nu_L_dot and xi_tilde with eta_D_dot. For apply and entries those two
        # tilde variables are zero, and what multiplies nu_L_dot and eta_D_dot is the seed times mu and the seed
        # times xi, which have no names. (The other two pairings hold for every kind: TestPairings.)
        for labels, P, kind, order in cases(kinds=('apply', 'entries'), orders=(0, 3)):
            with self.subTest(**labels):
                d = P.d
                base = Variables(P, kind, order)
                seed = base.w_tilde
                zero_H = [0 * H for H in P.H]
                for i in range(1, d):                       # a bond with sites on both sides
                    right = ([V if j >= i else 0 * V for j, V in enumerate(P.V)],
                             [H if j >= i else 0 * H for j, H in enumerate(P.H)])
                    v = Variables(P, kind, order, variations=right, core_directions=None)
                    paired_with_Jp = top_pair(seed[..., None], v.Jp[..., None])
                    self.assertGreater(abs(paired_with_Jp), 1e-3)
                    self.assertEqual(top_pair(base.mu_tilde[i], v.nu_L_dot[i]), 0.0)
                    self.assertLessEqual(abs(top_pair(times_seed(seed, base.mu[i]), v.nu_L_dot[i]) - paired_with_Jp),
                                         TOL * abs(paired_with_Jp))
                for i in range(d):
                    trunk = ([V if j != i else 0 * V for j, V in enumerate(P.V)], P.H)       # everything but V[i]
                    v = Variables(P, kind, order, variations=trunk, core_directions=None)
                    paired_with_Jp = top_pair(seed[..., None], v.Jp[..., None])
                    self.assertGreater(abs(paired_with_Jp), 1e-3)
                    self.assertEqual(top_pair(base.xi_tilde[i], v.eta_D_dot[i]), 0.0)
                    self.assertLessEqual(abs(top_pair(times_seed(seed, base.xi[i]), v.eta_D_dot[i]) - paired_with_Jp),
                                         TOL * abs(paired_with_Jp))

    def test_a_tilde_dot_variable_is_the_derivative_of_its_tilde_with_the_seed_held(self):
        # For every kind (for apply and entries both sides of the first two are zero).
        h = 1e-4
        for labels, P, kind, order in cases(problems=PROBLEMS[:2], orders=(0, 3)):
            with self.subTest(**labels):
                sample, args = P.sample(kind, order), P.arguments(kind, order)
                seed = sdef.w_tilde(kind, P.frame, sample, **args)

                def moved(e):                               # the base point (D, L) moved along the raw direction
                    return ([X + e * X_dot for X, X_dot in zip(P.D, P.D_dot)], P.U,
                            [X + e * X_dot for X, X_dot in zip(P.L, P.L_dot)], P.R)
                for name in ('xi_tilde', 'mu_tilde', 'nu_L_tilde', 'eta_D_tilde'):
                    at = lambda e: getattr(sdef, name)(kind, moved(e), sample, order=order, w_tilde=seed)
                    quotient = [(8 * (a - b) - (c - d_)) / (12 * h)
                                for a, b, c, d_ in zip(at(h), at(-h), at(2 * h), at(-2 * h))]
                    dotted = getattr(sdef, name + '_dot')(kind, P.frame, sample, order=order, w_tilde=seed,
                                                          core_directions=args['core_directions'])
                    for a, b in zip(dotted, quotient):
                        self.assertLessEqual(relative_error(a, b), 1e-8, name + '_dot')


class TestJetsTwoWays(unittest.TestCase):
    """Every variable at an order, against the Taylor expansion of the same variable at order 0 along
    ``ww + s pp``, by interpolation at two sets of nodes that are not the reference's. For the tilde grades the
    order-0 call is given the stored seed as the polynomial ``sum_t w_tilde[t] s^t``: that is what it means
    that they are stored turned."""

    JETTED = tuple(name for name in ALL_NAMES if name in BOND_NAMES + SITE_NAMES + MODE_NAMES
                   and name not in ('V_tilde', 'H_tilde', 'V_tilde_dot', 'H_tilde_dot', 'r', 'w_tilde'))

    def test_every_variable(self):
        self.assertEqual(len(self.JETTED), 25)              # w, z, Jp and the 22 edge variables
        for labels, P, kind, order in cases(problems=(PROBLEMS[0], PROBLEMS[2]), orders=(1, 3)):
            sample, args = P.sample(kind, order), P.arguments(kind, order)
            seed = sdef.w_tilde(kind, P.frame, sample, **args)
            inputs = P.inputs(kind)
            line_kind = 'apply' if kind == 'entries' else kind      # off the grid, entries is apply at e + s p

            def at(name, s, i):
                powers = s ** np.arange(order + 1)
                seed_s = ([np.tensordot(powers, c, axes=1) for c in seed] if kind == 'probe'
                          else np.tensordot(powers, seed, axes=1))
                moved = [w + s * p for w, p in zip(inputs, P.pp)]
                return getattr(sdef, name)(line_kind, P.frame, moved, i, variations=args['variations'],
                                           core_directions=args['core_directions'], w_tilde=seed_s)

            for name in self.JETTED:
                got = getattr(sdef, name)(kind, P.frame, sample, **args)
                indexed = isinstance(got, tuple)            # not so for z and Jp of apply and entries
                got = list(got) if indexed else [got]
                n_slots = 2 if name in AFFINE_NAMES else order + 1
                for nodes in node_sets(P.d + order):        # no variable has a higher degree in s
                    for i, got_i in enumerate(got):
                        index = i if indexed else None
                        with self.subTest(name=name, i=index, nodes=float(nodes[-1]), **labels):
                            expected = interpolate(lambda s: at(name, s, index), nodes, n_slots)
                            self.assertEqual(got_i.shape, expected.shape)       # structural: A or T slots
                            self.assertLessEqual(relative_error(got_i, expected), 1e-8)

    def test_slot_zero_of_a_plain_or_dot_jet_is_the_variable_at_order_zero(self):
        # Not so for the tilde grades, whose seed is turned: its slot 0 holds the top order.
        P = small()
        for kind in KINDS:
            args = P.arguments(kind, 0)
            for name in NAMES['plain'] + NAMES['dot, frame form'] + NAMES['dot, raw form'] + ('w', 'z', 'Jp'):
                with self.subTest(kind=kind, name=name):
                    order0 = getattr(sdef, name)(kind, P.frame, P.sample(kind, 0), **dict(args, order=0))
                    order3 = getattr(sdef, name)(kind, P.frame, P.sample(kind, 3), **dict(args, order=3))
                    as_list = lambda value: list(value) if isinstance(value, tuple) else [value]
                    self.assertEqual(len(as_list(order3)), len(as_list(order0)))
                    for jet, plain in zip(as_list(order3), as_list(order0)):
                        self.assertLessEqual(relative_error(jet[0], plain), TOL)


def blocks_of(pair_of_tuples):
    return list(pair_of_tuples[0]) + list(pair_of_tuples[1])


def jax_in_double_precision():
    """A context in which jax computes in float64, whatever the process is set to; it changes no setting."""
    import jax
    enable_x64 = getattr(jax, 'enable_x64', None)               # jax >= 0.8
    if enable_x64 is None:
        from jax.experimental import enable_x64                 # older jax
    return enable_x64(True)


class TestKp(unittest.TestCase):
    """``Kp``: the second derivative of the dense objective ``<seed, z>`` in the cores of the base point."""

    def jax_problem(self, P, kind, order):
        """The arguments of a case as jax arrays: ``(frame, sample, arguments)``."""
        import jax.numpy as jnp
        to_jax = common.tree_to_jax
        args = P.arguments(kind, order)
        first = jnp.asarray(P.index) if kind == 'entries' else to_jax(P.ww)
        return (tuple(to_jax(store) for store in P.frame),
                first if order == 0 else (first, to_jax(P.pp)),
                dict(order=order, variations=(to_jax(P.V), to_jax(P.H)),
                     core_directions=(to_jax(P.D_dot), to_jax(P.L_dot)),
                     y=to_jax(args['y']) if kind == 'probe' else jnp.asarray(args['y']),
                     weight=jnp.asarray(args['weight'])))

    def test_jax_against_finite_differences(self):
        # The two routes of Kp: automatic differentiation of the dense objective (jax arrays) and the central
        # finite differences (numpy arrays). jax is put in double precision for the comparison, inside a
        # context, so that no setting of the process changes.
        if not common.jax_available:
            self.skipTest('jax is not installed: Kp is the finite differences alone')
        for labels, P, kind, order in cases():
            with self.subTest(**labels):
                sample, args = P.sample(kind, order), P.arguments(kind, order)
                S = sdef._setting(kind, P.frame, sample, order)
                seed = sdef.w_tilde(kind, P.frame, sample, **args)
                by_differences = sdef._Kp_by_finite_differences(S, seed, args['core_directions'])
                with jax_in_double_precision():
                    jax_frame, jax_sample, jax_args = self.jax_problem(P, kind, order)
                    S_jax = sdef._setting(kind, jax_frame, jax_sample, order)
                    seed_jax = sdef.w_tilde(kind, jax_frame, jax_sample, **jax_args)
                    by_jax = sdef._Kp_by_jax(S_jax, seed_jax, jax_args['core_directions'])
                    for a in blocks_of(by_jax):
                        self.assertTrue(common.is_jax_ndarray(a))
                        self.assertEqual(a.dtype, np.float64)
                    by_jax = [np.asarray(a) for a in blocks_of(by_jax)]
                for a, b in zip(by_jax, blocks_of(by_differences)):
                    self.assertIsInstance(b, np.ndarray)
                    self.assertLessEqual(relative_error(a, b), TOL)

    def test_the_route_of_Kp_is_read_off_the_arrays(self):
        # numpy arrays: the finite differences, whether or not jax is installed (the automatic differentiation
        # is made to raise here, and is not reached). A jax array among them: automatic differentiation.
        P, kind, order = small(), 'probe', 1
        sample, args = P.sample(kind, order), P.arguments(kind, order)
        S = sdef._setting(kind, P.frame, sample, order)
        seed = sdef.w_tilde(kind, P.frame, sample, **args)
        by_differences = sdef._Kp_by_finite_differences(S, seed, args['core_directions'])

        def not_this_route(*args, **kwargs):
            raise AssertionError('the other route of Kp was taken')
        with unittest.mock.patch.object(sdef, '_Kp_by_jax', not_this_route):
            for name, blocks in (('Kp', blocks_of(sdef.Kp(kind, P.frame, sample, **args))),
                                 ('V_tilde_dot', list(sdef.V_tilde_dot(kind, P.frame, sample, **args))),
                                 ('H_tilde_dot', list(sdef.H_tilde_dot(kind, P.frame, sample, **args)))):
                expected = {'Kp': blocks_of(by_differences), 'V_tilde_dot': by_differences[0],
                            'H_tilde_dot': by_differences[1]}[name]
                for a, b in zip(blocks, expected):
                    self.assertIsInstance(a, np.ndarray)
                    self.assertTrue(np.array_equal(a, b))
        if not common.jax_available:
            return
        jax_frame, jax_sample, jax_args = self.jax_problem(P, kind, order)
        with unittest.mock.patch.object(sdef, '_Kp_by_finite_differences', not_this_route):
            all_jax = blocks_of(sdef.Kp(kind, jax_frame, jax_sample, **jax_args))
            mixed = dict(args, core_directions=jax_args['core_directions'])
            one_jax = blocks_of(sdef.Kp(kind, P.frame, sample, **mixed))
        for a, b, c in zip(all_jax, one_jax, blocks_of(by_differences)):
            self.assertTrue(common.is_jax_ndarray(a) and common.is_jax_ndarray(b))
            self.assertLessEqual(relative_error(np.asarray(a), c), 2e-3)        # float32, unless x64 is on
            self.assertLessEqual(relative_error(np.asarray(b), c), 2e-3)

    def test_Kp_is_the_derivative_of_the_corewise_gradient_with_the_seed_held(self):
        # g on the corewise frame (B, G, G, G) with a FIXED seed is the gradient of <seed, z(B, G)>; Kp is its
        # derivative along (D_dot, L_dot). g is pinned by the dense Jacobian (TestOutputsAgainstDense).
        h = 1e-4
        for labels, P, kind, order in cases(problems=PROBLEMS[:2], orders=(0, 3)):
            with self.subTest(**labels):
                sample, args = P.sample(kind, order), P.arguments(kind, order)
                seed = sdef.w_tilde(kind, P.frame, sample, **args)

                def gradient(e):
                    B = [X + e * X_dot for X, X_dot in zip(P.D, P.D_dot)]
                    G = [X + e * X_dot for X, X_dot in zip(P.L, P.L_dot)]
                    return blocks_of(sdef.g(kind, (B, G, G, G), sample, order=order, w_tilde=seed))
                quotient = [(8 * (a - b) - (c - d)) / (12 * h)             # a fourth-order central difference
                            for a, b, c, d in zip(gradient(h), gradient(-h), gradient(2 * h), gradient(-2 * h))]
                Kp = blocks_of(sdef.Kp(kind, P.frame, sample, **args))
                self.assertEqual([b.shape for b in Kp], [X.shape for X in P.D + P.L])       # the shapes of D, L
                for a, b in zip(Kp, quotient):
                    self.assertLessEqual(relative_error(a, b), 1e-8)

    def test_Kp_is_the_second_derivative_of_the_dense_objective(self):
        # By a route with no part of the reference in it but the seed: phi(B, G) = <weight^2 r, z(B, G)> with
        # the plain inner product, z from t3_to_dense sampled by the tests' own route; <q, Kp> is the mixed
        # second derivative of phi along q and along the direction.
        nodes = np.linspace(-1.0, 1.0, 2 * small().d + 1)
        for labels, P, kind, order in cases(problems=PROBLEMS[:1], orders=(0, 3)):
            with self.subTest(**labels):
                sample, args = P.sample(kind, order), P.arguments(kind, order)
                r = lift(modes_of(kind, sdef.r(kind, P.frame, sample, **args)), order)
                functional = weighted(r, args['weight'], kind, order)
                rng = np.random.default_rng(5)
                q = [rng.standard_normal(X.shape) for X in P.D + P.L]
                p = P.D_dot + P.L_dot
                x = P.D + P.L

                def phi(a, b):
                    cores = [X + a * Q + b * X_dot for X, Q, X_dot in zip(x, q, p)]
                    T = t3_conversions.t3_to_dense((cores[:P.d], cores[P.d:]))
                    z = lift(modes_of(kind, sample_dense_jet(P, kind, T, order)), order)
                    return sum(np.sum(c * z_i) for c, z_i in zip(functional, z))
                # phi is a polynomial of degree at most 2 d along each of the two: its mixed derivative at 0,
                # exactly, by interpolation in the one and then in the other
                slope = lambda f: interpolate(f, nodes, 2)[1]
                mixed = slope(lambda a: slope(lambda b: phi(a, b)))
                Kp = blocks_of(sdef.Kp(kind, P.frame, sample, **args))
                paired = sum(np.sum(a * b) for a, b in zip(q, Kp))
                self.assertLessEqual(abs(paired - mixed), 1e-9 * max(abs(mixed), 1.0))

    def test_Kp_is_symmetric_and_linear_in_the_direction(self):
        P, kind, order = small(), 'probe', 1
        sample, args = P.sample(kind, order), P.arguments(kind, order)
        rng = np.random.default_rng(6)
        a = ([rng.standard_normal(X.shape) for X in P.D], [rng.standard_normal(X.shape) for X in P.L])
        b = args['core_directions']
        K = lambda direction: blocks_of(sdef.Kp(kind, P.frame, sample, **dict(args, core_directions=direction)))
        dot = lambda u, v: sum(np.sum(x * y) for x, y in zip(u, v))
        Ka, Kb = K(a), K(b)
        self.assertLessEqual(abs(dot(blocks_of(a), Kb) - dot(blocks_of(b), Ka)), TOL * abs(dot(blocks_of(a), Kb)))
        both = ([x + 2 * y for x, y in zip(a[0], b[0])], [x + 2 * y for x, y in zip(a[1], b[1])])
        for got, x, y in zip(K(both), Ka, Kb):
            self.assertLessEqual(relative_error(got, x + 2 * y), TOL)


class TestCorewiseFrame(unittest.TestCase):
    """SAMPLE_VARIABLES.md section 9: the corewise frame ``(B, G, G, G)`` has one basis on each edge."""

    def test_the_two_bases_are_one(self):
        P = small()
        corewise = (P.D, P.L, P.L, P.L)
        raw = (P.D_dot, P.L_dot)
        for kind in KINDS:
            for order in (0, 3):
                sample, args = P.sample(kind, order), P.arguments(kind, order)
                get = lambda name, **more: getattr(sdef, name)(kind, corewise, sample, **dict(args, **more))
                pairs = [('nu', 'nu_L', {}), ('eta', 'eta_D', {}), ('nu_tilde', 'nu_L_tilde', {}),
                         ('eta_tilde', 'eta_D_tilde', {}),
                         # a direction at a corewise point is a raw direction: the frame's dots are the raw ones
                         ('mu_R_dot', 'mu_dot', dict(variations=raw)), ('xi_U_dot', 'xi_dot', dict(variations=raw))]
                for one, other, more in pairs:
                    with self.subTest(kind=kind, order=order, one=one, other=other):
                        for a, b in zip(get(one, **more), get(other, **more)):
                            self.assertLessEqual(relative_error(a, b), TOL)
                for name in ('nu_L_dot', 'eta_D_dot'):      # one name, two forms, the same array here
                    with self.subTest(kind=kind, order=order, name=name):
                        frame_form = get(name, variations=raw, core_directions=None)
                        raw_form = get(name, variations=None, core_directions=raw)
                        for a, b in zip(frame_form, raw_form):
                            self.assertLessEqual(relative_error(a, b), TOL)

    def test_with_both_directions_the_shared_names_are_the_raw_form(self):
        # On a manifold frame the two forms differ; with both arguments given the name means the raw form
        # (the one Kp reads).
        P, kind = small(), 'probe'
        sample, args = P.sample(kind, 1), P.arguments(kind, 1)
        for name in ('nu_L_dot', 'eta_D_dot'):
            both = getattr(sdef, name)(kind, P.frame, sample, **args)
            raw_only = getattr(sdef, name)(kind, P.frame, sample, **dict(args, variations=None))
            frame_only = getattr(sdef, name)(kind, P.frame, sample, **dict(args, core_directions=None))
            for a, b, c in zip(both[:P.d], raw_only[:P.d], frame_only[:P.d]):
                self.assertTrue(np.array_equal(a, b))
                self.assertGreater(relative_error(a, c), 1e-2)


ALLOWED_MODULES = {'probing': ('t3toolbox.backend.probing', 'dense_probe'),      # name in the module: (module,
                   'apply': ('t3toolbox.backend.apply', 'dense_apply'),          #   the one attribute of it that
                   'entries': ('t3toolbox.backend.entries', 'dense_entries')}    #   the reference may touch)
ALLOWED_OUTSIDE = {'math', 'typing', 'dataclasses', 'numpy', 'jax'}              # everything else it imports
ALLOWED_FROM_COMMON = {'NDArray', 'jax_available', 'get_backend', 'tree_contains_jax', 'tree_to_jax'}
FORBIDDEN_NAMES = {'contract', 'jet_contract', 'contractions', 'safety', 'xscan', 'xmap', 't3toolbox',
                   # no dynamic road to a module's contents either
                   'getattr', 'vars', 'globals', 'locals', 'eval', 'exec', 'compile', '__import__', 'importlib',
                   'sys', '__dict__', '__getattribute__', 'modules'}


def independence_violations(source):
    """What a source of the reference module reaches of the library beyond what it may: a list of messages, empty
    if it is independent. It may: ``from t3toolbox.backend.common import *`` and use the names in
    `ALLOWED_FROM_COMMON`; import the three sampling modules under their own names and touch, of each, the one
    dense function, as ``probing.dense_probe``. Nothing else of the library, by any road: no other import of it,
    no name imported from a module, no other attribute, no module passed around or looked into by name."""
    tree = ast.parse(source)
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split('.')[0] != 't3toolbox':
                    if alias.name.split('.')[0] not in ALLOWED_OUTSIDE:
                        found.append('import %s' % alias.name)
                    continue
                allowed = {module: name for name, (module, _) in ALLOWED_MODULES.items()}
                if allowed.get(alias.name) != alias.asname:
                    found.append('import %s as %s' % (alias.name, alias.asname))
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                found.append('a relative import')
            elif (node.module or '').split('.')[0] == 't3toolbox':
                if node.module != 't3toolbox.backend.common' or [alias.name for alias in node.names] != ['*']:
                    found.append('from %s import %s' % (node.module, ', '.join(alias.name for alias in node.names)))
            elif (node.module or '').split('.')[0] not in ALLOWED_OUTSIDE:
                found.append('from %s import ...' % node.module)

    # a module's name occurs only as `module.its_dense_function`
    allowed_uses = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id in ALLOWED_MODULES and node.attr == ALLOWED_MODULES[node.value.id][1]):
            allowed_uses.add(id(node.value))
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in ALLOWED_MODULES and id(node) not in allowed_uses:
            found.append('the module %s used other than as %s.%s (line %d)'
                         % (node.id, node.id, ALLOWED_MODULES[node.id][1], node.lineno))

    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    for name in sorted((names | attributes) & FORBIDDEN_NAMES):
        found.append('the name %s' % name)
    for name in sorted((names & set(common.__all__)) - ALLOWED_FROM_COMMON):
        found.append('%s, from common' % name)
    # get_backend hands out a map and a scan: the reference takes its array module from it and nothing else
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name) and node.value.func.id == 'get_backend'):
            targets = node.targets[0]
            taken = [t.id for t in targets.elts] if isinstance(targets, ast.Tuple) else None
            if taken is None or taken[1:] != ['_', '_']:
                found.append('get_backend unpacked as %s (line %d)' % (taken, node.lineno))
    n_calls = sum(1 for node in ast.walk(tree) if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Name) and node.func.id == 'get_backend')
    n_assigned = sum(1 for node in ast.walk(tree) if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                     and isinstance(node.value.func, ast.Name) and node.value.func.id == 'get_backend')
    if n_calls != n_assigned:
        found.append('get_backend called other than in `xnp, _, _ = get_backend(...)`')
    return found


class TestStructure(unittest.TestCase):
    """Exact: the names, the one signature, the shapes, the errors, the doctests, and what the module calls."""

    def test_the_names_are_those_of_the_plan(self):
        self.assertEqual(tuple(sdef.__all__), ALL_NAMES)
        self.assertEqual(len(ALL_NAMES), 37)
        self.assertEqual(set(BOND_NAMES + SITE_NAMES + MODE_NAMES + UNINDEXED_NAMES), set(ALL_NAMES))
        for name in ALL_NAMES:
            self.assertIs(getattr(sdef, name).__module__, sdef.__name__)

    def test_every_function_has_the_same_arguments(self):
        expected = [('kind', inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
                    ('frame', inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
                    ('sample', inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
                    ('i', inspect.Parameter.POSITIONAL_OR_KEYWORD, None),
                    ('order', inspect.Parameter.POSITIONAL_OR_KEYWORD, 0),
                    ('variations', inspect.Parameter.KEYWORD_ONLY, None),
                    ('core_directions', inspect.Parameter.KEYWORD_ONLY, None),
                    ('y', inspect.Parameter.KEYWORD_ONLY, None),
                    ('weight', inspect.Parameter.KEYWORD_ONLY, None),
                    ('w_tilde', inspect.Parameter.KEYWORD_ONLY, None)]
        for name in ALL_NAMES:
            with self.subTest(name=name):
                parameters = inspect.signature(getattr(sdef, name)).parameters.values()
                self.assertEqual([(p.name, p.kind, p.default) for p in parameters], expected)

    def test_every_function_has_a_doctest(self):
        for name in ALL_NAMES:
            with self.subTest(name=name):
                doc = getattr(sdef, name).__doc__
                self.assertRegex(doc, r">>> .*\b%s\b" % name)        # an example, and it is about this name

    def test_the_module_calls_nothing_of_the_library_but_the_dense_functions(self):
        # The reference must share no code with what it will be used to test: no sweep, no kernel, no
        # jet_contract, no contract.
        self.assertEqual(independence_violations(inspect.getsource(sdef)), [])

    def test_the_guard_of_independence_sees_every_way_around_it(self):
        # The guard above, turned on the module's own source with one line added: each line reaches library code
        # the reference must not use, by a different road, and each is seen.
        source = inspect.getsource(sdef)
        roads = (
            "_old = probing.t3_probe",                                              # another attribute
            "from t3toolbox.backend.probing import t3_probe as _old",               # a name imported FROM it
            "from t3toolbox.backend.probing import dense_probe",                    # even an allowed function
            "_old = getattr(probing, 't3_' + 'probe')",                             # getattr on the module
            "_old = getattr(probing, 'dense_probe')",
            "_module = probing",                                                    # the module under another name
            "_old = probing.__dict__['t3_probe']",
            "_old = vars(apply)['t3_apply']",
            "import t3toolbox.backend.contractions as _c",                          # another library module
            "import t3toolbox.backend.probing as _p",                               # an allowed one, renamed
            "import t3toolbox.backend.probing",                                     # ... or not renamed at all
            "from t3toolbox.backend import probing as _p",
            "from t3toolbox.backend.common import jax_scan as _scan",               # a named import from common
            "from . import probing as _p",                                          # a relative import
            "_scan = ragged_scan",                                                  # a scan out of common's star
            "_, _map, _ = get_backend(True, False)",                                # a map out of get_backend
            "_xnp, xmap, xscan = get_backend(True, False)",
            "_old = __import__('t3toolbox.backend.probing')",
            "import importlib",
            "from importlib import import_module",
            "_old = sys.modules['t3toolbox.backend.probing']",
            "_old = globals()['probing']",
            "_old = eval('probing.t3_probe')",
        )
        for line in roads:
            with self.subTest(line=line):
                self.assertNotEqual(independence_violations(source + '\n' + line + '\n'), [])

    def sizes(self, P):
        """The size of the open edge of each indexed edge variable, by name: a function of the index."""
        D, U, L, R = P.frame
        d = P.d
        rL = lambda i: L[i].shape[0] if i < d else L[d - 1].shape[2]
        rR = lambda i: R[i].shape[0] if i < d else R[d - 1].shape[2]
        nD = lambda i: D[i].shape[0]
        nU = lambda i: U[i].shape[1]
        N = lambda i: D[i].shape[1]
        return {'w': N, 'xi': nD, 'mu': rL, 'nu': rR, 'eta': nU, 'nu_L': rL, 'eta_D': nD,
                'xi_U_dot': nU, 'mu_R_dot': rR, 'nu_L_dot': rL, 'eta_D_dot': nD, 'xi_dot': nD, 'mu_dot': rL,
                'xi_tilde': nD, 'mu_tilde': rL, 'nu_tilde': rR, 'eta_tilde': nU, 'nu_L_tilde': rL,
                'eta_D_tilde': nD, 'xi_tilde_dot': nD, 'mu_tilde_dot': rL, 'nu_L_tilde_dot': rL,
                'eta_D_tilde_dot': nD}

    def test_the_shapes_are_those_of_the_table(self):
        # SAMPLE_VARIABLES.md section 7: T + W + (size,), with A in place of T for the affine variables; one
        # array per bond (d + 1) or per site (d); the blocks have the shapes of the cores they stand for.
        for labels, P, kind, order in cases(problems=PROBLEMS[:2], orders=(0, 1, 3)):
            sample, args = P.sample(kind, order), P.arguments(kind, order)
            T = (order + 1,) if order else ()
            A = (2,) if order else ()
            d = P.d
            for name, size in self.sizes(P).items():
                with self.subTest(name=name, **labels):
                    values = getattr(sdef, name)(kind, P.frame, sample, **args)
                    self.assertIsInstance(values, tuple)
                    self.assertEqual(len(values), d + 1 if name in BOND_NAMES else d)
                    group = A if name in AFFINE_NAMES else T
                    self.assertEqual([v.shape for v in values], [group + P.W + (size(i),) for i in range(len(values))])
                    for v in values:
                        self.assertIsInstance(v, np.ndarray)
                        self.assertEqual(v.dtype, np.float64)
            with self.subTest(what='outputs and blocks', **labels):
                for name in ('z', 'r', 'Jp', 'w_tilde'):
                    value = getattr(sdef, name)(kind, P.frame, sample, **args)
                    if kind == 'probe':
                        self.assertEqual([v.shape for v in value], [T + P.W + (n,) for n in P.shape])
                    else:
                        self.assertEqual(value.shape, T + P.W)
                V_shapes = [V.shape for V in P.V]
                H_shapes = [H.shape for H in P.H]
                for name in ('g', 'JtJp'):
                    V_blocks, H_blocks = getattr(sdef, name)(kind, P.frame, sample, **args)
                    self.assertEqual(([b.shape for b in V_blocks], [b.shape for b in H_blocks]), (V_shapes, H_shapes))
                V_blocks, H_blocks = sdef.Kp(kind, P.frame, sample, **args)
                self.assertEqual(([b.shape for b in V_blocks], [b.shape for b in H_blocks]),
                                 ([B.shape for B in P.D], [G.shape for G in P.L]))
                self.assertEqual([b.shape for b in sdef.V_tilde(kind, P.frame, sample, **args)], V_shapes)
                self.assertEqual([b.shape for b in sdef.H_tilde(kind, P.frame, sample, **args)], H_shapes)

    def test_one_index_is_the_entry_of_the_whole(self):
        P = small()
        for kind in KINDS:
            sample, args = P.sample(kind, 1), P.arguments(kind, 1)
            for name in BOND_NAMES + SITE_NAMES + (MODE_NAMES if kind == 'probe' else ()):
                with self.subTest(kind=kind, name=name):
                    whole = getattr(sdef, name)(kind, P.frame, sample, **args)
                    for i, value in enumerate(whole):
                        self.assertTrue(np.array_equal(getattr(sdef, name)(kind, P.frame, sample, i, **args), value))
                    with self.assertRaises(ValueError):                         # one past the last index
                        getattr(sdef, name)(kind, P.frame, sample, len(whole), **args)
                    with self.assertRaises(ValueError):
                        getattr(sdef, name)(kind, P.frame, sample, -1, **args)

    def test_an_index_where_there_is_none_raises(self):
        P = small()
        for kind in KINDS:
            sample, args = P.sample(kind, 0), P.arguments(kind, 0)
            for name in UNINDEXED_NAMES + (MODE_NAMES if kind != 'probe' else ()):
                with self.subTest(kind=kind, name=name):
                    with self.assertRaises(ValueError):
                        getattr(sdef, name)(kind, P.frame, sample, 0, **args)

    def test_a_missing_argument_raises(self):
        P, kind = small(), 'probe'
        sample, args = P.sample(kind, 0), P.arguments(kind, 0)
        needs_variations = ('xi_U_dot', 'mu_R_dot', 'Jp', 'JtJp', 'Jp_norm_sq')
        needs_core_directions = ('xi_dot', 'mu_dot', 'xi_tilde_dot', 'mu_tilde_dot', 'nu_L_tilde_dot',
                                 'eta_D_tilde_dot', 'Kp', 'V_tilde_dot', 'H_tilde_dot')
        needs_a_seed = NAMES['tilde'] + NAMES['tilde-dot'] + ('w_tilde', 'g', 'Kp', 'V_tilde', 'H_tilde',
                                                               'V_tilde_dot', 'H_tilde_dot')
        needs_y = ('r', 'f', 'r_block_sumsq')
        for names, without in ((needs_variations, dict(variations=None)),
                               (needs_core_directions, dict(core_directions=None)),
                               (('nu_L_dot', 'eta_D_dot'), dict(variations=None, core_directions=None)),
                               (needs_a_seed, dict(y=None, w_tilde=None)),
                               (needs_y, dict(y=None))):
            for name in names:
                with self.subTest(name=name, without=sorted(without)):
                    with self.assertRaises(ValueError):
                        getattr(sdef, name)(kind, P.frame, sample, **dict(args, **without))
        # and nothing else is needed: every function runs with only what it reads
        reads = {name: set() for name in ALL_NAMES}
        for name in needs_variations + ('nu_L_dot', 'eta_D_dot'):
            reads[name].add('variations')
        for name in needs_core_directions:
            reads[name].add('core_directions')
        for name in needs_a_seed + needs_y:
            reads[name].add('y')
        for name in ALL_NAMES:
            with self.subTest(name=name, what='runs with only what it reads'):
                getattr(sdef, name)(kind, P.frame, sample, **{key: args[key] for key in reads[name]})

    def test_what_cannot_be_run_raises(self):
        P = small()
        with self.assertRaises(ValueError):
            sdef.mu('probes', P.frame, P.ww)                                # an unknown kind
        with self.assertRaises(ValueError):
            sdef.mu('probe', (P.D, P.L), P.ww)                              # a corewise point is (B, G, G, G)
        many = 17
        frame = tuple([np.ones((1, 2))] * many if k == 0 else [np.ones((1, 1, 1))] * many for k in range(4))
        with self.assertRaises(ValueError):
            sdef.mu('probe', frame, [np.ones((1, 2))] * many)               # more sites than einsum letters

    def test_sixteen_sites_run(self):
        d = 16
        rng = np.random.default_rng(0)
        D = [rng.standard_normal((1, 2)) for _ in range(d)]
        G = [rng.standard_normal((1, 1, 1)) for _ in range(d)]
        ww = [rng.standard_normal((2, 2)) for _ in range(d)]
        expected = np.prod([G[i][0, 0, 0] * (ww[i] @ D[i][0]) for i in range(d)], axis=0)
        self.assertLessEqual(relative_error(sdef.mu('probe', (D, G, G, G), ww, d)[:, 0], expected), TOL)

    def test_the_input_is_returned_exactly(self):
        P = small()
        for kind in KINDS:
            w = sdef.w(kind, P.frame, P.sample(kind, 3), order=3)
            for w_i, x_i, p_i in zip(w, P.inputs(kind), P.pp):
                self.assertEqual(w_i.shape, (2,) + P.W + x_i.shape[-1:])
                self.assertTrue(np.array_equal(w_i[0], x_i) and np.array_equal(w_i[1], p_i))

    def test_the_plain_and_dot_variables_do_not_depend_on_the_kind(self):
        # probe and apply take the same inputs and differ only in the outputs and in where the seed goes.
        P = small()
        for order in (0, 3):
            args = P.arguments('probe', order)
            for name in NAMES['plain'] + NAMES['dot, frame form'] + NAMES['dot, raw form'] + ('w',):
                with self.subTest(name=name, order=order):
                    for a, b in zip(getattr(sdef, name)('probe', P.frame, P.sample('probe', order), **args),
                                    getattr(sdef, name)('apply', P.frame, P.sample('apply', order), **args)):
                        self.assertTrue(np.array_equal(a, b))

    def test_entries_is_apply_at_unit_vectors(self):
        P = small()
        for order in (0, 3):
            args = P.arguments('entries', order)
            unit = P.inputs('entries')
            apply_sample = unit if order == 0 else (unit, P.pp)
            for name in ALL_NAMES:
                with self.subTest(name=name, order=order):
                    a = getattr(sdef, name)('entries', P.frame, P.sample('entries', order), **args)
                    b = getattr(sdef, name)('apply', P.frame, apply_sample, **args)
                    flatten = lambda value: (sum((flatten(v) for v in value), []) if isinstance(value, tuple)
                                             else [np.asarray(value)])
                    for x, y in zip(flatten(a), flatten(b)):
                        self.assertLessEqual(relative_error(x, y), TOL)


class TestJax(unittest.TestCase):
    """Dispatch is inferred from the arrays: jax in, jax out, with the numbers of numpy (to the precision jax
    is set to: float32, unless x64 is on, which another test module of the same process may have done)."""

    def tearDown(self):
        if common.jax_available:
            import jax
            jax.clear_caches()              # every eager jax einsum of a new shape is a compiled program

    def test_jax_in_gives_jax_out(self):
        if not common.jax_available:
            self.skipTest('jax is not installed')
        P = small()
        flatten = lambda value: (sum((flatten(v) for v in value), []) if isinstance(value, (tuple, list))
                                 else [value])
        for kind, order in (('probe', 1), ('apply', 0), ('entries', 1)):
            sample, args = P.sample(kind, order), P.arguments(kind, order)
            jax_frame, jax_sample, jax_args = TestKp.jax_problem(None, P, kind, order)
            for name in ALL_NAMES:
                with self.subTest(kind=kind, order=order, name=name):
                    got = flatten(getattr(sdef, name)(kind, jax_frame, jax_sample, **jax_args))
                    expected = flatten(getattr(sdef, name)(kind, P.frame, sample, **args))
                    self.assertEqual(len(got), len(expected))
                    for a, b in zip(got, expected):
                        self.assertTrue(common.is_jax_ndarray(a), type(a))
                        self.assertEqual(tuple(a.shape), tuple(np.shape(b)))
                        self.assertLessEqual(relative_error(np.asarray(a), b), 2e-3)


class TestDenseFunctions(unittest.TestCase):
    """``dense_probe``, ``dense_apply``, ``dense_entries``: the dense ground truth of the three kinds."""

    def setUp(self):
        rng = np.random.default_rng(0)
        self.rng = rng
        self.shape = (4, 3, 5)
        self.T = rng.standard_normal(self.shape)
        self.TC = rng.standard_normal((2, 3) + self.shape)                  # a stack C = (2, 3) on the tensor
        self.ww = [rng.standard_normal((2, 2, n)) for n in self.shape]      # W = (2, 2)
        self.pp = [rng.standard_normal((2, 2, n)) for n in self.shape]
        self.index = np.stack([rng.integers(0, n, size=(2, 2)) for n in self.shape])

    def check(self, got, expected, tol=1e-12):
        self.assertLessEqual(relative_error(got, expected), tol)

    def test_order_0_dense_probe_is_the_tensor_contracted_with_all_vectors_but_one(self):
        T, TC, (u, v, w) = self.T, self.TC, self.ww
        z = probing.dense_probe(T, self.ww)
        self.assertIsInstance(z, tuple)
        self.check(z[0], np.einsum('ijk,abj,abk->abi', T, v, w))
        self.check(z[1], np.einsum('ijk,abi,abk->abj', T, u, w))
        self.check(z[2], np.einsum('ijk,abi,abj->abk', T, u, v))
        z = probing.dense_probe(TC, self.ww)                                # W outer, C inner
        self.check(z[0], np.einsum('xyijk,abj,abk->abxyi', TC, v, w))
        self.check(z[1], np.einsum('xyijk,abi,abk->abxyj', TC, u, w))
        self.check(z[2], np.einsum('xyijk,abi,abj->abxyk', TC, u, v))
        z = probing.dense_probe(T, [x[0, 0] for x in self.ww])              # no sample stack
        self.check(z[1], np.einsum('ijk,i,k->j', T, u[0, 0], w[0, 0]))

    def test_dense_probe_with_one_mode(self):
        # No vector is contracted: the probe is the tensor itself, broadcast over W (review 2026-08-22, C1).
        T = self.rng.standard_normal((2, 6))                                # C = (2,), N0 = 6
        ww, pp = [self.rng.standard_normal((3, 6))], [self.rng.standard_normal((3, 6))]
        (z,) = probing.dense_probe(T, ww)
        self.assertEqual(z.shape, (3, 2, 6))
        self.assertTrue(np.array_equal(z, np.broadcast_to(T, (3, 2, 6))))
        (z,) = probing.dense_probe(T, ww, pp, 2)                            # degree 0 in s: zero above slot 0
        self.assertEqual(z.shape, (3, 3, 2, 6))
        self.assertTrue(np.array_equal(z[0], np.broadcast_to(T, (3, 2, 6))) and np.all(z[1:] == 0))

    def test_order_0_dense_apply_is_the_probe_contracted_with_the_vector_of_its_mode(self):
        for T, C in ((self.T, ''), (self.TC, 'xy')):
            z = apply.dense_apply(T, self.ww)
            self.assertEqual(z.shape, (2, 2) + T.shape[:-3])                # W + C
            self.check(z, np.einsum(C + 'ijk,abi,abj,abk->ab' + C, T, *self.ww))
            for i, probe in enumerate(probing.dense_probe(T, self.ww)):
                self.check(z, np.einsum('ab...o,abo->ab...', probe, self.ww[i]))

    def test_order_0_dense_entries_is_indexing(self):
        i0, i1, i2 = self.index
        z = entries.dense_entries(self.T, self.index)
        self.assertEqual(z.shape, (2, 2))
        self.assertTrue(np.array_equal(z, self.T[i0, i1, i2]))             # one-hot contraction: exact
        z = entries.dense_entries(self.TC, self.index)
        self.assertEqual(z.shape, (2, 2, 2, 3))                             # W + C
        self.check(z, np.moveaxis(self.TC[:, :, i0, i1, i2], (0, 1), (2, 3)))
        # numpy's index semantics: a negative index wraps
        self.assertTrue(np.array_equal(entries.dense_entries(self.T, self.index - np.array(self.shape)[:, None, None]),
                                       self.T[i0, i1, i2]))
        self.check(entries.dense_entries(self.T, self.index.tolist()), self.T[i0, i1, i2])    # a nested list
        # indexing, with no arithmetic: an entry does not see the others. A non-finite entry elsewhere leaves
        # it as it is (a contraction with unit vectors would make it nan, 0 times inf).
        T = self.T.copy()
        T[(i0[0, 0] + 1) % 4, i1[0, 0], i2[0, 0]] = np.inf
        T[i0[0, 0], (i1[0, 0] + 1) % 3, i2[0, 0]] = np.nan
        expected = T[i0, i1, i2]
        self.assertTrue(np.isfinite(expected[0, 0]))
        self.assertTrue(np.array_equal(entries.dense_entries(T, self.index), expected, equal_nan=True))
        self.assertTrue(np.array_equal(entries.dense_entries(self.T, self.index[:, 0, 0]),         # W = ()
                                       self.T[i0[0, 0], i1[0, 0], i2[0, 0]]))

    def functions(self):
        """(name, the function of (T, inputs, pp, order), its inputs, the degree of its output in s)."""
        unit = [np.eye(n)[self.index[i]] for i, n in enumerate(self.shape)]
        return (('dense_probe', probing.dense_probe, self.ww, 2),
                ('dense_apply', apply.dense_apply, self.ww, 3),
                ('dense_entries', lambda T, index, pp=None, order=0: entries.dense_entries(T, index, pp, order),
                 self.index, 3)), unit

    def test_jets_are_the_interpolation_of_order_0_calls(self):
        # The coefficients come from the multilinear expansion; here they are recovered from order-0 calls along
        # ww + s pp, at two sets of nodes. With the W and C stacks, the order axis leading: T + W + C + ...
        functions, unit = self.functions()
        for name, function, first, degree in functions:
            # the same function at order 0 along the line; entries leaves the grid, where it is apply at e + s p
            along, start = (apply.dense_apply, unit) if name == 'dense_entries' else (function, first)
            for T in (self.T, self.TC):
                C = T.shape[:-3]
                for order in (1, 2, 3, 5):
                    got = function(T, first, self.pp, order)
                    for nodes in node_sets(max(degree, order)):
                        with self.subTest(name=name, C=C, order=order, nodes=float(nodes[-1])):
                            at = lambda s: along(T, [w + s * p for w, p in zip(start, self.pp)])
                            if name == 'dense_probe':
                                for i in range(3):
                                    self.assertEqual(got[i].shape, (order + 1, 2, 2) + C + self.shape[i:i + 1])
                                    self.check(got[i], interpolate(lambda s: at(s)[i], nodes, order + 1), 1e-9)
                            else:
                                self.assertEqual(got.shape, (order + 1, 2, 2) + C)
                                self.check(got, interpolate(at, nodes, order + 1), 1e-9)

    def test_slot_0_is_the_order_0_call_and_the_slots_above_the_degree_are_zero(self):
        functions, _ = self.functions()
        for name, function, first, degree in functions:
            with self.subTest(name=name):
                jet = function(self.TC, first, self.pp, degree + 2)
                plain = function(self.TC, first)
                for jet_i, plain_i in zip(modes_of('probe', jet) if name == 'dense_probe' else [jet],
                                          modes_of('probe', plain) if name == 'dense_probe' else [plain]):
                    self.assertTrue(np.array_equal(jet_i[0], plain_i))
                    self.assertGreater(np.abs(jet_i[degree]).min(), 0)
                    self.assertTrue(np.all(jet_i[degree + 1:] == 0))        # exactly: no term is formed there

    def test_the_coefficients_are_taylor_coefficients_not_derivatives(self):
        # Slot 2 of the apply along w + s p: the sum over the PAIRS of modes that take p, each pair once (the
        # second derivative is twice that).
        T, (u, v, w), (p, q, r) = self.T, self.ww, self.pp
        slot2 = (np.einsum('ijk,abi,abj,abk->ab', T, p, q, w) + np.einsum('ijk,abi,abj,abk->ab', T, p, v, r)
                 + np.einsum('ijk,abi,abj,abk->ab', T, u, q, r))
        self.check(apply.dense_apply(T, self.ww, self.pp, 2)[2], slot2)

    def test_pp_goes_with_order(self):
        functions, _ = self.functions()
        for name, function, first, _ in functions:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, name):
                    function(self.T, first, self.pp)                        # pp without an order
                with self.assertRaisesRegex(ValueError, name):
                    function(self.T, first, None, 2)                        # an order without pp
                with self.assertRaisesRegex(ValueError, name + ': order'):
                    function(self.T, first, self.pp, -1)                    # a negative order, with pp
                with self.assertRaisesRegex(ValueError, name + ': order'):
                    function(self.T, first, None, -1)                       # ... and without

    def test_wrong_shapes_raise(self):
        short = [x[..., :-1] for x in self.ww]
        other_stack = [x[0] for x in self.ww]
        mixed = [self.ww[0][0]] + self.ww[1:]
        with self.assertRaisesRegex(ValueError, 'dense_probe'):
            probing.dense_probe(self.T, short)                              # a wrong mode size
        with self.assertRaisesRegex(ValueError, 'dense_apply'):
            apply.dense_apply(self.T, mixed)                                # sample stacks that differ
        with self.assertRaisesRegex(ValueError, 'dense_apply'):
            apply.dense_apply(self.T, self.ww, other_stack, 1)              # pp unlike ww
        with self.assertRaisesRegex(ValueError, 'dense_probe'):
            probing.dense_probe(self.T, self.ww, self.pp[:2], 1)            # too few pp
        with self.assertRaisesRegex(ValueError, 'dense_entries'):
            entries.dense_entries(self.T, self.index, other_stack, 1)       # pp unlike index
        with self.assertRaisesRegex(ValueError, 'dense_entries'):
            entries.dense_entries(self.T[0, 0], self.index)                 # more index rows than axes
        with self.assertRaisesRegex(ValueError, 'dense_probe'):
            probing.dense_probe(self.T[0, 0], self.ww)                      # more vectors than axes

    def test_they_are_in_their_modules_public_names(self):
        self.assertIn('dense_probe', probing.__all__)
        self.assertIn('dense_apply', apply.__all__)
        self.assertIn('dense_entries', entries.__all__)
        for function in (probing.dense_probe, apply.dense_apply, entries.dense_entries):
            names = list(inspect.signature(function).parameters)
            self.assertEqual(names[0], 'T')
            self.assertEqual(names[2:], ['pp', 'order'])
            self.assertEqual([p.default for p in inspect.signature(function).parameters.values()][2:], [None, 0])
        self.assertEqual(list(inspect.signature(probing.dense_probe).parameters)[1], 'ww')
        self.assertEqual(list(inspect.signature(apply.dense_apply).parameters)[1], 'ww')
        self.assertEqual(list(inspect.signature(entries.dense_entries).parameters)[1], 'index')

    def test_numpy_in_numpy_out_and_the_dtype_is_kept(self):
        T32 = self.T.astype(np.float32)
        ww32, pp32 = [x.astype(np.float32) for x in self.ww], [x.astype(np.float32) for x in self.pp]
        for value in (probing.dense_probe(T32, ww32, pp32, 2)[0], apply.dense_apply(T32, ww32, pp32, 2),
                      entries.dense_entries(T32, self.index, pp32, 2), entries.dense_entries(T32, self.index)):
            self.assertIsInstance(value, np.ndarray)
            self.assertEqual(value.dtype, np.float32)

    def test_jax_in_gives_jax_out(self):
        if not common.jax_available:
            self.skipTest('jax is not installed')
        import jax.numpy as jnp
        T, ww, pp = jnp.asarray(self.T), common.tree_to_jax(self.ww), common.tree_to_jax(self.pp)
        for got, expected in ((probing.dense_probe(T, ww, pp, 2)[1],
                               probing.dense_probe(self.T, self.ww, self.pp, 2)[1]),
                              (apply.dense_apply(T, ww, pp, 2), apply.dense_apply(self.T, self.ww, self.pp, 2)),
                              (entries.dense_entries(T, jnp.asarray(self.index), pp, 2),
                               entries.dense_entries(self.T, self.index, self.pp, 2)),
                              (apply.dense_apply(T, self.ww), apply.dense_apply(self.T, self.ww))):  # mixed: jax wins
            self.assertTrue(common.is_jax_ndarray(got))
            self.check(np.asarray(got), expected, 1e-4)


if __name__ == '__main__':
    unittest.main()
