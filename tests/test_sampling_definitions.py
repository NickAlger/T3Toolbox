# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""``backend/sampling_definitions.py``: the reference module, pinned to the definitions.

The reference computes every sampling variable by one dense contraction of its part of the network. Here it is
checked by lines of evidence that share nothing with it but the library's conversions to dense tensors:

1. **The outputs against dense tensors** (`TestOutputsAgainstDense`): ``z`` and ``Jp`` against the dense tensor and
   the dense tangent tensor, ``g`` and ``JtJp`` against the dense Jacobian, transposed; ``r``, the seed
   ``w_tilde``, ``f``, ``r_block_sumsq`` and ``Jp_norm_sq`` against their formulas. The sampling of a dense tensor
   and its jets are computed here by ``np.tensordot`` and by interpolation on a real grid, not by the library's
   dense functions.
2. **The edge variables against the recurrences** (`TestRecurrences`): every recurrence of ``SAMPLE_VARIABLES.md``
   section 5, written out here with truncated products as explicit loops over multi-indices. The reference
   contracts each whole subnetwork at once, so this is the page's statement that the two definitions agree.
3. **The readouts** (`TestReadouts`): section 6 of the page. The outputs and blocks (dense tensor, dense Jacobian,
   second derivative) equal the terms of the edge variables (subnetworks). This is what ties the tilde variables to
   the dense Jacobian and the tilde-dot variables to the second derivative.
4. **The pairings** (`TestPairings`): section 8 of the page, at every cut edge.
5. **Jets two ways** (`TestJetsTwoWays`): every variable at a jet shape against the interpolation, on two real
   grids that are not the reference's, of the same variable with no jets at the input jets evaluated there (with the
   stored seed as the polynomial ``sum_k w_tilde[k] s^k``, which is what "stored turned" means).
6. **Kp** (`TestKp`): automatic differentiation (jax arrays) against finite differences (numpy arrays), and that
   the route is read off the arrays; against the difference quotient of ``g`` on the corewise frame with the seed
   held; against the second derivative of the dense objective.
7. **The old kernels** (`TestAgainstTheOldKernels`): with one variable and a line input, ``z`` and ``g`` of the
   probe kind against ``sampling_derivatives`` (until it is deleted), through the conversions: derivatives are
   ``t!`` times Taylor coefficients, and the old transpose takes the residual forward, not turned.
8. **The corewise frame** (`TestCorewiseFrame`), **structure** (`TestStructure`: the names, the one signature, the
   jet shape and its reader, the declared jet sizes, the errors, a doctest on every function, and that the module
   calls nothing of the library but the three dense functions) and **dispatch** (`TestJax`).
9. **The dense functions** (`TestDenseFunctions`): with no jets against hand-written contractions and indexing;
   their jets at input jets of every kind (a line, two directions, a curve, modes of different jet sizes)
   against the calls with no jets along the input jets at a grid of roots of unity; the ``W`` and ``C`` stacks,
   the errors. **The input jets** (`TestInputJets`): ``affine_jet`` and ``index_to_one_hot``.

**The jet matrix** (`JET_CASES`), over which 1 to 6 run: no jets; a line at ``(1,)`` and ``(3,)``; two directions
at ``(1, 2)`` (``affine_jet(ww, p1, p2)``); modes of input jet sizes ``(3, 2)``, ``(1, 1)``, ``(2, 3)`` at
``(1, 2)`` (a curve in one variable and a line in the other, a discrete mode, and an input larger than the box on
the first axis); a curve of three slots at ``(2,)``; modes of input jet sizes ``(1,)``, ``(4,)``, ``(2,)`` at
``(2,)`` (a mode that does not move, and one clipped). The entries kind runs with no jets only.

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
import t3toolbox.backend.contractions as contractions
import t3toolbox.backend.jets as jets
import t3toolbox.backend.sampling_definitions as sdef
import t3toolbox.backend.probing as probing
import t3toolbox.backend.apply as apply
import t3toolbox.backend.entries as entries
import t3toolbox.backend.t3_conversions as t3_conversions
import t3toolbox.backend.tv_operations as tv_operations

KINDS = ('probe', 'apply', 'entries')
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
LEAF_NAMES = ('xi', 'xi_dot', 'xi_U_dot')                   # the declared jet sizes: the input's, clipped to F


#######################################
########    The jet matrix    #########
#######################################

class JetCase:
    """One jet shape and one kind of input jet: ``'none'`` (no jets), ``'line'`` (``affine_jet(ww, pp)``),
    ``'plane'`` (``affine_jet(ww, pp, qq)``), ``'curve'`` (three random slots), or a tuple of jet sizes, one per mode
    (cycled over the modes; random slots, slot 0 the point ``ww``)."""

    def __init__(self, label, order, form):
        self.label, self.order, self.form = label, tuple(order), form
        self.m = len(order)
        self.box = tuple(n + 1 for n in order)

    def __repr__(self):
        return self.label


NO_JETS = JetCase('no jets', (), 'none')
LINE_1 = JetCase('a line at (1,)', (1,), 'line')
LINE_3 = JetCase('a line at (3,)', (3,), 'line')
PLANE = JetCase('two directions at (1, 2)', (1, 2), 'plane')
MIXED_MODES = JetCase('modes (3, 2), (1, 1), (2, 3) at (1, 2)', (1, 2), ((3, 2), (1, 1), (2, 3)))
CURVE = JetCase('a curve of three slots at (2,)', (2,), 'curve')
MIXED_SIZES = JetCase('modes (1,), (4,), (2,) at (2,)', (2,), ((1,), (4,), (2,)))
JET_CASES = (NO_JETS, LINE_1, LINE_3, PLANE, MIXED_MODES, CURVE, MIXED_SIZES)
WITH_JETS = JET_CASES[1:]


class Jets:
    """The algebra of jets on the box ``F`` of one jet shape, written out with explicit loops over multi-indices.
    Inside these tests every jet is lifted to the full box: ``m`` leading axes of sizes ``F`` (none with no jets)."""

    def __init__(self, order):
        self.order, self.m = tuple(order), len(order)
        self.box = tuple(n + 1 for n in order)
        self.corner = tuple(order)
        self.slots = list(np.ndindex(*self.box))

    def lift(self, value):
        """A value (one array, or a tuple of them) on the full box: cut to it, and padded with zeros up to it."""
        if isinstance(value, (tuple, list)):
            return [self.lift(v) for v in value]
        value = np.asarray(value)
        if self.m == 0:
            return value
        value = value[tuple(slice(0, f) for f in self.box)]
        return np.pad(value, [(0, f - s) for f, s in zip(self.box, value.shape)] + [(0, 0)] * (value.ndim - self.m))

    def tprod(self, term, *jets_):
        """The truncated product: slot ``t`` is the sum, over the slots that add up to ``t``, of ``term`` of them."""
        out = None
        for ks in itertools.product(self.slots, repeat=len(jets_)):
            t = tuple(sum(k[j] for k in ks) for j in range(self.m))
            if any(a >= f for a, f in zip(t, self.box)):
                continue
            value = term(*(jet[k] for jet, k in zip(jets_, ks)))
            if out is None:
                out = np.zeros(self.box + np.shape(value))
            out[t] += value
        return out

    def top(self, term, *jets_):
        """The coefficient at the corner of the product."""
        return self.tprod(term, *jets_)[self.corner]

    def flip(self, x):
        return np.asarray(x)[(slice(None, None, -1),) * self.m]

    # the terms of SAMPLE_VARIABLES.md section 5
    def xi(self, B, w):
        return np.einsum('io,...o->...i', B, w)

    def z(self, eta, B):
        return np.einsum('...i,io->...o', eta, B)

    def mu(self, mu, G, xi):
        return self.tprod(lambda a, x: np.einsum('...a,aib,...i->...b', a, G, x), mu, xi)

    def nu(self, G, xi, nu):
        return self.tprod(lambda x, b: np.einsum('aib,...i,...b->...a', G, x, b), xi, nu)

    def eta(self, mu, G, nu):
        return self.tprod(lambda a, b: np.einsum('...a,aib,...b->...i', a, G, b), mu, nu)

    def V_top(self, eta, w):                    # summed over the samples, at the corner
        return self.top(lambda e, v: np.einsum('Wi,Wo->io', flat(e), flat(v)), eta, w)

    def H_top(self, mu, xi, nu):
        return self.top(lambda a, x, b: np.einsum('Wa,Wi,Wb->aib', flat(a), flat(x), flat(b)), mu, xi, nu)

    def pair(self, a, b):
        """Two jets on the two sides of one edge, contracted over the edge: a jet ``F + W``."""
        return self.tprod(lambda p, q: np.einsum('...a,...a->...', p, q), a, b)

    def top_pair(self, a, b):
        """The number two jets stand for together: the corner coefficient of their product, summed over everything."""
        return float(np.sum(self.pair(a, b)[self.corner]))

    def times_seed(self, seed, jet):
        """The seed of apply or entries (a jet ``F + W``, one number per sample) times a jet ``F + W + (size,)``."""
        return self.tprod(lambda c, x: c[..., None] * x, seed, jet)

    def ones(self, W, size):                    # the start of a plain sweep: ones in slot 0, zero elsewhere
        out = np.zeros(self.box + W + (size,))
        out[(0,) * self.m] = 1.0
        return out

    def weight_of(self, weight, kind, i):
        """The weight of mode ``i`` (probe) as an array of the box."""
        return np.asarray(weight)[i] if kind == 'probe' else np.asarray(weight)

    def weighted(self, jets_, weight, kind):
        """``weight**2 * jet``, slot by slot, on lifted jets (a list over the modes)."""
        return [(self.weight_of(weight, kind, i) ** 2).reshape(self.box + (1,) * (jet.ndim - self.m)) * jet
                for i, jet in enumerate(jets_)]


def flat(x_WX):                 # the sample stack as one axis
    return x_WX.reshape((-1,) + x_WX.shape[-1:])


def evaluate(jet, s):
    """An input jet (clipped to its box, ``m = len(s)`` jet axes) at the point ``s``: the sum over its slots."""
    m = len(s)
    value = 0
    for k in np.ndindex(*jet.shape[:m]):
        value = value + jet[k] * np.prod([s[j] ** k[j] for j in range(m)])
    return np.asarray(value) * np.ones(jet.shape[m:])


#######################################
########    Problems to test on    ####
#######################################

class Problem:
    """Random stores (not orthogonal, not consistent with each other: no definition needs that), a sample for
    each kind and jet case, both forms of a direction, data and a weight."""

    def __init__(self, seed, shape, nD, nU, rL, rR, W):
        rng = np.random.default_rng(seed)
        rn = rng.standard_normal
        d = len(shape)
        self.seed, self.d, self.shape, self.W = seed, d, tuple(shape), tuple(W)
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
        self.qq = [rn(self.W + (n,)) for n in shape]
        self.index = np.stack([rng.integers(0, n, size=self.W) for n in shape])
        self.rng = rng

    def inputs(self, kind):
        """The input vectors at s = 0: ``ww``, or the unit vectors at ``index``."""
        if kind == 'entries':
            return [np.eye(n)[self.index[i]] for i, n in enumerate(self.shape)]
        return self.ww

    def input_jets(self, case):
        """The input jets of a jet case, slot 0 the point ``ww`` (the same for every case)."""
        if case.form == 'none':
            return list(self.ww)
        if case.form == 'line':
            return list(jets.affine_jet(self.ww, self.pp))
        if case.form == 'plane':
            return list(jets.affine_jet(self.ww, self.pp, self.qq))
        sizes = [(3,)] * self.d if case.form == 'curve' else [case.form[i % len(case.form)] for i in range(self.d)]
        rng = np.random.default_rng(1000 * self.seed + JET_CASES.index(case))
        out = []
        for i, size in enumerate(sizes):
            jet = rng.standard_normal(tuple(size) + self.W + (self.shape[i],))
            jet[(0,) * case.m] = self.ww[i]
            out.append(jet)
        return out

    def sample(self, kind, case):
        return self.index if kind == 'entries' else self.input_jets(case)

    def data(self, kind, case):
        """``(y, weight)``: of the type of ``z``, and of the shape the weight has for the kind and the jet shape."""
        rng = np.random.default_rng(2000 + JET_CASES.index(case))
        F = case.box
        if kind == 'probe':
            return [rng.standard_normal(F + self.W + (n,)) for n in self.shape], 0.5 + rng.random((self.d,) + F)
        return rng.standard_normal(F + self.W), 0.5 + rng.random(F)

    def arguments(self, kind, case):
        y, weight = self.data(kind, case)
        return dict(order=case.order, variations=(self.V, self.H), core_directions=(self.D_dot, self.L_dot),
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


def cases(problems=PROBLEMS, kinds=KINDS, jet_cases=JET_CASES):
    """``(labels, problem, kind, jet case)`` over the matrix; the entries kind with no jets only."""
    for name, make in problems:
        for kind in kinds:
            for case in jet_cases:
                if kind == 'entries' and case.m:
                    continue
                yield dict(problem=name, kind=kind, jets=case.label), make(), kind, case


def relative_error(got, expected):
    got, expected = np.asarray(got), np.asarray(expected)
    assert got.shape == expected.shape, (got.shape, expected.shape)
    return np.linalg.norm(got - expected) / max(np.linalg.norm(expected), 1.0)


#######################################################################
########    A dense tensor sampled, by a route of the tests' own    ####
#######################################################################

def sample_dense(kind, T, vectors):
    """No jets: a dense tensor ``T`` (no stack) contracted with vectors ``W + (Ni,)``, by ``np.tensordot`` one
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


def node_sets(degree):
    """Two sets of ``degree + 1`` real nodes, neither of them the reference's (roots of unity)."""
    return np.linspace(-1.0, 1.0, degree + 1), np.linspace(-0.6, 1.3, degree + 1)


def interpolate(value_at, degrees, sizes, which=0):
    """The Taylor coefficients, on the box ``sizes``, of a polynomial in ``s_1 ... s_m`` of degree at most
    ``degrees[j]`` in ``s_j``, from its values on a grid of real nodes (``node_sets``, set ``which``):
    ``value_at(s)`` returns a list of arrays; a tensor-product Vandermonde system, solved one axis at a time."""
    m = len(degrees)
    nodes = [node_sets(g)[which] for g in degrees]
    values = None
    for node in itertools.product(*(range(len(n)) for n in nodes)):
        value = [np.asarray(v) for v in value_at(tuple(nodes[j][node[j]] for j in range(m)))]
        if values is None:
            values = [np.zeros(tuple(len(n) for n in nodes) + v.shape) for v in value]
        for stored, v in zip(values, value):
            stored[node] = v
    out = []
    for stored in values:
        for j in range(m):
            inverse = np.linalg.inv(np.vander(nodes[j], increasing=True))
            stored = np.moveaxis(np.tensordot(inverse, stored, axes=([1], [j])), 0, j)
        box = np.zeros(tuple(sizes) + stored.shape[m:])
        inside = tuple(slice(0, min(s, g + 1)) for s, g in zip(sizes, degrees))
        box[inside] = stored[inside]
        out.append(box)
    return out


def clipped_inputs(P, kind, case):
    """The input jets of a case, cut to the box (slots beyond it reach no coefficient on it)."""
    return [x[tuple(slice(0, f) for f in case.box)] for x in P.input_jets(case)] if kind != 'entries' \
        else P.inputs(kind)


def degree_bound(P, kind, case, seed=False):
    """A bound on the degree in each ``s_j`` of a sampled output (and, with ``seed``, of a tilde variable)."""
    inputs = clipped_inputs(P, kind, case)
    return [sum(x.shape[j] - 1 for x in inputs) + (case.order[j] if seed else 0) for j in range(case.m)]


def sample_dense_jet(P, kind, T, case, which=0):
    """``T`` sampled at the input jets of a case, in Taylor coefficients on the box, of the type of ``z``: by
    interpolation of `sample_dense` on a real grid. With no jets, `sample_dense` itself."""
    inputs = clipped_inputs(P, kind, case)
    if case.m == 0:
        return sample_dense(kind, T, inputs)
    at = lambda s: (lambda value: value if kind == 'probe' else [value])(
        sample_dense(kind, T, [evaluate(x, s) for x in inputs]))
    value = interpolate(at, degree_bound(P, kind, case), case.box, which)
    return value if kind == 'probe' else value[0]


def modes_of(kind, value):
    """An object of the type of ``z`` as a list of arrays: the modes for probe, one array otherwise."""
    return list(value) if kind == 'probe' else [value]


class TestOutputsAgainstDense(unittest.TestCase):
    """The outputs against dense tensors the library's conversions build, sampled by the tests' own route."""

    def check(self, got, expected, tol=TOL):
        self.assertLessEqual(relative_error(got, expected), tol)

    def test_z_is_the_dense_tensor_contracted_with_the_inputs(self):
        for labels, P, kind, case in cases():
            with self.subTest(**labels):
                T = t3_conversions.t3_to_dense((P.D, P.L))
                expected = modes_of(kind, sample_dense_jet(P, kind, T, case))
                got = modes_of(kind, sdef.z(kind, P.frame, P.sample(kind, case), order=case.order))
                self.assertEqual(len(got), len(expected))
                for a, b in zip(got, expected):
                    self.check(a, b)

    def test_Jp_is_the_dense_tangent_tensor_contracted_with_the_inputs(self):
        for labels, P, kind, case in cases():
            with self.subTest(**labels):
                T_dot = tv_operations.tv_to_dense(P.frame, (P.V, P.H))
                expected = modes_of(kind, sample_dense_jet(P, kind, T_dot, case))
                got = modes_of(kind, sdef.Jp(kind, P.frame, P.sample(kind, case), order=case.order,
                                             variations=(P.V, P.H)))
                for a, b in zip(got, expected):
                    self.check(a, b)

    def test_r_and_the_seed_and_the_sums_are_their_formulas(self):
        for labels, P, kind, case in cases():
            with self.subTest(**labels):
                J = Jets(case.order)
                sample, args = P.sample(kind, case), P.arguments(kind, case)
                y, weight = args['y'], args['weight']
                z = J.lift(modes_of(kind, sdef.z(kind, P.frame, sample, order=case.order)))
                r = J.lift(modes_of(kind, sdef.r(kind, P.frame, sample, **args)))
                for r_i, z_i, y_i in zip(r, z, J.lift(modes_of(kind, y))):
                    self.check(r_i, z_i - y_i)

                # the seed: the weights, then the turn, a flip of every jet axis
                seed = J.lift(modes_of(kind, sdef.w_tilde(kind, P.frame, sample, **args)))
                for seed_i, weighted_r_i in zip(seed, J.weighted(r, weight, kind)):
                    self.check(seed_i, J.flip(weighted_r_i))
                if case.m:                                  # and the turn is not a no-op here
                    self.assertGreater(relative_error(seed[0], J.weighted(r, weight, kind)[0]), 1e-3)

                f = sdef.f(kind, P.frame, sample, **args)
                self.assertEqual(np.shape(f), ())
                self.check(f, 0.5 * sum(np.sum(wr * r_i) for wr, r_i in zip(J.weighted(r, weight, kind), r)))

                sums = sdef.r_block_sumsq(kind, P.frame, sample, **args)
                self.assertEqual(np.shape(sums), np.shape(weight))          # structural: exact
                expected = np.array([np.sum(r_i ** 2, axis=tuple(range(case.m, r_i.ndim))) for r_i in r])
                self.check(np.reshape(sums, expected.shape), expected)

                Jp = J.lift(modes_of(kind, sdef.Jp(kind, P.frame, sample, **args)))
                Jp_norm_sq = sdef.Jp_norm_sq(kind, P.frame, sample, **args)
                self.assertEqual(np.shape(Jp_norm_sq), ())
                self.check(Jp_norm_sq, sum(np.sum(wJ * J_) for wJ, J_ in zip(J.weighted(Jp, weight, kind), Jp)))

    def jacobian_columns(self, P, kind, case):
        """The dense Jacobian of ``(V, H) -> Jp``, one column per unit direction: a list, over the blocks
        ``V[0..d-1], H[0..d-1]``, of arrays ``block shape + (list over modes of lifted jets)``."""
        J = Jets(case.order)
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
                    columns[entry] = J.lift(modes_of(kind, sample_dense_jet(P, kind, T_dot, case)))
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
        for labels, P, kind, case in cases(problems=PROBLEMS[:2], jet_cases=(NO_JETS, LINE_3, PLANE, MIXED_MODES)):
            with self.subTest(**labels):
                J = Jets(case.order)
                sample, args = P.sample(kind, case), P.arguments(kind, case)
                weight = args['weight']
                blocks = self.jacobian_columns(P, kind, case)

                r = J.lift(modes_of(kind, sdef.r(kind, P.frame, sample, **args)))
                expected = self.transposed(blocks, J.weighted(r, weight, kind))
                V_tilde, H_tilde = sdef.g(kind, P.frame, sample, **args)
                self.assertEqual((len(V_tilde), len(H_tilde)), (P.d, P.d))
                for a, b in zip(list(V_tilde) + list(H_tilde), expected):
                    self.check(a, b)
                # the two blocks' own functions are the same arrays
                for a, b in zip(sdef.V_tilde(kind, P.frame, sample, **args), V_tilde):
                    self.assertTrue(np.array_equal(a, b))
                for a, b in zip(sdef.H_tilde(kind, P.frame, sample, **args), H_tilde):
                    self.assertTrue(np.array_equal(a, b))

                Jp = J.lift(modes_of(kind, sdef.Jp(kind, P.frame, sample, **args)))
                expected = self.transposed(blocks, J.weighted(Jp, weight, kind))
                JtJp = sdef.JtJp(kind, P.frame, sample, **args)
                for a, b in zip(list(JtJp[0]) + list(JtJp[1]), expected):
                    self.check(a, b)

    def test_a_given_seed_is_used_as_it_is(self):
        # g with the residual's seed passed in as w_tilde is g from y and the weight; and a seed of one's own is
        # paired turned: the transpose of an UN-turned functional c is g(w_tilde = c turned).
        for labels, P, kind, case in cases(problems=PROBLEMS[:1], jet_cases=(NO_JETS, LINE_3, PLANE)):
            with self.subTest(**labels):
                J = Jets(case.order)
                sample, args = P.sample(kind, case), P.arguments(kind, case)
                seed = sdef.w_tilde(kind, P.frame, sample, **args)
                from_y = sdef.g(kind, P.frame, sample, **args)
                from_seed = sdef.g(kind, P.frame, sample, order=case.order, w_tilde=seed)
                for a, b in zip(from_y[0] + from_y[1], from_seed[0] + from_seed[1]):
                    self.assertTrue(np.array_equal(a, b))

                c = [P.rng.standard_normal(np.shape(s)) for s in modes_of(kind, seed)]      # forward order
                turned = [J.flip(c_i) for c_i in c]
                got = sdef.g(kind, P.frame, sample, order=case.order, w_tilde=turned if kind == 'probe' else turned[0])
                expected = self.transposed(self.jacobian_columns(P, kind, case), J.lift(c))
                for a, b in zip(list(got[0]) + list(got[1]), expected):
                    self.check(a, b)


class Variables:
    """Every variable of one case from the reference, lifted to the full box, by name: ``v.mu[i]`` is a jet with
    ``m`` axes of the box. Computed on first use."""

    def __init__(self, P, kind, case, **overrides):
        self.P, self.kind, self.case = P, kind, case
        self.J = Jets(case.order)
        self.args = dict(P.arguments(kind, case), **overrides)
        self.sample = P.sample(kind, case)
        self.cache = {}

    def __getattr__(self, name):
        if name not in ALL_NAMES:
            raise AttributeError(name)
        if name not in self.cache:
            self.cache[name] = self.J.lift(getattr(sdef, name)(self.kind, self.P.frame, self.sample, **self.args))
        return self.cache[name]


class TestRecurrences(unittest.TestCase):
    """SAMPLE_VARIABLES.md section 5: each variable, contracted whole by the reference, satisfies its
    recurrence. The recurrences are written out here from the page, with truncated products on the box."""

    def check(self, got, expected, what):
        self.assertLessEqual(relative_error(got, expected), TOL, what)

    def test_plain(self):
        for labels, P, kind, case in cases():
            with self.subTest(**labels):
                v, d, J = Variables(P, kind, case), P.d, Jets(case.order)
                D, U, L, R = P.frame
                for i in range(d):
                    self.check(v.xi[i], J.xi(D[i], v.w[i]), 'xi')
                    self.check(v.mu[i + 1], J.mu(v.mu[i], L[i], v.xi[i]), 'mu')
                    self.check(v.nu[i], J.nu(R[i], v.xi[i], v.nu[i + 1]), 'nu')
                    self.check(v.nu_L[i], J.nu(L[i], v.xi[i], v.nu_L[i + 1]), 'nu_L')
                    self.check(v.eta[i], J.eta(v.mu[i], U[i], v.nu[i + 1]), 'eta')
                    self.check(v.eta_D[i], J.eta(v.mu[i], L[i], v.nu_L[i + 1]), 'eta_D')
                # the starts: ones in slot 0 and zero elsewhere, sized by the end bond of the cores (tails). Exact
                # with no jets; with jets they come out of the interpolation, so to rounding.
                for start, size in ((v.mu[0], L[0].shape[0]), (v.nu[d], R[d - 1].shape[2]),
                                    (v.nu_L[d], L[d - 1].shape[2])):
                    self.assertEqual(start.shape, case.box + P.W + (size,))
                    if case.m == 0:
                        self.assertTrue(np.array_equal(start, J.ones(P.W, size)))
                    self.check(start, J.ones(P.W, size), 'a start')

    def test_dot_in_the_frame_form(self):
        for labels, P, kind, case in cases():
            with self.subTest(**labels):
                v, d, J = Variables(P, kind, case, core_directions=None), P.d, Jets(case.order)
                D, U, L, R = P.frame
                V, H = P.V, P.H
                for i in range(d):
                    self.check(v.xi_U_dot[i], J.xi(V[i], v.w[i]), 'xi_U_dot')
                    self.check(v.mu_R_dot[i + 1],
                               J.mu(v.mu_R_dot[i], R[i], v.xi[i]) + J.mu(v.mu[i], H[i], v.xi[i])
                               + J.mu(v.mu[i], U[i], v.xi_U_dot[i]), 'mu_R_dot')
                    self.check(v.nu_L_dot[i],
                               J.nu(L[i], v.xi[i], v.nu_L_dot[i + 1]) + J.nu(H[i], v.xi[i], v.nu[i + 1])
                               + J.nu(U[i], v.xi_U_dot[i], v.nu[i + 1]), 'nu_L_dot')
                    self.check(v.eta_D_dot[i],
                               J.eta(v.mu_R_dot[i], R[i], v.nu[i + 1]) + J.eta(v.mu[i], H[i], v.nu[i + 1])
                               + J.eta(v.mu[i], L[i], v.nu_L_dot[i + 1]), 'eta_D_dot')
                self.assertTrue(np.all(v.mu_R_dot[0] == 0) and v.mu_R_dot[0].shape[-1] == R[0].shape[0])
                self.assertTrue(np.all(v.nu_L_dot[d] == 0) and v.nu_L_dot[d].shape[-1] == L[d - 1].shape[2])

    def test_dot_in_the_raw_form(self):
        for labels, P, kind, case in cases():
            with self.subTest(**labels):
                v, d, J = Variables(P, kind, case, variations=None), P.d, Jets(case.order)
                D, U, L, R = P.frame
                D_dot, L_dot = P.D_dot, P.L_dot
                for i in range(d):
                    self.check(v.xi_dot[i], J.xi(D_dot[i], v.w[i]), 'xi_dot')
                    self.check(v.mu_dot[i + 1],
                               J.mu(v.mu_dot[i], L[i], v.xi[i]) + J.mu(v.mu[i], L_dot[i], v.xi[i])
                               + J.mu(v.mu[i], L[i], v.xi_dot[i]), 'mu_dot')
                    self.check(v.nu_L_dot[i],
                               J.nu(L[i], v.xi[i], v.nu_L_dot[i + 1]) + J.nu(L_dot[i], v.xi[i], v.nu_L[i + 1])
                               + J.nu(L[i], v.xi_dot[i], v.nu_L[i + 1]), 'nu_L_dot (raw)')
                    self.check(v.eta_D_dot[i],
                               J.eta(v.mu_dot[i], L[i], v.nu_L[i + 1]) + J.eta(v.mu[i], L_dot[i], v.nu_L[i + 1])
                               + J.eta(v.mu[i], L[i], v.nu_L_dot[i + 1]), 'eta_D_dot (raw)')
                self.assertTrue(np.all(v.mu_dot[0] == 0) and np.all(v.nu_L_dot[d] == 0))

    def seed_starts(self, v):
        """What the seed puts in: the seeded leaves, and the start of the right-to-left tilde sweeps. probe: the
        seed at the inputs, sweeps from zero. apply, entries: no seeded leaf, sweeps from the seed times ones."""
        P, d, J = v.P, v.P.d, v.J
        D, U, L, R = P.frame
        if v.kind == 'probe':
            xi_tilde = [J.xi(D[i], v.w_tilde[i]) for i in range(d)]
            xi_tilde_dot = [J.xi(P.D_dot[i], v.w_tilde[i]) for i in range(d)]
            start = lambda size: np.zeros(J.box + P.W + (size,))
        else:
            xi_tilde = [np.zeros(J.box + P.W + (D[i].shape[0],)) for i in range(d)]
            xi_tilde_dot = xi_tilde
            start = lambda size: v.w_tilde[..., None] * np.ones(size)
        return xi_tilde, xi_tilde_dot, start

    def test_tilde(self):
        for labels, P, kind, case in cases():
            with self.subTest(**labels):
                v, d, J = Variables(P, kind, case), P.d, Jets(case.order)
                D, U, L, R = P.frame
                xi_tilde, _, start = self.seed_starts(v)
                for i in range(d):
                    self.check(v.xi_tilde[i], xi_tilde[i], 'xi_tilde')
                    self.check(v.mu_tilde[i + 1],
                               J.mu(v.mu_tilde[i], L[i], v.xi[i]) + J.mu(v.mu[i], L[i], v.xi_tilde[i]), 'mu_tilde')
                    self.check(v.nu_tilde[i],
                               J.nu(R[i], v.xi[i], v.nu_tilde[i + 1]) + J.nu(R[i], v.xi_tilde[i], v.nu[i + 1]),
                               'nu_tilde')
                    self.check(v.nu_L_tilde[i],
                               J.nu(L[i], v.xi[i], v.nu_L_tilde[i + 1]) + J.nu(L[i], v.xi_tilde[i], v.nu_L[i + 1]),
                               'nu_L_tilde')
                    self.check(v.eta_tilde[i],
                               J.eta(v.mu_tilde[i], U[i], v.nu[i + 1]) + J.eta(v.mu[i], U[i], v.nu_tilde[i + 1]),
                               'eta_tilde')
                    self.check(v.eta_D_tilde[i],
                               J.eta(v.mu_tilde[i], L[i], v.nu_L[i + 1]) + J.eta(v.mu[i], L[i], v.nu_L_tilde[i + 1]),
                               'eta_D_tilde')
                self.assertTrue(np.all(v.mu_tilde[0] == 0))
                self.check(v.nu_tilde[d], start(R[d - 1].shape[2]), 'the start of nu_tilde')
                self.check(v.nu_L_tilde[d], start(L[d - 1].shape[2]), 'the start of nu_L_tilde')
                if kind != 'probe':                         # no left side and no leaf holds the seed: exact zeros
                    for i in range(d):
                        self.assertTrue(np.all(v.xi_tilde[i] == 0) and np.all(v.mu_tilde[i + 1] == 0))

    def test_tilde_dot(self):
        for labels, P, kind, case in cases():
            with self.subTest(**labels):
                v, d, J = Variables(P, kind, case, variations=None), P.d, Jets(case.order)    # the raw dots
                D, U, L, R = P.frame
                D_dot, L_dot = P.D_dot, P.L_dot
                _, xi_tilde_dot, _ = self.seed_starts(v)
                for i in range(d):
                    self.check(v.xi_tilde_dot[i], xi_tilde_dot[i], 'xi_tilde_dot')
                    self.check(v.mu_tilde_dot[i + 1],
                               J.mu(v.mu_tilde_dot[i], L[i], v.xi[i]) + J.mu(v.mu_dot[i], L[i], v.xi_tilde[i])
                               + J.mu(v.mu_tilde[i], L[i], v.xi_dot[i]) + J.mu(v.mu[i], L[i], v.xi_tilde_dot[i])
                               + J.mu(v.mu_tilde[i], L_dot[i], v.xi[i]) + J.mu(v.mu[i], L_dot[i], v.xi_tilde[i]),
                               'mu_tilde_dot')
                    self.check(v.nu_L_tilde_dot[i],
                               J.nu(L[i], v.xi[i], v.nu_L_tilde_dot[i + 1])
                               + J.nu(L[i], v.xi_tilde[i], v.nu_L_dot[i + 1])
                               + J.nu(L[i], v.xi_dot[i], v.nu_L_tilde[i + 1])
                               + J.nu(L[i], v.xi_tilde_dot[i], v.nu_L[i + 1])
                               + J.nu(L_dot[i], v.xi[i], v.nu_L_tilde[i + 1])
                               + J.nu(L_dot[i], v.xi_tilde[i], v.nu_L[i + 1]), 'nu_L_tilde_dot')
                    self.check(v.eta_D_tilde_dot[i],
                               J.eta(v.mu_tilde_dot[i], L[i], v.nu_L[i + 1])
                               + J.eta(v.mu_dot[i], L[i], v.nu_L_tilde[i + 1])
                               + J.eta(v.mu_tilde[i], L[i], v.nu_L_dot[i + 1])
                               + J.eta(v.mu[i], L[i], v.nu_L_tilde_dot[i + 1])
                               + J.eta(v.mu_tilde[i], L_dot[i], v.nu_L[i + 1])
                               + J.eta(v.mu[i], L_dot[i], v.nu_L_tilde[i + 1]), 'eta_D_tilde_dot')
                self.assertTrue(np.all(v.mu_tilde_dot[0] == 0) and np.all(v.nu_L_tilde_dot[d] == 0))


class TestReadouts(unittest.TestCase):
    """SAMPLE_VARIABLES.md section 6: the outputs and the blocks, which the reference takes from dense tensors,
    from the dense Jacobian and from the second derivative, are the terms of the edge variables, which it takes
    from subnetworks. This ties the tilde variables to the Jacobian and the tilde-dot variables to ``Kp``."""

    def check(self, got, expected, what):
        self.assertLessEqual(relative_error(got, expected), TOL, what)

    def test_z_and_Jp(self):
        for labels, P, kind, case in cases():
            with self.subTest(**labels):
                v, d, J = Variables(P, kind, case, core_directions=None), P.d, Jets(case.order)
                D, U, L, R = P.frame
                if kind == 'probe':
                    for i in range(d):
                        self.check(v.z[i], J.z(v.eta_D[i], D[i]), 'z')
                        self.check(v.Jp[i], J.z(v.eta_D_dot[i], D[i]) + J.z(v.eta[i], P.V[i]), 'Jp')
                else:                                       # read at the right end: bond d, summed with the ones
                    self.check(v.z, v.mu[d].sum(axis=-1), 'z')
                    self.check(v.Jp, v.mu_R_dot[d].sum(axis=-1), 'Jp')

    def test_the_blocks_of_g(self):
        for labels, P, kind, case in cases():
            with self.subTest(**labels):
                v, d, J = Variables(P, kind, case), P.d, Jets(case.order)
                V_tilde, H_tilde = sdef.g(kind, P.frame, v.sample, **v.args)
                for i in range(d):
                    V_block = J.V_top(v.eta_tilde[i], v.w[i])
                    if kind == 'probe':                     # the seed at input i itself
                        V_block = V_block + J.V_top(v.eta[i], v.w_tilde[i])
                    self.check(V_tilde[i], V_block, 'V_tilde')
                    self.check(H_tilde[i],
                               J.H_top(v.mu_tilde[i], v.xi[i], v.nu[i + 1]) + J.H_top(v.mu[i], v.xi_tilde[i], v.nu[i + 1])
                               + J.H_top(v.mu[i], v.xi[i], v.nu_tilde[i + 1]), 'H_tilde')

    def test_the_blocks_of_Kp(self):
        for labels, P, kind, case in cases():
            with self.subTest(**labels):
                v, d, J = Variables(P, kind, case, variations=None), P.d, Jets(case.order)    # the raw dots
                V_tilde_dot, H_tilde_dot = sdef.Kp(kind, P.frame, v.sample, **v.args)
                for i in range(d):
                    V_block = J.V_top(v.eta_D_tilde_dot[i], v.w[i])
                    if kind == 'probe':
                        V_block = V_block + J.V_top(v.eta_D_dot[i], v.w_tilde[i])
                    self.check(V_tilde_dot[i], V_block, 'V_tilde_dot')
                    mu = {'': v.mu[i], 'd': v.mu_dot[i], 't': v.mu_tilde[i], 'td': v.mu_tilde_dot[i]}
                    xi = {'': v.xi[i], 'd': v.xi_dot[i], 't': v.xi_tilde[i], 'td': v.xi_tilde_dot[i]}
                    nu = {'': v.nu_L[i + 1], 'd': v.nu_L_dot[i + 1], 't': v.nu_L_tilde[i + 1],
                          'td': v.nu_L_tilde_dot[i + 1]}
                    nine = [g for g in itertools.product(('', 'd', 't', 'td'), repeat=3)
                            if ''.join(sorted(''.join(g))) == 'dt']           # one tilde and one dot in all
                    self.assertEqual(len(nine), 9)
                    self.check(H_tilde_dot[i], sum(J.H_top(mu[a], xi[b], nu[c]) for a, b, c in nine), 'H_tilde_dot')
                # the two blocks' own functions are the same arrays
                for a, b in zip(sdef.V_tilde_dot(kind, P.frame, v.sample, **v.args), V_tilde_dot):
                    self.assertTrue(np.array_equal(a, b))
                for a, b in zip(sdef.H_tilde_dot(kind, P.frame, v.sample, **v.args), H_tilde_dot):
                    self.assertTrue(np.array_equal(a, b))


class TestPairings(unittest.TestCase):
    """SAMPLE_VARIABLES.md section 8, at every cut edge."""

    def test_the_direction_at_every_cut_edge(self):
        # At a cut bond the direction's sampled network is mu_R_dot paired with nu plus mu paired with nu_L_dot;
        # at a cut Tucker edge, xi_U_dot paired with eta plus xi paired with eta_D_dot. Every cut gives the same
        # jet per sample: the direction's network with every mode contracted, which is Jp of apply / entries.
        for labels, P, kind, case in cases(kinds=('apply', 'entries')):
            with self.subTest(**labels):
                v, d, J = Variables(P, kind, case, core_directions=None), P.d, Jets(case.order)
                cuts = ([J.pair(v.mu_R_dot[i], v.nu[i]) + J.pair(v.mu[i], v.nu_L_dot[i]) for i in range(d + 1)]
                        + [J.pair(v.xi_U_dot[i], v.eta[i]) + J.pair(v.xi[i], v.eta_D_dot[i]) for i in range(d)])
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
        for labels, P, kind, case in cases(kinds=('probe',)):
            with self.subTest(**labels):
                d, J = P.d, Jets(case.order)
                base = Variables(P, kind, case)
                seed = base.w_tilde

                def modes(direction, which):
                    Jp = Variables(P, kind, case, variations=direction).Jp
                    return sum(J.top_pair(seed[k], Jp[k]) for k in which)

                for i in range(d + 1):
                    right = self.one_sided(P, lambda j: j >= i)                 # mu_tilde multiplies nu_L_dot
                    v = Variables(P, kind, case, variations=right, core_directions=None)
                    self.check(J.top_pair(base.mu_tilde[i], v.nu_L_dot[i]), modes(right, range(i)))
                    left = self.one_sided(P, lambda j: j < i)                   # nu_tilde multiplies mu_R_dot
                    v = Variables(P, kind, case, variations=left, core_directions=None)
                    self.check(J.top_pair(base.nu_tilde[i], v.mu_R_dot[i]), modes(left, range(i, d)))
                for i in range(d):
                    leaf = (self.one_sided(P, lambda j: j == i)[0], [0 * H for H in P.H])   # only V[i]
                    v = Variables(P, kind, case, variations=leaf, core_directions=None)
                    self.check(J.top_pair(base.eta_tilde[i], v.xi_U_dot[i]),    # eta_tilde multiplies xi_U_dot
                               modes(leaf, [k for k in range(d) if k != i]))
                    trunk = (self.one_sided(P, lambda j: j != i)[0], P.H)       # everything but V[i]
                    v = Variables(P, kind, case, variations=trunk, core_directions=None)
                    self.check(J.top_pair(base.xi_tilde[i], v.eta_D_dot[i]), modes(trunk, [i]))   # xi_tilde: eta_D_dot

    def test_the_multipliers_of_apply_and_entries(self):
        # The seed sits at bond d, so it is on the right of every bond and in every trunk: nu_tilde multiplies
        # mu_R_dot and eta_tilde multiplies xi_U_dot, and there is no mu_tilde and no xi_tilde.
        for labels, P, kind, case in cases(kinds=('apply', 'entries')):
            with self.subTest(**labels):
                d, J = P.d, Jets(case.order)
                base = Variables(P, kind, case)

                def paired_with_Jp(direction):
                    Jp = Variables(P, kind, case, variations=direction).Jp
                    return J.top_pair(base.w_tilde[..., None], Jp[..., None])
                for i in range(d + 1):
                    left = self.one_sided(P, lambda j: j < i)
                    v = Variables(P, kind, case, variations=left, core_directions=None)
                    self.check(J.top_pair(base.nu_tilde[i], v.mu_R_dot[i]), paired_with_Jp(left))
                    self.assertTrue(np.all(base.mu_tilde[i] == 0))
                for i in range(d):
                    leaf = (self.one_sided(P, lambda j: j == i)[0], [0 * H for H in P.H])
                    v = Variables(P, kind, case, variations=leaf, core_directions=None)
                    self.check(J.top_pair(base.eta_tilde[i], v.xi_U_dot[i]), paired_with_Jp(leaf))
                    self.assertTrue(np.all(base.xi_tilde[i] == 0))


class TestApplyAndEntries(unittest.TestCase):
    """What the tilde and tilde-dot variables ARE for apply and entries, whose seed is one number per sample at
    bond ``d``: zero where the side does not reach that bond, the seed times the untilded variable where it does.
    And which of the pairings of section 8 hold for which kind. The docstrings of the reference say these."""

    def check(self, got, expected, what):
        self.assertLessEqual(relative_error(got, expected), TOL, what)

    def test_they_are_zero_or_the_seed_times_the_untilded_variable(self):
        for labels, P, kind, case in cases(kinds=('apply', 'entries')):
            with self.subTest(**labels):
                v, d, J = Variables(P, kind, case, variations=None), P.d, Jets(case.order)   # the raw dots
                seed = v.w_tilde
                for i in range(d + 1):
                    self.assertTrue(np.all(v.mu_tilde[i] == 0) and np.all(v.mu_tilde_dot[i] == 0))
                    self.check(v.nu_tilde[i], J.times_seed(seed, v.nu[i]), 'nu_tilde')
                    self.check(v.nu_L_tilde[i], J.times_seed(seed, v.nu_L[i]), 'nu_L_tilde')
                    self.check(v.nu_L_tilde_dot[i], J.times_seed(seed, v.nu_L_dot[i]), 'nu_L_tilde_dot')
                raw = sdef.xi_tilde(kind, P.frame, v.sample, **v.args)          # not lifted: its own jet sizes
                for i in range(d):
                    self.assertTrue(np.all(v.xi_tilde[i] == 0) and np.all(v.xi_tilde_dot[i] == 0))
                    self.assertEqual(raw[i].shape, case.box + P.W + (P.D[i].shape[0],))     # F, not the input's
                    self.check(v.eta_tilde[i], J.times_seed(seed, v.eta[i]), 'eta_tilde')
                    self.check(v.eta_D_tilde[i], J.times_seed(seed, v.eta_D[i]), 'eta_D_tilde')
                    self.check(v.eta_D_tilde_dot[i], J.times_seed(seed, v.eta_D_dot[i]), 'eta_D_tilde_dot')

    def test_their_blocks_have_only_the_terms_with_the_seed_on_the_right(self):
        for labels, P, kind, case in cases(kinds=('apply', 'entries')):
            with self.subTest(**labels):
                v, d, J = Variables(P, kind, case, variations=None), P.d, Jets(case.order)
                V_tilde, H_tilde = sdef.g(kind, P.frame, v.sample, **v.args)
                V_tilde_dot, H_tilde_dot = sdef.Kp(kind, P.frame, v.sample, **v.args)
                for i in range(d):
                    self.check(V_tilde[i], J.V_top(v.eta_tilde[i], v.w[i]), 'V_tilde')
                    self.check(H_tilde[i], J.H_top(v.mu[i], v.xi[i], v.nu_tilde[i + 1]), 'H_tilde')
                    self.check(V_tilde_dot[i], J.V_top(v.eta_D_tilde_dot[i], v.w[i]), 'V_tilde_dot')
                    self.check(H_tilde_dot[i],
                               J.H_top(v.mu_dot[i], v.xi[i], v.nu_L_tilde[i + 1])
                               + J.H_top(v.mu[i], v.xi_dot[i], v.nu_L_tilde[i + 1])
                               + J.H_top(v.mu[i], v.xi[i], v.nu_L_tilde_dot[i + 1]), 'H_tilde_dot')

    def test_two_of_the_four_pairings_are_of_the_probe_kind_only(self):
        # Section 8 pairs mu_tilde with nu_L_dot and xi_tilde with eta_D_dot. For apply and entries those two
        # tilde variables are zero, and what multiplies nu_L_dot and eta_D_dot is the seed times mu and the seed
        # times xi, which have no names. (The other two pairings hold for every kind: TestPairings.)
        for labels, P, kind, case in cases(kinds=('apply', 'entries'), jet_cases=(NO_JETS, LINE_3, MIXED_MODES)):
            with self.subTest(**labels):
                d, J = P.d, Jets(case.order)
                base = Variables(P, kind, case)
                seed = base.w_tilde
                for i in range(1, d):                       # a bond with sites on both sides
                    right = ([V if j >= i else 0 * V for j, V in enumerate(P.V)],
                             [H if j >= i else 0 * H for j, H in enumerate(P.H)])
                    v = Variables(P, kind, case, variations=right, core_directions=None)
                    paired_with_Jp = J.top_pair(seed[..., None], v.Jp[..., None])
                    self.assertGreater(abs(paired_with_Jp), 1e-3)
                    self.assertEqual(J.top_pair(base.mu_tilde[i], v.nu_L_dot[i]), 0.0)
                    self.assertLessEqual(abs(J.top_pair(J.times_seed(seed, base.mu[i]), v.nu_L_dot[i]) - paired_with_Jp),
                                         TOL * abs(paired_with_Jp))
                for i in range(d):
                    trunk = ([V if j != i else 0 * V for j, V in enumerate(P.V)], P.H)       # everything but V[i]
                    v = Variables(P, kind, case, variations=trunk, core_directions=None)
                    paired_with_Jp = J.top_pair(seed[..., None], v.Jp[..., None])
                    self.assertGreater(abs(paired_with_Jp), 1e-3)
                    self.assertEqual(J.top_pair(base.xi_tilde[i], v.eta_D_dot[i]), 0.0)
                    self.assertLessEqual(abs(J.top_pair(J.times_seed(seed, base.xi[i]), v.eta_D_dot[i]) - paired_with_Jp),
                                         TOL * abs(paired_with_Jp))

    def test_a_tilde_dot_variable_is_the_derivative_of_its_tilde_with_the_seed_held(self):
        # For every kind (for apply and entries both sides of the first two are zero).
        h = 1e-4
        for labels, P, kind, case in cases(problems=PROBLEMS[:2], jet_cases=(NO_JETS, LINE_3, MIXED_MODES)):
            with self.subTest(**labels):
                sample, args = P.sample(kind, case), P.arguments(kind, case)
                seed = sdef.w_tilde(kind, P.frame, sample, **args)

                def moved(e):                               # the base point (D, L) moved along the raw direction
                    return ([X + e * X_dot for X, X_dot in zip(P.D, P.D_dot)], P.U,
                            [X + e * X_dot for X, X_dot in zip(P.L, P.L_dot)], P.R)
                for name in ('xi_tilde', 'mu_tilde', 'nu_L_tilde', 'eta_D_tilde'):
                    at = lambda e: getattr(sdef, name)(kind, moved(e), sample, order=case.order, w_tilde=seed)
                    quotient = [(8 * (a - b) - (c - d_)) / (12 * h)
                                for a, b, c, d_ in zip(at(h), at(-h), at(2 * h), at(-2 * h))]
                    dotted = getattr(sdef, name + '_dot')(kind, P.frame, sample, order=case.order, w_tilde=seed,
                                                          core_directions=args['core_directions'])
                    for a, b in zip(dotted, quotient):
                        self.assertLessEqual(relative_error(a, b), 1e-8, name + '_dot')


class TestJetsTwoWays(unittest.TestCase):
    """Every variable at a jet shape, against the Taylor expansion of the same variable with no jets at the input
    jets evaluated at ``s``, by interpolation on two real grids that are not the reference's. For the tilde grades
    the no-jet call is given the stored seed as the polynomial ``sum_k w_tilde[k] s^k``: that is what it means that
    they are stored turned. The declared jet sizes are checked exactly."""

    JETTED = tuple(name for name in ALL_NAMES if name in BOND_NAMES + SITE_NAMES + MODE_NAMES
                   and name not in ('V_tilde', 'H_tilde', 'V_tilde_dot', 'H_tilde_dot', 'r', 'w_tilde', 'w'))

    def test_every_variable(self):
        self.assertEqual(len(self.JETTED), 24)              # z, Jp and the 22 edge variables
        # every jet case on three sites; on four (a larger grid, every coefficient nonzero) one variable and two
        matrix = itertools.chain(cases(problems=PROBLEMS[:1], kinds=('probe', 'apply'), jet_cases=WITH_JETS),
                                 cases(problems=PROBLEMS[2:3], kinds=('probe', 'apply'), jet_cases=(LINE_3, MIXED_MODES)))
        for labels, P, kind, case in matrix:
            sample, args = P.sample(kind, case), P.arguments(kind, case)
            seed = sdef.w_tilde(kind, P.frame, sample, **args)
            inputs = clipped_inputs(P, kind, case)
            degrees = degree_bound(P, kind, case, seed=True)      # no variable has a higher degree

            def at(name, s, i):
                seed_s = [evaluate(c, s) for c in seed] if kind == 'probe' else evaluate(seed, s)
                return getattr(sdef, name)(kind, P.frame, [evaluate(x, s) for x in inputs], i,
                                           variations=args['variations'], core_directions=args['core_directions'],
                                           w_tilde=seed_s)

            for name in self.JETTED:
                got = getattr(sdef, name)(kind, P.frame, sample, **args)
                indexed = isinstance(got, tuple)            # not so for z and Jp of apply
                got = list(got) if indexed else [got]
                for i, got_i in enumerate(got):
                    index = i if indexed else None
                    sizes = tuple(inputs[i].shape[:case.m]) if name in LEAF_NAMES else case.box
                    for which in (0, 1):
                        with self.subTest(name=name, i=index, nodes=which, **labels):
                            (expected,) = interpolate(lambda s: [at(name, s, index)], degrees, sizes, which)
                            self.assertEqual(got_i.shape, expected.shape)       # structural: the declared sizes
                            self.assertLessEqual(relative_error(got_i, expected), 1e-8)

    def test_slot_zero_is_the_variable_with_no_jets(self):
        # For the plain and dot grades, and the outputs: slot (0, ..., 0) of the jet at any jet shape is the variable
        # with no jets at the input jets' slot 0 (the point ww). Not so for the tilde grades, whose seed is turned:
        # their slot 0 holds the corner.
        P = small()
        for kind in ('probe', 'apply'):
            args = P.arguments(kind, NO_JETS)
            for case in WITH_JETS:
                for name in NAMES['plain'] + NAMES['dot, frame form'] + NAMES['dot, raw form'] + ('z', 'Jp'):
                    with self.subTest(kind=kind, jets=case.label, name=name):
                        plain = getattr(sdef, name)(kind, P.frame, P.sample(kind, NO_JETS), **args)
                        jet = getattr(sdef, name)(kind, P.frame, P.sample(kind, case), **dict(args, order=case.order))
                        as_list = lambda value: list(value) if isinstance(value, tuple) else [value]
                        self.assertEqual(len(as_list(jet)), len(as_list(plain)))
                        for jet_i, plain_i in zip(as_list(jet), as_list(plain)):
                            self.assertLessEqual(relative_error(jet_i[(0,) * case.m], plain_i), TOL)

    def test_a_second_variable_of_order_zero_is_one_variable(self):
        # The jet shape (n, 0) with inputs of sizes (2, 1) is the jet shape (n,) with the line: the second axis has
        # one slot everywhere, a variable nothing depends on.
        P = small()
        for kind in ('probe', 'apply'):
            line = P.input_jets(LINE_3)
            flat_args = P.arguments(kind, LINE_3)
            with_second = [x[:, None] for x in line]
            y = [y_i[:, None] for y_i in flat_args['y']] if kind == 'probe' else flat_args['y'][:, None]
            weight = flat_args['weight'][..., None]
            args = dict(flat_args, order=(3, 0), y=y, weight=weight)
            for name in ALL_NAMES:
                with self.subTest(kind=kind, name=name):
                    one = getattr(sdef, name)(kind, P.frame, line, **flat_args)
                    two = getattr(sdef, name)(kind, P.frame, with_second, **args)
                    flatten = lambda value: (sum((flatten(v) for v in value), []) if isinstance(value, (tuple, list))
                                             else [np.asarray(value)])
                    for a, b in zip(flatten(one), flatten(two)):
                        if b.ndim == a.ndim + 1:                    # a jet (or a block sum): drop the second jet
                            k = next(k for k in range(b.ndim)       # axis, of one slot
                                     if b.shape[k] == 1 and b.shape[:k] + b.shape[k + 1:] == a.shape)
                            b = np.squeeze(b, axis=k)
                        self.assertEqual(a.shape, b.shape)
                        self.assertLessEqual(relative_error(b, a), TOL)


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

    def jax_problem(self, P, kind, case):
        """The arguments of a case as jax arrays: ``(frame, sample, arguments)``."""
        import jax.numpy as jnp
        to_jax = common.tree_to_jax
        args = P.arguments(kind, case)
        sample = jnp.asarray(P.index) if kind == 'entries' else to_jax(P.input_jets(case))
        return (tuple(to_jax(store) for store in P.frame), sample,
                dict(order=case.order, variations=(to_jax(P.V), to_jax(P.H)),
                     core_directions=(to_jax(P.D_dot), to_jax(P.L_dot)),
                     y=to_jax(args['y']) if kind == 'probe' else jnp.asarray(args['y']),
                     weight=jnp.asarray(args['weight'])))

    def test_jax_against_finite_differences(self):
        # The two routes of Kp: automatic differentiation of the dense objective (jax arrays) and the central
        # finite differences (numpy arrays). jax is put in double precision for the comparison, inside a
        # context, so that no setting of the process changes.
        if not common.jax_available:
            self.skipTest('jax is not installed: Kp is the finite differences alone')
        for labels, P, kind, case in cases(jet_cases=(NO_JETS, LINE_3, MIXED_MODES)):
            with self.subTest(**labels):
                sample, args = P.sample(kind, case), P.arguments(kind, case)
                S = sdef._setting(kind, P.frame, sample, case.order)
                seed = sdef.w_tilde(kind, P.frame, sample, **args)
                by_differences = sdef._Kp_by_finite_differences(S, seed, args['core_directions'])
                with jax_in_double_precision():
                    jax_frame, jax_sample, jax_args = self.jax_problem(P, kind, case)
                    S_jax = sdef._setting(kind, jax_frame, jax_sample, case.order)
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
        P, kind, case = small(), 'probe', LINE_1
        sample, args = P.sample(kind, case), P.arguments(kind, case)
        S = sdef._setting(kind, P.frame, sample, case.order)
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
        jax_frame, jax_sample, jax_args = self.jax_problem(P, kind, case)
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
        for labels, P, kind, case in cases(problems=PROBLEMS[:2], jet_cases=(NO_JETS, LINE_3, PLANE)):
            with self.subTest(**labels):
                sample, args = P.sample(kind, case), P.arguments(kind, case)
                seed = sdef.w_tilde(kind, P.frame, sample, **args)

                def gradient(e):
                    B = [X + e * X_dot for X, X_dot in zip(P.D, P.D_dot)]
                    G = [X + e * X_dot for X, X_dot in zip(P.L, P.L_dot)]
                    return blocks_of(sdef.g(kind, (B, G, G, G), sample, order=case.order, w_tilde=seed))
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
        for labels, P, kind, case in cases(problems=PROBLEMS[:1], jet_cases=(NO_JETS, LINE_3, PLANE)):
            with self.subTest(**labels):
                J = Jets(case.order)
                sample, args = P.sample(kind, case), P.arguments(kind, case)
                r = J.lift(modes_of(kind, sdef.r(kind, P.frame, sample, **args)))
                functional = J.weighted(r, args['weight'], kind)
                rng = np.random.default_rng(5)
                q = [rng.standard_normal(X.shape) for X in P.D + P.L]
                p = P.D_dot + P.L_dot
                x = P.D + P.L

                def phi(a, b):
                    cores = [X + a * Q + b * X_dot for X, Q, X_dot in zip(x, q, p)]
                    T = t3_conversions.t3_to_dense((cores[:P.d], cores[P.d:]))
                    z = J.lift(modes_of(kind, sample_dense_jet(P, kind, T, case)))
                    return sum(np.sum(c * z_i) for c, z_i in zip(functional, z))
                # phi is a polynomial of degree at most 2 d along each of the two: its mixed derivative at 0,
                # exactly, by interpolation in the one and then in the other
                slope = lambda f: np.linalg.solve(np.vander(nodes, increasing=True), [f(t) for t in nodes])[1]
                mixed = slope(lambda a: slope(lambda b: phi(a, b)))
                Kp = blocks_of(sdef.Kp(kind, P.frame, sample, **args))
                paired = sum(np.sum(a * b) for a, b in zip(q, Kp))
                self.assertLessEqual(abs(paired - mixed), 1e-9 * max(abs(mixed), 1.0))

    def test_Kp_is_symmetric_and_linear_in_the_direction(self):
        P, kind, case = small(), 'probe', LINE_1
        sample, args = P.sample(kind, case), P.arguments(kind, case)
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


class TestAgainstTheOldKernels(unittest.TestCase):
    """The bridge to the library as it is today (``sampling_derivatives``, deleted in P20): with one jet variable and
    a line input, ``z`` and ``g`` of the probe kind. The old kernels take ``(ww, pp)`` and work in derivatives, and
    the old transpose takes its residual forward. So, slot by slot: an old ``z`` is ``t!`` times the reference's;
    and the old transpose given ``weight[t]**2 * r[t] / t!``, forward, is the reference's ``g``, whose seed is the
    same residual weighted and turned."""

    def test_z_and_g_of_the_probe_kind(self):
        import t3toolbox.backend.sampling_derivatives as sampling_derivatives
        for name, make in PROBLEMS[:3]:
            P = make()
            line = P.input_jets(LINE_1)
            for n in (1, 2, 3):
                with self.subTest(problem=name, order=n):
                    case = JetCase('a line', (n,), 'line')
                    factorials = np.array([math.factorial(t) for t in range(n + 1)], dtype=float)
                    per_slot = lambda x, c: c.reshape((n + 1,) + (1,) * (x.ndim - 1)) * x

                    z = sdef.z('probe', P.frame, line, order=(n,))
                    old = sampling_derivatives.t3_probe_derivatives(P.ww, P.pp, (P.D, P.L), n)
                    for a, b in zip(z, old):
                        self.assertLessEqual(relative_error(a, per_slot(b, 1.0 / factorials)), TOL)

                    rng = np.random.default_rng(n)
                    y = [rng.standard_normal((n + 1,) + P.W + (N,)) for N in P.shape]
                    weight = 0.5 + rng.random((P.d, n + 1))
                    g = sdef.g('probe', P.frame, line, order=(n,), y=y, weight=weight)
                    r = sdef.r('probe', P.frame, line, order=(n,), y=y)
                    forward = [per_slot(r[i], weight[i] ** 2 / factorials) for i in range(P.d)]
                    old = sampling_derivatives.tv_probe_derivatives_transpose(forward, P.ww, P.pp, P.frame, n,
                                                                              sum_over_probes=True)
                    for a, b in zip(blocks_of(g), blocks_of(old)):
                        self.assertLessEqual(relative_error(a, b), TOL)
                    self.assertEqual(case.m, 1)


class TestCorewiseFrame(unittest.TestCase):
    """SAMPLE_VARIABLES.md section 9: the corewise frame ``(B, G, G, G)`` has one basis on each edge."""

    def test_the_two_bases_are_one(self):
        P = small()
        corewise = (P.D, P.L, P.L, P.L)
        raw = (P.D_dot, P.L_dot)
        for kind in KINDS:
            for case in ((NO_JETS,) if kind == 'entries' else (NO_JETS, LINE_3, MIXED_MODES)):
                sample, args = P.sample(kind, case), P.arguments(kind, case)
                get = lambda name, **more: getattr(sdef, name)(kind, corewise, sample, **dict(args, **more))
                pairs = [('nu', 'nu_L', {}), ('eta', 'eta_D', {}), ('nu_tilde', 'nu_L_tilde', {}),
                         ('eta_tilde', 'eta_D_tilde', {}),
                         # a direction at a corewise point is a raw direction: the frame's dots are the raw ones
                         ('mu_R_dot', 'mu_dot', dict(variations=raw)), ('xi_U_dot', 'xi_dot', dict(variations=raw))]
                for one, other, more in pairs:
                    with self.subTest(kind=kind, jets=case.label, one=one, other=other):
                        for a, b in zip(get(one, **more), get(other, **more)):
                            self.assertLessEqual(relative_error(a, b), TOL)
                for name in ('nu_L_dot', 'eta_D_dot'):      # one name, two forms, the same array here
                    with self.subTest(kind=kind, jets=case.label, name=name):
                        frame_form = get(name, variations=raw, core_directions=None)
                        raw_form = get(name, variations=None, core_directions=raw)
                        for a, b in zip(frame_form, raw_form):
                            self.assertLessEqual(relative_error(a, b), TOL)

    def test_with_both_directions_the_shared_names_are_the_raw_form(self):
        # On a manifold frame the two forms differ; with both arguments given the name means the raw form
        # (the one Kp reads).
        P, kind, case = small(), 'probe', LINE_1
        sample, args = P.sample(kind, case), P.arguments(kind, case)
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
                    ('order', inspect.Parameter.POSITIONAL_OR_KEYWORD, ()),
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
        # PART1_PLAN.md section 2.3, the table of declared jet sizes, and SAMPLE_VARIABLES.md section 7: the input
        # w[i] has its own jet sizes, as given; the leaves (xi, xi_dot, xi_U_dot) the input's clipped to the box F;
        # every other variable F. Then W + (size,); one array per bond (d + 1) or per site (d); the blocks have the
        # shapes of the cores they stand for; the block sums the shape of the weight.
        for labels, P, kind, case in cases(problems=PROBLEMS[:2]):
            sample, args = P.sample(kind, case), P.arguments(kind, case)
            F = case.box
            d = P.d
            given = [tuple(x.shape[:case.m]) for x in P.input_jets(case)] if kind != 'entries' else [()] * d
            for name, size in self.sizes(P).items():
                with self.subTest(name=name, **labels):
                    values = getattr(sdef, name)(kind, P.frame, sample, **args)
                    self.assertIsInstance(values, tuple)
                    self.assertEqual(len(values), d + 1 if name in BOND_NAMES else d)
                    if name == 'w':
                        sizes = given
                    elif name in LEAF_NAMES:
                        sizes = [tuple(min(a, f) for a, f in zip(g, F)) for g in given]
                    else:
                        sizes = [F] * len(values)
                    self.assertEqual([v.shape for v in values],
                                     [sizes[i] + P.W + (size(i),) for i in range(len(values))])
                    for v in values:
                        self.assertIsInstance(v, np.ndarray)
                        self.assertEqual(v.dtype, np.float64)
            with self.subTest(what='outputs and blocks', **labels):
                for name in ('z', 'r', 'Jp', 'w_tilde'):
                    value = getattr(sdef, name)(kind, P.frame, sample, **args)
                    if kind == 'probe':
                        self.assertEqual([v.shape for v in value], [F + P.W + (n,) for n in P.shape])
                    else:
                        self.assertEqual(value.shape, F + P.W)
                self.assertEqual(np.shape(sdef.r_block_sumsq(kind, P.frame, sample, **args)), np.shape(args['weight']))
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
            case = NO_JETS if kind == 'entries' else LINE_1
            sample, args = P.sample(kind, case), P.arguments(kind, case)
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
            sample, args = P.sample(kind, NO_JETS), P.arguments(kind, NO_JETS)
            for name in UNINDEXED_NAMES + (MODE_NAMES if kind != 'probe' else ()):
                with self.subTest(kind=kind, name=name):
                    with self.assertRaises(ValueError):
                        getattr(sdef, name)(kind, P.frame, sample, 0, **args)

    def test_a_missing_argument_raises(self):
        P, kind = small(), 'probe'
        sample, args = P.sample(kind, NO_JETS), P.arguments(kind, NO_JETS)
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
        with self.assertRaisesRegex(ValueError, 'the entries kind has no jets'):
            sdef.mu('entries', P.frame, P.index, order=(1,))                # entries has no jets
        for bad in (1, 0, [1], (1.0,), (True,), (-1,)):                     # the jet shape is a tuple
            with self.subTest(order=bad):
                with self.assertRaises((TypeError, ValueError)):
                    sdef.mu('probe', P.frame, P.input_jets(LINE_1), order=bad)

    def test_the_jet_shape_is_read_as_the_library_reads_it(self):
        # The module reads the jet shape with a function of its own (it shares no code with the library but the
        # dense functions); on every input it agrees with the library's reader: the same result, or the same error.
        inputs = ((), (1,), (3, 0), (np.int64(2), 3), (np.uint8(1),), 3, 0, [1], (1.0,), (True,), (np.bool_(True),),
                  (-1,), (None,), np.array([1]), '1')
        for order in inputs:
            with self.subTest(order=order):
                outcomes = []
                for read in (sdef._read_jet_shape, lambda o: contractions._read_order(o, 'x')):
                    try:
                        value = read(order)
                        outcomes.append(('read', value, tuple(type(n) for n in value)))
                    except (TypeError, ValueError) as error:
                        outcomes.append(type(error))
                self.assertEqual(outcomes[0], outcomes[1])

    def test_sixteen_sites_run(self):
        d = 16
        rng = np.random.default_rng(0)
        D = [rng.standard_normal((1, 2)) for _ in range(d)]
        G = [rng.standard_normal((1, 1, 1)) for _ in range(d)]
        ww = [rng.standard_normal((2, 2)) for _ in range(d)]
        expected = np.prod([G[i][0, 0, 0] * (ww[i] @ D[i][0]) for i in range(d)], axis=0)
        self.assertLessEqual(relative_error(sdef.mu('probe', (D, G, G, G), ww, d)[:, 0], expected), TOL)

    def test_the_input_is_returned_exactly(self):
        # w is the input jet as given, slots beyond the box included (the mode of four slots at (2,)); for entries
        # the unit vectors
        P = small()
        for kind, case in (('probe', LINE_3), ('apply', MIXED_SIZES), ('apply', MIXED_MODES), ('entries', NO_JETS)):
            w = sdef.w(kind, P.frame, P.sample(kind, case), order=case.order)
            expected = P.inputs(kind) if kind == 'entries' else P.input_jets(case)
            with self.subTest(kind=kind, jets=case.label):
                for w_i, x_i in zip(w, expected):
                    self.assertEqual(w_i.shape, x_i.shape)
                    self.assertTrue(np.array_equal(w_i, x_i))

    def test_the_plain_and_dot_variables_do_not_depend_on_the_kind(self):
        # probe and apply take the same inputs and differ only in the outputs and in where the seed goes.
        P = small()
        for case in (NO_JETS, LINE_3, MIXED_MODES):
            args = P.arguments('probe', case)
            for name in NAMES['plain'] + NAMES['dot, frame form'] + NAMES['dot, raw form'] + ('w',):
                with self.subTest(name=name, jets=case.label):
                    for a, b in zip(getattr(sdef, name)('probe', P.frame, P.sample('probe', case), **args),
                                    getattr(sdef, name)('apply', P.frame, P.sample('apply', case), **args)):
                        self.assertTrue(np.array_equal(a, b))

    def test_entries_is_apply_at_unit_vectors(self):
        # with no jets (the entries kind has none): every name equals apply at index_to_one_hot
        P = small()
        args = P.arguments('entries', NO_JETS)
        unit = list(jets.index_to_one_hot(P.index, P.shape))
        for name in ALL_NAMES:
            with self.subTest(name=name):
                a = getattr(sdef, name)('entries', P.frame, P.index, **args)
                b = getattr(sdef, name)('apply', P.frame, unit, **args)
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
        for kind, case in (('probe', LINE_1), ('apply', NO_JETS), ('apply', PLANE), ('entries', NO_JETS)):
            sample, args = P.sample(kind, case), P.arguments(kind, case)
            jax_frame, jax_sample, jax_args = TestKp.jax_problem(None, P, kind, case)
            for name in ALL_NAMES:
                with self.subTest(kind=kind, jets=case.label, name=name):
                    got = flatten(getattr(sdef, name)(kind, jax_frame, jax_sample, **jax_args))
                    expected = flatten(getattr(sdef, name)(kind, P.frame, sample, **args))
                    self.assertEqual(len(got), len(expected))
                    for a, b in zip(got, expected):
                        self.assertTrue(common.is_jax_ndarray(a), type(a))
                        self.assertEqual(tuple(a.shape), tuple(np.shape(b)))
                        self.assertLessEqual(relative_error(np.asarray(a), b), 2e-3)


def jet_by_roots_of_unity(value_at, input_sizes, order):
    """The Taylor coefficients, on the box ``order + 1``, of a polynomial in ``s_1 ... s_m`` given only by its values:
    ``value_at(s)`` (``s`` a tuple of complex numbers) returns a list of arrays. The polynomial is evaluated on a grid of
    roots of unity, ``M_a`` of them along axis ``a``, with ``M_a - 1`` the degree bound along that axis (the sum over
    ``input_sizes`` of ``size_a - 1``: a product of the inputs has at most that degree), and the coefficients come
    back by a discrete Fourier transform along each axis, which is exact for a polynomial of that degree."""
    m = len(order)
    M = [1 + sum(sizes[a] - 1 for sizes in input_sizes) for a in range(m)]
    values = None
    for node in itertools.product(*(range(M_a) for M_a in M)):
        s = tuple(np.exp(2j * np.pi * node[a] / M[a]) for a in range(m))
        value = value_at(s)
        if values is None:
            values = [np.zeros(tuple(M) + v.shape, complex) for v in value]
        for stored, v in zip(values, value):
            stored[node] = v
    result = []
    for stored in values:
        coefficients = np.fft.fftn(stored, axes=tuple(range(m))) / np.prod(M) if m else stored
        box = np.zeros(tuple(n + 1 for n in order) + stored.shape[m:], complex)
        inside = tuple(slice(0, min(n + 1, M_a)) for n, M_a in zip(order, M))
        box[inside] = coefficients[inside]
        assert np.abs(box.imag).max(initial=0.0) <= 1e-9 * max(np.abs(box.real).max(initial=0.0), 1.0)
        result.append(box.real)
    return result


def evaluate_input_jet(w, s):
    """The input vector of the input jet ``w`` (``J_in + W + (N,)``, ``m = len(s)`` jet axes) at ``s``: the sum
    over its slots of the coefficient times ``s^k``."""
    m = len(s)
    value = 0
    for k in itertools.product(*(range(size) for size in w.shape[:m])):
        value = value + w[k] * np.prod([s[a] ** k[a] for a in range(m)])
    return np.asarray(value, dtype=complex) * np.ones(w.shape[m:])


class TestDenseFunctions(unittest.TestCase):
    """``dense_probe``, ``dense_apply``, ``dense_entries``: the dense ground truth of the three kinds.

    With no jets they are checked against hand-written contractions and indexing. Their jets are checked against
    the definition: the same function with no jets, called along the input jets at a grid of roots of unity (the
    input vectors evaluated there), and the coefficients recovered by a discrete Fourier transform. That route
    shares nothing with the functions' own (a truncated product, one mode at a time)."""

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

    def input_jets(self, sizes, W=(2, 2)):
        """Random input jets of the given jet sizes, one per mode: ``sizes[i] + W + (Ni,)``."""
        return [self.rng.standard_normal(tuple(s) + W + (n,)) for s, n in zip(sizes, self.shape)]

    # ---------------------------------------------------------------- no jets: today's values
    def test_dense_probe_is_the_tensor_contracted_with_all_vectors_but_one(self):
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
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(z, probing.dense_probe(T, [x[0, 0] for x in self.ww], ()))))

    def test_dense_probe_with_one_mode(self):
        # No vector is contracted: the probe is the tensor itself, broadcast over W (review 2026-08-22, C1); with
        # jets it is the same at slot 0 and zero above (degree 0 in s, whatever the input jet)
        T = self.rng.standard_normal((2, 6))                                # C = (2,), N0 = 6
        ww = [self.rng.standard_normal((3, 6))]
        (z,) = probing.dense_probe(T, ww)
        self.assertEqual(z.shape, (3, 2, 6))
        self.assertTrue(np.array_equal(z, np.broadcast_to(T, (3, 2, 6))))
        (z,) = probing.dense_probe(T, [self.rng.standard_normal((2, 2, 3, 6))], (2, 1))
        self.assertEqual(z.shape, (3, 2, 3, 2, 6))
        self.assertTrue(np.array_equal(z[0, 0], np.broadcast_to(T, (3, 2, 6))))
        self.assertTrue(np.all(z[1:] == 0) and np.all(z[:, 1:] == 0))

    def test_dense_apply_is_the_probe_contracted_with_the_vector_of_its_mode(self):
        for T, C in ((self.T, ''), (self.TC, 'xy')):
            z = apply.dense_apply(T, self.ww)
            self.assertEqual(z.shape, (2, 2) + T.shape[:-3])                # W + C
            self.check(z, np.einsum(C + 'ijk,abi,abj,abk->ab' + C, T, *self.ww))
            for i, probe in enumerate(probing.dense_probe(T, self.ww)):
                self.check(z, np.einsum('ab...o,abo->ab...', probe, self.ww[i]))

    def test_dense_entries_is_indexing(self):
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

    # ---------------------------------------------------------------- jets: against the definition
    def check_against_roots_of_unity(self, input_sizes, order, T=None, tol=1e-11):
        """dense_probe and dense_apply at the input jets of ``input_sizes`` and the jet shape ``order``, against the
        no-jet calls at the input vectors evaluated on a grid of roots of unity: the shapes exactly, the values to
        ``tol``."""
        T = self.T if T is None else T
        C = T.shape[:T.ndim - 3]
        ww = self.input_jets(input_sizes)
        box = tuple(n + 1 for n in order)
        inputs_at = lambda s: [evaluate_input_jet(w, s) for w in ww]                    # complex input vectors
        probe = probing.dense_probe(T, ww, order)
        expected = jet_by_roots_of_unity(lambda s: list(probing.dense_probe(T, inputs_at(s))), input_sizes, order)
        for i in range(3):
            self.assertEqual(probe[i].shape, box + (2, 2) + C + (self.shape[i],))
            self.check(probe[i], expected[i], tol)
        got = apply.dense_apply(T, ww, order)
        (expected,) = jet_by_roots_of_unity(lambda s: [apply.dense_apply(T, inputs_at(s))], input_sizes, order)
        self.assertEqual(got.shape, box + (2, 2) + C)
        self.check(got, expected, tol)

    def test_a_line_in_one_variable(self):
        for T in (self.T, self.TC):
            for n in (1, 2, 3, 5):
                with self.subTest(C=T.shape[:-3], order=(n,)):
                    self.check_against_roots_of_unity([(2,)] * 3, (n,), T)

    def test_two_directions_in_two_variables(self):
        for T in (self.T, self.TC):
            for order in ((1, 1), (1, 2), (2, 3)):
                with self.subTest(C=T.shape[:-3], order=order):
                    self.check_against_roots_of_unity([(2, 2)] * 3, order, T)

    def test_a_curve_of_three_slots(self):
        for order in ((1,), (2,), (4,)):
            with self.subTest(order=order):
                self.check_against_roots_of_unity([(3,)] * 3, order)

    def test_modes_of_different_input_jet_sizes(self):
        """Each mode with jet sizes of its own, below, at and above the box, and a mode of size 1 on every jet axis
        (a discrete mode, which does not move); and the case of the jet matrix: modes ``(3, 2)``, ``(1, 1)``,
        ``(2, 3)`` at ``(1, 2)`` (the first clipped on its first axis)."""
        for sizes, order in ((((4,), (1,), (2,)), (2,)), (((1,), (3,), (5,)), (3,)), (((1, 1),) * 3, (1, 2)),
                             (((3, 2), (1, 1), (2, 3)), (1, 2)), (((2, 1), (1, 4), (3, 2)), (2, 2)),
                             (((2, 1, 2), (1, 2, 1), (2, 2, 2)), (1, 1, 1))):
            with self.subTest(sizes=sizes, order=order):
                self.check_against_roots_of_unity(sizes, order)

    def test_input_slots_beyond_the_box_change_nothing(self):
        """NaN in every input slot beyond the box: the jets are bit for bit those of the inputs cut to the box."""
        order, box = (1, 2), (2, 3)
        ww = self.input_jets([(4, 3), (2, 5), (3, 4)])
        poisoned = []
        for w in ww:
            x = np.full_like(w, np.nan)
            x[:2, :3] = w[:2, :3]
            poisoned.append(x)
        cut = [w[:2, :3] for w in ww]
        for got, want in ((probing.dense_probe(self.TC, poisoned, order), probing.dense_probe(self.TC, cut, order)),
                          ([apply.dense_apply(self.TC, poisoned, order)], [apply.dense_apply(self.TC, cut, order)])):
            for a, b in zip(got, want):
                self.assertFalse(np.isnan(a).any())
                self.assertTrue(np.array_equal(a, b))
        self.assertEqual(box, apply.dense_apply(self.T, cut, order).shape[:2])

    def test_the_coefficients_are_taylor_coefficients_not_derivatives(self):
        # Slot 2 of the apply along w + s p: the sum over the PAIRS of modes that take p, each pair once (the
        # second derivative is twice that).
        T, (u, v, w), (p, q, r) = self.T, self.ww, self.pp
        slot2 = (np.einsum('ijk,abi,abj,abk->ab', T, p, q, w) + np.einsum('ijk,abi,abj,abk->ab', T, p, v, r)
                 + np.einsum('ijk,abi,abj,abk->ab', T, u, q, r))
        line = [np.stack([w_, p_]) for w_, p_ in zip(self.ww, self.pp)]
        self.check(apply.dense_apply(T, line, (2,))[2], slot2)

    def test_slot_0_is_the_call_with_no_jets_and_the_slots_above_the_degree_are_zero(self):
        line = [np.stack([w, p]) for w, p in zip(self.ww, self.pp)]
        for name, function, degree in (('dense_probe', probing.dense_probe, 2), ('dense_apply', apply.dense_apply, 3)):
            with self.subTest(name=name):
                jet = function(self.TC, line, (degree + 2,))
                plain = function(self.TC, self.ww)
                for jet_i, plain_i in zip(modes_of('probe', jet) if name == 'dense_probe' else [jet],
                                          modes_of('probe', plain) if name == 'dense_probe' else [plain]):
                    self.assertTrue(np.array_equal(jet_i[0], plain_i))
                    self.assertGreater(np.abs(jet_i[degree]).min(), 0)
                    self.assertTrue(np.all(jet_i[degree + 1:] == 0))        # exactly: no term is formed there

    def test_the_jets_of_entries_off_the_grid(self):
        """The recipe of entry 120: the released ``entries_derivatives`` is apply at the one-hot vectors moved
        along ``pp``, in Taylor coefficients, ``dense_entries_derivatives / t!`` (the old module stays until P20)."""
        import t3toolbox.backend.jets as jets
        import t3toolbox.backend.sampling_derivatives as sampling_derivatives
        T = self.T                                  # the old oracle takes one grid point of an unstacked tensor
        for n in (1, 2, 4):
            got = apply.dense_apply(T, jets.affine_jet(jets.index_to_one_hot(self.index, self.shape), self.pp), (n,))
            self.assertEqual((n + 1, 2, 2), got.shape)
            self.assertTrue(np.array_equal(got[0], entries.dense_entries(T, self.index)))
            factorials = np.array([math.factorial(t) for t in range(n + 1)], dtype=float)
            for a, b in itertools.product(range(2), range(2)):
                with self.subTest(order=n, sample=(a, b)):
                    old = sampling_derivatives.dense_entries_derivatives(
                        [int(i) for i in self.index[:, a, b]], [p[a, b] for p in self.pp], T, n)
                    self.check(got[:, a, b], old / factorials)

    # ---------------------------------------------------------------- structure
    def test_the_jet_shape_is_a_tuple(self):
        line = [np.stack([w, p]) for w, p in zip(self.ww, self.pp)]
        for name, function in (('dense_probe', probing.dense_probe), ('dense_apply', apply.dense_apply)):
            with self.subTest(name=name):
                for bad in (2, [2], 2.0):
                    with self.assertRaisesRegex(TypeError, name + ': order is the jet shape'):
                        function(self.T, line, bad)
                with self.assertRaisesRegex(ValueError, name):
                    function(self.T, line, (-1,))
                np.testing.assert_array_equal(function(self.T, line, (np.int64(2),))[0], function(self.T, line, (2,))[0])

    def test_wrong_shapes_raise(self):
        short = [x[..., :-1] for x in self.ww]
        mixed = [self.ww[0][0]] + self.ww[1:]
        line = [np.stack([w, p]) for w, p in zip(self.ww, self.pp)]
        with self.assertRaisesRegex(ValueError, 'dense_probe'):
            probing.dense_probe(self.T, short)                              # a wrong mode size
        with self.assertRaisesRegex(ValueError, 'dense_apply'):
            apply.dense_apply(self.T, mixed)                                # sample stacks that differ
        with self.assertRaisesRegex(ValueError, r'dense_apply: ww\[\d\] has sample stack'):
            apply.dense_apply(self.T, [line[0][0]] + line[1:], (1,))        # one input without its jet axis
        with self.assertRaisesRegex(ValueError, 'dense_probe: ww\\[1\\] has the jet sizes \\(0,\\)'):
            probing.dense_probe(self.T, [line[0], line[1][:0], line[2]], (1,))   # a jet axis of size 0
        with self.assertRaisesRegex(ValueError, 'dense_apply: ww\\[0\\] has 1 axes'):
            apply.dense_apply(self.T, [np.ones(4), np.ones((2, 3)), np.ones((2, 5))], (1, 1))    # too few axes
        with self.assertRaisesRegex(ValueError, 'dense_entries'):
            entries.dense_entries(self.T[0, 0], self.index)                 # more index rows than axes
        with self.assertRaisesRegex(ValueError, 'dense_probe'):
            probing.dense_probe(self.T[0, 0], self.ww)                      # more vectors than axes

    def test_they_are_in_their_modules_public_names(self):
        self.assertIn('dense_probe', probing.__all__)
        self.assertIn('dense_apply', apply.__all__)
        self.assertIn('dense_entries', entries.__all__)
        for function, second in ((probing.dense_probe, 'ww'), (apply.dense_apply, 'ww')):
            parameters = inspect.signature(function).parameters
            self.assertEqual(['T', second, 'order'], list(parameters))
            self.assertEqual((), parameters['order'].default)
        self.assertEqual(['T', 'index'], list(inspect.signature(entries.dense_entries).parameters))

    def test_numpy_in_numpy_out_and_the_dtype_is_kept(self):
        T32 = self.T.astype(np.float32)
        line32 = [np.stack([w, p]).astype(np.float32) for w, p in zip(self.ww, self.pp)]
        plane32 = [np.stack([line32[i], np.zeros_like(line32[i])], axis=1) for i in range(3)]   # jet sizes (2, 2)
        for value in (probing.dense_probe(T32, line32, (2,))[0], apply.dense_apply(T32, line32, (2,)),
                      apply.dense_apply(T32, plane32, (1, 2)), entries.dense_entries(T32, self.index)):
            self.assertIsInstance(value, np.ndarray)
            self.assertEqual(value.dtype, np.float32)

    def test_jax_in_gives_jax_out(self):
        if not common.jax_available:
            self.skipTest('jax is not installed')
        import jax.numpy as jnp
        line = [np.stack([w, p]) for w, p in zip(self.ww, self.pp)]
        T, jline = jnp.asarray(self.T), common.tree_to_jax(line)
        for got, expected in ((probing.dense_probe(T, jline, (2,))[1], probing.dense_probe(self.T, line, (2,))[1]),
                              (apply.dense_apply(T, jline, (2,)), apply.dense_apply(self.T, line, (2,))),
                              (entries.dense_entries(T, jnp.asarray(self.index)), entries.dense_entries(self.T, self.index)),
                              (apply.dense_apply(T, self.ww), apply.dense_apply(self.T, self.ww))):  # mixed: jax wins
            self.assertTrue(common.is_jax_ndarray(got))
            self.check(np.asarray(got), expected, 1e-4)


class TestInputJets(unittest.TestCase):
    """``jets.affine_jet`` and ``jets.index_to_one_hot``: the input jets of a point moved along directions, and the
    one-hot vectors of a grid point."""

    def setUp(self):
        self.rng = np.random.default_rng(1)
        self.shape = (3, 4, 2)
        self.ww = [self.rng.standard_normal((5, n)) for n in self.shape]
        self.directions = [[self.rng.standard_normal((5, n)) for n in self.shape] for _ in range(3)]

    def test_the_slots_of_affine_jet(self):
        import t3toolbox.backend.jets as jets
        for m in (1, 2, 3):
            directions = self.directions[:m]
            jet = jets.affine_jet(self.ww, *directions)
            self.assertIsInstance(jet, tuple)
            for i, x in enumerate(jet):
                with self.subTest(m=m, mode=i):
                    self.assertEqual((2,) * m + (5, self.shape[i]), x.shape)
                    for k in itertools.product(range(2), repeat=m):
                        if sum(k) == 0:
                            self.assertTrue(np.array_equal(x[k], self.ww[i]))               # slot 0: the point
                        elif sum(k) == 1:
                            self.assertTrue(np.array_equal(x[k], directions[k.index(1)][i]))   # slot e_j: p_j
                        else:
                            self.assertTrue(np.all(x[k] == 0))                            # every other slot: zero
        self.assertIs(self.ww, jets.affine_jet(self.ww))                                  # no direction: ww itself
        as_list = jets.affine_jet(list(self.ww), list(self.directions[0]))
        self.assertIsInstance(as_list, tuple)
        nan_point = [np.full_like(w, np.nan) for w in self.ww]                            # a zero slot is made,
        self.assertTrue(np.all(jets.affine_jet(nan_point, *self.directions[:2])[0][1, 1] == 0))   # not 0 * w

    def test_affine_jet_of_packed_vectors_is_packed(self):
        import t3toolbox.backend.jets as jets
        same = [self.rng.standard_normal((4, 5, 3)) for _ in range(3)]                    # (d,) + W + (N,), d = 4
        for m in (1, 2):
            packed = jets.affine_jet(same[0], *same[1:1 + m])
            ragged = jets.affine_jet(list(same[0]), *[list(p) for p in same[1:1 + m]])
            with self.subTest(m=m):
                self.assertEqual((4,) + (2,) * m + (5, 3), packed.shape)
                self.assertTrue(np.array_equal(packed, np.stack(ragged)))
        self.assertIs(same[0], jets.affine_jet(same[0]))

    def test_affine_jet_promotes_and_checks_only_the_shapes(self):
        import t3toolbox.backend.jets as jets
        w32 = [w.astype(np.float32) for w in self.ww]
        self.assertEqual(np.float32, jets.affine_jet(w32, [p.astype(np.float32) for p in self.directions[0]])[0].dtype)
        self.assertEqual(np.float64, jets.affine_jet(w32, self.directions[0])[0].dtype)
        with self.assertRaisesRegex(ValueError, 'affine_jet: direction 0 has the shape'):
            jets.affine_jet(self.ww, [p[0] for p in self.directions[0]])                # a mode of another shape
        with self.assertRaisesRegex(ValueError, 'affine_jet: direction 1 must be a sequence of 3'):
            jets.affine_jet(self.ww, self.directions[0], self.directions[1][:2])        # too few modes
        with self.assertRaisesRegex(ValueError, 'affine_jet: direction 0 must be packed'):
            jets.affine_jet(np.stack([w[:, :2] for w in self.ww]), self.directions[0])  # a tuple beside packed

    def test_index_to_one_hot_is_the_rows_of_the_identity(self):
        import t3toolbox.backend.jets as jets
        index = np.stack([self.rng.integers(-n, n, size=(2, 3)) for n in self.shape])
        e = jets.index_to_one_hot(index, self.shape)
        self.assertIsInstance(e, tuple)
        for i, n in enumerate(self.shape):
            self.assertEqual((2, 3, n), e[i].shape)
            self.assertEqual(np.float64, e[i].dtype)
            self.assertTrue(np.array_equal(e[i], np.eye(n)[index[i]]))                  # a negative index wraps
        # entries is apply at the one-hot vectors, exactly (each product has one nonzero term)
        T = self.rng.standard_normal(self.shape)
        self.assertTrue(np.array_equal(apply.dense_apply(T, e), entries.dense_entries(T, index)))
        # outside [-N, N): a numpy index raises, naming the mode and the values, as numpy's indexing does
        with self.assertRaisesRegex(IndexError, r'index_to_one_hot: mode 1 has the index or indices \[-9, 4\], outside \[-4, 4\)'):
            jets.index_to_one_hot(np.array([[0, 2], [-9, 4], [0, 1]]), self.shape)
        with self.assertRaisesRegex(IndexError, 'mode 0'):
            jets.index_to_one_hot(np.array([[3], [0], [0]]), self.shape)                # N: one past the end
        with self.assertRaisesRegex(IndexError, r'mode 1 has the index or indices \[-5\]'):
            jets.index_to_one_hot(np.array([[0], [-5], [0]]), self.shape)               # -N - 1, alone in its mode
        jets.index_to_one_hot(np.array([[-3, 2], [-4, 3], [-2, 1]]), self.shape)        # the ends of [-N, N): fine
        if common.jax_available:                # a jax index (possibly a tracer) is not checked: zeros out of range
            import jax.numpy as jnp
            e = jets.index_to_one_hot(jnp.asarray(np.array([[5, 0], [-9, 1], [0, 1]])), self.shape)
            self.assertTrue(np.all(np.asarray(e[0][0]) == 0) and np.all(np.asarray(e[1][0]) == 0))
            self.assertTrue(np.array_equal(np.asarray(e[0][1]), np.eye(3)[0]))
        for bad in (np.zeros((2, 3), int), np.zeros((3, 2))):
            with self.assertRaises(ValueError):
                jets.index_to_one_hot(bad, self.shape)                                    # rows; integer dtype


if __name__ == '__main__':
    unittest.main()
