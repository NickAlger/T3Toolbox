# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/T3Toolbox
"""The ragged backend sampling functions on the engine (graded messages, P12a), and the engine's addition for them.

- **the engine's second route to** ``JtJp``, from a held ``Jp`` with no direction: it is ``g`` from the same vector
  held as ``r`` (``K = ()``), the per-element ``g`` for a vector with a stack ``K``, the same steps as ``g``; a
  direction, when given, wins and its ``K`` must agree; a call with neither still names ``JtJp``'s missing argument;
- **the ragged mode size**: a ragged ``ww[i]`` whose mode axis is not ``N_i`` raises (a size-1 axis would broadcast);
- **the core sizes of the directions** (round 1): ``V``, ``H``, ``Vc``, ``D_dot``, ``L_dot`` with any core axis cut
  to size 1 raise, naming the operand, on a manifold frame and a corewise point, ragged and uniform;
- **a residual with a stack** ``K``: every transpose, at ``K = (2,)`` and ``(2, 3)``, summed or not, is the call on
  each element's residual, and each element satisfies its own adjoint identity;
- **the functions by definition**: ``t3_*`` and ``tv_*`` against the dense functions; the tangent and corewise
  transposes by the adjoint identity; the corewise transposes as gradients and the corewise Hessians as second
  derivatives, by central differences (automatic differentiation is P12b's); the defaults (a transpose sums the
  samples, an ambient transpose does not) and the signatures (entries has no ``order``).

numpy only (jax dispatch is ``tests/test_dispatch.py``'s).
"""
import inspect
import unittest

import numpy as np

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.uniform_manifold as ut3m
import t3toolbox.frame_variations_format as bvf
import t3toolbox.manifold as t3m
import t3toolbox.backend.fv_conversions as fvc
import t3toolbox.backend.jets as jets
import t3toolbox.backend.sampling_variables as sv
from t3toolbox.backend import apply as bapply
from t3toolbox.backend import entries as bentries
from t3toolbox.backend import probing

STRUCT = ((4, 5, 3), (2, 3, 2), (1, 2, 2, 1))
W = (3,)
KINDS = ('probe', 'apply', 'entries')
MODULES = {'probe': probing, 'apply': bapply, 'entries': bentries}


def leaves(x):
    """Every array of a nested tuple / list, flattened."""
    if isinstance(x, (tuple, list)):
        return [a for item in x for a in leaves(item)]
    return [np.asarray(x)]


def dot(a, b):
    return float(sum(np.vdot(p, q) for p, q in zip(leaves(a), leaves(b))))


def relerr(a, b):
    num = np.sqrt(sum(np.sum((p - q) ** 2) for p, q in zip(leaves(a), leaves(b))))
    den = np.sqrt(sum(np.sum(q ** 2) for q in leaves(b)))
    return float(num / max(den, 1e-300))


def add(a, b, s=1.0):
    if isinstance(a, (tuple, list)):
        return type(a)(add(p, q, s) for p, q in zip(a, b))
    return a + s * b


def randn_like(rng, x):
    if isinstance(x, (tuple, list)):
        return type(x)(randn_like(rng, item) for item in x)
    return rng.standard_normal(np.shape(x))


def sample_of(kind, rng, shape=STRUCT[0], order=()):
    """The kind's sample: ww (input jets of the given jet shape) or an index grid."""
    if kind == 'entries':
        return np.stack([rng.integers(0, N, size=W) for N in shape])
    if order == ():
        return [rng.standard_normal(W + (N,)) for N in shape]
    return list(jets.affine_jet([rng.standard_normal(W + (N,)) for N in shape],
                                [rng.standard_normal(W + (N,)) for N in shape]))


def dense_of(kind, T, sample, order=()):
    if kind == 'probe':
        return probing.dense_probe(T, sample, order)
    if kind == 'apply':
        return bapply.dense_apply(T, sample, order)
    return bentries.dense_entries(T, sample)


def order_kw(kind, order):
    return {} if kind == 'entries' else {'order': order}


def residual_like(kind, rng, z):
    return [rng.standard_normal(np.shape(a)) for a in z] if kind == 'probe' else rng.standard_normal(np.shape(z))


def point(rng, stack_shape=()):
    np.random.seed(int(rng.integers(0, 2 ** 31)))
    return t3.TuckerTensorTrain.randn(*STRUCT, stack_shape=stack_shape)


def call(kind, name, *args, **kwargs):
    return getattr(MODULES[kind], name % kind)(*args, **kwargs)


class TestTheHeldJpRoute(unittest.TestCase):
    """The engine's second ``JtJp`` route (P12): a held ``Jp`` with no direction."""

    def setUp(self):
        rng = np.random.default_rng(0)
        x = point(rng, (2,))
        self.frames = {'manifold': t3m.MANIFOLD.frame(x).data, 'corewise': x.data}
        self.samples = {kind: sample_of(kind, rng) for kind in KINDS}
        self.rng = rng

    def residual(self, kind, frame, K=()):
        (z,) = sv.sampling_variables(kind, frame, self.samples[kind], ('z',))
        lead = len(W)
        if kind == 'probe':
            return [self.rng.standard_normal(a.shape[:lead] + K + a.shape[lead:]) for a in z]
        return self.rng.standard_normal(z.shape[:lead] + K + z.shape[lead:])

    def test_JtJp_from_a_held_Jp_is_g_from_a_held_r(self):
        for kind in KINDS:
            for call_type, frame in self.frames.items():
                with self.subTest(kind=kind, call=call_type):
                    c = self.residual(kind, frame)
                    (JtJp,) = sv.sampling_variables(kind, frame, self.samples[kind], ('JtJp',),
                                                    held=sv.HeldVariables(Jp=c))
                    (g,) = sv.sampling_variables(kind, frame, self.samples[kind], ('g',), held=sv.HeldVariables(r=c))
                    self.assertLess(relerr(JtJp, g), 1e-14)

    def test_a_residual_with_a_stack_K_is_the_per_element_g(self):
        K = (2,)
        for kind in KINDS:
            for call_type, frame in self.frames.items():
                with self.subTest(kind=kind, call=call_type):
                    c = self.residual(kind, frame, K)
                    (JtJp,) = sv.sampling_variables(kind, frame, self.samples[kind], ('JtJp',),
                                                    held=sv.HeldVariables(Jp=c))
                    for k in range(K[0]):
                        ck = [a[:, k] for a in c] if kind == 'probe' else c[:, k]
                        (g,) = sv.sampling_variables(kind, frame, self.samples[kind], ('g',),
                                                     held=sv.HeldVariables(r=ck))
                        self.assertLess(relerr([B[k] for B in JtJp[0]] + [G[k] for G in JtJp[1]],
                                               list(g[0]) + list(g[1])), 1e-14)

    def test_a_direction_wins_and_its_K_must_agree(self):
        frame = self.frames['manifold']
        for kind in KINDS:
            with self.subTest(kind=kind):
                c = self.residual(kind, frame, (2,))
                held = sv.HeldVariables(Jp=c)
                (alone,) = sv.sampling_variables(kind, frame, self.samples[kind], ('JtJp',), held=held)
                np.random.seed(5)
                v2 = t3m.COREWISE.randn(bvf.T3Frame(*frame), stack_shape=(2,)).variations.data
                (both,) = sv.sampling_variables(kind, frame, self.samples[kind], ('JtJp',), held=held, variations=v2)
                self.assertLess(relerr(both, alone), 1e-14)                 # J^T of the held Jp either way
                v3 = t3m.COREWISE.randn(bvf.T3Frame(*frame), stack_shape=(3,)).variations.data
                with self.assertRaisesRegex(ValueError, r'the held Jp has the stack of directions K = \(2,\), where '
                                                        r'the directions give K = \(3,\)'):
                    sv.sampling_variables(kind, frame, self.samples[kind], ('JtJp',), held=held, variations=v3)

    def test_a_call_with_neither_names_the_missing_direction(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                with self.assertRaisesRegex(ValueError, "'JtJp' needs the argument variations, which was not given"):
                    sv.sampling_variables(kind, self.frames['manifold'], self.samples[kind], ('JtJp',))

    def test_the_step_count_is_g_s(self):
        """Structural: from a held ``Jp`` the engine runs the twins of exactly the rows ``g`` runs from a held ``r``,
        each once, and no direction row."""
        for kind in KINDS:
            for call_type in ('_manifold', '_corewise'):
                with self.subTest(kind=kind, call=call_type):
                    jtjp = sv._closure(kind, ('JtJp',), frozenset({'Jp'}), frozenset({call_type, sv._HELD_JP}))
                    g = sv._closure(kind, ('g',), frozenset({'r'}), frozenset({call_type}))
                    self.assertEqual([rule.step for rule in jtjp], [rule.step for rule in g])
                    self.assertEqual(len({rule.name for rule in jtjp}), len(jtjp))
                    self.assertFalse(any(rule.needs & {'variations', 'core_directions'} for rule in jtjp))


class TestTheRaggedModeSize(unittest.TestCase):

    def test_a_size_one_mode_axis_raises(self):
        """A ragged ``ww[i]`` of mode size 1 against ``N_i > 1`` would broadcast; the engine refuses it (P12)."""
        rng = np.random.default_rng(2)
        x = point(rng)
        for kind in ('probe', 'apply'):
            ww = sample_of(kind, rng)
            for i in range(len(ww)):
                bad = list(ww)
                bad[i] = ww[i][..., :1]
                with self.subTest(kind=kind, i=i):
                    with self.assertRaisesRegex(ValueError, r'w\[%d\] has shape \(3, 1\), where N_%d = %d'
                                                % (i, i, STRUCT[0][i])):
                        call(kind, 't3_%s', x.data, bad)
                    with self.assertRaisesRegex(ValueError, r'mode size N_i'):
                        call(kind, 'tv_%s_transpose', t3m.MANIFOLD.frame(x).data, bad,
                             residual_like(kind, rng, call(kind, 't3_%s', x.data, ww)))


class TestTheCoreSizesOfTheDirections(unittest.TestCase):
    """Round 1: a direction (or ``Vc``) whose core axis has size 1 where the stores give more would broadcast in a
    contraction and give a wrong answer with no error; the engine refuses it, naming the operand. Each operand alone,
    every core, every axis whose size is more than 1."""

    def setUp(self):
        rng = np.random.default_rng(11)
        x = point(rng, (2,))
        self.rng, self.x = rng, x
        self.ww = sample_of('apply', rng)

    @staticmethod
    def operands(frame, corewise, uniform, rng):
        """The good operands of a call: (name, the argument it is part of, the cores)."""
        D, L = frame[0], frame[1] if corewise else frame[2]
        if uniform:
            like = lambda A: rng.standard_normal(np.shape(A))
            out = [('D_dot', like(D)), ('L_dot', like(L))]
            if not corewise:
                U, R = frame[1], frame[3]
                C = np.shape(D)[1:-2]
                out += [('V', rng.standard_normal(np.shape(D)[:1] + C + (np.shape(U)[-2], np.shape(D)[-1]))),
                        ('H', rng.standard_normal(np.shape(D)[:1] + C + (np.shape(L)[-3], np.shape(D)[-2],
                                                                         np.shape(R)[-1])))]
                out.append(('Vc', out[2][1]))
            return out
        out = [('D_dot', tuple(rng.standard_normal(np.shape(B)) for B in D)),
               ('L_dot', tuple(rng.standard_normal(np.shape(G)) for G in L))]
        if not corewise:
            U, R = frame[1], frame[3]
            out += [('V', tuple(rng.standard_normal(np.shape(B)[:-2] + (np.shape(G)[-2], np.shape(B)[-1]))
                                for B, G in zip(D, U))),
                    ('H', tuple(rng.standard_normal(np.shape(G)[:-1] + (np.shape(Q)[-1],))
                                for G, Q in zip(L, R)))]
            out.append(('Vc', out[2][1]))
        return out

    def call(self, frame, sample, corewise, cores, name, value):
        given = dict(cores)
        given[name] = value
        kw = dict(core_directions=(given['D_dot'], given['L_dot']))
        if not corewise:
            kw.update(variations=(given['V'], given['H']), Vc=given['Vc'])
        return sv.sampling_variables('apply', frame, sample, ('z',), **kw)

    def test_a_core_axis_of_size_one_raises(self):
        layouts = {'ragged': (self.x.data, t3m.MANIFOLD.frame(self.x).data)}
        xu = ut3.UniformTuckerTensorTrain.from_t3(self.x, N=6, n=4, r=4)
        layouts['uniform'] = (xu.data, ut3m.UNIFORM_MANIFOLD.frame(xu).data)
        for layout, (point_data, frame_data) in layouts.items():
            for corewise, frame in ((True, point_data), (False, frame_data)):
                cores = dict(self.operands(frame, corewise, layout == 'uniform', self.rng))
                self.call(frame, self.ww, corewise, cores, 'D_dot', cores['D_dot'])         # the good call runs
                for name, good in cores.items():
                    seen = 0
                    for i in range(1 if layout == 'uniform' else len(good)):
                        core = good if layout == 'uniform' else good[i]
                        for axis in range(np.ndim(core) - (3 if name in ('H', 'L_dot') else 2), np.ndim(core)):
                            if np.shape(core)[axis] == 1:
                                continue
                            cut = np.take(core, [0], axis=axis)
                            bad = cut if layout == 'uniform' else good[:i] + (cut,) + good[i + 1:]
                            with self.subTest(layout=layout, corewise=corewise, operand=name, i=i, axis=axis):
                                with self.assertRaisesRegex(ValueError, r'core sizes of the frame: %s\[%d\] has shape'
                                                            % (name, i)):
                                    self.call(frame, self.ww, corewise, cores, name, bad)
                            seen += 1
                    self.assertGreater(seen, 0)


class TestAResidualWithAStack(unittest.TestCase):
    """Round 1: every transpose with a residual carrying a stack ``K`` (after ``W``) is the call on each element's
    residual, summed over the samples or not; and each element satisfies its own adjoint identity."""

    def test_per_element_and_adjoint(self):
        rng = np.random.default_rng(12)
        x = point(rng, (2,))
        frame = t3m.MANIFOLD.frame(x).data
        cf = fvc.t3_corewise_frame(x.data)
        np.random.seed(13)
        v = t3m.COREWISE.randn(bvf.T3Frame(*frame)).variations.data
        x_dot = randn_like(rng, x.data)
        lead = len(W)
        for kind in KINDS:
            s = sample_of(kind, rng)
            transposes = {
                'tv': (lambda c, sos: call(kind, 'tv_%s_transpose', frame, s, c, sum_over_samples=sos),
                       call(kind, 'tv_%s', frame, v, s), v),
                'corewise': (lambda c, sos: call(kind, 't3_%s_corewise_transpose', x.data, s, c, sum_over_samples=sos),
                             call(kind, 'tv_%s', cf, x_dot, s), x_dot),
            }
            for which, (transpose, Jp, p) in transposes.items():
                for K in [(2,), (2, 3)]:
                    with_K = lambda a: rng.standard_normal(a.shape[:lead] + K + a.shape[lead:])
                    c = [with_K(a) for a in Jp] if kind == 'probe' else with_K(Jp)
                    for sos in (True, False):
                        out = transpose(c, sos)
                        for k in np.ndindex(*K):
                            ck = [a[(slice(None),) * lead + k] for a in c] if kind == 'probe' \
                                else c[(slice(None),) * lead + k]
                            at = (slice(None),) * (0 if sos else lead) + k
                            got = [a[at] for a in leaves(out)]
                            with self.subTest(kind=kind, transpose=which, K=K, sum_over_samples=sos, k=k):
                                self.assertLess(relerr(got, leaves(transpose(ck, sos))), 1e-13)
                                if sos:
                                    lhs = dot(Jp, ck)
                                    self.assertLess(abs(lhs - dot(p, got)), 1e-12 * max(abs(lhs), 1.0))


class TestTheRaggedFunctions(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(3)

    def test_the_point_values_by_definition(self):
        for C in [(), (2,)]:
            x = point(self.rng, C)
            T = x.to_dense()
            for kind in KINDS:
                for order in ([(), (2,), (1, 1)] if kind != 'entries' else [()]):
                    with self.subTest(C=C, kind=kind, order=order):
                        s = sample_of(kind, self.rng, order=order)
                        got = call(kind, 't3_%s', x.data, s, **order_kw(kind, order))
                        self.assertLess(relerr(got, dense_of(kind, T, s, order)), 1e-12)

    def test_the_tangent_values_by_definition(self):
        x = point(self.rng, (2,))
        frame = t3m.MANIFOLD.frame(x)
        np.random.seed(7)
        v = t3m.COREWISE.randn(frame, stack_shape=(3,))
        T = v.to_dense()                                                    # K+C+shape
        for kind in KINDS:
            for order in ([(), (2,)] if kind != 'entries' else [()]):
                with self.subTest(kind=kind, order=order):
                    s = sample_of(kind, self.rng, order=order)
                    got = call(kind, 'tv_%s', frame.data, v.variations.data, s, **order_kw(kind, order))
                    self.assertLess(relerr(got, dense_of(kind, T, s, order)), 1e-12)

    def test_the_tangent_transposes_are_adjoint(self):
        """``<J v, c> = <v, J^T c>`` per element of a stack ``K`` of residuals; the default sums the samples."""
        x = point(self.rng)
        frame = t3m.MANIFOLD.frame(x).data
        np.random.seed(8)
        v = t3m.COREWISE.randn(bvf.T3Frame(*frame)).variations.data
        for kind in KINDS:
            for order in ([(), (2,)] if kind != 'entries' else [()]):
                with self.subTest(kind=kind, order=order):
                    s = sample_of(kind, self.rng, order=order)
                    Jv = call(kind, 'tv_%s', frame, v, s, **order_kw(kind, order))
                    c = residual_like(kind, self.rng, Jv)
                    g = call(kind, 'tv_%s_transpose', frame, s, c, **order_kw(kind, order))
                    self.assertLess(abs(dot(Jv, c) - dot(v, g)), 1e-12 * max(abs(dot(Jv, c)), 1.0))
                    unsummed = call(kind, 'tv_%s_transpose', frame, s, c, sum_over_samples=False,
                                    **order_kw(kind, order))
                    self.assertEqual(np.shape(unsummed[0][0])[:len(W)], W)
                    self.assertLess(relerr([a.sum(0) for a in leaves(unsummed)], leaves(g)), 1e-13)

    def test_the_corewise_transposes_are_adjoint_and_gradients(self):
        x = point(self.rng)
        cf = fvc.t3_corewise_frame(x.data)
        x_dot = randn_like(self.rng, x.data)
        h = 1e-5
        for kind in KINDS:
            for order in ([(), (2,)] if kind != 'entries' else [()]):
                with self.subTest(kind=kind, order=order):
                    s = sample_of(kind, self.rng, order=order)
                    kw = order_kw(kind, order)
                    Jx = call(kind, 'tv_%s', cf, x_dot, s, **kw)          # J x_dot: the corewise substitution
                    c = residual_like(kind, self.rng, Jx)
                    grad = call(kind, 't3_%s_corewise_transpose', x.data, s, c, **kw)
                    self.assertLess(abs(dot(Jx, c) - dot(x_dot, grad)), 1e-12 * max(abs(dot(Jx, c)), 1.0))
                    unsummed = call(kind, 't3_%s_corewise_transpose', x.data, s, c, sum_over_samples=False, **kw)
                    self.assertEqual(np.shape(unsummed[0][0])[:len(W)], W)
                    self.assertLess(relerr([a.sum(0) for a in leaves(unsummed)], leaves(grad)), 1e-13)
                    phi =lambda y: dot(c, call(kind, 't3_%s', y, s, **kw))
                    fd = (phi(add(x.data, x_dot, h)) - phi(add(x.data, x_dot, -h))) / (2 * h)
                    self.assertLess(abs(fd - dot(x_dot, grad)), 1e-7 * max(abs(fd), 1.0))

    def test_the_corewise_hessians_by_central_differences(self):
        """``K(x) x_dot`` is the derivative of the corewise gradient along ``x_dot``; a stack ``K`` of directions."""
        x = point(self.rng)
        h = 1e-5
        for kind in KINDS:
            for order in ([(), (2,)] if kind != 'entries' else [()]):
                with self.subTest(kind=kind, order=order):
                    s = sample_of(kind, self.rng, order=order)
                    kw = order_kw(kind, order)
                    c = residual_like(kind, self.rng, call(kind, 't3_%s', x.data, s, **kw))
                    dirs = tuple(tuple(self.rng.standard_normal((2,) + np.shape(a)) for a in part) for part in x.data)
                    got = call(kind, 't3_%s_corewise_hessian', x.data, s, c, dirs, **kw)
                    for k in range(2):
                        dk = tuple(tuple(a[k] for a in part) for part in dirs)
                        grad = lambda y: call(kind, 't3_%s_corewise_transpose', y, s, c, **kw)
                        fd = [(p - q) / (2 * h) for p, q in zip(leaves(grad(add(x.data, dk, h))),
                                                                  leaves(grad(add(x.data, dk, -h))))]
                        self.assertLess(relerr([a[k] for a in leaves(got)], fd), 1e-7)

    def test_the_ambient_transposes_by_definition(self):
        """``<J^T c, T> = <c, J T>`` on dense tensors; the default does not sum the samples (each sample its own
        CP term), ``sum_over_samples=True`` is the sum."""
        T = self.rng.standard_normal(STRUCT[0])
        cp_dense = lambda factors: np.einsum('...ra,...rb,...rc->...abc', *factors)
        for kind in KINDS:
            with self.subTest(kind=kind):
                s = sample_of(kind, self.rng)
                c = residual_like(kind, self.rng, dense_of(kind, T, s))
                if kind == 'entries':
                    summed = bentries.t3_entries_ambient_transpose(s, STRUCT[0], c, sum_over_samples=True)
                    each = bentries.t3_entries_ambient_transpose(s, STRUCT[0], c)
                else:
                    summed = call(kind, 't3_%s_ambient_transpose', s, c, sum_over_samples=True)
                    each = call(kind, 't3_%s_ambient_transpose', s, c)
                A = cp_dense(summed)
                self.assertLess(abs(np.vdot(A, T) - dot(c, dense_of(kind, T, s))), 1e-12 * np.abs(A).sum())
                self.assertEqual(np.shape(each[0])[:len(W)], W)
                self.assertLess(relerr(cp_dense(each).sum(0), A), 1e-13)

    def test_the_signatures(self):
        """The object first, then the sample, the residual, the directions; ``order`` keyword-defaulted ``()``
        (none for entries); the transposes sum by default, the ambient transposes do not."""
        for kind in KINDS:
            module = MODULES[kind]
            sample = 'index' if kind == 'entries' else 'ww'
            expected = {
                't3_%s': ['x', sample],
                'tv_%s': ['frame', 'variations', sample],
                'tv_%s_transpose': ['frame', sample, 'residual', 'sum_over_samples'],
                't3_%s_corewise_transpose': ['x', sample, 'residual', 'sum_over_samples'],
                't3_%s_corewise_hessian': ['x', sample, 'residual', 'core_directions'],
            }
            for name, params in expected.items():
                with self.subTest(kind=kind, name=name % kind):
                    sig = inspect.signature(getattr(module, name % kind))
                    names = [p for p in sig.parameters if p != 'order']
                    self.assertEqual(names, params)
                    self.assertEqual('order' in sig.parameters, kind != 'entries')
                    if kind != 'entries':
                        self.assertEqual(sig.parameters['order'].default, ())
                    if 'sum_over_samples' in sig.parameters:
                        self.assertIs(sig.parameters['sum_over_samples'].default, True)
            ambient = inspect.signature(getattr(module, 't3_%s_ambient_transpose' % kind))
            self.assertIs(ambient.parameters['sum_over_samples'].default, False)


if __name__ == '__main__':
    unittest.main()
