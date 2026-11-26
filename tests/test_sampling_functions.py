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

P12b adds:

- **the corewise tangent** ``t3_*_corewise_tangent``: against a numpy dense oracle (the sum of the ``2d`` dense tensors
  with one core replaced by its direction, through ``dense_*``; exact at every order, mixed input jet sizes), against
  the substitution ``tv_*`` at ``t3_corewise_frame(x)`` (bitwise: the same steps), the adjoint pair with the corewise
  transpose, and a structural step count (each sweep once);
- **the corewise Hessian's** ``sum_over_samples``, and the tangent and the Hessian against automatic differentiation
  in jax (``jax.jvp``, and ``jax.jvp`` of ``jax.grad``), at ``()``, ``(2,)``, ``(1, 1)``, skipped where jax is absent;
- **the uniform twins**: each equals its ragged function on the real parts, with garbage in every padded slot (the
  point, the frame, the directions, the variations and a packed sample's or residual's mode padding), exact zero
  padding on the outputs, and the mirror of the sample's form (ragged in ragged out, packed in packed out) at
  ``()``, ``(2,)`` and ``(1, 1)``.

numpy only, except the automatic-differentiation tests (jax dispatch is ``tests/test_dispatch.py``'s).
"""
import inspect
import unittest

import numpy as np

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.uniform_manifold as ut3m
import t3toolbox.uniform_frame_variations_format as ubv
import t3toolbox.backend.t3_conversions as t3c
import t3toolbox.backend.ut3_operations as ut3o
import t3toolbox.backend.ut3_sampling as ut3s
import t3toolbox.backend.utv_sampling as utvs
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
                'tv_%s_transpose': ['frame', sample, 'c', 'sum_over_samples'],
                't3_%s_corewise_tangent': ['x', sample, 'core_directions'],
                't3_%s_corewise_transpose': ['x', sample, 'c', 'sum_over_samples'],
                't3_%s_corewise_hessian': ['x', sample, 'c', 'core_directions', 'sum_over_samples'],
                'ut3_%s': ['x', sample],
                'utv_%s': ['frame', 'variations', sample],
                'utv_%s_transpose': ['frame', sample, 'c', 'sum_over_samples'],
                'ut3_%s_corewise_tangent': ['x', sample, 'core_directions'],
                'ut3_%s_corewise_transpose': ['x', sample, 'c', 'sum_over_samples'],
                'ut3_%s_corewise_hessian': ['x', sample, 'c', 'core_directions', 'sum_over_samples'],
            }
            for name, params in expected.items():
                with self.subTest(kind=kind, name=name % kind):
                    owner = ut3s if name.startswith('ut3') else utvs if name.startswith('utv') else module
                    sig = inspect.signature(getattr(owner, name % kind))
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


# ------------------------------------------------------------------------------------------------- P12b
ORDERS = {'probe': [(), (2,), (1, 1)], 'apply': [(), (2,), (1, 1)], 'entries': [()]}
JET_SIZES = {(): [(), (), ()], (2,): [(3,), (2,), (1,)], (1, 1): [(2, 1), (1, 2), (2, 2)]}   # mixed, per mode
PAD = dict(N=6, n=4, r=4)
FPAD = dict(N=6, nD=4, nU=4, rL=4, rR=4)
GARBAGE = 1e3


def jet_sample(kind, rng, order, shape=STRUCT[0]):
    """The kind's sample with input jets of mixed sizes per mode (a constant, a line, a full jet)."""
    if kind == 'entries':
        return np.stack([rng.integers(0, N, size=W) for N in shape])
    return [rng.standard_normal(F_in + W + (N,)) for F_in, N in zip(JET_SIZES[order], shape)]


def corrupt(obj):
    """``obj`` (a uniform point, frame or variations) with garbage in every padded slot; the real slots unchanged."""
    scs = obj.supercores
    ind = type(obj)(*([np.ones_like(sc) for sc in scs] + [obj.shape, obj.masks])).apply_masks().supercores
    return type(obj)(*([sc + GARBAGE * (1.0 - i) for sc, i in zip(scs, ind)] + [obj.shape, obj.masks]))


def packed_with_garbage(vectors, N, len_P=0):
    """Pack ragged per-mode vectors to ``N``, with garbage beyond each mode's ``N_i`` (the mode padding)."""
    packed = np.array(ut3o.pack_vectors(vectors, N, len_P=len_P))
    for i, v in enumerate(vectors):
        packed[i, ..., np.shape(v)[-1]:] = GARBAGE
    return packed


def embedded(ragged_blocks, uniform_blocks):
    """(the real-slot indicator, the ragged blocks embedded in zeros) on the uniform blocks' shapes: block ``i`` of
    the ragged pair sits at ``[i]`` of its supercore, from the origin of every axis."""
    ind, emb = [], []
    for rag, uni in zip(ragged_blocks, uniform_blocks):
        m, e = np.zeros(np.shape(uni), bool), np.zeros(np.shape(uni))
        for i, a in enumerate(rag):
            self_check = np.ndim(uni) - 1 == np.ndim(a)
            assert self_check, (np.shape(uni), np.shape(a))
            at = (i,) + tuple(slice(0, n) for n in np.shape(a))
            m[at], e[at] = True, a
        ind.append(m)
        emb.append(e)
    return ind, emb


class TestTheCorewiseTangent(unittest.TestCase):
    """``t3_*_corewise_tangent`` (P12b): ``J x_dot`` at a point along a stack ``K`` of core directions."""

    def setUp(self):
        self.rng = np.random.default_rng(21)
        self.x = point(self.rng, (2,))
        self.dirs = tuple(tuple(self.rng.standard_normal((2,) + np.shape(a)) for a in part) for part in self.x.data)

    def T_dot(self, k):
        """The dense derivative along direction ``k``: the sum of the 2d dense tensors with one core replaced."""
        x, d, total = self.x.data, len(self.x.data[0]), 0.0
        for part in (0, 1):
            for j in range(d):
                cores = [list(x[0]), list(x[1])]
                cores[part][j] = self.dirs[part][j][k]
                total = total + t3c.t3_to_dense((tuple(cores[0]), tuple(cores[1])))
        return total

    def test_the_dense_oracle(self):
        for kind in KINDS:
            for order in ORDERS[kind]:
                with self.subTest(kind=kind, order=order):
                    s = jet_sample(kind, self.rng, order)
                    kw = order_kw(kind, order)
                    J = call(kind, 't3_%s_corewise_tangent', self.x.data, s, self.dirs, **kw)
                    lead = (slice(None),) * (len(order) + len(W))           # F + W, then K
                    for k in range(2):
                        got = [a[lead + (k,)] for a in J] if kind == 'probe' else J[lead + (k,)]
                        self.assertLess(relerr(got, dense_of(kind, self.T_dot(k), s, order)), 1e-12)

    def test_the_substitution_and_the_adjoint_pair(self):
        """The substitution ``tv_*`` at ``t3_corewise_frame(x)`` runs the same steps (bitwise); ``<J x_dot, c> =
        <x_dot, J^T c>`` per element of ``K``."""
        cf = fvc.t3_corewise_frame(self.x.data)
        for kind in KINDS:
            for order in ORDERS[kind]:
                with self.subTest(kind=kind, order=order):
                    s = jet_sample(kind, self.rng, order)
                    kw = order_kw(kind, order)
                    J = call(kind, 't3_%s_corewise_tangent', self.x.data, s, self.dirs, **kw)
                    sub = call(kind, 'tv_%s', cf, self.dirs, s, **kw)
                    self.assertTrue(all(np.array_equal(a, b) for a, b in zip(leaves(J), leaves(sub))))
                    z = call(kind, 't3_%s', self.x.data, s, **kw)
                    c = residual_like(kind, self.rng, z)
                    g = call(kind, 't3_%s_corewise_transpose', self.x.data, s, c, **kw)
                    lead = (slice(None),) * (len(order) + len(W))
                    for k in range(2):
                        Jk = [a[lead + (k,)] for a in J] if kind == 'probe' else J[lead + (k,)]
                        dk = tuple(tuple(a[k] for a in part) for part in self.dirs)
                        self.assertLess(abs(dot(Jk, c) - dot(dk, g)), 1e-12 * max(abs(dot(Jk, c)), 1.0))

    def test_the_step_count(self):
        """Structural: the rows the engine runs for ``Jp`` on a corewise call with ``core_directions``, each once;
        the sweeps through the point's two stores, no frame row. The substitution runs the four-store frame's rows
        (its larger count, the record)."""
        for kind in KINDS:
            with self.subTest(kind=kind):
                rows = sv._closure(kind, ('Jp',), frozenset(), frozenset({'_corewise', 'core_directions'}))
                self.assertEqual(len({rule.name for rule in rows}), len(rows))          # each once
                self.assertFalse(any(rule.name.startswith('_frame.') for rule in rows))
                self.assertFalse(any(set(rule.reads) & {'U', 'R', 'V', 'H'} for rule in rows))
                frame_rows = sv._closure(kind, ('Jp',), frozenset(), frozenset({'_manifold', 'variations'}))
                self.assertLessEqual(len(rows), len(frame_rows))


class TestTheCorewiseHessianFlag(unittest.TestCase):

    def test_unsummed_sums_to_the_default(self):
        rng = np.random.default_rng(22)
        x = point(rng, (2,))
        dirs = tuple(tuple(rng.standard_normal((2,) + np.shape(a)) for a in part) for part in x.data)
        for kind in KINDS:
            for order in ORDERS[kind]:
                with self.subTest(kind=kind, order=order):
                    s = jet_sample(kind, rng, order)
                    kw = order_kw(kind, order)
                    c = residual_like(kind, rng, call(kind, 't3_%s', x.data, s, **kw))
                    H = call(kind, 't3_%s_corewise_hessian', x.data, s, c, dirs, **kw)
                    Hw = call(kind, 't3_%s_corewise_hessian', x.data, s, c, dirs, sum_over_samples=False, **kw)
                    self.assertEqual(np.shape(Hw[0][0])[:len(W) + 1], W + (2,))
                    self.assertLess(relerr([a.sum(0) for a in leaves(Hw)], leaves(H)), 1e-13)


class TestAgainstAutomaticDifferentiation(unittest.TestCase):
    """The tangent against ``jax.jvp`` of ``t3_*``, and the corewise Hessian against ``jax.jvp`` of the gradient of
    ``phi(x) = <c, t3_*(x)>`` (O-22), in float64; skipped where jax is absent."""

    def setUp(self):
        try:
            import jax
        except ImportError:
            self.skipTest('jax is not installed')
        self.jax = jax
        self.before = jax.config.read('jax_enable_x64')
        jax.config.update('jax_enable_x64', True)

    def tearDown(self):
        self.jax.config.update('jax_enable_x64', self.before)

    def test_jvp_and_hvp(self):
        jax = self.jax
        import jax.numpy as jnp
        rng = np.random.default_rng(23)
        x = point(rng)
        xj = jax.tree_util.tree_map(jnp.asarray, x.data)
        for kind in KINDS:
            for order in ORDERS[kind]:
                with self.subTest(kind=kind, order=order):
                    s = jet_sample(kind, rng, order)
                    kw = order_kw(kind, order)
                    x_dot = randn_like(rng, x.data)
                    f = lambda y: call(kind, 't3_%s', y, s, **kw)
                    _, jv = jax.jvp(f, (xj,), (jax.tree_util.tree_map(jnp.asarray, x_dot),))
                    J = call(kind, 't3_%s_corewise_tangent', x.data, s, x_dot, **kw)
                    self.assertLess(relerr(J, jax.tree_util.tree_map(np.asarray, jv)), 1e-12)
                    c = residual_like(kind, rng, f(x.data))
                    phi = lambda y: sum(jnp.vdot(jnp.asarray(a), b)
                                        for a, b in zip(leaves(c), jax.tree_util.tree_leaves(f(y))))
                    _, hv = jax.jvp(jax.grad(phi), (xj,), (jax.tree_util.tree_map(jnp.asarray, x_dot),))
                    H = call(kind, 't3_%s_corewise_hessian', x.data, s, c, x_dot, **kw)
                    self.assertLess(relerr(H, jax.tree_util.tree_map(np.asarray, hv)), 1e-12)


class TestTheUniformTwins(unittest.TestCase):
    """Each twin equals its ragged function on the real parts, whatever the padding holds; its padding is exactly
    zero; a probe output mirrors the sample's form."""

    def setUp(self):
        rng = np.random.default_rng(24)
        self.rng = rng
        self.x = point(rng, (2,))
        self.xu = corrupt(ut3.UniformTuckerTensorTrain.from_t3(self.x, **PAD))
        np.random.seed(25)
        self.dt = t3.TuckerTensorTrain.randn(*STRUCT, stack_shape=(2, 2))           # K + C
        self.du = corrupt(ut3.UniformTuckerTensorTrain.from_t3(self.dt, **PAD))
        self.rf = t3m.MANIFOLD.frame(self.x)
        self.uf = corrupt(ubv.UT3Frame.from_t3frame(self.rf, **FPAD))
        self.rv = t3m.COREWISE.randn(self.rf, stack_shape=(2,)).variations
        self.uv = corrupt(ubv.UT3Variations.from_t3variations(self.rv, **FPAD))
        self.N = PAD['N']

    def assert_vectors(self, kind, got, ref, packed):
        """A twin's ``z``-typed output against the ragged one: probe mirrors (a tuple, or packed with zero padding)."""
        if kind != 'probe':
            self.assertLess(relerr(got, ref), 1e-12)
            return
        if not packed:
            self.assertIsInstance(got, tuple)
            self.assertEqual([np.shape(a) for a in got], [np.shape(a) for a in ref])
            self.assertLess(relerr(got, ref), 1e-12)
            return
        self.assertEqual(np.shape(got), (len(ref),) + np.shape(ref[0])[:-1] + (self.N,))
        for i, a in enumerate(ref):
            n = np.shape(a)[-1]
            self.assertLess(relerr(np.asarray(got)[i, ..., :n], a), 1e-12)
            self.assertTrue(np.all(np.asarray(got)[i, ..., n:] == 0))

    def assert_blocks(self, got, ref):
        ind, emb = embedded(ref, got[:2])
        for o, m, e in zip(got[:2], ind, emb):
            o = np.asarray(o)
            self.assertTrue(np.all(o[~m] == 0), 'padding not zero')
            self.assertLess(relerr(o[m], e[m]), 1e-12)

    def samples(self, kind, order):
        s = jet_sample(kind, self.rng, order)
        yield 'ragged', s, False
        if kind != 'entries':
            yield 'packed', packed_with_garbage(s, self.N, len(order)), True

    def test_the_point_twins(self):
        x, xu, dirs = self.x.data, self.xu.data, self.dt.data
        for kind in KINDS:
            for order in ORDERS[kind]:
                kw = order_kw(kind, order)
                for form, s, packed in self.samples(kind, order):
                    with self.subTest(kind=kind, order=order, sample=form):
                        base = s if not packed else self._ragged_of(s, order)
                        z = call(kind, 't3_%s', x, base, **kw)
                        self.assert_vectors(kind, getattr(ut3s, 'ut3_%s' % kind)(xu, s, **kw), z, packed)
                        J = call(kind, 't3_%s_corewise_tangent', x, base, dirs, **kw)
                        self.assert_vectors(kind, getattr(ut3s, 'ut3_%s_corewise_tangent' % kind)(xu, s, self.du.data[:2], **kw),
                                            J, packed)
                        c = residual_like(kind, self.rng, z)
                        cu = packed_with_garbage(c, self.N) if (packed and kind == 'probe') else c
                        for sos in (True, False):
                            g = call(kind, 't3_%s_corewise_transpose', x, base, c, sum_over_samples=sos, **kw)
                            self.assert_blocks(getattr(ut3s, 'ut3_%s_corewise_transpose' % kind)(
                                xu, s, cu, sum_over_samples=sos, **kw), g)
                            Kp = call(kind, 't3_%s_corewise_hessian', x, base, c, dirs, sum_over_samples=sos, **kw)
                            self.assert_blocks(getattr(ut3s, 'ut3_%s_corewise_hessian' % kind)(
                                xu, s, cu, self.du.data[:2], sum_over_samples=sos, **kw), Kp)

    def test_the_tangent_twins(self):
        rf, uf, rv, uv = self.rf.data, self.uf.data, self.rv.data, self.uv.data
        for kind in KINDS:
            for order in ORDERS[kind]:
                kw = order_kw(kind, order)
                for form, s, packed in self.samples(kind, order):
                    with self.subTest(kind=kind, order=order, sample=form):
                        base = s if not packed else self._ragged_of(s, order)
                        Jp = call(kind, 'tv_%s', rf, rv, base, **kw)
                        self.assert_vectors(kind, getattr(utvs, 'utv_%s' % kind)(uf, uv, s, **kw), Jp, packed)
                        c = residual_like(kind, self.rng, Jp)                       # with the stack K of rv
                        cu = packed_with_garbage(c, self.N) if (packed and kind == 'probe') else c
                        for sos in (True, False):
                            g = call(kind, 'tv_%s_transpose', rf, base, c, sum_over_samples=sos, **kw)
                            self.assert_blocks(getattr(utvs, 'utv_%s_transpose' % kind)(
                                uf, s, cu, sum_over_samples=sos, **kw), g)

    def _ragged_of(self, packed, order):
        """The ragged sample a packed one was made from: each mode sliced to its ``N_i`` (the jet zeros are
        coefficients, so the polynomial is the same)."""
        return [np.asarray(packed)[i, ..., :N] for i, N in enumerate(STRUCT[0])]
