# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/T3Toolbox
"""The promoted oracle sweeps (Phase D of the 2026-08-22 review) -- the review's O1 (sampling/tangent:
dense einsum + exact combinatorial jet oracles + adjoint identities, ragged and uniform incl.
varying-rank stacks) and O2 (fitting: polynomial-interpolation oracle vs GaussNewtonModel; optimizer
trajectories vs hand recomputation, ragged == uniform) sweeps, run from verbatim copies in
``tests/oracle_sweeps/``.

TWO TIERS (Nick's ruling, 2026-08-25: no coverage is dropped from what gates a release):

- ALWAYS-ON: the full O1 sweep, the full O2 model sweep, and a 2-way-covering subset of the O2
  optimizer sweep (rule below).
- SLOW (set ``T3TOOLBOX_SLOW_TESTS=1``): the FULL optimizer matrix -- required at every release
  (the tag checklist) in addition to the always-on tier. Nothing the review matrix covered ever
  stops being enforced; only higher-order optimizer-trajectory crossings move off the per-commit path.

Optimizer subset rule (pairwise coverage of the review's bug distribution -- 2-way interactions +
degeneracy classes): keep a case iff (a) base options at d=3 (every geometry x kind pair), or
(b) manifold-unshared at both d (every kind x order x weight x regularizer pair and every d x factor
pair), or (c) probe_derivatives at d=3 (every geometry x weight/regularizer pair on the richest kind).

EXCLUDED ops (harness defect, not library defect): the six ``u_*_corewise[_derivatives]_transpose
(_vs_ragged)`` checks were broken in the review harness itself -- their real-block slicing uses the
structure tuple's ranks, which ``share()``-entered and padded objects outgrow, and the ARCHIVED review
results carry the same EXC rows, so these checks never delivered coverage. The ops themselves have
dedicated per-element uniform-vs-ragged tests (``test_uniform_manifold``, ``test_uniform_probing``)
and jit entries in ``test_dispatch``. Repairing the harness comparison is a recorded follow-up.

KNOWN CAVEAT rows (documented, not failures): ``retract_fd_jacobian`` / ``u_retract_fd_jacobian`` on
the SHARING structures probe the retraction at a non-minimal shared frame, where the FD ladder is
legitimately flat (the review's resolved D-item: retract preserves frame ranks only on a minimal
frame -- a caveat, not a precondition; ``docs/numerical_contracts.md``). Unshared FD rows accept a
convergence ratio >= 2.5 (h^2 truncation wobble, per the review's fd_convergence analysis).
"""
import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'oracle_sweeps'))
import o1_common
import o1_ragged
import o1_uniform


def _fd_ratio_ok(note):
    if 'ratio=' not in note:
        return False
    try:
        return float(note.split('ratio=')[1].split()[0]) >= 2.5
    except ValueError:
        return False


class _SweepCase(unittest.TestCase):
    _FD_OPS = ('retract_fd_jacobian', 'u_retract_fd_jacobian')

    def _assert_results(self):
        self.assertGreater(len(o1_common.RESULTS), 0)
        for op, sname, rep, C, W, K, sh, status, err, note in o1_common.RESULTS:
            if '_corewise' in op and '_vs_ragged' in op:
                continue                          # harness defect (see module docstring), never asserted
            with self.subTest(op=op, struct=sname, rep=rep, C=C, W=W, K=K, sharing=sh):
                if op in self._FD_OPS and sh is not None:
                    continue                      # the documented non-minimal-shared retract caveat
                if op in self._FD_OPS and status == 'FAIL' and _fd_ratio_ok(note):
                    continue                      # h^2 truncation wobble, still converging
                self.assertEqual(status, 'PASS', '%s err=%.2e %s' % (status, err, note))


class TestO1RaggedOracleSweep(_SweepCase):
    def test_full_sweep(self):
        o1_common.RESULTS.clear()
        for sname in o1_common.STRUCTS:
            o1_ragged.sweep_structure(sname, o1_common.STRUCTS[sname], o1_common.SHARING.get(sname))
        self._assert_results()


class TestO1UniformOracleSweep(_SweepCase):
    def test_full_sweep(self):
        o1_common.RESULTS.clear()
        for sname in ['d2', 'd3', 'd4', 'rank1', 'nonmin', 'sh2', 'shall']:
            for fp in (False, True):
                o1_uniform.sweep(sname, o1_common.STRUCTS[sname], o1_common.SHARING.get(sname), fp)
        o1_uniform.varying_rank()
        self._assert_results()


class TestO2ModelOracleSweep(unittest.TestCase):
    def test_full_sweep(self):
        import sweep_models
        sweep_models.failures.clear()
        sweep_models.main()
        self.assertEqual(sweep_models.failures, [])


def _optimizer_subset(key):
    d, _rep, geom_name, shared, kind, order, wspec, regspec = key
    base = (wspec == 'none' and not regspec)
    return ((d == 3 and base)
            or (geom_name == 'manifold' and not shared)
            or (d == 3 and kind == 'probe_derivatives'))


class TestO2OptimizerOracleSweep(unittest.TestCase):
    def test_pairwise_subset(self):
        import sweep_optimizers
        sweep_optimizers.failures.clear()
        sweep_optimizers.main(case_filter=_optimizer_subset)
        self.assertEqual(sweep_optimizers.failures, [])

    @unittest.skipUnless(os.environ.get('T3TOOLBOX_SLOW_TESTS'),
                         'full optimizer matrix: set T3TOOLBOX_SLOW_TESTS=1 (required at release)')
    def test_full_matrix_slow(self):
        import sweep_optimizers
        sweep_optimizers.failures.clear()
        sweep_optimizers.main()
        self.assertEqual(sweep_optimizers.failures, [])
