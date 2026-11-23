"""Tests for the backend Newton-CG diagnostic display (backend/optimizer_display.py, D3; P16 part 3: the oracle's
block sums, errors in the block shape, the multi-index table, error_table).

The formatter is pure (returns a string), so alignment/layout are asserted on the string directly -- no
stdout capture. Plus end-to-end `make_newton_display` + `newton_cg` runs recording the error arrays, checked against
dense residuals.
The whole capability is backend-owned (a raw-.data user gets the identical display; the anti-drift rule)."""
import dataclasses as dc
import math
import unittest

import numpy as np

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.backend.optimizers as bopt
import t3toolbox.backend.geometry as bgeo
import t3toolbox.backend.uniform_fitting as uf
import t3toolbox.backend.optimizer_display as bdisp
from t3toolbox.backend.fitting import LeastSquaresOracle, LeastSquaresBatch
from t3toolbox.backend.regularization import IdentityRegularizer
from t3toolbox.backend.jets import affine_jet, blocks_to_weight
import _derivative_data as dd
import _fits as F                    # a backend fit (geometry, oracle, batch, regularizer) from plain specs (P16)

SHAPE, TUCKER, TT = (8, 8, 8), (2, 2, 2), (1, 2, 2, 1)


def _info(**kw):
    """A representative non-converged NewtonInfo (step fields filled); override via kwargs."""
    base = dict(iteration=3, objective=5.23e-4, gnorm=1.44e-3, g0norm=6.8e-2, converged=False,
                forcing_eta=0.02, cg_tol=3.0e-4, cg_iters=14, cg_maxiter=200, cg_resid=8.7e-5,
                cg_converged=True, cg_truncated=False, ls_steps=2, alpha=0.25, slope=-1e-3,
                pHp=1e-3, delta_f=-1.2e-4, rho=0.98, step_rel=3e-2, wall_time=0.31)
    base.update(kw)
    return bopt.NewtonInfo(**base)


def _grid_lines(table):
    """The grid rows of a table (everything after the one-line legend)."""
    return table.split("\n")[1:]


class TestFormatter(unittest.TestCase):
    def test_cell_fixed_width(self):
        """%.1e cells are all 7 chars -- a `1.0e+00` and a `5.2e-04` line up with zero effort; `—` matches."""
        w = len("%.1e" % 1.0)
        self.assertEqual(w, 7)
        for v in (1.0, 5.2e-4, 8.1e-16, 3.3e-2, 1.0e-9):
            self.assertEqual(len(bdisp._cell(v, "%.1e", w)), w)
        self.assertEqual(len(bdisp._cell(float("nan"), "%.1e", w)), w)   # NaN -> — padded to width

    def test_two_axis_layout_aligns(self):
        """probe_derivatives layout (mode rows, order cols, train|val cells): every grid line is the SAME
        length even with `1.0e+00` next to `5.2e-04` next to `8.1e-16` -- the alignment guarantee."""
        train = np.array([[1.0, 5.2e-4, 8.1e-16], [3.3e-2, 1.0e+00, 2.7e-9]])   # (2 modes, 3 orders)
        val = train * 1.1
        table = bdisp._format_matrix(train, val, "%.1e")
        self.assertIn("rows=mode", table)
        self.assertIn("cols=order", table)
        self.assertIn("train|val", table)
        lines = _grid_lines(table)
        self.assertEqual(len(lines), 1 + 2)                              # header + 2 mode rows
        self.assertEqual(len({len(ln) for ln in lines}), 1)             # all lines equal length -> aligned
        self.assertIn("1.0e+00|", table)                                # a cell rendered

    def test_one_axis_plain_probe(self):
        """Plain probe (1 data axis = mode): dataset rows, mode cols; grid lines aligned."""
        train = np.array([[5.2e-4], [1.0e-2], [3.3e-2]])                 # (3 modes, 1 order)
        val = train * 1.2
        table = bdisp._format_matrix(train, val, "%.1e")
        self.assertIn("rows=train/val", table); self.assertIn("cols=mode", table)
        lines = _grid_lines(table)
        self.assertEqual(len(lines), 1 + 2)                              # header + train + val
        self.assertTrue(lines[1].lstrip().startswith("train"))
        self.assertTrue(lines[2].lstrip().startswith("val"))
        self.assertEqual(len({len(ln) for ln in lines}), 1)

    def test_one_axis_apply_derivatives(self):
        """apply/entries derivatives (1 data axis = order): dataset rows, order cols."""
        train = np.array([[5.2e-4, 2.7e-2, 9.1e-2, 1.8e-1]])            # (1 mode, 4 orders)
        table = bdisp._format_matrix(train, None, "%.1e")               # train-only
        self.assertIn("cols=order", table)
        lines = _grid_lines(table)
        self.assertEqual(len(lines), 1 + 1)                             # header + single train row (no val)
        self.assertEqual(len({len(ln) for ln in lines}), 1)

    def test_scalar_one_liner(self):
        """Plain apply/entries (0 data axes): a single line, no grid."""
        table = bdisp._format_matrix(np.array([[5.2e-4]]), np.array([[6.1e-4]]), "%.1e")
        self.assertNotIn("\n", table)
        self.assertIn("train", table); self.assertIn("val", table)

    def test_zero_norm_block_dash(self):
        """A zero-norm data block -> NaN relative error -> rendered `—`."""
        rel = bdisp.relative_errors(np.array([[1.0, 4.0]]), np.array([[4.0, 0.0]]))
        self.assertTrue(np.isclose(rel[0, 0], 0.5))
        self.assertTrue(np.isnan(rel[0, 1]))
        table = bdisp._format_matrix(rel, None, "%.1e")
        self.assertIn("—", table)

    def test_header_converged_vs_stepped(self):
        """The header is a full step line normally, a short `converged` line on the final iteration."""
        full = bdisp._format_header(_info())
        self.assertIn("CG 14/200", full); self.assertIn("ρ 0.98", full); self.assertIn("✓", full)
        conv = bdisp._format_header(_info(converged=True, cg_iters=None, cg_converged=None))
        self.assertIn("converged", conv); self.assertNotIn("CG ", conv)

    def test_converged_line_names_its_reference(self):
        """T29 / C.14: the converged line names the Newton-stop reference (`NewtonInfo.g0_reference`); a
        hand-built row with no reference kind prints exactly as in 2026.2.0 (a display change only)."""
        conv = dict(converged=True, cg_iters=None, cg_converged=None)
        self.assertTrue(bdisp._format_header(_info(**conv)).endswith("| converged (‖g‖ ≤ gtol·‖g₀‖)"))
        for kind, name in (('zero', 'zero tensor'), ('warm_start', 'warm start'), ('cone', 'cone (aligned zero tensor)'),
                           ('override', 'override')):
            with self.subTest(g0_reference=kind):
                line = bdisp._format_header(_info(g0_reference=kind, **conv))
                self.assertTrue(line.endswith("| converged (‖g‖ ≤ gtol·‖g₀‖, g₀ = %s)" % name), line)
        stepped = bdisp._format_header(_info(g0_reference='zero'))     # a stepped row: no reference text
        self.assertNotIn("g₀ =", stepped)

    def test_no_unweighted_figure(self):
        """M3: the header has no `(unwt …)` figure (the block sums are weighted, so it would be the misfit itself), and
        format_newton_iter has no obj_unweighted parameter."""
        self.assertNotIn("unwt", bdisp._format_header(_info(objective=1.0)))
        self.assertNotIn("unwt", bdisp._format_header(_info(objective=1.29, misfit=1.0, regularization=0.29)))
        with self.assertRaises(TypeError):
            bdisp.format_newton_iter(_info(), None, obj_unweighted=1.0)

    def test_header_misfit_reg_breakdown(self):
        """With a regularizer attached the objective splits as `obj = misfit + reg`; unregularized (the
        default, `regularization=None`) shows no split."""
        plain = bdisp._format_header(_info(objective=1.0))
        self.assertNotIn("misfit", plain); self.assertNotIn("+ reg", plain)      # regularization None -> no split
        reg = bdisp._format_header(_info(objective=1.29, misfit=1.0, regularization=0.29))
        self.assertIn("= misfit 1.000e+00 + reg 2.900e-01", reg)

    def test_truncated_symbol(self):
        h = bdisp._format_header(_info(cg_converged=False, cg_truncated=True))
        self.assertIn("⌇", h)

    # The 2026.2.0 header text, captured from the pre-trust-region formatter (fffdfde) for these rows: the
    # line-search `step` segment -- and every non-final Gauss-Newton row -- must stay byte-identical (less the
    # `(unwt …)` figure, dropped in M3).
    _LS_ROWS_2026_2 = [
        (dict(),
         'iter  3 | obj 5.230e-04  ‖g‖ 1.44e-03 (2.1e-02·g₀) | CG 14/200 tol 3.0e-04 resid 8.7e-05 ✓ '
         '| ls 2 α 2.50e-01 ‖Δx‖/‖x‖ 3.0e-02 | Δf -1.20e-04 ρ 0.98 | 0.31s'),
        (dict(ls_failed=True, ls_steps=40, alpha=2.0 ** -39, cg_converged=False, cg_truncated=True),
         'iter  3 | obj 5.230e-04  ‖g‖ 1.44e-03 (2.1e-02·g₀) | CG 14/200 tol 3.0e-04 resid 8.7e-05 ⌇ '
         '| ls 40! α 1.82e-12 ‖Δx‖/‖x‖ 3.0e-02 | Δf -1.20e-04 ρ 0.98 | 0.31s'),
        (dict(cg_converged=False, step_rel=float('nan'), rho=float('nan'), misfit=1e-4, regularization=4.23e-4),
         'iter  3 | obj 5.230e-04 = misfit 1.000e-04 + reg 4.230e-04  ‖g‖ 1.44e-03 (2.1e-02·g₀) '
         '| CG 14/200 tol 3.0e-04 resid 8.7e-05 ⋯ | ls 2 α 2.50e-01 ‖Δx‖/‖x‖ nan | Δf -1.20e-04 ρ nan | 0.31s'),
        (dict(cg_converged=False, g0_reference='zero'),
         'iter  3 | obj 5.230e-04  ‖g‖ 1.44e-03 (2.1e-02·g₀) | CG 14/200 tol 3.0e-04 resid 8.7e-05 ⋯ '
         '| ls 2 α 2.50e-01 ‖Δx‖/‖x‖ 3.0e-02 | Δf -1.20e-04 ρ 0.98 | 0.31s'),
    ]

    def test_line_search_rows_are_byte_identical(self):
        """C.14: the line-search `step` segment is the 2026.2.0 text byte for byte (a Gauss-Newton row of
        newton_cg also carries step_control / hessian / cg_boundary=False now -- none of which shows)."""
        for kw, expected in self._LS_ROWS_2026_2:
            for extra in (dict(), dict(step_control='line_search', hessian='gauss_newton', cg_boundary=False)):
                with self.subTest(row=kw, extra=extra):
                    self.assertEqual(bdisp._format_header(_info(**kw, **extra)), expected)

    def test_exact_mode_header(self):
        """T19: the trust-region `step` segment (`TR Δ … ✓/✗`), the boundary symbol `∂`, the exact cells
        `κ σ ε` (κ guarded against a zero denominator, σ / ε `—` when absent), and the stop reason on a run's
        last row (not on the converged line, which says so already)."""
        tr = dict(step_control='trust_region', hessian='exact', ls_steps=None, ls_failed=None, alpha=1.0,
                  tr_radius=3.1e-2, tr_accepted=True, pHp=8.0e-4, pKp=-2.0e-4, lift_sigma_min=4.2e-1,
                  lift_eps=1.3e-5)
        accepted = bdisp._format_header(_info(**tr))
        self.assertIn("| TR Δ 3.10e-02 ✓ ‖Δx‖/‖x‖ 3.0e-02 |", accepted)
        self.assertNotIn("ls ", accepted)
        self.assertTrue(accepted.endswith("| 0.31s | κ -2.0e-01 σ 4.2e-01 ε 1.3e-05"), accepted)   # κ = pKp/(pHp − pKp)
        self.assertIn("TR Δ 3.10e-02 ✗", bdisp._format_header(_info(**dict(tr, tr_accepted=False))))
        boundary = bdisp._format_header(_info(**dict(tr, cg_converged=False, cg_boundary=True)))
        self.assertIn("resid 8.7e-05 ∂ |", boundary)
        self.assertIn(" κ — ", bdisp._format_header(_info(**dict(tr, pKp=8.0e-4))))         # pHp − pKp = 0
        self.assertIn(" κ — ", bdisp._format_header(_info(**dict(tr, pKp=float('nan')))))
        absent = bdisp._format_header(_info(**dict(tr, lift_sigma_min=None, lift_eps=None)))
        self.assertTrue(absent.endswith("κ -2.0e-01 σ — ε —"), absent)                               # corewise / d = 1
        nan_radius = bdisp._format_header(_info(**dict(tr, tr_radius=float('nan'), step_rel=float('inf'))))
        self.assertIn("TR Δ — ✓ ‖Δx‖/‖x‖ — |", nan_radius)
        self.assertNotIn("κ", bdisp._format_header(_info(**dict(tr, hessian='gauss_newton', pKp=None))))
        for reason in ('objective_floor', 'trust_region_floor', 'max_newton', 'line_search'):
            with self.subTest(stop_reason=reason):
                self.assertTrue(bdisp._format_header(_info(**dict(tr, stop_reason=reason)))
                                .endswith("| stop: %s" % reason))
        conv = bdisp._format_header(_info(converged=True, cg_iters=None, stop_reason='gradient', g0_reference='zero',
                                          hessian='exact', step_control='trust_region'))
        self.assertTrue(conv.endswith("| converged (‖g‖ ≤ gtol·‖g₀‖, g₀ = zero tensor)"), conv)


class TestTables(unittest.TestCase):
    """P16 part 3: the table of errors in the block shape, laid out by the jet shape ``order`` (D4, O-42)."""

    def test_at_most_one_jet_variable_is_the_2026_5_table(self):
        """m <= 1: weight-shaped errors print as the 2026.5.0 table of their (n_mode, n_order) matrix: probe at ()
        (d,) -> one column per mode; probe at (n,) (d, n+1) -> mode rows, order columns; apply at (n,) (n+1,) -> order
        columns; apply and entries at () -> one line."""
        rng = np.random.default_rng(0)
        for shape, order, matrix, legend in (((3,), (), (3, 1), 'cols=mode'), ((3, 2), (1,), (3, 2), 'cols=order'),
                                             ((4,), (3,), (1, 4), 'cols=order'), ((), (), (1, 1), 'rel err   train')):
            with self.subTest(shape=shape, order=order):
                t, v = rng.random(shape), rng.random(shape)
                table = bdisp._format_table(t, v, '%.1e', order)
                self.assertEqual(table, bdisp._format_matrix(t.reshape(matrix), v.reshape(matrix), '%.1e'))
                self.assertIn(legend, table)

    @staticmethod
    def _cells(table):
        """{(row label, column label): cell} of a grid table (the legend line first)."""
        lines = table.split('\n')
        cols = lines[1].split()
        return {(line.split()[0], col): cell for line in lines[2:] for col, cell in zip(cols, line.split()[1:])}

    def test_two_or_more_jet_variables(self):
        """m >= 2: one column per multi-index of the jet axes, in row-major order; probe: mode rows with train|val
        cells; apply: train / val rows. F = (2, 3) is not square, so row-major is told from column-major: each cell is
        read back by its labels."""
        for lead in (0, 1):
            shape = ((3,) if lead else ()) + (2, 3)
            t = (1.0 + np.arange(np.prod(shape))).reshape(shape) * 1e-2          # every cell different
            v = 3.0 * t
            with self.subTest(kind='probe' if lead else 'apply'):
                table = bdisp._format_table(t, v, '%.1e', (1, 2))
                cols = table.split('\n')[1].split()
                self.assertEqual(cols, ['ord(%d,%d)' % (j, k) for j in range(2) for k in range(3)])
                cells = self._cells(table)
                for j in range(2):
                    for k in range(3):
                        col = 'ord(%d,%d)' % (j, k)
                        if lead:
                            for i in range(3):
                                self.assertEqual(cells[('m%d' % i, col)], '%.1e|%.1e' % (t[i, j, k], v[i, j, k]))
                        else:
                            self.assertEqual(cells[('train', col)], '%.1e' % t[j, k])
                            self.assertEqual(cells[('val', col)], '%.1e' % v[j, k])
                lines = table.split('\n')[1:]
                self.assertEqual(len({len(line) for line in lines}), 1)        # aligned

    def test_axes_by_position_without_an_order(self):
        """An oracle naming no jet shape: a scalar on one line; one axis in columns ``[j]``; more, the first axis in
        rows ``[i]`` and the others row-major in columns ``[j,k]``."""
        self.assertNotIn('\n', bdisp._format_table(np.array(0.5), None, '%.1e'))
        one = bdisp._format_table(np.array([0.1, 0.2]), None, '%.1e')
        self.assertEqual(one.split('\n')[1].split(), ['[0]', '[1]'])
        t = (1.0 + np.arange(12)).reshape(2, 2, 3) * 1e-2
        cells = self._cells(bdisp._format_table(t, None, '%.1e'))
        self.assertEqual(cells[('[1]', '[1,0]')], '%.1e' % t[1, 1, 0])
        self.assertEqual(cells[('[0]', '[0,2]')], '%.1e' % t[0, 0, 2])

    def test_a_shape_that_does_not_fit_raises(self):
        for t, order in ((np.ones((3, 3)), (1,)), (np.ones((2, 2, 2, 2)), (1,)), (np.ones(3), (1, 1))):
            with self.subTest(shape=t.shape, order=order):
                with self.assertRaises(ValueError):
                    bdisp._format_table(t, None, '%.1e', order)
        with self.assertRaises(ValueError):
            bdisp._format_table(np.ones((3, 2)), np.ones((2, 3)), '%.1e', (1,))

    def test_error_table_false_is_the_header_lines(self):
        """``error_table=False`` (and ``train_err=None``) give exactly the block's header lines -- the header, and the
        time line on a timed row -- and nothing of the table."""
        t = np.array([[1.9e-2, 1.3e-2], [1.0e+00, 5.2e-4]])
        for info, n_header in ((_info(), 1), (_info(timing=_timing()), 2)):
            with self.subTest(timed=n_header == 2):
                full = bdisp.format_newton_iter(info, t, 2 * t, order=(1,))
                off = bdisp.format_newton_iter(info, t, 2 * t, order=(1,), error_table=False)
                self.assertEqual(off, '\n'.join(full.split('\n')[:n_header]))
                self.assertNotIn('rel err', off)
                self.assertEqual(bdisp.format_newton_iter(info, None, order=(1,)), off)


SH = (5, 6, 7)


def _dense_probe(T, ww):
    """Per mode i, the dense tensor contracted with ww[j] on every mode j != i: (W, N_i)."""
    d = T.ndim
    out = []
    for i in range(d):
        ops = [T, list(range(d))]
        for j in range(d):
            if j != i:
                ops += [ww[j], [d, j]]
        out.append(np.einsum(*ops, [d, i]))
    return out


def _dense_apply(T, ww):
    d = T.ndim
    ops = [T, list(range(d))]
    for j in range(d):
        ops += [ww[j], [d, j]]
    return np.einsum(*ops, [d])


def _jets(order, W, rng):
    """Input jets: () the vectors ww; (1,) the line ww + s pp; (1, 1) a two-variable jet, (2, 2) + W + (N,)."""
    ww, pp, qq = ([rng.standard_normal((W, N)) for N in SH] for _ in range(3))
    if order == ():
        return ww
    if order == (1,):
        return affine_jet(ww, pp)
    return [np.stack([np.stack([w, p]), np.stack([q, np.zeros_like(w)])]) for w, p, q in zip(ww, pp, qq)]


def _batch(kind, order, W, rng, target, weight=None):
    """A batch of ``target``'s exact data at a random sample; ``weight`` one value per block (``_weight_shape``),
    placed on the data by ``blocks_to_weight``."""
    if kind == 'entries':
        index = np.stack([rng.integers(0, N, W) for N in SH])
        y = target.entries(index)
        return LeastSquaresBatch(index, y, None if weight is None else blocks_to_weight(kind, order, weight, y))
    sample = _jets(order, W, rng)
    y = target.probe(sample, order) if kind == 'probe' else target.apply(sample, order)
    return LeastSquaresBatch(sample, y, None if weight is None else blocks_to_weight(kind, order, weight, y))


def _weight_shape(kind, order):
    return ((len(SH),) if kind == 'probe' else ()) + tuple(n + 1 for n in order)


def _start():
    np.random.seed(11)
    return t3.TuckerTensorTrain.randn(SH, (2, 2, 2), (1, 2, 2, 1)).data


class _NoBlockSums(LeastSquaresOracle):
    """An oracle without the block sums (both, or only y_block_sumsq): the display prints the header lines only."""
    optional_outputs = ('Kp', 'Jp', 'r', 'f_resolution')


class _NoDataBlockSums(LeastSquaresOracle):
    optional_outputs = ('Kp', 'Jp', 'r', 'r_block_sumsq', 'f_resolution')


class TestMakeNewtonDisplay(unittest.TestCase):
    """P16 part 3: ``make_newton_display(oracle, batch, val_batch, ...)`` -- the errors through the oracle's block sums,
    the records in the block shape, the validation query without the training ``held``, ``error_table``."""

    def setUp(self):
        np.random.seed(0)
        self.rng = np.random.default_rng(0)
        self.target = t3.TuckerTensorTrain.randn(SH, (2, 2, 2), (1, 2, 2, 1))

    def run_display(self, oracle, batch, val_batch=None, x0=None, max_newton=3, newton=None, **kw):
        """A newton_cg run with the display: (stats, records, printed, infos)."""
        printed, infos = [], []
        display, records = bdisp.make_newton_display(oracle, batch, val_batch, print_fn=printed.append, **kw)

        def callback(info):
            infos.append(info)
            display(info)
        _, stats = bopt.newton_cg(bgeo.ManifoldGeometryOps(), oracle, batch, _start() if x0 is None else x0,
                                  max_newton=max_newton, callback=callback, **(newton or {}))
        return stats, records, printed, infos

    def test_records_in_the_weights_shape(self):
        """train_err and val_err keep the block shape, nothing reduced: () per kind, (2,) and (1, 1) for probe and
        apply; with and without validation (no val_err key without). The (1, 1) table has one column per multi-index."""
        cases = [('probe', ()), ('apply', ()), ('entries', ()), ('probe', (1,)), ('apply', (1,)), ('probe', (1, 1)),
                 ('apply', (1, 1))]
        for kind, order in cases:
            oracle = LeastSquaresOracle(kind, order)
            batch = _batch(kind, order, 40, self.rng, self.target)
            for validation in (False, True):
                with self.subTest(kind=kind, order=order, validation=validation):
                    val_batch = _batch(kind, order, 15, self.rng, self.target) if validation else None
                    stats, records, printed, _ = self.run_display(oracle, batch, val_batch, max_newton=2)
                    self.assertEqual(len(records), len(stats['history']))
                    for rec in records:
                        self.assertEqual(np.shape(rec['train_err']), _weight_shape(kind, order))
                        self.assertEqual('val_err' in rec, validation)
                        if validation:
                            self.assertEqual(np.shape(rec['val_err']), _weight_shape(kind, order))
                    if order == (1, 1):
                        cols = printed[0].split('\n')[2].split()
                        self.assertEqual(cols, ['ord(0,0)', 'ord(0,1)', 'ord(1,0)', 'ord(1,1)'])

    def test_the_errors_against_dense_residuals(self):
        """Each row's train and validation errors are the dense per-block relative errors at the row's point (probe
        per mode, apply and entries a scalar). The weight is constant within each block (one value per block through
        blocks_to_weight), so the weighted ratio ‖ω⊙r_b‖/‖ω⊙y_b‖ is the plain one, the 2026.5.0 numbers."""
        dense = {'probe': lambda T, b: _dense_probe(T, b.sample),
                 'apply': lambda T, b: _dense_apply(T, b.sample),
                 'entries': lambda T, b: T[tuple(b.sample)]}
        for kind in ('probe', 'apply', 'entries'):
            with self.subTest(kind=kind):
                weight = 0.5 + self.rng.random(_weight_shape(kind, ()))
                y_noise = 0.05
                batch = _batch(kind, (), 40, self.rng, self.target, weight)
                batch = dc.replace(batch, y=_perturb(batch.y, y_noise, self.rng))
                val_batch = _batch(kind, (), 15, self.rng, self.target)
                _, records, printed, infos = self.run_display(LeastSquaresOracle(kind, ()), batch, val_batch)
                for rec, info, text in zip(records, infos, printed):
                    T = t3.TuckerTensorTrain(*info.x_cores).to_dense()
                    for b, err in ((batch, rec['train_err']), (val_batch, rec['val_err'])):
                        z, y = dense[kind](T, b), b.y
                        if kind == 'probe':
                            ref = np.array([np.linalg.norm(zi - yi) / np.linalg.norm(yi) for zi, yi in zip(z, y)])
                        else:
                            ref = np.linalg.norm(z - y) / np.linalg.norm(y)
                        np.testing.assert_allclose(err, ref, rtol=1e-8)
                    self.assertNotIn('unwt', text)

    def test_a_zero_weighted_outlier_counts_in_neither_sum(self):
        """M3: the errors are ``‖ω⊙r_b‖ / ‖ω⊙y_b‖``, so a sample weighted zero is in neither sum. Exact data with one
        sample corrupted by ``1e3 ‖y‖`` and weighted zero: each row's errors are the plain dense ratio on the batch
        with that sample DELETED (``np.delete``, no weight anywhere), and the probe fit, exact on the rest, ends near 0
        while the plain ratio, outlier included, stays near 1 (it would dominate an unweighted table)."""
        bad = 7
        for kind, order in (('apply', ()), ('probe', (1,))):
            with self.subTest(kind=kind, order=order):
                m = len(order)
                batch = _batch(kind, order, 40, self.rng, self.target)
                ys = [np.array(a) for a in (batch.y if kind == 'probe' else [batch.y])]
                for a in ys:
                    a[(slice(None),) * m + (bad,)] += 1e3 * np.linalg.norm(a)
                omega = [np.ones((1,) * m + (40, 1)) for _ in SH] if kind == 'probe' else np.ones((40,))
                for a in (omega if kind == 'probe' else [omega]):
                    a[(0,) * m + (bad,)] = 0.0
                batch = LeastSquaresBatch(batch.sample, ys if kind == 'probe' else ys[0], omega)
                _, records, _, infos = self.run_display(LeastSquaresOracle(kind, order), batch, max_newton=12)
                for rec, info in zip(records, infos):
                    z = getattr(t3.TuckerTensorTrain(*info.x_cores), kind)(batch.sample, order)
                    zs = list(z) if kind == 'probe' else [z]
                    kept = [(np.delete(zi - yi, bad, axis=m), np.delete(yi, bad, axis=m)) for zi, yi in zip(zs, ys)]
                    ref = np.array([[np.linalg.norm(r[slot]) / np.linalg.norm(y[slot])
                                     for slot in np.ndindex(*r.shape[:m])]
                                    for r, y in kept]).reshape(np.shape(rec['train_err']))
                    np.testing.assert_allclose(rec['train_err'], ref, rtol=1e-8, atol=1e-12)   # atol: the rounding
                plain = max(np.linalg.norm(zi - yi) / np.linalg.norm(yi) for zi, yi in zip(zs, ys))
                self.assertGreater(plain, 0.5)                               # the outlier, were it counted
                if kind == 'probe':
                    self.assertLess(float(np.max(records[-1]['train_err'])), 1e-6)

    def test_each_table_is_under_its_own_batch_weight(self):
        """M3 (the design review's finding 1, ruled (a)): the training cells are under the run's weight, the validation
        cells under ``val_batch``'s own weight (here none), each against the dense ratio. The two weights give
        different numbers here, so a display that lent the training weight to the validation batch would fail."""
        for kind in ('apply', 'probe'):
            with self.subTest(kind=kind):
                omega = 0.2 + 2.0 * self.rng.random(40)                    # one weight per sample
                batch = _batch(kind, (), 40, self.rng, self.target)
                batch = dc.replace(batch, y=_perturb(batch.y, 0.2, self.rng),
                                   weight=[omega[:, None] for _ in SH] if kind == 'probe' else omega)
                val_batch = _batch(kind, (), 40, self.rng, self.target)
                val_batch = dc.replace(val_batch, y=_perturb(val_batch.y, 0.2, self.rng))
                _, records, _, infos = self.run_display(LeastSquaresOracle(kind, ()), batch, val_batch, max_newton=2)
                dense = (lambda T, b: _dense_probe(T, b.sample)) if kind == 'probe' else \
                    (lambda T, b: [_dense_apply(T, b.sample)])
                as_list = (lambda y: list(y)) if kind == 'probe' else (lambda y: [y])
                for rec, info in zip(records, infos):
                    T = t3.TuckerTensorTrain(*info.x_cores).to_dense()
                    w2 = (omega ** 2)[:, None] if kind == 'probe' else omega ** 2
                    train = [math.sqrt(np.sum(w2 * (z - y) ** 2) / np.sum(w2 * y ** 2))
                             for z, y in zip(dense(T, batch), as_list(batch.y))]
                    val = [np.linalg.norm(z - y) / np.linalg.norm(y) for z, y in zip(dense(T, val_batch),
                                                                                       as_list(val_batch.y))]
                    lent = [math.sqrt(np.sum(w2 * (z - y) ** 2) / np.sum(w2 * y ** 2))
                            for z, y in zip(dense(T, val_batch), as_list(val_batch.y))]
                    np.testing.assert_allclose(rec['train_err'], np.reshape(train, np.shape(rec['train_err'])),
                                               rtol=1e-8)
                    np.testing.assert_allclose(rec['val_err'], np.reshape(val, np.shape(rec['val_err'])), rtol=1e-8)
                    self.assertGreater(float(np.max(np.abs(np.array(lent) - np.array(val)) / np.array(val))), 1e-3)

    def test_no_unwt_figure_on_any_run(self):
        """M3: no row prints an ``(unwt …)`` figure: weighted, regularized, or on a drawn batch."""
        oracle = LeastSquaresOracle('apply', ())
        weighted = dc.replace(_batch('apply', (), 40, self.rng, self.target), weight=0.5 + self.rng.random(40))
        runs = [dict(batch=weighted), dict(batch=weighted, newton=dict(regularizer=IdentityRegularizer(1e-2))),
                dict(batch=oracle.take(weighted, np.arange(0, 40, 2)))]
        for i, run in enumerate(runs):
            with self.subTest(run=i):
                _, _, printed, _ = self.run_display(oracle, run['batch'], max_newton=2, newton=run.get('newton'))
                for text in printed:
                    self.assertNotIn('unwt', text)

    def test_the_queries(self):
        """What the display asks of the oracle: y_block_sumsq once per batch for the callback's life (two runs); each
        new point's r_block_sumsq at the model's frame, on its batch, with its held (no sweep: the residual is held);
        the validation r_block_sumsq at the same frame on val_batch, never with a held (the training one belongs to
        the training batch)."""
        import unittest.mock
        oracle = LeastSquaresOracle('probe', (1,))
        batch, val_batch = (_batch('probe', (1,), W, self.rng, self.target) for W in (40, 15))
        calls = []
        real = LeastSquaresOracle.query

        def recording(self_, frame, b, want, **kw):
            calls.append((frame, b, tuple(want), kw.get('held'), kw.get('Vc')))
            return real(self_, frame, b, want, **kw)
        display, records = bdisp.make_newton_display(oracle, batch, val_batch, print_fn=None)
        infos = []
        with unittest.mock.patch.object(LeastSquaresOracle, 'query', recording):
            for x0 in (_start(), t3.TuckerTensorTrain.randn(SH, (2, 2, 2), (1, 2, 2, 1)).data):
                bopt.newton_cg(bgeo.ManifoldGeometryOps(), oracle, batch, x0, max_newton=3,
                               callback=lambda info: (infos.append(info), display(info)))
        y_calls = [c for c in calls if 'y_block_sumsq' in c[2]]
        self.assertEqual([(c[1] is batch, c[1] is val_batch) for c in y_calls], [(True, False), (False, True)])
        train = [c for c in calls if c[2] == ('r_block_sumsq',) and c[1] is batch]
        val = [c for c in calls if c[2] == ('r_block_sumsq',) and c[1] is val_batch]
        self.assertEqual(len(train), len(infos))
        self.assertEqual(len(val), len(infos))
        for info, t, v in zip(infos, train, val):
            self.assertIs(t[0], info.lm.frame)
            self.assertIs(t[3], info.lm.held)
            self.assertIs(t[4], info.lm.centers[0])
            self.assertIs(v[0], info.lm.frame)
            self.assertIsNone(v[3])                                     # never the training held

    def test_the_errors_are_on_the_displays_batch(self):
        """Review round 1: numerator and denominator both on the display's ``batch``. A display on the full batch, runs on
        two different draws with the one callback: every row's train_err is the dense relative error on the FULL batch at
        the row's point (the measured case: 2.469 on the drawn third against 0.550 recorded before); those queries get
        no held (the run's held belongs to the drawn batch); and no (unwt …) figure is printed (dropped in M3)."""
        import unittest.mock
        oracle = LeastSquaresOracle('apply', ())
        full = _batch('apply', (), 60, self.rng, self.target)
        full = dc.replace(full, y=_perturb(full.y, 0.05, self.rng))
        geom = bgeo.ManifoldGeometryOps()
        printed, infos, calls = [], [], []
        display, records = bdisp.make_newton_display(oracle, full, print_fn=printed.append)
        real = LeastSquaresOracle.query

        def recording(self_, frame, b, want, **kw):
            calls.append((b, tuple(want), kw.get('held')))
            return real(self_, frame, b, want, **kw)
        with unittest.mock.patch.object(LeastSquaresOracle, 'query', recording):
            for start in (0, 1):
                drawn = oracle.take(full, np.arange(start, 60, 3))
                bopt.newton_cg(geom, oracle, drawn, _start(), max_newton=3,
                               callback=lambda info: (infos.append(info), display(info)))
        self.assertEqual(len(records), len(infos))
        for rec, info, text in zip(records, infos, printed):
            self.assertIsNot(info.lm.batch, full)
            T = t3.TuckerTensorTrain(*info.x_cores).to_dense()
            ref = np.linalg.norm(_dense_apply(T, full.sample) - full.y) / np.linalg.norm(full.y)
            np.testing.assert_allclose(rec['train_err'], ref, rtol=1e-8)
            self.assertNotIn('unwt', text)
        on_full = [c for c in calls if c[0] is full and c[1] == ('r_block_sumsq',)]
        self.assertEqual(len(on_full), len(infos))
        self.assertTrue(all(c[2] is None for c in on_full))
        self.assertEqual(sum(c[0] is full and c[1] == ('y_block_sumsq',) for c in calls), 1)

    def test_on_the_runs_batch_no_sweep(self):
        """When the display's batch is the run's, a row's numerator is read through the model's held r: the engine runs
        none of the plain sweep's rules (xi, mu, nu, eta, z, r) while the display works."""
        import t3toolbox.backend.sampling_variables as sv
        import unittest.mock
        oracle = LeastSquaresOracle('probe', (1,))
        batch = _batch('probe', (1,), 40, self.rng, self.target)
        display, records = bdisp.make_newton_display(oracle, batch, print_fn=None)
        ran, real = [], sv._run

        def recording(rule, *args, **kwargs):
            ran.append(rule.name)
            return real(rule, *args, **kwargs)

        def callback(info):
            with unittest.mock.patch.object(sv, '_run', recording):
                display(info)
        bopt.newton_cg(bgeo.ManifoldGeometryOps(), oracle, batch, _start(), max_newton=3, callback=callback)
        self.assertEqual(len(records), 3)
        self.assertFalse({'xi', 'mu', 'nu', 'eta', 'z', 'r'} & set(ran), ran)

    def test_rows_after_a_rejection_reuse_the_errors(self):
        """n14: a rejected trust-region trial keeps x (and its model), so the NEXT row reuses that row's errors -- no
        query for an unchanged point. The validation queries counted on an exact run with rejections equal the number
        of distinct points shown; a callback shared by two runs recomputes the second run's first row."""
        import unittest.mock
        np.random.seed(0)
        target = t3.TuckerTensorTrain.randn(SH, (3, 3, 3), (1, 3, 3, 1))
        oracle = LeastSquaresOracle('probe', (1,))
        batch, val_batch = (_batch('probe', (1,), W, self.rng, target) for W in (40, 15))
        geom = bgeo.ManifoldGeometryOps()
        x0 = _start()
        real = LeastSquaresOracle.query
        val_queries = []

        def recording(self_, frame, b, want, **kw):
            if b is val_batch and 'r_block_sumsq' in want:
                val_queries.append(1)
            return real(self_, frame, b, want, **kw)
        cb, records = bdisp.make_newton_display(oracle, batch, val_batch, print_fn=None)
        with unittest.mock.patch.object(LeastSquaresOracle, 'query', recording):
            _, stats = bopt.newton_cg(geom, oracle, batch, x0, max_newton=10, hessian='exact', tr_radius0=1e3,
                                      callback=cb)
        rows = stats['history']
        after_rejection = [k for k in range(1, len(rows)) if rows[k - 1]['tr_accepted'] is False]
        self.assertTrue(after_rejection, 'expected a rejected trial in this run')
        self.assertEqual(len(val_queries), len(rows) - len(after_rejection))
        for k in after_rejection:
            self.assertIs(records[k]['train_err'], records[k - 1]['train_err'])
            self.assertIs(records[k]['val_err'], records[k - 1]['val_err'])
        for k in set(range(1, len(rows))) - set(after_rejection):
            self.assertIsNot(records[k]['val_err'], records[k - 1]['val_err'])
        first_rejection = after_rejection[0]
        cb2, records2 = bdisp.make_newton_display(oracle, batch, val_batch, print_fn=None)
        bopt.newton_cg(geom, oracle, batch, x0, max_newton=first_rejection, hessian='exact', tr_radius0=1e3,
                       callback=cb2)
        self.assertIs(records2[-1]['tr_accepted'], False)
        np.random.seed(6)
        bopt.newton_cg(geom, oracle, batch, t3.TuckerTensorTrain.randn(SH, (2, 2, 2), (1, 2, 2, 1)).data,
                       max_newton=1, callback=cb2)
        self.assertIsNot(records2[-1]['train_err'], records2[-2]['train_err'])
        self.assertFalse(np.allclose(records2[-1]['val_err'], records2[-2]['val_err']))

    def test_an_oracle_without_block_sums_gets_the_header_lines(self):
        """An oracle lacking either block sum: each row prints its header line and no table, and its records carry the
        scalars and no error arrays; with or without a validation batch."""
        for cls in (_NoBlockSums, _NoDataBlockSums):
            for validation in (False, True):
                with self.subTest(oracle=cls.__name__, validation=validation):
                    oracle = cls('probe', ())
                    batch = _batch('probe', (), 40, self.rng, self.target)
                    val_batch = _batch('probe', (), 15, self.rng, self.target) if validation else None
                    stats, records, printed, _ = self.run_display(oracle, batch, val_batch)
                    self.assertEqual(len(printed), len(stats['history']))
                    for text, rec in zip(printed, records):
                        self.assertEqual(text.count('\n'), 0)
                        self.assertTrue(text.startswith('iter '))
                        self.assertNotIn('train_err', rec)
                        self.assertNotIn('val_err', rec)
                        self.assertIn('objective', rec)

    def test_error_table_false_changes_no_record(self):
        """error_table=False: each block is the error_table=True block's header line alone; the records are equal
        (arrays bitwise)."""
        import re
        clock = re.compile(r'\| \d+\.\d\ds\b')
        oracle = LeastSquaresOracle('apply', (1, 1))
        batch, val_batch = (_batch('apply', (1, 1), W, self.rng, self.target) for W in (40, 15))
        runs = [self.run_display(oracle, batch, val_batch, error_table=flag) for flag in (True, False)]
        (_, rec_on, text_on, _), (_, rec_off, text_off, _) = runs
        self.assertEqual([clock.sub('| T', t) for t in text_off],
                         [clock.sub('| T', t.split('\n')[0]) for t in text_on])
        self.assertEqual(len(rec_on), len(rec_off))
        for a, b in zip(rec_on, rec_off):
            self.assertEqual(list(a), list(b))
            for key in a:
                if key == 'wall_time':
                    continue
                if isinstance(a[key], np.ndarray):
                    self.assertTrue(np.array_equal(a[key], b[key], equal_nan=True), key)
                else:
                    self.assertEqual(a[key], b[key], key)

    def test_uniform_equals_ragged(self):
        """The uniform layout: the packed batches give the ragged run's errors, in the same (unpadded) shape."""
        oracle = LeastSquaresOracle('probe', (1,))
        batch, val_batch = (_batch('probe', (1,), W, self.rng, self.target) for W in (40, 15))
        x0 = _start()
        ux = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain(*x0), N=8, n=3, r=3)
        ufit, usc = F.uniform(F.Fit(bgeo.ManifoldGeometryOps(), oracle, batch), ux)
        uval = uf.pack_batch(oracle.kind, oracle.order, val_batch, ux.N)
        _, ragged, _, _ = self.run_display(oracle, batch, val_batch, x0=x0)
        display, uniform = bdisp.make_newton_display(oracle, ufit.batch, uval, print_fn=None)
        bopt.newton_cg(ufit.geom, oracle, ufit.batch, usc, max_newton=3, callback=display)
        self.assertEqual(len(ragged), len(uniform))
        for a, b in zip(ragged, uniform):
            for key in ('train_err', 'val_err'):
                self.assertEqual(np.shape(b[key]), np.shape(a[key]))
                np.testing.assert_allclose(b[key], a[key], rtol=1e-8)

    def test_a_drawn_batch(self):
        """On a drawn batch the scale is in both block sums and cancels: the errors equal the plain subset's (the same
        draw with the scale dropped), and no (unwt …) figure is printed."""
        oracle = LeastSquaresOracle('apply', ())
        full = _batch('apply', (), 60, self.rng, self.target)
        drawn = oracle.take(full, np.arange(0, 60, 3))
        plain = dc.replace(drawn, minibatch_scale=None)
        x0 = _start()
        _, on_drawn, printed, infos = self.run_display(oracle, drawn, x0=x0, max_newton=1)
        display, on_plain = bdisp.make_newton_display(oracle, plain, print_fn=None)
        lm = bopt.local_model(bgeo.ManifoldGeometryOps(), oracle, plain, x0)
        display(dc.replace(infos[0], lm=lm))
        np.testing.assert_allclose(on_drawn[0]['train_err'], on_plain[0]['train_err'], rtol=1e-12)
        T = t3.TuckerTensorTrain(*infos[0].x_cores).to_dense()
        plain = 0.5 * float(np.sum((_dense_apply(T, drawn.sample) - drawn.y) ** 2))
        self.assertAlmostEqual(infos[0].objective / plain, 3.0, delta=1e-6)        # the run's misfit: the scale, 60/20
        self.assertNotIn('unwt', printed[0])


def _perturb(y, noise, rng):
    """``y`` plus Gaussian noise of relative size ``noise`` (a probe ``y`` is a list per mode)."""
    if isinstance(y, (list, tuple)):
        return [yi + noise * np.linalg.norm(yi) / np.sqrt(yi.size) * rng.standard_normal(yi.shape) for yi in y]
    return y + noise * np.linalg.norm(y) / np.sqrt(max(np.size(y), 1)) * rng.standard_normal(np.shape(y))


def _seconds(**kw):
    """A PhaseSeconds of a Gauss-Newton line-search row after row 0 (model built, three trials); override via kwargs."""
    base = dict(previous_callback=0.01, frame=0.05, oracle=0.07, companion=0.001, curvature=None,
                gradient=0.03, reference=None, cg=0.40, forms=0.02, trial_retraction=0.15, trial_objective=0.06,
                tracking=None, remainder=0.01)
    base.update(kw)
    return bopt.PhaseSeconds(**base)


def _timing(seconds=None, **kw):
    """A NewtonTiming around ``_seconds()`` (three trials, the last accepted, no compile column)."""
    base = dict(window=0.801, seconds=_seconds() if seconds is None else seconds, compile_seconds=None, trials=3,
                accepted_retraction=0.07)
    base.update(kw)
    return bopt.NewtonTiming(**base)


def _compile_like(seconds, **kw):
    """A compile column for ``seconds`` where nothing compiled (0.0 where the phase ran, else None); override via
    kwargs."""
    zeros = {name: (None if value is None else 0.0) for name, value in dc.asdict(seconds).items()}
    return bopt.PhaseSeconds(**{**zeros, **kw})


def _zero_compile(**kw):
    return _compile_like(_seconds(), **kw)


class TestTimingDisplay(unittest.TestCase):
    """T8: ``newton_cg(timing=True)``'s rows in the display -- the ``time`` line between a timed row's header and its
    error table, and ``make_newton_display``'s closing table, once, after the run's last row. Untimed rows print as
    before."""

    def test_the_time_line_only_with_timing(self):
        """A timed row's block is the untimed block with one line inserted after the header; an untimed row has
        none."""
        train = np.array([[1.9e-2, 1.3e-2], [1.0e+00, 5.2e-4]])
        plain = bdisp.format_newton_iter(_info(), train, order=(1,))
        timed = bdisp.format_newton_iter(_info(timing=_timing()), train, order=(1,))
        self.assertNotIn('  time  ', plain)
        lines = timed.split('\n')
        self.assertTrue(lines[1].startswith('  time  model '), lines[1])
        self.assertEqual('\n'.join(lines[:1] + lines[2:]), plain)

    def test_the_time_line_cells(self):
        """Each cell, by the record that sets it: a phase that did not run is left out; seconds as ``%.1e``."""
        line = bdisp._format_timing(_info(timing=_timing()))
        self.assertEqual(line, '  time  model 1.2e-01 [frame 5.0e-02 oracle 7.0e-02 comp 1.0e-03] | '
                               'grad 3.0e-02 | cg 4.0e-01 (2.9e-02/it) | forms 2.0e-02 | '
                               'trials 2.1e-01 ×3 [retr 1.5e-01 obj 6.0e-02] (accepted 7.0e-02) | cb 1.0e-02 | '
                               'rest 1.0e-02')                           # cg per iteration: 0.40 / 14
        cell = lambda **kw: bdisp._format_timing(_info(**{'timing': _timing(), **kw}))
        one_trial = cell(timing=_timing(trials=1, accepted_retraction=0.15))
        self.assertIn('| trials 2.1e-01 ×1 [retr 1.5e-01 obj 6.0e-02] | cb', one_trial)       # no 'accepted' share
        rejected = cell(timing=_timing(trials=1, accepted_retraction=None))
        self.assertNotIn('accepted', rejected)
        exact = cell(timing=_timing(_seconds(curvature=0.2)))
        self.assertIn('model 3.2e-01 [frame 5.0e-02 oracle 7.0e-02 comp 1.0e-03 curv 2.0e-01] |', exact)
        tracked = cell(timing=_timing(_seconds(tracking=0.05)))
        self.assertIn('(accepted 7.0e-02) | track 5.0e-02 | cb 1.0e-02 |', tracked)
        reused = cell(timing=_timing(_seconds(frame=None, oracle=None, companion=None, gradient=None)))
        self.assertTrue(reused.startswith('  time  cg 4.0e-01'), reused)   # a row after a rejection: no build
        converged = cell(converged=True, cg_iters=None,
                         timing=_timing(_seconds(cg=None, forms=None, trial_retraction=None, trial_objective=None),
                                        trials=None, accepted_retraction=None))
        self.assertTrue(converged.endswith('| grad 3.0e-02 | cb 1.0e-02 | rest 1.0e-02'), converged)
        self.assertNotIn('/it', cell(cg_iters=0))                        # no CG iteration: no per-iteration time

    def test_row_zero_carries_the_one_offs(self):
        """Row 0: the reference with the cone's pass count, then the setup and the alignment with its pass count,
        after the window's cells; no callback cell (row 0 has no previous row)."""
        row0 = _timing(_seconds(previous_callback=None, reference=0.11), setup=1.3e-4, alignment=0.08,
                       alignment_passes=(0.02,) * 5, reference_passes=(0.01,) * 9)
        line = bdisp._format_timing(_info(iteration=0, timing=row0))
        self.assertIn('| grad 3.0e-02 | ref 1.1e-01 ×9 | cg ', line)
        self.assertTrue(line.endswith('| rest 1.0e-02 | setup 1.3e-04 | align 8.0e-02 ×5'), line)
        self.assertNotIn('| cb ', line)
        untimed_passes = bdisp._format_timing(_info(iteration=0, timing=dc.replace(row0, alignment_passes=None,
                                                                                   reference_passes=None)))
        self.assertIn('| ref 1.1e-01 | cg ', untimed_passes)            # the member did not report its passes
        self.assertTrue(untimed_passes.endswith('| align 8.0e-02'), untimed_passes)

    def test_the_compile_cell(self):
        """The compile wait by phase (above a microsecond) on a listened run, ``compile 0`` when nothing compiled,
        no cell without a listener; CG's seconds per iteration exclude its compile wait."""
        line = lambda timing: bdisp._format_timing(_info(timing=timing))
        self.assertNotIn('compile', line(_timing()))                                     # numpy / no listener
        self.assertTrue(line(_timing(compile_seconds=_zero_compile())).endswith('| rest 1.0e-02 | compile 0'))
        self.assertTrue(line(_timing(compile_seconds=_zero_compile(remainder=1e-12)))   # rounding, not a compile
                        .endswith('| compile 0'))
        compiled = _timing(_seconds(cg=2.0), compile_seconds=_zero_compile(cg=1.5, frame=5e-5),
                           setup_compile=0.0, alignment=4.0, alignment_compile=3.0)
        text = line(compiled)
        self.assertIn('| cg 2.0e+00 (3.6e-02/it) |', text)                             # (2.0 - 1.5) / 14
        self.assertTrue(text.endswith('| compile frame 5.0e-05 cg 1.5e+00 align 3.0e+00'), text)   # 50 µs is shown

    def test_the_closing_table(self):
        """Per phase that ran: seconds over the rows, % of the summed windows, compile ('—' without a listener),
        the mean over the rows it ran on; then the one-offs and the pass counts and CG's seconds per iteration."""
        row0 = _timing(_seconds(previous_callback=None, reference=0.2), window=1.0, setup=1e-3, alignment=0.5,
                       alignment_passes=(0.1,) * 5, reference_passes=(0.02,) * 9)
        last = _timing(_seconds(cg=None, forms=None, trial_retraction=None, trial_objective=None), window=0.6,
                       trials=None, accepted_retraction=None)
        run = [(row0, 10), (_timing(window=0.4), 6), (last, None)]
        table = bdisp._format_timing_table(run)
        lines = table.split('\n')
        self.assertEqual(lines[0], "  timing  3 rows, Σ window 2.0e+00 s (the last row's callback is not in it)")
        self.assertEqual(lines[1].split(), ['seconds', '%', 'window', 'compile', 'mean/row'])
        rows = {line.split()[0]: line.split()[1:] for line in lines[2:-1]}
        self.assertEqual(list(rows), ['previous_callback', 'frame', 'oracle', 'companion', 'gradient',
                                      'reference', 'cg', 'forms', 'trial_retraction', 'trial_objective', 'remainder',
                                      'setup', 'alignment'])             # no curvature, no tracking: they never ran
        self.assertEqual(rows['cg'], ['8.0e-01', '40.0', '—', '4.0e-01'])   # two rows ran CG
        self.assertEqual(rows['previous_callback'], ['2.0e-02', '1.0', '—', '1.0e-02'])
        self.assertEqual(rows['reference'], ['2.0e-01', '10.0', '—', '2.0e-01'])
        self.assertEqual(rows['alignment'], ['5.0e-01', '—', '—', '—'])
        self.assertEqual(lines[-1], '    alignment 5 passes, reference 9 passes, cg 5.0e-02 s per CG iteration')
        listened = [(dc.replace(timing, compile_seconds=_compile_like(timing.seconds)), n) for timing, n in run]
        listened[0] = (dc.replace(row0, compile_seconds=_compile_like(row0.seconds, cg=0.3), setup_compile=2e-4,
                                  alignment_compile=0.4), 10)               # CG and the one-offs compiled
        lines = bdisp._format_timing_table(listened).split('\n')
        rows = {line.split()[0]: line.split()[1:] for line in lines[2:-1]}
        self.assertEqual(rows['cg'], ['8.0e-01', '40.0', '3.0e-01', '4.0e-01'])
        self.assertEqual(rows['setup'], ['1.0e-03', '—', '2.0e-04', '—'])
        self.assertEqual(rows['alignment'], ['5.0e-01', '—', '4.0e-01', '—'])
        self.assertEqual(lines[-1], '    alignment 5 passes, reference 9 passes, cg 3.1e-02 s per CG iteration')

    def _problem(self):
        np.random.seed(0)
        rng = np.random.default_rng(0)
        shape = (6, 6, 6)
        target = t3.TuckerTensorTrain.randn(shape, (3, 3, 3), (1, 3, 3, 1))
        ww, pp = ([rng.standard_normal((40, n)) for n in shape] for _ in range(2))
        data = [z + 0.01 * rng.standard_normal(z.shape) for z in dd.forward(target, 'probe', ww, pp, 1)]
        problem = F.derivative_fit(bgeo.ManifoldGeometryOps(), 'probe', 1, ww, pp, data)
        return problem, t3.TuckerTensorTrain.randn(shape, (2, 2, 2), (1, 2, 2, 1)).data

    def test_a_timed_run(self):
        """End to end: every row's block has its time line after the header; the closing table is printed once, after
        the last row, for that run's rows only (a callback reused across runs resets at row 0); and the untimed run
        prints exactly the timed run's text without them (wall clocks masked). Three runs: one stopped by the cap, one
        converged, one converged at row 0 (one row, no CG iteration: a one-row table without the CG note)."""
        import re
        problem, x0 = self._problem()
        clock = re.compile(r'\| \d+\.\d\ds\b')                          # the header's wall-clock cell
        for name, kw in (('capped', dict(max_newton=4)), ('converged', dict(max_newton=30, gtol_rel=1e-3)),
                         ('converged at row 0', dict(max_newton=30, gtol_rel=10.0))):
            with self.subTest(name):
                timed, untimed = [], []
                cb, _ = bdisp.make_newton_display(problem.oracle, problem.batch, print_fn=timed.append)
                _, stats = problem.newton_cg(x0, timing=True, callback=cb, **kw)
                cb_plain, _ = bdisp.make_newton_display(problem.oracle, problem.batch, print_fn=untimed.append)
                problem.newton_cg(x0, callback=cb_plain, **kw)
                rows = len(stats['history'])
                self.assertEqual(len(timed), rows + 1)                       # a block per row, then the table
                header = '  timing  1 row, ' if rows == 1 else '  timing  %d rows, ' % rows
                self.assertTrue(timed[-1].startswith(header + 'Σ window '), timed[-1])
                iterated = any(row['cg_iters'] for row in stats['history'])
                self.assertEqual('per CG iteration' in timed[-1], iterated)
                if name == 'converged at row 0':
                    self.assertEqual((rows, iterated), (1, False))
                    self.assertTrue(timed[-1].split('\n')[-1].lstrip().startswith('setup'), timed[-1])   # no note line
                for block in timed[:-1]:
                    lines = block.split('\n')
                    self.assertTrue(lines[0].startswith('iter '))
                    self.assertTrue(lines[1].startswith('  time  '))
                    self.assertEqual(sum(line.startswith('  time  ') for line in lines), 1)
                stripped = ['\n'.join(line for line in block.split('\n') if not line.startswith('  time  '))
                            for block in timed[:-1]]
                self.assertEqual([clock.sub('| T', b) for b in stripped], [clock.sub('| T', b) for b in untimed])
                again = problem.newton_cg(x0, timing=True, callback=cb, max_newton=2)[1]   # the same callback
                self.assertEqual(len(again['history']), 2)
                self.assertTrue(timed[-1].startswith('  timing  2 rows, '), timed[-1])           # this run's rows only
                self.assertEqual(sum(text.startswith('  timing  ') for text in timed), 2)

    def test_the_last_row_closes_the_table(self):
        """The table follows a converged row without a stop reason (a hand-replayed run), and only the last row."""
        problem, x0 = self._problem()
        infos = []
        problem.newton_cg(x0, timing=True, max_newton=3, callback=infos.append)
        printed = []
        cb, _ = bdisp.make_newton_display(problem.oracle, problem.batch, print_fn=printed.append)
        last = dc.replace(infos[-1], converged=True, stop_reason=None)
        for info in infos[:-1] + [last]:
            cb(info)
        self.assertEqual([text.startswith('  timing  ') for text in printed], [False] * 3 + [True])
        self.assertTrue(printed[-1].startswith('  timing  3 rows, '))

    def test_a_silent_display_of_a_timed_run(self):
        """A records-only display (``print_fn=None``) on a timed run prints nothing, raises nothing at the last row, and
        records exactly what it records on the untimed run (wall clocks aside): ``timing`` is not a history field."""
        problem, x0 = self._problem()
        for name, kw in (('capped', dict(max_newton=4)), ('converged at row 0', dict(max_newton=30, gtol_rel=10.0))):
            with self.subTest(name):
                cb, timed = bdisp.make_newton_display(problem.oracle, problem.batch, print_fn=None)
                problem.newton_cg(x0, timing=True, callback=cb, **kw)
                cb_plain, untimed = bdisp.make_newton_display(problem.oracle, problem.batch, print_fn=None)
                problem.newton_cg(x0, callback=cb_plain, **kw)
                self.assertEqual(len(timed), len(untimed))
                for a, b in zip(timed, untimed):
                    self.assertEqual(list(a), list(b))                       # the same keys, in the same order
                    for key in a:
                        if key == 'wall_time':
                            continue
                        if isinstance(a[key], np.ndarray):
                            self.assertTrue(np.array_equal(a[key], b[key], equal_nan=True), key)
                        else:
                            self.assertEqual(a[key], b[key], key)


class TestSwitchDisplay(unittest.TestCase):
    """T10: ``newton_cg(hessian='switch')``'s rows in the display -- a Gauss-Newton-phase row's header ends with the
    ``sw`` segment (the tracking ratio ``e_N / e_GN``, then the count), the switching row with ``⇒ exact``; exact-phase
    rows show their ``κ σ ε`` cells; every row outside switch mode prints as before."""

    # A Gauss-Newton line-search row's header, as every untimed non-switch row prints it (TestFormatter's _info()).
    HEADER = ('iter  3 | obj 5.230e-04  ‖g‖ 1.44e-03 (2.1e-02·g₀) | CG 14/200 tol 3.0e-04 resid 8.7e-05 ✓ '
              '| ls 2 α 2.50e-01 ‖Δx‖/‖x‖ 3.0e-02 | Δf -1.20e-04 ρ 0.98 | 0.31s')
    GN_PHASE = dict(hessian='gauss_newton', switch_pKp=-3.1e-4, switch_e_gn=1.6e-4, switch_e_n=4.0e-5, switch_count=1,
                    switch_fired=False)                          # a tracked hit, the first of two

    def header(self, **kw):
        return bdisp._format_header(_info(**kw))

    def test_the_segment(self):
        """The ratio in ``%.1e``, the count after the row, the mark only where ``switch_fired``; ``—`` where there is
        no ratio (an untracked row, ``e_GN = 0``, a non-finite error)."""
        self.assertEqual(self.header(**self.GN_PHASE), self.HEADER + ' | sw 2.5e-01 ×1')
        fired = self.header(**dict(self.GN_PHASE, switch_e_n=1.1e-6, switch_count=2, switch_fired=True))
        self.assertEqual(fired, self.HEADER + ' | sw 6.9e-03 ×2 ⇒ exact')
        miss = self.header(**dict(self.GN_PHASE, switch_e_n=3.2e-4, switch_count=0))
        self.assertEqual(miss, self.HEADER + ' | sw 2.0e+00 ×0')
        unfired = self.header(**dict(self.GN_PHASE, switch_count=2))           # the mark reads switch_fired alone
        self.assertEqual(unfired, self.HEADER + ' | sw 2.5e-01 ×2')
        fired_at_one = self.header(**dict(self.GN_PHASE, switch_fired=True))  # (switch_k=1)
        self.assertEqual(fired_at_one, self.HEADER + ' | sw 2.5e-01 ×1 ⇒ exact')
        for name, fields in (('untracked', dict(switch_pKp=None, switch_e_n=None, switch_count=0)),
                             ('e_GN = 0', dict(switch_e_gn=0.0, switch_e_n=0.0, switch_count=0)),
                             ('e_GN inf', dict(switch_e_gn=float('inf'), switch_count=0)),
                             ('e_GN nan', dict(switch_e_gn=float('nan'), switch_count=0)),
                             ('e_N inf', dict(switch_e_n=float('inf'), switch_count=0)),
                             ('e_N nan', dict(switch_e_n=float('nan'), switch_count=0))):
            with self.subTest(name):
                self.assertEqual(self.header(**dict(self.GN_PHASE, **fields)), self.HEADER + ' | sw — ×0')

    def test_only_on_gauss_newton_phase_rows(self):
        """No ``sw`` cells on an exact-phase row (``switch_fired`` False, no count: the ``κ σ ε`` cells as in exact
        mode) or on a converged row (the short line); no ``κ`` on a Gauss-Newton-phase row, whose curvature product is
        ``switch_pKp`` (not ``pKp``); the segment sits after the wall clock and before a last row's stop reason."""
        tr = dict(step_control='trust_region', ls_steps=None, ls_failed=None, alpha=1.0, tr_radius=3.1e-2,
                  tr_accepted=True)
        exact = dict(tr, hessian='exact', pHp=8.0e-4, pKp=-2.0e-4, lift_sigma_min=4.2e-1, lift_eps=1.3e-5)
        self.assertEqual(self.header(**exact, switch_fired=False), self.header(**exact))
        self.assertTrue(self.header(**exact, switch_fired=False).endswith('| 0.31s | κ -2.0e-01 σ 4.2e-01 ε 1.3e-05'))
        gn_phase = self.header(**tr, **self.GN_PHASE)
        self.assertTrue(gn_phase.endswith('| TR Δ 3.10e-02 ✓ ‖Δx‖/‖x‖ 3.0e-02 | Δf -1.20e-04 ρ 0.98 | 0.31s '
                                          '| sw 2.5e-01 ×1'), gn_phase)
        self.assertNotIn('κ', gn_phase)
        last = self.header(**dict(self.GN_PHASE, stop_reason='max_newton'))
        self.assertTrue(last.endswith('| 0.31s | sw 2.5e-01 ×1 | stop: max_newton'), last)
        converged = self.header(converged=True, cg_iters=None, g0_reference='zero', hessian='gauss_newton',
                                switch_count=1, switch_fired=False, stop_reason='gradient')
        self.assertEqual(converged, 'iter  3 | obj 5.230e-04  ‖g‖ 1.44e-03 (2.1e-02·g₀) '
                                    '| converged (‖g‖ ≤ gtol·‖g₀‖, g₀ = zero tensor)')

    def test_unchanged_outside_switch_mode(self):
        """A row whose switch fields are all None prints byte for byte as before: the 2026.2.0 line-search texts, and
        format_newton_iter's whole block for every kind of row."""
        self.assertEqual(self.header(), self.HEADER)
        for kw, expected in TestFormatter._LS_ROWS_2026_2:
            with self.subTest(row=kw):
                self.assertEqual(bdisp._format_header(_info(**kw)), expected)
                self.assertNotIn(' sw ', expected)

    def _problem(self):
        """``newton_cg``'s docstring problem: probe derivatives of order 1, a noisy rank-3 target, the rank-2 fit
        warm-started by five Gauss-Newton steps (W1's T6: the switch fires there)."""
        np.random.seed(0)
        shape, rng = (5, 5, 5), np.random.default_rng(0)
        ww, pp = ([rng.standard_normal((40, n)) for n in shape] for _ in range(2))
        target = t3.TuckerTensorTrain.randn(shape, (3, 3, 3), (1, 3, 3, 1))
        data = [z + 0.01 * rng.standard_normal(z.shape) for z in dd.forward(target, 'probe', ww, pp, 1)]
        problem = F.derivative_fit(bgeo.ManifoldGeometryOps(), 'probe', 1, ww, pp, data)
        x0, _ = problem.newton_cg(t3.TuckerTensorTrain.randn(shape, (2, 2, 2), (1, 2, 2, 1)).data, max_newton=5)
        return problem, x0

    def test_a_switch_run(self):
        """End to end: each Gauss-Newton-phase step row's header ends with its own ratio and count (from the row's
        fields), the switching row alone with the mark; the exact rows carry ``κ`` and no ``sw``. A ``switch_tau=0`` run
        prints the Gauss-Newton trust-region run's text plus the segment (wall clocks masked): the segment is the only
        change."""
        import re
        problem, x0 = self._problem()
        printed, rows = [], []
        display, _ = bdisp.make_newton_display(problem.oracle, problem.batch, print_fn=printed.append)

        def callback(info):
            rows.append(info)
            display(info)
        _, stats = problem.newton_cg(x0, max_newton=40, hessian='switch', callback=callback)
        j = stats['switched_at']
        self.assertIsNotNone(j)
        headers = [block.split('\n')[0] for block in printed]
        for info, header in zip(rows, headers):
            with self.subTest(row=info.iteration):
                if info.converged:
                    self.assertNotIn(' sw ', header)
                elif info.hessian == 'gauss_newton':
                    ratio = '%.1e' % (info.switch_e_n / info.switch_e_gn)
                    mark = ' ⇒ exact' if info.iteration == j - 1 else ''
                    self.assertTrue(header.endswith(' | sw %s ×%d%s' % (ratio, info.switch_count, mark)), header)
                else:
                    self.assertNotIn(' sw ', header)
                    self.assertIn(' | κ ', header)
        self.assertEqual(sum('⇒ exact' in header for header in headers), 1)
        clock = re.compile(r'\| \d+\.\d\ds\b')
        segment = re.compile(r' \| sw [^|\n]*?(?= \||$)', re.MULTILINE)   # before a stop reason, or at the end
        texts = []
        for kw in (dict(hessian='switch', switch_tau=0.0), dict(step_control='trust_region')):
            text = []
            cb, _ = bdisp.make_newton_display(problem.oracle, problem.batch, print_fn=text.append)
            problem.newton_cg(x0, max_newton=6, callback=cb, **kw)
            texts.append('\n'.join(text))
        self.assertEqual(texts[0].count(' | sw '), 6)                   # every row a Gauss-Newton-phase step row
        self.assertEqual(clock.sub('| T', segment.sub('', texts[0])), clock.sub('| T', texts[1]))


if __name__ == "__main__":
    unittest.main()
