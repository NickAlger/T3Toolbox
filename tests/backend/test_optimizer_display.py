"""Tests for the backend Newton-CG diagnostic display (backend/optimizer_display.py, D3).

The formatter is pure (returns a string), so alignment/layout are asserted on the string directly -- no
stdout capture. Plus an end-to-end `make_newton_display` + `newton_cg` run recording the error matrices.
The whole capability is backend-owned (a raw-.data user gets the identical display; the anti-drift rule)."""
import dataclasses as dc
import unittest

import numpy as np

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.backend.optimizers as bopt
import t3toolbox.backend.geometry as bgeo
import t3toolbox.backend.fitting as bfit
import t3toolbox.backend.optimizer_display as bdisp

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
        table = bdisp._format_table(train, val, "%.1e")
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
        table = bdisp._format_table(train, val, "%.1e")
        self.assertIn("rows=train/val", table); self.assertIn("cols=mode", table)
        lines = _grid_lines(table)
        self.assertEqual(len(lines), 1 + 2)                              # header + train + val
        self.assertTrue(lines[1].lstrip().startswith("train"))
        self.assertTrue(lines[2].lstrip().startswith("val"))
        self.assertEqual(len({len(ln) for ln in lines}), 1)

    def test_one_axis_apply_derivatives(self):
        """apply/entries derivatives (1 data axis = order): dataset rows, order cols."""
        train = np.array([[5.2e-4, 2.7e-2, 9.1e-2, 1.8e-1]])            # (1 mode, 4 orders)
        table = bdisp._format_table(train, None, "%.1e")               # train-only
        self.assertIn("cols=order", table)
        lines = _grid_lines(table)
        self.assertEqual(len(lines), 1 + 1)                             # header + single train row (no val)
        self.assertEqual(len({len(ln) for ln in lines}), 1)

    def test_scalar_one_liner(self):
        """Plain apply/entries (0 data axes): a single line, no grid."""
        table = bdisp._format_table(np.array([[5.2e-4]]), np.array([[6.1e-4]]), "%.1e")
        self.assertNotIn("\n", table)
        self.assertIn("train", table); self.assertIn("val", table)

    def test_zero_norm_block_dash(self):
        """A zero-norm data block -> NaN relative error -> rendered `—`."""
        rel = bdisp.relative_errors(np.array([[1.0, 4.0]]), np.array([[4.0, 0.0]]))
        self.assertTrue(np.isclose(rel[0, 0], 0.5))
        self.assertTrue(np.isnan(rel[0, 1]))
        table = bdisp._format_table(rel, None, "%.1e")
        self.assertIn("—", table)

    def test_header_converged_vs_stepped(self):
        """The header is a full step line normally, a short `converged` line on the final iteration."""
        full = bdisp._format_header(_info(), obj_unweighted=None)
        self.assertIn("CG 14/200", full); self.assertIn("ρ 0.98", full); self.assertIn("✓", full)
        conv = bdisp._format_header(_info(converged=True, cg_iters=None, cg_converged=None), None)
        self.assertIn("converged", conv); self.assertNotIn("CG ", conv)

    def test_converged_line_names_its_reference(self):
        """T29 / C.14: the converged line names the Newton-stop reference (`NewtonInfo.g0_reference`); a
        hand-built row with no reference kind prints exactly as in 2026.2.0 (a display change only)."""
        conv = dict(converged=True, cg_iters=None, cg_converged=None)
        self.assertTrue(bdisp._format_header(_info(**conv), None).endswith("| converged (‖g‖ ≤ gtol·‖g₀‖)"))
        for kind, name in (('zero', 'zero tensor'), ('warm_start', 'warm start'), ('override', 'override')):
            with self.subTest(g0_reference=kind):
                line = bdisp._format_header(_info(g0_reference=kind, **conv), None)
                self.assertTrue(line.endswith("| converged (‖g‖ ≤ gtol·‖g₀‖, g₀ = %s)" % name), line)
        stepped = bdisp._format_header(_info(g0_reference='zero'), None)     # a stepped row: no reference text
        self.assertNotIn("g₀ =", stepped)

    def test_header_unweighted_shown_only_when_differs(self):
        """`(unwt …)` appears next to the objective only when the unweighted value differs (ω ≠ 1)."""
        self.assertNotIn("unwt", bdisp._format_header(_info(objective=1.0), obj_unweighted=1.0))
        self.assertIn("unwt", bdisp._format_header(_info(objective=1.0), obj_unweighted=2.5))

    def test_header_misfit_reg_breakdown(self):
        """With a regularizer attached the objective splits as `obj = misfit + reg`; unregularized (the
        default, `regularization=None`) shows no split. The `(unwt …)` gap sits on the misfit and reflects
        ω alone -- it must NOT fire merely because a regularizer shifts the total (the old comparison bug)."""
        plain = bdisp._format_header(_info(objective=1.0), obj_unweighted=None)
        self.assertNotIn("misfit", plain); self.assertNotIn("+ reg", plain)      # regularization None -> no split
        reg = bdisp._format_header(_info(objective=1.29, misfit=1.0, regularization=0.29), obj_unweighted=None)
        self.assertIn("= misfit 1.000e+00 + reg 2.900e-01", reg)
        self.assertNotIn("unwt", reg)                                            # no ω -> misfit == unweighted
        reg_w = bdisp._format_header(_info(objective=1.29, misfit=1.0, regularization=0.29), obj_unweighted=0.8)
        self.assertIn("misfit 1.000e+00 (unwt 8.00e-01)", reg_w)                 # ω gap annotates the misfit term
        reg_nobug = bdisp._format_header(_info(objective=1.29, misfit=1.0, regularization=0.29), obj_unweighted=1.0)
        self.assertNotIn("unwt", reg_nobug)   # unwt == misfit (only reg shifts the total) -> no spurious (unwt …)

    def test_truncated_symbol(self):
        h = bdisp._format_header(_info(cg_converged=False, cg_truncated=True), None)
        self.assertIn("⌇", h)

    # The 2026.2.0 header text, captured from the pre-trust-region formatter (fffdfde) for these rows: the
    # line-search `step` segment -- and every non-final Gauss-Newton row -- must stay byte-identical.
    _LS_ROWS_2026_2 = [
        (dict(),
         'iter  3 | obj 5.230e-04 (unwt 1.70e-04)  ‖g‖ 1.44e-03 (2.1e-02·g₀) | CG 14/200 tol 3.0e-04 resid 8.7e-05 ✓ '
         '| ls 2 α 2.50e-01 ‖Δx‖/‖x‖ 3.0e-02 | Δf -1.20e-04 ρ 0.98 | 0.31s'),
        (dict(ls_failed=True, ls_steps=40, alpha=2.0 ** -39, cg_converged=False, cg_truncated=True),
         'iter  3 | obj 5.230e-04 (unwt 1.70e-04)  ‖g‖ 1.44e-03 (2.1e-02·g₀) | CG 14/200 tol 3.0e-04 resid 8.7e-05 ⌇ '
         '| ls 40! α 1.82e-12 ‖Δx‖/‖x‖ 3.0e-02 | Δf -1.20e-04 ρ 0.98 | 0.31s'),
        (dict(cg_converged=False, step_rel=float('nan'), rho=float('nan'), misfit=1e-4, regularization=4.23e-4),
         'iter  3 | obj 5.230e-04 = misfit 1.000e-04 (unwt 1.70e-04) + reg 4.230e-04  ‖g‖ 1.44e-03 (2.1e-02·g₀) '
         '| CG 14/200 tol 3.0e-04 resid 8.7e-05 ⋯ | ls 2 α 2.50e-01 ‖Δx‖/‖x‖ nan | Δf -1.20e-04 ρ nan | 0.31s'),
        (dict(cg_converged=False, g0_reference='zero'),
         'iter  3 | obj 5.230e-04 (unwt 1.70e-04)  ‖g‖ 1.44e-03 (2.1e-02·g₀) | CG 14/200 tol 3.0e-04 resid 8.7e-05 ⋯ '
         '| ls 2 α 2.50e-01 ‖Δx‖/‖x‖ 3.0e-02 | Δf -1.20e-04 ρ 0.98 | 0.31s'),
    ]

    def test_line_search_rows_are_byte_identical(self):
        """C.14: the line-search `step` segment is the 2026.2.0 text byte for byte (a Gauss-Newton row of
        newton_cg also carries step_control / hessian / cg_boundary=False now -- none of which shows)."""
        for kw, expected in self._LS_ROWS_2026_2:
            for extra in (dict(), dict(step_control='line_search', hessian='gauss_newton', cg_boundary=False)):
                with self.subTest(row=kw, extra=extra):
                    self.assertEqual(bdisp._format_header(_info(**kw, **extra), obj_unweighted=1.7e-4), expected)

    def test_exact_mode_header(self):
        """T19: the trust-region `step` segment (`TR Δ … ✓/✗`), the boundary symbol `∂`, the exact cells
        `κ σ ε` (κ guarded against a zero denominator, σ / ε `—` when absent), and the stop reason on a run's
        last row (not on the converged line, which says so already)."""
        tr = dict(step_control='trust_region', hessian='exact', ls_steps=None, ls_failed=None, alpha=1.0,
                  tr_radius=3.1e-2, tr_accepted=True, pHp=8.0e-4, pKp=-2.0e-4, lift_sigma_min=4.2e-1,
                  lift_eps=1.3e-5)
        accepted = bdisp._format_header(_info(**tr), None)
        self.assertIn("| TR Δ 3.10e-02 ✓ ‖Δx‖/‖x‖ 3.0e-02 |", accepted)
        self.assertNotIn("ls ", accepted)
        self.assertTrue(accepted.endswith("| 0.31s | κ -2.0e-01 σ 4.2e-01 ε 1.3e-05"), accepted)   # κ = pKp/(pHp − pKp)
        self.assertIn("TR Δ 3.10e-02 ✗", bdisp._format_header(_info(**dict(tr, tr_accepted=False)), None))
        boundary = bdisp._format_header(_info(**dict(tr, cg_converged=False, cg_boundary=True)), None)
        self.assertIn("resid 8.7e-05 ∂ |", boundary)
        self.assertIn(" κ — ", bdisp._format_header(_info(**dict(tr, pKp=8.0e-4)), None))         # pHp − pKp = 0
        self.assertIn(" κ — ", bdisp._format_header(_info(**dict(tr, pKp=float('nan'))), None))
        absent = bdisp._format_header(_info(**dict(tr, lift_sigma_min=None, lift_eps=None)), None)
        self.assertTrue(absent.endswith("κ -2.0e-01 σ — ε —"), absent)                               # corewise / d = 1
        nan_radius = bdisp._format_header(_info(**dict(tr, tr_radius=float('nan'), step_rel=float('inf'))), None)
        self.assertIn("TR Δ — ✓ ‖Δx‖/‖x‖ — |", nan_radius)
        self.assertNotIn("κ", bdisp._format_header(_info(**dict(tr, hessian='gauss_newton', pKp=None)), None))
        for reason in ('objective_floor', 'trust_region_floor', 'max_newton', 'line_search'):
            with self.subTest(stop_reason=reason):
                self.assertTrue(bdisp._format_header(_info(**dict(tr, stop_reason=reason)), None)
                                .endswith("| stop: %s" % reason))
        conv = bdisp._format_header(_info(converged=True, cg_iters=None, stop_reason='gradient', g0_reference='zero',
                                          hessian='exact', step_control='trust_region'), None)
        self.assertTrue(conv.endswith("| converged (‖g‖ ≤ gtol·‖g₀‖, g₀ = zero tensor)"), conv)


class TestMakeNewtonDisplay(unittest.TestCase):
    def _probe_problem(self, M=120):
        np.random.seed(0); rng = np.random.default_rng(0)
        A = t3.TuckerTensorTrain.randn(SHAPE, TUCKER, TT)
        ww = [rng.standard_normal((M, N)) for N in SHAPE]
        ww = [w / np.linalg.norm(w, axis=1, keepdims=True) for w in ww]
        def dprobe(A, ww):
            d = len(ww); out = []
            for f in range(d):
                ops = [A.to_dense(), list(range(d))]
                for j in range(d):
                    if j != f:
                        ops += [ww[j], [d, j]]
                ops += [[d, f]]; out.append(np.einsum(*ops))
            return out
        data = dprobe(A, ww)
        return bopt.least_squares_problem(bgeo.ManifoldGeometryOps(), bfit.PROBE, ww, data), ww, data, dprobe, A

    def test_records_and_silent(self):
        """make_newton_display records a self-contained per-iteration dict (scalars + train_err/val_err)
        and honours print_fn=None (silent). One extra val forward per iter; d-mode error matrices."""
        prob, ww, data, dprobe, A = self._probe_problem()
        rng = np.random.default_rng(1)
        wwv = [w / np.linalg.norm(w, axis=1, keepdims=True) for w in (rng.standard_normal((50, N)) for N in SHAPE)]
        datav = dprobe(A, wwv)
        x0 = t3.TuckerTensorTrain.zeros(SHAPE, TUCKER, TT).data
        cb, records = bdisp.make_newton_display(prob, wwv, datav, print_fn=None)   # silent
        x, stats = bopt.newton_cg(prob, x0, max_newton=5, callback=cb)
        self.assertEqual(len(records), len(stats['history']))
        for rec in records:
            self.assertIn('iteration', rec)                            # scalar fields present
            self.assertEqual(np.asarray(rec['train_err']).shape, (len(SHAPE), 1))   # plain probe: (d, 1)
            self.assertEqual(np.asarray(rec['val_err']).shape, (len(SHAPE), 1))
            self.assertTrue(np.all(np.asarray(rec['train_err']) >= 0))
        # the fit improves -> final train error below the initial
        self.assertLess(float(records[-1]['train_err'].mean()), float(records[0]['train_err'].mean()))

    def test_rows_after_a_rejection_reuse_the_tables(self):
        """n14: a rejected trust-region trial keeps x (and its model), so the NEXT row reuses that row's error
        tables -- no validation forward for an unchanged point. The validation forwards counted on an exact run
        with rejections equal the number of distinct points shown."""
        import unittest.mock
        rng = np.random.default_rng(0)
        np.random.seed(0)
        shape = (6, 6, 6)
        target = t3.TuckerTensorTrain.randn(shape, (3, 3, 3), (1, 3, 3, 1)).data
        ww, pp, wv, pv = ([rng.standard_normal((n_w, n)) for n in shape] for n_w in (40, 40, 15, 15))
        kind = bfit.ProbeDerivativesKind(order=1)
        prob = bopt.least_squares_problem(bgeo.ManifoldGeometryOps(), kind, (ww, pp), kind.point_forward(target, (ww, pp)))
        val_sample = (wv, pv)
        cb, records = bdisp.make_newton_display(prob, val_sample, kind.point_forward(target, val_sample), print_fn=None)
        np.random.seed(5)
        x0 = t3.TuckerTensorTrain.randn(shape, (2, 2, 2), (1, 2, 2, 1)).data
        original = bfit.ProbeDerivativesKind.point_forward
        with unittest.mock.patch.object(bfit.ProbeDerivativesKind, 'point_forward', autospec=True,
                                        side_effect=original) as spy:
            _, stats = bopt.newton_cg(prob, x0, max_newton=10, hessian='exact', tr_radius0=1e3, callback=cb)
        val_forwards = sum(1 for call in spy.call_args_list if call.args[2] is val_sample)
        rows = stats['history']
        after_rejection = [k for k in range(1, len(rows)) if rows[k - 1]['tr_accepted'] is False]
        self.assertTrue(after_rejection, 'expected a rejected trial in this run')
        self.assertEqual(val_forwards, len(rows) - len(after_rejection))
        for k in after_rejection:
            self.assertIs(records[k]['train_err'], records[k - 1]['train_err'])
            self.assertIs(records[k]['val_err'], records[k - 1]['val_err'])
        for k in set(range(1, len(rows))) - set(after_rejection):
            self.assertIsNot(records[k]['val_err'], records[k - 1]['val_err'])
        # One callback shared by two runs (e.g. continuation levels): the first run ENDS on a rejected trial (as a
        # floor stop always does); the second run's first row is a new point and recomputes its tables.
        first_rejection = after_rejection[0]                                     # rows[first_rejection - 1] is rejected
        cb2, records2 = bdisp.make_newton_display(prob, val_sample, kind.point_forward(target, val_sample),
                                                  print_fn=None)
        bopt.newton_cg(prob, x0, max_newton=first_rejection, hessian='exact', tr_radius0=1e3, callback=cb2)
        self.assertIs(records2[-1]['tr_accepted'], False)
        np.random.seed(6)
        bopt.newton_cg(prob, t3.TuckerTensorTrain.randn(shape, (2, 2, 2), (1, 2, 2, 1)).data, max_newton=1,
                       callback=cb2)
        self.assertIsNot(records2[-1]['train_err'], records2[-2]['train_err'])
        self.assertFalse(np.allclose(records2[-1]['val_err'], records2[-2]['val_err']))

    def test_needs_block_sumsq(self):
        """A kind that does not provide block_sumsq raises a clear error (all built-in kinds do).

        `block_sumsq` is optional, so ``has_block_sumsq`` is declared by whoever IMPLEMENTS it -- False on
        the base, True on the two output-shape bases. The case that matters is a user kind written from
        the base that simply omits it: it must trip the friendly guard, not sail through to a bare
        NotImplementedError from the base method (which is what an inherited True default caused)."""
        import dataclasses as dc

        prob, ww, data, _, _ = self._probe_problem()

        @dc.dataclass(frozen=True, eq=False)
        class KindOmittingBlockSumsq(bfit.SamplingKind):
            """A user kind written from the base, implementing the operations but not block_sumsq."""
            def w_axes(self, sample):
                return 1

        self.assertFalse(KindOmittingBlockSumsq().has_block_sumsq)
        with self.assertRaises(ValueError):
            bdisp.make_newton_display(dc.replace(prob, kind=KindOmittingBlockSumsq()))

        @dc.dataclass(frozen=True, eq=False)                     # and the explicit opt-out still works
        class ProbeKindOptingOut(bfit.ProbeKind):
            has_block_sumsq = False

        with self.assertRaises(ValueError):
            bdisp.make_newton_display(dc.replace(prob, kind=ProbeKindOptingOut()))

        for builtin in (bfit.APPLY, bfit.ENTRIES, bfit.PROBE, bfit.probe_derivatives_kind(2)):
            self.assertTrue(builtin.has_block_sumsq, f'{builtin.name} implements block_sumsq')



def _seconds(**kw):
    """A PhaseSeconds of a Gauss-Newton line-search row after row 0 (model built, three trials); override via kwargs."""
    base = dict(previous_callback=0.01, frame=0.05, sweep=0.06, companion=0.001, residual=0.01, curvature=None,
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
        plain = bdisp.format_newton_iter(_info(), train)
        timed = bdisp.format_newton_iter(_info(timing=_timing()), train)
        self.assertNotIn('  time  ', plain)
        lines = timed.split('\n')
        self.assertTrue(lines[1].startswith('  time  model '), lines[1])
        self.assertEqual('\n'.join(lines[:1] + lines[2:]), plain)

    def test_the_time_line_cells(self):
        """Each cell, by the record that sets it: a phase that did not run is left out; seconds as ``%.1e``."""
        line = bdisp._format_timing(_info(timing=_timing()))
        self.assertEqual(line, '  time  model 1.2e-01 [frame 5.0e-02 sweep 6.0e-02 comp 1.0e-03 resid 1.0e-02] | '
                               'grad 3.0e-02 | cg 4.0e-01 (2.9e-02/it) | forms 2.0e-02 | '
                               'trials 2.1e-01 ×3 [retr 1.5e-01 obj 6.0e-02] (accepted 7.0e-02) | cb 1.0e-02 | '
                               'rest 1.0e-02')                           # cg per iteration: 0.40 / 14
        cell = lambda **kw: bdisp._format_timing(_info(**{'timing': _timing(), **kw}))
        one_trial = cell(timing=_timing(trials=1, accepted_retraction=0.15))
        self.assertIn('| trials 2.1e-01 ×1 [retr 1.5e-01 obj 6.0e-02] | cb', one_trial)       # no 'accepted' share
        rejected = cell(timing=_timing(trials=1, accepted_retraction=None))
        self.assertNotIn('accepted', rejected)
        exact = cell(timing=_timing(_seconds(curvature=0.2)))
        self.assertIn('model 3.2e-01 [frame 5.0e-02 sweep 6.0e-02 comp 1.0e-03 resid 1.0e-02 curv 2.0e-01] |', exact)
        tracked = cell(timing=_timing(_seconds(tracking=0.05)))
        self.assertIn('(accepted 7.0e-02) | track 5.0e-02 | cb 1.0e-02 |', tracked)
        reused = cell(timing=_timing(_seconds(frame=None, sweep=None, companion=None, residual=None, gradient=None)))
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
        self.assertEqual(list(rows), ['previous_callback', 'frame', 'sweep', 'companion', 'residual', 'gradient',
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
        target = t3.TuckerTensorTrain.randn(shape, (3, 3, 3), (1, 3, 3, 1)).data
        ww, pp = ([rng.standard_normal((40, n)) for n in shape] for _ in range(2))
        kind = bfit.ProbeDerivativesKind(order=1)
        data = [z + 0.01 * rng.standard_normal(z.shape) for z in kind.point_forward(target, (ww, pp))]
        problem = bopt.least_squares_problem(bgeo.ManifoldGeometryOps(), kind, (ww, pp), data)
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
                cb, _ = bdisp.make_newton_display(problem, print_fn=timed.append)
                _, stats = bopt.newton_cg(problem, x0, timing=True, callback=cb, **kw)
                cb_plain, _ = bdisp.make_newton_display(problem, print_fn=untimed.append)
                bopt.newton_cg(problem, x0, callback=cb_plain, **kw)
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
                again = bopt.newton_cg(problem, x0, timing=True, callback=cb, max_newton=2)[1]   # the same callback
                self.assertEqual(len(again['history']), 2)
                self.assertTrue(timed[-1].startswith('  timing  2 rows, '), timed[-1])           # this run's rows only
                self.assertEqual(sum(text.startswith('  timing  ') for text in timed), 2)

    def test_the_last_row_closes_the_table(self):
        """The table follows a converged row without a stop reason (a hand-replayed run), and only the last row."""
        problem, x0 = self._problem()
        infos = []
        bopt.newton_cg(problem, x0, timing=True, max_newton=3, callback=infos.append)
        printed = []
        cb, _ = bdisp.make_newton_display(problem, print_fn=printed.append)
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
                cb, timed = bdisp.make_newton_display(problem, print_fn=None)
                bopt.newton_cg(problem, x0, timing=True, callback=cb, **kw)
                cb_plain, untimed = bdisp.make_newton_display(problem, print_fn=None)
                bopt.newton_cg(problem, x0, callback=cb_plain, **kw)
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
        return bdisp._format_header(_info(**kw), None)

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
                self.assertEqual(bdisp._format_header(_info(**kw), obj_unweighted=1.7e-4), expected)
                self.assertNotIn(' sw ', expected)

    def _problem(self):
        """``newton_cg``'s docstring problem: probe derivatives of order 1, a noisy rank-3 target, the rank-2 fit
        warm-started by five Gauss-Newton steps (W1's T6: the switch fires there)."""
        np.random.seed(0)
        shape, rng = (5, 5, 5), np.random.default_rng(0)
        ww, pp = ([rng.standard_normal((40, n)) for n in shape] for _ in range(2))
        kind = bfit.ProbeDerivativesKind(order=1)
        target = t3.TuckerTensorTrain.randn(shape, (3, 3, 3), (1, 3, 3, 1)).data
        data = [z + 0.01 * rng.standard_normal(z.shape) for z in kind.point_forward(target, (ww, pp))]
        problem = bopt.least_squares_problem(bgeo.ManifoldGeometryOps(), kind, (ww, pp), data)
        x0, _ = bopt.newton_cg(problem, t3.TuckerTensorTrain.randn(shape, (2, 2, 2), (1, 2, 2, 1)).data, max_newton=5)
        return problem, x0

    def test_a_switch_run(self):
        """End to end: each Gauss-Newton-phase step row's header ends with its own ratio and count (from the row's
        fields), the switching row alone with the mark; the exact rows carry ``κ`` and no ``sw``. A ``switch_tau=0`` run
        prints the Gauss-Newton trust-region run's text plus the segment (wall clocks masked): the segment is the only
        change."""
        import re
        problem, x0 = self._problem()
        printed, rows = [], []
        display, _ = bdisp.make_newton_display(problem, print_fn=printed.append)

        def callback(info):
            rows.append(info)
            display(info)
        _, stats = bopt.newton_cg(problem, x0, max_newton=40, hessian='switch', callback=callback)
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
            cb, _ = bdisp.make_newton_display(problem, print_fn=text.append)
            bopt.newton_cg(problem, x0, max_newton=6, callback=cb, **kw)
            texts.append('\n'.join(text))
        self.assertEqual(texts[0].count(' | sw '), 6)                   # every row a Gauss-Newton-phase step row
        self.assertEqual(clock.sub('| T', segment.sub('', texts[0])), clock.sub('| T', texts[1]))


class TestValidationArgsGoTogether(unittest.TestCase):
    """Review 2026-08-22 (C9): val_data without val_sample used to fail deep inside the kind, and
    val_sample without val_data was silently ignored."""

    def test_one_without_the_other_raises(self):
        np.random.seed(4)
        x = t3.TuckerTensorTrain.randn((4, 5, 3), (2, 2, 2), (1, 2, 2, 1))
        ww = tuple(np.random.randn(6, n) for n in x.shape)
        problem = bopt.least_squares_problem(bgeo.ManifoldGeometryOps(), bfit.APPLY, ww, x.apply(ww))
        with self.assertRaises(ValueError):
            bdisp.make_newton_display(problem, val_data=x.apply(ww))
        with self.assertRaises(ValueError):
            bdisp.make_newton_display(problem, val_sample=ww)

if __name__ == "__main__":
    unittest.main()
