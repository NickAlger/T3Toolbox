"""Tests for the backend Newton-CG diagnostic display (backend/optimizer_display.py, D3).

The formatter is pure (returns a string), so alignment/layout are asserted on the string directly -- no
stdout capture. Plus an end-to-end `make_newton_display` + `newton_cg` run recording the error matrices.
The whole capability is backend-owned (a raw-.data user gets the identical display; the anti-drift rule)."""
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
