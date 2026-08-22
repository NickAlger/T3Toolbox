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
