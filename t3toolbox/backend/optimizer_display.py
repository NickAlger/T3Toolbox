# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
"""Diagnostic display for the Newton-CG fitting loop -- **backend-owned**, so a raw-``.data`` user gets the
identical display without touching the frontend (the anti-drift rule):

    import t3toolbox.backend.optimizers as bopt
    import t3toolbox.backend.optimizer_display as bdisp
    cb, records = bdisp.make_newton_display(problem, val_sample=vs, val_data=vd)
    x, stats = bopt.newton_cg(problem, x0, callback=cb)     # prints each iter; records == the history

Two layers, so the pure algorithm modules stay pure and the I/O is isolated:

* :py:func:`format_newton_iter` -- a **pure** function returning the formatted string block (a scalar
  header line + the per-``(mode, order)`` relative-error table). Testable without capturing stdout.
* :py:func:`make_newton_display` -- builds a ``callback(NewtonInfo)`` for
  :py:func:`t3toolbox.backend.optimizers.newton_cg`: it precomputes the constant data-norm denominators
  once, then per iteration computes the train (and optional validation) error matrices, prints via an
  injectable ``print_fn``, and records each iteration.

The relative-error table is ``‖r_ij‖ / ‖y_ij‖`` from the kind's UNWEIGHTED :py:meth:`block_sumsq` (D2) --
the honest per-block recovery error, independent of any residual weight ``ω``. The **layout follows the
kind's axes** (``dev/newton_display_plan.md`` §2a): ``probe_derivatives`` (mode × order) -> mode rows,
order cols, ``train|val`` cells; a single data axis (plain ``probe`` = mode, ``apply/entries_derivatives``
= order) -> dataset rows, that axis in columns; a scalar (plain ``apply/entries``) -> a one-liner. The
stored matrices are always canonical ``(n_mode, n_order)`` -- the layout is cosmetic.

Formatting uses ``%.1e`` (Python pads the exponent to a signed 2-digit field, so every cell is exactly 7
chars -> columns align with no extra work). The callback is **host-side** (it reads the concrete residual),
so it composes with ``newton_cg(use_jit=True)`` (only the inner CG jits) but not a fully-jitted outer loop.
"""
import math
import typing as typ

import numpy as np

from t3toolbox.backend.common import *   # NDArray
import t3toolbox.backend.optimizers as bopt
from t3toolbox import corewise as _cw

__all__ = ['format_newton_iter', 'make_newton_display', 'relative_errors']


# --------------------------------------------------------------------------------------------------
# Relative-error matrix
# --------------------------------------------------------------------------------------------------
def relative_errors(
        residual_block_sumsq: NDArray,   # kind.block_sumsq(residual, n_w), shape (n_mode, n_order)
        data_block_sumsq:     NDArray,   # kind.block_sumsq(data, n_w),     shape (n_mode, n_order)
) -> NDArray:                            # sqrt(num/den) per block; NaN where the data block norm is 0
    """Per-``(mode, order)`` relative error ``‖r_ij‖ / ‖y_ij‖ = sqrt(block_sumsq(r) / block_sumsq(y))``.
    A zero-norm data block (no signal in that block) is undefined -> ``NaN`` (rendered ``—``)."""
    num = np.asarray(residual_block_sumsq, dtype=float)
    den = np.asarray(data_block_sumsq, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel = np.sqrt(num / den)
    return np.where(den > 0.0, rel, np.nan)


# --------------------------------------------------------------------------------------------------
# Pure formatting
# --------------------------------------------------------------------------------------------------
def _cell(value: typ.Optional[float], fmt: str, width: int) -> str:
    """One right-justified fixed-width cell; a ``None`` / non-finite value renders as ``—``."""
    if value is None or not math.isfinite(float(value)):
        return '—'.rjust(width)
    return (fmt % float(value)).rjust(width)


def _grid(col_labels: typ.Sequence[str], row_labels: typ.Sequence[str],
          cells: typ.Sequence[typ.Sequence[str]], lead: str = '    ', gap: str = '  ') -> str:
    """An aligned text grid: per-column width = max(label, widest cell); row labels left-justified."""
    ncol = len(col_labels)
    col_w = [max([len(col_labels[c])] + [len(cells[r][c]) for r in range(len(row_labels))])
             for c in range(ncol)]
    row_lab_w = max([len(rl) for rl in row_labels], default=0)
    header = lead + ' ' * row_lab_w + gap + gap.join(col_labels[c].rjust(col_w[c]) for c in range(ncol))
    lines = [header]
    for r, rl in enumerate(row_labels):
        lines.append(lead + rl.ljust(row_lab_w) + gap
                     + gap.join(cells[r][c].rjust(col_w[c]) for c in range(ncol)))
    return '\n'.join(lines)


def _format_table(train_err: NDArray, val_err: typ.Optional[NDArray], fmt: str) -> str:
    """The relative-error table, layout chosen from the matrix shape (Decision 2a)."""
    train_err = np.asarray(train_err)
    nm, no = train_err.shape
    w = len(fmt % 1.0)                                   # cell width (7 for %.1e)
    has_val = val_err is not None
    val_err = np.asarray(val_err) if has_val else None

    if nm == 1 and no == 1:                              # scalar -> one line
        t = _cell(train_err[0, 0], fmt, w).strip()
        if has_val:
            return "  rel err   train %s | val %s" % (t, _cell(val_err[0, 0], fmt, w).strip())
        return "  rel err   %s" % t

    if nm > 1 and no > 1:                                # 2 data axes -> mode rows, order cols, train|val cells
        col_labels = ["ord%d" % t for t in range(no)]
        row_labels = ["m%d" % i for i in range(nm)]
        cells = [[(_cell(train_err[i, t], fmt, w) + "|" + _cell(val_err[i, t], fmt, w)) if has_val
                  else _cell(train_err[i, t], fmt, w) for t in range(no)] for i in range(nm)]
        legend = "  rel err  rows=mode  cols=order" + ("  (cells train|val)" if has_val else "")
        return legend + "\n" + _grid(col_labels, row_labels, cells)

    # 1 data axis -> dataset rows, the (>1) data axis in columns
    is_mode = nm > 1
    n = nm if is_mode else no
    col_labels = [("m%d" if is_mode else "ord%d") % j for j in range(n)]
    at = (lambda err, j: err[j, 0]) if is_mode else (lambda err, j: err[0, j])
    row_labels = ["train"]
    cells = [[_cell(at(train_err, j), fmt, w) for j in range(n)]]
    if has_val:
        row_labels.append("val")
        cells.append([_cell(at(val_err, j), fmt, w) for j in range(n)])
    legend = "  rel err  rows=train/val  cols=%s" % ("mode" if is_mode else "order")
    return legend + "\n" + _grid(col_labels, row_labels, cells)


def _format_header(info: "bopt.NewtonInfo", obj_unweighted: typ.Optional[float]) -> str:
    """The one-line per-iteration scalar header (a shorter form on the converged final line)."""
    obj = "obj %.3e" % info.objective
    if info.regularization is not None and info.misfit is not None:   # regularizer attached -> obj = misfit + reg
        misfit_str = "%.3e" % info.misfit
        if obj_unweighted is not None and not math.isclose(obj_unweighted, info.misfit, rel_tol=1e-9):
            misfit_str += " (unwt %.2e)" % obj_unweighted             # ω-weighting gap sits on the misfit term
        obj += " = misfit %s + reg %.3e" % (misfit_str, info.regularization)
    elif obj_unweighted is not None and not math.isclose(obj_unweighted, info.objective, rel_tol=1e-9):
        obj += " (unwt %.2e)" % obj_unweighted                        # unregularized + nontrivial ω: unweighted vs total
    rel_g = info.gnorm / info.g0norm if info.g0norm else float('nan')
    obj += "  ‖g‖ %.2e (%.1e·g₀)" % (info.gnorm, rel_g)
    parts = ["iter %2d" % info.iteration, obj]
    if info.converged:
        parts.append("converged (‖g‖ ≤ gtol·‖g₀‖)")
        return " | ".join(parts)
    sym = "✓" if info.cg_converged else ("⌇" if info.cg_truncated else "⋯")   # tol / truncated / maxiter
    parts.append("CG %d/%d tol %.1e resid %.1e %s"
                 % (info.cg_iters, info.cg_maxiter, info.cg_tol, info.cg_resid, sym))
    parts.append("ls %d%s α %.2e ‖Δx‖/‖x‖ %.1e" % (
        info.ls_steps, '!' if getattr(info, 'ls_failed', False) else '', info.alpha, info.step_rel))
    parts.append("Δf %+.2e ρ %.2f" % (info.delta_f, info.rho))
    parts.append("%.2fs" % info.wall_time)
    return " | ".join(parts)


def format_newton_iter(
        info:           "bopt.NewtonInfo",       # the per-iteration diagnostics from newton_cg
        train_err:      NDArray,                  # (n_mode, n_order) training relative-error matrix
        val_err:        typ.Optional[NDArray] = None,   # (n_mode, n_order) validation matrix, or None
        obj_unweighted: typ.Optional[float]   = None,   # ½‖r‖²; shown next to the misfit iff a nontrivial ω makes it differ
        fmt:            str = '%.1e',
) -> str:                                         # the formatted header line + relative-error table
    """Format one Newton iteration as a header line + the relative-error table (pure -- returns a string).
    The layout of the table follows ``train_err``'s shape (Decision 2a). When a regularizer is attached the
    header splits the objective as ``obj = misfit + reg`` (``info.misfit`` / ``info.regularization``);
    ``obj_unweighted`` (the unweighted ½‖r‖²) is shown next to the **misfit** term only when a nontrivial
    residual weight ``ω`` makes it differ (unregularized, it annotates the objective, as before).

    Examples
    --------
    A converged final iteration (short header) with a 2-mode x 2-order error matrix; note ``1.0e+00`` and
    ``5.2e-04`` line up -- ``%.1e`` is fixed 7-char width, so columns align with no padding logic:

    >>> import numpy as np
    >>> import t3toolbox.backend.optimizers as bopt
    >>> import t3toolbox.backend.optimizer_display as disp
    >>> info = bopt.NewtonInfo(iteration=5, objective=3.78e-4, gnorm=4.8e-6, g0norm=6.5e-2, converged=True)
    >>> train = np.array([[1.9e-2, 1.3e-2], [1.0e+00, 5.2e-4]])   # (2 modes, 2 orders)
    >>> print(disp.format_newton_iter(info, train))
    iter  5 | obj 3.780e-04  ‖g‖ 4.80e-06 (7.4e-05·g₀) | converged (‖g‖ ≤ gtol·‖g₀‖)
      rel err  rows=mode  cols=order
               ord0     ord1
        m0  1.9e-02  1.3e-02
        m1  1.0e+00  5.2e-04

    With a regularizer attached, the objective splits as ``obj = misfit + reg`` (here a single apply
    dataset, so a 1x1 error table):

    >>> info = bopt.NewtonInfo(iteration=3, objective=6.003e1, gnorm=1.60, g0norm=1.07e1, converged=False,
    ...                        misfit=5.974e1, regularization=2.9e-1, cg_iters=8, cg_maxiter=200,
    ...                        cg_tol=6.2e-1, cg_resid=4.9e-1, cg_converged=True, cg_truncated=False,
    ...                        ls_steps=0, alpha=1.0, step_rel=1.2, delta_f=-2.22, rho=0.07, wall_time=0.03)
    >>> print(disp.format_newton_iter(info, np.array([[3.7e-1]])))
    iter  3 | obj 6.003e+01 = misfit 5.974e+01 + reg 2.900e-01  ‖g‖ 1.60e+00 (1.5e-01·g₀) | CG 8/200 tol 6.2e-01 resid 4.9e-01 ✓ | ls 0 α 1.00e+00 ‖Δx‖/‖x‖ 1.2e+00 | Δf -2.22e+00 ρ 0.07 | 0.03s
      rel err   3.7e-01
    """
    return _format_header(info, obj_unweighted) + "\n" + _format_table(train_err, val_err, fmt)


# --------------------------------------------------------------------------------------------------
# The callback builder
# --------------------------------------------------------------------------------------------------
def make_newton_display(
        problem,                                  # backend.optimizers.Problem (holds the kind + full data)
        val_sample: typ.Any = None,               # optional validation sample (same kind/layout as problem.sample)
        val_data:   typ.Any = None,               # optional validation data; both given -> a train|val table
        print_fn:   typ.Optional[typ.Callable] = print,   # where to send each iteration's text (None = silent)
        fmt:        str  = '%.1e',
        record:     bool = True,                  # accumulate per-iteration records (scalars + err matrices)
) -> typ.Tuple[typ.Callable, list]:               # (callback for newton_cg, records list filled as it runs)
    """Build a ``callback(NewtonInfo)`` (+ its ``records`` list) that displays each Newton iteration.

    Precomputes the constant per-block **data norms** ``block_sumsq(data)`` once (the relative-error
    denominators). Each call: recomputes the residual block norms, forms the train (and, if ``val_data``
    is given, validation -- one extra ``point_forward``, no transpose/sweep) relative-error matrices,
    prints via ``print_fn``, and appends a self-contained record (the scalar fields + ``train_err`` /
    ``val_err``). Requires ``problem.kind.has_block_sumsq`` (all built-in kinds have it)."""
    kind = problem.kind
    if not kind.has_block_sumsq:
        raise ValueError("this sampling kind has no block_sumsq -- the relative-error table needs it "
                         "(all built-in kinds provide it; a custom kind must implement block_sumsq and "
                         "set has_block_sumsq = True).")
    n_w = kind.w_axes(problem.sample)
    data_bs = np.asarray(kind.block_sumsq(problem.data, n_w), dtype=float)
    if (val_sample is None) != (val_data is None):
        raise ValueError("make_newton_display: pass val_sample and val_data together, or neither "
                         "(val_sample=%s, val_data=%s)" % ('None' if val_sample is None else 'given',
                                                           'None' if val_data is None else 'given'))
    have_val = val_data is not None
    n_w_val = kind.w_axes(val_sample) if have_val else None
    val_data_bs = np.asarray(kind.block_sumsq(val_data, n_w_val), dtype=float) if have_val else None
    records: list = []

    def callback(info):
        res_bs = np.asarray(kind.block_sumsq(info.lm.residual, n_w), dtype=float)
        train_err = relative_errors(res_bs, data_bs)
        val_err = None
        if have_val:
            r_val = _cw.corewise_sub(kind.point_forward(info.x_cores, val_sample), val_data)
            val_err = relative_errors(kind.block_sumsq(r_val, n_w_val), val_data_bs)
        obj_unweighted = 0.5 * float(res_bs.sum())
        if print_fn is not None:
            print_fn(format_newton_iter(info, train_err, val_err, obj_unweighted, fmt))
        if record:
            rec = dict(bopt._newton_scalar_record(info))
            rec['train_err'] = train_err
            if val_err is not None:
                rec['val_err'] = val_err
            records.append(rec)

    return callback, records
