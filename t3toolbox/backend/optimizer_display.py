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

A run with ``newton_cg(timing=True)`` adds a ``time`` line to each row's block (the row's window by phase) and a
closing table after the run's last row; untimed rows print exactly as before.

The relative-error table is ``‖r_ij‖ / ‖y_ij‖`` from the kind's UNWEIGHTED :py:meth:`block_sumsq` (D2) --
the honest per-block recovery error, independent of any residual weight ``ω``. The **layout follows the
kind's axes**: ``probe_derivatives`` (mode × order) -> mode rows,
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


_REFERENCE_NAMES = {'zero': 'zero tensor', 'warm_start': 'warm start', 'override': 'override'}   # NewtonInfo.g0_reference


def _num(value: typ.Optional[float], fmt: str) -> str:
    """A number in ``fmt``, or ``—`` for ``None`` / non-finite (an unpadded :py:func:`_cell`)."""
    return _cell(value, fmt, 0)


def _cg_symbol(info: "bopt.NewtonInfo") -> str:
    """How CG stopped: ``✓`` tolerance, ``∂`` the trust-region boundary, ``⌇`` truncated (nonpositive
    curvature, no trust region), ``⋯`` the iteration cap."""
    if info.cg_converged:
        return "✓"
    if getattr(info, 'cg_boundary', None):
        return "∂"
    return "⌇" if info.cg_truncated else "⋯"


def _step_segment(info: "bopt.NewtonInfo") -> str:
    """The step-control segment: ``ls k α … ‖Δx‖/‖x‖ …`` (line search; the 2026.2.0 text, byte for byte) or
    ``TR Δ … ✓/✗ ‖Δx‖/‖x‖ …`` (trust region: the radius this trial was bounded by, accepted or rejected)."""
    if getattr(info, 'step_control', None) == 'trust_region' or getattr(info, 'tr_radius', None) is not None:
        verdict = {True: "✓", False: "✗"}.get(getattr(info, 'tr_accepted', None), "—")
        return "TR Δ %s %s ‖Δx‖/‖x‖ %s" % (_num(info.tr_radius, "%.2e"), verdict, _num(info.step_rel, "%.1e"))
    return "ls %d%s α %.2e ‖Δx‖/‖x‖ %.1e" % (
        info.ls_steps, '!' if getattr(info, 'ls_failed', False) else '', info.alpha, info.step_rel)


def _curvature_segment(info: "bopt.NewtonInfo") -> typ.Optional[str]:
    """The exact-Hessian cells (``None`` on a Gauss-Newton row): ``κ = pKp / (pHp − pKp)``, the curvature
    term's share of the model's quadratic relative to the Gauss-Newton part (``—`` on a zero denominator);
    ``σ``, the smallest lift singular value; ``ε``, the lift damping."""
    pKp = getattr(info, 'pKp', None)
    if pKp is None:
        return None
    kappa = None
    if info.pHp is not None and math.isfinite(float(pKp)) and math.isfinite(float(info.pHp)):
        gn_part = float(info.pHp) - float(pKp)
        kappa = float(pKp) / gn_part if gn_part != 0.0 else None
    return "κ %s σ %s ε %s" % (_num(kappa, "%+.1e"), _num(getattr(info, 'lift_sigma_min', None), "%.1e"),
                                _num(getattr(info, 'lift_eps', None), "%.1e"))


def _format_header(info: "bopt.NewtonInfo", obj_unweighted: typ.Optional[float]) -> str:
    """The one-line per-iteration scalar header (a shorter form on the converged final line). A run's last
    row names its ``stop_reason`` when that is not the gradient test (the converged line says so already)."""
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
        ref = _REFERENCE_NAMES.get(getattr(info, 'g0_reference', None))   # None (a hand-built row): unnamed
        parts.append("converged (‖g‖ ≤ gtol·‖g₀‖%s)" % (", g₀ = " + ref if ref else ""))
        return " | ".join(parts)
    parts.append("CG %d/%d tol %.1e resid %.1e %s"
                 % (info.cg_iters, info.cg_maxiter, info.cg_tol, info.cg_resid, _cg_symbol(info)))
    parts.append(_step_segment(info))
    parts.append("Δf %+.2e ρ %.2f" % (info.delta_f, info.rho))
    parts.append("%.2fs" % info.wall_time)
    curvature = _curvature_segment(info)
    if curvature is not None:
        parts.append(curvature)
    stop = getattr(info, 'stop_reason', None)
    if stop is not None and stop != 'gradient':
        parts.append("stop: %s" % stop)
    return " | ".join(parts)


# --------------------------------------------------------------------------------------------------
# Timing: newton_cg(timing=True)'s NewtonInfo.timing, as a line per row and a closing table per run
# --------------------------------------------------------------------------------------------------
_PHASE_LABELS = {'previous_callback': 'cb', 'frame': 'frame', 'sweep': 'sweep', 'companion': 'comp',
                 'residual': 'resid', 'curvature': 'curv', 'gradient': 'grad', 'reference': 'ref', 'cg': 'cg',
                 'forms': 'forms', 'trial_retraction': 'retr', 'trial_objective': 'obj', 'tracking': 'track',
                 'remainder': 'rest'}                      # PhaseSeconds field -> its label on the time line
_MODEL_PHASES = ('frame', 'sweep', 'companion', 'residual', 'curvature')
_COMPILE_SHOWN = 1e-6    # seconds: a compile wait below this is the remainder's rounding, not a compile


def _seconds(value: float) -> str:
    """A time in seconds as a fixed 7-character ``%.1e`` cell, like the error table's."""
    return "%.1e" % value


def _passes(passes: typ.Optional[typ.Tuple[float, ...]]) -> str:
    """`` ×n`` for a tuple of n per-pass times; nothing when the passes were not timed."""
    return "" if passes is None else " ×%d" % len(passes)


def _format_timing(
        info:  "bopt.NewtonInfo",   # a row with timing (newton_cg(timing=True))
) -> str:                           # the row's time line
    """The ``time`` line of a timed row: its window split by phase, in seconds -- the row's own work in run order, then
    the previous row's callback and the remainder.

    ``model`` is the local model's build, its parts in brackets (an overriding ``Problem.local_model`` is timed whole,
    in ``frame``); ``grad`` the misfit transpose to the lift diagnostics; ``ref`` the reference ``‖g0‖`` (row 0; ``×n``
    the ``'cone'`` loop's passes); ``cg`` the CG solve, with its seconds per CG iteration less its compile wait;
    ``forms`` the step's scalars; ``trials`` the retractions and objectives tried (``×n`` trials; on a backtracked
    row, ``accepted`` is the accepted trial's retraction); ``track`` the switch's tracking product; ``cb`` the previous
    row's callback; ``rest`` the window's remainder. Row 0 adds the run's one-offs, outside every window: ``setup``
    and ``align`` (``padded_from``'s alignment, ``×n`` its passes). ``compile`` lists jax's compile wait by phase (on
    a jax run whose ``jax.monitoring`` can unregister a listener; ``0`` when nothing compiled). A phase that did not
    run on this row is left out."""
    timing = info.timing
    seconds, compile_seconds = timing.seconds, timing.compile_seconds
    parts = []
    model = [(name, getattr(seconds, name)) for name in _MODEL_PHASES if getattr(seconds, name) is not None]
    if model:
        parts.append("model %s [%s]" % (_seconds(sum(s for _, s in model)),
                                        " ".join("%s %s" % (_PHASE_LABELS[name], _seconds(s)) for name, s in model)))
    if seconds.gradient is not None:
        parts.append("grad %s" % _seconds(seconds.gradient))
    if seconds.reference is not None:
        parts.append("ref %s%s" % (_seconds(seconds.reference), _passes(timing.reference_passes)))
    if seconds.cg is not None:
        cg = "cg %s" % _seconds(seconds.cg)
        if info.cg_iters:
            compiling = compile_seconds.cg if compile_seconds is not None else 0.0
            cg += " (%s/it)" % _seconds((seconds.cg - compiling) / info.cg_iters)
        parts.append(cg)
    if seconds.forms is not None:
        parts.append("forms %s" % _seconds(seconds.forms))
    if seconds.trial_retraction is not None:
        trials = "trials %s ×%d [retr %s obj %s]" % (
            _seconds(seconds.trial_retraction + seconds.trial_objective), timing.trials,
            _seconds(seconds.trial_retraction), _seconds(seconds.trial_objective))
        if timing.accepted_retraction is not None and timing.trials > 1:
            trials += " (accepted %s)" % _seconds(timing.accepted_retraction)
        parts.append(trials)
    if seconds.tracking is not None:
        parts.append("track %s" % _seconds(seconds.tracking))
    if seconds.previous_callback is not None:
        parts.append("cb %s" % _seconds(seconds.previous_callback))
    parts.append("rest %s" % _seconds(seconds.remainder))
    if timing.setup is not None:
        parts.append("setup %s" % _seconds(timing.setup))
    if timing.alignment is not None:
        parts.append("align %s%s" % (_seconds(timing.alignment), _passes(timing.alignment_passes)))
    if compile_seconds is not None:
        waits = [(_PHASE_LABELS[name], getattr(compile_seconds, name)) for name in _PHASE_LABELS]
        waits += [('setup', timing.setup_compile), ('align', timing.alignment_compile)]
        shown = ["%s %s" % (label, _seconds(c)) for label, c in waits if c is not None and c > _COMPILE_SHOWN]
        parts.append("compile %s" % (" ".join(shown) if shown else "0"))
    return "  time  " + " | ".join(parts)


def _format_timing_table(
        rows:  typ.Sequence[typ.Tuple["bopt.NewtonTiming", typ.Optional[int]]],   # each timed row's (timing, cg_iters)
) -> str:                                                                         # the run's closing table
    """The closing table of a timed run: per phase, its seconds over the rows, its share of the summed windows, its
    compile wait, and its mean over the rows it ran on; then the one-offs (``setup``, ``alignment``) and the pass
    counts and CG's seconds per iteration (less its compile wait). Phases that ran on no row are left out. The last
    row's callback is not in it (it is running): ``stats['timing']['last_callback']`` has it."""
    timings = [timing for timing, _ in rows]
    window = sum(timing.window for timing in timings)
    listened = timings[0].compile_seconds is not None
    labels, cells = [], []
    for name in _PHASE_LABELS:
        ran = [timing for timing in timings if getattr(timing.seconds, name) is not None]
        if not ran:
            continue
        seconds = sum(getattr(timing.seconds, name) for timing in ran)
        compiling = sum(getattr(timing.compile_seconds, name) for timing in ran) if listened else None
        labels.append(name)
        cells.append([_seconds(seconds), "%.1f" % (100.0 * seconds / window) if window > 0 else "—",
                      _num(compiling, "%.1e"), _seconds(seconds / len(ran))])
    first = timings[0]
    for name, seconds, compiling in (('setup', first.setup, first.setup_compile),
                                     ('alignment', first.alignment, first.alignment_compile)):
        if seconds is not None:
            labels.append(name)
            cells.append([_seconds(seconds), "—", _num(compiling, "%.1e"), "—"])
    lines = ["  timing  %d row%s, Σ window %s s (the last row's callback is not in it)"
             % (len(rows), '' if len(rows) == 1 else 's', _seconds(window)),
             _grid(["seconds", "% window", "compile", "mean/row"], labels, cells)]
    notes = []
    for name, passes in (('alignment', first.alignment_passes), ('reference', first.reference_passes)):
        if passes is not None:
            notes.append("%s %d passes" % (name, len(passes)))
    iterations = sum(cg_iters or 0 for _, cg_iters in rows)
    if iterations > 0:
        cg = sum(timing.seconds.cg - (timing.compile_seconds.cg if listened else 0.0)
                 for timing in timings if timing.seconds.cg is not None)
        notes.append("cg %s s per CG iteration" % _seconds(cg / iterations))
    if notes:
        lines.append("    " + ", ".join(notes))
    return "\n".join(lines)


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

    A row from ``newton_cg`` carries its reference kind (``NewtonInfo.g0_reference``), and the converged
    line names it -- here the default zero-tensor reference:

    >>> info = bopt.NewtonInfo(iteration=7, objective=1.2e-9, gnorm=3.1e-7, g0norm=4.4e1, converged=True,
    ...                        g0_reference='zero')
    >>> print(disp.format_newton_iter(info, np.array([[2.0e-5]])))
    iter  7 | obj 1.200e-09  ‖g‖ 3.10e-07 (7.0e-09·g₀) | converged (‖g‖ ≤ gtol·‖g₀‖, g₀ = zero tensor)
      rel err   2.0e-05

    A trust-region row of the exact Hessian (``hessian='exact'``): CG stopped on the region's boundary (``∂``),
    the trial was rejected (``✗``; the next row keeps ``x`` and a quarter of the step length as ``Δ``), and the
    exact cells give ``κ`` (the curvature term's share of ``pᵀHp``, relative to the Gauss-Newton part), the
    smallest lift singular value ``σ`` and the lift damping ``ε``:

    >>> info = bopt.NewtonInfo(iteration=4, objective=2.31e-3, gnorm=7.2e-4, g0norm=5.1e1, converged=False,
    ...                        cg_iters=6, cg_maxiter=200, cg_tol=1.9e-5, cg_resid=3.3e-5, cg_converged=False,
    ...                        cg_truncated=False, cg_boundary=True, alpha=1.0, step_rel=4.0e-2, delta_f=6.1e-5,
    ...                        rho=-0.52, wall_time=0.12, step_control='trust_region', hessian='exact',
    ...                        tr_radius=3.1e-2, tr_accepted=False, pHp=8.0e-4, pKp=-2.0e-4,
    ...                        lift_sigma_min=4.2e-1, lift_eps=1.3e-5)
    >>> print(disp.format_newton_iter(info, np.array([[2.0e-3]])))
    iter  4 | obj 2.310e-03  ‖g‖ 7.20e-04 (1.4e-05·g₀) | CG 6/200 tol 1.9e-05 resid 3.3e-05 ∂ | TR Δ 3.10e-02 ✗ ‖Δx‖/‖x‖ 4.0e-02 | Δf +6.10e-05 ρ -0.52 | 0.12s | κ -2.0e-01 σ 4.2e-01 ε 1.3e-05
      rel err   2.0e-03

    A row of a timed run (``newton_cg(timing=True)``, its ``NewtonInfo.timing``) adds a ``time`` line under the header:
    the row's window by phase, in seconds (:py:class:`~t3toolbox.backend.optimizers.PhaseSeconds`). ``model`` is the
    local model's build with its parts in brackets, ``cg`` also gives its seconds per CG iteration, ``trials`` the
    retractions and objectives tried (``×1``), ``cb`` the previous row's callback and ``rest`` what no phase covers; a
    phase that did not run on the row (here ``ref``, row 0's) is left out:

    >>> seconds = bopt.PhaseSeconds(previous_callback=2.4e-4, frame=4.6e-4, sweep=9.1e-4, companion=3.4e-6,
    ...                             residual=9.8e-4, curvature=1.1e-3, gradient=2.7e-3, reference=None, cg=7.3e-2,
    ...                             forms=1.2e-2, trial_retraction=7.8e-4, trial_objective=1.0e-3, tracking=None,
    ...                             remainder=1.6e-4)
    >>> timing = bopt.NewtonTiming(window=9.3e-2, seconds=seconds, compile_seconds=None, trials=1,
    ...                            accepted_retraction=7.8e-4)
    >>> info = bopt.NewtonInfo(iteration=2, objective=4.62e2, gnorm=5.23e2, g0norm=5.6e4, converged=False,
    ...                        cg_iters=5, cg_maxiter=200, cg_tol=4.8, cg_resid=2.5, cg_converged=True,
    ...                        cg_truncated=False, cg_boundary=False, alpha=1.0, step_rel=1.3e-2, delta_f=-4.62e2,
    ...                        rho=1.01, wall_time=9.3e-2, step_control='trust_region', hessian='exact',
    ...                        tr_radius=6.92e1, tr_accepted=True, pHp=9.2e2, pKp=1.1e1, lift_sigma_min=1.0e1,
    ...                        lift_eps=1.4e-4, timing=timing)
    >>> print(disp.format_newton_iter(info, np.array([[1.2e-2]])))
    iter  2 | obj 4.620e+02  ‖g‖ 5.23e+02 (9.3e-03·g₀) | CG 5/200 tol 4.8e+00 resid 2.5e+00 ✓ | TR Δ 6.92e+01 ✓ ‖Δx‖/‖x‖ 1.3e-02 | Δf -4.62e+02 ρ 1.01 | 0.09s | κ +1.2e-02 σ 1.0e+01 ε 1.4e-04
      time  model 3.5e-03 [frame 4.6e-04 sweep 9.1e-04 comp 3.4e-06 resid 9.8e-04 curv 1.1e-03] | grad 2.7e-03 | cg 7.3e-02 (1.5e-02/it) | forms 1.2e-02 | trials 1.8e-03 ×1 [retr 7.8e-04 obj 1.0e-03] | cb 2.4e-04 | rest 1.6e-04
      rel err   1.2e-02
    """
    header = _format_header(info, obj_unweighted)
    if getattr(info, 'timing', None) is not None:
        header += "\n" + _format_timing(info)
    return header + "\n" + _format_table(train_err, val_err, fmt)


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
    ``val_err``). After a **rejected** trust-region trial (``tr_accepted is False``) the next row starts from
    the same point with the same local model, so it reuses that row's tables -- no second validation
    forward for an unchanged ``x``. (Reuse is keyed on that model object, so a callback shared by several
    runs, e.g. continuation levels, recomputes the next run's first row even after a run that ended on a
    rejected trial.) Requires ``problem.kind.has_block_sumsq`` (all built-in kinds have it).

    **Timed rows** (``newton_cg(timing=True)``) print their ``time`` line (:py:func:`format_newton_iter`), and after a
    run's last row (converged, or a ``stop_reason`` set) the callback prints a closing table: per phase that ran, its
    seconds over the rows, its share of the summed windows, its compile wait (``—`` without one) and its mean over the
    rows it ran on; then the setup and the alignment, the pass counts, and CG's seconds per iteration less its compile
    wait. The last row's own callback is not in it (it is still running; ``stats['timing']['last_callback']`` has it).
    A callback reused across runs starts a new table at each run's row 0. The records are unchanged: ``timing`` is not
    a history field."""
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
    previous = {'lm': None}              # the last row's model if it was a rejected TR trial, and its tables
    timed: list = []                     # this run's (timing, cg_iters) per timed row, for the closing table

    def callback(info):
        if previous['lm'] is not None and info.lm is previous['lm']:   # a rejected trial keeps x and its model:
            train_err, val_err, obj_unweighted = previous['tables']     #   the tables are unchanged
        else:
            res_bs = np.asarray(kind.block_sumsq(info.lm.residual, n_w), dtype=float)
            train_err = relative_errors(res_bs, data_bs)
            val_err = None
            if have_val:
                r_val = _cw.corewise_sub(kind.point_forward(info.x_cores, val_sample), val_data)
                val_err = relative_errors(kind.block_sumsq(r_val, n_w_val), val_data_bs)
            obj_unweighted = 0.5 * float(res_bs.sum())
        previous['lm'] = info.lm if getattr(info, 'tr_accepted', None) is False else None   # the SAME model next row
        previous['tables'] = (train_err, val_err, obj_unweighted)                          # (not a reused callback's next run)
        if print_fn is not None:
            print_fn(format_newton_iter(info, train_err, val_err, obj_unweighted, fmt))
        if record:
            rec = dict(bopt._newton_scalar_record(info))
            rec['train_err'] = train_err
            if val_err is not None:
                rec['val_err'] = val_err
            records.append(rec)
        if info.iteration == 0:                                    # a new run (a callback reused across levels)
            timed.clear()
        if getattr(info, 'timing', None) is not None:
            timed.append((info.timing, info.cg_iters))
            if print_fn is not None and (info.converged or getattr(info, 'stop_reason', None) is not None):
                print_fn(_format_timing_table(timed))                  # the run's last row

    return callback, records
