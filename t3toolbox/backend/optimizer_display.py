# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
"""Diagnostic display for the Newton-CG fitting loop -- **backend-owned**, so a raw-``.data`` user gets the
identical display without touching the frontend (the anti-drift rule):

    import t3toolbox.backend.optimizers as bopt
    import t3toolbox.backend.optimizer_display as bdisp
    cb, records = bdisp.make_newton_display(oracle, batch, val_batch=val_batch)
    x, stats = bopt.newton_cg(geom, oracle, batch, x0, callback=cb)     # prints each row; records == the history

Two layers, so the pure algorithm modules stay pure and the I/O is isolated:

* :py:func:`format_newton_iter` -- a **pure** function returning the formatted string block (a scalar
  header line, a ``time`` line on a timed row, and the relative-error table). Testable without capturing stdout.
* :py:func:`make_newton_display` -- builds a ``callback(NewtonInfo)`` for
  :py:func:`t3toolbox.backend.optimizers.newton_cg`: it reads the data's block norms once (the denominators), then
  per row the residual's block norms through the oracle (train, and validation when a ``val_batch`` is given),
  prints via an injectable ``print_fn``, and records each row.

A run with ``newton_cg(timing=True)`` adds a ``time`` line to each row's block (the row's window by phase) and a
closing table after the run's last row; untimed rows print exactly as before. In a ``newton_cg(hessian='switch')``
run, each Gauss-Newton-phase step row's header carries an ``sw`` segment, before a last row's stop reason (the
tracking ratio ``e_N / e_GN`` and the hit count, the switching row marked ``⇒ exact``); its exact-phase rows, its
converged line and every row of the other modes print as before.

The relative-error table is ``‖ω⊙r_b‖ / ‖ω⊙y_b‖`` per block ``b``, from the oracle's block sums ``<r, M r>`` and
``<y, M y>`` (``r_block_sumsq`` at the model's frame with its ``held``, ``y_block_sumsq``), ``M`` the data metric: the
per-block recovery error under the fit's own weight. A draw's scale is in both sums and cancels; a weight constant
within each block cancels too (the plain ``‖r_b‖ / ‖y_b‖``); a sample weighted zero counts in neither. The errors have
the block shape: ``(d,) + F`` for probe (modes, then the jet axes ``F``), ``F`` for apply, ``()`` for entries; the
records keep them in that shape. The printed layout follows the jet shape ``order`` (D4: given, never read off the
model): with at most one jet variable it is the 2026.5.0 table -- modes in rows and the orders in columns
(``train|val`` cells) for probe along a line, a single axis in columns with ``train``/``val`` rows otherwise, a one-line
scalar for a plain apply or entries fit; with two or more, one column per multi-index of the jet axes in row-major
order (O-42). ``error_table=False`` prints the header lines only and changes no record. An oracle without both block
sums gets the header lines and no table.

Formatting uses ``%.1e`` (Python pads the exponent to a signed 2-digit field, so every cell is exactly 7
chars -> columns align with no extra work). The callback is **host-side** (it reads concrete block sums),
so it composes with ``newton_cg(use_jit=True)`` (only the inner CG jits) but not a fully-jitted outer loop.
"""
import math
import typing as typ

import numpy as np

from t3toolbox.backend.common import *   # NDArray
import t3toolbox.backend.optimizers as bopt

__all__ = ['format_newton_iter', 'make_newton_display', 'relative_errors']


# --------------------------------------------------------------------------------------------------
# Relative-error matrix
# --------------------------------------------------------------------------------------------------
def relative_errors(
        residual_block_sumsq: NDArray,   # the oracle's r_block_sumsq, <r, M r> per block, the block shape (d,)+F|F|()
        data_block_sumsq:     NDArray,   # the oracle's y_block_sumsq, <y, M y> per block, the same shape
) -> NDArray:                            # sqrt(num/den) per block, the same shape; NaN where the data block norm is 0
    """Per-block relative error ``‖ω⊙r_b‖ / ‖ω⊙y_b‖ = sqrt(r_block_sumsq / y_block_sumsq)``, elementwise (a draw's
    scale, in both sums, cancels). A zero data block (no signal in that block, or a weight of zero throughout it) is
    undefined -> ``NaN`` (rendered ``—``)."""
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


def _format_matrix(train_err: NDArray, val_err: typ.Optional[NDArray], fmt: str) -> str:
    """The 2026.5.0 table of a ``(n_mode, n_order)`` matrix, layout chosen from its shape (Decision 2a)."""
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


def _format_columns(
        train_err:   NDArray,                  # lead + column axes
        val_err:     typ.Optional[NDArray],
        fmt:         str,
        lead:        int,                      # 1: the first axis is the rows; 0: rows train/val
        col_labels:  typ.Sequence[str],        # one per column, the column axes unfolded row-major
        row_label:   str,                      # the row label's format (lead 1)
        rows_name:   str,                      # the legend's names
        cols_name:   str,
) -> str:
    """A table whose columns unfold the trailing axes in row-major order: rows by the leading axis with ``train|val``
    cells (``lead = 1``), or ``train`` / ``val`` rows (``lead = 0``)."""
    w = len(fmt % 1.0)
    t = train_err.reshape(train_err.shape[:lead] + (-1,))
    v = None if val_err is None else val_err.reshape(val_err.shape[:lead] + (-1,))
    n = t.shape[-1]
    if lead:
        row_labels = [row_label % i for i in range(t.shape[0])]
        cells = [[(_cell(t[i, c], fmt, w) + "|" + _cell(v[i, c], fmt, w)) if v is not None
                  else _cell(t[i, c], fmt, w) for c in range(n)] for i in range(t.shape[0])]
        legend = "  rel err  rows=%s  cols=%s" % (rows_name, cols_name) + ("  (cells train|val)" if v is not None
                                                                           else "")
    else:
        row_labels = ["train"] + (["val"] if v is not None else [])
        cells = [[_cell(e[c], fmt, w) for c in range(n)] for e in ([t] + ([v] if v is not None else []))]
        legend = "  rel err  rows=train/val  cols=%s" % cols_name
    return legend + "\n" + _grid(col_labels, row_labels, cells)


def _format_table(
        train_err:  NDArray,                          # relative errors in the block shape
        val_err:    typ.Optional[NDArray],            # the same shape, or None
        fmt:        str,
        order:      typ.Optional[typ.Tuple[int, ...]] = None,   # the jet shape (D4); None: labelled by position
) -> str:
    """The relative-error table of arrays in the block shape (``(d,) + F`` probe, ``F`` apply, ``()``
    entries). With the jet shape ``order`` (``m = len(order)``), the leading ``ndim - m`` axes (none or one) are the
    modes: at ``m <= 1`` the 2026.5.0 table (:py:func:`_format_matrix` of the ``(n_mode, n_order)`` matrix); at
    ``m >= 2`` one column per multi-index of the jet axes, row-major (O-42), mode rows with ``train|val`` cells for
    probe, ``train`` / ``val`` rows for apply. Without ``order`` (an oracle that names none), the axes are labelled by
    position: a scalar on one line, one axis in columns, else the first axis in rows and the others unfolded row-major
    in columns. A malformed shape raises a ``ValueError`` (structural)."""
    train_err = np.asarray(train_err, dtype=float)
    val_err = None if val_err is None else np.asarray(val_err, dtype=float)
    if val_err is not None and val_err.shape != train_err.shape:
        raise ValueError("the validation errors have shape %s, the training errors %s: they must agree (the block "
                         "shape)" % (val_err.shape, train_err.shape))
    if order is None:
        if train_err.ndim == 0:
            return _format_matrix(train_err.reshape(1, 1), None if val_err is None else val_err.reshape(1, 1), fmt)
        lead = 1 if train_err.ndim >= 2 else 0
        cols = ["[%s]" % ",".join(str(i) for i in ix) for ix in np.ndindex(*train_err.shape[lead:])]
        return _format_columns(train_err, val_err, fmt, lead, cols, "[%d]", "axis 0", "position (row-major)")
    m = len(order)
    lead = train_err.ndim - m
    if lead not in (0, 1) or tuple(train_err.shape[lead:]) != tuple(n + 1 for n in order):
        raise ValueError("relative errors of shape %s do not fit the jet shape order=%r: expected the block "
                         "shape, (d,) + F or F with F = %r" % (train_err.shape, tuple(order),
                                                               tuple(n + 1 for n in order)))
    if m <= 1:
        shape2 = (train_err.shape[0] if lead else 1, train_err.shape[-1] if m else 1)
        return _format_matrix(train_err.reshape(shape2), None if val_err is None else val_err.reshape(shape2), fmt)
    cols = ["ord(%s)" % ",".join(str(i) for i in ix) for ix in np.ndindex(*train_err.shape[lead:])]
    return _format_columns(train_err, val_err, fmt, lead, cols, "m%d", "mode", "order multi-index (row-major)")


_REFERENCE_NAMES = {'zero': 'zero tensor', 'warm_start': 'warm start', 'cone': 'cone (aligned zero tensor)',
                    'override': 'override'}   # NewtonInfo.g0_reference


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


def _switch_segment(info: "bopt.NewtonInfo") -> typ.Optional[str]:
    """The switch cells of a Gauss-Newton-phase row of ``newton_cg(hessian='switch')`` (``None`` on a row without
    ``switch_count``, which only those rows carry; the converged one carries it too, but its short line never reaches
    the segments): ``sw``, the tracking ratio ``e_N / e_GN`` (``—`` on an untracked row, on a non-finite error or at
    ``e_GN = 0``), then ``×`` the count of consecutive hits after the row (``×0``: a reset). The row whose count
    reached ``switch_k`` (``switch_fired``) ends with ``⇒ exact``: the next row's model is exact."""
    count = getattr(info, 'switch_count', None)
    if count is None:
        return None
    e_gn, e_n = getattr(info, 'switch_e_gn', None), getattr(info, 'switch_e_n', None)
    ratio = None
    if e_gn is not None and e_n is not None and math.isfinite(float(e_gn)) and math.isfinite(float(e_n)):
        ratio = float(e_n) / float(e_gn) if float(e_gn) != 0.0 else None
    return "sw %s ×%d%s" % (_num(ratio, "%.1e"), count, " ⇒ exact" if getattr(info, 'switch_fired', None) else "")


def _format_header(info: "bopt.NewtonInfo") -> str:
    """The one-line per-iteration scalar header (a shorter form on the converged final line). A run's last
    row names its ``stop_reason`` when that is not the gradient test (the converged line says so already)."""
    obj = "obj %.3e" % info.objective
    if info.regularization is not None and info.misfit is not None:   # regularizer attached -> obj = misfit + reg
        obj += " = misfit %.3e + reg %.3e" % (info.misfit, info.regularization)
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
    switch = _switch_segment(info)
    if switch is not None:
        parts.append(switch)
    stop = getattr(info, 'stop_reason', None)
    if stop is not None and stop != 'gradient':
        parts.append("stop: %s" % stop)
    return " | ".join(parts)


# --------------------------------------------------------------------------------------------------
# Timing: newton_cg(timing=True)'s NewtonInfo.timing, as a line per row and a closing table per run
# --------------------------------------------------------------------------------------------------
_PHASE_LABELS = {'previous_callback': 'cb', 'frame': 'frame', 'oracle': 'oracle', 'companion': 'comp',
                 'curvature': 'curv', 'gradient': 'grad', 'reference': 'ref', 'cg': 'cg',
                 'forms': 'forms', 'trial_retraction': 'retr', 'trial_objective': 'obj', 'tracking': 'track',
                 'remainder': 'rest'}                      # PhaseSeconds field -> its label on the time line
_MODEL_PHASES = ('frame', 'oracle', 'companion', 'curvature')
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
        train_err:      typ.Optional[NDArray],    # training relative errors in the block shape; None: no table
        val_err:        typ.Optional[NDArray] = None,   # validation relative errors, the same shape, or None
        fmt:            str = '%.1e',
        order:          typ.Optional[typ.Tuple[int, ...]] = None,   # the jet shape (D4); None: axes labelled by position
        error_table:    bool = True,              # False: the header lines only
) -> str:                                         # the formatted header line(s) + relative-error table
    """Format one Newton iteration as a header line (a ``time`` line under it on a timed row) and the relative-error
    table (pure -- returns a string). The errors have the block shape (``(d,) + F`` probe, ``F`` apply, ``()``
    entries); ``order``, the jet shape, says which axes are the jets (D4: given explicitly, never read off the
    model) and so the layout: the 2026.5.0 table for at most one jet variable, one column per multi-index in row-major
    order for two or more (O-42); without it the axes are labelled by position. ``error_table=False`` (or
    ``train_err=None``) gives the header lines alone. When a regularizer is attached the header splits the objective
    as ``obj = misfit + reg`` (``info.misfit`` / ``info.regularization``).

    Examples
    --------
    A converged final iteration (short header) of a probe fit along lines (``order=(1,)``: 2 modes x 2 orders); note
    ``1.0e+00`` and ``5.2e-04`` line up -- ``%.1e`` is fixed 7-char width, so columns align with no padding logic:

    >>> import numpy as np
    >>> import t3toolbox.backend.optimizers as bopt
    >>> import t3toolbox.backend.optimizer_display as disp
    >>> info = bopt.NewtonInfo(iteration=5, objective=3.78e-4, gnorm=4.8e-6, g0norm=6.5e-2, converged=True)
    >>> train = np.array([[1.9e-2, 1.3e-2], [1.0e+00, 5.2e-4]])   # (d,) + F = (2 modes, 2 orders)
    >>> print(disp.format_newton_iter(info, train, order=(1,)))
    iter  5 | obj 3.780e-04  ‖g‖ 4.80e-06 (7.4e-05·g₀) | converged (‖g‖ ≤ gtol·‖g₀‖)
      rel err  rows=mode  cols=order
               ord0     ord1
        m0  1.9e-02  1.3e-02
        m1  1.0e+00  5.2e-04

    Two jet variables (``order=(1, 1)``: the jet axes ``F = (2, 2)``) give one column per multi-index, in
    row-major order (O-42); here an apply fit with validation, so ``train`` / ``val`` rows:

    >>> train = np.array([[2.1e-3, 4.0e-2], [3.3e-2, 1.0e+00]])
    >>> print(disp.format_newton_iter(info, train, 1.5 * train, order=(1, 1)))
    iter  5 | obj 3.780e-04  ‖g‖ 4.80e-06 (7.4e-05·g₀) | converged (‖g‖ ≤ gtol·‖g₀‖)
      rel err  rows=train/val  cols=order multi-index (row-major)
               ord(0,0)  ord(0,1)  ord(1,0)  ord(1,1)
        train   2.1e-03   4.0e-02   3.3e-02   1.0e+00
        val     3.2e-03   6.0e-02   5.0e-02   1.5e+00

    ``error_table=False`` keeps the header and drops the table:

    >>> print(disp.format_newton_iter(info, train, order=(1, 1), error_table=False))
    iter  5 | obj 3.780e-04  ‖g‖ 4.80e-06 (7.4e-05·g₀) | converged (‖g‖ ≤ gtol·‖g₀‖)

    With a regularizer attached, the objective splits as ``obj = misfit + reg`` (here a plain apply fit,
    ``order=()``: a scalar error, one line):

    >>> info = bopt.NewtonInfo(iteration=3, objective=6.003e1, gnorm=1.60, g0norm=1.07e1, converged=False,
    ...                        misfit=5.974e1, regularization=2.9e-1, cg_iters=8, cg_maxiter=200,
    ...                        cg_tol=6.2e-1, cg_resid=4.9e-1, cg_converged=True, cg_truncated=False,
    ...                        ls_steps=0, alpha=1.0, step_rel=1.2, delta_f=-2.22, rho=0.07, wall_time=0.03)
    >>> print(disp.format_newton_iter(info, np.array(3.7e-1), order=()))
    iter  3 | obj 6.003e+01 = misfit 5.974e+01 + reg 2.900e-01  ‖g‖ 1.60e+00 (1.5e-01·g₀) | CG 8/200 tol 6.2e-01 resid 4.9e-01 ✓ | ls 0 α 1.00e+00 ‖Δx‖/‖x‖ 1.2e+00 | Δf -2.22e+00 ρ 0.07 | 0.03s
      rel err   3.7e-01

    A row from ``newton_cg`` carries its reference kind (``NewtonInfo.g0_reference``), and the converged
    line names it -- here the default zero-tensor reference:

    >>> info = bopt.NewtonInfo(iteration=7, objective=1.2e-9, gnorm=3.1e-7, g0norm=4.4e1, converged=True,
    ...                        g0_reference='zero')
    >>> print(disp.format_newton_iter(info, np.array(2.0e-5), order=()))
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
    >>> print(disp.format_newton_iter(info, np.array(2.0e-3), order=()))
    iter  4 | obj 2.310e-03  ‖g‖ 7.20e-04 (1.4e-05·g₀) | CG 6/200 tol 1.9e-05 resid 3.3e-05 ∂ | TR Δ 3.10e-02 ✗ ‖Δx‖/‖x‖ 4.0e-02 | Δf +6.10e-05 ρ -0.52 | 0.12s | κ -2.0e-01 σ 4.2e-01 ε 1.3e-05
      rel err   2.0e-03

    A Gauss-Newton-phase row of ``hessian='switch'``: the ``sw`` segment gives the tracking ratio ``e_N / e_GN``, how
    far the exact model's predicted change missed the actual one relative to Gauss-Newton's miss (``—`` on a row that
    ran no tracking product), then ``×`` the count of consecutive hits (``e_N < switch_tau·e_GN`` on a countable step)
    after this row. Here the count reaches ``switch_k = 2``, so the row ends with ``⇒ exact``: from the next row on
    the model is the exact Hessian, and the rows show its ``κ σ ε`` cells instead:

    >>> info = bopt.NewtonInfo(iteration=1, objective=2.537e5, gnorm=4.17e2, g0norm=9.84e2, converged=False,
    ...                        cg_iters=1, cg_maxiter=200, cg_tol=2.1e2, cg_resid=1.2e2, cg_converged=True,
    ...                        cg_truncated=False, cg_boundary=False, alpha=1.0, step_rel=8.7e-3, delta_f=-2.03e2,
    ...                        rho=0.85, wall_time=0.03, step_control='trust_region', hessian='gauss_newton',
    ...                        tr_radius=2.51, tr_accepted=True, pHp=4.1e2, switch_pKp=7.43e1, switch_e_gn=3.70e1,
    ...                        switch_e_n=1.1e-1, switch_count=2, switch_fired=True)
    >>> print(disp.format_newton_iter(info, np.array(2.8e-1), order=()))
    iter  1 | obj 2.537e+05  ‖g‖ 4.17e+02 (4.2e-01·g₀) | CG 1/200 tol 2.1e+02 resid 1.2e+02 ✓ | TR Δ 2.51e+00 ✓ ‖Δx‖/‖x‖ 8.7e-03 | Δf -2.03e+02 ρ 0.85 | 0.03s | sw 3.0e-03 ×2 ⇒ exact
      rel err   2.8e-01

    A row of a timed run (``newton_cg(timing=True)``, its ``NewtonInfo.timing``) adds a ``time`` line under the header:
    the row's window by phase, in seconds (:py:class:`~t3toolbox.backend.optimizers.PhaseSeconds`). ``model`` is the
    local model's build with its parts in brackets, ``cg`` also gives its seconds per CG iteration, ``trials`` the
    retractions and objectives tried (``×1``), ``cb`` the previous row's callback and ``rest`` what no phase covers; a
    phase that did not run on the row (here ``ref``, row 0's) is left out:

    >>> seconds = bopt.PhaseSeconds(previous_callback=2.4e-4, frame=4.6e-4, oracle=1.9e-3, companion=3.4e-6,
    ...                             curvature=1.1e-3, gradient=2.7e-3, reference=None, cg=7.3e-2,
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
    >>> print(disp.format_newton_iter(info, np.array(1.2e-2), order=()))
    iter  2 | obj 4.620e+02  ‖g‖ 5.23e+02 (9.3e-03·g₀) | CG 5/200 tol 4.8e+00 resid 2.5e+00 ✓ | TR Δ 6.92e+01 ✓ ‖Δx‖/‖x‖ 1.3e-02 | Δf -4.62e+02 ρ 1.01 | 0.09s | κ +1.2e-02 σ 1.0e+01 ε 1.4e-04
      time  model 3.5e-03 [frame 4.6e-04 oracle 1.9e-03 comp 3.4e-06 curv 1.1e-03] | grad 2.7e-03 | cg 7.3e-02 (1.5e-02/it) | forms 1.2e-02 | trials 1.8e-03 ×1 [retr 7.8e-04 obj 1.0e-03] | cb 2.4e-04 | rest 1.6e-04
      rel err   1.2e-02
    """
    header = _format_header(info)
    if getattr(info, 'timing', None) is not None:
        header += "\n" + _format_timing(info)
    if not error_table or train_err is None:
        return header
    return header + "\n" + _format_table(train_err, val_err, fmt, order)


# --------------------------------------------------------------------------------------------------
# The callback builder
# --------------------------------------------------------------------------------------------------
def _host(value) -> NDArray:
    """A block-sum value on the host, float64."""
    return np.asarray(value, dtype=float)


def make_newton_display(
        oracle,                                   # the fit's Oracle (its block sums: r_block_sumsq, y_block_sumsq)
        batch,                                    # its batch (the training data: the denominators)
        val_batch:   typ.Any = None,              # optional validation batch, laid out like batch -> a train|val table
        print_fn:    typ.Optional[typ.Callable] = print,   # where to send each row's text (None = silent)
        fmt:         str  = '%.1e',
        record:      bool = True,                 # accumulate per-row records (scalars + error arrays)
        error_table: bool = True,                 # False: print the header lines only (the records are unchanged)
) -> typ.Tuple[typ.Callable, list]:               # (callback for newton_cg, records list filled as it runs)
    """Build a ``callback(NewtonInfo)`` (+ its ``records`` list) that displays each Newton iteration.

    **The errors are on the batch the display is given**: numerator and denominator both on ``batch`` (and on
    ``val_batch``). The denominators, the data's block sums ``y_block_sumsq``, are queried once, at the first row's
    frame (a packed probe ``y`` reads its mode sizes off the frame). Each row's numerator is the residual's block sums
    ``r_block_sumsq`` at the row's model's frame (``info.lm``, with its centers' ``Vc``):

    - when ``batch`` is the run's batch (``info.lm.batch is batch``), with the model's ``held``: the residual the
      build computed, no sweep;
    - otherwise (a run on a drawn batch, on other data, or under ``use_jit``, which converts the batch to a new jax
      object) a forward on ``batch``, never with the model's ``held`` (that belongs to the run's batch): one forward
      more per row. Hand the display the very batch object the run uses to take the first path.

    The validation numerator is likewise a forward at the same frame on ``val_batch``, never with the model's ``held``.
    The block sums are ``<r, M r>`` and ``<y, M y>`` per block, so each error is ``‖ω⊙r_b‖ / ‖ω⊙y_b‖``: the per-block
    recovery error under the weight, a draw's scale cancelling (a weight constant within each block cancels too: the
    plain ``‖r_b‖ / ‖y_b‖``). **Each table is under its own batch's weight**: the validation cells use ``val_batch``'s
    weight, so give ``val_batch`` a weight of the same form as ``batch``'s, or the two halves of a ``train|val`` cell
    are different quantities (a weighted ratio beside an unweighted one, say). An oracle that lacks either block sum
    (not in its ``optional_outputs``) gets the header lines and no table, and its records no error arrays.

    **The records.** Each row appends a self-contained record: the history's scalar fields, ``train_err`` and (with a
    ``val_batch``) ``val_err`` in the block shape (``(d,) + F`` probe, ``F`` apply, ``()`` entries), nothing
    reduced. ``error_table=False`` changes the printed text only. The layout of the printed table follows the oracle's
    ``order`` (:py:func:`format_newton_iter`; an oracle without one: axes labelled by position).

    After a **rejected** trust-region trial (``tr_accepted is False``) the next row starts from the same point with
    the same local model, so it reuses that row's errors -- no queries for an unchanged ``x``. (Reuse is keyed on that
    model object, so a callback shared by several runs, e.g. continuation levels, recomputes the next run's first row
    even after a run that ended on a rejected trial.)

    **Timed rows** (``newton_cg(timing=True)``) print their ``time`` line (:py:func:`format_newton_iter`), and after a
    run's last row (converged, or a ``stop_reason`` set) the callback prints a closing table: per phase that ran, its
    seconds over the rows, its share of the summed windows, its compile wait (``—`` without one) and its mean over the
    rows it ran on; then the setup and the alignment, the pass counts, and CG's seconds per iteration less its compile
    wait. The last row's own callback is not in it (it is still running; ``stats['timing']['last_callback']`` has it).
    A callback reused across runs starts a new table at each run's row 0. The records are unchanged: ``timing`` is not
    a history field.

    **Switch rows** (``newton_cg(hessian='switch')``): each Gauss-Newton-phase step row's header carries the ``sw``
    segment, before a last row's stop reason (:py:func:`format_newton_iter`), and every record of such a run carries
    the five ``switch_*`` history fields.

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.backend.optimizers as bopt
    >>> import t3toolbox.backend.optimizer_display as bdisp
    >>> from t3toolbox.backend.geometry import ManifoldGeometryOps
    >>> from t3toolbox.backend.fitting import LeastSquaresOracle, LeastSquaresBatch
    >>> np.random.seed(0)
    >>> target = t3.TuckerTensorTrain.randn((5, 6, 7), (2, 2, 2), (1, 2, 2, 1))
    >>> ww, wv = ([np.random.randn(W, N) for N in target.shape] for W in (60, 20))
    >>> oracle = LeastSquaresOracle('probe', order=())
    >>> batch, val_batch = LeastSquaresBatch(ww, target.probe(ww)), LeastSquaresBatch(wv, target.probe(wv))
    >>> cb, records = bdisp.make_newton_display(oracle, batch, val_batch, print_fn=None)
    >>> x0 = t3.TuckerTensorTrain.randn((5, 6, 7), (2, 2, 2), (1, 2, 2, 1)).data
    >>> x, stats = bopt.newton_cg(ManifoldGeometryOps(), oracle, batch, x0, max_newton=4, callback=cb)
    >>> print(len(records) == len(stats['history']), records[0]['train_err'].shape, records[0]['val_err'].shape)
    True (3,) (3,)
    >>> print(bool(records[-1]['val_err'].max() < records[0]['val_err'].min()))
    True
    """
    outputs = tuple(getattr(oracle, 'optional_outputs', ()))
    tables = 'r_block_sumsq' in outputs and 'y_block_sumsq' in outputs
    order = getattr(oracle, 'order', None)
    order = None if order is None else tuple(order)
    denominators: dict = {}              # 'train' (and 'val'): y_block_sumsq, queried once at the first row's frame
    records: list = []
    previous = {'lm': None}              # the last row's model if it was a rejected TR trial, and its errors
    timed: list = []                     # this run's (timing, cg_iters) per timed row, for the closing table

    def errors(lm):
        if not denominators:
            (y_bs,), _ = oracle.query(lm.frame, batch, ('y_block_sumsq',))
            denominators['train'] = _host(y_bs)
            if val_batch is not None:
                (y_bs,), _ = oracle.query(lm.frame, val_batch, ('y_block_sumsq',))
                denominators['val'] = _host(y_bs)
        Vc = None if lm.centers is None else lm.centers[0]
        own = lm.batch is batch                                         # the run's batch: its held r, no sweep
        if own:
            (r_bs,), _ = lm.oracle.query(lm.frame, batch, ('r_block_sumsq',), Vc=Vc, held=lm.held)
        else:                                                           # another batch: a forward on it, never held
            (r_bs,), _ = lm.oracle.query(lm.frame, batch, ('r_block_sumsq',), Vc=Vc)
        r_bs = _host(r_bs)
        train_err = relative_errors(r_bs, denominators['train'])
        val_err = None
        if val_batch is not None:
            (r_val,), _ = lm.oracle.query(lm.frame, val_batch, ('r_block_sumsq',), Vc=Vc)   # never the model's held
            val_err = relative_errors(_host(r_val), denominators['val'])
        return train_err, val_err

    def callback(info):
        if previous['lm'] is not None and info.lm is previous['lm']:   # a rejected trial keeps x and its model:
            train_err, val_err = previous['tables']                     #   the errors are unchanged
        elif tables:
            train_err, val_err = errors(info.lm)
        else:
            train_err = val_err = None
        previous['lm'] = info.lm if getattr(info, 'tr_accepted', None) is False else None   # the SAME model next row
        previous['tables'] = (train_err, val_err)                                 # (not a reused callback's next run)
        if print_fn is not None:
            print_fn(format_newton_iter(info, train_err, val_err, fmt, order=order,
                                        error_table=error_table))
        if record:
            rec = dict(bopt._newton_scalar_record(info))
            if train_err is not None:
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
