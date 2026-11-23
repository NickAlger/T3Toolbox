"""M5, the hand edits: docstrings and comments outside the signature tables of ``types.py``, as exact replacements.

Run from the repository root after ``types.py``: ``python dev/graded_messages/implementation/m5/prose.py``. Each edit
is ``(file, scope, old, new)``: ``old`` must occur exactly once in ``scope`` (a function or class by qualname, or
``None`` for the whole file). The engine's table gains its type column by rule (``TYPE``), checked name by name.
"""
import ast
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[4]

HESSIAN_BACKEND = ('applied here: for the gradient of a weighted least-squares fit pass ``data_metric(r, weight, '
                   'minibatch_scale)`` (or\n    ``batch.data_metric(r)``), which is ``r`` itself with no weight and no '
                   'scale. One call of\n')
HESSIAN_BACKEND_NEW = ('applied here: for the curvature term of a weighted least-squares fit (``Kp`` at ``c = M r``), '
                       'pass\n    ``data_metric(r, weight, minibatch_scale)`` (or ``batch.data_metric(r)``), which is '
                       '``r`` itself with no weight and\n    no scale. One call of\n')

EDITS = [
    # --- M4 review, minor 1: the corewise Hessians' sentence (four docstrings carry it)
    ('t3toolbox/backend/probing.py', 't3_probe_corewise_hessian',
     'and none is applied here: for the gradient of a weighted least-squares fit pass\n',
     'and none is applied here: for the curvature term of a weighted least-squares fit (``Kp`` at ``c = M r``), pass\n'),
    ('t3toolbox/backend/apply.py', 't3_apply_corewise_hessian', HESSIAN_BACKEND, HESSIAN_BACKEND_NEW),
    ('t3toolbox/backend/entries.py', 't3_entries_corewise_hessian', HESSIAN_BACKEND, HESSIAN_BACKEND_NEW),
    ('t3toolbox/tucker_tensor_train.py', 'TuckerTensorTrain.probe_corewise_hessian',
     "        (for a weighted fit's, pass ``data_metric(r, weight, minibatch_scale)``). ``core_directions`` as in\n"
     "        :py:meth:`probe_corewise_tangent`.",
     "        (for the curvature term of a weighted fit, ``Kp`` at ``c = M r``, pass\n"
     "        ``data_metric(r, weight, minibatch_scale)``). ``core_directions`` as in :py:meth:`probe_corewise_tangent`."),
    # --- D5: the tangent transposes return X* in a tangent container
    ('t3toolbox/manifold.py', 'T3Tangent.probe_transpose',
     "        Bare ``(J^(s))^T`` (no gauge projector). See Section 6.2.3 (Algorithm 8) of Alger et al. (2026).\n",
     "        Bare ``(J^(s))^T`` (no gauge projector): the result is an element of ``X*``, carried in a ``T3Tangent`` at\n"
     "        ``frame`` before the projection (the geometry's ``project`` maps it into ``X``). See Section 6.2.3\n"
     "        (Algorithm 8) of Alger et al. (2026).\n"),
    ('t3toolbox/manifold.py', 'T3Tangent.apply_transpose',
     "        stack (one tangent per sample). Needs only the frame sweep + a single-term scatter assembly (cheaper than a\n"
     "        general :py:meth:`probe_transpose`).\n",
     "        stack (one tangent per sample). Needs only the frame sweep + a single-term scatter assembly (cheaper than a\n"
     "        general :py:meth:`probe_transpose`). The result is ``X*`` in a ``T3Tangent``, before the projection, as\n"
     "        there.\n"),
    ('t3toolbox/manifold.py', 'T3Tangent.entries_transpose',
     "        as there. ``sum_over_samples`` as in :py:meth:`apply_transpose` (see *Batching & stacking* §11).\n",
     "        as there. ``sum_over_samples`` as in :py:meth:`apply_transpose` (see *Batching & stacking* §11). The result\n"
     "        is ``X*`` in a ``T3Tangent``, before the projection, as there.\n"),
    ('t3toolbox/uniform_manifold.py', 'UT3Tangent.probe_transpose',
     "``False`` keeps it as the result's tangent stack (``W+K``). The bare ``𝒥ᵀ`` (no gauge projector\n"
     "        ``Π``).",
     "``False`` keeps it as the result's tangent stack (``W+K``). The bare ``𝒥ᵀ`` (no gauge projector\n"
     "        ``Π``): ``X*`` in a ``UT3Tangent``, before the projection."),
    ('t3toolbox/uniform_manifold.py', 'UT3Tangent.apply_transpose',
     "The bare\n        ``𝒥ᵀ`` (no gauge projector ``Π``).\"\"\"",
     "The bare\n        ``𝒥ᵀ`` (no gauge projector ``Π``): ``X*`` in a ``UT3Tangent``, before the projection.\"\"\""),
    ('t3toolbox/uniform_manifold.py', 'UT3Tangent.entries_transpose',
     "Uniform mirror of :py:meth:`~t3toolbox.manifold.T3Tangent.entries_transpose`. The bare ``𝒥ᵀ``.\"\"\"",
     "Uniform mirror of :py:meth:`~t3toolbox.manifold.T3Tangent.entries_transpose`. The bare ``𝒥ᵀ``: ``X*`` in a\n"
     "        ``UT3Tangent``, before the projection.\"\"\""),
    ('t3toolbox/uniform_manifold.py', 'UT3Tangent.probe_transpose',
     "# X*, in a T3Tangent at frame", "# X*, in a UT3Tangent at frame"),
    ('t3toolbox/uniform_manifold.py', 'UT3Tangent.apply_transpose',
     "# X*, in a T3Tangent at frame", "# X*, in a UT3Tangent at frame"),
    ('t3toolbox/uniform_manifold.py', 'UT3Tangent.entries_transpose',
     "# X*, in a T3Tangent at frame", "# X*, in a UT3Tangent at frame"),
    # --- the engine: the prose around the table (the column is added by rule, below)
    ('t3toolbox/backend/sampling_variables.py', None,
     "core's shape); a private node's shape is its base's. ``f`` has the shape ``C``; ``r_block_sumsq``, ``<r, M r>`` per\n"
     "block (weighted as ``f`` is), the block shape ``(d,) + F`` (``F`` for apply); ``Jp_norm_sq`` the shape ``K + C``;\n"
     "``g`` is the pair ``(V_tilde, H_tilde)``, ``JtJp`` the pair of their twins and ``Kp`` the pair ``(V_tilde_dot,\n"
     "H_tilde_dot)``.",
     "core's shape); a private node's shape is its base's. ``f`` (``ℝ``) has the shape ``C``; ``r_block_sumsq`` (``ℝ`` per\n"
     "block), ``<r, M r>`` per block (weighted as ``f`` is), the block shape ``(d,) + F`` (``F`` for apply);\n"
     "``Jp_norm_sq`` (``ℝ``) the shape ``K + C``; ``g`` is the pair ``(V_tilde, H_tilde)``, ``JtJp`` the pair of their\n"
     "twins and ``Kp`` the pair ``(V_tilde_dot, H_tilde_dot)``, each in ``X*``.\n"
     "\n"
     "**The types** (the last column; the letters are the glossary's Symbols). ``P``: a jet on an edge of the network,\n"
     "slot ``k`` the coefficient of ``s^k`` (the plain variables and the dots, linear in the direction). ``Y*``: the\n"
     "measurements, probe ``P⊗V_i*`` per mode, apply ``P``, entries ``ℝ``. ``Y turned``: ``w_tilde``, an element of\n"
     "``Y`` (probe ``P*⊗V_i``, apply ``P*``, entries ``ℝ``) with every jet axis turned, slot ``k`` holding the\n"
     "coefficient ``n - k`` of the dual jet. ``P* turned``: every other tilde variable, a dual jet on an edge, turned\n"
     "likewise. ``X*``: a block of the parameters' dual, before any projection. So under ``s_j -> ell_j s_j`` a ``P``\n"
     "slot ``k`` scales as ``ell^k``, a turned slot ``k`` as ``ell^-(n-k)``, and a block not at all."),
    ('t3toolbox/backend/sampling_variables.py', None,
     "``F + W + [K] + C``, with no index and no mode edge, and ``r_block_sumsq`` has the block shape ``F``.\n",
     "``F + W + [K] + C``, with no index and no mode edge, and ``r_block_sumsq`` has the block shape ``F``. The last\n"
     "column is the variable's type, explained below the table.\n"),
    ('t3toolbox/backend/sampling_definitions.py', None,
     "check a fast computation against (the library's own tests do), to see what a variable means, or to check a\n"
     "modification of your own.\n",
     "check a fast computation against (the library's own tests do), to see what a variable means, or to check a\n"
     "modification of your own. Each variable's algebraic type (``P``, ``Y*``, a turned ``P*`` or ``Y``, ``X*``) is the\n"
     "last column of the variable table of :py:mod:`t3toolbox.backend.sampling_variables`.\n"),
    # --- the oracle and its batch
    ('t3toolbox/backend/fitting.py', 'LeastSquaresBatch',
     "    sample:           typ.Any                                    # ww (input jets) | index (entries)\n",
     "    sample:           typ.Any                                    # P_in_i⊗V_i (the input jets ww) | index^d (entries)\n"),
    ('t3toolbox/backend/fitting.py', 'LeastSquaresBatch',
     "# n / batch_size on a drawn batch; None = 1", "# gamma: n / batch_size on a drawn batch; None = 1"),
    ('t3toolbox/backend/fitting.py', 'LeastSquaresOracle',
     "    **What it holds.** ``query`` returns, with its values, a\n",
     "    **The outputs, typed** (the glossary's Symbols): ``r`` and ``Jp`` are in ``Y*``, of the type of ``z``; ``g``,\n"
     "    ``JtJp`` and ``Kp`` in ``X*``, the blocks before the geometry's projection; ``f``, ``Jp_norm_sq`` and\n"
     "    ``f_resolution`` are numbers, ``ℝ`` per element of ``C`` (``K + C`` for ``Jp_norm_sq``); the two block sums\n"
     "    ``ℝ`` per block.\n"
     "\n"
     "    **What it holds.** ``query`` returns, with its values, a\n"),
    # --- D7: LocalModel, backend: the class docstring and the comment lines
    ('t3toolbox/backend/optimizers.py', 'LocalModel',
     "``hvp(p) = Pi J^T omega^2 J Pi p (+ H_R p)``, the\n      oracle's ``JtJp`` at ``Pi p``, projected.",
     "``hvp(p) = Pi J^T M J Pi p (+ H_R p)``, the\n      oracle's ``JtJp`` at ``Pi p``, projected (``M`` the batch's "
     "data metric, ``gamma * omega**2``)."),
    ('t3toolbox/backend/optimizers.py', 'LocalModel',
     "      ``hvp(p) = Pi [J^T omega^2 J Pi p + L^T (K_{A^T omega^2 r} - K_{grad f}) L Pi p] (+ H_R p)``,\n",
     "      ``hvp(p) = Pi [J^T M J Pi p + L^T (K_{A^T M r} - K_{grad f}) L Pi p] (+ H_R p)``,\n"),
    ('t3toolbox/backend/optimizers.py', 'LocalModel',
     "with ``L`` the geometry's lift from coordinates to raw core variations, ``K_{A^T omega^2 r}`` the oracle's\n"
     "      ``Kp`` along the lifted direction (the second derivative of ``<omega^2 r, z>`` at the fixed residual; one query\n",
     "with ``L`` the geometry's lift from coordinates to raw core variations, ``K_{A^T M r}`` the oracle's\n"
     "      ``Kp`` along the lifted direction (the second derivative of ``<M r, z>`` at the fixed residual; one query\n"),
    ('t3toolbox/backend/optimizers.py', 'LocalModel',
     "      of ``A^T omega^2 r``).", "      of ``A^T M r``)."),
    ('t3toolbox/backend/optimizers.py', 'LocalModel',
     "(``gn_quadratic`` is always ``‖ω J Pi p‖² (+ <p, H_R p>)``)",
     "(``gn_quadratic`` is always ``<M J Pi p, J Pi p> (+ <p, H_R p>)``)"),
    ('t3toolbox/backend/optimizers.py', 'LocalModel',
     "# ½‖ω⊙r‖²  (the weighted data misfit alone): the build's f",
     "# ½<r, M r>  (the weighted data misfit alone): the build's f"),
    ('t3toolbox/backend/optimizers.py', 'LocalModel',
     "# c = ½‖ω⊙r‖²  (+ ρ(X) if regularized)", "# ½<r, M r>  (+ ρ(X) if regularized)"),
    ('t3toolbox/backend/optimizers.py', 'LocalModel',
     "# r = z(X) - y, of the type of z (the oracle's optional r)", "# r = z(X) - y: Y*, of the type of z (the oracle's optional r)"),
    ('t3toolbox/backend/optimizers.py', 'LocalModel',
     "# g = Π 𝒥ᵀ ω² r  (+ g_R)", "# g = Π Jᵀ(M r)  (+ g_R), M = gamma * omega**2"),
    ('t3toolbox/backend/optimizers.py', 'LocalModel',
     "# 𝒥 Π p, of the type of z with K (the oracle's optional Jp)", "# J Π p: Y*, of the type of z with K (the oracle's optional Jp)"),
    ('t3toolbox/backend/optimizers.py', 'LocalModel',
     "# ‖ω 𝒥 Π p‖²  (+ ⟨p, H_R p⟩) -- the Cauchy denominator; K+C", "# <M J Π p, J Π p>  (+ ⟨p, H_R p⟩) -- the Cauchy denominator; K+C"),
    ('t3toolbox/backend/optimizers.py', 'LocalModel',
     "# GN: H p = Π 𝒥ᵀ ω² 𝒥 Π p  (+ H_R p)", "# GN: H p = Π Jᵀ M J Π p  (+ H_R p)"),
    ('t3toolbox/backend/optimizers.py', 'LocalModel',
     "# exact: Π [𝒥ᵀω²𝒥 Πp + Lᵀ K_N L Πp]  (+ H_R p)", "# exact: Π [Jᵀ M J Πp + Lᵀ K_N L Πp]  (+ H_R p)"),
    ('t3toolbox/backend/optimizers.py', 'LocalModel._curvature_from',
     "# Lᵀ (K_{Aᵀω²r} − K_{grad f}) θ̂ -- UNGAUGED, K+C", "# Lᵀ (K_{Aᵀ M r} − K_{grad f}) θ̂ -- UNGAUGED, K+C"),
    ('t3toolbox/backend/optimizers.py', 'LocalModel._curvature_from',
     "# the oracle's Kp along it: K_{Aᵀω²r} θ̂ (ω² inside)", "# the oracle's Kp along it: K_{Aᵀ M r} θ̂ (M inside)"),
    ('t3toolbox/backend/optimizers.py', 'LocalModel._curvature_block',
     "# Lᵀ (K_{Aᵀω²r} − K_{grad f}) L pp -- UNGAUGED, K+C", "# Lᵀ (K_{Aᵀ M r} − K_{grad f}) L pp -- UNGAUGED, K+C"),
    # --- D7: LocalModel, frontend
    ('t3toolbox/fitting.py', 'LocalModel',
     "(``Π Lᵀ(K_{Aᵀω²r} − K_{grad f}) L Π``)", "(``Π Lᵀ(K_{Aᵀ M r} − K_{grad f}) L Π``)"),
    ('t3toolbox/fitting.py', 'LocalModel',
     "the gradient ``g = Π 𝒥ᵀ r`` is a gauged tangent", "the gradient ``g = Π Jᵀ(M r)`` is a gauged tangent"),
    ('t3toolbox/fitting.py', 'LocalModel',
     "# c = ½‖ω⊙r‖² (+ ρ(X)), shape C", "# f = ½<r, M r> (+ ρ(X)), shape C"),
    ('t3toolbox/fitting.py', 'LocalModel',
     "# r = z(X) - y, of the type of z (the oracle's optional output)",
     "# r = z(X) - y: Y*, of the type of z (the oracle's optional output)"),
    ('t3toolbox/fitting.py', 'LocalModel',
     "# g = Π 𝒥ᵀ ω² r (+ g_R), a tangent at frame", "# g = Π Jᵀ(M r) (+ g_R), M = gamma * omega**2: a tangent at frame"),
    ('t3toolbox/fitting.py', 'LocalModel',
     "# J p = 𝒥(Π p), of the type of z with p's stack K", "# J p = J(Π p): Y*, of the type of z with p's stack K"),
    ('t3toolbox/fitting.py', 'LocalModel',
     "# pᵀ H p = ‖ω J p‖² (+ ⟨p, H_R p⟩), shape K+C", "# pᵀ H p = <M J p, J p> (+ ⟨p, H_R p⟩), shape K+C"),
    ('t3toolbox/fitting.py', 'LocalModel',
     "# H p = Π 𝒥ᵀ ω² 𝒥 Π p (+ H_R p): Gauss-Newton in either mode", "# H p = Π Jᵀ M J Π p (+ H_R p): Gauss-Newton in either mode"),
    ('t3toolbox/fitting.py', 'LocalModel',
     "# m(p) = c + ⟨g, Π p⟩ + ½ pᵀ H p, shape K+C", "# m(p) = f + ⟨g, Π p⟩ + ½ pᵀ H p, shape K+C"),
    # --- D4: the vocabulary, in the docs
    ('docs/glossary.md', None,
     "`P*` its dual, in slot coordinates. Also:",
     "`P*` its dual, in slot coordinates; `P_in,i` the input jets of mode `i`, on their own box `F_in,i` (written "
     "`P_in_i` in comments: `ww[i]` is in `P_in_i⊗V_i`); `index^d` the entries sample, `d` indices per sample, which "
     "stand for the basis vectors `e_index` and do not move with a change of coordinates. Also:"),
    ('docs/contributor/signature_style.md', None,
     "  Symbols: `P` (the jets) and `P*` (its dual, slot by slot), `V_i` and `V_i*`, `X` and `X*` (parameters), `Y*` (the\n"
     "  measurements) and `Y`, `E` and `E*`, `ℝ`;",
     "  Symbols: `P` (the jets) and `P*` (its dual, slot by slot), `P_in_i` (the input jets of mode `i`, on their sizes\n"
     "  `F_in_i`: `ww[i]` is `P_in_i⊗V_i`), `V_i` and `V_i*`, `X` and `X*` (parameters), `Y*` (the measurements) and\n"
     "  `Y`, `E` and `E*`, `ℝ`, `index^d` (the entries sample);"),
    ('docs/contributor/signature_style.md', None,
     "A signature whose arguments are\n  not data- or parameter-space objects keeps `len` first, with no type.",
     "A signature whose arguments are\n  not data- or parameter-space objects keeps `len` first, with no type. The point of"
     " a corewise call is\n  `X` (`# the point: X, (B, G), ...`); a manifold frame, its centers `Vc` and the weight "
     "`omega` take a role\n  and no type (the frame is not in its own `X`; `omega**2` is the diagonal of `M`). A "
     "transpose at a frame\n  returns `X*` in a tangent's container, before the projection."),
]

# the engine's variable table: the type of each variable, by rule (see the module docstring's explanation)
TYPE = {}
for _n in ('xi', 'mu', 'nu', 'nu_L', 'eta', 'eta_D', 'xi_U_dot', 'mu_R_dot', 'nu_L_dot', 'eta_D_dot', 'xi_dot',
           'mu_dot', 'nu_dot', 'eta_dot'):
    TYPE[_n] = 'P'
TYPE.update(z='Y*', r='Y*', Jp='Y*', w_tilde='Y turned')
for _n in ('xi_tilde', 'mu_tilde', 'nu_tilde', 'eta_tilde', '_eta_tilde_top', 'nu_L_tilde', 'eta_D_tilde',
           '_eta_D_tilde_top', 'xi_tilde_dot', 'mu_tilde_dot', 'nu_L_tilde_dot', 'eta_D_tilde_dot',
           '_eta_D_tilde_dot_top', 'nu_tilde_dot', 'eta_tilde_dot', '_eta_tilde_dot_top'):
    TYPE[_n] = 'P* turned'
for _n in ('V_tilde', 'H_tilde', 'V_tilde_dot', 'H_tilde_dot'):
    TYPE[_n] = 'X*'


def _span(text, scope):
    if scope is None:
        return 0, len(text)
    lines = text.split('\n')
    body, node = ast.parse(text).body, None
    for part in scope.split('.'):
        node = next(n for n in body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == part)
        body = node.body
    start = sum(len(l) + 1 for l in lines[:node.lineno - 1])
    end = sum(len(l) + 1 for l in lines[:node.end_lineno])
    return start, end


def _table(text):
    header = '| Variable | Index | Jet sizes | Size |'
    lines = text.split('\n')
    at = next(i for i, l in enumerate(lines) if l == header)
    lines[at] = header + ' Type |'
    lines[at + 1] = lines[at + 1] + ' --- |'
    i, seen = at + 2, []
    while lines[i].startswith('|'):
        name = lines[i].split('|')[1].strip().strip('`')
        lines[i] = lines[i] + ' %s |' % TYPE[name]
        seen.append(name)
        i += 1
    if set(seen) != set(TYPE):
        raise SystemExit('the table and TYPE differ: %s' % (set(seen) ^ set(TYPE)))
    return '\n'.join(lines)


def main():
    for rel, scope, old, new in EDITS:
        path = REPO / rel
        text = path.read_text()
        start, end = _span(text, scope)
        count = text[start:end].count(old)
        if count != 1:
            raise SystemExit('%s %s: %d occurrences of %r' % (rel, scope, count, old[:80]))
        text = text[:start] + text[start:end].replace(old, new) + text[end:]
        path.write_text(text)
        print('EDIT\t%s\t%s\t%r' % (rel, scope, old[:70]))
    path = REPO / 't3toolbox/backend/sampling_variables.py'
    path.write_text(_table(path.read_text()))
    print('TABLE\tsampling_variables: %d rows typed' % len(TYPE))


if __name__ == '__main__':
    main()
