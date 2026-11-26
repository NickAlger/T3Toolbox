"""M4 step B: the prose and the `c` lines, by explicit edits (each old text matched exactly once)."""
import ast
import pathlib
import re
import sys

ROOT = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(ROOT / 'dev/graded_messages/implementation/m4'))
from rename import TARGETS          # noqa: E402

WIDTH = 120


def atoms(text):
    """Words, with an inline literal ``...`` (which may hold spaces) kept whole."""
    out, cur = [], None
    for token in text.split():
        cur = token if cur is None else cur + ' ' + token
        if cur.count('``') % 2 == 0:
            out.append(cur)
            cur = None
    assert cur is None, text
    return out


def wrap(text, indent):
    lines, cur = [], indent
    for a in atoms(text):
        trial = (cur + ' ' + a) if cur.strip() else (indent + a)
        if len(trial) > WIDTH and cur.strip():
            lines.append(cur)
            cur = indent + a
        else:
            cur = trial
    lines.append(cur)
    return '\n'.join(lines)


class File:
    def __init__(self, rel):
        self.rel, self.path = rel, ROOT / rel
        self.text = self.path.read_text()

    def rep(self, old, new, count=1):
        n = self.text.count(old)
        if n != count:
            raise SystemExit('%s: %r found %d times, expected %d' % (self.rel, old[:70], n, count))
        self.text = self.text.replace(old, new)

    def para(self, marker, new_text, nth=1):
        """Replace the paragraph (consecutive non-blank lines) holding ``marker`` (its nth occurrence) by
        ``new_text`` wrapped at the paragraph's indentation."""
        lines = self.text.split('\n')
        hits = [i for i, l in enumerate(lines) if marker in l]
        if len(hits) != 1:
            raise SystemExit('%s: marker %r found %d times' % (self.rel, marker, len(hits)))
        i = hits[nth - 1]
        quotes = ('"""', "'''")
        a = i
        while (a > 0 and not lines[a].lstrip().startswith(quotes) and lines[a - 1].strip()
               and set(lines[a - 1].strip()) != {'-'}):
            a -= 1
        b = i
        while (b + 1 < len(lines) and not lines[b].rstrip().endswith(quotes) and lines[b + 1].strip()
               and lines[b + 1].strip() not in quotes):
            b += 1
        indent = re.match(r'\s*', lines[a]).group(0)
        lines[a:b + 1] = wrap(new_text, indent).split('\n')
        self.text = '\n'.join(lines)

    def save(self):
        self.path.write_text(self.text)


# ------------------------------------------------------------------ the `c` lines (D1: the type on the `c` line)
def c_type(fname, comment):
    kind = 'probe' if 'probe' in fname else 'entries' if 'entries' in fname else 'apply'
    ambient = 'ambient' in fname
    rest = comment
    for junk in ('c, ', 'of the type of z: ', 'probe residuals, ', '; no weight'):
        rest = rest.replace(junk, '')
    if kind == 'probe':
        t = 'V_i' if ambient else 'P*⊗V_i'
        return '# %s, %s' % (t, rest)
    t = 'ℝ' if (ambient or kind == 'entries') else 'P*'
    return '# %s, %s' % (t, rest)


def c_lines():
    for rel, names in TARGETS.items():
        f = File(rel)
        tree = ast.parse(f.text)
        lines = f.text.split('\n')
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in names:
                for a in node.args.args + node.args.kwonlyargs:
                    if a.arg == 'c':
                        line = lines[a.lineno - 1]
                        m = re.match(r'^(\s*)c:(\s+)(\S.*?)(\s+)# (.*)$', line)
                        assert m, line
                        indent, gap, annot, gap2, comment = m.groups()
                        new = '%sc:%s%s%s%s' % (indent, gap + ' ' * 7, annot, gap2, c_type(node.name, comment))
                        lines[a.lineno - 1] = new
        f.text = '\n'.join(lines)
        f.save()


# ------------------------------------------------------------------ the texts
Y = {'probe': '``P*⊗V_i`` per mode', 'apply': '``P*``', 'entries': '``ℝ``'}
WEIGHTED = ('It carries no metric, and none is applied here: for the gradient of a weighted least-squares fit '
            'pass ``data_metric(r, weight, minibatch_scale)`` (or ``batch.data_metric(r)``), which is ``r`` itself '
            'with no weight and no scale.')
CALL_TAIL = ('For repeated calls at one frame (inside an iterative solver), call '
             ':py:func:`~t3toolbox.backend.sampling_variables.sampling_variables` with ``held=``: the frame\'s sweeps '
             'are then computed once.')


def c_sentence(kind, stack=True):
    k = '; a stack ``K`` of them allowed' if stack else ''
    return ('``c`` is an element of ``Y`` (%s per sample%s), paired with ``z`` slot by slot. %s'
            % (Y[kind], k, WEIGHTED))


def tv_para(kind):
    return (c_sentence(kind) + ' One call of :py:func:`~t3toolbox.backend.sampling_variables.sampling_variables` '
            'wanting ``JtJp`` with ``c`` held as ``Jp`` (the route of a held ``Jp``: ``J^T`` of the vector held; the '
            'engine turns ``c`` into its dual-jet form inside). The blocks are summed over the samples by default. '
            + CALL_TAIL)


def cw_para(kind):
    return (c_sentence(kind) + ' One call of :py:func:`~t3toolbox.backend.sampling_variables.sampling_variables` on '
            'the corewise point ``x`` wanting ``JtJp`` with ``c`` held as ``Jp``. Summed over the samples by default. '
            + CALL_TAIL)


def h_para(kind):
    return (c_sentence(kind, stack=False) + ' One call of '
            ':py:func:`~t3toolbox.backend.sampling_variables.sampling_variables` on the corewise point ``x`` wanting '
            '``Kp`` with ``c`` held as ``r``. Summed over the samples (``sum_over_samples=False``: per sample, ``W`` '
            'leading the blocks\' stack). ' + CALL_TAIL)


PHI = '``phi(x) = <c, z(x)>`` (the pairing of ``Y`` with ``Y*``, slot by slot: no metric)'


def backend_ragged():
    for kind, rel in (('apply', 't3toolbox/backend/apply.py'), ('entries', 't3toolbox/backend/entries.py'),
                      ('probe', 't3toolbox/backend/probing.py')):
        f = File(rel)
        f.para('``c`` is a vector of the type of ``z`` (a stack ``K`` of them allowed), with no weight; it is turned',
               tv_para(kind))
        f.rep('``phi(x) = <residual, z(x)>`` (plain inner product).', PHI + '.')
        f.para('``c`` is a vector of the type of ``z`` (a stack ``K`` allowed), with no weight, turned inside.',
               cw_para(kind))
        f.rep('the second derivative of ``phi(x) = <residual, z(x)>`` in the cores',
              'the second derivative of ``phi(x) = <c, z(x)>`` in the cores')
        f.para('``c`` is a vector of the type of ``z``, no weight, turned inside.', h_para(kind))
        f.save()
    # the corewise transposes' first paragraph, rewrapped (PHI is longer than the old text)
    for rel in ('t3toolbox/backend/apply.py', 't3toolbox/backend/entries.py', 't3toolbox/backend/probing.py'):
        f = File(rel)
        f.para("'''The corewise transpose: ``J^T`` applied to ``c`` in the cores of the point ``x``, the gradient of",
               "'''The corewise transpose: ``J^T`` applied to ``c`` in the cores of the point ``x``, the gradient of "
               + PHI + '.')
        f.save()

    f = File('t3toolbox/backend/apply.py')
    f.rep('# True: W becomes the CP rank (ambient J^T r)', '# True: W becomes the CP rank (ambient J^T c)')
    f.para('The *ambient* adjoint -- the transpose of ``apply`` as a linear map on the **full tensor space**',
           'The *ambient* adjoint -- the transpose of ``apply`` as a linear map on the **full tensor space** '
           '(``X -> ( <X, w0^W (x) ... (x) w_{d-1}^W> )_W``). Frame-free; the back-projection '
           '``c * (w0 (x) ... (x) w_{d-1})`` is rank-1, whose natural representation is a **canonical (CP) '
           'decomposition** (``apply`` consumes one vector per mode; its adjoint emits one scaled vector per mode). '
           'This is distinct from the *corewise* transpose (gradient w.r.t. a base point\'s cores) and the *tangent* '
           'transpose (Riemannian gradient) -- see ``docs/transposes.md`` for the full taxonomy. ``c`` is an element '
           'of ``Y`` (``ℝ`` per sample), paired with ``apply``\'s output: ``<J^T c, T> = <c, apply(T, ww)>``, with no '
           'metric (for a weighted fit\'s, pass ``data_metric(r, weight, minibatch_scale)``).')
    f.rep('``sum_W residual_W * (w0^W (x) ...)`` (the ambient ``J^T r``).', '``sum_W c_W * (w0^W (x) ...)`` (the ambient ``J^T c``).')
    f.save()

    f = File('t3toolbox/backend/entries.py')
    f.para('The ``entries`` counterpart of :py:func:`t3_apply_ambient_transpose` -- identical',
           'The ``entries`` counterpart of :py:func:`t3_apply_ambient_transpose` -- identical with the apply vectors '
           'replaced by the unit vectors ``e_{index_k}``, so the CP factors are one-hots and the back-projection is '
           '``c * e_{idx_0} (x) ... (x) e_{idx_{d-1}}`` (``c`` an element of ``Y``, ``ℝ`` per sample, as there). '
           '``sum_over_samples=True`` makes ``W`` the CP rank (scatter-adding colliding indices -- the ``J^T c`` for '
           'entry sampling). ``shape`` supplies the ambient dims, which (unlike the apply case, where ``ww`` carries '
           'them) ``c`` and ``index`` alone do not determine. Returns CP factors (see the apply version).')
    f.save()

    f = File('t3toolbox/backend/probing.py')
    f.rep("'''Ambient transpose of :py:func:`t3_probe`: back-project probe residuals into CP factors.",
          "'''Ambient transpose of :py:func:`t3_probe`: back-project ``c`` into CP factors.")
    f.para('Probe returns ``d`` vectors (one free mode each), so the residual ``c`` is ``d`` vectors; the',
           'The *ambient* adjoint -- the transpose of ``probe`` as a linear map on the **full tensor space**. Probe '
           'returns ``d`` vectors (one free mode each), so ``c`` (an element of ``Y``: ``V_i`` per mode and sample, '
           'paired with ``probe``\'s output, with no metric) is ``d`` vectors; the back-projection is the rank-``d`` '
           'tensor')
    f.rep('sum_i  w0 (x) ... (x) w_{i-1} (x) residual_i (x) w_{i+1} (x) ... (x) w_{d-1}',
          'sum_i  w0 (x) ... (x) w_{i-1} (x) c_i (x) w_{i+1} (x) ... (x) w_{d-1}')
    f.para("(term ``i`` has the residual ``residual_i`` in slot ``i`` and the probe vectors elsewhere), whose",
           '(term ``i`` has ``c_i`` in slot ``i`` and the probe vectors elsewhere), whose natural representation is a '
           '**canonical (CP) decomposition** of rank ``d``. Frame-free. Distinct from the *corewise* transpose '
           '(gradient w.r.t. a frame\'s cores) and the *tangent* transpose (Riemannian gradient); see '
           '``docs/transposes.md``. The ``apply``/``entries`` analog is the rank-1 (or rank-``|W|``) '
           ':py:func:`t3_apply_ambient_transpose`.')
    f.rep("rank slot ``k`` = ``residual_k``,", "rank slot ``k`` = ``c_k``,")
    f.para('The adjoint of ``probe`` on the full tensor space, ``<J^T r, T> = <r, dense_probe(T, ww)>``.',
           'The adjoint of ``probe`` on the full tensor space, ``<J^T c, T> = <c, dense_probe(T, ww)>``. **The '
           'argument order** (changed in this release: the sample, then ``c``): ``c`` and the sample are both ``d`` '
           'vectors, so the old order ``(c, ww)`` runs and returns the transpose of a different problem:')
    f.rep('C  = c[0].shape[nW:-1]          # frame stack (residual[i] is W + C + (Ni,))',
          'C  = c[0].shape[nW:-1]          # frame stack (c[i] is W + C + (Ni,))')
    f.para('**The argument order** (changed in this release: the point, the sample, then the residual).',
           '**The argument order** (changed in this release: the point, the sample, then ``c``). ``c`` and the sample '
           'are both ``d`` vectors, so ``(x, c, ww)`` runs and returns the transpose of a different problem. Checked '
           'against the forward derivative at the intended sample:')
    f.rep('# the residual and the sample swapped', '# c and the sample swapped')
    f.save()


def backend_uniform():
    f = File('t3toolbox/backend/ut3_sampling.py')
    for kind, mod in (('apply', 'apply'), ('entries', 'entries'), ('probe', 'probing')):
        packed = ' (a ragged ``c`` packed here)' if kind == 'probe' else ''
        f.para(':py:func:`~t3toolbox.backend.%s.t3_%s_corewise_transpose`, the gradient of ``<residual, z(x)>`` in the'
               % (mod, kind),
               '"""The corewise transpose on supercores: the uniform twin of '
               ':py:func:`~t3toolbox.backend.%s.t3_%s_corewise_transpose`, the gradient of ``<c, z(x)>`` in the cores '
               '(``c`` an element of ``Y``, with no metric, as there). The engine\'s ``JtJp`` with ``c`` held as '
               '``Jp``%s; the output is masked by the point\'s masks."""' % (mod, kind, packed))
    f.para('Hessian of ``theta -> <z~, probe(Phi(theta))>`` applied to ``theta_hat``, on supercores.',
           '"""The uniform twin of :py:func:`~t3toolbox.backend.probing.t3_probe_corewise_hessian`: the core-coordinate '
           'Hessian of ``theta -> <c, probe(Phi(theta))>`` applied to ``theta_hat``, on supercores (``c`` an element '
           'of ``Y``, with no metric, as there). ``theta`` is masked with its masks, ``theta_hat`` with the SAME masks, '
           '``c`` / ``ww`` packed to ``N``, and the output re-masked; per stack element the real block is the ragged '
           'two-hole of the real blocks, whatever the padding holds.')
    f.save()

    f = File('t3toolbox/backend/utv_sampling.py')
    f.para("projector): the engine's ``JtJp`` with the residual held as ``Jp`` (a ragged residual packed here), then the",
           '"""The uniform twin of :py:func:`~t3toolbox.backend.probing.tv_probe_transpose` (the bare ``J^T``, no '
           'gauge projector; ``c`` an element of ``Y``, with no metric, as there): the engine\'s ``JtJp`` with ``c`` '
           'held as ``Jp`` (a ragged ``c`` packed here), then the result variation masks attached, the frame\'s gauge '
           'masks broadcast over the new tangent stack ``K_new`` (``W+K`` if ``sum_over_samples=False``, else '
           '``K``), and the output re-masked with them (exact zero padding)."""')
    for kind in ('apply', 'entries'):
        f.para(':py:func:`~t3toolbox.backend.%s.tv_%s_transpose` (the bare ``J^T``, no gauge' % (kind, kind),
               '"""The uniform twin of :py:func:`~t3toolbox.backend.%s.tv_%s_transpose` (the bare ``J^T``, no gauge '
               'projector; ``c`` an element of ``Y``, with no metric, as there): the engine\'s ``JtJp`` with ``c`` '
               'held as ``Jp``, then the result variation masks attached, the frame\'s gauge masks broadcast over the '
               'new tangent stack ``K_new`` (``W+K`` if ``sum_over_samples=False``, else ``K``), and the output '
               're-masked with them (exact zero padding)."""' % (kind, kind))
    f.save()


GN = ('the Gauss-Newton gradient ``J^T(M r)`` when ``c = data_metric(r, weight, minibatch_scale)``')


def frontend():
    f = File('t3toolbox/manifold.py')
    f.rep('"""Apply the transpose ``(J^(s))^T`` of the probe map to residuals; returns a T3Tangent at ``frame``.',
          '"""Apply the transpose ``(J^(s))^T`` of the probe map to ``c``; returns a T3Tangent at ``frame``.')
    f.para('The adjoint of :py:meth:`probe`. The residuals live in the forward probe space,',
           'The adjoint of :py:meth:`probe`. ``c`` is an element of ``Y`` (``P*⊗V_i`` per mode), paired with '
           '``probe``\'s output slot by slot, and has its shape, ``elm_shape = W + K + C + (Ni,)`` (probe stack ``W`` '
           'outer, optional tangent batch ``K``, frame stack ``C`` inner -- the output space of a ``K``-stacked '
           ':py:meth:`probe`; ``K`` is empty in the common case). ' + WEIGHTED + ' The tangent batch ``K`` is always '
           'carried to the result\'s tangent stack; the probe stack ``W`` is summed or kept per '
           '``sum_over_samples``:')
    f.rep("""          ``K`` (frame stack ``C``) -- the usual Gauss-Newton ``J^T r`` (a single tangent when ``K = ()``).
        - ``sum_over_samples=False``: each probe residual becomes one tangent -- the result's tangent stack""",
          """          ``K`` (frame stack ``C``) -- ``J^T c`` (a single tangent when ``K = ()``); the Gauss-Newton gradient
          ``J^T(M r)`` when ``c = data_metric(r, weight, minibatch_scale)``.
        - ``sum_over_samples=False``: each sample's ``c`` becomes one tangent -- the result's tangent stack""")
    f.para('``sum_over_samples=True == Σ_W sum_over_samples=False``. With ``order``, the residual is of the type of',
           '``sum_over_samples=True == Σ_W sum_over_samples=False``. With ``order``, ``c`` has the shape of '
           '``probe(ww, order)``, a vector per jet slot, and the jet axes are summed (the slot-by-slot pairing of '
           '``Y`` with ``Y*``). See *Batching & stacking* §11 (``docs/batching_and_stacking.md``) for which mode to '
           'use and why.')
    f.rep('With ``K``-stacked residuals (``W + K + C``), the tangent batch', 'With a ``K``-stacked ``c`` (``W + K + C``), the tangent batch')
    f.rep('"""Apply the transpose ``apply^T`` of :py:meth:`apply`: back-project a residual into a tangent.',
          '"""Apply the transpose ``apply^T`` of :py:meth:`apply`: back-project ``c`` into a tangent.')
    f.para('The adjoint of the (linear-in-the-variation) all-modes :py:meth:`apply`. With',
           'The adjoint of the (linear-in-the-variation) all-modes :py:meth:`apply`. ``c`` is an element of ``Y`` '
           '(``P*`` per sample), paired with ``apply``\'s output slot by slot. ' + WEIGHTED + ' With '
           '``sum_over_samples=True`` (default) the sample stack ``W`` is summed -- ``J^T c``, ' + GN + '; with '
           '``sum_over_samples=False`` it becomes the result\'s tangent stack (one tangent per sample). Needs only '
           'the frame sweep + a single-term scatter assembly (cheaper than a general :py:meth:`probe_transpose`).')
    f.para('The adjoint of :py:meth:`entries` -- identical to :py:meth:`apply_transpose` with the upward',
           'The adjoint of :py:meth:`entries` -- identical to :py:meth:`apply_transpose` with the upward ``ξ̂`` from '
           'fiber slicing and unit apply-vectors ``e_{index}``; ``c`` is an element of ``Y`` (``ℝ`` per sample), with '
           'no metric, as there. ``sum_over_samples`` as in :py:meth:`apply_transpose` (see *Batching & stacking* '
           '§11).')
    f.para('that turns a bare adjoint ``J^T r`` into the Riemannian gradient. For the vector-preserving',
           'Orthogonally projects ``v``\'s variations onto the gauged tangent space (the gauge conditions (47)-(48), '
           'Appendix A.3). Represents a DIFFERENT tangent vector than ``v``. This is the map that turns a bare adjoint '
           '``J^T c`` (with ``c = M r``: the misfit\'s raw gradient) into the Riemannian gradient. For the '
           'vector-preserving gauge fix see :py:meth:`project_oblique`.')
    f.save()

    f = File('t3toolbox/uniform_manifold.py')
    f.para('"""Apply the transpose ``𝒥ᵀ`` of the probe map to residuals; returns a :py:class:`UT3Tangent` at',
           '"""Apply the transpose ``𝒥ᵀ`` of the probe map to ``c``; returns a :py:class:`UT3Tangent` at ``frame``. '
           'The adjoint of :py:meth:`probe`. Uniform mirror of :py:meth:`~t3toolbox.manifold.T3Tangent.probe_transpose`: '
           '``c`` (an element of ``Y``, with no metric, as there) has the forward probe shape (``W+K+C``); '
           '``sum_over_samples=True`` (default) sums the probe stack ``W`` (``K``; ``𝒥ᵀc``, the Gauss-Newton gradient '
           'when ``c = M r``), ``False`` keeps it as the result\'s tangent stack (``W+K``). The bare ``𝒥ᵀ`` (no gauge '
           'projector ``Π``).')
    f.rep('The defining property -- the adjoint identity ``<r, 𝒥v> = <𝒥ᵀr, v>`` (summed over probes):',
          'The defining property -- the adjoint identity ``<c, 𝒥v> = <𝒥ᵀc, v>`` (summed over probes):')
    f.para('"""Apply the transpose ``𝒥ᵀ`` of :py:meth:`apply` -- back-project a residual into a tangent',
           '"""Apply the transpose ``𝒥ᵀ`` of :py:meth:`apply` -- back-project ``c`` into a tangent at ``frame``. The '
           'adjoint of :py:meth:`apply`. Uniform mirror of :py:meth:`~t3toolbox.manifold.T3Tangent.apply_transpose` '
           '(``c`` an element of ``Y``, with no metric, as there); ``sum_over_samples`` as in '
           ':py:meth:`probe_transpose`. The bare ``𝒥ᵀ`` (no gauge projector ``Π``)."""')
    f.para('Appendix A.3). Represents a DIFFERENT tangent vector than ``v`` -- the map that turns a bare adjoint',
           'Orthogonally projects ``v``\'s variations onto the gauged tangent space (conditions (47)-(48), Appendix '
           'A.3). Represents a DIFFERENT tangent vector than ``v`` -- the map that turns a bare adjoint ``J^T c`` (with '
           '``c = M r``: the misfit\'s raw gradient) into the Riemannian gradient. For the vector-preserving fix see '
           ':py:meth:`project_oblique`. **Safe mode** requires ``v``\'s frame orthogonal (raises otherwise; skipped '
           'under ``safety.unsafe()`` / a jax trace)."""')
    f.save()

    f = File('t3toolbox/tucker_tensor_train.py')
    f.rep('"""Ambient transpose of :py:meth:`probe`: back-project probe residuals into canonical (CP) factors.',
          '"""Ambient transpose of :py:meth:`probe`: back-project ``c`` into canonical (CP) factors.')
    f.para('``docs/transposes.md``). ``probe`` returns ``d`` vectors, so the residual is ``d``',
           'The probe counterpart of :py:meth:`apply_ambient_transpose` -- the literal adjoint of ``probe`` as a '
           'linear map on the full tensor space (one of **three** probe transposes; full taxonomy in '
           '``docs/transposes.md``). ``probe`` returns ``d`` vectors, so ``c`` (an element of ``Y``: ``V_i`` per mode '
           'and sample, with no metric) is ``d`` vectors; the back-projection is the **rank-``d``** tensor '
           '``sum_i (w0 (x) ... (x) c_i (x) ... (x) w_{d-1})`` (``c_i`` in slot ``i``, probe vectors elsewhere), '
           'returned as **CP factors**. Frame-free. Realize a ``TuckerTensorTrain`` with :py:meth:`from_canonical`.')
    f.rep('(one rank-``d|W|`` CP, the ambient ``J^T r``)', '(one rank-``d|W|`` CP, the ambient ``J^T c``)')
    f.para('**The argument order** (changed in this release: the sample first, then the residual). A call in the',
           '**The argument order** (changed in this release: the sample first, then ``c``). A call in the old order '
           '``(c, ww)`` runs, since both are ``d`` vectors, and returns the transpose of a different problem. Checked '
           'against ``probe`` at the intended sample, ``<J^T c, y> = <c, y.probe(ww)>``:')
    f.rep('# the sample, then the residual', '# the sample, then c', count=2)
    f.para('-- the tangent probe transpose with this tensor\'s own cores substituted for the orthogonal',
           'The probe counterpart of :py:meth:`apply_corewise_transpose` -- the gradient of ``self.probe(ww)`` w.r.t. '
           '``self``\'s cores (for core-wise optimizers, Adam / L-BFGS, fitting from probes), returned as a raw '
           '``(tucker_grads, tt_grads)`` tuple shaped like ``self.data`` (a gradient, not a tensor; no ``|W|`` '
           'blow-up). Computed by the Section 6.3 corewise simplification -- the tangent probe transpose with this '
           'tensor\'s own cores substituted for the orthogonal frames. See ``docs/transposes.md`` for the '
           'ambient/corewise/tangent distinction; the ambient and Riemannian probe transposes are '
           ':py:meth:`probe_ambient_transpose` and ``T3Tangent.probe_transpose``. ``c`` is an element of ``Y`` '
           '(``P*⊗V_i`` per mode), paired with ``probe``\'s output slot by slot. ' + WEIGHTED + ' '
           '``sum_over_samples=True`` (the default) is the summed ``J^T c`` (' + GN + '); ``False`` keeps the probe '
           'stack ``W``.')
    f.para('**The argument order** (changed in this release: the sample first, then the residual, as everywhere in',
           '**The argument order** (changed in this release: the sample first, then ``c``, as everywhere in the '
           'library). ``c`` and the sample are both ``d`` vectors here, so a call in the old order ``(c, ww)`` runs, '
           'and returns the transpose of a different problem. Checked against the forward derivative at the intended '
           'sample, ``<c, J x_dot> = <x_dot, J^T c>``:')
    f.para('Returns the CP ``factors`` (``c`` folded into the first), in the layout',
           'Returns the CP ``factors`` (``c`` folded into the first), in the layout :py:meth:`from_canonical` '
           'consumes. ``c`` is an element of ``Y`` (``ℝ`` per sample), paired with ``apply``\'s output, with no metric '
           '(for a weighted fit\'s, pass ``data_metric(r, weight, minibatch_scale)``). See *Batching & stacking* §11 '
           '(``docs/batching_and_stacking.md``) for the stacking conventions.')
    f.rep('``sum_W c_W (w0^W (x) ...)`` (the ambient ``J^T r``).', '``sum_W c_W (w0^W (x) ...)`` (the ambient ``J^T c``).')
    f.para('``sum_over_samples=True`` makes ``W`` the CP rank (scatter-adding colliding indices -- the',
           '``sum_over_samples=True`` makes ``W`` the CP rank (scatter-adding colliding indices -- the ``J^T c`` for '
           'entry sampling); ``False`` keeps ``W`` as a stacking axis. ``shape`` supplies the ambient dims ``(N0, ..., '
           'N(d-1))``, which (unlike :py:meth:`apply_ambient_transpose`, where ``ww`` carries them) ``c`` and '
           '``index`` alone do not determine. ``c`` is an element of ``Y`` (``ℝ`` per sample), with no metric. Returns '
           'CP ``factors`` for :py:meth:`from_canonical`.')
    f.para('Back-projecting a residual scatters it at the index (realize the CP factors with',
           'Back-projecting ``c`` scatters it at the index (realize the CP factors with :py:meth:`from_canonical`):')
    f.para('own cores substituted for the orthogonal frames (orthogonality is not required).',
           'Computed by the Section 6.3 corewise simplification -- the tangent transpose with this tensor\'s own cores '
           'substituted for the orthogonal frames (orthogonality is not required). ``c`` is an element of ``Y`` '
           '(``P*`` per sample), paired with ``apply``\'s output slot by slot. ' + WEIGHTED + ' '
           '``sum_over_samples=True`` (the default) is the summed ``J^T c`` (' + GN + '); ``False`` keeps the probe '
           'stack ``W`` (one gradient set per probe, e.g. to assemble ``J^T J``).')
    f.para('``shape`` argument -- the ambient dims come from ``self``. ``sum_over_samples=True`` (the default) scatter-adds',
           'The ``entries`` counterpart of :py:meth:`apply_corewise_transpose` -- see it (and ``docs/transposes.md``) '
           'for the ambient-vs-corewise-vs-tangent distinction; this is the *corewise* gradient (w.r.t. the cores, for '
           'Adam / L-BFGS), returned as a raw ``(tucker_grads, tt_grads)`` tuple. Unlike '
           ':py:meth:`entries_ambient_transpose` it needs **no** ``shape`` argument -- the ambient dims come from '
           '``self``. ``c`` is an element of ``Y`` (``ℝ`` per sample), with no metric, as there. '
           '``sum_over_samples=True`` (the default) scatter-adds colliding indices (``J^T c``).')
    f.para('"""The corewise Hessian of probing: the second derivative of ``phi(x) = <residual, x.probe(ww)>`` in the',
           '"""The corewise Hessian of probing: the second derivative of ``phi(x) = <c, x.probe(ww)>`` in the cores at '
           '``self``, applied to ``core_directions`` (the derivative of :py:meth:`probe_corewise_transpose` along '
           'them). ``c`` is an element of ``Y`` (``P*⊗V_i`` per mode), paired with ``probe``\'s output slot by slot, '
           'with no metric (for a weighted fit\'s, pass ``data_metric(r, weight, minibatch_scale)``). '
           '``core_directions`` as in :py:meth:`probe_corewise_tangent`. Summed over the samples unless '
           '``sum_over_samples=False``.')
    f.save()

    f = File('t3toolbox/uniform_tucker_tensor_train.py')
    f.para('Section 6.3 ``(U,L,R)->G`` substitution; no orthogonality required. ``sum_over_samples=True`` (default) sums the',
           'Returns the raw gradient supercores ``(tucker_grad, tt_grad)`` (a gradient, NOT a tensor). The Section 6.3 '
           '``(U,L,R)->G`` substitution; no orthogonality required. ``c`` is an element of ``Y`` (``P*`` per sample), '
           'with no metric, as in the ragged method. ``sum_over_samples=True`` (default) sums the apply stack ``W`` '
           '(``Jᵀc``; the Gauss-Newton gradient when ``c = M r``), so the grads are shaped exactly like the '
           'supercores; ``False`` keeps ``W`` as a leading grad stack. Uniform mirror of '
           ':py:meth:`~t3toolbox.tucker_tensor_train.TuckerTensorTrain.apply_corewise_transpose`.')
    f.rep('# residual, shape=W', '# c, shape=W')
    f.para(':py:meth:`apply_corewise_transpose` but the residual carries one free mode per probe. See',
           '"""Corewise transpose of :py:meth:`probe`: gradient w.r.t. the supercores. Like '
           ':py:meth:`apply_corewise_transpose` but ``c`` carries one free mode per probe. See '
           ':py:meth:`apply_corewise_transpose` for the return contract and ``sum_over_samples``."""')
    for kind in ('probe', 'apply', 'entries'):
        f.para('"""The corewise Hessian of :py:meth:`%s` on supercores: the second derivative of ``<residual, z>`` in the'
               % kind,
               '"""The corewise Hessian of :py:meth:`%s` on supercores: the second derivative of ``<c, z>`` in the '
               'supercores, applied to ``core_directions`` (as :py:meth:`%s_corewise_tangent` takes them). The uniform '
               'mirror of :py:meth:`~t3toolbox.tucker_tensor_train.TuckerTensorTrain.%s_corewise_hessian`; the output\'s '
               'padding is exactly zero."""' % (kind, kind, kind))
    f.save()

    f = File('t3toolbox/backend/sampling_variables.py')
    f.para('``Jp``, so **holding any vector of** ``Jp``\'s **type as** ``Jp`` **gives** ``J^T`` **of that vector** (unweighted',
           '**The adjoint seeded by** ``Jp`` (``JtJp``) is the private twin of each row on the residual\'s path from '
           '``w_tilde`` to ``V_tilde``, ``H_tilde`` and the short node, named ``\'_Jp.\' + name.lstrip(\'_\')``: the same '
           'step and start, the call\'s ``K``, the seed reading ``Jp`` for ``r``. ``JtJp`` is the twin of ``g``. The '
           'full ``eta_tilde`` has no twin. '
           '``JtJp`` has a second route, from a held ``Jp`` with no direction (``held:Jp``, P12): the twin path is '
           'linear in ``Jp``, so **holding any vector of** ``Jp``\'s **shape as** ``Jp`` **gives** ``J^T`` **of that '
           'vector** (the seed is the data metric of the held vector at ``gamma = 1``, ``weight**2`` times it, turned; '
           'with no weight, the vector itself). The transposes take this route for ``c``, an element of ``Y`` with '
           'the shape of ``z`` (they pass no weight), with a stack ``K`` read off the held ``Jp``; a direction, when '
           'given, wins, and its ``K`` must agree.')
    f.para('direction. ``JtJp`` is linear in ``Jp``, so holding any vector ``c`` of ``Jp``\'s shape as ``Jp`` (with no',
           'It checks no shape: the engine checks every held variable it is handed, at every call, against the shapes '
           'of its table, and trusts the values. **One contract on a value:** a held ``Jp`` need not be the output of '
           'a direction. ``JtJp`` is linear in ``Jp``, so holding any vector ``c`` of ``Jp``\'s shape as ``Jp`` (with '
           'no direction) gives ``J^T`` of ``weight**2 c``: this is how the transposes compute ``J^T c`` (they pass no '
           'weight), ``K`` read off ``c``. Where jax is importable it is a registered pytree: the filled slots are the '
           'leaves, the set of filled slots is the static structure. It refuses ``==`` and is unhashable, as every '
           'array-carrying class of the library is: compare the slots you mean.')
    f.save()


if __name__ == '__main__':
    c_lines()
    backend_ragged()
    backend_uniform()
    frontend()
    print('done')
