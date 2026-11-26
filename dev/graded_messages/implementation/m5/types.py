"""M5, the types: the algebraic types written into the signature comments of the public sampling surface.

Run from the repository root: ``python dev/graded_messages/implementation/m5/types.py [--check]``. It edits comments
only: for each function in ``SCOPE`` it reads the signature block (from ``def`` to the first statement of the body)
and replaces each trailing comment found in ``COMMENTS`` (old text -> new text, the ``#`` kept in its column; a line
over 120 characters wraps with ``#   `` continuation lines in the same column). A ``c`` line whose ``#`` is one column
off its predecessor's is realigned (M4 review, minor 2). ``--check`` changes nothing and reports. Every edit is printed;
the unmapped comments in scope are listed, so that what is left alone is left on purpose.
"""
import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[4] / 't3toolbox'
WIDTH = 120
KINDS = ('probe', 'apply', 'entries')
MOD = {'probe': 'probing', 'apply': 'apply', 'entries': 'entries'}

SCOPE = {}
for k in KINDS:
    SCOPE.setdefault('backend/%s.py' % MOD[k], []).extend(
        ['t3_%s' % k, 'tv_%s' % k, 'tv_%s_transpose' % k, 't3_%s_ambient_transpose' % k,
         't3_%s_corewise_tangent' % k, 't3_%s_corewise_transpose' % k, 't3_%s_corewise_hessian' % k, 'dense_%s' % k])
    SCOPE.setdefault('backend/ut3_sampling.py', []).extend(
        ['ut3_%s' % k] + ['ut3_%s_corewise_%s' % (k, f) for f in ('tangent', 'transpose', 'hessian')])
    SCOPE.setdefault('backend/utv_sampling.py', []).extend(['utv_%s' % k, 'utv_%s_transpose' % k])
    SCOPE.setdefault('tucker_tensor_train.py', []).extend(
        ['TuckerTensorTrain.%s' % k] + ['TuckerTensorTrain.%s_%s' % (k, f) for f in
                                        ('ambient_transpose', 'corewise_transpose', 'corewise_tangent', 'corewise_hessian')])
    SCOPE.setdefault('uniform_tucker_tensor_train.py', []).extend(
        ['UniformTuckerTensorTrain.%s' % k] + ['UniformTuckerTensorTrain.%s_%s' % (k, f) for f in
                                               ('corewise_transpose', 'corewise_tangent', 'corewise_hessian')])
    SCOPE.setdefault('manifold.py', []).extend(['T3Tangent.%s' % k, 'T3Tangent.%s_transpose' % k])
    SCOPE.setdefault('uniform_manifold.py', []).extend(['UT3Tangent.%s' % k, 'UT3Tangent.%s_transpose' % k])
SCOPE['backend/jets.py'] = ['data_metric', 'blocks_to_weight', 'jet_to_dual_jet', 'dual_jet_to_jet',
                            'derivatives_to_taylor', 'taylor_to_derivatives', 'derivative_weight_to_taylor_weight',
                            'prepare_derivative_data', 'affine_jet', 'index_to_one_hot', 'check_sample', 'check_data',
                            'check_weight']
SCOPE['backend/data_preparation.py'] = ['data_block_norms', 'equalizing_weight', 'rescale_direction',
                                        'rescale_coordinates']
SCOPE['fitting.py'] = ['rescale_coordinates']
SCOPE['backend/sampling_variables.py'] = ['sampling_variables']
SCOPE['backend/uniform_fitting.py'] = ['pack_sample', 'pack_data']
SCOPE['backend/fitting.py'] = ['LeastSquaresBatch.data_metric', 'LeastSquaresOracle.query']
SCOPE['backend/optimizers.py'] = ['Oracle.query']

COMMENTS = {
    # --- the point, the frame, the directions
    "(tucker_cores, tt_cores), elm_shapes C+(nDi,Ni), C+(ri,nDi,r(i+1))":
        "the point: X, (tucker_cores, tt_cores), elm_shapes C+(nDi,Ni), C+(ri,nDi,r(i+1))",
    "(tucker_cores, tt_cores): the point": "the point: X, (tucker_cores, tt_cores)",
    "(B, G): the point whose cores the directions vary": "the point: X, (B, G), whose cores the directions vary",
    "the point's .data: (B, G, shape, masks), stack C": "the point: X, its .data (B, G, shape, masks), stack C",
    "(D, U, L, R)": "the base point: (D, U, L, R)",
    "UT3Frame .data: (D, U, L, R, shape, masks), stack C": "the base point: UT3Frame .data (D, U, L, R, shape, masks), stack C",
    "(V, H), elm_shapes K+C+(nUi,Ni), K+C+(rLi,nDi,rR(i+1))": "X, (V, H), elm_shapes K+C+(nUi,Ni), K+C+(rLi,nDi,rR(i+1))",
    "UT3Variations .data: (V, H) read, the frame's masks": "X, UT3Variations .data: (V, H) read, the frame's masks",
    "(B_dot, G_dot), the shapes of B, G with K+C": "X, (B_dot, G_dot), the shapes of B, G with K+C",
    "(B_dot, G_dot): supercores (d,)+K+C+(...), the point's sizes":
        "X, (B_dot, G_dot): supercores (d,)+K+C+(...), the point's sizes",
    "(tucker_dirs, tt_dirs): the cores' shapes with K": "X, (tucker_dirs, tt_dirs): the cores' shapes with K",
    "(tucker, tt) supercores (d,)+K+C+(...): the point's sizes":
        "X, (tucker, tt) supercores (d,)+K+C+(...): the point's sizes",
    # --- the inputs
    "len=d, elm_shape=F_in_i+W+(Ni,): input jets": "input jets: P_in_i⊗V_i, len=d, elm_shape=F_in_i+W+(Ni,)",
    "len=d, elm_shape=F_in_i+W+(Ni,): the input jets": "the input jets: P_in_i⊗V_i, len=d, elm_shape=F_in_i+W+(Ni,)",
    "input jets, len=d, elm_shape=F_in_i+W+(Ni,)": "input jets: P_in_i⊗V_i, len=d, elm_shape=F_in_i+W+(Ni,)",
    "input jets: len=d, F_in_i+W+(Ni,); or packed (d,)+F_in+W+(N,)":
        "input jets: P_in_i⊗V_i, len=d, F_in_i+W+(Ni,); or packed (d,)+F_in+W+(N,)",
    "int, shape=(d,)+W: the grid points": "index^d, shape=(d,)+W, int: the grid points",
    "int, shape=(d,)+W (a nested list is converted)": "index^d, shape=(d,)+W, int (a nested list is converted)",
    "int, shape=(d,)+W": "index^d, shape=(d,)+W, int",
    "probe vectors, len=d, elm_shape=W+(Ni,)": "probe vectors: V_i, len=d, elm_shape=W+(Ni,)",
    "apply vectors, len=d, elm_shape=W+(Ni,)": "apply vectors: V_i, len=d, elm_shape=W+(Ni,)",
    "shape=C+(N0,...,N(d-1))": "E, shape=C+(N0,...,N(d-1))",
    # --- the measurements (Y*)
    "z. len=d, elm_shape=F+W+C+(Ni,)": "z: P⊗V_i*, len=d, elm_shape=F+W+C+(Ni,)",
    "z. shape=F+W+C": "z: P, shape=F+W+C",
    "z. shape=W+C": "z: ℝ, shape=W+C",
    "z, MIRRORS ww: len=d, F+W+C+(Ni,); or packed": "z: P⊗V_i*, MIRRORS ww: len=d, F+W+C+(Ni,); or packed",
    "Jp. len=d, elm_shape=F+W+K+C+(Ni,)": "Jp: P⊗V_i*, len=d, elm_shape=F+W+K+C+(Ni,)",
    "Jp. shape=F+W+K+C": "Jp: P, shape=F+W+K+C",
    "Jp. shape=W+K+C": "Jp: ℝ, shape=W+K+C",
    "Jp, MIRRORS ww: len=d, F+W+K+C+(Ni,); or packed": "Jp: P⊗V_i*, MIRRORS ww: len=d, F+W+K+C+(Ni,); or packed",
    "J x_dot. len=d, elm_shape=F+W+K+C+(Ni,)": "J x_dot: P⊗V_i*, len=d, elm_shape=F+W+K+C+(Ni,)",
    "J x_dot. shape=F+W+K+C": "J x_dot: P, shape=F+W+K+C",
    "J x_dot. shape=W+K+C": "J x_dot: ℝ, shape=W+K+C",
    "J x_dot, MIRRORS ww: len=d, F+W+K+C+(Ni,); or packed":
        "J x_dot: P⊗V_i*, MIRRORS ww: len=d, F+W+K+C+(Ni,); or packed",
    # --- the transposes' outputs (X*, E*)
    "V blocks. len=d, elm_shape=[W]+[K]+C+(nUi,Ni)": "V blocks: X*, len=d, elm_shape=[W]+[K]+C+(nUi,Ni)",
    "H blocks. len=d, elm_shape=[W]+[K]+C+(rLi,nDi,rR(i+1))": "H blocks: X*, len=d, elm_shape=[W]+[K]+C+(rLi,nDi,rR(i+1))",
    "Tucker-core blocks. len=d, elm_shape=[W]+[K]+C+(nDi,Ni)": "Tucker-core blocks: X*, len=d, elm_shape=[W]+[K]+C+(nDi,Ni)",
    "TT-core blocks.     len=d, elm_shape=[W]+[K]+C+(ri,nDi,r(i+1))":
        "TT-core blocks:     X*, len=d, elm_shape=[W]+[K]+C+(ri,nDi,r(i+1))",
    "Tucker block. len=d, elm_shape=[W]+K+C+(nDi,Ni)": "Tucker block: X*, len=d, elm_shape=[W]+K+C+(nDi,Ni)",
    "TT block.     len=d, elm_shape=[W]+K+C+(ri,nDi,r(i+1))": "TT block:     X*, len=d, elm_shape=[W]+K+C+(ri,nDi,r(i+1))",
    "Tucker block, (d,)+[W]+[K]+C+(n,N)": "Tucker block: X*, (d,)+[W]+[K]+C+(n,N)",
    "TT block,     (d,)+[W]+[K]+C+(r,n,r)": "TT block:     X*, (d,)+[W]+[K]+C+(r,n,r)",
    "Tucker block, (d,)+[W]+K+C+(n,N)": "Tucker block: X*, (d,)+[W]+K+C+(n,N)",
    "TT block,     (d,)+[W]+K+C+(r,n,r)": "TT block:     X*, (d,)+[W]+K+C+(r,n,r)",
    "-> UT3Variations .data: (V_tilde, H_tilde, shape, masks); stack K_new + C":
        "-> X*: UT3Variations .data (V_tilde, H_tilde, shape, masks); stack K_new + C",
    "Tucker-core gradients, len=d, elm_shape=[W]+[K]+C+(ni,Ni)": "Tucker-core blocks: X*, len=d, elm_shape=[W]+[K]+C+(ni,Ni)",
    "TT-core gradients,     len=d, elm_shape=[W]+[K]+C+(ri,ni,r(i+1))":
        "TT-core blocks:     X*, len=d, elm_shape=[W]+[K]+C+(ri,ni,r(i+1))",
    "Tucker blocks, len=d, elm_shape=[W]+K+C+(ni,Ni)": "Tucker blocks: X*, len=d, elm_shape=[W]+K+C+(ni,Ni)",
    "TT blocks,     len=d, elm_shape=[W]+K+C+(ri,ni,r(i+1))": "TT blocks:     X*, len=d, elm_shape=[W]+K+C+(ri,ni,r(i+1))",
    "(tucker, tt) supercores, (d,)+[W]+[K]+C+(...)": "(tucker, tt) supercores: X*, (d,)+[W]+[K]+C+(...)",
    "(tucker, tt) blocks, (d,)+[W]+K+C+(...)": "(tucker, tt) blocks: X*, (d,)+[W]+K+C+(...)",
    "at frame; tangent stack [W]+[K], frame stack C":
        "X*, in a T3Tangent at frame (before the projection); tangent stack [W]+[K], frame stack C",
    "canonical (CP) factors. len=d, ith elm_shape=stack_shape+(R, Ni)":
        "canonical (CP) factors: E*, len=d, ith elm_shape=stack_shape+(R, Ni)",
    "CP factors, len=d, elm_shape=stack+(R,Ni)": "CP factors: E*, len=d, elm_shape=stack+(R,Ni)",
    # --- jets
    "Taylor coefficients: probe len=d, elm_shape=F+W+(Ni,); apply F+W; entries W":
        "Y* in Taylor coordinates (or Y in derivative ones: the contragredient): probe len=d, elm_shape=F+W+(Ni,); "
        "apply F+W; entries W",
    "derivatives, same shapes: slot k multiplied by k_1! ... k_m!. probe: a tuple":
        "Y* in derivative coordinates (or Y in Taylor ones), same shapes: slot k multiplied by k_1! ... k_m!. "
        "probe: a tuple",
    "derivatives: probe len=d, elm_shape=F+W+(Ni,); apply F+W; entries W":
        "Y* in derivative coordinates (or Y in Taylor ones: the contragredient): probe len=d, elm_shape=F+W+(Ni,); "
        "apply F+W; entries W",
    "Taylor coefficients, same shapes: slot k divided by k_1! ... k_m!. probe: a tuple":
        "Y* in Taylor coordinates (or Y in derivative ones), same shapes: slot k divided by k_1! ... k_m!. "
        "probe: a tuple",
    "derivatives, of the type of z: probe len=d, elm_shape=F+W+C+(Ni,); apply F+W+C; entries W+C":
        "Y*, in derivative coordinates: probe len=d, elm_shape=F+W+C+(Ni,); apply F+W+C; entries W+C",
    "validation derivatives, of the type of z, or None": "Y*, validation data in derivative coordinates, or None",
    "y in Taylor coefficients, same type and shapes": "y: Y*, in Taylor coordinates, same shapes",
    "the Taylor weight, of the type of z: k_1! ... k_m! per slot when weight is None":
        "omega, the Taylor weight, of the type of z: k_1! ... k_m! per slot when weight is None",
    "validation in Taylor coefficients, or None": "validation: Y*, in Taylor coordinates, or None",
    "len=d, elm_shape=W+(Ni,); or packed (d,)+W+(N,)": "V_i, len=d, elm_shape=W+(Ni,); or packed (d,)+W+(N,)",
    "m of them, each of the form and shapes of ww": "V_i, m of them, each of the form and shapes of ww",
    "len=d, elm_shape=(2,)*m+W+(Ni,); or packed (d,)+(2,)*m+W+(N,)":
        "P_in_i⊗V_i, len=d, elm_shape=(2,)*m+W+(Ni,); or packed (d,)+(2,)*m+W+(N,)",
    "len=d, elm_shape=W+(Ni,): e_index, in the default float dtype":
        "e_index: V_i, len=d, elm_shape=W+(Ni,), in the default float dtype",
    "ww (probe, apply: input jets) | index (entries)": "P_in_i⊗V_i (probe, apply: the input jets ww) | index^d (entries)",
    "d arrays F_in_i+W+(Ni,), each with len(order) jet axes of size >= 1 and":
        "P_in_i⊗V_i: d arrays F_in_i+W+(Ni,), each with len(order) jet axes",
    "one W for all; or index (d,)+W, integer": "of size >= 1 and one W for all; or index^d, (d,)+W, integer",
    "ww (input jets) | index, a good sample": "P_in_i⊗V_i (the input jets ww) | index^d, a good sample",
    # --- data preparation
    "the data: probe len=d, elm_shape=F+W+(Ni,); apply F+W; entries W":
        "the data: Y*, probe len=d, elm_shape=F+W+(Ni,); apply F+W; entries W",
    "the block shape: probe (d,)+F; apply F; entries ()": "ℝ per block, the block shape: probe (d,)+F; apply F; entries ()",
    "from data_block_norms: probe (d,)+F; apply F; entries ()":
        "ℝ per block, from data_block_norms: probe (d,)+F; apply F; entries ()",
    "ww, the input jets: len=d, elm_shape=F_in_i+W+(Ni,)": "the input jets ww: P_in_i⊗V_i, len=d, elm_shape=F_in_i+W+(Ni,)",
    "of the type of z: probe len=d, elm_shape=F+W+(Ni,); apply F+W": "Y*, the data: probe len=d, elm_shape=F+W+(Ni,); apply F+W",
    "the sample: the input jets rescaled, a tuple": "the sample: P_in_i⊗V_i, the input jets rescaled, a tuple",
    "y, same type and shapes, rescaled the same way": "y: Y*, same shapes, rescaled the same way",
    "the sample: every slot of every input jet times its mode's scales":
        "the sample: P_in_i⊗V_i, every slot of every input jet times its mode's scales",
    "y: for probe, mode i's data divided by scales[i]; for apply, as given":
        "y: Y*; for probe, mode i's data divided by scales[i]; for apply, as given",
    # --- the engine
    "(D, U, L, R), four tuples of d arrays; or (B, G)":
        "the base point (D, U, L, R), four tuples of d arrays; or the point (B, G): X",
    "ww: the input jets, len=d, elm_shape=F_in_i+W+(Ni,);": "ww: P_in_i⊗V_i, the input jets, len=d, elm_shape=F_in_i+W+(Ni,);",
    "entries: index, int, shape=(d,)+W": "entries: index^d, shape=(d,)+W, int",
    "the data, of the type of z, each axis 1 or z's size:": "the data: Y*, of the type of z, each axis 1 or z's size:",
    "(D_dot, L_dot), the shapes of D, L with K+C": "X, (D_dot, L_dot), the shapes of D, L with K+C",
    "len=d, elm_shape=C+(nUi,Ni)": "the frame's Tucker centers: len=d, elm_shape=C+(nUi,Ni)",
    "aligned with want; shapes in the module's table": "aligned with want; types and shapes in the module's table",
    # --- uniform packing
    "ww (input jets, len=d, elm_shape=F_in_i+W+(Ni,)) | index ((d,)+W, int)":
        "P_in_i⊗V_i (the input jets ww, len=d, elm_shape=F_in_i+W+(Ni,)) | index^d ((d,)+W, int)",
    "ww packed (d,)+F_in+W+(N,) | index unchanged": "P_in_i⊗V_i packed (d,)+F_in+W+(N,) | index^d unchanged",
    "of the type of z (the data, a weight, a residual): probe len=d,":
        "Y* or omega, of the type of z (the data, a weight, a residual):",
    "elm_shape=F+W+C+(Ni,); apply F+W+C; entries W+C; each axis 1 or z's size":
        "probe len=d, elm_shape=F+W+C+(Ni,); apply F+W+C; entries W+C; each axis 1 or z's size",
    # --- the oracle
    "(V, H): JtJp, Jp_norm_sq, Jp; on a point, the raw form": "X, (V, H): JtJp, Jp_norm_sq, Jp; raw on a point",
    "(V, H): JtJp, Jp_norm_sq, Jp": "X, (V, H): JtJp, Jp_norm_sq, Jp",
    "(D_dot, L_dot): Kp": "X, (D_dot, L_dot): Kp",
    "values, aligned with want; blocks before projection":
        "values, aligned with want: r, Jp Y*; g, JtJp, Kp X* (before the projection); f, Jp_norm_sq, "
        "f_resolution ℝ; r_block_sumsq, y_block_sumsq ℝ per block",
}


def _signature_lines(tree, qualname):
    parts = qualname.split('.')
    body = tree.body
    for i, name in enumerate(parts):
        kinds = (ast.ClassDef,) if i < len(parts) - 1 else (ast.FunctionDef,)
        node = next(n for n in body if isinstance(n, kinds) and n.name == name)
        body = node.body
    return node.lineno - 1, node.body[0].lineno - 1          # [start, end) in 0-based lines


def _comment_start(line):
    """The column of the trailing ``#``: the first ``#`` outside a string (signature lines hold no ``#`` in strings,
    except a ``'...'`` default, which this guards)."""
    quote = None
    for col, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = None
        elif ch in '\'"':
            quote = ch
        elif ch == '#':
            return col
    return -1


def _wrap(code, hashcol, text, continuation=False):
    """``code`` (padded to ``hashcol``) + ``# text`` (``#   text`` on a continuation line), wrapped at WIDTH with
    ``#   `` continuations in that column."""
    lines, current, fresh = [], code + ('#   ' if continuation else '# '), True
    for word in text.split(' '):
        trial = current + ('' if fresh else ' ') + word
        if len(trial) > WIDTH and not fresh:
            lines.append(current)
            current, fresh = ' ' * hashcol + '#   ' + word, False
        else:
            current, fresh = trial, False
    lines.append(current)
    return lines


def run(check):
    edits, unmapped, used = [], [], set()
    for rel, names in SCOPE.items():
        path = ROOT / rel
        lines = path.read_text().split('\n')
        tree = ast.parse('\n'.join(lines))
        spans = sorted((_signature_lines(tree, q) + (q,) for q in names), reverse=True)
        for start, end, qualname in spans:                       # bottom up: inserted lines keep earlier spans
            new_block, previous_col = [], None
            for line in lines[start:end]:
                col = _comment_start(line)
                if col < 0:
                    new_block.append(line)
                    continue
                code, rest = line[:col], line[col + 1:]
                text = rest.strip()
                if code.strip().startswith('c:') and previous_col is not None and abs(col - previous_col) == 1:
                    code, col = code.rstrip().ljust(previous_col), previous_col
                    edits.append((rel, qualname, 'the # column of c', 'realigned to %d' % col))
                continuation = not code.strip()                         # a '#   ...' continuation line
                key = text
                if key in COMMENTS:
                    used.add(key)
                    new_block.extend(_wrap(code, col, COMMENTS[key], continuation))
                    edits.append((rel, qualname, key, COMMENTS[key]))
                else:
                    new_block.append(code + '#' + rest)
                    if not continuation:
                        unmapped.append((rel, qualname, key))
                if not continuation:
                    previous_col = col
            lines[start:end] = new_block
        if not check:
            path.write_text('\n'.join(lines))
    for e in edits:
        print('EDIT\t%s\t%s\t%s\t-> %s' % e)
    for u in unmapped:
        print('LEFT\t%s\t%s\t%s' % u)
    for key in COMMENTS:
        if key not in used:
            print('UNUSED\t%s' % key)
    print('%d edits, %d comments left as they are' % (len(edits), len(unmapped)), file=sys.stderr)


if __name__ == '__main__':
    run('--check' in sys.argv)
