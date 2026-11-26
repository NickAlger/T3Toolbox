#!/usr/bin/env python3
"""K0's prose rewrites, after `delete_library.py`: each the before and after of `k0/prose_edits.md`, applied as an
exact replacement (each `before` must occur exactly once). Run once.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))

EDITS = {}

EDITS['t3toolbox/backend/probing.py'] = [
('''"""The ``probe`` sampling type AND the shared sampling machinery it exemplifies.

Probing contracts a T3 with vectors in all but one mode, giving ``d`` vectors (T4S §6). This
module holds the general machinery -- the ``xi/mu/nu/eta`` helper chain, ``assemble_z``, the tv
probe + frame sweeps + transposes, and the corewise transpose's derivative in the cores (the exact Hessian's
two-hole) -- which ``apply``/``entries`` specialize (probe ⊃ apply ⊃
entries; they import from here, never the reverse). Reference module for the signature style
(``docs/contributor/signature_style.md``).
"""''',
'''"""The ``probe`` sampling type: contract a T3 with vectors in all but one mode, giving ``d`` vectors (T4S §6).

The probe of a point and of a tangent, their transposes (ambient, corewise, tangent), the corewise forward
derivative and the corewise Hessian (the exact Hessian's sampling term), each but the ambient transpose one call
of the engine :py:func:`~t3toolbox.backend.sampling_variables.sampling_variables`; and the dense reference
:py:func:`dense_probe`, whose dense jets ``apply`` builds on (probe ⊃ apply ⊃ entries). Reference module
for the signature style (``docs/contributor/signature_style.md``).
"""'''),
("""__all__ = [
    # Probe a Tucker tensor train
    't3_probe',
    # Probe a tangent vector
    'tv_probe',
    # Apply -- frame-sweep reuse split (precompute the LEAN (xi,mu) frame edge vars once; inject into the
    # bare J / Jᵀ; adjoint-state transpose recomputes the right context as sigma_hat; for fitting.py)
    # Entries -- frame-sweep reuse split (the fiber-sliced seed; one-hot adjoint-state transpose; for fitting.py)
    # Probe -- frame-sweep reuse split (the FULL (xi,mu,nu,eta) sweep; for fitting.py)
    # Corewise (non-manifold) transpose -- the tangent transpose with the frame's cores in place of the frames
    # Transpose of map from tangent vector to probes
    'tv_probe_transpose',
    # Ambient / corewise probe transposes (the plain-probe analogs of the apply/entries transposes)
    't3_probe_ambient_transpose',
    't3_probe_corewise_tangent',
    't3_probe_corewise_transpose',
    # Corewise two-hole: the probe transpose's derivative in the cores (the exact Hessian's sampling term)
    't3_probe_corewise_hessian',
    # Probe a dense tensor
    'dense_probe',
]""",
"""__all__ = [
    # Probe a Tucker tensor train
    't3_probe',
    # Probe a tangent vector, and its transpose
    'tv_probe',
    'tv_probe_transpose',
    # Ambient / corewise probe transposes, the corewise tangent and Hessian
    't3_probe_ambient_transpose',
    't3_probe_corewise_tangent',
    't3_probe_corewise_transpose',
    't3_probe_corewise_hessian',
    # Probe a dense tensor
    'dense_probe',
]"""),
('''###############################################################
###########    Transpose of tangent to probes map    ##########
###############################################################''',
'''###############################################################
######    Transposes of the probe; the corewise tangent    ######
###############################################################'''),
]

EDITS['t3toolbox/backend/apply.py'] = [
('''Holds the t3 + tv ops, the ambient/tangent/corewise transposes, the corewise transpose's derivative in the
cores (the exact Hessian's sampling term), and the frame sweeps for applies.
Specializes the general probing machinery (containment probe ⊃ apply ⊃ entries; this module
imports from ``probing``, never the reverse). Costs and the role in Riemannian least-squares:
``docs/entries_apply_probe.md``.''',
'''Holds the t3 + tv ops, the ambient/tangent/corewise transposes, the corewise forward derivative and the
corewise Hessian (the exact Hessian's sampling term), each but the ambient transpose one call of the engine,
and the dense reference :py:func:`dense_apply` (containment probe ⊃ apply ⊃ entries; this module imports from
``probing``, never the reverse). Costs and the role in Riemannian least-squares:
``docs/entries_apply_probe.md``.'''),
]

EDITS['t3toolbox/backend/entries.py'] = [
('''Holds the t3 + tv ops, the ambient/tangent/corewise transposes, the corewise transpose's derivative in the
cores (the exact Hessian's sampling term), and the frame sweeps for entry sampling. The most-special case of
probe ⊃ apply ⊃ entries; imports the general machinery from ``probing``, never the reverse. See
``docs/entries_apply_probe.md``.''',
'''Holds the t3 + tv ops, the ambient/tangent/corewise transposes, the corewise forward derivative and the
corewise Hessian (the exact Hessian's sampling term), each but the ambient transpose one call of the engine,
and the dense reference :py:func:`dense_entries`. The most-special case of probe ⊃ apply ⊃ entries; imports
from ``apply`` (its ambient transpose), never the reverse. See ``docs/entries_apply_probe.md``.'''),
('''    # (a negative index wraps), exactly like the forward t3_entries and the tangent/corewise transposes
    # (_onehot_vectors); the former `arange(N) == index` scatter matched nothing for a negative index and
    # silently returned zero factors.''',
'''    # (a negative index wraps), exactly like the forward t3_entries and the tangent/corewise transposes; the
    # former `arange(N) == index` scatter matched nothing for a negative index and silently returned zero factors.'''),
]

EDITS['t3toolbox/backend/ut3_sampling.py'] = [
('''"""Uniform plain-layer sampling wrappers: probe/apply/entries on ut3 data (+ transposes, jets).

Grouped by OBJECT type, deliberately asymmetric with the ragged by-sampling-type modules: each
function here is a thin wrapper -- mask once, pack, delegate to the shared polymorphic ragged
machinery (the corewise transposes and the two-holes then re-mask their output) -- so it
lives with the object it wraps. The algorithm story is in
``probing``/``apply``/``entries``/``sampling_derivatives`` (``docs/naming_conventions.md``).

The twins of the ragged sampling functions (``ut3_probe``, ``ut3_apply``, ``ut3_entries`` and their
``*_corewise_tangent``, ``*_corewise_transpose``, ``*_corewise_hessian``; graded messages P12) take the ragged
functions' signatures, with ``x`` the uniform point's ``.data``, and are one call of the engine
(:py:func:`~t3toolbox.backend.sampling_variables.sampling_variables`), which packs a ragged sample, masks every
operand on entry and every output on exit. A probe output mirrors the sample's form: ragged in, ragged out; packed
in, packed out. The jet wrappers (``*_derivatives*``) still share ``sampling_derivatives``.
"""''',
'''"""Uniform plain-layer sampling: probe/apply/entries on ut3 data, their corewise derivative, transpose and
Hessian, and the full sum.

Grouped by OBJECT type, deliberately asymmetric with the ragged by-sampling-type modules
(``docs/naming_conventions.md``). The twins of the ragged sampling functions (``ut3_probe``, ``ut3_apply``,
``ut3_entries`` and their ``*_corewise_tangent``, ``*_corewise_transpose``, ``*_corewise_hessian``) take the
ragged functions' signatures, with ``x`` the uniform point's ``.data``, and are one call of the engine
(:py:func:`~t3toolbox.backend.sampling_variables.sampling_variables`), which packs a ragged sample, masks every
operand on entry and every output on exit. A probe output mirrors the sample's form: ragged in, ragged out;
packed in, packed out.
"""'''),
('''# ----------------------------------------------------------------- corewise (non-manifold) transposes
# Gradient of a sampling op w.r.t. the frame's own supercores (the Section 6.3 (U,L,R)->G substitution, via
# the now-polymorphic probing.*_corewise_transpose -- the tangent transpose at the frame (B, G, G, G)). For
# a uniform core-wise optimizer (Adam, L-BFGS). Mask-once + pack at the boundary; return the RAW gradient
# supercores (dB, dG) -- the uniform mirror of the ragged TuckerTensorTrain.*_corewise_transpose raw-tuple
# return -- RE-MASKED on exit with the same point masks that masked the input. The raw padding is not clean
# (the boundary-bond squash is a sum over the whole bond, so its gradient broadcasts into the padded
# boundary-bond slots too); the re-mask zeroes it, so a consumer gets exact zeros there. The equivalence
# contract still declares the padding don't-care (every consumer masks on entry -- ut3_squash_tails
# included, since 2026-08-22), and the masks, not the values, govern rank.''',
'''# ----------------------------------------------------------------- corewise (non-manifold) tangents and transposes
# The forward derivative of a sampling op along raw directions in the point's own supercores, and its gradient
# w.r.t. them (for a uniform core-wise optimizer: Adam, L-BFGS). The transposes return the RAW gradient
# supercores (dB, dG) -- the uniform mirror of the ragged TuckerTensorTrain.*_corewise_transpose raw-tuple
# return -- with their padding exactly zero: the engine masks every input on entry and every output on exit.
# The equivalence contract still declares the padding don't-care (every consumer masks on entry --
# ut3_squash_tails included, since 2026-08-22), and the masks, not the values, govern rank.'''),
('''# ----------------------------------------------------------- corewise two-holes: the plain kinds' Hessians
# The same wrapper, for the plain probe / apply / entries two-holes (the derivatives of ut3_*_corewise_transpose
# along raw core directions): theta masked with its masks, theta_hat with the SAME masks, the residual and the
# vectors packed (entries: the integer index is not packed -- the one-hots are built packed inside the kernel),
# the polymorphic kernel run on the supercores, the output re-masked.''',
'''# ----------------------------------------------------------- corewise two-holes: the plain kinds' Hessians
# The derivatives of ut3_*_corewise_transpose along raw core directions, one engine call each (Kp): theta and
# theta_hat masked with theta's masks on entry, the output masked on exit.'''),
]

EDITS['t3toolbox/backend/utv_sampling.py'] = [
('''The uniform mirror of the ragged ``T3Tangent`` sampling frontend. **Since graded messages P12a** the plain
(jet-free) functions ``utv_{probe,apply,entries}`` and their transposes are one call of the engine
(:py:func:`~t3toolbox.backend.sampling_variables.sampling_variables`) on the frame's six elements: the engine masks
the frame and **the variations with the frame's masks** (the variations' own masks are not read; a uniform tangent's
variation masks are the frame's gauge masks), packs the sample, and masks every output. They take the ragged
``tv_*`` signatures (P12b), and ``utv_probe`` mirrors the sample's form. What follows describes the jet functions and the ``*_from_sweep`` seam, which still share the ragged
machinery. The boundary work is the same as the plain layer's
``ut3_sampling`` (the precedent), lifted to the frame-variations pair:

1. **mask-once** the frame (4) and variation (2) supercores -- so the padded rank slots are zero and the
   core algorithm runs mask-free (the padded bonds contract to zero);
2. **pack** the probe vectors into one supercore-shaped tensor (``apply``/``probe``; ``entries`` slices
   fibers, so its integer ``index`` needs no packing);
3. call the polymorphic backend (``is_uniform`` is inferred from the masked supercores being bare ndarrays);
4. **unpack** the ``probe`` output back to ragged-width vectors (``apply``/``entries`` return a scalar --
   nothing to unpack).

These are the bare Jacobian ``𝒥`` (no gauge projector ``Π``); the transpose ``𝒥ᵀ`` is the last section of this module.
The transposes **re-mask their output** with the variation masks they attach, so the padded slots come back as
exact zeros (one multiply by a host mask; the real block is unchanged bit for bit). The ``*_transpose_from_sweep``
seam halves stay bare: their callers, the uniform fitting kinds, re-mask.''',
'''The uniform mirror of the ragged ``T3Tangent`` sampling frontend. The functions ``utv_{probe,apply,entries}`` and
their transposes take the ragged ``tv_*`` signatures and are one call of the engine
(:py:func:`~t3toolbox.backend.sampling_variables.sampling_variables`) on the frame's six elements: the engine masks
the frame and **the variations with the frame's masks** (the variations' own masks are not read; a uniform
tangent's variation masks are the frame's gauge masks), packs the sample, and masks every output; ``utv_probe``
mirrors the sample's form.

These are the bare Jacobian ``𝒥`` (no gauge projector ``Π``) and its transpose ``𝒥ᵀ``. The transposes attach the
result tangent's variation masks (the frame's gauge masks over the new tangent stack) and **re-mask their
output** with them, so the padded slots come back as exact zeros (one multiply by a host mask; the real block is
unchanged bit for bit).'''),
('# ----------------------------------------------------------------- the transpose 𝒥ᵀ (probe)\n',
 '# ----------------------------------------------------------------- the transpose 𝒥ᵀ (probe, apply, entries)\n'),
('''``K_new`` -- the leading stack of the transpose output minus the frame stack ``C``. ``sum_over_probes``''',
 '''``K_new`` -- the leading stack of the transpose output minus the frame stack ``C``. ``sum_over_samples``'''),
]

EDITS['t3toolbox/backend/fitting.py'] = [
("""'''Sampling-kind primitives for the geometry-generic Gauss-Newton fitting model.

The Gauss-Newton model ``m(p) = c + gᵀp + ½ pᵀ(JᵀJ)p`` of a least-squares objective factors over two
*independent* choices, and the fitting layer is generic in both:

  * the sampling **kind** (apply / entries / probe) -- the bare single-sample Jacobian ``𝒥`` / its
    transpose ``𝒥ᵀ`` (from :py:mod:`t3toolbox.backend.probing`), plus the kind-specific ``‖·‖²``
    reduction over the probe stack; and
  * the **geometry** (manifold / corewise) -- the gauge projection ``Π`` (``geometry.project``), with
    the Riemannian forward ``J = 𝒥∘Π`` and gradient ``Jᵀr = Π∘𝒥ᵀr``.

This module supplies the **kind** half: a :py:class:`SamplingKind` descriptor bundling the bare probing
primitives + the reduction per kind (:py:data:`APPLY` / :py:data:`ENTRIES` / :py:data:`PROBE`). The
**geometry** half lives in the geometry (:py:data:`t3toolbox.manifold.MANIFOLD` /
:py:data:`~t3toolbox.manifold.COREWISE`); the frontend :py:class:`t3toolbox.fitting.LocalModel`
composes the two. The §6.3 corewise substitution ``(D,U,L,R) -> (B,G,G,G)`` is now nothing but
``CorewiseGeometry.frame`` -- there is no separate corewise backend.

``probe`` is vector-valued (one free mode per probe), so its residual / forward output is a sequence of
``d`` arrays and its reduction sums the free mode too; apply / entries are the scalar all-modes special
case. The objective constant ``c = ½‖r‖²`` is just ``½ · sumsq(r)`` (the same reduction as the model's
quadratic term ``½‖𝒥Πp‖²``), so the kind needs no separate objective function.
'''""",
"""'''The least-squares oracle and its batch: what the optimizers ask of the data.

A fit is ``f = 1/2 <M r, r>`` with ``r = z(x) - y`` the residual of a sampling kind (probe, apply or entries) at
a jet shape ``order``, and ``M`` the batch's data metric. :py:class:`LeastSquaresOracle` (the kind and the order,
a frozen description with no arrays) answers what an optimizer asks at a frame or a corewise point (``f``,
``g``, ``Jp``, ``JtJp``, ``Kp``, the block sums, ...) through one call of the engine
:py:func:`~t3toolbox.backend.sampling_variables.sampling_variables`, and checks, draws from and counts a batch.
:py:class:`LeastSquaresBatch` holds the data, arrays only: the sample, ``y``, the weight, the minibatch scale and
the drawn indices. What the oracle returns is bare, before any projection: ``Jp = 𝒥p``, ``g = 𝒥ᵀ(M r)`` and
``JtJp = 𝒥ᵀ M 𝒥p``, with ``𝒥`` the Jacobian of ``z`` at the frame and ``M`` the batch's data metric (its minibatch
scale inside). The geometry (manifold or corewise, :py:mod:`t3toolbox.backend.geometry`) is the other half of a
fit: it supplies the projection ``Π`` (``geometry.project``: the gauge projection on the manifold; on the
corewise geometry the identity, or the per-group mean of tied factors), so that the forward is ``J = 𝒥∘Π`` and
the gradient ``Jᵀr = Π∘𝒥ᵀr``; :py:mod:`t3toolbox.backend.optimizers` composes the two.
'''"""),
("""# --------------------------------------------------------------------------------------------------
# Default-draw layout helpers (the minimal layout the kind exposes so the optimizer can build its
# DEFAULT flat minibatch draw -- a random subset across the whole probe stack W. A USER-supplied draw
# bypasses these entirely.) `n_measurements` = the flat W size; `take` gathers a flat W subset out of
# (sample, data) -- both kind-specific, since W sits in a different place per kind (and the derivative
# data has a leading order axis).
# --------------------------------------------------------------------------------------------------""",
"""# --------------------------------------------------------------------------------------------------
# The flat draw over the sample stack W: flatten W's axes into one and gather the drawn positions. The
# oracle's `take` uses it for the sample (and, through `_take_W`, for y and the weight), wherever W sits in
# the array. A USER-supplied draw bypasses it entirely.
# --------------------------------------------------------------------------------------------------"""),
("""# The fitting layer's new extension point (graded messages, Part 1): an oracle answers a query at a frame through
# the engine, its batch holds the data. Beside the old kinds above, which the optimizers use until P16.""",
"""# The fitting layer's extension point (graded messages, Part 1): an oracle answers a query at a frame through the
# engine, its batch holds the data."""),
]

EDITS['t3toolbox/backend/uniform_fitting.py'] = [
('''"""Uniform-layer fitting: packing a least-squares batch for the uniform layout, the minimal-rank reduction of a
uniform start, and the old packed ``SamplingKind`` classes.''',
'''"""Uniform-layer fitting: packing a least-squares batch for the uniform layout, and the minimal-rank reduction of
a uniform start.'''),
('''uniform start to minimal ranks, which the uniform manifold geometry requires.

The uniform twins of :py:mod:`t3toolbox.backend.fitting`'s old kinds -- subclasses that override the
layer-specific operations and, for the probe kinds, where ``omega``'s axes sit in the PACKED output -- are no longer
called by the optimizers (graded messages, P16) and go with the old kinds (P20).
Each carries the fixed rank it was built at (``shape`` + the plain-UT3 ``masks``) as FIELDS, so a
rebuilt kind of the same rank is the same jax cache key
(:py:mod:`t3toolbox.backend.geometry` follows the same rule; the reasoning is
``docs/contributor/parameters_not_closures.md``).

**Protocol note (ruled: keep and document):** the kinds' ``forward`` / ``transpose`` keep
the ragged Kind protocol's ``(v, sample, frame_data, sweep)`` signature even though the packed
``sweep`` already carries the sample and the frame -- the extra arguments are deliberately unread.
Protocol uniformity is the point: generic code (the shared ``LocalModel``) calls every kind the same
way, ragged or uniform.
"""''',
'''uniform start to minimal ranks, which the uniform manifold geometry requires.
"""'''),
]

EDITS['t3toolbox/backend/common.py'] = [
('''    an aliased writeable caller array mutated in place would leave a stale cached jit key
    (the ``_weight_matrix`` precedent)."""''',
'''    an aliased writeable caller array mutated in place would leave a stale cached jit key."""'''),
]

EDITS['docs/api_reference.rst'] = [
('''- The six model factories (in :py:mod:`~t3toolbox.fitting`):
  :py:func:`~t3toolbox.fitting.entries_model`, :py:func:`~t3toolbox.fitting.apply_model`,
  :py:func:`~t3toolbox.fitting.probe_model`,
  :py:func:`~t3toolbox.fitting.entries_derivatives_model`,
  :py:func:`~t3toolbox.fitting.apply_derivatives_model`,
  :py:func:`~t3toolbox.fitting.probe_derivatives_model`.
''', ''),
('''- Module-level (not at the root): :py:func:`~t3toolbox.optimizers.least_squares_problem` -- the
  constructor the four optimizers use, public for scripts that drive the backend directly: it builds the
  backend :py:class:`~t3toolbox.backend.optimizers.Problem` from the frontend vocabulary and returns a
  :py:class:`~t3toolbox.optimizers.LeastSquaresSetup` (``.problem``, ``.x0_cores``, ``.rewrap``).
''', ''),
('''  :py:mod:`~t3toolbox.backend.entries`, :py:mod:`~t3toolbox.backend.sampling_derivatives`, and the''',
 '''  :py:mod:`~t3toolbox.backend.entries`, and the'''),
]


def main():
    for path, edits in EDITS.items():
        full = os.path.join(REPO, path)
        text = open(full, encoding='utf-8').read()
        for before, after in edits:
            n = text.count(before)
            if n != 1:
                sys.exit('%s: a before-text occurs %d times: %r' % (path, n, before[:80]))
            text = text.replace(before, after)
        with open(full, 'w', encoding='utf-8') as f:
            f.write(text)
        print('%s: %d edits' % (path, len(edits)))


if __name__ == '__main__':
    main()
