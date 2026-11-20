# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""The sampling engine: every sampling variable of a Tucker tensor train, from one declared graph.

Sampling a Tucker tensor train at input jets makes a family of intermediate arrays, the *sampling variables*.
Their definitions, one dense function each, and their names are in :py:mod:`t3toolbox.backend.sampling_definitions`.
:py:func:`sampling_variables` computes the ones you ask for by their recurrences, each exactly once, and only those
that what you ask for reads.

**The declared graph** of a kind is plain data: a tuple of rows ``Rule(name, step, reads, needs, carries_K,
start)``.

- ``name``: the variable the row makes; a name with a leading underscore is a private node, which cannot be wanted
  or held. A name may have several rows, its *routes*: the first row whose ``needs``
  are all present is taken. The rows of a name run from the most specific to the least, so the earlier applicable
  row wins. A held variable never changes the route: it is a leaf of the closure, whatever route is taken.
- ``step``: ``(driver, step_name)``. ``step_name`` names a function of
  :py:mod:`t3toolbox.backend.sampling_steps`, looked up when it runs. The driver says how it runs over the sites:
  ``'site'``, a map over the sites ``i = 0 .. d-1``; ``'left'``, a sweep making the bonds ``0 .. d`` from the start
  value at bond ``0``; ``'right'``, the sweep making them from the start value at bond ``d``; ``'whole'``, once,
  on whole variables.
- ``reads``: the step's arrays, in order, each a name of one of three disjoint namespaces: a variable some row
  makes; a store of the frame (``D``, ``U``, ``L``, ``R``); an argument of the call (``w``, the sample; ``y``;
  ``weight``; ``Vc``). A sweep's carry is its first argument and is not listed; its first read is the TT store it
  sweeps through, and that store's end bond sizes its start value (``R`` for ``nu``, ``L`` for ``nu_L``: with tails
  they differ).
- ``needs``: the facts of the call the row requires: the arguments given (``y``, ``Vc``, ``variations``,
  ``core_directions``) and the call type (``_manifold``, ``_corewise``).
- ``carries_K``: whether the row runs with the call's stack of directions ``K``, or with ``K = 0``.
- ``start``: for a sweep, its start value: ``'ones'`` (the plain grade: the vector of ones that closes the end bond)
  or ``'zeros'`` (a tilde, dot or tilde-dot sweep); ``None`` for the other drivers.

At a site op, a bond variable is read at the bond its letter says: the left of bond ``i`` (the ``mu`` family) and
the right of bond ``i + 1`` (the ``nu`` family), the table ``_BOND_OFFSET``.

The graph of the probe kind on a manifold frame (the graph of a corewise point is derived from it, below):

| Variable | Driver | Step | Reads | Needs | K | Start |
| --- | --- | --- | --- | --- | --- | --- |
| `xi` | site | `xi_step` | `D`, `w` |  | no |  |
| `mu` | left | `mu_step` | `L`, `xi` |  | no | ones |
| `nu` | right | `nu_step` | `R`, `xi` |  | no | ones |
| `nu_L` | right | `nu_step` | `L`, `xi` |  | no | ones |
| `eta` | site | `eta_step` | `mu`, `U`, `nu` |  | no |  |
| `eta_D` | site | `eta_step` | `mu`, `L`, `nu_L` |  | no |  |
| `z` | site | `z_step` | `eta`, `Vc` | `Vc` | no |  |
| `z` | site | `z_step` | `eta_D`, `D` |  | no |  |
| `r` | whole | `r_step` | `z`, `y` | `y` | no |  |
| `f` | whole | `f_step` | `r`, `weight` |  | no |  |
| `r_block_sumsq` | whole | `r_block_sumsq_step` | `r` |  | no |  |
| `w_tilde` | whole | `w_tilde_step` | `r`, `weight` |  | no |  |
| `xi_tilde` | site | `xi_tilde_step` | `D`, `w_tilde` |  | no |  |
| `mu_tilde` | left | `mu_tilde_step` | `L`, `mu`, `xi`, `xi_tilde` |  | no | zeros |
| `nu_tilde` | right | `nu_tilde_step` | `R`, `nu`, `xi`, `xi_tilde` |  | no | zeros |
| `eta_tilde` | site | `eta_tilde_step` | `mu_tilde`, `mu`, `U`, `nu_tilde`, `nu` |  | no |  |
| `_eta_tilde_top` | site | `eta_tilde_top_step` | `mu_tilde`, `mu`, `U`, `nu_tilde`, `nu`, `w` |  | no |  |
| `nu_L_tilde` | right | `nu_tilde_step` | `L`, `nu_L`, `xi`, `xi_tilde` |  | no | zeros |
| `eta_D_tilde` | site | `eta_tilde_step` | `mu_tilde`, `mu`, `L`, `nu_L_tilde`, `nu_L` |  | no |  |
| `_eta_D_tilde_top` | site | `eta_tilde_top_step` | `mu_tilde`, `mu`, `L`, `nu_L_tilde`, `nu_L`, `w` |  | no |  |
| `V_tilde` | site | `V_tilde_step` | `_eta_D_tilde_top`, `eta_D`, `w`, `w_tilde` | `_corewise` | no |  |
| `V_tilde` | site | `V_tilde_step` | `_eta_tilde_top`, `eta`, `w`, `w_tilde` |  | no |  |
| `H_tilde` | site | `H_tilde_step` | `mu_tilde`, `mu`, `xi_tilde`, `xi`, `nu_L_tilde`, `nu_L` | `_corewise` | no |  |
| `H_tilde` | site | `H_tilde_step` | `mu_tilde`, `mu`, `xi_tilde`, `xi`, `nu_tilde`, `nu` |  | no |  |
| `g` | whole | `g_step` | `V_tilde`, `H_tilde` |  | no |  |
| `xi_U_dot` | site | `xi_dot_step` | `V`, `w` | `variations` | yes |  |
| `mu_R_dot` | left | `mu_dot_step` | `R`, `mu`, `xi`, `xi_U_dot`, `U`, `H` | `variations` | yes | zeros |
| `_frame.nu_L_dot` | right | `nu_dot_step` | `L`, `nu`, `xi`, `xi_U_dot`, `U`, `H` | `variations` | yes | zeros |
| `_frame.eta_D_dot` | site | `eta_dot_step` | `mu_R_dot`, `R`, `mu`, `H`, `L`, `_frame.nu_L_dot`, `nu` | `variations` | yes |  |
| `xi_dot` | site | `xi_dot_step` | `D_dot`, `w` | `core_directions` | yes |  |
| `mu_dot` | left | `mu_dot_step` | `L`, `mu`, `xi`, `xi_dot`, `L`, `L_dot` | `core_directions` | yes | zeros |
| `_raw.nu_L_dot` | right | `nu_dot_step` | `L`, `nu_L`, `xi`, `xi_dot`, `L`, `L_dot` | `core_directions` | yes | zeros |
| `_raw.eta_D_dot` | site | `eta_dot_step` | `mu_dot`, `L`, `mu`, `L_dot`, `L`, `_raw.nu_L_dot`, `nu_L` | `core_directions` | yes |  |
| `nu_L_dot` | whole | `identity_step` | `_raw.nu_L_dot` | `core_directions` | yes |  |
| `nu_L_dot` | whole | `identity_step` | `_frame.nu_L_dot` | `variations` | yes |  |
| `eta_D_dot` | whole | `identity_step` | `_raw.eta_D_dot` | `core_directions` | yes |  |
| `eta_D_dot` | whole | `identity_step` | `_frame.eta_D_dot` | `variations` | yes |  |
| `Jp` | site | `Jp_step` | `_frame.eta_D_dot`, `D`, `eta`, `V` | `variations` | yes |  |
| `Jp` | site | `Jp_step` | `_raw.eta_D_dot`, `D`, `eta_D`, `D_dot` | `core_directions` | yes |  |
| `Jp_norm_sq` | whole | `Jp_norm_sq_step` | `Jp`, `weight` |  | yes |  |
| `xi_tilde_dot` | site | `xi_tilde_dot_step` | `D_dot`, `w_tilde` | `core_directions` | yes |  |
| `mu_tilde_dot` | left | `mu_tilde_dot_step` | `L`, `mu_dot`, `mu_tilde`, `mu`, `xi_tilde_dot`, `xi_dot`, `xi_tilde`, `xi`, `L_dot` | `core_directions` | yes | zeros |
| `nu_L_tilde_dot` | right | `nu_tilde_dot_step` | `L`, `_raw.nu_L_dot`, `nu_L_tilde`, `nu_L`, `xi_tilde_dot`, `xi_dot`, `xi_tilde`, `xi`, `L_dot` | `core_directions` | yes | zeros |
| `eta_D_tilde_dot` | site | `eta_tilde_dot_step` | `mu_tilde_dot`, `mu_dot`, `mu_tilde`, `mu`, `L`, `nu_L_tilde_dot`, `_raw.nu_L_dot`, `nu_L_tilde`, `nu_L`, `L_dot` | `core_directions` | yes |  |
| `_eta_D_tilde_dot_top` | site | `eta_tilde_dot_top_step` | `mu_tilde_dot`, `mu_dot`, `mu_tilde`, `mu`, `L`, `nu_L_tilde_dot`, `_raw.nu_L_dot`, `nu_L_tilde`, `nu_L`, `L_dot`, `w` | `core_directions` | yes |  |
| `V_tilde_dot` | site | `V_tilde_dot_step` | `_eta_D_tilde_dot_top`, `_raw.eta_D_dot`, `w`, `w_tilde` | `core_directions` | yes |  |
| `H_tilde_dot` | site | `H_tilde_dot_step` | `mu_tilde_dot`, `mu_dot`, `mu_tilde`, `mu`, `xi_tilde_dot`, `xi_dot`, `xi_tilde`, `xi`, `nu_L_tilde_dot`, `_raw.nu_L_dot`, `nu_L_tilde`, `nu_L` | `core_directions` | yes |  |
| `Kp` | whole | `g_step` | `V_tilde_dot`, `H_tilde_dot` | `core_directions` | yes |  |
| `_Jp.w_tilde` | whole | `w_tilde_step` | `Jp`, `weight` |  | yes |  |
| `_Jp.xi_tilde` | site | `xi_tilde_step` | `D`, `_Jp.w_tilde` |  | yes |  |
| `_Jp.mu_tilde` | left | `mu_tilde_step` | `L`, `mu`, `xi`, `_Jp.xi_tilde` |  | yes | zeros |
| `_Jp.nu_tilde` | right | `nu_tilde_step` | `R`, `nu`, `xi`, `_Jp.xi_tilde` |  | yes | zeros |
| `_Jp.eta_tilde_top` | site | `eta_tilde_top_step` | `_Jp.mu_tilde`, `mu`, `U`, `_Jp.nu_tilde`, `nu`, `w` |  | yes |  |
| `_Jp.V_tilde` | site | `V_tilde_step` | `_Jp.eta_tilde_top`, `eta`, `w`, `_Jp.w_tilde` |  | yes |  |
| `_Jp.H_tilde` | site | `H_tilde_step` | `_Jp.mu_tilde`, `mu`, `_Jp.xi_tilde`, `xi`, `_Jp.nu_tilde`, `nu` |  | yes |  |
| `JtJp` | whole | `g_step` | `_Jp.V_tilde`, `_Jp.H_tilde` | `variations` | yes |  |

**The dot grades and the direction stack.** A direction comes in two forms: ``variations = (V, H)``, the frame's form,
and ``core_directions = (D_dot, L_dot)``, the raw form at the base point ``(D, L)``. Their leading axes before the
frame's ``C`` are the one stack of directions ``K`` of the call, read off ``V[0]`` (or ``D_dot[0]``): every ``V[i]``,
``H[i]``, ``D_dot[i]`` and ``L_dot[i]`` must share that ``K + C`` (a structural check at entry; the curvature pairs
element ``k`` of a direction with element ``k`` of its lift). A row with ``K = yes`` runs with the call's ``K``, and its
variable carries ``K`` after ``W``; the others run with ``K = 0``. The two names with both dot forms, ``nu_L_dot`` and
``eta_D_dot``, are route rows over private nodes: the raw ``_raw.*`` when ``core_directions`` is given, else the
frame's ``_frame.*`` (O-12). ``Jp`` reads the frame nodes, ``Kp`` the raw ones; so a call with both directions computes
each family once, ``Jp`` along ``(V, H)`` and the public names along ``(D_dot, L_dot)``, and a held ``nu_L_dot`` is a
leaf for the public name only. ``Jp`` has a raw route too, the direction's output at the base point, taken without
``variations``. ``JtJp`` needs ``variations`` itself and is not reached from ``core_directions`` alone on a manifold
call: its blocks are the frame's (``V_tilde`` in the ``U`` basis, ``H`` between ``L`` and ``R`` cores), and seeding them
from a raw ``Jp`` would give a mixed product with no use.

**The adjoint seeded by** ``Jp`` (``JtJp``) is the private twin of each row on the residual's path from ``w_tilde`` to
``V_tilde``, ``H_tilde`` and the short node, named ``'_Jp.' + name.lstrip('_')``: the same step and start, the call's
``K``, the seed reading ``Jp`` for ``r``. ``JtJp`` is the twin of ``g``. The full ``eta_tilde`` has no twin.

**The corewise point** ``(B, G)``: one basis on each edge. It binds the stores ``D = B`` and ``L = G``, and its graph
is derived from the manifold one: every row that reads ``U``, ``R``, ``V``, ``H`` or ``Vc``, or a ``_frame.*`` node, is
dropped, with every row that reads a dropped name; the rest are renamed, in their names, reads and needs, by one table
(the base point's basis is the only one, so a name loses its basis; the direction is ``core_directions``); then the
twins are derived. The rows of a name that need ``_corewise`` (the gradient in the base point's basis) are taken only
there. So each sweep runs once, and a name with a basis is unknown there (O-13). On a corewise point a direction is
passed as ``core_directions``; ``variations`` raises.

| On a manifold frame | On a corewise point |
| --- | --- |
| `nu_L` | `nu` |
| `eta_D` | `eta` |
| `nu_L_tilde` | `nu_tilde` |
| `eta_D_tilde` | `eta_tilde` |
| `nu_L_dot` | `nu_dot` |
| `eta_D_dot` | `eta_dot` |
| `nu_L_tilde_dot` | `nu_tilde_dot` |
| `eta_D_tilde_dot` | `eta_tilde_dot` |
| `_raw.nu_L_dot` | `_raw.nu_dot` |
| `_raw.eta_D_dot` | `_raw.eta_dot` |
| `_eta_D_tilde_top` | `_eta_tilde_top` |
| `_eta_D_tilde_dot_top` | `_eta_tilde_dot_top` |
| `variations` | `core_directions` |

**The short form of** ``eta_tilde``. ``V_tilde`` reads only the slots of ``eta_tilde`` that meet the input at the
corner: the top ``a = min(J_in_i, F)`` slots on every jet axis (two for a line). The private node ``_eta_tilde_top``
holds those slots, ``eta_tilde[i][c - a + 1 : c + 1]`` (``c`` the corner), each made by readouts at a lowered corner,
and ``V_tilde`` always reads it. The full ``eta_tilde`` has its own row, run only when it is wanted, so what ``g`` is
never depends on whether ``eta_tilde`` is wanted. ``eta_D_tilde`` (read by the corewise gradient) and
``eta_D_tilde_dot`` (read by ``V_tilde_dot``) have short nodes in the same way.

**The shapes** (ragged layout): a variable at index ``i`` is
``J + W + C + (size,)``, with ``J`` its declared jet sizes, ``W`` the sample stack and ``C`` the stack of base
points; a block (no jet sizes) is ``C`` and the shape of the core it stands for, summed over the samples.
``F = tuple(n + 1 for n in order)`` and ``J_in_i`` are the jet sizes of ``w[i]``. The tilde variables are dual jets,
stored with every jet axis turned.

| Variable | Index | Jet sizes | Size |
| --- | --- | --- | --- |
| `xi` | site | min(J_in_i, F) | nD_i |
| `mu` | bond | F | rL_i |
| `nu` | bond | F | rR_i |
| `nu_L` | bond | F | rL_i |
| `eta` | site | F | nU_i |
| `eta_D` | site | F | nD_i |
| `z` | mode | F | N_i |
| `r` | mode | F | N_i |
| `w_tilde` | mode | F | N_i |
| `xi_tilde` | site | F | nD_i |
| `mu_tilde` | bond | F | rL_i |
| `nu_tilde` | bond | F | rR_i |
| `eta_tilde` | site | F | nU_i |
| `_eta_tilde_top` | site | min(J_in_i, F) | nU_i |
| `V_tilde` | site | none | (nU_i, N_i) |
| `H_tilde` | site | none | (rL_i, nD_i, rR_(i+1)) |
| `xi_U_dot` | site | min(J_in_i, F) | nU_i |
| `mu_R_dot` | bond | F | rR_i |
| `nu_L_dot` | bond | F | rL_i |
| `eta_D_dot` | site | F | nD_i |
| `Jp` | mode | F | N_i |
| `xi_dot` | site | min(J_in_i, F) | nD_i |
| `mu_dot` | bond | F | rL_i |
| `nu_L_tilde` | bond | F | rL_i |
| `eta_D_tilde` | site | F | nD_i |
| `_eta_D_tilde_top` | site | min(J_in_i, F) | nD_i |
| `xi_tilde_dot` | site | F | nD_i |
| `mu_tilde_dot` | bond | F | rL_i |
| `nu_L_tilde_dot` | bond | F | rL_i |
| `eta_D_tilde_dot` | site | F | nD_i |
| `_eta_D_tilde_dot_top` | site | min(J_in_i, F) | nD_i |
| `V_tilde_dot` | site | none | (nD_i, N_i) |
| `H_tilde_dot` | site | none | (rL_i, nD_i, rL_(i+1)) |
| `nu_dot` | bond | F | rL_i |
| `eta_dot` | site | F | nD_i |
| `nu_tilde_dot` | bond | F | rL_i |
| `eta_tilde_dot` | site | F | nD_i |
| `_eta_tilde_dot_top` | site | min(J_in_i, F) | nD_i |

A variable made by a row with ``K = yes`` has ``K`` after ``W`` (``J + W + K + C + (size,)``; a block ``K + C`` and the
core's shape); a private node's shape is its base's. ``f`` has the shape ``C``; ``r_block_sumsq`` the shape of the
weight, ``(d,) + F``; ``Jp_norm_sq`` the shape ``K + C``; ``g`` is the pair ``(V_tilde, H_tilde)``, ``JtJp`` the
pair of their twins and ``Kp`` the pair ``(V_tilde_dot, H_tilde_dot)``. On a corewise point the sizes of ``R`` and ``U``
are those of ``L`` and ``D``. A bond index runs over ``0 .. d`` (``d + 1`` arrays), a site or a mode index over
``0 .. d-1``.

**What a call does, in order**: it checks what it needs to run, and nothing more; reads the
layout and the call type from the structure of the frame; takes the closure of what is wanted over the graph, the
held variables its leaves; chooses numpy or jax over every array input and the result dtype, the lowest floating
precision among the arrays the closure reads and the held variables read or wanted, and casts those once; runs the rows; returns a tuple aligned with
``want``. Every scan and map body is module-level, memoized on ``(step_name, lens, order, form)``: a body closes
over the record, the box and the form, nothing per site.

Only the probe kind on the ragged layout, on a manifold frame and on a corewise point, is built here; the other
kinds and the uniform layout are not taken yet.
"""
import dataclasses as dc
import functools
import typing as typ

import numpy as np

import t3toolbox.backend.contractions as contractions
import t3toolbox.backend.sampling_steps as sampling_steps
from t3toolbox.backend.common import *
from t3toolbox.backend.jets import _lowest_precision_dtype

if jax_available:
    import jax

__all__ = [
    'sampling_variables',
    'HeldVariables',
]


#####################################################
########    The declared graph    ###################
#####################################################

class Rule(typ.NamedTuple):
    '''One row of the declared graph.'''
    name:       str                         # the variable the row makes
    step:       typ.Tuple[str, str]         # (driver, step_name): 'site' | 'left' | 'right' | 'whole'
    reads:      typ.Tuple[str, ...]         # the step's arrays, in order: variables, stores, arguments
    needs:      typ.FrozenSet[str]          # the facts of the call the row requires
    carries_K:  bool                        # True: the call's K; False: K = 0
    start:      typ.Optional[str] = None    # a sweep's start value: 'ones' | 'zeros'; None for the other drivers


_NONE = frozenset()
_VAR = frozenset({'variations'})
_COR = frozenset({'core_directions'})
_COREWISE = frozenset({'_corewise'})

_PROBE_ROWS = (
    # the plain grade
    Rule('xi',             ('site', 'xi_step'),             ('D', 'w'),                    _NONE,             False),
    Rule('mu',             ('left', 'mu_step'),             ('L', 'xi'),                   _NONE,             False, 'ones'),
    Rule('nu',             ('right', 'nu_step'),            ('R', 'xi'),                   _NONE,             False, 'ones'),
    Rule('nu_L',           ('right', 'nu_step'),            ('L', 'xi'),                   _NONE,             False, 'ones'),
    Rule('eta',            ('site', 'eta_step'),            ('mu', 'U', 'nu'),             _NONE,             False),
    Rule('eta_D',          ('site', 'eta_step'),            ('mu', 'L', 'nu_L'),           _NONE,             False),
    Rule('z',              ('site', 'z_step'),              ('eta', 'Vc'),                 frozenset({'Vc'}), False),
    Rule('z',              ('site', 'z_step'),              ('eta_D', 'D'),                _NONE,             False),
    Rule('r',              ('whole', 'r_step'),             ('z', 'y'),                    frozenset({'y'}),  False),
    Rule('f',              ('whole', 'f_step'),             ('r', 'weight'),               _NONE,             False),
    Rule('r_block_sumsq',  ('whole', 'r_block_sumsq_step'), ('r',),                        _NONE,             False),
    # the tilde grade: the residual's adjoint
    Rule('w_tilde',        ('whole', 'w_tilde_step'),       ('r', 'weight'),               _NONE,             False),
    Rule('xi_tilde',       ('site', 'xi_tilde_step'),       ('D', 'w_tilde'),              _NONE,             False),
    Rule('mu_tilde',       ('left', 'mu_tilde_step'),       ('L', 'mu', 'xi', 'xi_tilde'), _NONE,             False, 'zeros'),
    Rule('nu_tilde',       ('right', 'nu_tilde_step'),      ('R', 'nu', 'xi', 'xi_tilde'), _NONE,             False, 'zeros'),
    Rule('eta_tilde',      ('site', 'eta_tilde_step'),      ('mu_tilde', 'mu', 'U', 'nu_tilde', 'nu'),
                                                                                           _NONE,             False),
    Rule('_eta_tilde_top', ('site', 'eta_tilde_top_step'),  ('mu_tilde', 'mu', 'U', 'nu_tilde', 'nu', 'w'),
                                                                                           _NONE,             False),
    Rule('nu_L_tilde',     ('right', 'nu_tilde_step'),      ('L', 'nu_L', 'xi', 'xi_tilde'),
                                                                                           _NONE,             False, 'zeros'),
    Rule('eta_D_tilde',    ('site', 'eta_tilde_step'),      ('mu_tilde', 'mu', 'L', 'nu_L_tilde', 'nu_L'),
                                                                                           _NONE,             False),
    Rule('_eta_D_tilde_top', ('site', 'eta_tilde_top_step'), ('mu_tilde', 'mu', 'L', 'nu_L_tilde', 'nu_L', 'w'),
                                                                                           _NONE,             False),
    # the blocks: on a corewise point in the base point's basis (the routes renamed into its g), else the frame's
    Rule('V_tilde',        ('site', 'V_tilde_step'),        ('_eta_D_tilde_top', 'eta_D', 'w', 'w_tilde'),
                                                                                           _COREWISE,         False),
    Rule('V_tilde',        ('site', 'V_tilde_step'),        ('_eta_tilde_top', 'eta', 'w', 'w_tilde'),
                                                                                           _NONE,             False),
    Rule('H_tilde',        ('site', 'H_tilde_step'),        ('mu_tilde', 'mu', 'xi_tilde', 'xi', 'nu_L_tilde', 'nu_L'),
                                                                                           _COREWISE,         False),
    Rule('H_tilde',        ('site', 'H_tilde_step'),        ('mu_tilde', 'mu', 'xi_tilde', 'xi', 'nu_tilde', 'nu'),
                                                                                           _NONE,             False),
    Rule('g',              ('whole', 'g_step'),             ('V_tilde', 'H_tilde'),        _NONE,             False),
    # the dot grade, the frame's form: the direction (V, H) at one site, summed over the sites
    Rule('xi_U_dot',       ('site', 'xi_dot_step'),         ('V', 'w'),                    _VAR,              True),
    Rule('mu_R_dot',       ('left', 'mu_dot_step'),         ('R', 'mu', 'xi', 'xi_U_dot', 'U', 'H'),
                                                                                           _VAR,              True, 'zeros'),
    Rule('_frame.nu_L_dot', ('right', 'nu_dot_step'),       ('L', 'nu', 'xi', 'xi_U_dot', 'U', 'H'),
                                                                                           _VAR,              True, 'zeros'),
    Rule('_frame.eta_D_dot', ('site', 'eta_dot_step'),      ('mu_R_dot', 'R', 'mu', 'H', 'L', '_frame.nu_L_dot', 'nu'),
                                                                                           _VAR,              True),
    # the dot grade, the raw form: the direction (D_dot, L_dot) at the base point, every other core an L core
    Rule('xi_dot',         ('site', 'xi_dot_step'),         ('D_dot', 'w'),                _COR,              True),
    Rule('mu_dot',         ('left', 'mu_dot_step'),         ('L', 'mu', 'xi', 'xi_dot', 'L', 'L_dot'),
                                                                                           _COR,              True, 'zeros'),
    Rule('_raw.nu_L_dot',  ('right', 'nu_dot_step'),        ('L', 'nu_L', 'xi', 'xi_dot', 'L', 'L_dot'),
                                                                                           _COR,              True, 'zeros'),
    Rule('_raw.eta_D_dot', ('site', 'eta_dot_step'),        ('mu_dot', 'L', 'mu', 'L_dot', 'L', '_raw.nu_L_dot', 'nu_L'),
                                                                                           _COR,              True),
    # the two names with both forms (O-12): the raw form when core_directions is given, else the frame's
    Rule('nu_L_dot',       ('whole', 'identity_step'),      ('_raw.nu_L_dot',),            _COR,              True),
    Rule('nu_L_dot',       ('whole', 'identity_step'),      ('_frame.nu_L_dot',),          _VAR,              True),
    Rule('eta_D_dot',      ('whole', 'identity_step'),      ('_raw.eta_D_dot',),           _COR,              True),
    Rule('eta_D_dot',      ('whole', 'identity_step'),      ('_frame.eta_D_dot',),         _VAR,              True),
    # the direction's output: through the frame's form when variations is given, else the raw form at the base point
    Rule('Jp',             ('site', 'Jp_step'),             ('_frame.eta_D_dot', 'D', 'eta', 'V'),
                                                                                           _VAR,              True),
    Rule('Jp',             ('site', 'Jp_step'),             ('_raw.eta_D_dot', 'D', 'eta_D', 'D_dot'),
                                                                                           _COR,              True),
    Rule('Jp_norm_sq',     ('whole', 'Jp_norm_sq_step'),    ('Jp', 'weight'),              _NONE,             True),
    # the tilde-dot grade: the direction (raw form) and the residual's seed, both, at the base point
    Rule('xi_tilde_dot',   ('site', 'xi_tilde_dot_step'),   ('D_dot', 'w_tilde'),          _COR,              True),
    Rule('mu_tilde_dot',   ('left', 'mu_tilde_dot_step'),   ('L', 'mu_dot', 'mu_tilde', 'mu', 'xi_tilde_dot', 'xi_dot',
                                                             'xi_tilde', 'xi', 'L_dot'),   _COR,              True, 'zeros'),
    Rule('nu_L_tilde_dot', ('right', 'nu_tilde_dot_step'),  ('L', '_raw.nu_L_dot', 'nu_L_tilde', 'nu_L', 'xi_tilde_dot',
                                                             'xi_dot', 'xi_tilde', 'xi', 'L_dot'),
                                                                                           _COR,              True, 'zeros'),
    Rule('eta_D_tilde_dot', ('site', 'eta_tilde_dot_step'), ('mu_tilde_dot', 'mu_dot', 'mu_tilde', 'mu', 'L',
                                                             'nu_L_tilde_dot', '_raw.nu_L_dot', 'nu_L_tilde', 'nu_L',
                                                             'L_dot'),                     _COR,              True),
    Rule('_eta_D_tilde_dot_top', ('site', 'eta_tilde_dot_top_step'),
                                                            ('mu_tilde_dot', 'mu_dot', 'mu_tilde', 'mu', 'L',
                                                             'nu_L_tilde_dot', '_raw.nu_L_dot', 'nu_L_tilde', 'nu_L',
                                                             'L_dot', 'w'),                _COR,              True),
    Rule('V_tilde_dot',    ('site', 'V_tilde_dot_step'),    ('_eta_D_tilde_dot_top', '_raw.eta_D_dot', 'w', 'w_tilde'),
                                                                                           _COR,              True),
    Rule('H_tilde_dot',    ('site', 'H_tilde_dot_step'),    ('mu_tilde_dot', 'mu_dot', 'mu_tilde', 'mu', 'xi_tilde_dot',
                                                             'xi_dot', 'xi_tilde', 'xi', 'nu_L_tilde_dot',
                                                             '_raw.nu_L_dot', 'nu_L_tilde', 'nu_L'),
                                                                                           _COR,              True),
    Rule('Kp',             ('whole', 'g_step'),             ('V_tilde_dot', 'H_tilde_dot'), _COR,             True),
)

# The adjoint seeded by Jp (row 3.3; JtJp): a private twin of every row on the path from the seed to the blocks,
# named twin(name), with the same step and start and the call's K; JtJp is the twin of g. No twin of the full
# eta_tilde (nothing reads it, and a private name cannot be wanted). The path is given by role, per call type: on a
# corewise point it is the manifold path of the base point's basis, through the rename below.
_JP_PATH = ('w_tilde', 'xi_tilde', 'mu_tilde', 'nu_tilde', '_eta_tilde_top', 'V_tilde', 'H_tilde')
_JP_PATH_COREWISE = ('w_tilde', 'xi_tilde', 'mu_tilde', 'nu_L_tilde', '_eta_D_tilde_top', 'V_tilde', 'H_tilde')


def _twin(name: str) -> str:
    '''The name of the twin, in the adjoint seeded by ``Jp``, of a row on the residual's path.'''
    return '_Jp.' + name.lstrip('_')


def _jp_twins(
        graph:      typ.Tuple[Rule, ...],
        path:       typ.Tuple[str, ...],    # the residual's rows on the path to the blocks, in this graph's names
        direction:  str,                    # the argument the direction comes in: 'variations' | 'core_directions'
        call:       str,                    # the call type of this graph: '_manifold' | '_corewise'
) -> typ.Tuple[Rule, ...]:
    '''The rows of the adjoint seeded by ``Jp``, derived from the residual's: each read on the path renamed by
    ``_twin``, the seed reading ``Jp`` for ``r``, the call's ``K``; and ``JtJp``, the twin of ``g``, which needs the
    direction itself (so that a call without it names ``JtJp``, not a chain of private nodes).'''
    rename = lambda read: _twin(read) if read in path else ('Jp' if read == 'r' else read)
    out, jtjp = [], None
    other = '_corewise' if call == '_manifold' else '_manifold'
    for rule in graph:
        if other in rule.needs:                         # a route of the other call type has no twin here
            continue
        if rule.name in path:
            out.append(rule._replace(name=_twin(rule.name), reads=tuple(map(rename, rule.reads)), carries_K=True))
        elif rule.name == 'g':
            jtjp = rule._replace(name='JtJp', reads=tuple(map(rename, rule.reads)), carries_K=True,
                                 needs=rule.needs | {direction})
    return tuple(out) + (jtjp,)


# The corewise point (B, G): one basis on each edge. Its graph is derived from the manifold one: every row that reads
# the frame's other stores (U, R), the frame's form of a direction (V, H), the Tucker centers or a frame node is
# dropped, with every row that reads a dropped name; the rest are renamed, in their names, reads AND needs, by one
# table (the base point's basis is the only one, so its names lose their basis; the direction is core_directions);
# then the twins are derived from the result.
_COREWISE_RENAME = {
    'nu_L': 'nu', 'eta_D': 'eta', 'nu_L_tilde': 'nu_tilde', 'eta_D_tilde': 'eta_tilde',
    'nu_L_dot': 'nu_dot', 'eta_D_dot': 'eta_dot', 'nu_L_tilde_dot': 'nu_tilde_dot', 'eta_D_tilde_dot': 'eta_tilde_dot',
    '_raw.nu_L_dot': '_raw.nu_dot', '_raw.eta_D_dot': '_raw.eta_dot',
    '_eta_D_tilde_top': '_eta_tilde_top', '_eta_D_tilde_dot_top': '_eta_tilde_dot_top',
    'variations': 'core_directions',
}
_COREWISE_DROP = frozenset({'U', 'R', 'V', 'H', 'Vc'})


def _corewise(rows: typ.Tuple[Rule, ...]) -> typ.Tuple[Rule, ...]:
    '''The graph of a corewise call, derived from the manifold rows (without their twins).'''
    dropped_rows = set()
    while True:
        alive = {rule.name for k, rule in enumerate(rows) if k not in dropped_rows}
        more = {k for k, rule in enumerate(rows) if k not in dropped_rows and (
            rule.name.startswith('_frame.') or '_manifold' in rule.needs or set(rule.reads) & _COREWISE_DROP
            or any(read not in alive and read in {r.name for r in rows} for read in rule.reads))}
        if not more:
            break
        dropped_rows |= more
    kept = [rule for k, rule in enumerate(rows) if k not in dropped_rows]
    rename = lambda name: _COREWISE_RENAME.get(name, name)
    unrenamed = {rule.name for rule in kept if rule.name not in _COREWISE_RENAME}
    for rule in kept:                                    # a rename never lands on a name a surviving row keeps
        if rule.name in _COREWISE_RENAME and rename(rule.name) in unrenamed:     # raise, not assert: python -O
            raise RuntimeError('the corewise rename of %r lands on the surviving %r' % (rule.name, rename(rule.name)))
    renamed = tuple(rule._replace(name=rename(rule.name), reads=tuple(map(rename, rule.reads)),
                                  needs=frozenset(map(rename, rule.needs))) for rule in kept)
    path = tuple(map(rename, _JP_PATH_COREWISE))
    if not set(path) <= {rule.name for rule in renamed}:
        raise RuntimeError('the corewise path of the adjoint seeded by Jp is incomplete')
    return renamed + _jp_twins(renamed, path, 'core_directions', '_corewise')


_PROBE_GRAPH = _PROBE_ROWS + _jp_twins(_PROBE_ROWS, _JP_PATH, 'variations', '_manifold')
_PROBE_COREWISE_GRAPH = _corewise(_PROBE_ROWS)

_GRAPHS = {('probe', '_manifold'): _PROBE_GRAPH, ('probe', '_corewise'): _PROBE_COREWISE_GRAPH}
_KINDS = ('probe', 'apply', 'entries')

_STORES = frozenset({'D', 'U', 'L', 'R'})                   # the stores of a manifold frame
_ARGUMENTS = frozenset({'w', 'y', 'weight', 'Vc', 'V', 'H', 'D_dot', 'L_dot'})   # what a row may read but the stores:
                                                    # 'w' the sample; V, H the variations; D_dot, L_dot the core directions

# The direction operands and the masks P11 masks each with on the uniform layout: the frame's variation masks for the
# frame's form and the Tucker centers, the store's own masks for a core direction (the frame's D and L, never the
# point's, which are a superset at TT slack).
_DIRECTION_MASKS = {'V': 'variations', 'H': 'variations', 'Vc': 'variations', 'D_dot': 'D', 'L_dot': 'L'}
_DRIVERS = ('site', 'left', 'right', 'whole')
_STARTS = ('ones', 'zeros')

# A site op at site i reads the left of bond i (the mu family) and the right of bond i + 1 (the nu family).
_BOND_OFFSET = {'mu': 0, 'nu': 1}


def _base(name: str) -> str:
    '''The name of a private node with its prefix and leading underscore taken off: what its letter and its table
    entry are read from (``'_Jp.mu_tilde'`` -> ``'mu_tilde'``, ``'_frame.nu_L_dot'`` -> ``'nu_L_dot'``,
    ``'_eta_tilde_top'`` -> ``'eta_tilde_top'``).'''
    return name.split('.', 1)[-1].lstrip('_')


def _letter(name: str) -> str:
    '''The letter of a variable (its side): ``mu``, ``nu``, ``xi``, ``eta``, ...'''
    return _base(name).split('_')[0]


def _table_name(name: str) -> str:
    '''The entry of ``_SHAPES`` a variable's shape is read from: its own, or its base's (a twin, a frame node).'''
    if name in _SHAPES:
        return name
    base = _base(name)
    return base if base in _SHAPES else '_' + base

# name: (index, jet sizes, size). The index: 'bond' (0 .. d), 'site' or 'mode' (0 .. d-1). The jet sizes: 'F';
# 'leaf' (the input's, clipped to F); or 'none' (a block: no jet axes and no W). The size of the last axis at index i,
# or for a block the shape of the core it stands for ('V': (nUi, Ni); 'H': (rLi, nDi, rR(i+1))).
_SHAPES = {
    'xi':              ('site', 'leaf', 'nD'),
    'mu':              ('bond', 'F',    'rL'),
    'nu':              ('bond', 'F',    'rR'),
    'nu_L':            ('bond', 'F',    'rL'),
    'eta':             ('site', 'F',    'nU'),
    'eta_D':           ('site', 'F',    'nD'),
    'z':               ('mode', 'F',    'N'),
    'r':               ('mode', 'F',    'N'),
    'w_tilde':         ('mode', 'F',    'N'),
    'xi_tilde':        ('site', 'F',    'nD'),
    'mu_tilde':        ('bond', 'F',    'rL'),
    'nu_tilde':        ('bond', 'F',    'rR'),
    'eta_tilde':       ('site', 'F',    'nU'),
    '_eta_tilde_top':  ('site', 'leaf', 'nU'),
    'V_tilde':         ('site', 'none', 'V'),
    'H_tilde':         ('site', 'none', 'H'),
    'xi_U_dot':        ('site', 'leaf', 'nU'),
    'mu_R_dot':        ('bond', 'F',    'rR'),
    'nu_L_dot':        ('bond', 'F',    'rL'),
    'eta_D_dot':       ('site', 'F',    'nD'),
    'Jp':              ('mode', 'F',    'N'),
    'xi_dot':          ('site', 'leaf', 'nD'),
    'mu_dot':          ('bond', 'F',    'rL'),
    'nu_L_tilde':      ('bond', 'F',    'rL'),
    'eta_D_tilde':     ('site', 'F',    'nD'),
    '_eta_D_tilde_top': ('site', 'leaf', 'nD'),
    'xi_tilde_dot':    ('site', 'F',    'nD'),
    'mu_tilde_dot':    ('bond', 'F',    'rL'),
    'nu_L_tilde_dot':  ('bond', 'F',    'rL'),
    'eta_D_tilde_dot': ('site', 'F',    'nD'),
    '_eta_D_tilde_dot_top': ('site', 'leaf', 'nD'),
    'V_tilde_dot':     ('site', 'none', 'VD'),
    'H_tilde_dot':     ('site', 'none', 'HL'),
    # the names of a corewise point (its sizes: the R and U sizes are those of L and D there)
    'nu_dot':          ('bond', 'F',    'rL'),
    'eta_dot':         ('site', 'F',    'nD'),
    'nu_tilde_dot':    ('bond', 'F',    'rL'),
    'eta_tilde_dot':   ('site', 'F',    'nD'),
    '_eta_tilde_dot_top': ('site', 'leaf', 'nD'),
}


def _made(graph: typ.Tuple[Rule, ...]) -> typ.Tuple[str, ...]:
    '''The variables a graph makes, each once, in the order of their first row; private nodes included.'''
    return tuple(dict.fromkeys(rule.name for rule in graph))


def _public(graph: typ.Tuple[Rule, ...]) -> typ.Tuple[str, ...]:
    '''The public variables a graph makes: those that can be wanted and held.'''
    return tuple(name for name in _made(graph) if not name.startswith('_'))


@functools.lru_cache(maxsize=None)
def _closure(
        kind:     str,
        want:     typ.Tuple[str, ...],      # the wanted variables (not 'w')
        held:     typ.FrozenSet[str],       # the filled slots of held: leaves of the closure
        present:  typ.FrozenSet[str],       # the facts of the call: the arguments given and the call type
) -> typ.Tuple[Rule, ...]:                  # the rows to run, in dependency order
    '''The rows that make what is wanted, from the declared graph of ``kind``: for each name not held, its first
    row whose ``needs`` are present, after the rows of the variables it reads. Raises a ``ValueError`` when no row
    of a needed name applies, naming what is missing and through which variables it was needed.'''
    graph = _GRAPHS[kind, '_corewise' if '_corewise' in present else '_manifold']
    rows: typ.Dict[str, typ.List[Rule]] = {}
    for rule in graph:
        rows.setdefault(rule.name, []).append(rule)
    done, out = set(held), []

    def visit(name, chain):
        if name in done or name not in rows:            # held, already made, or a store or an argument
            return
        for rule in rows[name]:
            if rule.needs <= present:
                break
        else:
            missing = sorted({fact for rule in rows[name] for fact in rule.needs - present})
            through = (' (needed through %s)' % ' -> '.join(chain + (name,))) if chain else ''
            raise ValueError('sampling_variables: %r needs the argument %s, which was not given%s'
                             % (name, ' or '.join(missing), through))
        for read in rule.reads:
            visit(read, chain + (name,))
        done.add(name)
        out.append(rule)

    for name in want:
        visit(name, ())
    return tuple(out)


# Every public name of a corewise point is reachable there, with the direction as core_directions and data.
for _name in _public(_PROBE_COREWISE_GRAPH):
    _closure('probe', (_name,), frozenset(), frozenset({'_corewise', 'y', 'core_directions'}))
del _name


#####################################################
########    Held variables    #######################
#####################################################

@dc.dataclass(frozen=True, eq=False)
class HeldVariables:
    """Sampling variables kept at one frame, to be handed back to :py:func:`sampling_variables` as ``held=``.

    One slot per public computed variable of the engine's graphs (``w``, the sample, and the private nodes
    excepted); a slot is ``None`` when not held. Each holds the variable as the engine returns it: a tuple over its
    index with the shapes of the module's table, one array for ``f`` and ``r_block_sumsq``, the pair of tuples for
    ``g``. It reads like a dictionary over its filled slots: ``held['mu']``, ``'mu' in held``, ``held.keys()``,
    iteration and ``len``. An unknown name raises, at construction, in :py:meth:`with_values` and at lookup.
    :py:meth:`with_values` returns a new record; a record is never changed in place.

    It checks no shape: the engine checks every held variable it is handed, at every call, against the shapes of
    its table, and trusts the values. Where jax is importable it is a registered pytree: the filled slots are the
    leaves, the set of filled slots is the static structure. It refuses ``==`` and is unhashable, as every
    array-carrying class of the library is: compare the slots you mean.

    >>> import numpy as np
    >>> from t3toolbox.backend.sampling_variables import HeldVariables
    >>> held = HeldVariables(mu=(np.ones((3, 1)), np.ones((3, 2)), np.ones((3, 1))))
    >>> print(held.keys(), 'mu' in held, 'nu' in held, len(held))
    ('mu',) True False 1
    >>> held2 = held.with_values(r=(np.zeros((3, 4)),))
    >>> print(held.keys(), held2.keys())
    ('mu',) ('mu', 'r')
    >>> held['nu']                                      # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    KeyError: "HeldVariables: the slot 'nu' is not filled"
    """
    xi:             typ.Optional[typ.Sequence[NDArray]] = None      # len=d,   elm_shape=min(J_in_i,F)+W+C+(nDi,)
    mu:             typ.Optional[typ.Sequence[NDArray]] = None      # len=d+1, elm_shape=F+W+C+(rLi,)
    nu:             typ.Optional[typ.Sequence[NDArray]] = None      # len=d+1, elm_shape=F+W+C+(rRi,)
    nu_L:           typ.Optional[typ.Sequence[NDArray]] = None      # len=d+1, elm_shape=F+W+C+(rLi,)
    eta:            typ.Optional[typ.Sequence[NDArray]] = None      # len=d,   elm_shape=F+W+C+(nUi,)
    eta_D:          typ.Optional[typ.Sequence[NDArray]] = None      # len=d,   elm_shape=F+W+C+(nDi,)
    z:              typ.Optional[typ.Sequence[NDArray]] = None      # len=d,   elm_shape=F+W+C+(Ni,)
    r:              typ.Optional[typ.Sequence[NDArray]] = None      # len=d,   elm_shape=F+W+C+(Ni,)
    f:              typ.Optional[NDArray] = None                    # shape=C
    r_block_sumsq:  typ.Optional[NDArray] = None                    # shape=(d,)+F
    w_tilde:        typ.Optional[typ.Sequence[NDArray]] = None      # len=d,   elm_shape=F+W+C+(Ni,), turned
    xi_tilde:       typ.Optional[typ.Sequence[NDArray]] = None      # len=d,   elm_shape=F+W+C+(nDi,), turned
    mu_tilde:       typ.Optional[typ.Sequence[NDArray]] = None      # len=d+1, elm_shape=F+W+C+(rLi,), turned
    nu_tilde:       typ.Optional[typ.Sequence[NDArray]] = None      # len=d+1, elm_shape=F+W+C+(rRi,), turned
    eta_tilde:      typ.Optional[typ.Sequence[NDArray]] = None      # len=d,   elm_shape=F+W+C+(nUi,), turned
    nu_L_tilde:     typ.Optional[typ.Sequence[NDArray]] = None      # len=d+1, elm_shape=F+W+C+(rLi,), turned
    eta_D_tilde:    typ.Optional[typ.Sequence[NDArray]] = None      # len=d,   elm_shape=F+W+C+(nDi,), turned
    V_tilde:        typ.Optional[typ.Sequence[NDArray]] = None      # len=d,   elm_shape=C+(nUi,Ni)
    H_tilde:        typ.Optional[typ.Sequence[NDArray]] = None      # len=d,   elm_shape=C+(rLi,nDi,rR(i+1))
    g:              typ.Optional[typ.Tuple] = None                  # (V_tilde, H_tilde)
    xi_U_dot:       typ.Optional[typ.Sequence[NDArray]] = None      # len=d,   elm_shape=min(J_in_i,F)+W+K+C+(nUi,)
    mu_R_dot:       typ.Optional[typ.Sequence[NDArray]] = None      # len=d+1, elm_shape=F+W+K+C+(rRi,)
    xi_dot:         typ.Optional[typ.Sequence[NDArray]] = None      # len=d,   elm_shape=min(J_in_i,F)+W+K+C+(nDi,)
    mu_dot:         typ.Optional[typ.Sequence[NDArray]] = None      # len=d+1, elm_shape=F+W+K+C+(rLi,)
    nu_L_dot:       typ.Optional[typ.Sequence[NDArray]] = None      # len=d+1, elm_shape=F+W+K+C+(rLi,)
    eta_D_dot:      typ.Optional[typ.Sequence[NDArray]] = None      # len=d,   elm_shape=F+W+K+C+(nDi,)
    Jp:             typ.Optional[typ.Sequence[NDArray]] = None      # len=d,   elm_shape=F+W+K+C+(Ni,)
    Jp_norm_sq:     typ.Optional[NDArray] = None                    # shape=K+C
    xi_tilde_dot:   typ.Optional[typ.Sequence[NDArray]] = None      # len=d,   elm_shape=F+W+K+C+(nDi,), turned
    mu_tilde_dot:   typ.Optional[typ.Sequence[NDArray]] = None      # len=d+1, elm_shape=F+W+K+C+(rLi,), turned
    nu_L_tilde_dot: typ.Optional[typ.Sequence[NDArray]] = None      # len=d+1, elm_shape=F+W+K+C+(rLi,), turned
    eta_D_tilde_dot: typ.Optional[typ.Sequence[NDArray]] = None     # len=d,   elm_shape=F+W+K+C+(nDi,), turned
    V_tilde_dot:    typ.Optional[typ.Sequence[NDArray]] = None      # len=d,   elm_shape=K+C+(nDi,Ni)
    H_tilde_dot:    typ.Optional[typ.Sequence[NDArray]] = None      # len=d,   elm_shape=K+C+(rLi,nDi,rL(i+1))
    Kp:             typ.Optional[typ.Tuple] = None                  # (V_tilde_dot, H_tilde_dot)
    JtJp:           typ.Optional[typ.Tuple] = None                  # the pair of blocks, K+C+(nUi,Ni), K+C+(rLi,nDi,rR(i+1))
    # the names of a corewise point that a manifold call spells with a basis
    nu_dot:         typ.Optional[typ.Sequence[NDArray]] = None      # len=d+1, elm_shape=F+W+K+C+(ri,)
    eta_dot:        typ.Optional[typ.Sequence[NDArray]] = None      # len=d,   elm_shape=F+W+K+C+(ni,)
    nu_tilde_dot:   typ.Optional[typ.Sequence[NDArray]] = None      # len=d+1, elm_shape=F+W+K+C+(ri,), turned
    eta_tilde_dot:  typ.Optional[typ.Sequence[NDArray]] = None      # len=d,   elm_shape=F+W+K+C+(ni,), turned

    @classmethod
    def slots(cls) -> typ.Tuple[str, ...]:
        '''Every slot, filled or not.'''
        return tuple(field.name for field in dc.fields(cls))

    def _require_slot(self, name: str) -> None:
        if name not in self.slots():
            raise KeyError('HeldVariables: %r is not a variable that can be held; the slots are %s'
                           % (name, ', '.join(self.slots())))

    def keys(self) -> typ.Tuple[str, ...]:
        '''The filled slots, in the order of the slots.'''
        return tuple(name for name in self.slots() if getattr(self, name) is not None)

    def __getitem__(self, name: str):
        self._require_slot(name)
        value = getattr(self, name)
        if value is None:
            raise KeyError('HeldVariables: the slot %r is not filled' % (name,))
        return value

    def __contains__(self, name) -> bool:
        return name in self.keys()

    def __iter__(self):
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self.keys())

    def __eq__(self, other):
        if self is other:
            return True
        raise TypeError('HeldVariables does not define ==: compare the slots you mean (`a is b` is identity)')

    __hash__ = None

    def with_values(self, **values) -> 'HeldVariables':
        '''A new record with these slots set (``None`` empties a slot); this one is unchanged.'''
        unknown = sorted(set(values) - set(self.slots()))
        if unknown:
            raise TypeError('HeldVariables.with_values: %s %s not a variable that can be held; the slots are %s'
                            % (', '.join(map(repr, unknown)), 'is' if len(unknown) == 1 else 'are',
                               ', '.join(self.slots())))
        return dc.replace(self, **values)


if jax_available:
    jax.tree_util.register_pytree_node(
        HeldVariables,
        lambda held: (tuple(getattr(held, name) for name in held.keys()), held.keys()),
        lambda names, values: HeldVariables(**dict(zip(names, values))),
    )


#####################################################
########    The setting of a call    ################
#####################################################

class _Setting(typ.NamedTuple):
    '''What the shapes of a call are made of, read once from the frame, the sample and the jet shape.'''
    d:      int
    order:  typ.Tuple[int, ...]
    box:    typ.Tuple[int, ...]             # F
    J_in:   typ.Tuple[typ.Tuple[int, ...], ...]     # the jet sizes of each input, len=d
    W:      typ.Tuple[int, ...]
    C:      typ.Tuple[int, ...]
    sizes:  typ.Dict[str, typ.Tuple[int, ...]]      # 'rL', 'rR' (len=d+1), 'nD', 'nU', 'N' (len=d)
    K:      typ.Tuple[int, ...] = ()        # the stack of directions, read off variations
    with_K: typ.FrozenSet[str] = frozenset()        # the variables whose rows carry K


def _bonds(tt: typ.Sequence[NDArray]) -> typ.Tuple[int, ...]:
    '''The sizes of the bonds 0 .. d in the basis of the TT cores ``tt``.'''
    return tuple(np.shape(G)[-3] for G in tt) + (np.shape(tt[-1])[-1],)


def _read_K(
        variations:       typ.Optional[typ.Tuple],  # (V, H), elm_shapes K+C+(nUi,Ni), K+C+(rLi,nDi,rR(i+1)); or None
        core_directions:  typ.Optional[typ.Tuple],  # (D_dot, L_dot), the shapes of D and L with K+C; or None
        C:                typ.Tuple[int, ...],      # the stack of base points, read off the frame
) -> typ.Tuple[int, ...]:                           # K, the stack of directions; () with no direction
    '''The one stack of directions of the call, read off ``V[0]`` (or ``D_dot[0]`` with no variations); every
    ``V[i]``, ``H[i]``, ``D_dot[i]`` and ``L_dot[i]`` must have the prefix ``K + C`` and the rank of its core
    (structural: a ``ValueError``, naming the operand). A size-1 axis where another operand has more would otherwise
    broadcast in a contraction and give a wrong answer with no error; and the curvature pairs element ``k`` of a
    direction with element ``k`` of its lift, so both directions share one ``K``.'''
    operands = []
    if variations is not None:
        operands += [('V', variations[0], 2), ('H', variations[1], 3)]
    if core_directions is not None:
        operands += [('D_dot', core_directions[0], 2), ('L_dot', core_directions[1], 3)]
    if not operands:
        return ()
    name0, cores0, core_ndim0 = operands[0]
    K = tuple(np.shape(cores0[0])[:max(np.ndim(cores0[0]) - core_ndim0 - len(C), 0)])
    for name, cores, core_ndim in operands:
        for i, core in enumerate(cores):
            shape = tuple(np.shape(core))
            if len(shape) != len(K) + len(C) + core_ndim or shape[:len(K) + len(C)] != K + C:
                raise ValueError('sampling_variables: the directions must share one stack of directions K and the '
                                 "frame's stack C: %s[%d] has shape %s, where %s[0] gives K = %s and the frame C = %s, "
                                 'so K + C + (the %d axes of the core) is expected'
                                 % (name, i, shape, name0, K, C, core_ndim))
    return K


def _read_setting(
        stores:           typ.Dict[str, typ.Sequence[NDArray]],   # D, U, L, R; or on a corewise point D, L only
        sample:           typ.Sequence[NDArray],
        order:            typ.Tuple[int, ...],
        variations:       typ.Optional[typ.Tuple] = None,
        graph:            typ.Tuple[Rule, ...] = (),
        core_directions:  typ.Optional[typ.Tuple] = None,
) -> _Setting:
    D, L = stores['D'], stores['L']
    U, R = stores.get('U', L), stores.get('R', L)       # a corewise point: one basis on each edge
    m = len(order)
    C = tuple(np.shape(D[0])[:-2])
    return _Setting(
        d=len(D), order=order, box=tuple(n + 1 for n in order),
        J_in=tuple(tuple(np.shape(w)[:m]) for w in sample),
        W=tuple(np.shape(sample[0])[m:-1]), C=C,
        sizes=dict(rL=_bonds(L), rR=_bonds(R), nD=tuple(np.shape(B)[-2] for B in D),
                   nU=tuple(np.shape(G)[-2] for G in U), N=tuple(np.shape(B)[-1] for B in D)),
        K=_read_K(variations, core_directions, C), with_K=frozenset(rule.name for rule in graph if rule.carries_K))


def _shape(
        name:     str,
        i:        typ.Optional[int],
        setting:  _Setting,
) -> typ.Tuple[int, ...]:
    '''The shape of the variable ``name`` at index ``i``, from the table ``_SHAPES`` (``i`` None for ``f`` and
    ``r_block_sumsq``, which have no index).'''
    if name == 'f':
        return setting.C
    if name == 'r_block_sumsq':
        return (setting.d,) + setting.box
    K = setting.K if name in setting.with_K else ()
    if name == 'Jp_norm_sq':
        return K + setting.C
    _, jets, size = _SHAPES[_table_name(name)]
    sizes = setting.sizes
    if jets == 'none':
        core = {'V':  lambda: (sizes['nU'][i], sizes['N'][i]),
                'H':  lambda: (sizes['rL'][i], sizes['nD'][i], sizes['rR'][i + 1]),
                'VD': lambda: (sizes['nD'][i], sizes['N'][i]),
                'HL': lambda: (sizes['rL'][i], sizes['nD'][i], sizes['rL'][i + 1])}[size]()
        return K + setting.C + core
    J = tuple(min(a, f) for a, f in zip(setting.J_in[i], setting.box)) if jets == 'leaf' else setting.box
    return J + setting.W + K + setting.C + (sizes[size][i],)


def _count(name: str, setting: _Setting) -> int:
    '''How many arrays an indexed variable has: ``d + 1`` for a bond variable, ``d`` otherwise.'''
    return setting.d + 1 if _SHAPES[_table_name(name)][0] == 'bond' else setting.d


def _expected(name: str, setting: _Setting) -> typ.Any:
    '''The shapes of the whole variable ``name``, in its form: a shape for ``f`` and ``r_block_sumsq``, a tuple of
    shapes over the index for an indexed variable, the pair of those of ``V_tilde`` and ``H_tilde`` for ``g``.'''
    if name in ('f', 'r_block_sumsq', 'Jp_norm_sq'):
        return _shape(name, None, setting)
    if name in ('g', 'JtJp', 'Kp'):
        blocks = {'g': ('V_tilde', 'H_tilde'), 'JtJp': (_twin('V_tilde'), _twin('H_tilde')),
                  'Kp': ('V_tilde_dot', 'H_tilde_dot')}[name]
        return tuple(_expected(block, setting) for block in blocks)
    return tuple(_shape(name, i, setting) for i in range(_count(name, setting)))


def _check_form(where: str, value: typ.Any, expected: typ.Any) -> None:
    '''``value`` has the form of ``expected``: one array of the shape ``expected`` (a tuple of ints), or a
    sequence of the length of ``expected`` (a tuple of such), entry by entry.'''
    if all(isinstance(n, int) for n in expected):
        if isinstance(value, (tuple, list)) or tuple(np.shape(value)) != expected:
            raise ValueError('sampling_variables: the held %s has shape %s; expected %s'
                             % (where, type(value).__name__ if isinstance(value, (tuple, list)) else
                                tuple(np.shape(value)), expected))
        return
    if not isinstance(value, (tuple, list)) or len(value) != len(expected):
        raise ValueError('sampling_variables: the held %s must be a tuple of %d; got %s'
                         % (where, len(expected), type(value).__name__ if not isinstance(value, (tuple, list))
                            else 'a %s of length %d' % (type(value).__name__, len(value))))
    for i, (item, shape) in enumerate(zip(value, expected)):
        _check_form('%s[%d]' % (where, i), item, shape)


def _check_held(held: 'HeldVariables', setting: _Setting) -> None:
    '''Every filled slot has the form and the shapes of the table. Structural: raises a ``ValueError``.'''
    for name in held.keys():
        _check_form(name, held[name], _expected(name, setting))


#####################################################
########    Running the rows    #####################
#####################################################

@functools.lru_cache(maxsize=None)
def _site_body(step_name: str, lens: contractions.GroupLengths, order: typ.Tuple[int, ...], form: str):
    '''The map body of a site step: one stable object per statics (scan and map bodies close over hashable
    statics only: the record, the box and the form, nothing per site).'''
    def body(x):
        return (getattr(sampling_steps, step_name)(*x, lens=lens, order=order, form=form),)
    return body


@functools.lru_cache(maxsize=None)
def _sweep_body(step_name: str, lens: contractions.GroupLengths, order: typ.Tuple[int, ...], form: str):
    '''The scan body of a sweep step: the carry first, then the site's arrays.'''
    def body(carry, x):
        new = getattr(sampling_steps, step_name)(carry, *x, lens=lens, order=order, form=form)
        return new, (new,)
    return body


def _with_end(
        bonds:     typ.Any,                 # the d bonds a sweep made (a tuple on ragged)
        start:     NDArray,                 # its start value
        at_start:  bool,                    # True: the start is bond 0 (a left sweep); False: bond d
) -> typ.Any:                               # the d + 1 bonds
    '''The one place a sweep's result is joined to its start value.'''
    return xprepend(start, bonds) if at_start else xappend(bonds, start)


def _at_sites(name: str, value: typ.Any, d: int) -> typ.Any:
    '''What a site op reads of a variable at the sites 0 .. d-1: a bond variable at the bond its letter says.'''
    if _table_name(name) in _SHAPES and _SHAPES[_table_name(name)][0] == 'bond':
        offset = _BOND_OFFSET[_letter(name)]
        return value[offset:offset + d]
    return value


def _run(
        rule:     Rule,
        values:   typ.Dict[str, typ.Any],   # every variable made or held so far, the stores and the arguments
        lens:     contractions.GroupLengths,
        setting:  _Setting,
        form:     str,
        dtype:    typ.Any,
        xnp, xmap, xscan,
) -> typ.Any:
    driver, step_name = rule.step
    args = [values[name] for name in rule.reads]
    order = setting.order
    if driver == 'whole':
        return getattr(sampling_steps, step_name)(*args, lens=lens, order=order, form=form)
    per_site = tuple(_at_sites(name, arg, setting.d) for name, arg in zip(rule.reads, args))
    if driver == 'site':
        return xmap(_site_body(step_name, lens, order, form), per_site)[0]
    store = args[0]                                         # a sweep's first read: the TT store it sweeps through
    at_start = driver == 'left'
    size = np.shape(store[0])[-3] if at_start else np.shape(store[-1])[-1]
    if rule.start not in _STARTS:
        raise ValueError('sampling_variables: the sweep %r declares the start %r' % (rule.name, rule.start))
    stacks = (setting.K if rule.carries_K else ()) + tuple(np.shape(store[0])[:-3])     # [K] + C
    start = getattr(sampling_steps, 'start_' + rule.start)(setting.box, setting.W, stacks, size, dtype, xnp)
    body = _sweep_body(step_name, lens, order, form)
    if at_start:
        _, (bonds,) = xscan(body, start, per_site)
        return _with_end(bonds, start, at_start=True)
    _, (bonds,) = xscan(body, start, tuple(arg[::-1] for arg in per_site))
    return _with_end(bonds[::-1], start, at_start=False)


def _leaves(value: typ.Any) -> typ.List[typ.Any]:
    '''The arrays of a value: an array, or nested tuples and lists of them; ``None`` has none.'''
    if value is None:
        return []
    if isinstance(value, (tuple, list)):
        return [leaf for item in value for leaf in _leaves(item)]
    return [value]


def _cast(value: typ.Any, dtype: typ.Any, xnp) -> typ.Any:
    '''An array, or nested tuples and lists of them, cast to ``dtype`` by ``xnp``.'''
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        return tuple(_cast(item, dtype, xnp) for item in value)
    return xnp.asarray(value, dtype=dtype)


#####################################################
########    The engine    ###########################
#####################################################

def sampling_variables(
        kind:              str,                              # 'probe' | 'apply' | 'entries'
        frame:             typ.Tuple,                        # (D, U, L, R), four tuples of d arrays; or (B, G)
        sample:            typ.Any,                          # ww: the input jets, len=d, elm_shape=J_in_i+W+(Ni,)
        want:              typ.Sequence[str],                # variable names; the return is aligned with them

        order:             typ.Tuple[int, ...] = (),         # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...)
        y:                 typ.Optional[typ.Any] = None,     # the data: len=d, elm_shape=F+W+(Ni,); for r
        weight:            typ.Optional[NDArray] = None,     # (d,)+F; None = ones
        variations:        typ.Optional[typ.Tuple] = None,   # (V, H), elm_shapes K+C+(nUi,Ni), K+C+(rLi,nDi,rR(i+1))
        core_directions:   typ.Optional[typ.Tuple] = None,   # (D_dot, L_dot), the shapes of D, L with K+C
        Vc:                typ.Optional[typ.Sequence[NDArray]] = None,   # len=d, elm_shape=C+(nUi,Ni)
        sum_over_samples:  bool = True,                      # on the blocks: not read yet
        held:              typ.Optional[HeldVariables] = None,   # held variables; shapes checked, values trusted
) -> typ.Tuple[typ.Any, ...]:                                # aligned with want; shapes in the module's table
    """The sampling variables ``want`` of a Tucker tensor train at a frame, sampled at input jets.

    Each variable is computed by its recurrence over the sites, once, and only the variables that what is wanted
    reads (the closure over the declared graph, in the module docstring). Held variables are used, not recomputed (cast
    once to the result dtype, like any counted input); their shapes are checked against the table of the module docstring, their values trusted.

    The engine checks what it needs in order to run, and nothing more: an unknown kind or name raises; a jet shape
    that is not a tuple of nonnegative integers raises; an entries call with jets raises; a wanted variable whose
    argument is missing raises (``r`` without ``y`` or a held ``r``), and so does a held variable of the wrong
    shape. It does not check the form of the sample, its sizes against the frame's, ``y`` or the weight (those are
    the checkers of :py:mod:`t3toolbox.backend.jets`), and nothing numerical: four inconsistent stores give a wrong
    answer, with no error.

    Numpy or jax is read off every array input. Every output is in one dtype: the lowest floating precision among the
    counted inputs (jax over numpy), which are cast to it once. The counted inputs are the arrays the computation
    reads, and every held variable that a row reads or that is wanted (a wanted held variable is an output: it is
    returned cast). A held variable neither read nor wanted is not counted and not returned; an argument nothing
    wanted reads does not count either. The sample ``w``, when wanted, is returned as given.

    Parameters
    ----------
    kind: str
        ``'probe'`` (the only kind built so far).
    frame: tuple
        ``(D, U, L, R)``, each ``len=d``: ``D[i]`` of shape ``C+(nDi, Ni)``, ``U[i]`` ``C+(rLi, nUi, rR(i+1))``,
        ``L[i]`` ``C+(rLi, nDi, rL(i+1))``, ``R[i]`` ``C+(rRi, nDi, rR(i+1))``. The base point is ``(D, L)``.
        Or a corewise point ``(B, G)``, the cores of a Tucker tensor train: one basis on each edge, and the names
        without a basis (the module docstring).
    sample: sequence of NDArray
        The input jets ``ww``: ``len=d``, ``elm_shape=J_in_i+W+(Ni,)``, with ``len(order)`` leading jet axes.
    want: sequence of str
        The names wanted (the module's graph, and ``'w'``, the sample as given).
    order: tuple of int
        The jet shape. ``()`` for no jets.
    y: sequence of NDArray, optional
        The data, of the type of ``z``: ``len=d``, ``elm_shape=F+W+(Ni,)``. Needed for ``r`` and what reads it.
    weight: NDArray, optional
        ``(d,)+F``. ``None`` is ones.
    variations: tuple, optional
        The direction in the frame's form, ``(V, H)``: ``V[i]`` of shape ``K+C+(nUi, Ni)``, ``H[i]``
        ``K+C+(rLi, nDi, rR(i+1))``, the stack of directions ``K`` the same for all. Needed for the dot variables,
        ``Jp``, ``Jp_norm_sq`` and ``JtJp``.
    core_directions: tuple, optional
        The direction in the raw form at the base point, ``(D_dot, L_dot)``, with the shapes of ``D`` and ``L`` and the
        stack ``K`` in front (the same ``K`` as ``variations``). Needed for the raw dots, the tilde-dot variables and
        ``Kp``; on a corewise point, the only form of a direction.
    sum_over_samples
        Not read by the variables built so far.
    Vc: sequence of NDArray, optional
        The Tucker centers, ``len=d``, ``elm_shape=C+(nUi, Ni)``: ``z`` is then read off the frame through them,
        with no sweep through the base point's ``L`` cores.
    held: HeldVariables, optional
        Variables computed earlier at this frame and sample.

    Returns
    -------
    tuple
        One entry per name of ``want``: a tuple over the index for an indexed variable (``d + 1`` bonds, or ``d``
        sites or modes), an array for ``f`` and ``r_block_sumsq``.

    Raises
    ------
    ValueError
        For what the engine cannot run, as listed above.
    TypeError
        If ``order`` is not a tuple of integers, or ``held`` is not a ``HeldVariables``.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.sampling_variables as sv
    >>> from t3toolbox.backend.jets import affine_jet
    >>> rng = np.random.default_rng(0)
    >>> N, nD, nU, rL, rR = (4, 5, 6), (2, 3, 2), (3, 2, 3), (1, 2, 3, 1), (1, 3, 2, 1)
    >>> D = [rng.standard_normal((nD[i], N[i])) for i in range(3)]
    >>> U = [rng.standard_normal((rL[i], nU[i], rR[i + 1])) for i in range(3)]
    >>> L = [rng.standard_normal((rL[i], nD[i], rL[i + 1])) for i in range(3)]
    >>> R = [rng.standard_normal((rR[i], nD[i], rR[i + 1])) for i in range(3)]
    >>> frame = (D, U, L, R)
    >>> ww = [rng.standard_normal((7, n)) for n in N]                   # W = (7,)
    >>> z, mu = sv.sampling_variables('probe', frame, ww, ('z', 'mu'))
    >>> print([zi.shape for zi in z], [m.shape for m in mu])
    [(7, 4), (7, 5), (7, 6)] [(7, 1), (7, 2), (7, 3), (7, 1)]

    With jets along the line ``ww + s pp``, at the jet shape ``(3,)``, and the misfit against data:

    >>> pp = [rng.standard_normal((7, n)) for n in N]
    >>> y = [rng.standard_normal((4, 7, n)) for n in N]                  # F + W + (Ni,)
    >>> line = affine_jet(ww, pp)                                         # J_in = (2,)
    >>> z, xi, f = sv.sampling_variables('probe', frame, line, ('z', 'xi', 'f'), order=(3,), y=y)
    >>> print([zi.shape for zi in z], [x.shape for x in xi], f.shape)    # the leaves keep the line's two slots
    [(4, 7, 4), (4, 7, 5), (4, 7, 6)] [(2, 7, 2), (2, 7, 3), (2, 7, 2)] ()

    A held variable is used, not recomputed; ``r`` needs ``y`` or a held ``r``:

    >>> held = sv.HeldVariables(z=z)
    >>> (r,) = sv.sampling_variables('probe', frame, line, ('r',), order=(3,), y=y, held=held)
    >>> print(all(bool(np.allclose(ri, zi - yi)) for ri, zi, yi in zip(r, z, y)))
    True
    >>> sv.sampling_variables('probe', frame, ww, ('f',))              # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    ValueError: sampling_variables: 'r' needs the argument y, which was not given (needed through f -> r)
    """
    # 1. what the engine needs in order to run
    if kind not in _KINDS:
        raise ValueError('sampling_variables: unknown sampling kind %r; expected one of %s' % (kind, _KINDS))
    order = contractions._read_order(order, 'sampling_variables')
    if kind == 'entries' and order != ():
        raise ValueError('sampling_variables: the entries kind has no jets: its order is (), got %r' % (order,))
    if (kind, '_manifold') not in _GRAPHS:
        raise NotImplementedError('sampling_variables: the %s kind is not built yet' % kind)
    if isinstance(want, str):
        raise TypeError('sampling_variables: want is a sequence of names, not one name; got %r' % (want,))
    want = tuple(want)
    if held is not None and not isinstance(held, HeldVariables):
        raise TypeError('sampling_variables: held is a HeldVariables or None; got a %s' % type(held).__name__)

    # 2. the layout and the call type, from the structure of the frame
    if not isinstance(frame, (tuple, list)) or not frame:
        raise ValueError('sampling_variables: the frame is a tuple of stores; got a %s' % type(frame).__name__)
    if is_ndarray(frame[0]):
        raise NotImplementedError('sampling_variables: the uniform layout is not built yet')
    if len(frame) == 2:                                 # a corewise point (B, G): it binds the stores D and L
        call = '_corewise'
        stores = dict(zip(('D', 'L'), frame))
        if variations is not None:
            raise ValueError('sampling_variables: a corewise point has one form of a direction, the raw one: pass it '
                             'as core_directions, not variations')
    elif len(frame) == 4:
        call = '_manifold'
        stores = dict(zip(('D', 'U', 'L', 'R'), frame))
    else:
        raise ValueError('sampling_variables: a ragged frame has four stores (D, U, L, R), or a corewise point two '
                         '(B, G); got %d' % len(frame))
    graph = _GRAPHS[kind, call]
    names = _public(graph) + ('w',)
    unknown = [name for name in want if name not in names]
    if unknown:
        raise ValueError('sampling_variables: unknown variable %s for the %s kind on a %s call; the names are %s'
                         % (', '.join(map(repr, unknown)), kind, call.lstrip('_'), ', '.join(names)))
    setting = _read_setting(stores, sample, order, variations, graph, core_directions)
    if held is not None:
        _check_held(held, setting)

    # 3. the closure
    given = {'y': y, 'Vc': Vc, 'variations': variations, 'core_directions': core_directions}
    present = frozenset(name for name, value in given.items() if value is not None) | {call}
    held_names = frozenset(held.keys()) if held is not None else frozenset()
    rules = _closure(kind, tuple(name for name in want if name != 'w'), held_names, present)

    # 4. numpy or jax, and the result dtype over the arrays the closure reads; those are cast once
    held_values = {name: held[name] for name in held_names}
    use_jax = tree_contains_jax((frame, sample, y, weight, variations, core_directions, Vc,
                                 tuple(held_values.values())))
    xnp, xmap, xscan = get_backend(False, use_jax)
    V, H = variations if variations is not None else (None, None)
    D_dot, L_dot = core_directions if core_directions is not None else (None, None)
    inputs = dict(stores, w=sample, y=y, weight=weight, Vc=Vc, V=V, H=H, D_dot=D_dot, L_dot=L_dot)
    read = {name for rule in rules for name in rule.reads}
    held_counted = held_names & (read | set(want))         # a held variable read by a row, or wanted (an output)
    counted = [inputs[name] for name in sorted(read & set(inputs))] + [held_values[name] for name in
                                                                       sorted(held_counted)]
    dtype = _lowest_precision_dtype(xnp, *[leaf for value in counted for leaf in _leaves(value)])
    if dtype is None:
        dtype = np.dtype(xnp.result_type(float))
    values = {name: _cast(inputs[name], dtype, xnp) for name in read & set(inputs)}
    values.update({name: _cast(held_values[name], dtype, xnp) for name in held_counted})

    # 5. to 7. run the rows
    m = len(order)
    lens = contractions.GroupLengths(J=m, W=len(setting.W), K=len(setting.K), C=len(setting.C))
    lens_no_K = dc.replace(lens, K=0)
    form = 'loop'                                           # the ragged layout
    for rule in rules:
        values[rule.name] = _run(rule, values, lens if rule.carries_K else lens_no_K, setting, form, dtype,
                                 xnp, xmap, xscan)

    # 8. the return, aligned with want
    return tuple(sample if name == 'w' else values[name] for name in want)
