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
- ``start``: for a sweep, its start value: ``'ones'`` (the plain grade: the vector of ones that closes the end bond),
  ``'zeros'`` (a tilde, dot or tilde-dot sweep), or the name of a node, which times the vector of ones that closes the
  end bond (the apply kind's seed); ``None`` for the other drivers. A start node is not read per site, so it is not
  among the reads; what a row depends on is its reads and its start node (``_deps``), and every walk of the graph
  (the closure, the dtype count, the twins, the corewise derivation) reads both.

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
| `r_block_sumsq` | whole | `r_block_sumsq_step` | `r`, `weight` |  | no |  |
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
| `JtJp` | whole | `g_step` | `_Jp.V_tilde`, `_Jp.H_tilde` | `held:Jp` | yes |  |

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
``K``, the seed reading ``Jp`` for ``r``. ``JtJp`` is the twin of ``g``. The full ``eta_tilde`` has no twin. ``JtJp``
has a second route, from a held ``Jp`` with no direction (``held:Jp``, P12): the twin path is linear in ``Jp``, so
**holding any vector of** ``Jp``'s **shape as** ``Jp`` **gives** ``J^T`` **of that vector** (the seed is the data metric
of the held vector at ``gamma = 1``, ``weight**2`` times it, turned; with no weight, the vector itself). The transposes
take this route for ``c``, an element of ``Y`` with the shape of ``z`` (they pass no weight), with a stack ``K`` read
off the held ``Jp``; a direction, when given, wins, and its ``K`` must agree.

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

**The apply kind.** Its output is one number per sample and slot: the left side ``mu`` at bond ``d`` closed by the
vector of ones of its end bond (a sum: with tails the end bond is larger than one), and ``Jp`` likewise of the
direction's left side, ``mu_R_dot`` (closed by the ``R`` store's end bond) or the raw ``mu_dot``. The seed of the
adjoint is one number per sample and slot too, and it sits on those closing ones. So a right side or a trunk is the
plain recurrence started from the seed (``nu_tilde`` is the jet product of the seed and ``nu``), and a left side or a
leaf has no tilde: ``xi_tilde``, ``mu_tilde``, ``xi_tilde_dot`` and ``mu_tilde_dot`` are zero for apply, and are not
names of its graphs (asking for one raises, and the message says so). The tilde-dots are the dot recurrences with the
tilde forms in the plain ones' places, and the blocks have no seed at a leaf. The apply graphs are the plain and dot
blocks of rows shared with the probe kind (with ``r``, ``f``, ``r_block_sumsq``, ``w_tilde`` and ``Jp_norm_sq``,
whose whole steps take the kind) and the apply kind's own rows, its twins included:

| Apply's own row | Driver | Step | Reads | Needs | K | Start |
| --- | --- | --- | --- | --- | --- | --- |
| `z` | whole | `close_step` | `mu` |  | no |  |
| `nu_tilde` | right | `nu_step` | `R`, `xi` |  | no | w_tilde |
| `eta_tilde` | site | `eta_step` | `mu`, `U`, `nu_tilde` |  | no |  |
| `_eta_tilde_top` | site | `eta_top_step` | `mu`, `U`, `nu_tilde`, `w` |  | no |  |
| `nu_L_tilde` | right | `nu_step` | `L`, `xi` |  | no | w_tilde |
| `eta_D_tilde` | site | `eta_step` | `mu`, `L`, `nu_L_tilde` |  | no |  |
| `_eta_D_tilde_top` | site | `eta_top_step` | `mu`, `L`, `nu_L_tilde`, `w` |  | no |  |
| `V_tilde` | site | `V_top_step` | `_eta_D_tilde_top`, `w` | `_corewise` | no |  |
| `V_tilde` | site | `V_top_step` | `_eta_tilde_top`, `w` |  | no |  |
| `H_tilde` | site | `H_readout_step` | `mu`, `xi`, `nu_L_tilde` | `_corewise` | no |  |
| `H_tilde` | site | `H_readout_step` | `mu`, `xi`, `nu_tilde` |  | no |  |
| `g` | whole | `g_step` | `V_tilde`, `H_tilde` |  | no |  |
| `Jp` | whole | `close_step` | `mu_R_dot` | `variations` | yes |  |
| `Jp` | whole | `close_step` | `mu_dot` | `core_directions` | yes |  |
| `nu_L_tilde_dot` | right | `nu_dot_step` | `L`, `nu_L_tilde`, `xi`, `xi_dot`, `L`, `L_dot` | `core_directions` | yes | zeros |
| `eta_D_tilde_dot` | site | `eta_dot_step` | `mu_dot`, `L`, `mu`, `L_dot`, `L`, `nu_L_tilde_dot`, `nu_L_tilde` | `core_directions` | yes |  |
| `_eta_D_tilde_dot_top` | site | `eta_dot_top_step` | `mu_dot`, `L`, `mu`, `L_dot`, `L`, `nu_L_tilde_dot`, `nu_L_tilde`, `w` | `core_directions` | yes |  |
| `V_tilde_dot` | site | `V_top_step` | `_eta_D_tilde_dot_top`, `w` | `core_directions` | yes |  |
| `H_tilde_dot` | site | `H_tilde_step` | `mu_dot`, `mu`, `xi_dot`, `xi`, `nu_L_tilde_dot`, `nu_L_tilde` | `core_directions` | yes |  |
| `Kp` | whole | `g_step` | `V_tilde_dot`, `H_tilde_dot` | `core_directions` | yes |  |
| `_Jp.w_tilde` | whole | `w_tilde_step` | `Jp`, `weight` |  | yes |  |
| `_Jp.nu_tilde` | right | `nu_step` | `R`, `xi` |  | yes | _Jp.w_tilde |
| `_Jp.eta_tilde_top` | site | `eta_top_step` | `mu`, `U`, `_Jp.nu_tilde`, `w` |  | yes |  |
| `_Jp.V_tilde` | site | `V_top_step` | `_Jp.eta_tilde_top`, `w` |  | yes |  |
| `_Jp.H_tilde` | site | `H_readout_step` | `mu`, `xi`, `_Jp.nu_tilde` |  | yes |  |
| `JtJp` | whole | `g_step` | `_Jp.V_tilde`, `_Jp.H_tilde` | `variations` | yes |  |
| `JtJp` | whole | `g_step` | `_Jp.V_tilde`, `_Jp.H_tilde` | `held:Jp` | yes |  |

The corewise point of the apply kind is derived from these rows in the same way as the probe kind's.

**The entries kind** is the apply kind at the unit vectors ``e_index``, with no jets (``order = ()`` only). Its sample
is the integer ``index``, ``(d,) + W``, which is neither counted for the dtype nor cast. Its rows are derived from the
apply kind's by one declared table of steps:

| A step of apply | Its step for entries | What it does |
| --- | --- | --- |
| `xi_step` | `xi_gather_step` | the leaf: the columns of ``D`` at ``index`` (a slice, not a product) |
| `xi_dot_step` | `xi_gather_step` | ``xi_U_dot`` and ``xi_dot``: the columns of ``V`` and ``D_dot`` |
| `V_top_step` | `V_scatter_step` | a block of a Tucker core's shape: a scatter-add into the columns ``index`` |

Every read of ``w`` is a read of ``index``, and the scatter also reads ``D`` for the mode size. ``w`` itself is the
entries kind's own row: the one-hot vectors ``e_index`` (``w_one_hot_step``, reading ``index`` and ``D``), made only
when wanted, in the result dtype; for probe and apply ``'w'`` is the sample as given. At ``order = ()`` the
short form of a trunk's tilde is the whole trunk, so the short nodes go and the blocks read the full ``eta_tilde``,
``eta_D_tilde`` and ``eta_D_tilde_dot`` (its path of the adjoint seeded by ``Jp`` has the full ``eta_tilde``). The
scatter-add is ``W x n`` in memory, not the one-hot product's ``W x N``, and adds duplicate indices up. Its zero
names, its corewise point and its shapes (``z``, ``r``, ``w_tilde`` and ``Jp`` are ``W + [K] + C``,
``r_block_sumsq`` a number) are the apply kind's.

**The short form of** ``eta_tilde``. ``V_tilde`` reads only the slots of ``eta_tilde`` that meet the input at the
corner: the top ``a = min(F_in_i, F)`` slots on every jet axis (two for a line). The private node ``_eta_tilde_top``
holds those slots, ``eta_tilde[i][c - a + 1 : c + 1]`` (``c`` the corner), each made by readouts at a lowered corner,
and ``V_tilde`` always reads it. The full ``eta_tilde`` has its own row, run only when it is wanted, so what ``g`` is
never depends on whether ``eta_tilde`` is wanted. ``eta_D_tilde`` (read by the corewise gradient) and
``eta_D_tilde_dot`` (read by ``V_tilde_dot``) have short nodes in the same way.

**The shapes** (ragged layout): a variable at index ``i`` is
``P + W + C + (size,)``, with ``P`` its jet axes (of the declared sizes below), ``W`` the sample stack and ``C`` the
stack of base points; a block (no jet sizes) is ``C`` and the shape of the core it stands for, summed over the
samples (with ``sum_over_samples=False``, ``W`` first: ``W + [K] + C`` and the core's shape).
``F = tuple(n + 1 for n in order)`` and ``F_in_i`` are the jet sizes of ``w[i]``. The tilde variables are dual jets,
stored with every jet axis turned. For the apply kind ``z``, ``r``, ``w_tilde`` and ``Jp`` are one array each,
``F + W + [K] + C``, with no index and no mode edge, and ``r_block_sumsq`` has the block shape ``F``.

| Variable | Index | Jet sizes | Size |
| --- | --- | --- | --- |
| `xi` | site | min(F_in_i, F) | nD_i |
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
| `_eta_tilde_top` | site | min(F_in_i, F) | nU_i |
| `V_tilde` | site | none | (nU_i, N_i) |
| `H_tilde` | site | none | (rL_i, nD_i, rR_(i+1)) |
| `xi_U_dot` | site | min(F_in_i, F) | nU_i |
| `mu_R_dot` | bond | F | rR_i |
| `nu_L_dot` | bond | F | rL_i |
| `eta_D_dot` | site | F | nD_i |
| `Jp` | mode | F | N_i |
| `xi_dot` | site | min(F_in_i, F) | nD_i |
| `mu_dot` | bond | F | rL_i |
| `nu_L_tilde` | bond | F | rL_i |
| `eta_D_tilde` | site | F | nD_i |
| `_eta_D_tilde_top` | site | min(F_in_i, F) | nD_i |
| `xi_tilde_dot` | site | F | nD_i |
| `mu_tilde_dot` | bond | F | rL_i |
| `nu_L_tilde_dot` | bond | F | rL_i |
| `eta_D_tilde_dot` | site | F | nD_i |
| `_eta_D_tilde_dot_top` | site | min(F_in_i, F) | nD_i |
| `V_tilde_dot` | site | none | (nD_i, N_i) |
| `H_tilde_dot` | site | none | (rL_i, nD_i, rL_(i+1)) |
| `nu_dot` | bond | F | rL_i |
| `eta_dot` | site | F | nD_i |
| `nu_tilde_dot` | bond | F | rL_i |
| `eta_tilde_dot` | site | F | nD_i |
| `_eta_tilde_dot_top` | site | min(F_in_i, F) | nD_i |

A variable made by a row with ``K = yes`` has ``K`` after ``W`` (``P + W + K + C + (size,)``; a block ``K + C`` and the
core's shape); a private node's shape is its base's. ``f`` has the shape ``C``; ``r_block_sumsq``, ``<r, M r>`` per
block (weighted as ``f`` is), the block shape ``(d,) + F`` (``F`` for apply); ``Jp_norm_sq`` the shape ``K + C``;
``g`` is the pair ``(V_tilde, H_tilde)``, ``JtJp`` the pair of their twins and ``Kp`` the pair ``(V_tilde_dot,
H_tilde_dot)``. On a corewise point the sizes of ``R`` and ``U`` are those of ``L`` and ``D``. A bond index runs over
``0 .. d`` (``d + 1`` arrays), a site or a mode index over ``0 .. d-1``.

**What a call does, in order**: it checks what it needs to run, and nothing more; reads the
layout and the call type from the structure of the frame; takes the closure of what is wanted over the graph, the
held variables its leaves; chooses numpy or jax over every array input and the result dtype, the lowest floating
precision among the arrays the closure reads and the held variables read or wanted, and casts those once; runs the
rows; returns a tuple aligned with ``want``. Every scan and map body is module-level, memoized on ``(step_name, lens,
order, form)`` and, for a block row, its ``keep_W``: a body closes over the record, the box and the form, nothing per
site.

**The stacks.** ``W`` is read once off ``w[0]`` (after its jet axes; for entries off ``index``), ``C`` off ``D[0]`` and
``K`` off ``V[0]`` or ``D_dot[0]``; the steps see only their lengths (``GroupLengths``), and every variable is laid out
``P + W + [K] + C`` and its edge, a block ``[W] + [K] + C`` and its core. Every other operand that carries a stack must
have exactly it (``_check_stacks`` for ``W`` and ``C``, ``_read_K`` for ``K``: tuple equality, not broadcast
compatibility), since a size-1 axis would broadcast silently. With ``sum_over_samples=False`` the block rows (read off
the shape table) keep ``W``: ``g``, ``JtJp`` and ``Kp`` are then ``W + [K] + C``; ``f``, ``r_block_sumsq`` and
``Jp_norm_sq`` are sums, and stay summed over ``W``.

**The uniform layout.** A uniform frame is ``(D, U, L, R, shape, masks)``, today's uniform frame ``.data``, and a
uniform point ``(B, G, shape, masks)``, ``UniformTuckerTensorTrain.data`` (the layout first, by ``is_ndarray``, then the
count). The supercores have the site index ``d`` leading; the masks and ``shape`` are host values, static: a traced
mask raises (under ``jit``, pass the frame inside its object, where the masks and ``shape`` are ``aux_data``, or close
over them). The engine
masks once on entry, every input with the masks of the object it is defined at (``_UNIFORM_MASKS``): the stores with
their own; ``V``, ``H`` and ``Vc`` with the frame's variation masks; ``D_dot`` and ``L_dot`` with the frame's ``D`` and
``L`` masks, never the point's (a superset at TT slack); every Tucker-shaped input, the sample and probe's ``y``
with the shape mask on the mode axis ``N`` (the sample's jet axes untouched: their slots are coefficients; a ``y``
with an edge of 1 is broadcast to ``N`` by its mask, so that its value cannot reach the padding of ``r``), and probe's
weight when its edge is ``N`` (an edge of 1 has no padding, and multiplies padding that is zero); a held
variable with its table's masks (padding is not a value). The start values carry ones on the real slots of their end
bond only. The sample is packed (``pack_if_ragged`` with ``len_P=len(order)``: the modes' jet axes brought to their
largest size by zero coefficients, which are data). The entries kind's numpy index outside ``[-N_i, N_i)`` raises an
``IndexError`` as on ragged, and a negative one counts from ``N_i``; a jax index is not checked (out of range it reads
a zero column). The steps see no mask: the site rows go through ``xmap`` and the sweeps through ``xscan`` over the
leading ``d``, the terms with ``over_orders='scan'``; the whole steps act on the representation they are given. On
exit every output but a ``'w'`` the caller passed (returned as given, its padding included) is masked by its table
(``_mask_variable``): the variables and the blocks have exactly zero padding.
The masks are layered (each family alone keeps the real parts right when every other operand is masked). A uniform
call returns the uniform layout: a variable is one array, its index leading (``(d,)`` or ``(d + 1,)``); mirroring a
ragged sample's form is the business of the functions built on the engine.

The three kinds are built here, on both layouts, on a manifold frame and on a corewise point.
"""
import dataclasses as dc
import functools
import typing as typ

import numpy as np

import t3toolbox.backend.contractions as contractions
import t3toolbox.backend.sampling_steps as sampling_steps
import t3toolbox.backend.ut3_operations as ut3_operations
from t3toolbox.backend.common import *
from t3toolbox.backend.jets import _lowest_precision_dtype

if jax_available:
    import jax

__all__ = [
    'sampling_variables',
    'HeldVariables',
    'frame_form',
    'frame_shape',
    'computed_names',
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
_STARTS = ('ones', 'zeros')                 # the start values by name; any other start names a node
_HELD_JP = 'held:Jp'                        # the fact of a held Jp, in present for a held Jp only (P12)

# The rows split into blocks: the plain and dot grades depend only on the frame and the inputs, so every kind shares
# them; the outputs, the seed's adjoint, the direction's output and the curvature are each kind's own. The residual's
# whole rows are shared too: their steps take the kind as a static.
_PLAIN_ROWS = (
    Rule('xi',             ('site', 'xi_step'),             ('D', 'w'),                    _NONE,             False),
    Rule('mu',             ('left', 'mu_step'),             ('L', 'xi'),                   _NONE,             False, 'ones'),
    Rule('nu',             ('right', 'nu_step'),            ('R', 'xi'),                   _NONE,             False, 'ones'),
    Rule('nu_L',           ('right', 'nu_step'),            ('L', 'xi'),                   _NONE,             False, 'ones'),
    Rule('eta',            ('site', 'eta_step'),            ('mu', 'U', 'nu'),             _NONE,             False),
    Rule('eta_D',          ('site', 'eta_step'),            ('mu', 'L', 'nu_L'),           _NONE,             False),
)

_RESIDUAL_ROWS = (
    Rule('r',              ('whole', 'r_step'),             ('z', 'y'),                    frozenset({'y'}),  False),
    Rule('f',              ('whole', 'f_step'),             ('r', 'weight'),               _NONE,             False),
    Rule('r_block_sumsq',  ('whole', 'r_block_sumsq_step'), ('r', 'weight'),               _NONE,             False),
    Rule('w_tilde',        ('whole', 'w_tilde_step'),       ('r', 'weight'),               _NONE,             False),
)

_DOT_ROWS = (
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
)

_DOT_SUM_ROWS = (
    Rule('Jp_norm_sq',     ('whole', 'Jp_norm_sq_step'),    ('Jp', 'weight'),              _NONE,             True),
)

_PROBE_Z_ROWS = (
    Rule('z',              ('site', 'z_step'),              ('eta', 'Vc'),                 frozenset({'Vc'}), False),
    Rule('z',              ('site', 'z_step'),              ('eta_D', 'D'),                _NONE,             False),
)

_PROBE_ADJOINT_ROWS = (
    # the tilde grade: the residual's adjoint
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
)

_PROBE_JP_ROWS = (
    # the direction's output: through the frame's form when variations is given, else the raw form at the base point
    Rule('Jp',             ('site', 'Jp_step'),             ('_frame.eta_D_dot', 'D', 'eta', 'V'),
                                                                                           _VAR,              True),
    Rule('Jp',             ('site', 'Jp_step'),             ('_raw.eta_D_dot', 'D', 'eta_D', 'D_dot'),
                                                                                           _COR,              True),
)

_PROBE_CURVATURE_ROWS = (
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

_PROBE_ROWS = (_PLAIN_ROWS + _PROBE_Z_ROWS + _RESIDUAL_ROWS + _PROBE_ADJOINT_ROWS + _DOT_ROWS + _PROBE_JP_ROWS
               + _DOT_SUM_ROWS + _PROBE_CURVATURE_ROWS)

# The apply kind: its output is the left side at bond d closed by the ones of the end bond, and the seed of its
# adjoint sits on those ones. So a right side or a trunk is the plain recurrence started from the seed (a start that
# names a node), and a left side or a leaf has no tilde (xi_tilde, mu_tilde and their tilde-dots are zero, and are
# not names of the kind: _ZERO_NAMES). The tilde-dots are the dot recurrences with the tilde forms in the plain ones'
# places; the blocks have no seed at a leaf.
_APPLY_Z_ROWS = (
    Rule('z',              ('whole', 'close_step'),         ('mu',),                       _NONE,             False),
)

_APPLY_ADJOINT_ROWS = (
    Rule('nu_tilde',       ('right', 'nu_step'),            ('R', 'xi'),                   _NONE,             False, 'w_tilde'),
    Rule('eta_tilde',      ('site', 'eta_step'),            ('mu', 'U', 'nu_tilde'),       _NONE,             False),
    Rule('_eta_tilde_top', ('site', 'eta_top_step'),        ('mu', 'U', 'nu_tilde', 'w'),  _NONE,             False),
    Rule('nu_L_tilde',     ('right', 'nu_step'),            ('L', 'xi'),                   _NONE,             False, 'w_tilde'),
    Rule('eta_D_tilde',    ('site', 'eta_step'),            ('mu', 'L', 'nu_L_tilde'),     _NONE,             False),
    Rule('_eta_D_tilde_top', ('site', 'eta_top_step'),      ('mu', 'L', 'nu_L_tilde', 'w'), _NONE,            False),
    Rule('V_tilde',        ('site', 'V_top_step'),          ('_eta_D_tilde_top', 'w'),     _COREWISE,         False),
    Rule('V_tilde',        ('site', 'V_top_step'),          ('_eta_tilde_top', 'w'),       _NONE,             False),
    Rule('H_tilde',        ('site', 'H_readout_step'),      ('mu', 'xi', 'nu_L_tilde'),    _COREWISE,         False),
    Rule('H_tilde',        ('site', 'H_readout_step'),      ('mu', 'xi', 'nu_tilde'),      _NONE,             False),
    Rule('g',              ('whole', 'g_step'),             ('V_tilde', 'H_tilde'),        _NONE,             False),
)

_APPLY_JP_ROWS = (
    # the direction's output: mu_R_dot (or the raw mu_dot) at bond d, closed by the ones of its end bond
    Rule('Jp',             ('whole', 'close_step'),         ('mu_R_dot',),                 _VAR,              True),
    Rule('Jp',             ('whole', 'close_step'),         ('mu_dot',),                   _COR,              True),
)

_APPLY_CURVATURE_ROWS = (
    Rule('nu_L_tilde_dot', ('right', 'nu_dot_step'),        ('L', 'nu_L_tilde', 'xi', 'xi_dot', 'L', 'L_dot'),
                                                                                           _COR,              True, 'zeros'),
    Rule('eta_D_tilde_dot', ('site', 'eta_dot_step'),       ('mu_dot', 'L', 'mu', 'L_dot', 'L', 'nu_L_tilde_dot',
                                                             'nu_L_tilde'),                _COR,              True),
    Rule('_eta_D_tilde_dot_top', ('site', 'eta_dot_top_step'),
                                                            ('mu_dot', 'L', 'mu', 'L_dot', 'L', 'nu_L_tilde_dot',
                                                             'nu_L_tilde', 'w'),           _COR,              True),
    Rule('V_tilde_dot',    ('site', 'V_top_step'),          ('_eta_D_tilde_dot_top', 'w'), _COR,              True),
    Rule('H_tilde_dot',    ('site', 'H_tilde_step'),        ('mu_dot', 'mu', 'xi_dot', 'xi', 'nu_L_tilde_dot',
                                                             'nu_L_tilde'),                _COR,              True),
    Rule('Kp',             ('whole', 'g_step'),             ('V_tilde_dot', 'H_tilde_dot'), _COR,             True),
)

_APPLY_ROWS = (_PLAIN_ROWS + _APPLY_Z_ROWS + _RESIDUAL_ROWS + _APPLY_ADJOINT_ROWS + _DOT_ROWS + _APPLY_JP_ROWS
               + _DOT_SUM_ROWS + _APPLY_CURVATURE_ROWS)

# The entries kind: the apply kind at the unit vectors e_index, with no jets. Its rows are derived from apply's by one
# declared table of steps: the leaves are gathers (a slice of the core at index, not a product) and the blocks of a
# Tucker core's shape scatter-adds (into the columns index; the step reads D for the mode size); the input w is read
# as index; and at order () the short form of a trunk's tilde is the whole trunk, so the short nodes go and their
# readers read the full node.
_ENTRIES_STEPS = {'xi_step': 'xi_gather_step', 'xi_dot_step': 'xi_gather_step', 'V_top_step': 'V_scatter_step'}
_ENTRIES_SIZED = frozenset({'V_scatter_step'})          # a step that also reads D, for the mode size N
_ENTRIES_FULL = {'_eta_tilde_top': 'eta_tilde', '_eta_D_tilde_top': 'eta_D_tilde',
                 '_eta_D_tilde_dot_top': 'eta_D_tilde_dot'}


def _entries(rows: typ.Tuple[Rule, ...]) -> typ.Tuple[Rule, ...]:
    '''The entries kind's rows, derived from the apply kind's by the table of steps. Its invariants raise a
    ``RuntimeError``: every row that reads the input has its step in the table, and no derived row reads ``w`` or a
    short node.'''
    out = []
    for rule in rows:
        if rule.name in _ENTRIES_FULL:                  # a short node: at order () it is the full trunk
            continue
        reads = tuple(_ENTRIES_FULL.get(read, read) for read in rule.reads)
        step = rule.step
        if 'w' in reads:
            if step[1] not in _ENTRIES_STEPS:
                raise RuntimeError('the entries kind has no step for %r, which reads the input' % (step[1],))
            step = (step[0], _ENTRIES_STEPS[step[1]])
            reads = tuple('index' if read == 'w' else read for read in reads)
            reads += ('D',) if step[1] in _ENTRIES_SIZED else ()
        out.append(rule._replace(step=step, reads=reads))
    for rule in out:
        if set(_deps(rule)) & ({'w'} | set(_ENTRIES_FULL)):
            raise RuntimeError('the entries row %r reads the input jets or a short node' % (rule.name,))
    return tuple(out)


# The names a kind does not have, though the definitions give them, and why (the message of an unknown name says so):
# for apply and entries the seed is on bond d, which a left side or a leaf never reaches, so their tilde variables
# are zero.
_ZERO_NAMES = {
    'apply': frozenset({'xi_tilde', 'mu_tilde', 'xi_tilde_dot', 'mu_tilde_dot'}),
    'entries': frozenset({'xi_tilde', 'mu_tilde', 'xi_tilde_dot', 'mu_tilde_dot'}),
}
_ZERO_REASON = {
    'apply': 'the seed sits on the ones that close bond d, which a left side or a leaf never reaches',
    'entries': 'the seed sits on the ones that close bond d, which a left side or a leaf never reaches',
}


def _made(graph: typ.Tuple[Rule, ...]) -> typ.Tuple[str, ...]:
    '''The variables a graph makes, each once, in the order of their first row; private nodes included.'''
    return tuple(dict.fromkeys(rule.name for rule in graph))


def _public(graph: typ.Tuple[Rule, ...]) -> typ.Tuple[str, ...]:
    '''The public variables a graph makes: those that can be wanted and held.'''
    return tuple(name for name in _made(graph) if not name.startswith('_'))


def _deps(rule: Rule) -> typ.Tuple[str, ...]:
    '''What a row depends on: its reads, and its start when the start names a node (a start node is not read per
    site, so it is not among the reads). Every walk of the graph reads it: the closure, the dtype count, the twins and
    the corewise derivation.'''
    return rule.reads + ((rule.start,) if rule.start is not None and rule.start not in _STARTS else ())


# The adjoint seeded by Jp (row 3.3; JtJp): a private twin of every row on the path from the seed to the blocks,
# named twin(name), with the same step and the call's K, its start renamed when it names a node on the path; JtJp is
# the twin of g. No twin of the full eta_tilde (nothing reads it, and a private name cannot be wanted). One table of
# the paths, keyed like _GRAPHS, each in its graph's own names (on a corewise point, the names after the rename).
_JP_PATHS = {
    ('probe', '_manifold'): ('w_tilde', 'xi_tilde', 'mu_tilde', 'nu_tilde', '_eta_tilde_top', 'V_tilde', 'H_tilde'),
    ('probe', '_corewise'): ('w_tilde', 'xi_tilde', 'mu_tilde', 'nu_tilde', '_eta_tilde_top', 'V_tilde', 'H_tilde'),
    ('apply', '_manifold'): ('w_tilde', 'nu_tilde', '_eta_tilde_top', 'V_tilde', 'H_tilde'),
    ('apply', '_corewise'): ('w_tilde', 'nu_tilde', '_eta_tilde_top', 'V_tilde', 'H_tilde'),
    ('entries', '_manifold'): ('w_tilde', 'nu_tilde', 'eta_tilde', 'V_tilde', 'H_tilde'),
    ('entries', '_corewise'): ('w_tilde', 'nu_tilde', 'eta_tilde', 'V_tilde', 'H_tilde'),
}


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
    ``_twin`` (and a start node on it), the seed reading ``Jp`` for ``r``, the call's ``K``; and ``JtJp``, the twin of
    ``g``, which needs the direction itself (so that a call without it names ``JtJp``, not a chain of private
    nodes).'''
    rename = lambda read: _twin(read) if read in path else ('Jp' if read == 'r' else read)
    out, jtjp = [], None
    other = '_corewise' if call == '_manifold' else '_manifold'
    for rule in graph:
        if other in rule.needs:                         # a route of the other call type has no twin here
            continue
        if rule.name in path:
            start = rename(rule.start) if rule.start is not None and rule.start not in _STARTS else rule.start
            out.append(rule._replace(name=_twin(rule.name), reads=tuple(map(rename, rule.reads)), carries_K=True,
                                     start=start))
        elif rule.name == 'g':
            jtjp = rule._replace(name='JtJp', reads=tuple(map(rename, rule.reads)), carries_K=True,
                                 needs=rule.needs | {direction})
    # a second route: from a held Jp, with no direction (P12). The twin path is linear in Jp, so holding any vector of
    # Jp's type as Jp gives J^T of that vector: the transposes' route for a residual with a stack K
    return tuple(out) + (jtjp, jtjp._replace(needs=(jtjp.needs - {direction}) | {_HELD_JP}))


# The corewise point (B, G): one basis on each edge. Its graph is derived from the manifold rows of its kind: every
# row that reads (or starts from) the frame's other stores (U, R), the frame's form of a direction (V, H), the Tucker
# centers or a frame node is dropped, with every row that depends on a dropped name; the rest are renamed, in their
# names, reads, starts AND needs, by one table (the base point's basis is the only one, so its names lose their basis;
# the direction is core_directions); then the twins are derived from the result.
_COREWISE_RENAME = {
    'nu_L': 'nu', 'eta_D': 'eta', 'nu_L_tilde': 'nu_tilde', 'eta_D_tilde': 'eta_tilde',
    'nu_L_dot': 'nu_dot', 'eta_D_dot': 'eta_dot', 'nu_L_tilde_dot': 'nu_tilde_dot', 'eta_D_tilde_dot': 'eta_tilde_dot',
    '_raw.nu_L_dot': '_raw.nu_dot', '_raw.eta_D_dot': '_raw.eta_dot',
    '_eta_D_tilde_top': '_eta_tilde_top', '_eta_D_tilde_dot_top': '_eta_tilde_dot_top',
    'variations': 'core_directions',
}
_COREWISE_DROP = frozenset({'U', 'R', 'V', 'H', 'Vc'})


def _corewise(
        rows:  typ.Tuple[Rule, ...],        # a kind's manifold rows, without their twins
        kind:  str,                         # whose path of the adjoint seeded by Jp the twins follow
) -> typ.Tuple[Rule, ...]:
    '''The graph of a corewise call, derived from a kind's manifold rows. Its invariants raise a ``RuntimeError``
    (never an ``assert``, which ``python -O`` removes).'''
    names = {r.name for r in rows}
    dropped_rows = set()
    while True:
        alive = {rule.name for k, rule in enumerate(rows) if k not in dropped_rows}
        more = {k for k, rule in enumerate(rows) if k not in dropped_rows and (
            rule.name.startswith('_frame.') or '_manifold' in rule.needs or set(_deps(rule)) & _COREWISE_DROP
            or any(dep not in alive and dep in names for dep in _deps(rule)))}
        if not more:
            break
        dropped_rows |= more
    kept = [rule for k, rule in enumerate(rows) if k not in dropped_rows]
    rename = lambda name: _COREWISE_RENAME.get(name, name)
    unrenamed = {rule.name for rule in kept if rule.name not in _COREWISE_RENAME}
    for rule in kept:                                    # a rename never lands on a name a surviving row keeps
        if rule.name in _COREWISE_RENAME and rename(rule.name) in unrenamed:
            raise RuntimeError('the corewise rename of %r lands on the surviving %r' % (rule.name, rename(rule.name)))
    renamed = tuple(rule._replace(name=rename(rule.name), reads=tuple(map(rename, rule.reads)),
                                  needs=frozenset(map(rename, rule.needs)),
                                  start=None if rule.start is None else rename(rule.start)) for rule in kept)
    path = _JP_PATHS[kind, '_corewise']
    if not set(path) <= {rule.name for rule in renamed}:
        raise RuntimeError('the corewise path of the adjoint seeded by Jp of the %s kind is incomplete' % kind)
    return renamed + _jp_twins(renamed, path, 'core_directions', '_corewise')


def _manifold(
        rows:  typ.Tuple[Rule, ...],        # a kind's manifold rows, without their twins
        kind:  str,
) -> typ.Tuple[Rule, ...]:
    '''The graph of a manifold call: the rows and the twins along the kind's path.'''
    path = _JP_PATHS[kind, '_manifold']
    if not set(path) <= {rule.name for rule in rows}:
        raise RuntimeError('the path of the adjoint seeded by Jp of the %s kind is incomplete' % kind)
    return rows + _jp_twins(rows, path, 'variations', '_manifold')


# The entries kind's own row: its input vectors w, the one-hot vectors of index by their definition (for the probe and
# apply kinds the sample is the input vectors, so 'w' is the sample as given and has no row).
_ENTRIES_OWN_ROWS = (
    Rule('w',              ('whole', 'w_one_hot_step'),     ('index', 'D'),                _NONE,             False),
)
_ENTRIES_ROWS = _ENTRIES_OWN_ROWS + _entries(_APPLY_ROWS)
_ROWS = {'probe': _PROBE_ROWS, 'apply': _APPLY_ROWS, 'entries': _ENTRIES_ROWS}
_GRAPHS = {}
for _kind, _rows in _ROWS.items():
    _GRAPHS[_kind, '_manifold'] = _manifold(_rows, _kind)
    _GRAPHS[_kind, '_corewise'] = _corewise(_rows, _kind)
    for _call in ('_manifold', '_corewise'):         # a zero name is not a name of the kind's graphs
        _clash = _ZERO_NAMES.get(_kind, frozenset()) & set(_made(_GRAPHS[_kind, _call]))
        if _clash:
            raise RuntimeError('the %s kind makes %s, declared zero for it' % (_kind, sorted(_clash)))
del _kind, _rows, _call, _clash
_PROBE_GRAPH = _GRAPHS['probe', '_manifold']
_PROBE_COREWISE_GRAPH = _GRAPHS['probe', '_corewise']
_KINDS = ('probe', 'apply', 'entries')

_STORES = frozenset({'D', 'U', 'L', 'R'})                   # the stores of a manifold frame
_ARGUMENTS = frozenset({'w', 'index', 'y', 'weight', 'Vc', 'V', 'H', 'D_dot', 'L_dot'})   # what a row may read but
                                                    # the stores: 'w' the sample (input jets) or 'index' (the entries
                                                    # kind's grid points); V, H the variations; D_dot, L_dot the core
                                                    # directions
_INTEGER_ARGUMENTS = frozenset({'index'})           # integer: it indexes; never counted for the dtype, never cast

# The direction operands and the masks P11 masks each with on the uniform layout: the frame's variation masks for the
# frame's form and the Tucker centers, the store's own masks for a core direction (the frame's D and L, never the
# point's, which are a superset at TT slack).
_DIRECTION_MASKS = {'V': 'variations', 'H': 'variations', 'Vc': 'variations', 'D_dot': 'D', 'L_dot': 'L'}
_DRIVERS = ('site', 'left', 'right', 'whole')

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

# What differs by kind: the type of z. The apply kind's output, its residual, its seed and its direction's output are
# one array each (the index 'one': no index and no mode edge), P+W+[K]+C.
_KIND_SHAPES = {
    'apply': {'z': ('one', 'F', None), 'r': ('one', 'F', None), 'w_tilde': ('one', 'F', None),
              'Jp': ('one', 'F', None)},
    'entries': {'z': ('one', 'F', None), 'r': ('one', 'F', None), 'w_tilde': ('one', 'F', None),
                'Jp': ('one', 'F', None)},
}


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
            missing = sorted({fact for rule in rows[name] for fact in rule.needs - present
                              if fact != _HELD_JP})     # a held Jp is not an argument to name
            through = (' (needed through %s)' % ' -> '.join(chain + (name,))) if chain else ''
            raise ValueError('sampling_variables: %r needs the argument %s, which was not given%s'
                             % (name, ' or '.join(missing), through))
        for dep in _deps(rule):
            visit(dep, chain + (name,))
        done.add(name)
        out.append(rule)

    for name in want:
        visit(name, ())
    return tuple(out)


# Every public name of a corewise point is reachable there, with the direction as core_directions and data (a missing
# row raises from the closure, at import).
for _kind in _ROWS:
    for _name in _public(_GRAPHS[_kind, '_corewise']):
        _closure(_kind, (_name,), frozenset(), frozenset({'_corewise', 'y', 'core_directions'}))
del _kind, _name


def _names(
        kind:  str,                         # 'probe' | 'apply' | 'entries'
        call:  str,                         # '_manifold' | '_corewise'
) -> typ.Tuple[str, ...]:
    '''The names a call of ``kind`` on that call type accepts in ``want``: the public variables of its graph, in the
    order of their first row, and ``'w'`` (a row of the entries kind's graph; for the other kinds the sample as
    given). (Private for now: a public reader is a new name, for the maintainer.)'''
    public = _public(_GRAPHS[kind, call])
    return public if 'w' in public else public + ('w',)


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

    It checks no shape: the engine checks every held variable it is handed, at every call, against the shapes of its
    table, and trusts the values. **One contract on a value:** a held ``Jp`` need not be the output of a direction.
    ``JtJp`` is linear in ``Jp``, so holding any vector ``c`` of ``Jp``'s shape as ``Jp`` (with no direction) gives
    ``J^T`` of ``weight**2 c``: this is how the transposes compute ``J^T c`` (they pass no weight), ``K`` read off
    ``c``. Where jax is importable it is a registered pytree: the filled slots are the leaves, the set of filled slots
    is the static structure. It refuses ``==`` and is unhashable, as every array-carrying class of the library is:
    compare the slots you mean.

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
    xi:             typ.Optional[typ.Sequence[NDArray]] = None      # len=d,   elm_shape=min(F_in_i,F)+W+C+(nDi,)
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
    xi_U_dot:       typ.Optional[typ.Sequence[NDArray]] = None      # len=d,   elm_shape=min(F_in_i,F)+W+K+C+(nUi,)
    mu_R_dot:       typ.Optional[typ.Sequence[NDArray]] = None      # len=d+1, elm_shape=F+W+K+C+(rRi,)
    xi_dot:         typ.Optional[typ.Sequence[NDArray]] = None      # len=d,   elm_shape=min(F_in_i,F)+W+K+C+(nDi,)
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
    F_in:   typ.Tuple[typ.Tuple[int, ...], ...]     # the jet sizes of each input, len=d
    W:      typ.Tuple[int, ...]
    C:      typ.Tuple[int, ...]
    sizes:  typ.Dict[str, typ.Tuple[int, ...]]      # 'rL', 'rR' (len=d+1), 'nD', 'nU', 'N' (len=d)
    K:      typ.Tuple[int, ...] = ()        # the stack of directions, read off variations
    with_K: typ.FrozenSet[str] = frozenset()        # the variables whose rows carry K
    kind:   str = 'probe'
    keep_W: bool = False                    # sum_over_samples=False: a block keeps W, W+[K]+C+core
    masks:  typ.Optional[typ.Dict[str, NDArray]] = None    # the uniform layout's mask families; None on ragged


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
        return None
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
        kind:             str,
        stores:           typ.Dict[str, typ.Sequence[NDArray]],   # D, U, L, R; or on a corewise point D, L only
        sample:           typ.Any,                  # the input jets; or the entries kind's index, (d,)+W
        order:            typ.Tuple[int, ...],
        variations:       typ.Optional[typ.Tuple] = None,
        graph:            typ.Tuple[Rule, ...] = (),
        core_directions:  typ.Optional[typ.Tuple] = None,
        keep_W:           bool = False,
        masks:            typ.Optional[typ.Dict[str, NDArray]] = None,
        held_Jp:          typ.Any = None,           # a held Jp: K is read off it when no direction gives one
) -> _Setting:
    D, L = stores['D'], stores['L']
    U, R = stores.get('U', L), stores.get('R', L)       # a corewise point: one basis on each edge
    m = len(order)
    C = tuple(np.shape(D[0])[:-2])
    if kind == 'entries':                               # the grid points, (d,) + W: no jets
        F_in, W = ((),) * len(D), tuple(np.shape(sample)[1:])
    else:
        F_in, W = tuple(tuple(np.shape(w)[:m]) for w in sample), tuple(np.shape(sample[0])[m:-1])
    K = _read_K(variations, core_directions, C)
    if held_Jp is not None:                             # K + C after the jets and W, before the mode edge (probe)
        first = held_Jp[0] if kind == 'probe' else held_Jp
        shape = tuple(np.shape(first))
        held_K = shape[m + len(W):len(shape) - len(C) - (1 if kind == 'probe' else 0)]
        if K is None:
            K = held_K
        elif held_K != K:
            raise ValueError('sampling_variables: the held Jp has the stack of directions K = %s, where the directions '
                             'give K = %s' % (held_K, K))
    return _Setting(
        d=len(D), order=order, box=tuple(n + 1 for n in order),
        F_in=F_in, W=W, C=C,
        sizes=dict(rL=_bonds(L), rR=_bonds(R), nD=tuple(np.shape(B)[-2] for B in D),
                   nU=tuple(np.shape(G)[-2] for G in U), N=tuple(np.shape(B)[-1] for B in D)),
        K=() if K is None else K, with_K=frozenset(rule.name for rule in graph if rule.carries_K),
        kind=kind, keep_W=keep_W, masks=masks)


def _check_stacks(
        setting:  _Setting,
        stores:   typ.Dict[str, typ.Sequence[NDArray]],     # D, U, L, R; or on a corewise point D, L
        sample:   typ.Any,                                  # the input jets; or the entries kind's index
        Vc:       typ.Optional[typ.Sequence[NDArray]],
        variations:       typ.Optional[typ.Tuple] = None,           # (V, H), or None
        core_directions:  typ.Optional[typ.Tuple] = None,           # (D_dot, L_dot), or None
) -> None:
    '''The stacks ``W`` and ``C`` are read once, ``W`` from ``w[0]`` and ``C`` from ``D[0]``; every other operand that
    carries one must have exactly it (tuple equality, not broadcast compatibility): ``C`` as the prefix of every core
    of every store and of every ``Vc[i]``, ``W`` after the jet axes of every ``w[i]``. A size-1 axis where another
    operand has more would otherwise broadcast in a contraction and give a wrong answer with no error (as for ``K``,
    ``_read_K``); and on the ragged layout every input's mode size ``N_i`` that of ``D[i]`` (a size-1 mode axis would
    broadcast too; the packed sample's ``N`` is checked on the uniform layout); and the core axes of every direction
    and of ``Vc`` exactly the sizes the stores give (P12a, round 1): ``V[i]`` and ``Vc[i]`` ``(nU_i, N_i)``, ``H[i]``
    ``(rL_i, nD_i, rR_(i+1))``, ``D_dot[i]`` ``(nD_i, N_i)`` and ``L_dot[i]`` ``(rL_i, nD_i, rL_(i+1))`` (on a corewise
    point, its own; on the uniform layout, the supercores' padded sizes). Structural: a ``ValueError`` naming the
    operand. ``y`` and the weight are checked by ``_check_data_space`` (each axis 1 or ``z``'s size: there a size
    of 1 is meant), the jet sizes by the checkers of :py:mod:`t3toolbox.backend.jets`. Its cost is about ``5d`` comparisons of short integer tuples per
    call, at trace time only under ``jit``. (A removable check, pending O-18: delete this function, its one call and
    its test class.)'''
    C, W, m = setting.C, setting.W, len(setting.order)
    core_ndim = {'D': 2, 'U': 3, 'L': 3, 'R': 3}
    operands = [(name, cores, core_ndim[name]) for name, cores in stores.items()]
    if Vc is not None:
        operands.append(('Vc', Vc, 2))
    for name, cores, ndim in operands:
        for i, core in enumerate(cores):
            shape = tuple(np.shape(core))
            if len(shape) != len(C) + ndim or shape[:len(C)] != C:
                raise ValueError('sampling_variables: every core must have the stack C of the frame, read off D[0]: '
                                 '%s[%d] has shape %s, where C = %s, so C + (the %d axes of the core) is expected'
                                 % (name, i, shape, C, ndim))
    sizes = setting.sizes
    trailing = {
        'V': lambda i: (sizes['nU'][i], sizes['N'][i]), 'Vc': lambda i: (sizes['nU'][i], sizes['N'][i]),
        'H': lambda i: (sizes['rL'][i], sizes['nD'][i], sizes['rR'][i + 1]),
        'D_dot': lambda i: (sizes['nD'][i], sizes['N'][i]),
        'L_dot': lambda i: (sizes['rL'][i], sizes['nD'][i], sizes['rL'][i + 1]),
    }
    directions = [('Vc', Vc)] if Vc is not None else []
    if variations is not None:
        directions += [('V', variations[0]), ('H', variations[1])]
    if core_directions is not None:
        directions += [('D_dot', core_directions[0]), ('L_dot', core_directions[1])]
    for name, cores in directions:
        if len(cores) != setting.d:
            raise ValueError('sampling_variables: %s has %d cores, where the frame has d = %d'
                             % (name, len(cores), setting.d))
        for i, core in enumerate(cores):
            expected = trailing[name](i)
            shape = tuple(np.shape(core))
            if shape[len(shape) - len(expected):] != expected:
                raise ValueError('sampling_variables: every direction must have the core sizes of the frame: %s[%d] '
                                 'has shape %s, where the stores give the core axes %s'
                                 % (name, i, shape, expected))
    if setting.kind != 'entries':
        for i, w in enumerate(sample):
            shape = tuple(np.shape(w))
            if len(shape) != m + len(W) + 1 or shape[m:-1] != W:
                raise ValueError('sampling_variables: every input must have the sample stack W, read off w[0]: w[%d] '
                                 'has shape %s, where W = %s, so %d jet axes, W and the mode edge are expected'
                                 % (i, shape, W, m))
            if setting.masks is None and shape[-1] != np.shape(stores['D'][i])[-1]:     # ragged: N_i read off D[i]
                raise ValueError('sampling_variables: every input must have its mode size N_i, read off D[%d]: w[%d] '
                                 'has shape %s, where N_%d = %d' % (i, i, shape, i, np.shape(stores['D'][i])[-1]))
    if setting.masks is not None:                       # uniform: the masks' C, the packed sample's d and N
        for family, mask in setting.masks.items():
            if family != 'N' and tuple(np.shape(mask)[1:-1]) != C:
                raise ValueError('sampling_variables: every mask must have the stack C of the frame: the %s mask has '
                                 'shape %s, where C = %s' % (family, np.shape(mask), C))
        if setting.kind != 'entries':
            d, N = setting.d, np.shape(stores['D'])[-1]
            if np.shape(sample)[0] != d or np.shape(sample)[-1] != N:
                raise ValueError("sampling_variables: the packed sample must have the frame's d and N: it has shape "
                                 '%s, where d = %d and N = %d' % (np.shape(sample), d, N))


def _check_data_space(
        setting:  _Setting,
        name:     str,                      # 'y' or 'weight', for the message
        value:    typ.Any,                  # of the type of z: probe len=d (or a mode-leading array); else one array
        edges:    typ.Tuple[int, ...],      # probe: the mode edge of each array, N_i (ragged) or the padded N (packed)
) -> None:
    '''``y`` and the weight are of the type of ``z``: for probe exactly ``d`` arrays (a tuple, a list, or one array
    whose first axis is the mode), each of the rank of ``z``'s array, ``F + W + C`` and the mode edge, each axis 1 or
    ``z``'s size (the edge ``N_i``; ``N`` for a packed one). Structural: a ``ValueError`` naming the object, the mode
    and the shapes. Static, so it runs under ``jit``. Without it, ``z - y`` would right-align a ``y`` of another rank:
    a ``y`` with no ``C`` on a stacked model, with ``W`` of the size of ``C``, would be read along ``C`` with no error.
    The stack of directions ``K`` is never in it (the engine places it, ``_with_K``).'''
    body = setting.box + setting.W + setting.C
    kind = setting.kind
    if kind != 'probe':
        if isinstance(value, (tuple, list)):
            raise ValueError('sampling_variables: %s of the %s kind is one array, of the type of z; got a %s'
                             % (name, kind, type(value).__name__))
        expected = [(None, body if kind == 'apply' else setting.W + setting.C)]
        arrays = [value]
    else:
        count = len(value) if isinstance(value, (tuple, list)) else (np.shape(value)[0] if np.ndim(value) else 0)
        if count != setting.d:
            raise ValueError('sampling_variables: %s of the probe kind has exactly d = %d arrays, one per mode (a '
                             'tuple, a list, or one array whose first axis is the mode, of length d); got %d'
                             % (name, setting.d, count))
        expected = [(i, body + (int(edges[i]),)) for i in range(setting.d)]
        arrays = [value[i] for i in range(setting.d)]
    for (i, shape), array in zip(expected, arrays):
        got = tuple(np.shape(array))
        if len(got) != len(shape) or any(a not in (1, b) for a, b in zip(got, shape)):
            raise ValueError('sampling_variables: %s%s has shape %s; it is of the type of z, whose array there has '
                             'shape %s (F + W + C%s): the same rank, each axis 1 or that size, and never the stack '
                             'of directions K' % (name, '' if i is None else ' (mode %d)' % i, got, shape,
                                                  ' and the mode edge' if kind == 'probe' else ''))


def _entry(name: str, kind: str) -> typ.Tuple[str, str, typ.Optional[str]]:
    '''The entry of the shape table of a variable for ``kind``: the kind's own (``_KIND_SHAPES``), else ``_SHAPES``.'''
    table = _table_name(name)
    return _KIND_SHAPES.get(kind, {}).get(table, _SHAPES[table])


def _shape(
        name:     str,
        i:        typ.Optional[int],
        setting:  _Setting,
) -> typ.Tuple[int, ...]:
    '''The shape of the variable ``name`` at index ``i``, from the table ``_SHAPES`` (``i`` None for ``f``,
    ``r_block_sumsq`` and a variable of the index ``'one'``, which have no index).'''
    if name == 'f':
        return setting.C
    if name == 'r_block_sumsq':
        return ((setting.d,) if setting.kind == 'probe' else ()) + setting.box
    K = setting.K if name in setting.with_K else ()
    if name == 'Jp_norm_sq':
        return K + setting.C
    index, jets, size = _entry(name, setting.kind)
    if index == 'one':
        return setting.box + setting.W + K + setting.C
    sizes = setting.sizes
    if jets == 'none':
        core = {'V':  lambda: (sizes['nU'][i], sizes['N'][i]),
                'H':  lambda: (sizes['rL'][i], sizes['nD'][i], sizes['rR'][i + 1]),
                'VD': lambda: (sizes['nD'][i], sizes['N'][i]),
                'HL': lambda: (sizes['rL'][i], sizes['nD'][i], sizes['rL'][i + 1])}[size]()
        return (setting.W if setting.keep_W else ()) + K + setting.C + core
    jet_sizes = tuple(min(a, f) for a, f in zip(setting.F_in[i], setting.box)) if jets == 'leaf' else setting.box
    return jet_sizes + setting.W + K + setting.C + (sizes[size][i],)


def _count(name: str, setting: _Setting) -> int:
    '''How many arrays an indexed variable has: ``d + 1`` for a bond variable, ``d`` otherwise.'''
    return setting.d + 1 if _SHAPES[_table_name(name)][0] == 'bond' else setting.d


def _expected(name: str, setting: _Setting) -> typ.Any:
    '''The shapes of the whole variable ``name``, in its form: a shape for ``f``, ``r_block_sumsq`` and a variable
    of the index ``'one'`` (the apply kind's ``z``, ...), a tuple of shapes over the index for an indexed variable, the
    pair of those of ``V_tilde`` and ``H_tilde`` for ``g``.'''
    if name in ('g', 'JtJp', 'Kp'):
        blocks = {'g': ('V_tilde', 'H_tilde'), 'JtJp': (_twin('V_tilde'), _twin('H_tilde')),
                  'Kp': ('V_tilde_dot', 'H_tilde_dot')}[name]
        return tuple(_expected(block, setting) for block in blocks)
    if name in ('f', 'r_block_sumsq', 'Jp_norm_sq') or _entry(name, setting.kind)[0] == 'one':
        return _shape(name, None, setting)
    if setting.masks is not None:                       # uniform: one array, the index leading
        return (_count(name, setting),) + _shape(name, 0, setting)
    return tuple(_shape(name, i, setting) for i in range(_count(name, setting)))


def _is_block(name: str, kind: str) -> bool:
    '''Whether a variable is a block (no jet axes; summed over the samples unless ``sum_over_samples=False``): read off
    the shape table, so the rows a ``keep_W`` reaches are not a declared set of their own.'''
    table = _table_name(name)
    return table in _SHAPES and _entry(name, kind)[1] == 'none'


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
def _site_body(
        step_name:  str,
        lens:       contractions.GroupLengths,
        order:      typ.Tuple[int, ...],
        form:       str,
        keep_W:     typ.Optional[bool] = None,  # a block row's flag; None for any other row (not passed)
):
    '''The map body of a site step: one stable object per statics (scan and map bodies close over hashable
    statics only: the record, the box, the form and a block's ``keep_W``, nothing per site).'''
    statics = dict(lens=lens, order=order, form=form) if keep_W is None else dict(lens=lens, order=order, form=form,
                                                                                    keep_W=keep_W)
    def body(x):
        return (getattr(sampling_steps, step_name)(*x, **statics),)
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
        return getattr(sampling_steps, step_name)(*args, lens=lens, order=order, form=form, kind=setting.kind)
    per_site = tuple(_at_sites(name, arg, setting.d) for name, arg in zip(rule.reads, args))
    if driver == 'site':
        keep_W = setting.keep_W if _is_block(rule.name, setting.kind) else None     # a block row: summed or not
        return xmap(_site_body(step_name, lens, order, form, keep_W), per_site)[0]
    store = args[0]                                         # a sweep's first read: the TT store it sweeps through
    at_start = driver == 'left'
    size = np.shape(store[0])[-3] if at_start else np.shape(store[-1])[-1]
    if rule.start is None:
        raise ValueError('sampling_variables: the sweep %r declares no start' % (rule.name,))
    if rule.start in _STARTS:
        stacks = (setting.K if rule.carries_K else ()) + tuple(np.shape(store[0])[:-3])     # [K] + C
        start = getattr(sampling_steps, 'start_' + rule.start)(setting.box, setting.W, stacks, size, dtype, xnp)
    else:                                                   # a start that names a node: it times the closing ones
        start = sampling_steps.start_node(values[rule.start], size, xnp)
    if setting.masks is not None:                           # uniform: ones on the real slots of the end bond only
        bond = setting.masks[_STORE_FAMILY[rule.reads[0]]][0 if at_start else -1]       # C+(size,)
        start = start * bond.reshape((1,) * (np.ndim(start) - bond.ndim) + bond.shape)
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
########    The uniform layout    ###################
#####################################################
# A uniform call masks once on entry, every input with the masks of the object it is defined at, and re-masks the
# core-shaped outputs on exit; the steps between see no mask. The masks are host numpy, static structure, never
# traced (docs/contributor/uniform_pytree_composition.md). A family is a mask over one kind of edge: 'down', 'up'
# (the Tucker edges, (d,)+C+(n,)), 'left', 'right' (the bonds, (d+1,)+C+(r,)), and 'N' (the mode axis, (d, N), from
# the static shape). On a corewise point (B, G) the up family is the down one and the right the left one.

# The masks of each input, one per trailing (core) axis: (family, part), the part of a bond family 'lo' (bonds
# 0 .. d-1, the left leg of a TT core) or 'hi' (bonds 1 .. d); a site family whole. Every Tucker-shaped input has the
# shape mask on N beside its rank mask (the variation masks are rank masks only).
_TUCKER = (('down', None), ('N', None))
_UNIFORM_MASKS = {
    'D': _TUCKER, 'D_dot': _TUCKER,
    'U': (('left', 'lo'), ('up', None), ('right', 'hi')),
    'L': (('left', 'lo'), ('down', None), ('left', 'hi')), 'L_dot': (('left', 'lo'), ('down', None), ('left', 'hi')),
    'R': (('right', 'lo'), ('down', None), ('right', 'hi')),
    'V': (('up', None), ('N', None)), 'Vc': (('up', None), ('N', None)),
    'H': (('left', 'lo'), ('down', None), ('right', 'hi')),
    'w': (('N', None),), 'y': (('N', None),), 'weight': (('N', None),),
}
# A variable's mask, by the size of its table entry: a bond, site or mode variable over its edge; a block over the
# axes of the core it stands for ('V' and 'VD' with the shape mask on N).
_SIZE_MASKS = {
    'rL': (('left', None),), 'rR': (('right', None),), 'nD': (('down', None),), 'nU': (('up', None),),
    'N': (('N', None),),
    'V': _UNIFORM_MASKS['V'], 'H': _UNIFORM_MASKS['H'], 'VD': _TUCKER, 'HL': _UNIFORM_MASKS['L'],
}
_STORE_FAMILY = {'L': 'left', 'R': 'right'}         # the bond family of the store a sweep goes through


def _families(
        frame:  typ.Tuple,                  # (D, U, L, R, shape, masks) or (B, G, shape, masks)
) -> typ.Dict[str, NDArray]:                # the mask families, host numpy
    """The mask families of a uniform frame or point, checked to be concrete (a traced mask raises)."""
    if len(frame) == 6:
        shape, (down, up, left, right) = frame[4], frame[5]
    else:
        shape, (down, left) = frame[2], frame[3]
        up, right = down, left
    require_concrete_masks(down, up, left, right)
    return dict(down=down, up=up, left=left, right=right, N=prefix_mask(np.asarray(shape), np.shape(frame[0])[-1]))


def _masked(
        x:      NDArray,                    # (lead,) + stacks + core: lead d or d+1, the stacks ending in C
        spec:   typ.Tuple,                  # one (family, part) per trailing core axis
        masks:  typ.Dict[str, NDArray],     # the families
) -> NDArray:                               # x with its padding zero
    """``x`` times its masks, each a host array ``(lead,) + C + (size,)`` (``N``'s with no ``C``) placed on its axis,
    right-aligned against the stacks so that a ``W`` or ``K`` before ``C`` broadcasts."""
    if not spec:
        return x
    n_core = len(spec)
    n_stack = np.ndim(x) - 1 - n_core
    total = None
    for j, (family, part) in enumerate(spec):
        mask = masks[family]
        mask = mask[:-1] if part == 'lo' else (mask[1:] if part == 'hi' else mask)
        stack = mask.shape[1:-1]
        mask = mask.reshape(mask.shape[:1] + (1,) * (n_stack - len(stack)) + stack + (1,) * j + mask.shape[-1:]
                            + (1,) * (n_core - 1 - j))
        total = mask if total is None else total * mask
    return x * total


def _mask_variable(
        name:     str,
        value:    typ.Any,                  # a variable of the uniform layout, or a pair of blocks
        setting:  _Setting,
) -> typ.Any:
    """A variable, a block or a pair of blocks with its padding zero, by its table entry; a variable of no index or
    with no padded axis (the apply kind's ``z``, ``f``, the sums) as it is."""
    if name in ('g', 'JtJp', 'Kp'):
        blocks = {'g': ('V_tilde', 'H_tilde'), 'JtJp': (_twin('V_tilde'), _twin('H_tilde')),
                  'Kp': ('V_tilde_dot', 'H_tilde_dot')}[name]
        return tuple(_mask_variable(block, part, setting) for block, part in zip(blocks, value))
    if name in ('f', 'r_block_sumsq', 'Jp_norm_sq', 'w') or _table_name(name) not in _SHAPES:
        return value
    index, _, size = _entry(name, setting.kind)
    if index == 'one':
        return value
    return _masked(value, _SIZE_MASKS[size], setting.masks)


def _uniform_index(
        index:  NDArray,                    # the entries kind's grid points, int, (d,)+W
        shape:  typ.Sequence[int],          # the real mode sizes N_i
) -> NDArray:                               # the index, a negative one wrapped by its real N_i
    """The entries kind's index on a padded ``D``: as on the ragged layout, a numpy index outside ``[-N_i, N_i)``
    raises an ``IndexError`` (on the padded layout it would read a masked column, a silent zero), and a negative one
    counts from ``N_i``, not from the padded ``N``. A jax index (maybe traced) cannot be checked: one out of range
    reads a zero column."""
    sizes = np.asarray(shape).reshape((len(shape),) + (1,) * (np.ndim(index) - 1))
    if not is_jax_ndarray(index):
        array = np.asarray(index)
        bad = (array >= sizes) | (array < -sizes)
        if bad.any():
            mode = int(np.argwhere(bad)[0][0])
            raise IndexError('sampling_variables: index %d is out of range for mode %d of size %d'
                             % (int(array[bad][0]), mode, int(shape[mode])))
        return np.where(array < 0, array + sizes, array)
    import jax.numpy as jnp
    return jnp.where(index < 0, index + sizes, index)


#####################################################
########    The engine    ###########################
#####################################################

def _closure_facts(
        kind:             str,
        frame:            typ.Tuple,
        want:             typ.Sequence[str],
        order:            typ.Any,
        y:                typ.Any,
        variations:       typ.Any,
        core_directions:  typ.Any,
        Vc:               typ.Any,
        held:             typ.Any,
) -> typ.Tuple:                             # (order, want, uniform, call, held_names, w_given, rules)
    '''The facts of a call that decide what it computes, and its closure: the one reading shared by
    :py:func:`sampling_variables` and :py:func:`computed_names`, so the two cannot diverge. It checks what can be
    checked from the arguments' presence and the frame's structure (the kind, the jet shape, ``want``'s type and names,
    ``held``'s type, the frame's form, a direction's form on a corewise point, and a wanted name's missing argument,
    from the closure) and reads no array.'''
    if kind not in _KINDS:
        raise ValueError('sampling_variables: unknown sampling kind %r; expected one of %s' % (kind, _KINDS))
    order = contractions._read_order(order, 'sampling_variables')
    if kind == 'entries' and order != ():
        raise ValueError('sampling_variables: the entries kind has no jets: its order is (), got %r' % (order,))
    if isinstance(want, str):
        raise TypeError('sampling_variables: want is a sequence of names, not one name; got %r' % (want,))
    want = tuple(want)
    if held is not None and not isinstance(held, HeldVariables):
        raise TypeError('sampling_variables: held is a HeldVariables or None; got a %s' % type(held).__name__)
    layout, call_name = frame_form(frame)               # the layout and the call type, from the frame's structure
    uniform, call = layout == 'uniform', '_' + call_name
    if call == '_corewise' and variations is not None:
        raise ValueError('sampling_variables: a corewise point has one form of a direction, the raw one: pass it as '
                         'core_directions, not variations')
    names = _names(kind, call)
    unknown = [name for name in want if name not in names]
    if unknown:
        zero = [name for name in unknown if name in _ZERO_NAMES.get(kind, ())]
        why = (' (%s %s zero for the %s kind: %s)' % (', '.join(map(repr, zero)), 'is' if len(zero) == 1 else 'are',
                                                       kind, _ZERO_REASON[kind])) if zero else ''
        raise ValueError('sampling_variables: unknown variable %s for the %s kind on a %s call%s; the names are %s'
                         % (', '.join(map(repr, unknown)), kind, call.lstrip('_'), why, ', '.join(names)))
    given = {'y': y, 'Vc': Vc, 'variations': variations, 'core_directions': core_directions}
    held_names = frozenset(held.keys()) if held is not None else frozenset()
    present = (frozenset(name for name, value in given.items() if value is not None) | {call}
               | ({_HELD_JP} if 'Jp' in held_names else frozenset()))
    w_given = 'w' not in _made(_GRAPHS[kind, call])    # probe, apply: 'w' is the sample as given, no row
    rules = _closure(kind, tuple(name for name in want if not (name == 'w' and w_given)), held_names, present)
    return order, want, uniform, call, held_names, w_given, rules


def frame_shape(
        frame:  typ.Tuple,                  # any of the four forms (frame_form)
) -> typ.Tuple[int, ...]:                   # the mode sizes (N_0, ..., N_(d-1))
    '''The mode sizes ``(N_0, ..., N_(d-1))`` of the tensors a frame (or a corewise point) is for, for any of its four
    forms (:py:func:`frame_form`): read off the last axis of each Tucker core on the ragged layout, and on the
    uniform layout the static ``shape`` element (the second to last; the supercore's ``N`` is a padded width, not a
    mode size). It raises as :py:func:`frame_form` does.

    >>> import numpy as np
    >>> from t3toolbox.backend.sampling_variables import frame_shape
    >>> B = (np.ones((2, 4)), np.ones((2, 5)))
    >>> G = (np.ones((1, 2, 2)), np.ones((2, 2, 1)))
    >>> frame_shape((B, G)), frame_shape((B, G, G, G))
    ((4, 5), (4, 5))
    >>> Bsc, Gsc = np.ones((2, 2, 6)), np.ones((2, 2, 2, 2))        # supercores padded to N = 6
    >>> frame_shape((Bsc, Gsc, (4, 5), None)), frame_shape((Bsc, Gsc, Gsc, Gsc, (4, 5), None))
    ((4, 5), (4, 5))
    '''
    layout, _ = frame_form(frame)
    if layout == 'uniform':
        return tuple(int(n) for n in frame[-2])
    return tuple(int(np.shape(core)[-1]) for core in frame[0])


def computed_names(
        kind:              str,                              # 'probe' | 'apply' | 'entries'
        frame:             typ.Tuple,                        # any of the four forms (frame_form)
        sample:            typ.Any,                          # accepted and not read (None allowed)
        want:              typ.Sequence[str],                # variable names
        order:             typ.Tuple[int, ...] = (),
        y:                 typ.Optional[typ.Any] = None,     # presence only
        weight:            typ.Optional[NDArray] = None,     # not read
        variations:        typ.Optional[typ.Tuple] = None,   # presence only
        core_directions:   typ.Optional[typ.Tuple] = None,   # presence only
        Vc:                typ.Optional[typ.Sequence[NDArray]] = None,   # presence only
        sum_over_samples:  bool = True,                      # not read
        held:              typ.Optional[HeldVariables] = None,   # its filled slots
) -> typ.FrozenSet[str]:                                     # the public names the call would compute
    '''The names of the variables a call of :py:func:`sampling_variables` with these arguments would compute: a pure
    query, which computes nothing. Its arguments are exactly an engine call's, so ``computed_names(*args, **kwargs)``
    and ``sampling_variables(*args, **kwargs)`` take the same ones.

    It reads the facts the engine reads, through the engine's own reading and its cached closure: which of ``y``,
    ``Vc``, ``variations`` and ``core_directions`` are given (presence only, never the values), the call type of the
    frame (:py:func:`frame_form`), and ``held``'s filled slots. The sample, the weight, the jet sizes and every value
    are not read. It raises where the engine raises on those facts, with the engine's messages.

    It returns the **public** names (the graph's private nodes are internals), as a ``frozenset``. A name that
    ``held`` fills is not in it, wanted or not: the engine reuses it and does not compute it. The sample ``'w'`` is not
    in it either. A route is chosen from the call's facts, never from ``want``, so wanting a name the call computes
    anyway adds no work: an oracle that holds what is free adds to ``want`` the names it would like to hold that this
    returns.

    Examples
    --------
    A line search asks for ``f`` alone, a gradient step for ``f`` and ``g``; the second computes more:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.manifold as t3m
    >>> from t3toolbox.backend.sampling_variables import computed_names, sampling_variables, HeldVariables
    >>> np.random.seed(0)
    >>> x = t3.TuckerTensorTrain.randn((4, 5, 6), (2, 3, 2), (1, 2, 2, 1))
    >>> frame = t3m.MANIFOLD.frame(x).data
    >>> ww, y = ([np.random.randn(3, N) for N in x.shape] for _ in range(2))
    >>> line_search = computed_names('probe', frame, ww, ('f',), y=y)
    >>> sorted(line_search)
    ['eta_D', 'f', 'mu', 'nu_L', 'r', 'xi', 'z']
    >>> sorted(computed_names('probe', frame, ww, ('f', 'g'), y=y) - line_search)       # what g adds
    ['H_tilde', 'V_tilde', 'eta', 'g', 'mu_tilde', 'nu', 'nu_tilde', 'w_tilde', 'xi_tilde']

    The pattern of an oracle that holds what is free: one dictionary of arguments for both calls, the hold-list names
    the call computes anyway added to ``want``, and split off:

    >>> args = dict(y=y)
    >>> hold = ('xi', 'mu', 'nu', 'eta', 'r')
    >>> want = ('f',)
    >>> extras = tuple(n for n in hold if n in computed_names('probe', frame, ww, want, **args) and n not in want)
    >>> *values, = sampling_variables('probe', frame, ww, want + extras, **args)
    >>> held = HeldVariables(**dict(zip(extras, values[len(want):])))
    >>> print(extras, held.keys())
    ('xi', 'mu', 'r') ('xi', 'mu', 'r')
    '''
    rules = _closure_facts(kind, frame, want, order, y, variations, core_directions, Vc, held)[-1]
    return frozenset(rule.name for rule in rules if not rule.name.startswith('_'))


def frame_form(
        frame:  typ.Tuple,                  # (D, U, L, R) or (B, G); uniform: + (shape, masks) after the cores
) -> typ.Tuple[
        str,                                # the layout: 'ragged' | 'uniform'
        str,                                # the call: 'manifold' (a frame) | 'corewise' (a point)
]:
    '''The form of a frame: its layout and the call it makes, read from its structure alone.

    The layout first, from the type of ``frame[0]`` (a tuple of cores on the ragged layout, a supercore on the uniform
    one), then the count: a ragged frame has four stores ``(D, U, L, R)`` and a corewise point two ``(B, G)``; a
    uniform frame six elements ``(D, U, L, R, shape, masks)`` and a uniform point four ``(B, G, shape, masks)``. **A
    count of four is a ragged frame or a uniform point**, which only the layout tells apart. The engine reads a frame
    with this function, and so should an oracle that passes a direction in the form the call takes (on a corewise
    point, as ``core_directions``). Anything else raises a ``ValueError``.

    >>> import numpy as np
    >>> from t3toolbox.backend.sampling_variables import frame_form
    >>> B, G = (np.ones((2, 4)),), (np.ones((1, 2, 1)),)
    >>> frame_form((B, G)), frame_form((B, G, G, G))
    (('ragged', 'corewise'), ('ragged', 'manifold'))
    >>> frame_form((np.ones((1, 2, 4)), np.ones((1, 1, 2, 1)), (4,), None))      # four elements, uniform: a point
    ('uniform', 'corewise')
    '''
    if not isinstance(frame, (tuple, list)) or not frame:
        raise ValueError('sampling_variables: the frame is a tuple of stores; got a %s' % type(frame).__name__)
    uniform = is_ndarray(frame[0])                      # the layout first, then the count
    counts = {(False, 2): 'corewise', (False, 4): 'manifold', (True, 4): 'corewise', (True, 6): 'manifold'}
    if (uniform, len(frame)) not in counts:
        raise ValueError('sampling_variables: a ragged frame has four stores (D, U, L, R), or a corewise point two '
                         '(B, G); a uniform frame six elements (D, U, L, R, shape, masks), or a uniform point four '
                         '(B, G, shape, masks); got %d on the %s layout'
                         % (len(frame), 'uniform' if uniform else 'ragged'))
    return ('uniform' if uniform else 'ragged'), counts[uniform, len(frame)]


def sampling_variables(
        kind:              str,                              # 'probe' | 'apply' | 'entries'
        frame:             typ.Tuple,                        # (D, U, L, R), four tuples of d arrays; or (B, G)
        sample:            typ.Any,                          # ww: the input jets, len=d, elm_shape=F_in_i+W+(Ni,);
                                                             #   entries: index, int, shape=(d,)+W
        want:              typ.Sequence[str],                # variable names; the return is aligned with them

        order:             typ.Tuple[int, ...] = (),         # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...)
        y:                 typ.Optional[typ.Any] = None,     # the data, of the type of z, each axis 1 or z's size:
                                                             #   probe len=d, elm_shape=F+W+C+(Ni,); apply F+W+C; entries W+C
        weight:            typ.Optional[typ.Any] = None,     # omega, of the type of z likewise (no K); None = ones
        variations:        typ.Optional[typ.Tuple] = None,   # (V, H), elm_shapes K+C+(nUi,Ni), K+C+(rLi,nDi,rR(i+1))
        core_directions:   typ.Optional[typ.Tuple] = None,   # (D_dot, L_dot), the shapes of D, L with K+C
        Vc:                typ.Optional[typ.Sequence[NDArray]] = None,   # len=d, elm_shape=C+(nUi,Ni)
        sum_over_samples:  bool = True,                      # on the blocks; False: W joins their stack, W+K+C
        held:              typ.Optional[HeldVariables] = None,   # held variables; shapes checked, values trusted
) -> typ.Tuple[typ.Any, ...]:                                # aligned with want; shapes in the module's table
    """The sampling variables ``want`` of a Tucker tensor train at a frame, sampled at input jets.

    Each variable is computed by its recurrence over the sites, once, and only the variables that what is wanted
    reads (the closure over the declared graph, in the module docstring). Held variables are used, not recomputed (cast
    once to the result dtype, like any counted input); their shapes are checked against the table of the module docstring, their values trusted.

    The engine checks what it needs in order to run, and nothing more: an unknown kind or name raises; a jet shape
    that is not a tuple of nonnegative integers raises; an entries call with jets raises; a wanted variable whose
    argument is missing raises (``r`` without ``y`` or a held ``r``), and so does a held variable of the wrong
    shape. The stacks are checked: every core, ``Vc`` and every input has exactly the ``C`` of ``D[0]`` and the ``W`` of
    ``w[0]``, and every direction the ``K + C`` of the first (a size-1 axis would broadcast silently); ``y`` and the
    weight are of the type of ``z``, each array of ``z``'s rank and each axis 1 or ``z``'s size (for probe exactly
    ``d`` arrays), or a ``ValueError``. It does not check the mode sizes of the sample against the frame's or the jet
    sizes (the checkers of :py:mod:`t3toolbox.backend.jets`), and nothing numerical: four inconsistent stores give a wrong answer, with no
    error.

    Numpy or jax is read off every array input. Every output is in one dtype: the lowest floating precision among the
    counted inputs (jax over numpy), which are cast to it once. The counted inputs are the arrays the computation
    reads, and every held variable that a row reads or that is wanted (a wanted held variable is an output: it is
    returned cast). A held variable neither read nor wanted is not counted and not returned; an argument nothing
    wanted reads does not count either. For probe and apply the sample ``w``, when wanted, is returned as given; for
    entries ``w`` is a row of the graph (the one-hot vectors, reading ``index`` and ``D``), in the result dtype.

    Parameters
    ----------
    kind: str
        ``'probe'``, ``'apply'`` or ``'entries'``. The kind decides the output's type: probe, one array per mode with
        the mode edge open; apply, one number per sample and slot; entries, one number per grid point (no jets).
    frame: tuple
        ``(D, U, L, R)``, each ``len=d``: ``D[i]`` of shape ``C+(nDi, Ni)``, ``U[i]`` ``C+(rLi, nUi, rR(i+1))``,
        ``L[i]`` ``C+(rLi, nDi, rL(i+1))``, ``R[i]`` ``C+(rRi, nDi, rR(i+1))``. The base point is ``(D, L)``.
        Or a corewise point ``(B, G)``, the cores of a Tucker tensor train: one basis on each edge, and the names
        without a basis (the module docstring). On the uniform layout ``(D, U, L, R, shape, masks)`` or
        ``(B, G, shape, masks)``: supercores with ``d`` leading, the static ``shape`` and host masks.
    sample: sequence of NDArray, or NDArray
        The input jets ``ww``: ``len=d``, ``elm_shape=F_in_i+W+(Ni,)``, with ``len(order)`` leading jet axes. For
        ``'entries'``, the grid points ``index``: integer, ``shape=(d,)+W`` (never cast, and not counted for the
        dtype). On the uniform layout the input jets may come packed, ``(d,)+F_in+W+(N,)``, or ragged (the engine packs
        them).
    want: sequence of str
        The names wanted (the module's graph, and ``'w'``, the input vectors: for probe and apply the sample as
        given; for entries the one-hot vectors ``e_index`` of its definition, made only when wanted, in the result
        dtype).
    order: tuple of int
        The jet shape. ``()`` for no jets.
    y: sequence of NDArray or NDArray, optional
        The data, of the type of ``z``: probe ``len=d`` (a tuple, a list, or one array whose first axis is the mode),
        ``elm_shape=F+W+C+(Ni,)``; apply ``shape=F+W+C``; entries ``shape=W+C``; each axis 1 or that size (a size of 1
        is the same datum along the axis: data shared by a stack have size-1 ``C`` axes). Needed for ``r`` and what
        reads it. On the uniform layout probe data may come ragged (packed here) or packed, ``(d,)+F+W+C+(e,)`` with
        the edge ``e`` 1 or ``N``.
    weight: sequence of NDArray or NDArray, optional
        ``omega``, of the type of ``z`` as ``y`` is, never with the stack of directions ``K`` (the engine places it
        against ``Jp``); it enters squared, through :py:func:`~t3toolbox.backend.jets.data_metric` at ``gamma = 1``.
        ``None`` is ones. :py:func:`~t3toolbox.backend.jets.blocks_to_weight` makes one from a value per block.
    variations: tuple, optional
        The direction in the frame's form, ``(V, H)``: ``V[i]`` of shape ``K+C+(nUi, Ni)``, ``H[i]``
        ``K+C+(rLi, nDi, rR(i+1))``, the stack of directions ``K`` the same for all. Needed for the dot variables,
        ``Jp``, ``Jp_norm_sq`` and ``JtJp``.
    core_directions: tuple, optional
        The direction in the raw form at the base point, ``(D_dot, L_dot)``, with the shapes of ``D`` and ``L`` and the
        stack ``K`` in front (the same ``K`` as ``variations``). Needed for the raw dots, the tilde-dot variables and
        ``Kp``; on a corewise point, the only form of a direction.
    sum_over_samples: bool
        On the blocks (``V_tilde``, ``H_tilde``, ``V_tilde_dot``, ``H_tilde_dot``, and so ``g``, ``JtJp``, ``Kp``):
        ``True`` sums them over the samples; ``False`` keeps one block per sample, ``W`` first, ``W + [K] + C`` and the
        core's shape. ``f``, ``r_block_sumsq`` and ``Jp_norm_sq`` are sums and ignore it.
    Vc: sequence of NDArray, optional
        The Tucker centers, ``len=d``, ``elm_shape=C+(nUi, Ni)``: the probe kind's ``z`` is then read off the frame
        through them, with no sweep through the base point's ``L`` cores. Only the probe kind's ``z`` reads it; given
        to apply or entries it is read by nothing and accepted, as any argument nothing wanted reads.
    held: HeldVariables, optional
        Variables computed earlier at this frame and sample (on the uniform layout one array each, masked on entry).

    Returns
    -------
    tuple
        One entry per name of ``want``: a tuple over the index for an indexed variable (``d + 1`` bonds, or ``d``
        sites or modes), an array for ``f`` and ``r_block_sumsq``, and for the apply and entries kinds' ``z``,
        ``r``, ``w_tilde`` and ``Jp``.

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
    >>> line = affine_jet(ww, pp)                                         # F_in = (2,)
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

    The apply kind: one number per sample and slot, ``mu`` at bond ``d`` closed by the ones; its data ``F + W``:

    >>> y_apply = rng.standard_normal((4, 7))                             # F + W
    >>> z, f, g = sv.sampling_variables('apply', frame, line, ('z', 'f', 'g'), order=(3,), y=y_apply)
    >>> print(z.shape, f.shape, [V.shape for V in g[0]])
    (4, 7) () [(3, 4), (2, 5), (3, 6)]
    >>> (mu,) = sv.sampling_variables('apply', frame, line, ('mu',), order=(3,))
    >>> print(bool(np.allclose(z, mu[-1].sum(axis=-1))))
    True

    Its left sides and leaves have no tilde (the seed sits on bond ``d``); asking for one says so:

    >>> sv.sampling_variables('apply', frame, ww, ('mu_tilde',))      # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    ValueError: sampling_variables: unknown variable 'mu_tilde' for the apply kind on a manifold call ('mu_tilde' is
    zero for the apply kind: ...); the names are ...

    The entries kind, at integer grid points ``index`` (``(d,) + W``, no jets): ``apply`` at the unit vectors.

    >>> from t3toolbox.backend.jets import index_to_one_hot
    >>> index = np.array([[0, 3, 3], [1, 1, 4], [5, 0, 2]])                # (d,) + W, W = (3,)
    >>> z, xi = sv.sampling_variables('entries', frame, index, ('z', 'xi'))
    >>> print(z.shape, [x.shape for x in xi])                             # the leaves are columns of D
    (3,) [(3, 2), (3, 3), (3, 2)]
    >>> (z1,) = sv.sampling_variables('apply', frame, index_to_one_hot(index, N), ('z',))
    >>> print(bool(np.allclose(z, z1)))
    True
    """
    # 1. to 3. the facts of the call and its closure: the one reading computed_names shares
    facts = _closure_facts(kind, frame, want, order, y, variations, core_directions, Vc, held)
    order, want, uniform, call, held_names, w_given, rules = facts
    stores = dict(zip(('D', 'L') if call == '_corewise' else ('D', 'U', 'L', 'R'), frame))
    if kind == 'entries':                               # the index's length d: what the engine reads the sample by
        d = len(frame[0])
        if np.ndim(sample) < 1 or np.shape(sample)[0] != d:
            raise ValueError('sampling_variables: the entries kind\'s index has one row per mode, shape (d,) + W with '
                             'd = %d; got shape %s' % (d, np.shape(sample)))
    y_given, weight_given = y, weight                   # as given, for their check (before any packing)
    masks = None
    if uniform:                                         # the families; the sample packed (zero jet coefficients and
        masks = _families(frame)                        # zero mode padding), its index checked as on ragged
        shape = frame[-2]
        N = np.shape(frame[0])[-1]
        if kind == 'entries':
            sample = _uniform_index(sample, shape)
        else:
            sample = ut3_operations.pack_if_ragged(sample, N, len_P=len(order))
        if kind == 'probe':                             # y and the weight: the data-space packing (edge 1 kept)
            y = None if y is None else ut3_operations._pack_data_space(y, N)
            weight = None if weight is None else ut3_operations._pack_data_space(weight, N)
    graph = _GRAPHS[kind, call]
    held_Jp = held['Jp'] if held is not None and 'Jp' in held else None
    setting = _read_setting(kind, stores, sample, order, variations, graph, core_directions,
                            keep_W=not sum_over_samples, masks=masks, held_Jp=held_Jp)
    _check_stacks(setting, stores, sample, Vc, variations, core_directions)
    for name, given in (('y', y_given), ('the weight', weight_given)):
        if given is not None:                           # as given: ragged against N_i, packed against N
            if not uniform:
                edges = setting.sizes['N']
            elif isinstance(given, (tuple, list)):
                edges = tuple(frame[-2])                # the real mode sizes
            else:
                edges = (np.shape(frame[0])[-1],) * setting.d
            _check_data_space(setting, name, given, edges)
    if held is not None:
        _check_held(held, setting)

    # 4. numpy or jax, and the result dtype over the arrays the closure reads; those are cast once
    held_values = {name: held[name] for name in held_names}
    use_jax = tree_contains_jax((frame, sample, y, weight, variations, core_directions, Vc,
                                 tuple(held_values.values())))
    xnp, xmap, xscan = get_backend(uniform, use_jax)
    V, H = variations if variations is not None else (None, None)
    D_dot, L_dot = core_directions if core_directions is not None else (None, None)
    inputs = dict(stores, y=y, weight=weight, Vc=Vc, V=V, H=H, D_dot=D_dot, L_dot=L_dot,
                  **{('index' if kind == 'entries' else 'w'): sample})
    read = {name for rule in rules for name in _deps(rule)}
    held_counted = held_names & (read | set(want))         # a held variable read by a row, or wanted (an output)
    floating = (read & set(inputs)) - _INTEGER_ARGUMENTS   # the integer index is neither counted nor cast
    counted = [inputs[name] for name in sorted(floating)] + [held_values[name] for name in sorted(held_counted)]
    dtype = _lowest_precision_dtype(xnp, *[leaf for value in counted for leaf in _leaves(value)])
    if dtype is None:
        dtype = np.dtype(xnp.result_type(float))
    values = {name: _cast(inputs[name], dtype, xnp) for name in floating}
    values.update({name: xnp.asarray(inputs[name]) for name in (read & set(inputs)) & _INTEGER_ARGUMENTS})
    values.update({name: _cast(held_values[name], dtype, xnp) for name in held_counted})
    if uniform:                                         # mask once: every input by the object it is defined at,
        for name in floating:                           # a held variable by its table (padding is not a value)
            if name in ('y', 'weight') and kind != 'probe':          # apply's and entries' have no N edge
                continue
            if name == 'weight' and (values[name] is None or np.shape(values[name])[-1] == 1):
                continue                                # None is ones; an edge of 1 has no padding to mask (the probe
                                                        # y always: its value must not reach r's padding)
            if name in _UNIFORM_MASKS:
                values[name] = _masked(values[name], _UNIFORM_MASKS[name], masks)
        for name in held_counted:
            values[name] = _mask_variable(name, values[name], setting)

    # 5. to 7. run the rows
    m = len(order)
    lens = contractions.GroupLengths(P=m, W=len(setting.W), K=len(setting.K), C=len(setting.C))
    lens_no_K = dc.replace(lens, K=0)
    form = 'scan' if uniform else 'loop'                   # over the orders: a real scan on the uniform layout
    for rule in rules:
        values[rule.name] = _run(rule, values, lens if rule.carries_K else lens_no_K, setting, form, dtype,
                                 xnp, xmap, xscan)

    # 8. the return, aligned with want; on the uniform layout the core-shaped outputs re-masked
    out = tuple(sample if (name == 'w' and w_given) else values[name] for name in want)
    if uniform:
        out = tuple(value if (name == 'w' and w_given) else _mask_variable(name, value, setting)
                    for name, value in zip(want, out))
    return out
