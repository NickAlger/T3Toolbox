# P11 design note: Engine G, the uniform layout and sharding over `W`

Implementer, 2026-11-20, read only, against `graded-messages` with P10 built (not committed). Sources:
- the plan's P11, section 2.2 (the frame structures), section 2.3 steps 2 and 8, and section 7.2;
- `docs/uniform_equivalence_contract.md`, `docs/contributor/testing_strategy.md` (exact masks, garbage, the two-hole
  note), `docs/uniform_masks_vs_ranks.md`, `docs/contributor/uniform_pytree_composition.md`,
  `docs/uniform_backend_jit_recipe.md`;
- today's uniform wrappers (`ut3_sampling`, `utv_sampling`: mask once, pack, delegate, re-mask) and `ufv_masking`.

## 1. Entry and exit

**The structures, by structure.** Ragged or uniform is read off `is_ndarray(frame[0])`; then the count:
- **manifold:** `(D, U, L, R, shape, masks)`, the six-element `UT3Frame.data`, with
  `masks = (down, up, left, right)`, host booleans `(d or d+1,) + C + (size,)`;
- **corewise:** `(B_sc, G_sc, shape, masks)`, the four-element `UniformTuckerTensorTrain.data`, with
  `masks = (tucker, tt)`;
- a ragged frame has two or four elements, as today. A count of three or five raises.

**Mask once on entry**, every input with the masks of the object it is defined at. One table extends
`_DIRECTION_MASKS` to every masked input:

| Input | Its masks |
| --- | --- |
| stores `D`, `U`, `L`, `R` (manifold) | the frame's own: `ufv_apply_frame_masks` |
| stores `B`, `G` (corewise) | the point's own: `ut3_apply_masks` |
| `V`, `H`, `Vc` | the frame's variation masks (`ufv_variation_masks_over_stack`, `K + C`) |
| `D_dot`, `L_dot` | the frame's `D` and `L` masks over `K + C` (the point's own on a corewise call) |
| `y` (probe) | the shape mask on the mode axis `N` |
| the sample `w` | none (see below) |

- **`core_directions`.** On a manifold call they take the frame's `D` and `L` masks, never the point's: at TT slack
  the point's are a strict superset, and a direction's padded slot would be multiplied into real slots (the two-hole
  note).
- **`y` must be masked.** `r = z - y` carries `y`'s padding, and `f` and `r_block_sumsq` sum over `N`, so garbage there
  would change `f`.
- **The sample needs no mask.** Every contraction with `w` meets a masked `D`, except the `V` readout, whose padded
  columns are re-masked on exit. The engine packs a ragged sample itself, `pack_if_ragged(ww, N, len_J=len(order))`.
- **Held variables are not masked.** Their values are trusted (section 2.3 step 1). The engine's own outputs have exact
  zero padding (next point), so a round trip stays clean.

**The start values are masked** by the end bond's mask. The padded bond `r` carries ones only on its real slots,
including the seed's start node. Every operand is then zero in its padding, so every variable comes out with exactly
zero padding. That is cheaper to state, and to test, than "don't care". It costs one host mask per start.

**Re-mask on exit** the core-shaped outputs. The masks come from the block's size in the shape table, so there is no
new set of names:
- `'V'` and `'H'` (`V_tilde`, `H_tilde` and their twins, so `g` and `JtJp`) take the variation masks;
- `'VD'` and `'HL'` (`V_tilde_dot`, `H_tilde_dot`, so `Kp`) take the frame's `D` and `L` masks;
- on a corewise call, all four take the point's masks;
- the masks are broadcast over `[W] + K + C` (`_masks_over_stack`, with `W` when `sum_over_samples=False`);
- `z`, `r` and `Jp` need nothing: their mode padding is zero by construction.

**The uniform branches:**
- **`_read_setting`:**
  - `d = D.shape[0]`, `C = D.shape[1:-2]`;
  - `W = sample.shape[1 + m:-1]`, the packed sample; `J_in` is the packed jet sizes, the same for every mode;
  - `sizes` are the padded sizes; the real sizes are in `shape` and the masks.
- **`_read_K`:** `K` from `V.shape[1:-2-len(C)]` (or `D_dot`).
- **`_check_stacks`**, removable as before. `C` follows the leading `d` on every supercore and every mask. The packed
  sample's leading `d` and trailing `N` must equal `D`'s. An `N` of 1 would broadcast silently through the `'o'` label
  in a hand-packed sample.
- **The shape table** gains the layout: a variable is one array `(d or d+1,) + J + W + [K] + C + (padded size,)`, so
  `_expected` checks a held uniform variable as one array.

**Masks are host numpy, always** (`CLAUDE.md`): `np`, never `xnp`, and never traced. Under `jit` a backend user passes
the frame inside its object, where the masks are `aux_data`, or closes over the masks (`uniform_backend_jit_recipe.md`).
A traced mask raises in `require_concrete_masks`, as today. The engine's docstring says so.

## 2. The steps, unchanged

- **A step sees one site's arrays on both layouts.** The site rows go through `xmap` (`jax.lax.map` with jax, a loop
  on numpy) over the leading `d`. The sweeps go through `xscan` (a real `lax.scan`). `_at_sites` slices bond variables
  the same way on arrays. The terms run `over_orders='scan'`: a real scan over a full partner at `m = 1`, the loop at
  `m >= 2`. The memo key (P10) keeps one body per statics.
- **What does change: the whole steps.** Today they iterate modes and return tuples. `r_step`, `w_tilde_step`
  (through `jet_to_dual_jet`, which on a packed array returns a tuple), `g_step` (`tuple(V_tilde)` would split a
  supercore) and `w_one_hot_step` must return packed arrays on uniform. They gain a `layout` static beside `kind`.
  The probe kind's per-mode sums become one vectorized sum over the leading `d`: the weight `(d,) + F` broadcasts
  against `(d,) + F + ...`, and `y` gains its `C` axes before the edge.
  - **This is not in the plan.** I'd add it to P11's text.

## 3. `pack_vectors` and `pack_if_ragged` with `len_J=0` (N41)

- **The change:** `len_J` leading axes are brought to their largest size across the modes with zeros; the mode axis
  is padded to `N` as today; every other axis must agree, and a mismatch raises as today (the `xnp.stack` fails). Add an
  explicit check with a message naming the mode, since today's error comes from inside `pad`/`stack`.
- **`len_J=0` is today's behaviour exactly**, so the 55 library calls and 36 test calls are unchanged.
- **The zeros are coefficients, not padding (O-45).** The packed input is the same polynomial, so no mask on any jet
  axis.
- **`unpack_vectors`** gets no `len_J`. Unpacking a jet axis would need each mode's own sizes, which a packed array
  does not carry. The test compares the unpacked mode axis after slicing the jet axes by hand.

## 4. Entries on a padded `D`

- **The gather** `D[..., index]` reads only columns `index < N_i <= N`: real columns. The padded rows `n` are masked,
  so the leaf's padding is zero.
- **The scatter** writes only into columns `index`, also real; the exit re-mask zeroes everything else regardless.
- **Entries' one-hot `'w'`** on uniform is `index_to_one_hot(index, (N,) * d)`, stacked: the padded length `N`, the
  one in the real column, zeros beyond `N_i`. That equals `pack_vectors` of the ragged one-hot vectors, and is in the
  result dtype.
- **A numpy index of `N_i` or more** is out of range on ragged and raises there. On uniform it lands in a masked
  column (zero), or raises only at `N`. That is a silent difference, but the engine checks no values (O-18), and the
  checkers (`check_sample`) are where `index < N_i` is tested. The docstring will say so.

## 5. The tests, and their size

**One uniform call proves three things.** Every uniform input is built with garbage (1e3 times random) in every masked
slot: the padding of the cores, of `V`/`H`/`Vc`, of `D_dot`/`L_dot`, of `N` in the sample and in `y`, and of the
ranks across a varying-rank `C`. Each uniform call is then compared with ragged on its real parts:
- **equivalence**: `to_uniform -> op -> to_ragged == op_ragged`, per `C` element (each its own ragged train when the
  ranks vary);
- **garbage robustness**: the garbage changes nothing real;
- **exact masks**: every core-shaped output's padding is exactly zero, and every variable's too (start values
  masked). The masks are derived from the ragged ranks with `np.arange(size) < rank`, not from the engine's masks.

So there is no separate clean pass.

**The covering design.**
- The factors are:
  - the 8 `_CONFIGS` entries (no stack, `C`, `K`, `K + C`, forced-larger padding, forced padding with `C`, `C = (2, 3)`,
    `K = (2, 3)`), plus a ranks-varying-across-`C` configuration (9);
  - the jet cases `()`, `(2,)` with a line, `(1, 1)`, and mixed input jet sizes `(1, 2)` (4);
  - the call type (2);
  - the kind (3; entries at `()`).
- **Every configuration with every jet case once for probe and apply** (2 × 36 calls). Entries gets every
  configuration once (9). The call type alternates. That is 81 uniform calls, every name each.
- **Why the pairs and not more:**
  - every uniform-specific bug P11 can have is in masking (a configuration's concern) or in packing (a jet case's);
  - their pairwise interaction is where a mask meets a packed jet axis;
  - kind and call type touch which rows run, which P5 to P10 already cover on ragged.
  - P10's maximal case has a uniform twin: forced padding, varying ranks across `C = (2, 3)`, `K = (2, 3)`,
    `W = (2, 2)`, `(1, 1)`, one per kind.
- **Cost:** about 81 uniform calls plus about 300 ragged per-element calls. On P10's measure (54 stacked plus 380
  unstacked calls took 10 s), that is about 12 s single-process.
- **The other tests:**
  - **the packed jet zeros**: write a nonzero into a jet slot packing filled, on one probe case at `(2,)` with mixed
    jet sizes; the uniform result equals the ragged call whose input jet has that slot added;
  - **`Kp` at TT slack**: a frame whose `D`/`L` masks are a strict subset of its point's. Garbage in the padding of
    `core_directions` leaves the real blocks unchanged. A built-in negative control, `unittest.mock.patch` making the
    direction mask a no-op, must turn the comparison red, so the mutation check is permanent;
  - **`pack_vectors(len_J)`** beside the packing tests;
  - **dispatch**: the engine jitted on uniform data; program size equal at `(2,)` and `(6,)`; an eager call twice
    adds no compilation (the jax compile cache counted around the second call);
  - **the sharding class**: `g` and `Kp` in one call per program, at `()`, `(3,)` and `(1, 1)` on uniform, and at
    `(3,)` on ragged. Four compiled programs, not twelve.
- **Estimated cost at `-n 2`:** the matrix about 6 s; jit and sharding about 15 to 25 s of compilation; the rest a
  few seconds. About 30 s added to the suite's wall time.

## 6. Plan changes I'd propose

1. **The whole steps gain a `layout` static** (section 2), to return packed arrays on uniform; not in the plan's text.
2. **`y` is masked on entry** (probe's mode padding), and **the start values are masked** (zero padding on every
   variable). Both are new in the plan's entry table.
3. **The engine's output is always in the uniform layout** on a uniform call. Mirroring the sample's packedness, as
   `utv_probe` does today, is the twins' job (P12).
4. **`_check_stacks`'s uniform branch also checks the packed sample's `d` and `N`.**
5. **The tests run garbage-padded only**, on the covering design above. The sharding class runs four programs.
6. **A note for P12's twins:** a backend user jitting the engine on raw `.data` tuples must keep the masks static. The
   engine raises on a traced mask (today's `require_concrete_masks`) rather than computing wrongly.
