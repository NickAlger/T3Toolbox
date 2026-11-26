### `sampling_variables` (`t3toolbox/backend/sampling_variables.py:1678`)

```python
def sampling_variables(
        kind:              str,                              # 'probe' | 'apply' | 'entries'
        frame:             typ.Tuple,                        # the base point (D, U, L, R), four tuples of d arrays; or
                                                             #   the point (B, G): X
        sample:            typ.Any,                          # ww: P_in_i⊗V_i, the input jets, len=d,
                                                             #   elm_shape=F_in_i+W+(Ni,);
                                                             #   entries: index^d, shape=(d,)+W, int
        want:              typ.Sequence[str],                # variable names; the return is aligned with them

        order:             typ.Tuple[int, ...] = (),         # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...)
        y:                 typ.Optional[typ.Any] = None,     # the data: Y*, of the type of z, each axis 1 or z's size:
                                                             #   probe len=d, elm_shape=F+W+C+(Ni,); apply F+W+C; entries W+C
        weight:            typ.Optional[typ.Any] = None,     # omega, of the type of z likewise (no K); None = ones
        variations:        typ.Optional[typ.Tuple] = None,   # X, (V, H), elm_shapes K+C+(nUi,Ni), K+C+(rLi,nDi,rR(i+1))
        core_directions:   typ.Optional[typ.Tuple] = None,   # X, (D_dot, L_dot), the shapes of D, L with K+C
        Vc:                typ.Optional[typ.Sequence[NDArray]] = None,   # the frame's Tucker centers: len=d,
                                                                         #   elm_shape=C+(nUi,Ni)
        sum_over_samples:  bool = True,                      # on the blocks; False: W joins their stack, W+K+C
        held:              typ.Optional[HeldVariables] = None,   # held variables; shapes checked, values trusted
) -> typ.Tuple[typ.Any, ...]:                                # aligned with want; types and shapes in the module's table
```

### `computed_names` (`t3toolbox/backend/sampling_variables.py:1580`)

```python
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
```

### `_check_data_space` (`t3toolbox/backend/sampling_variables.py:1172`)

```python
def _check_data_space(
        setting:  _Setting,
        name:     str,                      # 'y' or 'weight', for the message
        value:    typ.Any,                  # of the type of z: probe len=d (or a mode-leading array); else one array
        edges:    typ.Tuple[int, ...],      # probe: the mode edge of each array, N_i (ragged) or the padded N (packed)
) -> None:
```

### `_modes` (`t3toolbox/backend/sampling_steps.py:915`)

```python
def _modes(
        kind:    str,                       # 'probe' | 'apply' | 'entries'
        value:   typ.Any,                   # of the type of z: probe len=d (or packed (d,)+...), elm_shape=F+W+[K]+C+(Ni,); apply F+W+[K]+C; entries W+[K]+C
        weight:  typ.Any,                   # omega, of the type of z with [K] placed (_with_K): probe len=d or (d,)+...; else one array; None = ones
        xnp,
) -> typ.List[typ.Tuple[NDArray, typ.Any, int]]:     # (array, its weight's array or None, mode-edge axes)
```

### `_with_K` (`t3toolbox/backend/sampling_steps.py:926`)

```python
def _with_K(
        weight:  typ.Any,                   # omega, of the type of z with no K: probe len=d or (d,)+F+W+C+(e,); else F+W+C or W+C
        kind:    str,
        lens:    contractions.GroupLengths,
) -> typ.Any:                               # omega with lens.K size-1 axes after F + W: F+W+(1,)*K+C(+(e,))
```

### `_weighted_sum` (`t3toolbox/backend/sampling_steps.py:948`)

```python
def _weighted_sum(
        kind:    str,
        value:   typ.Any,                   # of the type of z (r, or Jp with K)
        weight:  typ.Any,                   # placed against value (with K where value has it); None = ones
        n_sum:   int,                       # the leading axes summed: the jet axes and W
        xnp,
) -> NDArray:                               # sum of value * data_metric(value, weight, None) over F, W and the edge
```

### `Jp_norm_sq_step` (`t3toolbox/backend/sampling_steps.py:993`)

```python
def Jp_norm_sq_step(
        Jp:      typ.Any,                   # of the type of z: probe len=d, elm_shape=F+W+K+C+(Ni,); apply F+W+K+C
        weight:  typ.Any,                   # omega, of the type of z (no K: placed here); None = ones
        *,
        lens:    contractions.GroupLengths,
        order:   typ.Tuple[int, ...],
        form:    str,
        kind:    str,
) -> NDArray:                               # <Jp, M Jp> at gamma = 1, keeping K + C. shape=K+C
```

### `w_tilde_step` (`t3toolbox/backend/sampling_steps.py:1009`)

```python
def w_tilde_step(
        r:       typ.Any,                   # of the type of z: probe len=d (or packed), elm_shape=F+W+[K]+C+(Ni,); apply F+W+[K]+C
        weight:  typ.Any,                   # omega, of the type of z (no K: placed here); None = ones
        *,
        lens:    contractions.GroupLengths,
        order:   typ.Tuple[int, ...],
        form:    str,
        kind:    str,
) -> typ.Any:                               # the seed: Y in the engine's turned coordinates, the type of z
```

### `r_step` (`t3toolbox/backend/sampling_steps.py:1048`)

```python
def r_step(
        z:      typ.Any,                    # of the type of z: probe len=d (or packed), elm_shape=F+W+C+(Ni,); apply F+W+C
        y:      typ.Any,                    # the data, of the type of z: each axis 1 or z's size (checked on entry)
        *,
        lens:   contractions.GroupLengths,
        order:  typ.Tuple[int, ...],
        form:   str,
        kind:   str,
) -> typ.Any:                               # r = z - y, of the type of z
```

### `f_step` (`t3toolbox/backend/sampling_steps.py:1066`)

```python
def f_step(
        r:       typ.Any,                   # of the type of z: probe len=d, elm_shape=F+W+C+(Ni,); apply F+W+C
        weight:  typ.Any,                   # omega, of the type of z; None = ones
        *,
        lens:    contractions.GroupLengths,
        order:   typ.Tuple[int, ...],
        form:    str,
        kind:    str,
) -> NDArray:                               # f = 1/2 <r, M r> at gamma = 1, keeping C. shape=C
```

### `r_block_sumsq_step` (`t3toolbox/backend/sampling_steps.py:1083`)

```python
def r_block_sumsq_step(
        r:       typ.Any,                   # Y*, of the type of z: probe len=d, elm_shape=F+W+C+(Ni,); apply F+W+C
        weight:  typ.Any,                   # omega, of the type of z; None = ones
        *,
        lens:    contractions.GroupLengths,
        order:   typ.Tuple[int, ...],
        form:    str,
        kind:    str,
) -> NDArray:                               # <r, M r> at gamma = 1 per block: the block shape (d,)+F | F | ()
```

### `data_metric` (`t3toolbox/backend/jets.py:579`)

```python
def data_metric(
        r:                typ.Any,                  # Y*, of the type of z, no K: probe len=d (or packed (d,)+...), elm_shape=F+W+C+(Ni,); apply F+W+C; entries W+C
        weight:           typ.Optional[typ.Any],    # omega, of the type of z: each axis 1 or r's size; None = ones
        minibatch_scale:  typ.Optional[typ.Any],    # gamma: a number or a 0-d array (traced under jit); None = 1
) -> typ.Any:                                       # M r: Y, gamma * omega**2 * r elementwise, r's structure (a tuple for a tuple or a list)
```

### `blocks_to_weight` (`t3toolbox/backend/jets.py:767`)

```python
def blocks_to_weight(
        kind:    str,                               # 'probe' | 'apply' | 'entries'
        order:   typ.Tuple[int, ...],               # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...). entries: ()
        blocks:  typ.Any,                           # one value per block: probe (d,)+F; apply F; entries ()
        y:       typ.Any,                           # Y*, of the type of z: only its rank is read (C included)
) -> NDArray:                                       # omega: probe (d,)+F+(1,)*(len W+len C)+(1,); apply F+(1,)*(len W+len C); entries (1,)*(len W+len C)
```

### `jet_to_dual_jet` (`t3toolbox/backend/jets.py:324`)

```python
def jet_to_dual_jet(
        kind:    str,                      # 'probe' | 'apply' | 'entries': which type the jet is of
        order:   typ.Tuple[int, ...],      # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...). entries: ()
        jet:     typ.Any,                  # Y*, of the type of z: probe len=d (or packed), elm_shape=F+W+[K]+C+(Ni,); apply F+W+[K]+C; entries W+[K]+C
        weight:  typ.Optional[typ.Any],    # omega, of the type of z (each axis 1 or the jet's size, no K); None = ones
) -> typ.Any:                              # Y, turned: data_metric(jet, weight, None), every jet axis flipped. probe: a tuple
```

### `dual_jet_to_jet` (`t3toolbox/backend/jets.py:456`)

```python
def dual_jet_to_jet(
        kind:      str,                    # 'probe' | 'apply' | 'entries': which type the dual jet is of
        order:     typ.Tuple[int, ...],    # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...). entries: ()
        dual_jet:  typ.Any,                # Y, turned: same type and shapes as jet_to_dual_jet's result
        weight:    typ.Optional[typ.Any],  # omega, of the type of z (each axis 1 or the dual jet's size); None = ones
) -> typ.Any:                              # Y*: every jet axis flipped back, then divided by weight**2. probe: a tuple
```

### `derivative_weight_to_taylor_weight` (`t3toolbox/backend/jets.py:997`)

```python
def derivative_weight_to_taylor_weight(
        kind:    str,                      # 'probe' | 'apply' | 'entries': which type the weight is of
        order:   typ.Tuple[int, ...],      # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...). entries: ()
        weight:  typ.Any,                  # omega on derivative residuals, of the type of z: probe len=d or (d,)+...; else one array
) -> typ.Any:                              # omega on Taylor residuals, in the form given: slot k times k_1! ... k_m!
```

### `prepare_derivative_data` (`t3toolbox/backend/jets.py:1080`)

```python
def prepare_derivative_data(
        kind:        str,                              # 'probe' | 'apply' | 'entries'
        order:       typ.Tuple[int, ...],              # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...). entries: ()
        y:           typ.Any,                          # Y*, in derivative coordinates: probe len=d,
                                                       #   elm_shape=F+W+C+(Ni,); apply F+W+C; entries W+C
        weight:      typ.Optional[typ.Any] = None,     # omega on derivative residuals, of the type of z; None = ones on the derivatives
        validation:  typ.Optional[typ.Any] = None,     # Y*, validation data in derivative coordinates, or None
) -> typ.Tuple[
        typ.Any,                                       # y: Y*, in Taylor coordinates, same shapes
        typ.Any,                                       # omega, the Taylor weight, of the type of z: k_1! ... k_m! per
                                                       #   slot when weight is None
        typ.Optional[typ.Any],                         # validation: Y*, in Taylor coordinates, or None
]:
```

### `check_data` (`t3toolbox/backend/jets.py:1568`)

```python
def check_data(
        kind:         str,                      # 'probe' | 'apply' | 'entries'
        order:        typ.Tuple[int, ...],      # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...). entries: ()
        sample:       typ.Any,                  # P_in_i⊗V_i (the input jets ww) | index^d, a good sample
        y:            typ.Any,                  # Y*, of the type of z: probe len=d (or (d,)+...), elm_shape=F+W+C+(Ni,); apply F+W+C; entries W+C
        stack_shape:  typ.Tuple[int, ...] = (), # C, the model's stack of base points; () for an unstacked model
) -> None:
```

### `check_weight` (`t3toolbox/backend/jets.py:1657`)

```python
def check_weight(
        kind:         str,                      # 'probe' | 'apply' | 'entries'
        order:        typ.Tuple[int, ...],      # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...). entries: ()
        sample:       typ.Any,                  # P_in_i⊗V_i (the input jets ww) | index^d, a good sample
        weight:       typ.Optional[typ.Any],    # omega, of the type of z (each axis 1 or z's size), or None
        stack_shape:  typ.Tuple[int, ...] = (), # C, the model's stack of base points; () for an unstacked model
) -> None:
```

### `LeastSquaresBatch` (`t3toolbox/backend/fitting.py:68`)

```python
class LeastSquaresBatch:
    sample:           typ.Any                                    # P_in_i⊗V_i (the input jets ww) | index^d (entries)
    y:                typ.Any = None                             # Y*: probe len=d, F+W+C+(Ni,); apply F+W+C; entries W+C
    weight:           typ.Any = None                             # omega, of the type of z likewise; None = ones
    minibatch_scale:  typ.Optional[typ.Any] = None               # gamma: n / batch_size on a drawn batch; None = 1
    drawn_indices:    typ.Optional[NDArray] = None               # int, (batch_size,): the draw, flat over W
    def data_metric(
            self,
            r:  typ.Any,        # Y*, of the type of z (no K), on this batch's samples
    ) -> typ.Any:               # M r: Y, minibatch_scale * weight**2 * r elementwise, r's structure
```

### `LeastSquaresOracle` (`t3toolbox/backend/fitting.py:290`)

```python
class LeastSquaresOracle(ValueHashedFields):
    kind:   str                                         # 'probe' | 'apply' | 'entries'
    order:  typ.Tuple[int, ...]                         # the jet shape; required (O-44); entries: () only (O-46)
    optional_outputs: typ.ClassVar[typ.Tuple[str, ...]] = (
    def __post_init__(self):
    def query(
            self,
            frame:            typ.Tuple,                      # a manifold frame or a corewise point (frame_form)
            batch:            LeastSquaresBatch,              # its data
            want:             typ.Sequence[str],              # f, g, JtJp, Jp_norm_sq; optional: optional_outputs
            Vc:               typ.Optional[typ.Sequence[NDArray]] = None,
            variations:       typ.Optional[typ.Tuple] = None, # X, (V, H): JtJp, Jp_norm_sq, Jp; raw on a point
            core_directions:  typ.Optional[typ.Tuple] = None, # X, (D_dot, L_dot): Kp
            held:             typ.Any = None,                 # a HeldVariables from an earlier query here, or None
    ) -> typ.Tuple[
            typ.Tuple[typ.Any, ...],                          # values, aligned with want: r, Jp Y*; g, JtJp, Kp X*
                                                              #   (before the projection); f, Jp_norm_sq, f_resolution
                                                              #   ℝ; r_block_sumsq, y_block_sumsq ℝ per block
            typ.Any,                                          # held: a HeldVariables
    ]:
    def check_batch(
            self,
            batch:        LeastSquaresBatch,                  # ragged (check before packing)
            shape:        typ.Sequence[int],                  # the mode sizes (N0, ..., N(d-1))
            stack_shape:  typ.Tuple[int, ...] = (),           # C, the model's stack of base points; () unstacked
    ) -> None:
    def _w_axes(self, batch):
    def _W(self, batch) -> typ.Tuple[int, ...]:
    def n_measurements(
            self,
            batch:  LeastSquaresBatch,
    ) -> int:                                                 # |W|, the number of samples
    def take(
            self,
            batch:    LeastSquaresBatch,
            indices:  NDArray,                                # int, (b,): flat positions over W; len(indices) static
    ) -> LeastSquaresBatch:                                   # the drawn batch, W flattened to (b,)
```

### `pack_data` (`t3toolbox/backend/uniform_fitting.py:61`)

```python
def pack_data(
        kind:   str,                        # 'probe' | 'apply' | 'entries'
        order:  typ.Tuple[int, ...],        # the jet shape (the probe data's leading F axes ride along)
        y:      typ.Any,                    # Y* or omega, of the type of z (the data, a weight, a residual):
                                            #   probe len=d, elm_shape=F+W+C+(Ni,); apply F+W+C; entries W+C; each axis
                                            #   1 or z's size
        N:      int,                        # the packed mode width
) -> typ.Any:                               # probe: (d,)+F+W+C+(e,), e = 1 or N; apply, entries: y unchanged
```

### `pack_batch` (`t3toolbox/backend/uniform_fitting.py:94`)

```python
def pack_batch(
        kind:   str,                                # 'probe' | 'apply' | 'entries'
        order:  typ.Tuple[int, ...],                # the jet shape
        batch:  'bfit.LeastSquaresBatch',           # ragged (or already packed: kept)
        N:      int,                                # the packed mode width
) -> 'bfit.LeastSquaresBatch':                      # the sample's vectors and the probe data packed to N
```

### `_pack_data_space` (`t3toolbox/backend/ut3_operations.py:211`)

```python
def _pack_data_space(
        value,          # a probe data-space object: packed (returned as is), or len=d arrays, elm_shape=F+W+C+(e_i,)
        N:  int,        # the packed mode width
) -> NDArray:           # (d,)+F+W+C+(e,): each axis of F+W+C the join across the modes; e = 1 if every e_i is 1, else N
```

### `flat_draw` (`t3toolbox/backend/optimizers.py:625`)

```python
def flat_draw(
        oracle:      typ.Any,       # an oracle with n_measurements(batch) and take(batch, indices)
        batch:       typ.Any,       # the full batch
        batch_size:  int,           # samples per minibatch
) -> typ.Callable:                  # draw(rng) -> a drawn batch
```

### `_require_draw` (`t3toolbox/backend/optimizers.py:644`)

```python
def _require_draw(
        oracle:  typ.Any,
        draw:    typ.Optional[typ.Callable],   # the user's draw, or None (the default flat_draw)
        who:     str,
) -> None:
```

### `make_newton_display` (`t3toolbox/backend/optimizer_display.py:549`)

```python
def make_newton_display(
        oracle,                                   # the fit's Oracle (its block sums: r_block_sumsq, y_block_sumsq)
        batch,                                    # its batch (the training data: the denominators)
        val_batch:   typ.Any = None,              # optional validation batch, laid out like batch -> a train|val table
        print_fn:    typ.Optional[typ.Callable] = print,   # where to send each row's text (None = silent)
        fmt:         str  = '%.1e',
        record:      bool = True,                 # accumulate per-row records (scalars + error arrays)
        error_table: bool = True,                 # False: print the header lines only (the records are unchanged)
) -> typ.Tuple[typ.Callable, list]:               # (callback for newton_cg, records list filled as it runs)
```

### `_is_packed` (`t3toolbox/fitting.py:582`)

```python
def _is_packed(oracle, batch) -> bool:
```

### `prepare_batch` (`t3toolbox/fitting.py:589`)

```python
def prepare_batch(
        oracle,
        batch,
        x,                           # the point: its shape, and on the uniform layer its width N
        check:  bool = True,         # False: pack only (a draw's batch, made by the caller's draw every step)
):                                   # the batch the backend takes (packed on the uniform layer)
```

### `_frontend_draw` (`t3toolbox/optimizers.py:117`)

```python
def _frontend_draw(oracle, draw, x0):
```

### `_old_fit` (`t3toolbox/optimizers.py:171`)

```python
def _old_fit(
        kind:         str,                 # the old kind name ('probe', 'apply_derivatives', ...), case-insensitive
        sample:       typ.Any,             # ww / index, or (ww, pp) / (index, pp)
        data:         typ.Any,             # the old data (derivatives for a derivative kind)
        shape:        typ.Tuple[int, ...], # the mode sizes
        order:        typ.Optional[int],   # derivative kinds: the highest order
        weight:       typ.Any,             # the old residual weight, or None
        who:          str,
        stack_shape:  typ.Tuple[int, ...] = (),   # the point's stack C (the old data are shared by it)
) -> typ.Tuple[LeastSquaresOracle, LeastSquaresBatch]:
```

### `_old_draw` (`t3toolbox/optimizers.py:221`)

```python
def _old_draw(draw, kind, shape, order, weight, scale, who, stack_shape=()):
```

