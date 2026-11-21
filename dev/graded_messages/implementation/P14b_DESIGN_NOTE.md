# P14b design note: `computed_names` in place of `keep=`

Implementer, 2026-11-21, against `graded-messages` at `0e71fec8`. Read-only so far. It is built to
`TARGET_DECISIONS.md`, the note "The hold list: `computed_names`, not `keep=`". `frame_form` is left as it is.

## 1. The signature

```python
def computed_names(                         # backend/sampling_variables.py, in __all__
        kind:              str,             # 'probe' | 'apply' | 'entries'
        frame:             typ.Tuple,       # any of the four forms (frame_form reads it)
        want:              typ.Sequence[str],
        order:             typ.Tuple[int, ...] = (),
        y=None, weight=None, variations=None, core_directions=None, Vc=None,
        sum_over_samples:  bool = True,
        held:              typ.Optional[HeldVariables] = None,
) -> typ.FrozenSet[str]:                    # the public names the call would compute
```

**The keywords are exactly those of `sampling_variables`, without `sample`**, so the oracle builds one dictionary
and passes it to both: `computed_names(kind, frame, want, **args)`, then `sampling_variables(kind, frame, sample,
want + extras, **args)`. The sample is not a fact of the closure, because it is always given.

**What it reads:**

| Argument | What it reads |
| --- | --- |
| `y`, `Vc`, `variations`, `core_directions` | **presence only** (`is not None`) |
| `weight`, `order`, `sum_over_samples` | nothing; accepted so the dictionary can be shared |
| `held` | its filled slots, and `held:Jp` when `Jp` is held |
| `frame` | its form, through `frame_form` (the layout does not matter to the closure) |

It reads no array and checks no shape. It does check what the engine checks before the closure, with the engine's
messages:
- an unknown kind;
- an entries call with jets;
- `want` as a bare string;
- `held` that is not a `HeldVariables`;
- unknown names (including a name that is zero for the kind);
- a bad frame structure;
- `variations` on a corewise point (the engine refuses it there);
- a wanted name whose argument is missing (the closure's own error).

So `computed_names` raises exactly where the engine would, for the same reason.

## 2. What it returns

- **Public names only,** as a `frozenset`: the names of the closure's rules that do not start with `_`. Private nodes
  (`_frame.*`, `_raw.*`, `_Jp.*`, the short nodes) are the graph's internals; exposing them would make the
  return change whenever the graph is refactored. The oracle needs only the hold list, which is public.
- **A `want` name that `held` fills is not in the result:** the engine reuses it and does not compute it. So the
  result is `want`'s computed names plus everything they read that is not held, and it is disjoint from
  `held.keys()`.
- **`'w'`** (the sample) is never in it: it is an argument.

## 3. One reading of the facts

Today the engine builds `present`, `held_names` and the call type inline (its step 2 and the start of step 3).
I would move that into one private function, `_closure_facts(kind, frame, want, held, given) -> (call,
held_names, present)`, called by both `sampling_variables` and `computed_names`. They then share one cached
`_closure` call on identical facts and cannot diverge; this is the review's concern from P14 (two readings of one
set of facts).

## 4. The oracle

```python
args = dict(order=..., y=batch.y, weight=batch.weight, variations=..., core_directions=..., Vc=Vc, held=held)
extras = tuple(sorted(set(HOLD) & computed_names(kind, frame, engine_want, **args) - set(engine_want)))
*values, *extra_values = sampling_variables(kind, frame, batch.sample, engine_want + extras, **args)
held_out = (held or HeldVariables()).with_values(**dict(zip(extras, extra_values)))
```

`held_out` is the `held` passed in plus the extras. A hold-list name the caller also wants is not an extra: its value
is in `values`, and it is held too (`held_out` takes it from `values`). This keeps P14's tested behaviour: `held`
equals `HOLD ∩ computed`.

## 5. Tests

- **`computed_names` equals what the engine runs.** The engine's executed rules are recorded by wrapping
  `sampling_steps` with a counter, in the style of P11's monitoring listener. The public names of the executed rules
  must equal `computed_names(...)`, checked:
  - for every kind, both forms (manifold and corewise), ragged and uniform;
  - on the routes with and without `Vc`, with a held `Jp`, with a held `r` or `mu`;
  - on the oracle's want patterns (`('f',)`, `('f', 'g')`, `('JtJp',)`, `('Jp_norm_sq',)`, `('Kp',)`, and with
    `held` from an earlier call).
- **Checks:** its errors are the engine's for the same arguments.
- **The oracle:**
  - `held.keys() == HOLD ∩ computed_names(...)`, plus the held passed in;
  - each held value equals a fresh engine call for that name;
  - P14's `keep` tests are replaced by these.
- **`keep` is gone:** passing `keep=` to `sampling_variables` raises a `TypeError`.

## 6. Anything I would change

Only §3, the shared private reading of the facts, which I recommend.
