# Worked example

> **Status after the review of 2026-11-14.** The example stands in substance: it computes the value with its own sweep, which is the route kept after the review. Its names are the old ones. `scripts/review/jet_steps_check.py` is the same idea with the decided names (`xi_term`, `mu_dot_step`, `_tilde`, `gamma`), with jets and residual weights. See [DESIGN_AFTER_REVIEW.md](DESIGN_AFTER_REVIEW.md). One known inconsistency: the example already uses bond indexing and boundary operands that the proposal's order of work places at a later step (D5).

The probing objective, its gradient and one Gauss-Newton Hessian-vector product on the manifold take about 30 lines when written with seven rule strings and two drivers. Every result matches the current library to about 7e-16.

This is the explicit term-list layer of the proposal (layer 2 in [PROPOSAL.md](PROPOSAL.md)): backend style, on tuples of arrays. The runnable file is `scripts/probe_fit_step_example.py`.

## The building blocks

Two drivers do all the work: `sitewise` applies a rule at every site, and `sweep` threads a carry along the chain. Each takes one rule string and a list of operand tuples, one tuple per term of the sum.

```python
UP, RIGHT, LEFT, DOWN, OUT = 'io,o->i', 'a,aib,i->b', 'aib,i,b->a', 'a,aib,b->i', 'i,io->o'
GRAD_G, GRAD_B = 'a,i,b->aib', 'i,o->io'
CARRY = object()          # marks the slot a sweep threads through


def _term(rule, ops, sum_w=False):
    """One contraction. An operand with one more axis than its legs carries the sample axis W."""
    ins, out = rule.split('->')
    ins = ins.split(',')
    subs = ','.join(('w' + s) if op.ndim == len(s) + 1 else s for s, op in zip(ins, ops))
    return np.einsum(subs + '->' + ('' if sum_w else 'w') + out, *ops)


def sitewise(rule, terms, sum_w=False):
    """Apply one rule at every site. Each operand in a tuple is a sequence over sites."""
    d = min(len(op) for op in terms[0])
    return [sum(_term(rule, [op[i] for op in t], sum_w) for t in terms) for i in range(d)]


def sweep(rule, start, terms, reverse=False):
    """Thread a carry along the chain. Returns the message on every bond 0..d."""
    d = min(len(op) for op in terms[0] if op is not CARRY)
    out = [None] * (d + 1)
    out[d if reverse else 0] = carry = start
    for i in (reversed(range(d)) if reverse else range(d)):
        carry = sum(_term(rule, [carry if op is CARRY else op[i] for op in t]) for t in terms)
        out[i if reverse else i + 1] = carry
    return out
```

The conventions the example relies on:

- **One rule, many tuples.** All terms of a sum have the rule's leg pattern, so a sum is a list of operand tuples under one string.
- **`CARRY`** marks the slot that receives the running message. A term without it is a source term.
- **Bond variables are indexed by bond.** A sweep returns `d+1` messages. `mu[i]` is the left input of core `i`, and `nu[1:][i]` is its right input.
- **Stacks come from the operands.** Here the only stack is the sample axis `W`, detected from the operand's shape. In the library this is where operand signatures for `W`, `K` and `C` go.
- **`sum_w`** sums the sample axis in the gradient rules. It is the `sum_over_probes` option of the per-site driver.

`CARRY` is a bare `object()` here only for brevity. In the library it should be a named singleton, such as an enum member, so that it prints readably and survives pickling. `None` is wrong because a forgotten operand would silently be treated as the carry. A string is wrong because it is itself a sequence and would not fail loudly when misplaced.

## The example

Seven steps: a point and data, the objective, the frame, the frame sweep, the Jacobian transpose, the Jacobian, and then the gradient and Hessian-vector product assembled from them.

```python
# 1. A point, some samples, some data
B, G = t3_corewise_randn(shape, tucker_ranks, tt_ranks)           # the point x = (B, G)
w = [rng.standard_normal((W, N)) for N in shape]                  # sample vectors
y = [rng.standard_normal((W, N)) for N in shape]                  # data, one vector per mode
b_L, b_R = np.ones((W, tt_ranks[0])), np.ones((W, tt_ranks[-1]))  # boundary vectors
zero_L, zero_R = np.zeros_like(b_L), np.zeros_like(b_R)

# 2. Objective: probe the point (plain grade; every variant is the point's own core)
xi  = sitewise(UP, [(B, w)])                                # mode edges, up
mu  = sweep(RIGHT, b_L, [(CARRY, G, xi)])                   # bonds 0..d, from the left
nu  = sweep(LEFT,  b_R, [(G, xi, CARRY)], reverse=True)     # bonds 0..d, from the right
eta = sitewise(DOWN, [(mu, G, nu[1:])])                     # mode edges, down
z   = sitewise(OUT,  [(eta, B)])                            # probes of x
r   = [z_i - y_i for z_i, y_i in zip(z, y)]                 # residual
objective = 0.5 * sum(np.sum(r_i ** 2) for r_i in r)

# 3. Orthogonal frame at the point (existing library function)
frame, _ = t3_orthogonal_representations((B, G))
D, U, L, R = frame

# 4. Frame sweep: the plain grade in the manifold frame. Computed once, reused below.
xi  = sitewise(UP, [(D, w)])
mu  = sweep(RIGHT, b_L, [(CARRY, L, xi)])
nu  = sweep(LEFT,  b_R, [(R, xi, CARRY)], reverse=True)
eta = sitewise(DOWN, [(mu, U, nu[1:])])

# 5. Jacobian transpose: the residual grade z. One residual is inserted, on a leaf.
def jacobian_transpose(w_z):
    xi_z  = sitewise(UP, [(D, w_z)])
    mu_z  = sweep(RIGHT, zero_L, [(CARRY, L, xi), (mu, L, xi_z)])
    nu_z  = sweep(LEFT,  zero_R, [(R, xi, CARRY), (R, xi_z, nu[1:])], reverse=True)
    eta_z = sitewise(DOWN, [(mu_z, U, nu[1:]), (mu, U, nu_z[1:])])
    dH = sitewise(GRAD_G, [(mu_z, xi, nu[1:]), (mu, xi_z, nu[1:]), (mu, xi, nu_z[1:])], sum_w=True)
    dV = sitewise(GRAD_B, [(eta_z, w), (eta, w_z)], sum_w=True)
    return dV, dH

# 6. Jacobian: the tangent grade t. One core is replaced by its variation.
def jacobian(V, H):
    xi_t  = sitewise(UP, [(V, w)])
    mu_t  = sweep(RIGHT, zero_L, [(CARRY, R, xi), (mu, H, xi), (mu, U, xi_t)])
    nu_t  = sweep(LEFT,  zero_R, [(L, xi, CARRY), (H, xi, nu[1:]), (U, xi_t, nu[1:])], reverse=True)
    eta_t = sitewise(DOWN, [(mu_t, R, nu[1:]), (mu, H, nu[1:]), (mu, L, nu_t[1:])])
    return sitewise(OUT, [(eta_t, D), (eta, V)])

# 7. Gradient and one Gauss-Newton Hessian-vector product
project  = lambda v: tv_orthogonal_gauge_projection(frame, v)    # existing library function
gradient = project(jacobian_transpose(r))
Hp       = project(jacobian_transpose(jacobian(*p)))             # p: a gauged tangent (V, H)
```

The frame construction and the gauge projection are the existing `fv_conversions.t3_orthogonal_representations` and `tv_operations.tv_orthogonal_gauge_projection`. Everything else is the two drivers.

## Reading the example

Each line is one rule at one grade, and each corresponds to one current kernel.

| Line | Rule, grade | Current kernel |
| --- | --- | --- |
| `xi` | UP, plain | `compute_xi` |
| `mu` | RIGHT, plain | `compute_mu` |
| `nu` | LEFT, plain | `compute_nu` |
| `eta` | DOWN, plain | `compute_eta` |
| `z` | OUT, plain | `assemble_z` |
| `xi_z` | UP, `z` | `compute_deta_tilde` |
| `mu_z` | RIGHT, `z` | `compute_tau_tilde` |
| `nu_z` | LEFT, `z` | `compute_sigma_tilde` |
| `eta_z` | DOWN, `z` | `compute_dxi_tilde` |
| `dH` | GRAD_G, `z` | `assemble_tt_variations` |
| `dV` | GRAD_B, `z` | `assemble_tucker_variations` |
| `xi_t` | UP, `t` | `compute_dxi` |
| `mu_t` | RIGHT, `t` | `compute_sigma` |
| `nu_t` | LEFT, `t` | `compute_tau` |
| `eta_t` | DOWN, `t` | `compute_deta` |
| return of `jacobian` | OUT, `t` | `assemble_tangent_z` |

Things to notice:

- **The value needs its own sweep.** Step 2 runs through the point's cores, because a manifold frame has no plain probe output. Step 4 repeats the plain sweep in the frame.
- **The frame variant in each tuple follows the pointing rule.** In `mu_t` the carry term uses `R` because the hole is to the left, the `H` term is the hole itself, and the `xi_t` term uses `U` because the hole is in the Tucker core below.
- **Residual-grade terms use the variant pointed at the output.** `mu_z` flows right, so both its terms use `L`; `nu_z` uses `R`; `eta_z` uses `U`.
- **Apply and entries change only the seeds.** The scalar residual multiplies `b_R` as the start of `nu_z`, the leaf term drops out, and `mu_z` and `xi_z` vanish. `dH` and `dV` are then left with one term each.
- **Jets change only the product.** Each array gains a leading order axis and each contraction becomes a truncated polynomial product. The lines stay the same.

The Hessian-vector product shown is the Gauss-Newton one. The exact Hessian adds a curvature term, which needs the `tz` grade at the corewise frame and the lift between frame and core coordinates. It is not shown.

## Checked against the library

All five quantities agree with T3Toolbox at commit `89ac203f` to rounding error, and the frame is bit-identical to the one the library's model builds.

| Quantity | Compared with | Relative difference |
| --- | --- | --- |
| Probes of the point | `probing.t3_probe` | 2.9e-16 |
| Bare transpose applied to the residual | `probing.tv_probe_transpose` with `sum_over_probes=True` | 2.4e-16 |
| Jacobian applied to a tangent | `probing.tv_probe` | 3.1e-16 |
| Gradient | `fitting.probe_model(MANIFOLD, ...).gradient` | 4.0e-16 |
| Gauss-Newton Hessian-vector product | `.gn_hessian(p)` of the same model | 6.9e-16 |

The objective also matches the model's `objective_value`.

Scope of the sketch:

- **numpy only**, with Python loops over sites in place of scans.
- **One sample axis `W`.** No `K` or `C` stacks, and `W` is always summed in the gradients.
- **Order 0.** No jets.
- **Tails of size 1**, with boundary vectors of ones.
- **Explicit term lists.** The tuples are written by hand here. Generating them from grades is the next layer of the proposal.
