"""Probing objective, gradient and one Gauss-Newton Hessian-vector product on the manifold,
written at the level of the proposed sweep building blocks (backend style: tuples of arrays).

A sketch of the PROPOSED interface, not library code: numpy only, one sample axis W, no K or C stacks,
order 0 (no jets). The frame construction and the gauge projection are existing T3Toolbox functions.
The second half checks every result against the current library.

Usage:  python3 probe_fit_step_example.py /path/to/T3Toolbox
"""
import sys
import numpy as np

# ------------------------------------------------------------------------------------------------
# The building blocks: seven rules and two drivers
# ------------------------------------------------------------------------------------------------
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
    """Apply one rule at every site. `terms` is a list of operand tuples; each operand is a sequence
    over sites. The tuples are the terms of a sum and all share the rule's leg pattern."""
    d = min(len(op) for op in terms[0])
    return [sum(_term(rule, [op[i] for op in t], sum_w) for t in terms) for i in range(d)]


def sweep(rule, start, terms, reverse=False):
    """Thread a carry along the chain. Returns the message on every bond 0..d (bond-indexed).
    Left to right: out[0] = start, out[i+1] from site i. Reversed: out[d] = start, out[i] from site i."""
    d = min(len(op) for op in terms[0] if op is not CARRY)
    out = [None] * (d + 1)
    out[d if reverse else 0] = carry = start
    for i in (reversed(range(d)) if reverse else range(d)):
        carry = sum(_term(rule, [carry if op is CARRY else op[i] for op in t]) for t in terms)
        out[i if reverse else i + 1] = carry
    return out


def main(repo):
    sys.path.append(repo)
    import t3toolbox.backend.t3_constructors as t3_constructors
    import t3toolbox.backend.fv_conversions as fv_conversions
    import t3toolbox.backend.tv_operations as tv_operations

    np.random.seed(0)
    rng = np.random.default_rng(0)

    # --------------------------------------------------------------------------------------------
    # 1. A point, some samples, some data
    # --------------------------------------------------------------------------------------------
    shape, tucker_ranks, tt_ranks, W = (10, 11, 12, 9), (4, 5, 4, 3), (1, 3, 4, 3, 1), 20
    d = len(shape)
    B, G = t3_constructors.t3_corewise_randn(shape, tucker_ranks, tt_ranks)        # the point x = (B, G)
    w = [rng.standard_normal((W, N)) for N in shape]                               # sample vectors
    y = [rng.standard_normal((W, N)) for N in shape]                               # data: one vector per mode

    b_L, b_R = np.ones((W, tt_ranks[0])), np.ones((W, tt_ranks[-1]))               # boundary vectors
    zero_L, zero_R = np.zeros_like(b_L), np.zeros_like(b_R)

    # --------------------------------------------------------------------------------------------
    # 2. Objective: probe the point (plain grade; every frame variant is the point's own core)
    # --------------------------------------------------------------------------------------------
    xi  = sitewise(UP, [(B, w)])                                    # mode edges, up
    mu  = sweep(RIGHT, b_L, [(CARRY, G, xi)])                       # bonds 0..d, from the left
    nu  = sweep(LEFT,  b_R, [(G, xi, CARRY)], reverse=True)         # bonds 0..d, from the right
    eta = sitewise(DOWN, [(mu, G, nu[1:])])                         # mode edges, down
    z   = sitewise(OUT,  [(eta, B)])                                # probes of x

    r = [z_i - y_i for z_i, y_i in zip(z, y)]                       # residual
    objective = 0.5 * sum(np.sum(r_i ** 2) for r_i in r)

    # --------------------------------------------------------------------------------------------
    # 3. Orthogonal frame at the point (existing library function)
    # --------------------------------------------------------------------------------------------
    frame, _ = fv_conversions.t3_orthogonal_representations((B, G))
    D, U, L, R = frame

    # --------------------------------------------------------------------------------------------
    # 4. Frame sweep: the plain grade in the manifold frame. Computed once, reused below.
    # --------------------------------------------------------------------------------------------
    xi  = sitewise(UP, [(D, w)])
    mu  = sweep(RIGHT, b_L, [(CARRY, L, xi)])
    nu  = sweep(LEFT,  b_R, [(R, xi, CARRY)], reverse=True)
    eta = sitewise(DOWN, [(mu, U, nu[1:])])

    # --------------------------------------------------------------------------------------------
    # 5. Jacobian transpose: the residual grade `z`. One residual is inserted, on a leaf.
    # --------------------------------------------------------------------------------------------
    def jacobian_transpose(w_z):
        xi_z  = sitewise(UP, [(D, w_z)])
        mu_z  = sweep(RIGHT, zero_L, [(CARRY, L, xi), (mu, L, xi_z)])
        nu_z  = sweep(LEFT,  zero_R, [(R, xi, CARRY), (R, xi_z, nu[1:])], reverse=True)
        eta_z = sitewise(DOWN, [(mu_z, U, nu[1:]), (mu, U, nu_z[1:])])
        dH = sitewise(GRAD_G, [(mu_z, xi, nu[1:]), (mu, xi_z, nu[1:]), (mu, xi, nu_z[1:])], sum_w=True)
        dV = sitewise(GRAD_B, [(eta_z, w), (eta, w_z)], sum_w=True)
        return dV, dH

    # --------------------------------------------------------------------------------------------
    # 6. Jacobian: the tangent grade `t`. One core is replaced by its variation.
    # --------------------------------------------------------------------------------------------
    def jacobian(V, H):
        xi_t  = sitewise(UP, [(V, w)])
        mu_t  = sweep(RIGHT, zero_L, [(CARRY, R, xi), (mu, H, xi), (mu, U, xi_t)])
        nu_t  = sweep(LEFT,  zero_R, [(L, xi, CARRY), (H, xi, nu[1:]), (U, xi_t, nu[1:])], reverse=True)
        eta_t = sitewise(DOWN, [(mu_t, R, nu[1:]), (mu, H, nu[1:]), (mu, L, nu_t[1:])])
        return sitewise(OUT, [(eta_t, D), (eta, V)])

    project = lambda v: tv_operations.tv_orthogonal_gauge_projection(frame, v)   # existing library function

    # --------------------------------------------------------------------------------------------
    # 7. Gradient and one Gauss-Newton Hessian-vector product
    # --------------------------------------------------------------------------------------------
    gradient = project(jacobian_transpose(r))

    p = project(([rng.standard_normal(V.shape) for V in gradient[0]],            # a tangent direction
                 [rng.standard_normal(H.shape) for H in gradient[1]]))
    Hp = project(jacobian_transpose(jacobian(*p)))

    # ============================================================================================
    # Check against the current library
    # ============================================================================================
    import t3toolbox.tucker_tensor_train as t3
    import t3toolbox.frame_variations_format as fvf
    import t3toolbox.manifold as t3m
    import t3toolbox.fitting as fitting
    import t3toolbox.backend.probing as probing

    def rel(mine, theirs):
        flat = lambda v: np.concatenate([np.ravel(a) for part in v for a in part]) if isinstance(v[0], (list, tuple)) \
            else np.concatenate([np.ravel(a) for a in v])
        a, b = flat(mine), flat(theirs)
        return np.linalg.norm(a - b) / np.linalg.norm(b)

    model = fitting.probe_model(t3m.MANIFOLD, t3.TuckerTensorTrain(B, G), w, r)
    same_frame = all(np.array_equal(a, b) for fa, fb in zip(frame, model.frame.data) for a, b in zip(fa, fb))
    p_tangent = t3m.T3Tangent(model.frame, fvf.T3Variations(*p))

    print('objective %.6e   (library %.6e)' % (objective, float(model.objective_value)))
    print('same frame as the library model:', same_frame)
    print('relative differences from the library:')
    print('  probes of the point       %.1e' % rel(z, probing.t3_probe(w, (B, G))))
    print('  bare J^T r                %.1e' % rel(jacobian_transpose(r), probing.tv_probe_transpose(r, w, frame, sum_over_probes=True)))
    print('  J p                       %.1e' % rel(jacobian(*p), probing.tv_probe(w, p, frame)))
    print('  gradient                  %.1e' % rel(gradient, model.gradient.data[1].data))
    print('  Gauss-Newton Hessian * p  %.1e' % rel(Hp, model.gn_hessian(p_tangent).data[1].data))


if __name__ == '__main__':
    main(sys.argv[1])
