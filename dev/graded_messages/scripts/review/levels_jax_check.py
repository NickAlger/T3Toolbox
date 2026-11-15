"""Do blocking LEVELS (scan over axes of the sample stack) hold up under jax: jit, nested scans, the uniform
layout, jets, padding with a mask, and sharding across devices?

The per-block kernels are the LIBRARY'S OWN (probing, entries, the jet kind, the uniform probe kind); only the
loop over a sample-stack axis is new. Nothing is timed. Reported: differences from an unblocked numpy or
unsharded reference, the number of all-gathers in the compiled program, and XLA's own scratch-memory figure.

Usage: python3 levels_jax_check.py /path/to/T3Toolbox      (sets up 4 virtual CPU devices)
"""
import inspect
import os
import sys

os.environ['XLA_FLAGS'] = (os.environ.get('XLA_FLAGS', '') + ' --xla_force_host_platform_device_count=4').strip()
sys.path.append(sys.argv[1])

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

jax.config.update('jax_enable_x64', True)

import t3toolbox.backend.t3_constructors as t3c
import t3toolbox.backend.fv_conversions as fvc
import t3toolbox.backend.probing as probing
import t3toolbox.backend.entries as entries
import t3toolbox.backend.fitting as bfit

tmap = jax.tree.map
flat = lambda v: np.concatenate([np.ravel(np.asarray(a)) for a in jax.tree.leaves(v)])
rel = lambda a, b: float(np.linalg.norm(flat(a) - flat(b)) / np.linalg.norm(flat(b)))


# ----------------------------------------------------------------------------------------------------
# The one new thing: scan a per-block kernel over one axis of the sample stack.
# ----------------------------------------------------------------------------------------------------
def scan_axis(kernel, per_sample, axis):
    """kernel(block) -> (summed, mapped).  `per_sample` is a pytree; `axis` is the ARRAY axis to loop over, or a
    pytree of the same structure giving it per leaf (the sample stack sits at different axes in different operands).
    Summed parts are added across the loop; mapped parts come back with the looped axis leading."""
    axes = tmap(lambda a: axis, per_sample) if isinstance(axis, int) else axis
    take = lambda i: tmap(lambda a, ax: a[(slice(None),) * ax + (i,)], per_sample, axes)
    n = [a.shape[ax] for a, ax in zip(jax.tree.leaves(per_sample), jax.tree.leaves(axes))][0]
    shapes = jax.eval_shape(lambda: kernel(take(0)))
    init = tmap(lambda s: jnp.zeros(s.shape, s.dtype), shapes[0])

    def body(carry, i):
        summed, mapped = kernel(take(i))
        return tmap(jnp.add, carry, summed), mapped

    return lax.scan(body, init, jnp.arange(n))


def all_gathers(fn, *args):
    return jax.jit(fn).lower(*args).compile().as_text().count('all-gather')


def shard_map(body, mesh, in_specs, out_specs):
    sm = getattr(jax, 'shard_map', None)
    if sm is None:
        from jax.experimental.shard_map import shard_map as sm
    off = {k: False for k in ('check_vma', 'check_rep') if k in inspect.signature(sm).parameters}
    return sm(body, mesh=mesh, in_specs=in_specs, out_specs=out_specs, **off)


def main():
    rng = np.random.default_rng(0)
    np.random.seed(0)
    shape = (9, 8, 10, 7)
    d = len(shape)
    x = t3c.t3_corewise_randn(shape, (4, 4, 4, 4), (1, 3, 4, 3, 1))
    frame, _ = fvc.t3_orthogonal_representations(x)
    V = [rng.standard_normal((u.shape[1], dd.shape[1])) for u, dd in zip(frame[1], frame[0])]
    H = [rng.standard_normal((l.shape[0], l.shape[1], r.shape[2])) for l, r in zip(frame[2], frame[3])]
    v = (V, H)
    jframe, jv = tmap(jnp.asarray, frame), tmap(jnp.asarray, v)
    mesh = Mesh(np.array(jax.devices()), ('w',))
    n_dev = jax.device_count()
    print('jax %s, %d devices' % (jax.__version__, n_dev))

    def gn(w, sweep=None, fr=jframe, vv=jv):                        # J^T J v for probes, library kernels
        sweep = probing.tv_precompute_probe_frame_sweep(fr, w) if sweep is None else sweep
        Jp = probing.tv_probe_jacobian_from_sweep(vv, w, fr, sweep)
        return probing.tv_probe_transpose_from_sweep(Jp, w, fr, sweep, sum_over_probes=True)

    # ------------------------------------------------------------------ A. jit, one device, levels
    stack = (4, 6, 8)
    W = int(np.prod(stack))
    w_np = [rng.standard_normal(stack + (N,)) for N in shape]
    w = [jnp.asarray(a) for a in w_np]
    ref = gn([a.reshape((W, -1)) for a in w_np], fr=frame, vv=v)    # numpy, flat, unblocked

    def outer(w):                                                   # scan axis 0, nothing kept
        return scan_axis(lambda b: (gn(b), ()), w, 0)[0]

    def inner(w):                                                   # keep the sweep for all samples, scan axis 0
        sweep = probing.tv_precompute_probe_frame_sweep(jframe, w)
        return scan_axis(lambda b: (gn(b[0], b[1]), ()), (w, sweep), 0)[0]

    def nested(w):                                                  # scan 0; keep the sweep per block; scan 1
        def block(wb):
            sweep = probing.tv_precompute_probe_frame_sweep(jframe, wb)
            return scan_axis(lambda b: (gn(b[0], b[1]), ()), (wb, sweep), 0)[0], ()
        return scan_axis(block, w, 0)[0]

    def middle(w):                                                  # scan a non-leading axis of the stack
        return scan_axis(lambda b: (gn(b), ()), w, 1)[0]

    print('A. Gauss-Newton product under jit, sample stack %s, against unblocked numpy' % (stack,))
    for name, fn in (('vectorized', gn), ('scan axis 0', outer), ('keep the sweep, scan axis 0', inner),
                     ('scan 0, keep the sweep per block, scan 1', nested), ('scan axis 1 (not leading)', middle)):
        print('   %-44s %.0e' % (name, rel(jax.jit(fn)(w), ref)))

    # ------------------------------------------------------------------ B. the scan is not unrolled, and bounds memory
    def eqns(jaxpr):                                                # lines of the printed program, sub-programs included
        return len(str(jaxpr).splitlines())

    counts = []
    for nb in (4, 64):
        wb = [jnp.zeros((nb, 16, N)) for N in shape]
        counts.append(eqns(jax.make_jaxpr(outer)(wb)))
    print('B. traced program size for 4 blocks and for 64 blocks: %d and %d lines' % tuple(counts))
    big = 4096
    for label, fn, shp in (('all %d samples vectorized' % big, gn, (big,)), ('blocks of 512, scanned', outer, (8, 512)),
                           ('blocks of 64, scanned', outer, (64, 64))):
        wb = [jnp.zeros(shp + (N,)) for N in shape]
        mem = jax.jit(fn).lower(wb).compile().memory_analysis()
        print('   XLA scratch for the product, %-28s %7.2f MB' % (label + ':', mem.temp_size_in_bytes / 2**20))

    # ------------------------------------------------------------------ C. sharding
    nb, B = 8, 16
    w_np = [rng.standard_normal((nb, B, N)) for N in shape]
    w = [jnp.asarray(a) for a in w_np]
    ref = gn([a.reshape((nb * B, -1)) for a in w_np], fr=frame, vv=v)
    spec = [P('w', None, None)] * d
    placed = [jax.device_put(a, NamedSharding(mesh, s)) for a, s in zip(w, spec)]

    c1 = shard_map(lambda w: tmap(lambda a: lax.psum(a, 'w'), outer(w)), mesh, (spec,), P())
    print('C. sharding over %d devices; %d blocks of %d, the block axis sharded' % (n_dev, nb, B))
    print('   1. shard_map, each device scans its own blocks:        difference %.0e   all-gathers %d'
          % (rel(jax.jit(c1)(placed), ref), all_gathers(c1, placed)))
    print('   2. plain jit, scanning the sharded axis:               difference %.0e   all-gathers %d'
          % (rel(jax.jit(outer)(placed), ref), all_gathers(outer, placed)))

    w3_np = [rng.standard_normal((n_dev, 2, B, N)) for N in shape]             # (shard, block, B)
    ref3 = gn([a.reshape((n_dev * 2 * B, -1)) for a in w3_np], fr=frame, vv=v)
    placed3 = [jax.device_put(jnp.asarray(a), NamedSharding(mesh, P('w', None, None, None))) for a in w3_np]
    print('   3. plain jit, a separate sharded axis left vectorized, scanning the next axis:'
          '\n                                                           difference %.0e   all-gathers %d'
          % (rel(jax.jit(middle)(placed3), ref3), all_gathers(middle, placed3)))

    # per-sample output under shard_map: stays sharded
    def resid(w):
        return scan_axis(lambda b: ((), probing.tv_probe_jacobian_from_sweep(
            jv, b, jframe, probing.tv_precompute_probe_frame_sweep(jframe, b))), w, 0)[1]
    c5 = shard_map(lambda w: tuple(jnp.moveaxis(a, 0, 0) for a in resid(w)), mesh, (spec,), tuple(spec))
    out = jax.jit(c5)(placed)
    ref_J = probing.tv_probe(w_np, v, frame)
    print('   4. a per-sample result (J p) under shard_map:          difference %.0e   all-gathers %d   sharded: %s'
          % (rel(out, ref_J), all_gathers(c5, placed), not out[0].sharding.is_fully_replicated))

    # a whole CG solve compiled as one while_loop, with the scanned product inside, inside shard_map
    tdot = lambda a, b: sum(jnp.vdot(x, y) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)))
    axpy = lambda a, x, y: tmap(lambda u, w_: a * u + w_, x, y)

    def cg(apply_H, b, iters, xp=jnp, loop=None):
        def step(state):
            pk, r, dk, rs, i = state
            Hd = apply_H(dk)
            alpha = rs / tdot(dk, Hd)
            pk, r = axpy(alpha, dk, pk), axpy(-alpha, Hd, r)
            rs_new = tdot(r, r)
            return pk, r, axpy(rs_new / rs, dk, r), rs_new, i + 1
        state = (tmap(lambda z: 0.0 * z, b), b, b, tdot(b, b), 0)
        if loop is None:                                             # eager reference
            for _ in range(iters):
                state = step(state)
            return state[0]
        return loop(lambda st: st[4] < iters, step, state)[0]

    rhs_np = gn([a.reshape((nb * B, -1)) for a in w_np], fr=frame, vv=v)
    damped_np = lambda q: axpy(1.0, gn([a.reshape((nb * B, -1)) for a in w_np], fr=frame, vv=q), q)
    tdot_np = lambda a, b: sum(np.vdot(x, y) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)))
    cg_ref = cg(damped_np, rhs_np, 6)
    rhs = tmap(jnp.asarray, rhs_np)

    def solve(w):                                                    # runs per device; psum finishes each product
        product = lambda q: axpy(1.0, tmap(lambda a: lax.psum(a, 'w'), scan_axis(lambda b: (gn(b, vv=q), ()), w, 0)[0]), q)
        return cg(product, rhs, 6, loop=lax.while_loop)
    c6 = shard_map(solve, mesh, (spec,), P())
    print('   5. six CG iterations as one compiled while_loop, the scanned product inside, under shard_map:'
          '\n                                                           difference %.0e   all-gathers %d'
          % (rel(jax.jit(c6)(placed), cg_ref), all_gathers(c6, placed)))

    # ------------------------------------------------------------------ D. entries: padding with a mask, sharded
    Wreal, Bp = 1000, 32
    pad = (-Wreal) % (Bp * n_dev)
    index = np.stack([rng.integers(0, N, size=Wreal) for N in shape])
    ref_e = entries.tv_entries_transpose(entries.tv_entries(index, v, frame), index, frame, sum_over_probes=True)
    ip = np.concatenate([index, np.zeros((d, pad), dtype=index.dtype)], axis=1).reshape((d, -1, Bp))
    mask = np.concatenate([np.ones(Wreal), np.zeros(pad)]).reshape((-1, Bp))

    def gn_entries(args, use_mask=True):
        def kernel(b):
            ib, mb = b
            c = entries.tv_entries(ib, jv, jframe)
            c = c * mb if use_mask else c
            return entries.tv_entries_transpose(c, ib, jframe, sum_over_probes=True), ()
        return scan_axis(kernel, args, (1, 0))[0]                    # the index stacks W after the mode axis

    e_specs = (P(None, 'w', None), P('w', None))
    e_placed = (jax.device_put(jnp.asarray(ip), NamedSharding(mesh, e_specs[0])),
                jax.device_put(jnp.asarray(mask), NamedSharding(mesh, e_specs[1])))
    d1 = shard_map(lambda a: tmap(lambda z: lax.psum(z, 'w'), gn_entries(a)), mesh, (e_specs,), P())
    print('D. entries, %d samples padded to %d (blocks of %d on %d devices), mask on the padded samples' % (Wreal, Wreal + pad, Bp, n_dev))
    print('   one device, jit:  difference %.0e;   without the mask %.0e'
          % (rel(jax.jit(gn_entries)((jnp.asarray(ip), jnp.asarray(mask))), ref_e),
             rel(jax.jit(lambda a: gn_entries(a, False))((jnp.asarray(ip), jnp.asarray(mask))), ref_e)))
    print('   shard_map:        difference %.0e   all-gathers %d' % (rel(jax.jit(d1)(e_placed), ref_e), all_gathers(d1, e_placed)))

    # ------------------------------------------------------------------ E. jets: the sample axis is not leading
    order = 2
    kind = bfit.ProbeDerivativesKind(order=order, chunk_size=None)
    ww = [rng.standard_normal((nb, B, N)) for N in shape]
    pp = [rng.standard_normal((nb, B, N)) for N in shape]
    fl = lambda t: [a.reshape((nb * B, -1)) for a in t]
    s_ref = (fl(ww), fl(pp))
    sw_ref = kind.precompute(frame, s_ref)
    z_ref = kind.forward(v, s_ref, frame, sw_ref)                    # (order+1, W, N) per mode
    out_ref = kind.transpose(z_ref, s_ref, frame, sw_ref)
    zb = [jnp.asarray(z.reshape((order + 1, nb, B, -1))) for z in z_ref]
    js = (tmap(jnp.asarray, ww), tmap(jnp.asarray, pp))

    def jets(args):
        def kernel(b):
            s, z = b
            return kind.transpose(z, s, jframe, kind.precompute(jframe, s)), ()
        s, z = args
        return scan_axis(kernel, (s, z), (tmap(lambda a: 0, s), tmap(lambda a: 1, z)))[0]

    j_specs = ((spec, spec), [P(None, 'w', None, None)] * d)
    j_placed = tmap(lambda a, s: jax.device_put(a, NamedSharding(mesh, s)), (js, zb), j_specs)
    e2 = shard_map(lambda a: tmap(lambda z: lax.psum(z, 'w'), jets(a)), mesh, (j_specs,), P())
    print('E. jets (order %d): the seed has the order axis before the sample axis' % order)
    print('   one device, jit:  difference %.0e' % rel(jax.jit(jets)((js, zb)), out_ref))
    print('   shard_map:        difference %.0e   all-gathers %d' % (rel(jax.jit(e2)(j_placed), out_ref), all_gathers(e2, j_placed)))

    # ------------------------------------------------------------------ F. the uniform layout: a scan inside the scan
    import t3toolbox.tucker_tensor_train as t3
    import t3toolbox.uniform_tucker_tensor_train as ut3
    import t3toolbox.uniform_manifold as um
    import t3toolbox.fitting as fitting

    ushape, Wu = (9, 9, 9, 9), nb * B
    xu = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn(ushape, (4, 4, 4, 4), (1, 3, 3, 3, 1)))
    wu = [jnp.asarray(rng.standard_normal((Wu, N))) for N in ushape]
    model = fitting.probe_model(um.UNIFORM_MANIFOLD, xu, wu, [jnp.zeros((Wu, N)) for N in ushape])
    ukind, uframe, usample = model.kind, model.frame.data, model.sample           # packed sample (d, W, N)
    uv = fitting._coordinates_of(model._project(um.UNIFORM_MANIFOLD.randn(model.frame)))
    u_ref = ukind.transpose(ukind.forward(uv, usample, uframe, model.sweep), usample, uframe, model.sweep)

    def gn_uniform(sample):
        def kernel(sb):
            sweep = ukind.precompute(uframe, sb)
            return ukind.transpose(ukind.forward(uv, sb, uframe, sweep), sb, uframe, sweep), ()
        return scan_axis(kernel, sample, 1)[0]                        # the packed sample stacks W after the site axis

    ub = usample.reshape((len(ushape), nb, B, -1))
    u_spec = P(None, 'w', None, None)
    f1 = shard_map(lambda s: tmap(lambda z: lax.psum(z, 'w'), gn_uniform(s)), mesh, (u_spec,), P())
    u_placed = jax.device_put(ub, NamedSharding(mesh, u_spec))
    print('F. uniform layout (the kernels scan over sites inside the scan over blocks), against the unblocked model')
    print('   one device, jit:  difference %.0e' % rel(jax.jit(gn_uniform)(ub), u_ref))
    print('   shard_map:        difference %.0e   all-gathers %d' % (rel(jax.jit(f1)(u_placed), u_ref), all_gathers(f1, u_placed)))


if __name__ == '__main__':
    main()
