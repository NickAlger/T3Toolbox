"""Prototype for target decision 6.6: ONE frontend `probe` for the plain and the jetted computation.

Signatures only, in the house style (shape comments as the type).  The bodies call the library's existing
methods and convert derivatives to Taylor coefficients, so that the shapes and the checks below are real.
Nothing in T3Toolbox is changed; these are free functions that take the tensor as their first argument.

`T` is the optional order axis: T = () at order 0, T = (order+1,) above.  It is outermost, where the order
axis is today.  It is never read off an array alone:
  - where it is created, `order` declares it;
  - where it is consumed (the transposes), it is recovered by pairing the residual with the sample, and
    checked against `order`.
The inputs never carry it: a jetted input is a point `ww` and one direction `pp`.

Usage:  python3 -I merged_frontend_prototype.py /path/to/T3Toolbox
"""
import math
import sys
import typing as typ

import numpy as np

sys.path.append(sys.argv[1])
import t3toolbox.tucker_tensor_train as t3

NDArray = np.ndarray
Sequence = typ.Sequence


def _check_jet_arguments(pp, order):
    if order < 0:
        raise ValueError('order must be >= 0, got %d' % order)
    if order == 0 and pp is not None:
        raise ValueError('pp was given but order is 0. A direction is read only when order >= 1; '
                         'pass order, or drop pp.')
    if order >= 1 and pp is None:
        raise ValueError('order is %d but no direction pp was given. Pass pp, with the same shapes as ww.' % order)


def _taylor(order: int, ndim: int) -> NDArray:           # 1 / t!, shaped to broadcast against T + ...
    return np.array([1.0 / math.factorial(t) for t in range(order + 1)]).reshape((order + 1,) + (1,) * (ndim - 1))


# ---------------------------------------------------------------------------------------------------------
def probe(
        x:      t3.TuckerTensorTrain,
        ww:     Sequence[NDArray],                  # probe vectors X,     len=d, elm_shape=W+(Ni,)
        pp:     typ.Optional[Sequence[NDArray]] = None,  # direction P,    len=d, elm_shape=W+(Ni,). Given iff order >= 1
        order:  int = 0,                            # highest Taylor order.  T=(order+1,) if order else ()
) -> Sequence[NDArray]:                             # zz, len=d, elm_shape=T+W+C+(Ni,)
    """Probe a Tucker tensor train, or the Taylor coefficients of probing it along a line.

    ``order == 0``:  ``zz[i] = probe(X)_i``, with no order axis.
    ``order >= 1``:  ``zz[i][t]`` is the coefficient of ``s^t`` in ``probe(X + s P)_i``, for ``t = 0..order``.
    """
    _check_jet_arguments(pp, order)
    if order == 0:
        return x.probe(ww)
    return tuple(_taylor(order, z.ndim) * z for z in x.probe_derivatives(ww, pp, order))


def probe_corewise_transpose(
        x:       t3.TuckerTensorTrain,
        ztildes: Sequence[NDArray],                 # residual,            len=d, elm_shape=T+W+C+(Ni,)
        ww:      Sequence[NDArray],                 # probe vectors X,     len=d, elm_shape=W+(Ni,)
        pp:      typ.Optional[Sequence[NDArray]] = None,  # direction P,   len=d, elm_shape=W+(Ni,). Given iff order >= 1
        order:   int = 0,                           # highest Taylor order.  T=(order+1,) if order else ()
        sum_over_probes: bool = False,
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],                        # tucker-core gradients, same shapes as x.tucker_cores
    typ.Tuple[NDArray, ...],                        # tt-core gradients,     same shapes as x.tt_cores
]:
    """Corewise transpose of :py:func:`probe`.  ``ztildes[i][t]`` pairs with ``zz[i][t]``."""
    _check_jet_arguments(pp, order)
    len_T = ztildes[0].ndim - ww[0].ndim - len(x.stack_shape)       # recovered by pairing, not read off the array
    if len_T != (0 if order == 0 else 1):
        raise ValueError('ztildes has %d axis(es) beyond W+C+(Ni,), but order=%d calls for %d.'
                         % (len_T, order, 0 if order == 0 else 1))
    if order == 0:
        return x.probe_corewise_transpose(ztildes, ww, sum_over_probes=sum_over_probes)
    return x.probe_corewise_derivatives_transpose([_taylor(order, c.ndim) * c for c in ztildes], ww, pp, order,
                                                  sum_over_probes=sum_over_probes, chunk_size=None)


# ---------------------------------------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(0)
    rn = rng.standard_normal
    x = t3.TuckerTensorTrain(tuple(rn((n, N)) for n, N in zip((4, 5, 6), (14, 15, 16))),      # seeded, so the run repeats
                             tuple(rn((a, n, b)) for a, n, b in zip((1, 3, 2), (4, 5, 6), (3, 2, 1))))
    flat = lambda parts: np.concatenate([np.ravel(a) for a in parts])
    err = lambda a, b: np.linalg.norm(flat(a) - flat(b)) / np.linalg.norm(flat(b))

    print('probe: shapes of zz[0] (N0 = 14)')
    for W in ((), (5,), (2, 5)):
        ww = [rn(W + (N,)) for N in (14, 15, 16)]
        pp = [rn(W + (N,)) for N in (14, 15, 16)]
        z0 = probe(x, ww)
        z3 = probe(x, ww, pp, order=3)
        print('   W = %-7s  order 0: %-14s  order 3: %-16s  order-0 result is x.probe: %s;  z3[t] * t! is '
              'x.probe_derivatives: %.0e;  z3[0] is the order-0 result: %.0e'
              % (W, z0[0].shape, z3[0].shape, all(np.array_equal(a, b) for a, b in zip(z0, x.probe(ww))),
                 err([z / _taylor(3, z.ndim) for z in z3], x.probe_derivatives(ww, pp, 3)),
                 err([z[0] for z in z3], z0)))

    print('probe_corewise_transpose: the transpose is the adjoint of probe in the Taylor pairing')
    W = (5,)
    ww = [rn(W + (N,)) for N in (14, 15, 16)]
    pp = [rn(W + (N,)) for N in (14, 15, 16)]
    for order in (0, 3):
        T = () if order == 0 else (order + 1,)
        c = [rn(T + W + (N,)) for N in (14, 15, 16)]
        args = (ww,) if order == 0 else (ww, pp)
        g = probe_corewise_transpose(x, c, *args, order=order, sum_over_probes=True)
        # <c, d probe(x)[dx]> against <g, dx> for a small change dx of the cores, by central differences
        dB = [rn(b.shape) for b in x.tucker_cores]
        dG = [rn(G.shape) for G in x.tt_cores]
        eps = 1e-6
        shifted = lambda s: t3.TuckerTensorTrain(tuple(b + s * e for b, e in zip(x.tucker_cores, dB)),
                                                 tuple(G + s * e for G, e in zip(x.tt_cores, dG)))
        dz = [(a - b) / (2 * eps) for a, b in zip(probe(shifted(eps), *args, order=order),
                                                  probe(shifted(-eps), *args, order=order))]
        lhs = sum(float(np.sum(ci * zi)) for ci, zi in zip(c, dz))
        rhs = sum(float(np.sum(a * b)) for a, b in zip(g[0], dB)) + sum(float(np.sum(a * b)) for a, b in zip(g[1], dG))
        print('   order %d: residual elm_shape %-12s  <c, d probe> = %.10f   <transpose(c), dx> = %.10f   rel diff %.0e'
              % (order, c[0].shape, lhs, rhs, abs(lhs - rhs) / abs(rhs)))

    print('what a wrong call says')
    ww = [rn((5, N)) for N in (14, 15, 16)]
    pp = [rn((5, N)) for N in (14, 15, 16)]
    for label, call in (('probe(x, ww, pp)', lambda: probe(x, ww, pp)),
                        ('probe(x, ww, order=2)', lambda: probe(x, ww, order=2)),
                        ('probe_corewise_transpose(x, <jet residual>, ww)',
                         lambda: probe_corewise_transpose(x, [rn((3, 5, N)) for N in (14, 15, 16)], ww))):
        try:
            call()
            print('   %-48s no error' % label)
        except ValueError as e:
            print('   %-48s ValueError: %s' % (label, e))


if __name__ == '__main__':
    main()
