"""Prototype of the required example (target decision 6.7): badly scaled data, a bad fit as given, a good fit
after the data are processed with the standalone preparation functions.

The problem.  A Tucker tensor train with 4 modes is fitted from jets of its probes (kind 'probe_derivatives',
order 3) with the library's Newton solver, from the zero tensor, with the solver's default settings.  The data
are exact (no noise).  They are badly scaled in two ways that a user's raw data often are:

  coordinates   the coordinates of each mode's sample vectors have scales spread over four decades
                (as if some were in metres and others in millimetres);
  direction     the directions are 100 times too short, so each order of the jet is about 100 times
                smaller than the one before.

The preparation functions, as decided in 6.7 and 2.5.  All are pure and standalone.

  block_norms(data)                         the norm of each (mode, order) block
  filtered_inverse(norms, floor=1e-6)       1 / max(norms, floor * max(norms)): the equalizing weight
  rescale_direction(pp, data, length)       loss-only: the direction times `length`, slot t of the data times length^t
  rescale_coordinates(ww, pp, data, scale)  changes the unknowns AND, for probe, the loss: sample vectors times
                                            `scale`, that mode's data divided by it
  restore_coordinates(x, scale)             the inverse of the previous one, applied to the fitted tensor

Two cases are run.  With few samples the higher orders of the jets are needed to determine the tensor, and both
kinds of preparation matter.  With many samples the order-0 data determine it alone, and only the coordinates
matter.

Today's library holds derivatives, not Taylor coefficients.  The functions above do not care; slot t scales the
same way in both.  Nothing in T3Toolbox is changed.

Usage:  python3 -I bad_scaling_example.py /path/to/T3Toolbox [--numbers]
"""
import sys
import time
import warnings

import numpy as np

sys.path.append(sys.argv[1])
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.manifold as t3m
import t3toolbox.optimizers as optimizers


# ---- the standalone preparation functions ---------------------------------------------------------------
def block_norms(
        data,           # len=d, elm_shape=(order+1,)+W+(Ni,)
):                      # shape=(d, order+1)
    return np.array([[np.linalg.norm(z[t]) for t in range(z.shape[0])] for z in data])


def filtered_inverse(
        norms,          # shape=(d, order+1)
        floor=1e-6,     # relative to the largest block
):                      # shape=(d, order+1)
    return 1.0 / np.maximum(norms, floor * np.max(norms))


def rescale_direction(
        pp,             # len=d, elm_shape=W+(Ni,)
        data,           # len=d, elm_shape=(order+1,)+W+(Ni,)
        length,         # float > 0
):
    powers = lambda z: (length ** np.arange(z.shape[0])).reshape((-1,) + (1,) * (z.ndim - 1))
    return [p * length for p in pp], [powers(z) * z for z in data]


def rescale_coordinates(
        ww,             # len=d, elm_shape=W+(Ni,)
        pp,             # len=d, elm_shape=W+(Ni,)
        data,           # len=d, elm_shape=(order+1,)+W+(Ni,)
        scale,          # len=d, elm_shape=(Ni,): the new coordinate j of mode i is scale[i][j] times the old one
):
    return ([w * s for w, s in zip(ww, scale)], [p * s for p, s in zip(pp, scale)],
            [z / s for z, s in zip(data, scale)])


def restore_coordinates(
        x,              # the tensor fitted in the new coordinates
        scale,          # len=d, elm_shape=(Ni,)
):                      # the same tensor, as a function of the old coordinates
    return t3.TuckerTensorTrain(tuple(B * s[None, :] for B, s in zip(x.tucker_cores, scale)), tuple(x.tt_cores))


# ---- two estimates a user can read off the data ---------------------------------------------------------------
def direction_length_from_norms(norms):     # the factor on the directions that makes the first and last order equally large
    per_order = np.sqrt(np.sum(norms ** 2, axis=0))
    J = per_order.shape[0] - 1
    return float((per_order[0] / per_order[J]) ** (1.0 / J))


def coordinate_scale_from_samples(ww):      # one over the root mean square of each coordinate
    return [1.0 / np.sqrt(np.mean(w ** 2, axis=0)) for w in ww]


# ---- the problem ------------------------------------------------------------------------------------------------------
SHAPE, TUCKER_RANKS, TT_RANKS, ORDER = (8, 8, 8, 8), (3, 3, 3, 3), (1, 3, 3, 3, 1), 3
SHORT_DIRECTION, W_TEST = 1e-2, 400
VERBOSE = len(sys.argv) > 2 and sys.argv[2] == '--numbers'     # also print each error and the seconds (not reproducible to the digit)


def case(W_train, seed, which):
    rng = np.random.default_rng(seed)
    rn = rng.standard_normal
    d = len(SHAPE)
    sigma = [10.0 ** rng.permutation(np.linspace(-2.0, 2.0, N)) for N in SHAPE]     # the scale of each coordinate

    # the truth is an ordinary tensor in well-scaled coordinates, seen through the badly scaled ones
    nice = [np.linalg.qr(rn((N, n)))[0].T for N, n in zip(SHAPE, TUCKER_RANKS)]
    x_true = t3.TuckerTensorTrain(tuple(B / s[None, :] for B, s in zip(nice, sigma)),
                                  tuple(rn((TT_RANKS[i], TUCKER_RANKS[i], TT_RANKS[i + 1])) for i in range(d)))
    draw = lambda W: [rn((W, N)) * s for N, s in zip(SHAPE, sigma)]
    ww, pp = draw(W_train), [SHORT_DIRECTION * p for p in draw(W_train)]
    data = [np.asarray(z) for z in x_true.probe_derivatives(ww, pp, ORDER)]
    ww_test = draw(W_TEST)
    truth = np.asarray(x_true.apply(ww_test))

    def fit_and_score(ww_, pp_, data_, weight, scale=None):
        x0 = t3.TuckerTensorTrain.zeros(SHAPE, TUCKER_RANKS, TT_RANKS)
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            x, stats = optimizers.newton_cg(t3m.MANIFOLD, 'probe_derivatives', (ww_, pp_), data_, x0,
                                            order=ORDER, weight=weight, chunk_size=None)
        seconds = time.perf_counter() - t0
        if scale is not None:
            x = restore_coordinates(x, scale)
        pred = np.asarray(x.apply(ww_test))
        return float(np.linalg.norm(pred - truth) / np.linalg.norm(truth)), seconds

    show = lambda a: '      ' + np.array2string(a, formatter={'float_kind': lambda v: '%.1e' % v}).replace('\n', '\n      ')
    norms = block_norms(data)
    print('   The data as given.  Norm of each block (rows: modes, columns: orders 0..%d):' % ORDER)
    print(show(norms))
    print('   largest / smallest block: %.0e.   The coordinate scales of the samples span %.0e.'
          % (norms.max() / norms.min(), max(s.max() / s.min() for s in sigma)))

    length = direction_length_from_norms(norms)
    scale = coordinate_scale_from_samples(ww)
    print('   Read off the data: a factor of %.0f on the directions (they were made %.0f times too short), and'
          % (length, 1 / SHORT_DIRECTION))
    print('   coordinate scales within a factor %.1f of the true ones.'
          % max(np.max(np.maximum(s * t, 1 / (s * t))) for s, t in zip(scale, sigma)))

    pp_s, data_s = rescale_direction(pp, data, length)
    ww_c, pp_c, data_c = rescale_coordinates(ww, pp, data, scale)
    ww_cs, pp_cs, data_cs = rescale_coordinates(ww, pp_s, data_s, scale)
    eq = lambda z: filtered_inverse(block_norms(z))
    arms = {
        'as given':                                     ((ww, pp, data, None), None),
        'equalizing weight only':                       ((ww, pp, data, eq(data)), None),
        'coordinates only':                             ((ww_c, pp_c, data_c, None), scale),
        'coordinates, then the direction':              ((ww_cs, pp_cs, data_cs, None), scale),
        'coordinates, then equalizing weight':          ((ww_c, pp_c, data_c, eq(data_c)), scale),
        'coordinates and direction, then equalizing weight': ((ww_cs, pp_cs, data_cs, eq(data_cs)), scale),
    }
    print('   %-50s %s' % ('preparation', 'fit'))
    for label in which:
        args, sc = arms[label]
        error, seconds = fit_and_score(*args, scale=sc)
        verdict = ('good  (test error below 1e-6)' if error < 1e-6 else
                   'bad   (test error above 0.1)' if error > 0.1 else 'test error %.0e' % error)
        print('   %-50s %s' % (label, verdict) + ('   [%.1e, %.1f s]' % (error, seconds) if VERBOSE else ''))
    after = block_norms(data_cs)
    print('   Blocks after the full preparation: largest / smallest %.1f.' % (after.max() / after.min()))


def main():
    print('The fit: Newton from the zero tensor, default settings.  Test error = relative error of the tensor applied')
    print('to %d fresh samples, in the ORIGINAL coordinates (a scalar, so it does not depend on any scaling).' % W_TEST)
    print('Case 1: 8 samples.  The higher orders of the jets are needed.')
    case(8, 0, ('as given', 'equalizing weight only', 'coordinates only', 'coordinates, then the direction',
                'coordinates, then equalizing weight', 'coordinates and direction, then equalizing weight'))
    print('Case 2: 200 samples.  The order-0 data determine the tensor alone.')
    case(200, 0, ('as given', 'equalizing weight only', 'coordinates only', 'coordinates, then equalizing weight'))


if __name__ == '__main__':
    main()
