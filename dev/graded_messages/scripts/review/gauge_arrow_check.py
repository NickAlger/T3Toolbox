"""Are the arrows of the frame cores the library's orthogonality conditions?

Arrow rule (the maintainer's, 2026-11-15): every arrow points away from its store, towards the hole.  Claim: in
each family, the legs where the arrow ENTERS the core are the ones summed in the orthogonality condition, and the
one leg where it LEAVES is free.  Equivalently: as a map along its arrows, each core M satisfies M M^T = I.

    D (Tucker, B_io)   enters o,     leaves i     sum_o     B_io  B_i'o   = delta_ii'
    L (TT, G_aib)      enters a, i,  leaves b     sum_{a,i} G_aib G_aib'  = delta_bb'
    R (TT, G_aib)      enters i, b,  leaves a     sum_{i,b} G_aib G_a'ib  = delta_aa'
    U (TT, G_aib)      enters a, b,  leaves i     sum_{a,b} G_aib G_ai'b  = delta_ii'

Reads T3Toolbox, changes nothing.   Usage:  python3 -I gauge_arrow_check.py /path/to/T3Toolbox
"""
import sys

import numpy as np

sys.path.append(sys.argv[1])
import t3toolbox.tucker_tensor_train as t3


def worst(cores, spec):
    out = 0.0
    for G in cores:
        M = np.einsum(spec, G, G)
        out = max(out, float(np.max(np.abs(M - np.eye(M.shape[0])))))
    return out


def main():
    rng = np.random.default_rng(0)
    rn = rng.standard_normal
    def make(Ns, ns, rs):
        return t3.TuckerTensorTrain(tuple(rn((n, N)) for n, N in zip(ns, Ns)),
                                    tuple(rn((rs[i], ns[i], rs[i + 1])) for i in range(len(ns))))
    x = make((14, 15, 16), (4, 5, 6), (1, 3, 2, 1))
    xu = make((14, 15, 16), (3, 5, 2), (1, 4, 3, 1))           # ranks for which the up condition is possible
    print('D  down_orthogonalize_tucker_cores  sum over o       : %.0e' % worst(x.down_orthogonalize_tucker_cores().tucker_cores, 'io,jo->ij'))
    print('L  left_orthogonalize_tt_cores      sum over a, i    : %.0e   (all cores but the last)'
          % worst(x.left_orthogonalize_tt_cores().tt_cores[:-1], 'aib,aic->bc'))
    print('R  right_orthogonalize_tt_cores     sum over i, b    : %.0e   (all cores but the first)'
          % worst(x.right_orthogonalize_tt_cores().tt_cores[1:], 'aib,cib->ac'))
    last_L = x.left_orthogonalize_tt_cores().tt_cores[-1:]
    first_R = x.right_orthogonalize_tt_cores().tt_cores[:1]
    print('   the known exceptions: the last L core is off by %.0e, the first R core by %.0e (they carry the norm)'
          % (worst(last_L, 'aib,aic->bc'), worst(first_R, 'aib,cib->ac')))
    print('U  up_orthogonalize_tt_cores        sum over a, b    : %.0e' % worst(xu.up_orthogonalize_tt_cores().tt_cores, 'aib,ajb->ij'))


if __name__ == '__main__':
    main()
