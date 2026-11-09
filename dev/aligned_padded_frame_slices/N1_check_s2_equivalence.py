"""One-time migration check for slice N1: the new ``padded_frames.fv_padded_frame`` against the construction it
replaced, bitwise, on the replaced construction's test cases.

The replaced construction is the template-signature ``fv_operations.fv_padded_frame`` of commit 92640502 (S3), whose
hint-free path is 2562264a's (S2) unchanged; N1 deleted it. This script loads that file from git (``git show
92640502:t3toolbox/backend/fv_operations.py``) into a temporary module and runs its hint-free path live on 30 cases:
every case of the S2 tests, the deleted dispatch row's two cases, and one extra seed. It runs the new function on the
same old frames and target ranks (``fv_empty_frame`` for the zero tensor) and prints the bitwise count. Nothing is
written. Run from the repository root:

    PYTHONPATH=$PWD python dev/aligned_padded_frame_slices/N1_check_s2_equivalence.py

Result on the implementer's machine (2026-11-09; py3.11, numpy 2.x): 30 cases, 380 of 380 arrays bitwise equal. The
cases' S2 templates all have exactly the stores ``compute_orthogonal_representation_ranks`` gives at the target
ranks (asserted), so each case is expressible in the new signature. The committed test (T5) pins the
platform-independent facts instead (the zero tensor bitwise; the Tucker stream against an independent QR; the
convention's properties).
"""
import importlib.util
import os
import subprocess
import sys
import tempfile

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
SOURCE_COMMIT = '92640502'
sys.path.insert(0, REPO)

import t3toolbox.tucker_tensor_train as t3            # noqa: E402
import t3toolbox.backend.fv_conversions as fvc         # noqa: E402
import t3toolbox.backend.ranks as ranks                # noqa: E402
import t3toolbox.backend.sharing as sharing_module     # noqa: E402

_source = subprocess.run(['git', 'show', SOURCE_COMMIT + ':t3toolbox/backend/fv_operations.py'], cwd=REPO,
                         check=True, capture_output=True, text=True).stdout
_tmp = tempfile.mkdtemp()
with open(os.path.join(_tmp, 'fv_operations_s3.py'), 'w') as _f:
    _f.write(_source)
spec = importlib.util.spec_from_file_location('fv_operations_s3', os.path.join(_tmp, 'fv_operations_s3.py'))
fvo_head = importlib.util.module_from_spec(spec)
sys.modules['fv_operations_s3'] = fvo_head           # dataclasses resolve annotations through sys.modules
spec.loader.exec_module(fvo_head)

STORES = ('up', 'down', 'left', 'right')


def frame_of(x):
    return fvc.t3_orthogonal_representations(x.data)[0]


def store_widths(frame):
    up, down, left, right = frame
    return (tuple(int(u.shape[-2]) for u in up), tuple(int(o.shape[-2]) for o in down),
            tuple(int(g.shape[-3]) for g in left) + (int(left[-1].shape[-1]),),
            tuple(int(g.shape[-3]) for g in right) + (int(right[-1].shape[-1]),))


def padded_case(shape, old_ranks, new_ranks, stack_shape=(), sharing=None, seed=0):
    """The S2 tests' construction: x_old random at old_ranks (shared first under sharing), the template the frame
    of its zero-padded resize."""
    np.random.seed(seed)
    x_old = t3.TuckerTensorTrain.randn(shape, *old_ranks, stack_shape=stack_shape)
    if sharing is not None:
        x_old = x_old.share(sharing)
    x_new = x_old.resize(shape, *new_ranks, sharing=sharing)
    return x_old, frame_of(x_old), frame_of(x_new)


CASES = []   # (name, shape, target tucker, target tt, stack, sharing, seed, old_frame or None, template)

# T2(a) of S2: the zero tensor (old_frame=None) reproduces fv_zero_point_frame; seeds 0 and 5
ZERO = [
    ((7,), (3,), (1, 1), None),
    ((5, 6, 7), (2, 3, 2), (1, 2, 2, 1), None),
    ((6, 6, 7), (3, 3, 2), (1, 3, 2, 1), (0, 0, 1)),
    ((4, 5, 6, 3), (2, 3, 3, 2), (1, 2, 3, 2, 1), None),
    ((5, 5, 5, 4), (3, 3, 3, 2), (1, 3, 3, 2, 1), (0, 1, 0, 2)),
]
for shape, tucker, tt, sharing in ZERO:
    template = frame_of(t3.TuckerTensorTrain.zeros(shape, tucker, tt))
    for seed in (0, 5):
        CASES.append(('zero %r %r %r sharing=%r seed=%d' % (shape, tucker, tt, sharing, seed),
                      shape, tucker, tt, (), sharing, seed, None, template))
# the stacked zero template (broadcast over C)
template = frame_of(t3.TuckerTensorTrain.zeros((5, 6, 7), (2, 3, 2), (1, 2, 2, 1), stack_shape=(2, 3)))
CASES.append(('zero stacked C=(2,3)', (5, 6, 7), (2, 3, 2), (1, 2, 2, 1), (2, 3), None, 0, None, template))
# the dispatch row's zero case: shared, seed 3, d = 4
template = frame_of(t3.TuckerTensorTrain.zeros((4, 4, 4, 4), (2, 2, 2, 3), (1, 2, 3, 2, 1)))
CASES.append(('zero d=4 shared seed=3', (4, 4, 4, 4), (2, 2, 2, 3), (1, 2, 3, 2, 1), (), (0, 1, 0, 2), 3, None,
              template))

# T2(b): valid at resized points (Tucker-only, TT-only, joint, d = 4, d = 1, no increase)
RESIZED = {
    'tucker only': ((5, 6, 7), ((2, 2, 2), (1, 2, 2, 1)), ((3, 4, 2), (1, 2, 2, 1))),
    'tt only':     ((5, 6, 7), ((3, 3, 3), (1, 2, 2, 1)), ((3, 3, 3), (1, 3, 3, 1))),
    'joint':       ((5, 6, 7), ((2, 2, 2), (1, 2, 2, 1)), ((3, 4, 3), (1, 3, 3, 1))),
    'joint d=4':   ((4, 5, 6, 3), ((2, 2, 2, 2), (1, 2, 2, 2, 1)), ((3, 3, 4, 3), (1, 3, 4, 3, 1))),
    'd=1':         ((7,), ((2,), (1, 1)), ((5,), (1, 1))),
    'no increase': ((5, 6, 7), ((2, 3, 2), (1, 2, 2, 1)), ((2, 3, 2), (1, 2, 2, 1))),
}
for name, (shape, old_ranks, new_ranks) in RESIZED.items():
    _, old, template = padded_case(shape, old_ranks, new_ranks)
    CASES.append((name, shape, new_ranks[0], new_ranks[1], (), None, 0, old, template))
# ... and another seed on the joint case
_, old, template = padded_case((5, 6, 7), ((2, 2, 2), (1, 2, 2, 1)), ((3, 4, 3), (1, 3, 3, 1)))
CASES.append(('joint seed=3', (5, 6, 7), (3, 4, 3), (1, 3, 3, 1), (), None, 3, old, template))
# T2(b) sharing: a tied old frame, one factor per group
_, old, template = padded_case((6, 6, 7), ((2, 2, 2), (1, 2, 2, 1)), ((5, 5, 3), (1, 4, 3, 1)), sharing=(0, 0, 1))
CASES.append(('sharing', (6, 6, 7), (5, 5, 3), (1, 4, 3, 1), (), (0, 0, 1), 0, old, template))

# T2(c): in-block completions (a bond growing while its neighbours do not; a down core at saturated bonds)
INBLOCK = {
    'in-block TT d=4':       ((4, 4, 4, 4), ((2, 2, 2, 2), (1, 2, 2, 2, 1)), ((2, 2, 2, 2), (1, 2, 3, 2, 1))),
    'in-block TT bond 1':    ((5, 6, 7), ((3, 3, 3), (1, 2, 2, 1)), ((3, 3, 3), (1, 3, 2, 1))),
    'in-block down':         ((5, 6, 7), ((2, 3, 2), (1, 2, 2, 1)), ((2, 4, 2), (1, 2, 2, 1))),
    'outside then in-block': ((4, 5, 6, 3), ((2, 2, 2, 2), (1, 2, 2, 2, 1)), ((3, 3, 4, 3), (1, 3, 4, 3, 1))),
}
for name, (shape, old_ranks, new_ranks) in INBLOCK.items():
    _, old, template = padded_case(shape, old_ranks, new_ranks)
    CASES.append((name, shape, new_ranks[0], new_ranks[1], (), None, 0, old, template))

# T2(d): nesting -- the second Tucker count at mode 1
_, old, template = padded_case((5, 6, 7), ((2, 2, 2), (1, 2, 2, 1)), ((3, 3, 3), (1, 3, 3, 1)))
CASES.append(('nested 3', (5, 6, 7), (3, 3, 3), (1, 3, 3, 1), (), None, 0, old, template))

# T2(e): stacks C = (2,) and (2, 3)
for C in [(2,), (2, 3)]:
    for shape, old_ranks, new_ranks in [((5, 6, 7), ((2, 2, 2), (1, 2, 2, 1)), ((3, 4, 3), (1, 3, 3, 1))),
                                        ((4, 4, 4, 4), ((2, 2, 2, 2), (1, 2, 2, 2, 1)),
                                         ((2, 2, 2, 2), (1, 2, 3, 2, 1)))]:
        _, old, template = padded_case(shape, old_ranks, new_ranks, stack_shape=C)
        CASES.append(('stacked C=%r %r' % (C, shape), shape, new_ranks[0], new_ranks[1], C, None, 0, old, template))

# the dispatch row's nonzero case: (4,4,4,4) -> L_1 in-block, U_3 Haar-completed, seed 3
np.random.seed(1)
x_old = t3.TuckerTensorTrain.randn((4, 4, 4, 4), (2, 2, 2, 2), (1, 2, 2, 2, 1))
x_new = x_old.resize((4, 4, 4, 4), (2, 2, 2, 3), (1, 2, 3, 2, 1))
CASES.append(('dispatch d=4 seed=3', (4, 4, 4, 4), (2, 2, 2, 3), (1, 2, 3, 2, 1), (), None, 3, frame_of(x_old),
              frame_of(x_new)))


def main():
    import t3toolbox.backend.padded_frames as padded_frames
    n_bitwise, n_arrays = 0, 0
    for name, shape, tucker, tt, C, sharing, seed, old, template in CASES:
        # the S2 template is exactly the frame layout of the target ranks (what N1's signature computes)
        assert store_widths(template) == ranks.compute_orthogonal_representation_ranks(shape, tucker, tt), name
        groups = () if sharing is None else sharing_module.canonical_groups(sharing, shape)
        s2 = fvo_head.fv_padded_frame(template, old, None, groups=groups, seed=seed)
        new_old = padded_frames.fv_empty_frame(shape, C) if old is None else old
        new = padded_frames.fv_padded_frame(new_old, tucker, tt, sharing=sharing, seed=seed)
        equal = [np.array_equal(a, b) and a.shape == b.shape and a.dtype == b.dtype
                 for f, g in zip(s2, new) for a, b in zip(f, g)]
        tied = all(new[0][i] is new[0][g[0]] for g in groups for i in g)
        n_bitwise += sum(equal)
        n_arrays += len(equal)
        print('  %-48s %d/%d bitwise%s' % (name, sum(equal), len(equal), ', groups tied' if groups else '')
              + ('' if tied else '  GROUPS NOT TIED'))
    print('%d cases, %d of %d arrays bitwise equal' % (len(CASES), n_bitwise, n_arrays))


if __name__ == '__main__':
    main()
