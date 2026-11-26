# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
"""The guard of ``tests/conftest.py``: a ``subTest`` label that is not a builtin raises, wherever the tests run."""

import unittest

import numpy as np


class TestSubTestLabelsAreBuiltins(unittest.TestCase):

    def test_builtin_labels_pass(self):
        for label in (None, True, 3, 2.5, 1j, 'a', b'a', (1, 'a'), [1, (2, 3)], {1, 2}, frozenset({'a'}),
                      {'k': (1, 2.0)}):
            with self.subTest(label=label):
                pass
        with self.subTest('a message', n=1):
            pass

    def test_numpy_labels_raise(self):
        for bad in (np.int64(3), np.bool_(True), np.float64(1.0), np.array([1]), (1, np.int32(2)),
                    {'k': np.uint8(1)}, np.float32):
            with self.assertRaisesRegex(TypeError, 'subTest label .* cannot send it'):
                with self.subTest(label=bad):
                    pass
        with self.assertRaisesRegex(TypeError, 'subTest label msg='):
            with self.subTest(np.int64(1)):
                pass

    def test_repr_is_the_way_out(self):
        with self.subTest(label=repr(np.int64(3))):
            pass
