# %%
import unittest

import numpy as np

from cora.helpers import softmax


class TestSoftmaxFunction(unittest.TestCase):
    def test_softmax_basic(self):
        vec = np.array([[1, 2, 3], [1, 2, 3]])
        expected = np.array([[0.09003057, 0.24472847, 0.66524096], [0.09003057, 0.24472847, 0.66524096]])
        result = softmax(vec)
        np.testing.assert_almost_equal(result, expected, decimal=6, err_msg="Basic softmax test failed")

    def test_softmax_sum_to_one(self):
        vec = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = softmax(vec)
        row_sums = np.sum(result, axis=1)
        np.testing.assert_almost_equal(row_sums, np.ones(row_sums.shape), decimal=6, err_msg="Sum to one test failed")


if __name__ == "__main__":
    unittest.main()
