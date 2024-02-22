import unittest

import numpy as np
import pinocchio as pin
import torch

# from numpy.testing import assert_equal as np.allclose
from happypose.toolbox.lib3d.transform import Transform


class TestTransform(unittest.TestCase):
    """
    Test the valid inputs and behavior of happypose custom SE(3) transform class.
    """

    def test_constructor(self):
        M1 = pin.SE3.Random()
        M2 = pin.SE3.Random()
        M3 = M1 * M2

        # 1 arg constructor
        T1 = Transform(M1)
        self.assertTrue(T1._T == M1)
        T1b = Transform(T1)
        T1c = Transform(M1.homogeneous)
        T1d = Transform(torch.from_numpy(M1.homogeneous))
        self.assertTrue(T1 == T1b == T1c == T1d)

        # 2 args constructor
        R1 = M1.rotation
        t1 = M1.translation

        T1e = Transform(R1, t1)
        T1f = Transform(torch.from_numpy(R1), t1)
        T1g = Transform(pin.Quaternion(R1), t1)
        T1h = Transform(pin.Quaternion(R1).coeffs(), t1)
        T1h = Transform(tuple(pin.Quaternion(R1).coeffs()), t1)
        T1i = Transform(list(pin.Quaternion(R1).coeffs()), t1)
        T1j = Transform(torch.from_numpy(pin.Quaternion(R1).coeffs().copy()), t1)

        self.assertTrue(T1 == T1e == T1f)
        self.assertTrue(np.allclose(T1._T, T1g._T))
        self.assertTrue(T1g == T1h == T1h == T1i == T1j)

        # Conversions
        self.assertTrue(np.allclose(M1.homogeneous, T1.toHomogeneousMatrix()))
        self.assertTrue(np.allclose(M1.homogeneous, T1.matrix))
        self.assertTrue(torch.allclose(torch.from_numpy(M1.homogeneous), T1.tensor))

        # Composition
        T2 = Transform(M2)
        T3 = Transform(M3)
        T3m = T1 * T2
        self.assertTrue(T3 == T3m)

        # Inverse
        T1inv = Transform(T1.inverse())
        self.assertTrue(T1inv == T1.inverse())


if __name__ == "__main__":
    unittest.main()
