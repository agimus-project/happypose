import unittest

import numpy as np
import pinocchio as pin
import torch

from happypose.toolbox.lib3d.rotations import (
    angle_axis_to_rotation_matrix,
    compute_rotation_matrix_from_quaternions,
    euler2quat,
    quat2mat,
    quaternion_to_angle_axis,
)

# from numpy.testing import assert_equal as np.allclose
from happypose.toolbox.lib3d.transform import Transform
from happypose.toolbox.lib3d.transform_ops import (
    invert_transform_matrices,
    transform_pts,
)


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


class TestRotations(unittest.TestCase):

    """
    Compare results with their equivalent pinocchio implementation.
    """

    N = 4

    def setUp(self) -> None:
        self.quats_ts = torch.rand(self.N, 4)
        self.quats_ts_norm = self.quats_ts / torch.norm(
            self.quats_ts, p=2, dim=-1, keepdim=True
        )
        self.quats_arr = self.quats_ts.numpy()
        self.quats_arr_norm = self.quats_ts_norm.numpy()

    def test_euler2quat(self):
        rpy = np.random.random(3)
        q_xyzw = euler2quat(rpy, axes="sxyz")
        q_pin_xyzw = pin.Quaternion(pin.rpy.rpyToMatrix(rpy)).coeffs()
        self.assertTrue(np.allclose(q_xyzw, q_pin_xyzw))

    def test_angle_axis_to_rotation_matrix(self):
        aa_ts = torch.rand(self.N, 3)
        R_ts = angle_axis_to_rotation_matrix(aa_ts)
        R = pin.exp3(aa_ts.numpy()[1])
        # Rotation matrices are actually transformations with zero translation
        self.assertTrue(R_ts.shape == (self.N, 4, 4))
        self.assertTrue(np.allclose(R_ts[1, :3, :3].numpy(), R, atol=1e-6))

    def test_quaternion_to_angle_axis(self):
        # quaternion_to_angle_axis assumes a wxyz quaternion order convention
        quats_ts_norm = torch.hstack(
            [self.quats_ts_norm[:, -1:], self.quats_ts_norm[:, :3]]
        )
        aa_ts = quaternion_to_angle_axis(quats_ts_norm)
        aa = pin.log3(pin.Quaternion(self.quats_arr_norm[1]).toRotationMatrix())
        self.assertTrue(np.allclose(aa_ts.numpy()[1], aa, atol=1e-6))

    def test_quat2mat(self):
        # quat2mat assumes a wxyz quaternion order convention
        R_ts = quat2mat(self.quats_ts)
        R = pin.Quaternion(self.quats_arr_norm[1]).toRotationMatrix()
        self.assertTrue(np.allclose(R_ts[1, :3, :3].numpy(), R, atol=1e-6))

    def test_compute_rotation_matrix_from_quaternions(self):
        # quaternion_to_angle_axis assumes a xyzw quaternion order convention
        R_ts = compute_rotation_matrix_from_quaternions(self.quats_ts)
        R = pin.Quaternion(self.quats_arr_norm[1]).toRotationMatrix()
        self.assertTrue(np.allclose(R_ts.numpy()[1], R, atol=1e-6))


class TestTransformOps(unittest.TestCase):
    n_pts = 10
    n_T = 5

    def setUp(self) -> None:
        self.pts = torch.rand(1, self.n_pts, 3)  # (1,n_pts,3)
        self.Tpin_lst = [pin.SE3.Random() for _ in range(self.n_T)]
        self.T_ts = torch.stack(
            [torch.Tensor(Tpin.homogeneous) for Tpin in self.Tpin_lst]
        )  # (n_T,4,4)
        self.T_ts = self.T_ts.unsqueeze(0)  # (1,n_T,4,4)

    def test_transform_pts(self):
        pts_trans_ts = transform_pts(self.T_ts, self.pts)
        pts_trans_arr = np.zeros((1, self.n_T, self.n_pts, 3))
        for i in range(self.n_pts):
            for j in range(self.n_T):
                pts_trans_arr[0, j, i] = self.Tpin_lst[j] * self.pts[0, i].numpy()
        self.assertTrue(np.allclose(pts_trans_ts.numpy(), pts_trans_arr, atol=1e-6))

    def test_invert_transform_matrices(self):
        T_ts_inv = invert_transform_matrices(self.T_ts)
        T_arr_inv = np.zeros((self.n_T, 4, 4))
        for i in range(self.n_T):
            T_arr_inv[i] = self.Tpin_lst[i].inverse().homogeneous
        self.assertTrue(np.allclose(T_ts_inv.numpy(), T_arr_inv, atol=1e-6))


class TestsDistances(unittest.TestCase):
    # TODO
    pass


if __name__ == "__main__":
    unittest.main()
