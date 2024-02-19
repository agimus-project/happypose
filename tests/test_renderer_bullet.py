"""Set of unit tests for bullet renderer."""

import unittest
from pathlib import Path

import numpy as np
import torch
from numpy.testing import assert_array_less as np_assert_array_less
from numpy.testing import assert_equal as np_assert_equal
from torch.testing import assert_allclose as tr_assert_allclose

from happypose.toolbox.datasets.object_dataset import RigidObject, RigidObjectDataset
from happypose.toolbox.lib3d.transform import Transform
from happypose.toolbox.renderer.bullet_batch_renderer import BulletBatchRenderer
from happypose.toolbox.renderer.bullet_scene_renderer import BulletSceneRenderer


class TestBulletRenderer(unittest.TestCase):
    """Unit tests for bullet renderer."""

    def setUp(self) -> None:
        self.obj_label = (
            "obj_000001"  # TODO: current limitation: label must be equal to object name
        )
        self.obj_path = Path(__file__).parent.joinpath("data/obj_000001.ply")
        self.rigid_object_dataset = RigidObjectDataset(
            objects=[
                RigidObject(
                    label=self.obj_label, mesh_path=self.obj_path, mesh_units="mm"
                )
            ]
        )

        self.z_obj = 0.3
        self.TWO = Transform((0.5, 0.5, -0.5, 0.5), (0, 0, self.z_obj))
        self.object_datas = [
            {
                "name": self.obj_label,
                "TWO": self.TWO,
            }
        ]

        fx, fy = 300, 300
        cx, cy = 320, 240
        self.K = np.array(
            [
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1],
            ]
        )

        self.width, self.height = 640, 480
        self.TWC = Transform.Identity()
        self.Nc = 2
        self.camera_datas = self.Nc * [
            {
                "K": self.K,
                "resolution": (self.width, self.height),
                "TWC": self.TWC,
            }
        ]

    def test_scene_renderer(self):
        """
        Render an example object and check that output image match expectation.
        """
        SAVEFIG = False

        renderer = BulletSceneRenderer(
            asset_dataset=self.rigid_object_dataset, gpu_renderer=False
        )

        renderings = renderer.render_scene(
            self.object_datas,
            self.camera_datas,
            render_depth=True,
            render_binary_mask=True,
        )

        self.assertEqual(len(renderings), self.Nc)
        rgb = renderings[0].rgb
        depth = renderings[0].depth
        binary_mask = renderings[0].binary_mask

        assert renderings[0].normals is None

        self.assertEqual(rgb.shape, (self.height, self.width, 3))
        self.assertEqual(depth.shape, (self.height, self.width, 1))
        self.assertEqual(binary_mask.shape, (self.height, self.width, 1))

        self.assertEqual(rgb.dtype, np.dtype(np.uint8))
        self.assertEqual(depth.dtype, np.dtype(np.float32))
        self.assertEqual(binary_mask.dtype, np.dtype(bool))

        if SAVEFIG:
            import matplotlib.pyplot as plt

            plt.subplot(2, 2, 1)
            plt.imshow(rgb)
            plt.subplot(2, 2, 3)
            plt.imshow(depth, cmap=plt.cm.gray_r)
            plt.subplot(2, 2, 4)
            plt.imshow(binary_mask, cmap=plt.cm.gray)
            fig_path = (
                self.obj_path.parent / f"bullet_{self.obj_label}_scene_render.png"
            )
            print(f"Saved {fig_path}")
            plt.savefig(fig_path)

        # ================================
        self.assertIsNone(np_assert_equal(rgb[0, 0], (0, 0, 0)))
        self.assertIsNone(
            np_assert_array_less((0, 0, 0), rgb[self.height // 2, self.width // 2])
        )
        self.assertEqual(depth[0, 0], 0)
        self.assertLess(depth[self.height // 2, self.width // 2], self.z_obj)
        self.assertEqual(binary_mask[0, 0], 0)
        self.assertEqual(binary_mask[self.height // 2, self.width // 2], 1)

        # ===========================
        # Partial renderings
        # ===========================
        renderings = renderer.render_scene(
            self.object_datas,
            self.camera_datas,
            render_depth=True,
            render_binary_mask=False,
        )
        self.assertIsNotNone(renderings[0].rgb)
        self.assertIsNotNone(renderings[0].depth)
        self.assertIsNone(renderings[0].binary_mask)

        renderings = renderer.render_scene(
            self.object_datas,
            self.camera_datas,
            render_depth=False,
            render_binary_mask=False,
        )
        self.assertIsNotNone(renderings[0].rgb)
        self.assertIsNone(renderings[0].depth)
        self.assertIsNone(renderings[0].binary_mask)

        renderings = renderer.render_scene(
            self.object_datas,
            self.camera_datas,
            render_depth=False,
            render_binary_mask=True,
        )
        self.assertIsNotNone(renderings[0].rgb)
        self.assertIsNone(renderings[0].depth)
        self.assertIsNotNone(renderings[0].binary_mask)

    def test_batch_renderer(self):
        """
        Render an example object and check that output image match expectation.
        """
        SAVEFIG = False

        renderer = BulletBatchRenderer(
            asset_dataset=self.rigid_object_dataset,
            n_workers=0,
            preload_cache=True,
            gpu_renderer=False,
        )

        TCO = torch.from_numpy((self.TWC.inverse() * self.TWO).matrix)
        K = torch.from_numpy(self.K)
        TCO = TCO.unsqueeze(0)
        K = K.unsqueeze(0)
        TCO = TCO.repeat(self.Nc, 1, 1)
        K = K.repeat(self.Nc, 1, 1)

        renderings = renderer.render(
            labels=self.Nc * [self.obj_label],
            TCO=TCO,
            K=K,
            resolution=(self.width, self.height),
            render_depth=True,
            render_binary_mask=True,
        )

        self.assertIsNone(renderings.normals)  # normals not supported

        self.assertEqual(renderings.rgbs.shape, (self.Nc, 3, self.height, self.width))
        self.assertEqual(renderings.depths.shape, (self.Nc, 1, self.height, self.width))
        self.assertEqual(
            renderings.binary_masks.shape, (self.Nc, 1, self.height, self.width)
        )

        self.assertEqual(renderings.rgbs.dtype, torch.float32)
        self.assertEqual(renderings.depths.dtype, torch.float32)
        self.assertEqual(renderings.binary_masks.dtype, torch.bool)

        # Renders from 2 identical cams are equals
        self.assertIsNone(tr_assert_allclose(renderings.rgbs[0], renderings.rgbs[1]))
        self.assertIsNone(
            tr_assert_allclose(renderings.depths[0], renderings.depths[1])
        )
        self.assertIsNone(
            tr_assert_allclose(renderings.binary_masks[0], renderings.binary_masks[1])
        )

        rgb = renderings.rgbs[0].movedim(0, -1).numpy()  # (Nc,3,h,w) -> (h,w,3)
        depth = renderings.depths[0].movedim(0, -1).numpy()  # (Nc,1,h,w) -> (h,w,1)
        binary_mask = (
            renderings.binary_masks[0].movedim(0, -1).numpy()
        )  # (Nc,1,h,w) -> (h,w,1)

        if SAVEFIG:
            import matplotlib.pyplot as plt

            plt.subplot(2, 2, 1)
            plt.imshow(rgb)
            plt.subplot(2, 2, 3)
            plt.imshow(depth, cmap=plt.cm.gray_r)
            plt.subplot(2, 2, 4)
            plt.imshow(binary_mask, cmap=plt.cm.gray)
            fig_path = (
                self.obj_path.parent / f"bullet_{self.obj_label}_batch_render.png"
            )
            print(f"Saved {fig_path}")
            plt.savefig(fig_path)

        # ================================
        self.assertIsNone(np_assert_equal(rgb[0, 0], (0, 0, 0)))
        self.assertIsNone(
            np_assert_array_less((0, 0, 0), rgb[self.height // 2, self.width // 2])
        )
        self.assertEqual(depth[0, 0], 0)
        self.assertLess(depth[self.height // 2, self.width // 2], self.z_obj)
        self.assertEqual(binary_mask[0, 0], 0)
        self.assertEqual(binary_mask[self.height // 2, self.width // 2], 1)

        # ==================
        # Partial renderings
        # ==================
        renderings = renderer.render(
            self.Nc * [self.obj_label],
            TCO,
            K,
            resolution=(self.width, self.height),
            render_depth=True,
            render_binary_mask=False,
        )
        self.assertIsNotNone(renderings.rgbs)
        self.assertIsNotNone(renderings.depths)
        self.assertIsNone(renderings.binary_masks)

        renderings = renderer.render(
            self.Nc * [self.obj_label],
            TCO,
            K,
            resolution=(self.width, self.height),
            render_depth=False,
            render_binary_mask=False,
        )
        self.assertIsNotNone(renderings.rgbs)
        self.assertIsNone(renderings.depths)
        self.assertIsNone(renderings.binary_masks)

        renderings = renderer.render(
            self.Nc * [self.obj_label],
            TCO,
            K,
            resolution=(self.width, self.height),
            render_depth=False,
            render_binary_mask=True,
        )
        self.assertIsNotNone(renderings.rgbs)
        self.assertIsNone(renderings.depths)
        self.assertIsNotNone(renderings.binary_masks)


if __name__ == "__main__":
    unittest.main()
