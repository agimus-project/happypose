"""Set of unit tests for bullet renderer."""

from pathlib import Path

import numpy as np
import pytest
import torch
from numpy.testing import assert_array_less as np_assert_array_less
from numpy.testing import assert_equal as np_assert_equal
from torch.testing import assert_close as tr_assert_close

from happypose.toolbox.datasets.object_dataset import RigidObject, RigidObjectDataset
from happypose.toolbox.lib3d.transform import Transform
from happypose.toolbox.renderer.bullet_batch_renderer import BulletBatchRenderer
from happypose.toolbox.renderer.bullet_scene_renderer import BulletSceneRenderer

from .config.test_config import DEVICE


class TestBulletRenderer:
    """Unit tests for bullet renderer."""

    @pytest.fixture(autouse=True)
    def setUp(self) -> None:
        self.obj_label = "my_favorite_object_label"
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

    @pytest.mark.parametrize("device", DEVICE)
    def test_scene_renderer(self, device):
        """
        Render an example object and check that output image match expectation.
        """
        SAVEFIG = False
        if device == "cpu":
            renderer = BulletSceneRenderer(
                asset_dataset=self.rigid_object_dataset, gpu_renderer=False
            )
        else:
            renderer = BulletSceneRenderer(
                asset_dataset=self.rigid_object_dataset, gpu_renderer=True
            )

        renderings = renderer.render_scene(
            self.object_datas,
            self.camera_datas,
            render_depth=True,
            render_binary_mask=True,
        )

        assert len(renderings) == self.Nc
        rgb = renderings[0].rgb
        depth = renderings[0].depth
        binary_mask = renderings[0].binary_mask

        assert renderings[0].normals is None

        assert rgb.shape == (self.height, self.width, 3)
        assert depth.shape == (self.height, self.width, 1)
        assert binary_mask.shape == (self.height, self.width, 1)

        assert rgb.dtype == np.dtype(np.uint8)
        assert depth.dtype == np.dtype(np.float32)
        assert binary_mask.dtype == np.dtype(bool)

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
        assert np_assert_equal(rgb[0, 0], (0, 0, 0)) is None
        assert (
            np_assert_array_less((0, 0, 0), rgb[self.height // 2, self.width // 2])
            is None
        )
        assert depth[0, 0] == 0
        assert depth[self.height // 2, self.width // 2] < self.z_obj
        assert binary_mask[0, 0] == 0
        assert binary_mask[self.height // 2, self.width // 2] == 1

        # ===========================
        # Partial renderings
        # ===========================
        renderings = renderer.render_scene(
            self.object_datas,
            self.camera_datas,
            render_depth=True,
            render_binary_mask=False,
        )
        assert renderings[0].rgb is not None
        assert renderings[0].depth is not None
        assert renderings[0].binary_mask is None

        renderings = renderer.render_scene(
            self.object_datas,
            self.camera_datas,
            render_depth=False,
            render_binary_mask=False,
        )
        assert renderings[0].rgb is not None
        assert renderings[0].depth is None
        assert renderings[0].binary_mask is None

        renderings = renderer.render_scene(
            self.object_datas,
            self.camera_datas,
            render_depth=False,
            render_binary_mask=True,
        )
        assert renderings[0].rgb is not None
        assert renderings[0].depth is None
        assert renderings[0].binary_mask is not None

    @pytest.mark.parametrize("device", DEVICE)
    def test_batch_renderer(self, device):
        """
        Render an example object and check that output image match expectation.
        """
        SAVEFIG = False
        if device == "cpu":
            renderer = BulletBatchRenderer(
                asset_dataset=self.rigid_object_dataset,
                n_workers=0,
                preload_cache=True,
                gpu_renderer=False,
            )
        else:
            renderer = BulletBatchRenderer(
                asset_dataset=self.rigid_object_dataset,
                n_workers=0,
                preload_cache=True,
                gpu_renderer=True,
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

        assert renderings.normals is None  # normals not supported

        assert renderings.rgbs.shape == (self.Nc, 3, self.height, self.width)
        assert renderings.depths.shape == (self.Nc, 1, self.height, self.width)
        assert renderings.binary_masks.shape == (self.Nc, 1, self.height, self.width)

        assert renderings.rgbs.dtype == torch.float32
        assert renderings.depths.dtype == torch.float32
        assert renderings.binary_masks.dtype == torch.bool

        # Renders from 2 identical cams are equals
        assert tr_assert_close(renderings.rgbs[0], renderings.rgbs[1]) is None
        assert tr_assert_close(renderings.depths[0], renderings.depths[1]) is None
        assert (
            tr_assert_close(renderings.binary_masks[0], renderings.binary_masks[1])
            is None
        )

        if device == "cpu":
            rgb = renderings.rgbs[0].movedim(0, -1).numpy()  # (Nc,3,h,w) -> (h,w,3)
            depth = renderings.depths[0].movedim(0, -1).numpy()  # (Nc,1,h,w) -> (h,w,1)
            binary_mask = (
                renderings.binary_masks[0].movedim(0, -1).numpy()
            )  # (Nc,1,h,w) -> (h,w,1)
        else:
            rgb = (
                renderings.rgbs[0].movedim(0, -1).cpu().numpy()
            )  # (Nc,3,h,w) -> (h,w,3)
            depth = (
                renderings.depths[0].movedim(0, -1).cpu().numpy()
            )  # (Nc,1,h,w) -> (h,w,1)
            binary_mask = (
                renderings.binary_masks[0].movedim(0, -1).cpu().numpy()
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
        assert np_assert_equal(rgb[0, 0], (0, 0, 0)) is None
        assert (
            np_assert_array_less((0, 0, 0), rgb[self.height // 2, self.width // 2])
            is None
        )
        assert depth[0, 0] == 0
        assert depth[self.height // 2, self.width // 2] < self.z_obj
        assert binary_mask[0, 0] == 0
        assert binary_mask[self.height // 2, self.width // 2] == 1

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
        assert renderings.rgbs is not None
        assert renderings.depths is not None
        assert renderings.binary_masks is None

        renderings = renderer.render(
            self.Nc * [self.obj_label],
            TCO,
            K,
            resolution=(self.width, self.height),
            render_depth=False,
            render_binary_mask=False,
        )
        assert renderings.rgbs is not None
        assert renderings.depths is None
        assert renderings.binary_masks is None

        renderings = renderer.render(
            self.Nc * [self.obj_label],
            TCO,
            K,
            resolution=(self.width, self.height),
            render_depth=False,
            render_binary_mask=True,
        )
        assert renderings.rgbs is not None
        assert renderings.depths is None
        assert renderings.binary_masks is not None
