"""Set of unit tests for Panda3D renderer."""

import os
from pathlib import Path
from typing import List

import numpy as np
import pytest
from numpy.testing import assert_array_less as np_assert_array_less
from numpy.testing import assert_equal as np_assert_equal
from torch.testing import assert_close as tr_assert_close

from happypose.toolbox.datasets.object_dataset import RigidObject, RigidObjectDataset
from happypose.toolbox.lib3d.transform import Transform
from happypose.toolbox.renderer.panda3d_scene_renderer import Panda3dSceneRenderer
from happypose.toolbox.renderer.types import (
    CameraRenderingData,
    Panda3dCameraData,
    Panda3dLightData,
    Panda3dObjectData,
)

from .config.test_config import DEVICE

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class TestPanda3DBatchRenderer:
    """Unit tests for Panda3D renderer."""

    @pytest.fixture(autouse=True)
    def setUp(self) -> None:
        self.obj_label = "my_favorite_object_label"
        self.obj_path = Path(__file__).parent.joinpath("data/obj_000001.ply")
        self.asset_dataset = RigidObjectDataset(
            objects=[
                RigidObject(
                    label=self.obj_label, mesh_path=self.obj_path, mesh_units="mm"
                ),
                RigidObject(label="NOT_USED", mesh_path=self.obj_path, mesh_units="mm"),
            ]
        )

        self.z_obj = 0.3
        self.TWO = Transform((0.5, 0.5, -0.5, 0.5), (0, 0, self.z_obj))
        self.object_datas = [Panda3dObjectData(label=self.obj_label, TWO=self.TWO)]

        fx, fy = 300, 300
        cx, cy = 320, 240
        self.K = np.array(
            [
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1],
            ]
        )

        self.TWC = Transform(np.eye(4))
        self.width, self.height = 640, 480
        self.Nc = 4
        self.camera_datas = self.Nc * [
            Panda3dCameraData(
                K=self.K, resolution=(self.height, self.width), TWC=self.TWC
            )
        ]

        Nb_lights = 3
        self.light_datas = Nb_lights * [
            Panda3dLightData(light_type="ambient", color=(1.0, 1.0, 1.0, 1.0))
        ]

    @pytest.mark.order(4)
    @pytest.mark.parametrize("device", DEVICE)
    def test_scene_renderer(self, device):
        """
        Scene render an example object and check that output image match expectation.
        """
        SAVEFIG = False

        renderer = Panda3dSceneRenderer(asset_dataset=self.asset_dataset)

        renderings: List[CameraRenderingData] = renderer.render_scene(
            self.object_datas,
            self.camera_datas,
            self.light_datas,
            render_normals=True,
            render_depth=True,
            render_binary_mask=True,
        )

        assert len(renderings) == self.Nc

        # render from 2 identical cams are equals
        assert tr_assert_close(renderings[0].rgb, renderings[1].rgb) is None
        assert tr_assert_close(renderings[0].normals, renderings[1].normals) is None
        assert (
            tr_assert_close(
                renderings[0].depth, renderings[1].depth, atol=1e-3, rtol=1e-3
            )
            is None
        )
        assert (
            tr_assert_close(renderings[0].binary_mask, renderings[1].binary_mask)
            is None
        )

        rgb = renderings[0].rgb
        normals = renderings[0].normals
        depth = renderings[0].depth
        binary_mask = renderings[0].binary_mask

        assert rgb.shape == (self.height, self.width, 3)
        assert normals.shape == (self.height, self.width, 3)
        assert depth.shape == (self.height, self.width, 1)
        assert binary_mask.shape == (self.height, self.width, 1)

        assert rgb.dtype == np.dtype(np.uint8)
        assert normals.dtype == np.dtype(np.uint8)
        assert depth.dtype == np.dtype(np.float32)
        assert binary_mask.dtype == np.dtype(bool)

        if SAVEFIG:
            import matplotlib.pyplot as plt

            plt.subplot(2, 2, 1)
            plt.imshow(rgb)
            plt.subplot(2, 2, 2)
            plt.imshow(normals)
            plt.subplot(2, 2, 3)
            plt.imshow(depth, cmap=plt.cm.gray_r)
            plt.subplot(2, 2, 4)
            plt.imshow(binary_mask, cmap=plt.cm.gray)
            fig_path = (
                self.obj_path.parent / f"panda3d_{self.obj_label}_scene_render.png"
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
        assert np_assert_equal(normals[0, 0], (0, 0, 0)) is None
        assert (
            np_assert_array_less((0, 0, 0), normals[self.height // 2, self.width // 2])
            is None
        )
        assert binary_mask[0, 0] == 0
        assert binary_mask[self.height // 2, self.width // 2] == 1

        # ==================
        # Partial renderings
        # ==================
        renderings = renderer.render_scene(
            self.object_datas,
            self.camera_datas,
            self.light_datas,
            render_depth=True,
            render_normals=True,
            render_binary_mask=False,
        )
        assert renderings[0].rgb is not None
        assert renderings[0].depth is not None
        assert renderings[0].normals is not None
        assert renderings[0].binary_mask is None

        renderings = renderer.render_scene(
            self.object_datas,
            self.camera_datas,
            self.light_datas,
            render_depth=True,
            render_normals=False,
            render_binary_mask=False,
        )
        assert renderings[0].rgb is not None
        assert renderings[0].depth is not None
        assert renderings[0].normals is None
        assert renderings[0].binary_mask is None

        renderings = renderer.render_scene(
            self.object_datas,
            self.camera_datas,
            self.light_datas,
            render_depth=False,
            render_normals=True,
            render_binary_mask=False,
        )
        assert renderings[0].rgb is not None
        assert renderings[0].depth is None
        assert renderings[0].normals is not None
        assert renderings[0].binary_mask is None

        renderings = renderer.render_scene(
            self.object_datas,
            self.camera_datas,
            self.light_datas,
            render_depth=False,
            render_normals=False,
            render_binary_mask=False,
        )
        assert renderings[0].rgb is not None
        assert renderings[0].depth is None
        assert renderings[0].normals is None
        assert renderings[0].binary_mask is None

        with pytest.raises(AssertionError):
            renderer.render_scene(
                self.object_datas,
                self.camera_datas,
                self.light_datas,
                render_depth=False,
                render_normals=False,
                render_binary_mask=True,
            )
