"""Set of unit tests for Panda3D renderer."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from numpy.testing import assert_array_less as np_assert_array_less
from numpy.testing import assert_equal as np_assert_equal
from torch.testing import assert_close as tr_assert_close

from happypose.toolbox.datasets.object_dataset import RigidObject, RigidObjectDataset
from happypose.toolbox.lib3d.transform import Transform
from happypose.toolbox.renderer.panda3d_batch_renderer import Panda3dBatchRenderer
from happypose.toolbox.renderer.types import (
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

    @pytest.mark.order(2)
    @pytest.mark.parametrize("device", DEVICE)
    def test_batch_renderer(self, device):
        """
        Batch render an example object and check that output image match expectation.
        """
        SAVEFIG = False

        renderer = Panda3dBatchRenderer(
            asset_dataset=self.asset_dataset,
            n_workers=4,
            preload_cache=True,
            split_objects=False,
        )

        TCO = torch.from_numpy((self.TWC.inverse() * self.TWO).matrix)
        K = torch.from_numpy(self.K)
        TCO = TCO.unsqueeze(0)
        K = K.unsqueeze(0)
        TCO = TCO.repeat(self.Nc, 1, 1)
        K = K.repeat(self.Nc, 1, 1)

        # labels and light_datas need to have same size as TCO/K batch size
        renderings = renderer.render(
            labels=self.Nc * [self.obj_label],
            TCO=TCO,
            K=K,
            light_datas=self.Nc * [self.light_datas],
            resolution=(self.height, self.width),
            render_normals=True,
            render_depth=True,
            render_binary_mask=True,
        )

        assert renderings.rgbs.shape == (self.Nc, 3, self.height, self.width)
        assert renderings.depths.shape == (self.Nc, 1, self.height, self.width)
        assert renderings.normals.shape == (self.Nc, 3, self.height, self.width)
        assert renderings.binary_masks.shape == (self.Nc, 1, self.height, self.width)

        assert renderings.rgbs.dtype == torch.float32
        assert renderings.depths.dtype == torch.float32
        assert renderings.normals.dtype == torch.float32
        assert renderings.binary_masks.dtype == torch.bool

        # Renders from 2 identical cams are equals
        assert tr_assert_close(renderings.rgbs[0], renderings.rgbs[1]) is None
        assert tr_assert_close(renderings.normals[0], renderings.normals[1]) is None
        assert (
            tr_assert_close(
                renderings.depths[0], renderings.depths[1], atol=1e-3, rtol=1e-3
            )
            is None
        )
        assert (
            tr_assert_close(renderings.binary_masks[0], renderings.binary_masks[1])
            is None
        )

        if device == "cpu":
            rgb = renderings.rgbs[0].movedim(0, -1).numpy()  # (Nc,3,h,w) -> (h,w,3)
            normals = (
                renderings.normals[0].movedim(0, -1).numpy()
            )  # (Nc,1,h,w) -> (h,w,1)
            depth = renderings.depths[0].movedim(0, -1).numpy()  # (Nc,3,h,w) -> (h,w,3)
            binary_mask = (
                renderings.binary_masks[0].movedim(0, -1).numpy()
            )  # (Nc,1,h,w) -> (h,w,1)

        else:
            rgb = (
                renderings.rgbs[0].movedim(0, -1).cpu().numpy()
            )  # (Nc,3,h,w) -> (h,w,3)
            normals = (
                renderings.normals[0].movedim(0, -1).cpu().numpy()
            )  # (Nc,1,h,w) -> (h,w,1)
            depth = (
                renderings.depths[0].movedim(0, -1).cpu().numpy()
            )  # (Nc,3,h,w) -> (h,w,3)
            binary_mask = (
                renderings.binary_masks[0].movedim(0, -1).cpu().numpy()
            )  # (Nc,1,h,w) -> (h,w,1)

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
                self.obj_path.parent / f"panda3d_{self.obj_label}_batch_render.png"
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
        renderings = renderer.render(
            self.Nc * [self.obj_label],
            TCO,
            K,
            light_datas=self.Nc * [self.light_datas],
            resolution=(self.height, self.width),
            render_depth=True,
            render_normals=True,
            render_binary_mask=False,
        )
        assert renderings.rgbs is not None
        assert renderings.depths is not None
        assert renderings.normals is not None
        assert renderings.binary_masks is None

        renderings = renderer.render(
            self.Nc * [self.obj_label],
            TCO,
            K,
            light_datas=self.Nc * [self.light_datas],
            resolution=(self.height, self.width),
            render_depth=True,
            render_normals=False,
            render_binary_mask=False,
        )
        assert renderings.rgbs is not None
        assert renderings.depths is not None
        assert renderings.normals is None
        assert renderings.binary_masks is None

        renderings = renderer.render(
            self.Nc * [self.obj_label],
            TCO,
            K,
            light_datas=self.Nc * [self.light_datas],
            resolution=(self.height, self.width),
            render_depth=False,
            render_normals=True,
            render_binary_mask=False,
        )
        assert renderings.rgbs is not None
        assert renderings.depths is None
        assert renderings.normals is not None
        assert renderings.binary_masks is None

        renderings = renderer.render(
            self.Nc * [self.obj_label],
            TCO,
            K,
            light_datas=self.Nc * [self.light_datas],
            resolution=(self.height, self.width),
            render_depth=False,
            render_normals=False,
            render_binary_mask=False,
        )
        assert renderings.rgbs is not None
        assert renderings.depths is None
        assert renderings.normals is None
        assert renderings.binary_masks is None

        # TODO: not sure how to check that assertion is raised without crashing the test
        # -> AssertionError happens in a subprocess and is not caught
        # self.assertRaises(
        #     AssertionError, renderer.render,
        #     self.Nc*[self.obj_label],
        #     TCO,
        #     K,
        #     light_datas=self.Nc*[self.light_datas],
        #     resolution=(self.height, self.width),
        #     render_depth=False,
        #     render_normals=False,
        #     render_binary_mask=True
        # )
