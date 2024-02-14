"""Set of unit tests for Panda3D renderer."""
import unittest
from pathlib import Path

import numpy as np
import torch
from numpy.testing import assert_array_less, assert_equal

from happypose.toolbox.datasets.object_dataset import RigidObject, RigidObjectDataset
from happypose.toolbox.lib3d.transform import Transform
from happypose.toolbox.renderer.panda3d_batch_renderer import Panda3dBatchRenderer
from happypose.toolbox.renderer.panda3d_scene_renderer import Panda3dSceneRenderer
from happypose.toolbox.renderer.types import (
    Panda3dCameraData,
    Panda3dLightData,
    Panda3dObjectData,
)


class TestPanda3DRenderer(unittest.TestCase):
    """Unit tests for Panda3D renderer."""


    def setUp(self) -> None:

        self.obj_label = 'obj_000001'
        self.obj_path = Path(__file__).parent.joinpath(f"data/{self.obj_label}.ply")
        self.asset_dataset = RigidObjectDataset(
            objects=[
                RigidObject(
                    label="obj",
                    mesh_path=self.obj_path,
                    mesh_units="mm"
                )
            ]
        )
        
        self.z_obj = 0.3
        self.TWO=Transform((0.5, 0.5, -0.5, 0.5), (0, 0, self.z_obj))
        self.object_datas = [
            Panda3dObjectData(
                label="obj",
                TWO=self.TWO
            )
        ]

        fx, fy = 300, 300
        cx, cy = 320, 240
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ])
        
        self.TWC = Transform(np.eye(4))
        self.width, self.height = 640, 480
        self.camera_datas = [
            Panda3dCameraData(
                K=K,
                resolution=(self.height, self.width),
                TWC=self.TWC
            )
        ]

        self.light_datas = [
            Panda3dLightData(
                light_type="ambient",
                color=(1.0, 1.0, 1.0, 1.0)
            ),
        ]

    def test_scene_renderer(self):
        """
        Scene render an example object and check that output image match expectation.
        """
        SAVEFIG = False

        renderer = Panda3dSceneRenderer(
            asset_dataset=self.asset_dataset
        )

        renderings = renderer.render_scene(self.object_datas, self.camera_datas, self.light_datas,
                                           render_depth=True, render_normals=True, render_binary_mask=True)

        self.assertEqual(len(renderings), 1)
        rgb = renderings[0].rgb
        depth = renderings[0].depth
        normals = renderings[0].normals
        binary_mask = renderings[0].binary_mask

        if SAVEFIG:
            import matplotlib.pyplot as plt
            plt.subplot(1,3,1)
            plt.imshow(rgb)
            plt.subplot(1,3,2)
            plt.imshow(depth, cmap=plt.cm.gray_r)
            plt.subplot(1,3,3)
            plt.imshow(normals)
            fig_path = self.obj_path.parent / f'panda3d_{self.obj_label}_render.png'
            print(f'Saved {fig_path}')
            plt.savefig(fig_path)

        # ================================
        assert_equal(rgb[0, 0], (0, 0, 0))
        assert_array_less((0, 0, 0), rgb[self.height // 2, self.width // 2])

        assert depth[0, 0] == 0
        assert depth[self.height // 2, self.width // 2] < self.z_obj 

        assert_equal(normals[0, 0], (0, 0, 0))
        assert_array_less((0, 0, 0), normals[self.height // 2, self.width // 2])

        assert binary_mask[0,0] == 0        
        assert binary_mask[self.height // 2, self.width // 2] == 1        
  

    def test_batch_renderer(self):
        """
        Batch render an example object and check that output image match expectation.
        """
        SAVEFIG = False

        renderer = Panda3dBatchRenderer(
            object_dataset=self.asset_dataset,
            n_workers=1,
            preload_cache=True,
            split_objects=False,
        )

        TCO = torch.from_numpy((self.TWC.inverse() * self.TWO).matrix)
        K = torch.from_numpy(K)
        renderings = renderer.render([self.obj_label], TCO, K, self.light_datas,
                                    render_depth=True, render_normals=True, render_binary_mask=True)

        self.assertEqual(len(renderings), 1)
        rgb = renderings[0].rgb
        depth = renderings[0].depth
        normals = renderings[0].normals
        binary_mask = renderings[0].binary_mask

        if SAVEFIG:
            import matplotlib.pyplot as plt
            plt.subplot(1,3,1)
            plt.imshow(rgb)
            plt.subplot(1,3,2)
            plt.imshow(depth, cmap=plt.cm.gray_r)
            plt.subplot(1,3,3)
            plt.imshow(normals)
            fig_path = self.obj_path.parent / f'panda3d_{self.obj_label}_render.png'
            print(f'Saved {fig_path}')
            plt.savefig(fig_path)

        # ================================
        assert_equal(rgb[0, 0], (0, 0, 0))
        assert_array_less((0, 0, 0), rgb[self.height // 2, self.width // 2])

        assert depth[0, 0] == 0
        assert depth[self.height // 2, self.width // 2] < self.z_obj 

        assert_equal(normals[0, 0], (0, 0, 0))
        assert_array_less((0, 0, 0), normals[self.height // 2, self.width // 2])

        assert binary_mask[0,0] == 0        
        assert binary_mask[self.height // 2, self.width // 2] == 1        
  


if __name__ == "__main__":
    unittest.main()
