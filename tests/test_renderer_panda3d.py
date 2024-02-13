"""Set of unit tests for Panda3D renderer."""
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_array_less, assert_equal

from happypose.toolbox.datasets.object_dataset import RigidObject, RigidObjectDataset
from happypose.toolbox.lib3d.transform import Transform
from happypose.toolbox.renderer.panda3d_scene_renderer import Panda3dSceneRenderer
from happypose.toolbox.renderer.types import (
    Panda3dCameraData,
    Panda3dLightData,
    Panda3dObjectData,
)


class TestPanda3DRenderer(unittest.TestCase):
    """Unit tests for Panda3D renderer."""

    def test_scene_renderer(self):
        """
        Render an example object and check that output image match expectation.
        """
        SAVEFIG = False

        obj_label = 'obj_000001'
        obj_path = Path(__file__).parent.joinpath(f"data/{obj_label}.ply")
        
        renderer = Panda3dSceneRenderer(
            asset_dataset=RigidObjectDataset(
                objects=[
                    RigidObject(
                        label="obj",
                        mesh_path=obj_path,
                        mesh_units="mm",
                    ),
                ],
            ),
        )
        z_obj = 0.3
        object_datas = [
            Panda3dObjectData(
                label="obj",
                TWO=Transform((0.5, 0.5, -0.5, 0.5), (0, 0, z_obj)),
            ),
        ]

        fx, fy = 300, 300
        cx, cy = 320, 240
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ])
        
        width, height = 640, 480
        camera_datas = [
            Panda3dCameraData(
                K=K,
                resolution=(height, width),
                TWC=Transform(np.eye(4))
            ),
        ]

        light_datas = [
            Panda3dLightData(
                light_type="ambient",
                color=(1.0, 1.0, 1.0, 1.0),  # 
            ),
        ]

        renderings = renderer.render_scene(object_datas, camera_datas, light_datas,
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
            fig_path = obj_path.parent / f'panda3d_{obj_label}_render.png'
            print(f'Saved {fig_path}')
            plt.savefig(fig_path)


        # ================================
        assert_equal(rgb[0, 0], (0, 0, 0))
        assert_array_less((0, 0, 0), rgb[height // 2, width // 2])

        assert depth[0, 0] == 0
        assert depth[height // 2, width // 2] < z_obj 

        assert_equal(normals[0, 0], (0, 0, 0))
        assert_array_less((0, 0, 0), normals[height // 2, width // 2])

        assert binary_mask[0,0] == 0        
        assert binary_mask[height // 2, width // 2] == 1        
  


if __name__ == "__main__":
    unittest.main()
