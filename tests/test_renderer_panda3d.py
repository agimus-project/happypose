"""Set of unit tests for Panda3D renderer."""
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_equal

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
        TODO: 
        - check depth
        - check mask
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

        object_datas = [
            Panda3dObjectData(
                label="obj",
                TWO=Transform((0.5, 0.5, -0.5, 0.5), (0, 0, 0.3)),
            ),
        ]

        fx, fy = 300, 300
        cx, cy = 320, 240
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ])
        camera_datas = [
            Panda3dCameraData(
                K=K,
                resolution=(480, 640),
                TWC=Transform(np.eye(4))
            ),
        ]

        light_datas = [
            Panda3dLightData(
                light_type="ambient",
                color=(0.1, 0.1, 0.1, 1.0),  # 
            ),
        ]

        renderings = renderer.render_scene(object_datas, camera_datas, light_datas)

        self.assertEqual(len(renderings), 1)
        rgb = renderings[0].rgb

        if SAVEFIG:
            import matplotlib.pyplot as plt
            plt.imshow(rgb)
            fig_path = obj_path.parent / f'panda3d_{obj_label}_render.png'
            print(f'Saved {fig_path}')
            plt.savefig(fig_path)

        # Color check hard to implement since depends on luminosity of the scene
        # assert_equal(rgb[rgb.shape[0] // 2, rgb.shape[1] // 2], (255, 0, 0))        
        assert_equal(rgb[0, 0], (0, 0, 0))


if __name__ == "__main__":
    unittest.main()
