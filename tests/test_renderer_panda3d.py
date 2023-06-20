import unittest
from pathlib import Path
import numpy as np
from numpy.testing import assert_equal

from happypose.toolbox.datasets.object_dataset import RigidObjectDataset, RigidObject
from happypose.toolbox.lib3d.transform import Transform
from happypose.toolbox.renderer.panda3d_scene_renderer import Panda3dSceneRenderer
from happypose.toolbox.renderer.types import (
    Panda3dObjectData,
    Panda3dCameraData,
    Panda3dLightData,
)


class TestRendererPanda3D(unittest.TestCase):
    def test_simple_render(self):
        renderer = Panda3dSceneRenderer(
            asset_dataset=RigidObjectDataset(
                objects=[
                    RigidObject(
                        label="obj",
                        mesh_path=Path(__file__).parent.joinpath("data/obj_000001.ply"),
                        mesh_units="mm",
                    )
                ]
            )
        )

        object_datas = [
            Panda3dObjectData(
                label="obj", TWO=Transform((0, 0, 0, 1), (0, 0, 1)), color=(1, 0, 0, 1)
            )
        ]
        camera_datas = [
            Panda3dCameraData(
                K=np.array(
                    [
                        [600.0, 0.0, 160.0],
                        [0.0, 600.0, 160.0],
                        [0.0, 0.0, 1.0],
                    ]
                ),
                resolution=(320, 320),
            )
        ]

        light_datas = [
            Panda3dLightData(
                light_type="ambient",
                color=(1.0, 1, 1, 1),
            ),
        ]

        renderings = renderer.render_scene(object_datas, camera_datas, light_datas)
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(1, 1, squeeze=True)  # type: plt.Figure, plt.Axes
        # ax.imshow(renderings[0].rgb)
        # plt.show()

        self.assertEqual(len(renderings), 1)
        rgb = renderings[0].rgb

        assert_equal(rgb[rgb.shape[0] // 2, rgb.shape[1] // 2], (255, 0, 0))
        assert_equal(rgb[0, 0], (0, 0, 0))


if __name__ == "__main__":
    unittest.main()
