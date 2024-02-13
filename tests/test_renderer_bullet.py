"""Set of unit tests for Panda3D renderer."""
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_array_less, assert_equal

from happypose.toolbox.datasets.object_dataset import RigidObject, RigidObjectDataset
from happypose.toolbox.lib3d.transform import Transform
from happypose.toolbox.renderer.bullet_scene_renderer import BulletSceneRenderer


class TestBulletRenderer(unittest.TestCase):
    """Unit tests for Panda3D renderer."""

    def test_scene_renderer(self):
        """
        Render an example object and check that output image match expectation.
        """
        SAVEFIG = False

        obj_label = 'obj_000001'
        obj_path = Path(__file__).parent.joinpath(f"data/{obj_label}.ply")
        
        rigid_object_dataset = RigidObjectDataset(
            objects=[
                RigidObject(
                    label=obj_label,
                    mesh_path=obj_path,
                    mesh_units="mm",
                ),
            ],
        )
        renderer = BulletSceneRenderer(asset_dataset=rigid_object_dataset, gpu_renderer=False)

        z_obj = 0.3
        TWO = Transform((0.5, 0.5, -0.5, 0.5), (0, 0, z_obj))
        object_datas = [{
            "name": obj_label,
            "TWO": TWO,
        }]

        fx, fy = 300, 300
        cx, cy = 320, 240
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ])

        width, height = 640, 480
        camera_datas = [{
            "K": K,
            "resolution": (width, height),
            "TWC": Transform(np.eye(4)),
        }]

        renderings = renderer.render_scene(object_datas, camera_datas,
                                           render_depth=True, render_binary_mask=True)

        self.assertEqual(len(renderings), 1)
        rgb = renderings[0].rgb
        depth = renderings[0].depth
        binary_mask = renderings[0].binary_mask

        if SAVEFIG:
            import matplotlib.pyplot as plt
            plt.subplot(1,2,1)
            plt.imshow(rgb)
            plt.subplot(1,2,2)
            plt.imshow(depth, cmap=plt.cm.gray_r)
            fig_path = obj_path.parent / f'panda3d_{obj_label}_render.png'
            print(f'Saved {fig_path}')
            plt.savefig(fig_path)

        # ================================
        assert_equal(rgb[0, 0], (0, 0, 0))
        assert_array_less((0, 0, 0), rgb[height // 2, width // 2])

        assert depth[0, 0] == 0
        assert depth[height // 2, width // 2] < z_obj   

        assert binary_mask[0,0] == 0        
        assert binary_mask[height // 2, width // 2] == 1        


if __name__ == "__main__":
    unittest.main()
