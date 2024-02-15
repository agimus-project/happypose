"""Set of unit tests for Panda3D renderer."""
import unittest
from pathlib import Path
from typing import List

import numpy as np
import torch
from numpy.testing import assert_array_less as np_assert_array_less, assert_equal as np_assert_equal
from torch.testing import assert_allclose as tr_assert_allclose

from happypose.toolbox.datasets.object_dataset import RigidObject, RigidObjectDataset
from happypose.toolbox.lib3d.transform import Transform
from happypose.toolbox.renderer.panda3d_batch_renderer import Panda3dBatchRenderer
from happypose.toolbox.renderer.panda3d_scene_renderer import Panda3dSceneRenderer
from happypose.toolbox.renderer.types import (
    Panda3dCameraData,
    Panda3dLightData,
    Panda3dObjectData,
    CameraRenderingData
)


class TestPanda3DRenderer(unittest.TestCase):
    """Unit tests for Panda3D renderer."""


    def setUp(self) -> None:

        self.obj_label = "obj"
        self.obj_path = Path(__file__).parent.joinpath(f"data/obj_000001.ply")
        self.asset_dataset = RigidObjectDataset(
            objects=[
                RigidObject(
                    label=self.obj_label,
                    mesh_path=self.obj_path,
                    mesh_units="mm"
                ),
                RigidObject(
                    label='NOT_USED',
                    mesh_path=self.obj_path,
                    mesh_units="mm"
                )
            ]
        )
        
        self.z_obj = 0.3
        self.TWO = Transform((0.5, 0.5, -0.5, 0.5), (0, 0, self.z_obj))
        self.object_datas = [
            Panda3dObjectData(
                label=self.obj_label,
                TWO=self.TWO
            )
        ]

        fx, fy = 300, 300
        cx, cy = 320, 240
        self.K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ])
        
        self.TWC = Transform(np.eye(4))
        self.width, self.height = 640, 480
        self.Nc = 4
        self.camera_datas = self.Nc*[
            Panda3dCameraData(
                K=self.K,
                resolution=(self.height, self.width),
                TWC=self.TWC
            )
        ]

        Nb_lights = 3
        self.light_datas = Nb_lights*[
            Panda3dLightData(
                light_type="ambient",
                color=(1.0, 1.0, 1.0, 1.0)
            )
        ]

    def test_scene_renderer(self):
        """
        Scene render an example object and check that output image match expectation.
        """
        SAVEFIG = False

        renderer = Panda3dSceneRenderer(
            asset_dataset=self.asset_dataset
        )

        renderings: List[CameraRenderingData] = renderer.render_scene(self.object_datas, 
                                                                      self.camera_datas, 
                                                                      self.light_datas,
                                                                      render_normals=True, 
                                                                      render_depth=True, 
                                                                      render_binary_mask=True
                                                                      )

        self.assertEqual(len(renderings), self.Nc)
        # render from 2 identical cams are equals 
        self.assertIsNone(tr_assert_allclose(renderings[0].rgb, renderings[1].rgb))
        self.assertIsNone(tr_assert_allclose(renderings[0].normals, renderings[1].normals))
        self.assertIsNone(tr_assert_allclose(renderings[0].depth, renderings[1].depth))
        self.assertIsNone(tr_assert_allclose(renderings[0].binary_mask, renderings[1].binary_mask))
        
        rgb = renderings[0].rgb
        normals = renderings[0].normals
        depth = renderings[0].depth
        binary_mask = renderings[0].binary_mask

        self.assertEqual(rgb.shape, (self.height, self.width, 3))
        self.assertEqual(normals.shape, (self.height, self.width, 3))
        self.assertEqual(depth.shape, (self.height, self.width, 1))
        self.assertEqual(binary_mask.shape, (self.height, self.width, 1))

        self.assertEqual(rgb.dtype, np.dtype(np.uint8))
        self.assertEqual(normals.dtype, np.dtype(np.uint8))
        self.assertEqual(depth.dtype, np.dtype(np.float32))
        self.assertEqual(binary_mask.dtype, np.dtype(bool))

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
        self.assertIsNone(np_assert_equal(rgb[0, 0], (0, 0, 0)))
        self.assertIsNone(np_assert_array_less((0, 0, 0), rgb[self.height // 2, self.width // 2]))
        self.assertEqual(depth[0, 0], 0)
        self.assertLess(depth[self.height // 2, self.width // 2], self.z_obj) 
        self.assertIsNone(np_assert_equal(normals[0, 0], (0, 0, 0)))
        self.assertIsNone(np_assert_array_less((0, 0, 0), normals[self.height // 2, self.width // 2]))
        self.assertEqual(binary_mask[0,0], 0)     
        self.assertEqual(binary_mask[self.height // 2, self.width // 2], 1)    

        # ==================
        # Partial renderings
        # ==================
        renderings = renderer.render_scene(self.object_datas, 
                                           self.camera_datas, 
                                           self.light_datas,
                                           render_depth=True, 
                                           render_normals=True, 
                                           render_binary_mask=False
                                           )
        self.assertIsNotNone(renderings[0].rgb)
        self.assertIsNotNone(renderings[0].depth)
        self.assertIsNotNone(renderings[0].normals)
        self.assertIsNone(renderings[0].binary_mask)

        renderings = renderer.render_scene(self.object_datas, 
                                           self.camera_datas, 
                                           self.light_datas,
                                           render_depth=True, 
                                           render_normals=False, 
                                           render_binary_mask=False
                                           )
        self.assertIsNotNone(renderings[0].rgb)
        self.assertIsNotNone(renderings[0].depth)
        self.assertIsNone(renderings[0].normals)
        self.assertIsNone(renderings[0].binary_mask)

        renderings = renderer.render_scene(
            self.object_datas, 
            self.camera_datas, 
            self.light_datas,
            render_depth=False, 
            render_normals=True, 
            render_binary_mask=False
        )
        self.assertIsNotNone(renderings[0].rgb)
        self.assertIsNone(renderings[0].depth)
        self.assertIsNotNone(renderings[0].normals)
        self.assertIsNone(renderings[0].binary_mask)

        renderings = renderer.render_scene(
            self.object_datas, 
            self.camera_datas, 
            self.light_datas,
            render_depth=False, 
            render_normals=False, 
            render_binary_mask=False
        )
        self.assertIsNotNone(renderings[0].rgb)
        self.assertIsNone(renderings[0].depth)
        self.assertIsNone(renderings[0].normals)
        self.assertIsNone(renderings[0].binary_mask)

        # TODO: not sure how to check that assertion is raised without crashing the test
        # try:
            # renderings = renderer.render_scene(self.object_datas, 
            #                                    self.camera_datas, 
            #                                    self.light_datas,
            #                                    light_datas=self.Nc*[self.light_datas],
            #                                    resolution=(self.height, self.width),
            #                                    render_depth=False, 
            #                                    render_normals=False,
            #                                    render_binary_mask=True
            #                                    )
        # except AssertionError:
        #     assert True


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
        K = torch.from_numpy(self.K)
        TCO = TCO.unsqueeze(0)
        K = K.unsqueeze(0)
        TCO = TCO.repeat(self.Nc, 1, 1)
        K = K.repeat(self.Nc, 1, 1)

        # labels and light_datas need to have same size as TCO/K batch size
        renderings = renderer.render(
            labels=self.Nc*[self.obj_label], 
            TCO=TCO, 
            K=K, 
            light_datas=self.Nc*[self.light_datas],
            resolution=(self.height, self.width),
            render_normals=True,
            render_depth=True, 
            render_binary_mask=True
        )

        self.assertEqual(renderings.rgbs.shape, (self.Nc, 3, self.height, self.width))
        self.assertEqual(renderings.depths.shape, (self.Nc, 1, self.height, self.width))
        self.assertEqual(renderings.normals.shape, (self.Nc, 3, self.height, self.width))
        self.assertEqual(renderings.binary_masks.shape, (self.Nc, 1, self.height, self.width))

        self.assertEqual(renderings.rgbs.dtype, torch.float32)
        self.assertEqual(renderings.depths.dtype, torch.float32)
        self.assertEqual(renderings.normals.dtype, torch.float32)
        self.assertEqual(renderings.binary_masks.dtype, torch.bool)

        # Renders from 2 identical cams are equals 
        self.assertIsNone(tr_assert_allclose(renderings.rgbs[0], renderings.rgbs[1]))
        self.assertIsNone(tr_assert_allclose(renderings.normals[0], renderings.normals[1]))
        self.assertIsNone(tr_assert_allclose(renderings.depths[0], renderings.depths[1]))
        self.assertIsNone(tr_assert_allclose(renderings.binary_masks[0], renderings.binary_masks[1]))
        
        rgb = renderings.rgbs[0].movedim(0,-1).numpy()  # (Nc,3,h,w) -> (h,w,3)
        normals = renderings.normals[0].movedim(0,-1).numpy()  # (Nc,1,h,w) -> (h,w,1)
        depth = renderings.depths[0].movedim(0,-1).numpy()  # (Nc,3,h,w) -> (h,w,3)
        binary_mask = renderings.binary_masks[0].movedim(0,-1).numpy()  # (Nc,1,h,w) -> (h,w,1)



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
        self.assertIsNone(np_assert_equal(rgb[0, 0], (0, 0, 0)))
        self.assertIsNone(np_assert_array_less((0, 0, 0), rgb[self.height // 2, self.width // 2]))
        self.assertEqual(depth[0, 0], 0)
        self.assertLess(depth[self.height // 2, self.width // 2], self.z_obj) 
        self.assertIsNone(np_assert_equal(normals[0, 0], (0, 0, 0)))
        self.assertIsNone(np_assert_array_less((0, 0, 0), normals[self.height // 2, self.width // 2]))
        self.assertEqual(binary_mask[0,0], 0)     
        self.assertEqual(binary_mask[self.height // 2, self.width // 2], 1)     

        # ==================
        # Partial renderings
        # ==================
        renderings = renderer.render(
            self.Nc*[self.obj_label], 
            TCO, 
            K, 
            light_datas=self.Nc*[self.light_datas],
            resolution=(self.height, self.width),
            render_depth=True, 
            render_normals=True,
            render_binary_mask=False
        )
        self.assertIsNotNone(renderings.rgbs)
        self.assertIsNotNone(renderings.depths)
        self.assertIsNotNone(renderings.normals)
        self.assertIsNone(renderings.binary_masks)

        renderings = renderer.render(
            self.Nc*[self.obj_label], 
            TCO, 
            K, 
            light_datas=self.Nc*[self.light_datas],
            resolution=(self.height, self.width),
            render_depth=True, 
            render_normals=False,
            render_binary_mask=False
        )
        self.assertIsNotNone(renderings.rgbs)
        self.assertIsNotNone(renderings.depths)
        self.assertIsNone(renderings.normals)
        self.assertIsNone(renderings.binary_masks)

        renderings = renderer.render(
            self.Nc*[self.obj_label], 
            TCO, 
            K, 
            light_datas=self.Nc*[self.light_datas],
            resolution=(self.height, self.width),
            render_depth=False, 
            render_normals=True,
            render_binary_mask=False
        )
        self.assertIsNotNone(renderings.rgbs)
        self.assertIsNone(renderings.depths)
        self.assertIsNotNone(renderings.normals)
        self.assertIsNone(renderings.binary_masks)

        renderings = renderer.render(
            self.Nc*[self.obj_label], 
            TCO, 
            K, 
            light_datas=self.Nc*[self.light_datas],
            resolution=(self.height, self.width),
            render_depth=False, 
            render_normals=False,
            render_binary_mask=False
        )
        self.assertIsNotNone(renderings.rgbs)
        self.assertIsNone(renderings.depths)
        self.assertIsNone(renderings.normals)
        self.assertIsNone(renderings.binary_masks)

        # TODO: not sure how to check that assertion is raised without crashing the test
        # try:
        #     renderings = renderer.render(self.Nc*[self.obj_label], 
        #                     TCO, 
        #                     K, 
        #                     light_datas=self.Nc*[self.light_datas],
        #                     resolution=(self.height, self.width),
        #                     render_depth=False, 
        #                     render_normals=False,
        #                     render_binary_mask=True
        #                     )
        # except AssertionError:
        #     assert True


if __name__ == "__main__":
    unittest.main()
