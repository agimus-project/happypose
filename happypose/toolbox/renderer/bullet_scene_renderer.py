from typing import Dict, List, Union

import numpy as np
import pybullet as pb

from happypose.pose_estimators.cosypose.cosypose.config import LOCAL_DATA_DIR

# from happypose.toolbox.datasets.datasets_cfg import UrdfDataset
from happypose.pose_estimators.cosypose.cosypose.datasets.urdf_dataset import (
    UrdfDataset,
)

# TODO: move urdf utilities to happypose toolbox
from happypose.pose_estimators.cosypose.cosypose.libmesh.urdf_utils import (
    convert_rigid_body_dataset_to_urdfs,
)
from happypose.pose_estimators.cosypose.cosypose.simulator.base_scene import BaseScene
from happypose.pose_estimators.cosypose.cosypose.simulator.caching import BodyCache
from happypose.pose_estimators.cosypose.cosypose.simulator.camera import Camera
from happypose.toolbox.datasets.object_dataset import RigidObjectDataset
from happypose.toolbox.lib3d.transform import Transform
from happypose.toolbox.renderer.types import CameraRenderingData


class BulletSceneRenderer(BaseScene):
    def __init__(
        self,
        asset_dataset: Union[RigidObjectDataset,UrdfDataset],
        preload_cache=False,
        background_color=(0, 0, 0),
        gpu_renderer=True,
        gui=False,
    ):
        if isinstance(asset_dataset, UrdfDataset):
            self.urdf_ds = asset_dataset
        elif isinstance(asset_dataset, RigidObjectDataset):
            # Build urdfs files from RigidObjectDataset
            ds_name = 'tmp'
            urdf_dir = LOCAL_DATA_DIR / "urdfs" / ds_name
            convert_rigid_body_dataset_to_urdfs(asset_dataset, urdf_dir)
            self.urdf_ds = UrdfDataset(urdf_dir)
            # TODO: BodyCache assumes unique scale for all objects (true for bop datasets)
            self.urdf_ds.index["scale"] = asset_dataset[0].scale
        else:
            raise TypeError(f'asset_dataset of type {type(asset_dataset)} should be either UrdfDataset or RigidObjectDataset' )
        self.connect(gpu_renderer=gpu_renderer, gui=gui)
        self.body_cache = BodyCache(self.urdf_ds, self.client_id)
        if preload_cache:
            self.body_cache.get_bodies_by_ids(np.arange(len(self.urdf_ds)))
        self.background_color = background_color

    def setup_scene(self, obj_infos):
        labels = [obj["name"] for obj in obj_infos]
        bodies = self.body_cache.get_bodies_by_labels(labels)
        for obj_info, body in zip(obj_infos, bodies):
            body.pose = Transform(obj_info["TWO"])
            color = obj_info.get("color", None)
            if color is not None:
                pb.changeVisualShape(
                    body.body_id,
                    -1,
                    physicsClientId=0,
                    rgbaColor=color,
                )
        return bodies

    def render_images(self, 
                      cam_infos: Dict, 
                      render_depth: bool=False,
                      render_binary_mask: bool=False
                      ) -> List[CameraRenderingData]:
        cam_renderings = []
        for cam_info in cam_infos:
            
            K = cam_info["K"]
            TWC = Transform(cam_info["TWC"])
            resolution = cam_info["resolution"]
            cam = Camera(resolution=resolution, client_id=self.client_id)
            cam.set_intrinsic_K(K)
            cam.set_extrinsic_T(TWC)
            cam_obs_ = cam.get_state()
            rgb = cam_obs_["rgb"]
            mask = cam_obs_["mask"]

            background_indices = np.logical_or(mask < 0, mask == 255)

            if self.background_color is not None:
                rgb[background_indices] = self.background_color

            rendering = CameraRenderingData(rgb=rgb)

            if render_binary_mask:
                binary_mask = np.ones(rgb.shape[:2], dtype=bool)  # (h,w)
                binary_mask[background_indices] = False
                rendering.binary_mask = binary_mask[..., np.newaxis]  # (h,w,1)

            if render_depth:
                depth = cam_obs_["depth"]  # (h,w)
                near, far = cam_obs_["near"], cam_obs_["far"]
                z_n = 2 * depth - 1
                z_e = 2 * near * far / (far + near - z_n * (far - near))
                z_e[background_indices] = 0.0
                rendering.depth = z_e[..., np.newaxis]  # (h,w,1)

            cam_renderings.append(rendering)
            
        return cam_renderings

    def render_scene(self, 
                     obj_infos: Dict, 
                     cam_infos: Dict, 
                     render_depth=False,
                     render_binary_mask=False
                     ) -> List[CameraRenderingData]:
        self.setup_scene(obj_infos)
        return self.render_images(cam_infos, render_depth=render_depth, render_binary_mask=render_binary_mask)
