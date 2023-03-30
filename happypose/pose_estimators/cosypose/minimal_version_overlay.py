
import cv2
import numpy as np

# Standard Library
import json
from pathlib import Path

# Third Party
import numpy as np
from bokeh.io import export_png
from bokeh.plotting import gridplot

########################
# Add cosypose to my path -> dirty
import sys
########################

from happypose.pose_estimators.cosypose.cosypose.rendering.bullet_scene_renderer import BulletSceneRenderer
from happypose.pose_estimators.cosypose.cosypose.visualization.singleview import render_prediction_wrt_camera
from happypose.pose_estimators.cosypose.cosypose.config import LOCAL_DATA_DIR

from cosypose_wrapper import CosyPoseWrapper

# MegaPose
from happypose.toolbox.datasets.object_dataset import RigidObject, RigidObjectDataset
from happypose.toolbox.datasets.scene_dataset import CameraData, ObjectData
from happypose.pose_estimators.megapose.src.megapose.inference.types import (
    DetectionsType,
    ObservationTensor,
    PoseEstimatesType,
)
from happypose.pose_estimators.megapose.src.megapose.inference.utils import make_detections_from_object_data
from happypose.pose_estimators.megapose.src.megapose.lib3d.transform import Transform
from happypose.toolbox.renderer import Panda3dLightData
from happypose.toolbox.renderer.panda3d_scene_renderer import Panda3dSceneRenderer
from happypose.pose_estimators.megapose.src.megapose.utils.conversion import convert_scene_observation_to_panda3d
from happypose.pose_estimators.megapose.src.megapose.utils.load_model import NAMED_MODELS, load_named_model
from happypose.pose_estimators.megapose.src.megapose.utils.logging import get_logger, set_logging_level
from happypose.pose_estimators.megapose.src.megapose.visualization.bokeh_plotter import BokehPlotter
from happypose.pose_estimators.megapose.src.megapose.visualization.utils import make_contour_overlay
from happypose.pose_estimators.megapose.src.megapose.datasets.datasets_cfg import make_object_dataset


import glob
import json
import time

dataset_to_use = 'ycbv'  # tless or ycbv

IMG_RES = 480, 640
# Realsense 453i intrinsics (from rostopic camera_info)
"""

K_rs = np.array([615.1529541015625, 0.0, 324.5750732421875, 
    0.0, 615.2452392578125, 237.81765747070312, 
    0.0, 0.0, 1.0]).reshape((3,3))

img_dir = 'imgs'
# image_name = 'all_Color.png'
# image_name = 'all_far_Color.png'
# image_name = 'banana_Color.png'
image_name = 'cheezit_Color.png'
# image_name = 'wood_block_Color.png'


# imread stores color dim in the BGR order by default
brg = cv2.imread(img_dir + '/' + image_name)
# CosyPose uses a RGB representation internally?
rgb = cv2.cvtColor(brg, cv2.COLOR_BGR2RGB)
"""


sub_dir = "/bop_datasets/ycbv/test"
path_data = Path(str(LOCAL_DATA_DIR) + sub_dir)

scenes = glob.glob(str(path_data) + "/*")


for path_scene in scenes:
    scene_number = Path(path_scene.split('/')[-1])
    if path_scene.split('/')[-1] == "000057" or path_scene.split('/')[-1] == "000053":
        continue
    path_scene = Path(path_scene)
    path_images = path_scene / "rgb"


    cosy_pose = CosyPoseWrapper(dataset_name=dataset_to_use, n_workers=8)


    pictures = glob.glob(str(path_images) + "/*.png")

    with open(path_scene / "scene_camera.json", 'r') as f:
        cameras_scene = json.loads(f.read())
            
    for index, pic in enumerate(pictures):
        cam_pic = cameras_scene['{}'.format(index+1)]
        K_rs = np.array(cam_pic['cam_K']).reshape((3,3))
        brg = cv2.imread(pic)
        rgb = cv2.cvtColor(brg, cv2.COLOR_BGR2RGB)


        t = time.time()
        preds = cosy_pose.inference(rgb, K_rs)
        print('\nInference time (sec): ', time.time() - t)
        print('Number of detections: ', len(preds))

        labels = preds.infos.label.to_list()

        ### Object dataset is necessary for panda3d renderer


        output_dir = Path(str(LOCAL_DATA_DIR) + "/bop_datasets/ycbv/results/")

        object_dataset = make_object_dataset('ycbv')

        # rendering
        renderer = Panda3dSceneRenderer(object_dataset)
        cam = {
            'resolution': IMG_RES,
            'K': K_rs,
            'TWC': np.eye(4),
        }

        object_datas = []
        # Data necessary for image rendering
        for idx, pred in enumerate(preds):
            object = ObjectData(label=labels[idx], TWO=Transform(preds.poses[idx].numpy()))
            object_datas.append(object)
        camera_data = CameraData(K=K_rs, resolution=cam["resolution"], TWC=Transform(cam["TWC"]))



        camera_data, object_datas = convert_scene_observation_to_panda3d(camera_data, object_datas)
        light_datas = [
            Panda3dLightData(
                light_type="ambient",
                color=((0.6, 0.6, 0.6, 1)),
            ),
        ]
        renderings = renderer.render_scene(
            object_datas,
            [camera_data],
            light_datas,
            render_depth=False,
            render_binary_mask=False,
            render_normals=False,
            copy_arrays=True,
        )[0]

        rgb_render = renderings.rgb


        # render_prediction_wrt_camera calls BulletSceneRenderer.render_scene using only one camera at pose Identity and return only rgb values
        # BulletSceneRenderer.render_scene: gets a "object list" (prediction like object), a list of camera infos (with Km pose, res) and renders
        # a "camera observation" for each camera/viewpoint
        # Actually, renders: rgb, mask, depth, near, far
        #rgb_render = render_prediction_wrt_camera(renderer, preds, cam)
        mask = ~(rgb_render.sum(axis=-1) == 0)

        alpha = 0.1

        rgb_n_render = rgb.copy()
        rgb_n_render[mask] = rgb_render[mask]

        # make the image background a bit fairer than the render
        rgb_overlay = np.zeros_like(rgb_render)
        rgb_overlay[~mask] = rgb[~mask] * 0.6 + 255 * 0.4
        rgb_overlay[mask] = rgb_render[mask] * 0.8 + 255 * 0.2


        plotter = BokehPlotter()

        fig_rgb = plotter.plot_image(rgb)

        print("writing", scene_number, index)
        fig_mesh_overlay = plotter.plot_overlay(rgb, renderings.rgb)
        contour_overlay = make_contour_overlay(
            rgb, renderings.rgb, dilate_iterations=1, color=(0, 255, 0)
        )["img"]
        fig_contour_overlay = plotter.plot_image(contour_overlay)
        fig_all = gridplot([[fig_rgb, fig_contour_overlay, fig_mesh_overlay]], toolbar_location=None)
        vis_dir = output_dir / scene_number
        vis_dir.mkdir(exist_ok=True)
        export_png(fig_mesh_overlay, filename=vis_dir / "mesh_overlay_{}.png".format(index+1))
        export_png(fig_contour_overlay, filename=vis_dir / "contour_overlay_{}.png".format(index+1))
        export_png(fig_all, filename=vis_dir / "all_results_{}.png".format(index+1))