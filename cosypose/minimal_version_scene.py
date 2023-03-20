
import os
import cv2
import pickle
import numpy as np
from matplotlib import pyplot as plt

# Standard Library
import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Union

# Third Party
import numpy as np
from bokeh.io import export_png
from bokeh.plotting import gridplot
from PIL import Image

########################
# Add cosypose to my path -> dirty
import sys
sys.path.insert(0, '/home/emaitre/cosypose')
########################

import cosypose

from cosypose.rendering.bullet_scene_renderer import BulletSceneRenderer
from cosypose.visualization.singleview import render_prediction_wrt_camera

from cosypose_wrapper import CosyPoseWrapper

# MegaPose
from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
from megapose.datasets.scene_dataset import CameraData, ObjectData
from megapose.inference.types import (
    DetectionsType,
    ObservationTensor,
    PoseEstimatesType,
)
from megapose.inference.utils import make_detections_from_object_data
from megapose.lib3d.transform import Transform
from megapose.panda3d_renderer import Panda3dLightData
from megapose.panda3d_renderer.panda3d_scene_renderer import Panda3dSceneRenderer
from megapose.utils.conversion import convert_scene_observation_to_panda3d
from megapose.utils.load_model import NAMED_MODELS, load_named_model
from megapose.utils.logging import get_logger, set_logging_level
from megapose.visualization.bokeh_plotter import BokehPlotter
from megapose.visualization.utils import make_contour_overlay
from megapose.datasets.datasets_cfg import make_object_dataset


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

path_data = Path("/home/emaitre/cosypose/local_data/bop_datasets/ycbv/test")

scene_number = Path("000057")

path_scene = path_data / scene_number

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

    print(type)
    print("preds = ", preds)
    print("poses =", preds.poses)
    print("poses_input =", preds.poses_input)
    print("k_crop =", preds.K_crop )
    print("boxes_rend =", preds.boxes_rend)
    print("boxes_crop =", preds.boxes_crop)


    ### Object dataset is necessary for panda3d renderer


    example_dir = Path("/home/emaitre/cosypose/local_data/bop_datasets/ycbv/examples/")

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


    cv2.imshow('raw img', brg)
    # Detected object
    #cv2.imshow('rgb_n_render', cv2.cvtColor(rgb_n_render, cv2.COLOR_RGB2BGR))
    # Blured background version
    cv2.imshow('rgb_overlay',  cv2.cvtColor(rgb_overlay, cv2.COLOR_RGB2BGR))
    cv2.waitKey(10000)
    cv2.destroyAllWindows()

