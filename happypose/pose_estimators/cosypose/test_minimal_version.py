# General imports

import numpy as np
import cv2
import time
from pathlib import Path
import argparse

import torch
from cosypose_wrapper import CosyPoseWrapper
import cosypose
from torch.testing import assert_close
from happypose.pose_estimators.cosypose.cosypose.config import LOCAL_DATA_DIR

# Functions to test

#from happypose.pose_estimators.cosypose.cosypose.rendering.bullet_scene_renderer import BulletSceneRenderer
#from happypose.pose_estimators.cosypose.cosypose.visualization.singleview import render_prediction_wrt_camera

# HappyPose
from happypose.toolbox.datasets.object_dataset import RigidObject, RigidObjectDataset
from happypose.toolbox.datasets.scene_dataset import CameraData, ObjectData

# MegaPose
from happypose.pose_estimators.megapose.src.megapose.inference.types import (
    ObservationTensor
)
from happypose.pose_estimators.megapose.src.megapose.inference.utils import make_detections_from_object_data
from happypose.pose_estimators.megapose.src.megapose.lib3d.transform import Transform
from happypose.pose_estimators.megapose.src.megapose.panda3d_renderer import Panda3dLightData
from happypose.pose_estimators.megapose.src.megapose.panda3d_renderer.panda3d_scene_renderer import Panda3dSceneRenderer
from happypose.pose_estimators.megapose.src.megapose.utils.conversion import convert_scene_observation_to_panda3d
from happypose.pose_estimators.megapose.src.megapose.utils.load_model import NAMED_MODELS, load_named_model
from happypose.pose_estimators.megapose.src.megapose.utils.logging import get_logger, set_logging_level
from happypose.pose_estimators.megapose.src.megapose.visualization.bokeh_plotter import BokehPlotter
from happypose.pose_estimators.megapose.src.megapose.visualization.utils import make_contour_overlay


##################################
##################################
##################################
#
#   Parameters
#
##################################
##################################
##################################

"""
parser = argparse.ArgumentParser()
parser.add_argument("-r", "--renderer", help="Chose which renderer to use",
                    type=str, options=['panda3d', 'pybullet'], required=True)
args = parser.parse_args()
"""
dataset_to_use = 'ycbv'  # tless or ycbv

# TODO : May need to be changed to 480, 640 for panda3d
IMG_RES = 480, 640 
# Realsense 453i intrinsics (from rostopic camera_info)
K_rs = np.array([615.1529541015625, 0.0, 324.5750732421875, 
    0.0, 615.2452392578125, 237.81765747070312, 
    0.0, 0.0, 1.0]).reshape((3,3))

img_dir = 'imgs'
image_name = 'cheezit_Color.png'

# imread stores color dim in the BGR order by default
brg = cv2.imread(img_dir + '/' + image_name)
# CosyPose uses a RGB representation internally?
rgb = cv2.cvtColor(brg, cv2.COLOR_BGR2RGB)

cam = {
    'resolution': IMG_RES,
    'K': K_rs,
    'TWC': np.eye(4),
}



##################################
##################################
##################################
#
#   CosyPose Wrapper
#   & Inference
#
##################################
##################################
##################################

cosypose_wrapper = CosyPoseWrapper(dataset_name=dataset_to_use, n_workers=8)

# Test the initialization of the detection and pose predictor models
def test_init_wrapper():
    assert(type(cosypose_wrapper.detector) == cosypose.integrated.detector.Detector)
    assert(type(cosypose_wrapper.pose_predictor) == cosypose.integrated.pose_predictor.CoarseRefinePosePredictor)
    assert(cosypose_wrapper.dataset_name == dataset_to_use)
    
# Test that the number of detected objects are the same, and the score is similar
# Test that the number of predictions and the inference time corresponds to the original one
def test_inference_integration():
    t = time.time()
    preds = cosypose_wrapper.inference(rgb, K_rs)
    
    # This test doesn't always pass
    assert(time.time() - t < 5)
    assert(len(preds) == 1)
    
# Test that the score and the bounding boxes are the same
def test_inference_detection():
    
    image = rgb
    images = torch.from_numpy(image).float().unsqueeze_(0)
    images = images.permute(0, 3, 1, 2) / 255
    box_detections = cosypose_wrapper.detector.get_detections(images=images, one_instance_per_class=False,
                                                      # detection_th=0.8, output_masks=False, mask_th=0.9)
                                                      detection_th=0.7, output_masks=False, mask_th=0.8)

    assert_close(box_detections.bboxes, torch.tensor([[218.9421, 121.9166, 415.8912, 381.7610]]))
    assert(box_detections.infos['score'][0] > 0.9999)
    
##########################################################
# Running the global inference to test the pose prediction
##########################################################

preds = cosypose_wrapper.inference(rgb, K_rs)

# Test that the output of the pose detection models are the same

# The results obtained here differ of the results obtained by running the scrip
# minimal_test.py. It is to be investigated to know why.
def test_inference(ndarrays_regression):
    dic_tensors = dict()
    dic_tensors['poses'] = preds.poses.detach().cpu().numpy()
    dic_tensors['poses_input'] = preds.poses_input.detach().cpu().numpy()
    dic_tensors['K_crop'] = preds.K_crop.detach().cpu().numpy()
    dic_tensors['boxes_rend'] = preds.boxes_rend.detach().cpu().numpy()
    dic_tensors['boxes_crop'] = preds.boxes_crop.detach().cpu().numpy()
    # Setting tolerance to arbitrary level
    #ndarrays_regression.check(dic_tensors)
    print(preds.poses)
      
    assert_close(preds.poses, 
                 torch.tensor([[[0.116784,  0.993015,  0.016808, -0.009588],
                            [ 0.189639, -0.005684, -0.981837,  0.012246],
                            [-0.974884,  0.117850, -0.188978,  0.523082],
                            [ 0.000000,  0.000000,  0.000000,  1.000000]]]))
    assert_close(preds.poses_input,
                 torch.tensor([[[ 0.154123,  0.987574,  0.030725, -0.009887],
                            [ 0.247309, -0.008452, -0.968900,  0.011040],
                            [-0.956600,  0.156928, -0.245539,  0.523046],
                            [ 0.000000,  0.000000,  0.000000,  1.000000]]]))
    assert_close(preds.K_crop,
                 torch.tensor([[[398.229767,   0.000000, 167.027328],
                            [  0.000000, 398.289490, 111.092880],
                            [  0.000000,   0.000000,   1.000000]]]))
    assert_close(preds.boxes_rend,
                 torch.tensor([[214.328354, 120.109589, 419.351959, 383.208740]]))
    assert_close(preds.boxes_crop,
                 torch.tensor([[ 65.792450,  65.438034, 560.102417, 436.170532]]))
    
    
##################################
##################################
##################################
#
#   Rendering
#
##################################
##################################
##################################

# TODO : Modify to use panda3d renderer
# If using panda3d, remember to modify IMG_RES as stated near its init
# renderer = BulletSceneRenderer('ycbv', gpu_renderer=False)
# rgb_render = render_prediction_wrt_camera(renderer, preds, cam)


def make_object_dataset(example_dir: Path) -> RigidObjectDataset:
    print(example_dir)
    rigid_objects = []
    mesh_units = "mm"
    object_dirs = (example_dir / "meshes").iterdir()
    print(object_dirs)
    for object_dir in object_dirs:
        print(object_dir)
        label = object_dir.name
        print(label)
        mesh_path = None
        for fn in object_dir.glob("*"):
            print("fn = ", fn)
            if fn.suffix in {".obj", ".ply"}:
                assert not mesh_path, f"there multiple meshes in the {label} directory"
                mesh_path = fn
        assert mesh_path, f"couldnt find a obj or ply mesh for {label}"
        rigid_objects.append(RigidObject(label=label, mesh_path=mesh_path, mesh_units=mesh_units))
        # TODO: fix mesh units
    rigid_object_dataset = RigidObjectDataset(rigid_objects)
    return rigid_object_dataset

example_dir = Path(str(LOCAL_DATA_DIR) + "/bop_datasets/ycbv/examples/")

object_dataset = make_object_dataset(Path(example_dir / "cheetos" ))

# rendering
renderer = Panda3dSceneRenderer(object_dataset)
cam = {
    'resolution': IMG_RES,
    'K': K_rs,
    'TWC': np.eye(4),
}

# Data necessary for image rendering

object_datas = [ObjectData(label="cheetos", TWO=Transform(preds.poses[0].numpy()))]
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


# Test if rgb rendered image returned is the same
def test_render_predicton_wrt_camera(ndarrays_regression):
    dic_render = {'render': rgb_render}
    ndarrays_regression.check(dic_render)
    

# Test of the final rendering
def test_rgb_overlay(ndarrays_regression):
    mask = ~(rgb_render.sum(axis=-1) == 0)
    alpha = 0.1
    rgb_n_render = rgb.copy()
    rgb_n_render[mask] = rgb_render[mask]
    # make the image background a bit fairer than the render
    rgb_overlay = np.zeros_like(rgb_render)
    rgb_overlay[~mask] = rgb[~mask] * 0.6 + 255 * 0.4
    rgb_overlay[mask] = rgb_render[mask] * 0.8 + 255 * 0.2
    dic_overlay = {'render': rgb_overlay}
    ndarrays_regression.check(dic_overlay)
