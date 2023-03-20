# General imports

import numpy as np
import cv2
import time
from pathlib import Path

import torch
from cosypose_wrapper import CosyPoseWrapper
import cosypose
from torch.testing import assert_close

# Functions to test

from cosypose.rendering.bullet_scene_renderer import BulletSceneRenderer
from cosypose.visualization.singleview import render_prediction_wrt_camera



##################################
##################################
##################################
#
#   Parameters
#
##################################
##################################
##################################

dataset_to_use = 'ycbv'  # tless or ycbv

# TODO : May need to be changed to 480, 640 for panda3d
IMG_RES = 640, 480 
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
    ndarrays_regression.check(dic_tensors, default_tolerance=dict(atol=1e-8, rtol=1e-8))  
    
    
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
renderer = BulletSceneRenderer('ycbv', gpu_renderer=False)
rgb_render = render_prediction_wrt_camera(renderer, preds, cam)


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
