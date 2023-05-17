import os
import cv2
import numpy as np

# Standard Library
import argparse
import json
import os
from pathlib import Path
from typing import List

# Third Party
import numpy as np
from bokeh.io import export_png
from bokeh.plotting import gridplot
from PIL import Image

########################
# Add cosypose to my path -> dirty
import sys
########################

import cosypose
import torch

#from happypose.pose_estimators.cosypose.cosypose.rendering.bullet_scene_renderer import BulletSceneRenderer
from happypose.pose_estimators.cosypose.cosypose.visualization.singleview import render_prediction_wrt_camera
from happypose.pose_estimators.cosypose.cosypose.config import LOCAL_DATA_DIR

from happypose.pose_estimators.cosypose.cosypose.scripts.cosypose_wrapper import CosyPoseWrapper

# HappyPose
from happypose.toolbox.renderer import Panda3dLightData
from happypose.toolbox.renderer.panda3d_scene_renderer import Panda3dSceneRenderer

# MegaPose
from happypose.toolbox.datasets.object_dataset import RigidObject, RigidObjectDataset
from happypose.toolbox.datasets.scene_dataset import CameraData, ObjectData
from happypose.toolbox.inference.types import (
    DetectionsType,
    ObservationTensor,
    PoseEstimatesType,
)
from happypose.toolbox.lib3d.transform import Transform
from happypose.toolbox.utils.conversion import convert_scene_observation_to_panda3d
from happypose.toolbox.utils.load_model import NAMED_MODELS, load_named_model
from happypose.toolbox.utils.logging import get_logger, set_logging_level
from happypose.toolbox.visualization.bokeh_plotter import BokehPlotter
from happypose.toolbox.visualization.utils import make_contour_overlay

logger = get_logger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_object_dataset(example_dir: Path) -> RigidObjectDataset:
    rigid_objects = []
    mesh_units = "mm"
    object_dirs = (example_dir / "meshes").iterdir()
    for object_dir in object_dirs:
        label = object_dir.name
        mesh_path = None
        for fn in object_dir.glob("*"):
            if fn.suffix in {".obj", ".ply"}:
                assert not mesh_path, f"there multiple meshes in the {label} directory"
                mesh_path = fn
        assert mesh_path, f"couldnt find a obj or ply mesh for {label}"
        rigid_objects.append(RigidObject(label=label, mesh_path=mesh_path, mesh_units=mesh_units))
        # TODO: fix mesh units
    rigid_object_dataset = RigidObjectDataset(rigid_objects)
    return rigid_object_dataset


def load_image(data_path: Path) -> List[ObjectData]:
    # imread stores color dim in the BGR order by default
    brg = cv2.imread(str(data_path) + "/cheezit_Color.png")
    # CosyPose uses a RGB representation internally?
    rgb = cv2.cvtColor(brg, cv2.COLOR_BGR2RGB)
    IMG_RES = 480, 640
    return rgb, IMG_RES


def load_camera() -> np.array: 
    K_rs = np.array([615.1529541015625, 0.0, 324.5750732421875, 
    0.0, 615.2452392578125, 237.81765747070312, 
    0.0, 0.0, 1.0]).reshape((3,3))
    return K_rs


def rendering(predictions, K_rs, img_res):
    object_dataset = make_object_dataset(example_dir)
    # rendering
    renderer = Panda3dSceneRenderer(object_dataset)
    cam = {
        'resolution': img_res,
        'K': K_rs,
        'TWC': np.eye(4),
    }
    # Data necessary for image rendering
    object_datas = [ObjectData(label="cheetos", TWO=Transform(predictions.poses[0].numpy()))]
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
    return renderings


def save_predictions(example_dir, renderings, rgb):
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

    fig_mesh_overlay = plotter.plot_overlay(rgb, renderings.rgb)
    contour_overlay = make_contour_overlay(
        rgb, renderings.rgb, dilate_iterations=1, color=(0, 255, 0)
    )["img"]
    fig_contour_overlay = plotter.plot_image(contour_overlay)
    fig_all = gridplot([[fig_rgb, fig_contour_overlay, fig_mesh_overlay]], toolbar_location=None)
    vis_dir = example_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    export_png(fig_mesh_overlay, filename=vis_dir / "mesh_overlay.png")
    export_png(fig_contour_overlay, filename=vis_dir / "contour_overlay.png")
    export_png(fig_all, filename=vis_dir / "all_results.png")


def run_inference(
    example_dir: Path,
    model_name: str,
    dataset_to_use: str,
) -> None:
    rgb, img_res = load_image(example_dir)
    K_rs = load_camera()
    CosyPose = CosyPoseWrapper(dataset_name=dataset_to_use, n_workers=8)
    predictions = CosyPose.inference(rgb, K_rs)
    renderings = rendering(predictions, K_rs, img_res)
    save_predictions(example_dir, renderings, rgb)


if __name__ == "__main__":
    set_logging_level("info")
    parser = argparse.ArgumentParser()
    parser.add_argument("example_name")
    parser.add_argument("--model", type=str, default="megapose-1.0-RGB-multi-hypothesis")
    parser.add_argument("--dataset", type=str, default="ycbv")
    #parser.add_argument("--vis-detections", action="store_true")
    parser.add_argument("--run-inference", action="store_true", default=True)
    #parser.add_argument("--vis-outputs", action="store_true")
    args = parser.parse_args()

    data_dir = os.getenv("MEGAPOSE_DATA_DIR")
    assert data_dir
    example_dir = Path(data_dir) / "examples" / args.example_name
    dataset_to_use = args.dataset  # tless or ycbv

    #if args.vis_detections:
    #    make_detections_visualization(example_dir)

    if args.run_inference:
        run_inference(example_dir, args.model, dataset_to_use)

    #if args.vis_outputs:
    #    make_output_visualization(example_dir)