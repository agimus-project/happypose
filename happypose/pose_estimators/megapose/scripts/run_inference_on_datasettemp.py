# Standard Library
import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Union

# Third Party
import numpy as np
import torch
from bokeh.io import export_png
from bokeh.plotting import gridplot
from PIL import Image

# HappyPose
import happypose
from happypose.toolbox.datasets.object_dataset import RigidObject, RigidObjectDataset
from happypose.toolbox.datasets.scene_dataset import CameraData, ObjectData
from happypose.toolbox.inference.types import (
    DetectionsType,
    ObservationTensor,
    PoseEstimatesType,
)
from happypose.toolbox.inference.utils import make_detections_from_object_data
from happypose.toolbox.lib3d.transform import Transform
from happypose.toolbox.renderer import Panda3dLightData
from happypose.toolbox.renderer.panda3d_scene_renderer import Panda3dSceneRenderer
from happypose.toolbox.utils.conversion import convert_scene_observation_to_panda3d
from happypose.toolbox.utils.load_model import NAMED_MODELS, load_named_model
from happypose.toolbox.utils.logging import get_logger, set_logging_level
from happypose.toolbox.visualization.bokeh_plotter import BokehPlotter
from happypose.toolbox.visualization.utils import make_contour_overlay
from happypose.toolbox.datasets.datasets_cfg import make_scene_dataset, make_object_dataset

# MegaPose
#from happypose.toolbox.datasets.object_dataset import RigidObject, RigidObjectDataset
#from happypose.toolbox.datasets.scene_dataset import CameraData, ObjectData
from happypose.pose_estimators.megapose.config import (
    BOP_DS_DIR
)

#scene_id = str(object['scene_id']).zfill(6)
#image_id = str(object['image_id']).zfill(6)

logger = get_logger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_observation(
    dataset_dir: Path,
    scene_id: str,
    image_id: str,
    load_depth: bool = False,
) -> Tuple[np.ndarray, Union[None, np.ndarray], CameraData]:
    camera_data_json = json.loads((dataset_dir / "scene_camera.json").read_text())
    camera_data = CameraData(
        K=np.array(camera_data_json[str(image_id)]['cam_K']).reshape(3,3),
        resolution=(480, 640))
    rgb = np.array(Image.open(dataset_dir / "rgb/{image_id}.png".format(image_id=str(image_id).zfill(6))),
                   dtype=np.uint8)
    assert rgb.shape[:2] == camera_data.resolution

    depth = None
    if load_depth:
        depth = np.array(Image.open(dataset_dir / "depth/{image_id}.png".format(image_id=str(image_id).zfill(6))),
                         dtype=np.float32) / 1000
        assert depth.shape[:2] == camera_data.resolution

    return rgb, depth, camera_data


def load_observation_tensor(
    dataset_dir: Path,
    scene_id: str,
    image_id: str,
    load_depth: bool = False,
) -> ObservationTensor:
    dataset_dir = dataset_dir / "test" / str(scene_id).zfill(6)
    rgb, depth, camera_data = load_observation(dataset_dir, scene_id, image_id, load_depth)
    observation = ObservationTensor.from_numpy(rgb, depth, camera_data.K)
    return observation


def load_object_data(data_path: Path) -> List[ObjectData]:
    object_data = json.loads(data_path.read_text())
    for object in object_data:
        object['bbox_modal'] = object['bbox']
        object['label'] = "ycbv-obj_{}".format(str(object['category_id']).zfill(6))
        scene_id = object['scene_id']
        image_id = object['image_id']
        break
    for object in object_data:
        print("object_data = ", object)
        object_data = [ObjectData.from_json(object)]
        break
    return object_data, scene_id, image_id


def load_detections(
    example_dir: Path,
) -> DetectionsType:
    input_object_data, scene_id, image_id = load_object_data(example_dir / "baseline.json")
    detections = make_detections_from_object_data(input_object_data).to(device)
    print(detections)
    return detections, scene_id, image_id

"""
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
"""

def save_predictions(
    example_dir: Path,
    pose_estimates: PoseEstimatesType,
) -> None:
    labels = pose_estimates.infos["label"]
    poses = pose_estimates.poses.cpu().numpy()
    object_data = [
        ObjectData(label=label, TWO=Transform(pose)) for label, pose in zip(labels, poses)
    ]
    object_data_json = json.dumps([x.to_json() for x in object_data])
    output_fn = example_dir / "outputs" / "object_data.json"
    output_fn.parent.mkdir(exist_ok=True)
    output_fn.write_text(object_data_json)
    logger.info(f"Wrote predictions: {output_fn}")
    return


def run_inference(
    example_dir: Path,
    dataset_dir: Path,
    dataset_name: str,
    model_name: str,
) -> None:

    model_info = NAMED_MODELS[model_name]
    

    detections, scene_id, image_id = load_detections(example_dir)
    detections = detections.to(device)

    observation = load_observation_tensor(
        dataset_dir, scene_id, image_id, load_depth=model_info["requires_depth"]
    )
    if torch.cuda.is_available():
        observation.cuda()
    #object_dataset = make_object_dataset(dataset_dir)

    ds_kwargs = dict(load_depth=True)
    dataset_name = "ycbv.bop19"
    scene_ds = make_scene_dataset(dataset_name, **ds_kwargs)
    urdf_ds_name, obj_ds_name = happypose.toolbox.datasets.datasets_cfg.get_obj_ds_info(dataset_name)
    object_dataset = make_object_dataset(obj_ds_name)

    
    logger.info(f"Loading model {model_name}.")
    pose_estimator = load_named_model(model_name, object_dataset).to(device)
    logger.info(f"Running inference.")
    output, _ = pose_estimator.run_inference_pipeline(
        observation, detections=detections, **model_info["inference_parameters"]
    )
    
    save_predictions(example_dir, output)
    return
    


# def make_mesh_visualization(RigidObject) -> List[Image]:
#     return


# def make_scene_visualization(CameraData, List[ObjectData]) -> List[Image]:
#     return


# def run_inference(example_dir, use_depth: bool = False):
#     return


if __name__ == "__main__":
    set_logging_level("info")
    parser = argparse.ArgumentParser()
    parser.add_argument("ds_name")
    parser.add_argument("--model", type=str, default="megapose-1.0-RGB-multi-hypothesis")
    parser.add_argument("--vis-detections", action="store_true")
    parser.add_argument("--run-inference", action="store_true")
    parser.add_argument("--vis-outputs", action="store_true")
    args = parser.parse_args()

    data_dir = os.getenv("HAPPYPOSE_DATA_DIR")
    assert data_dir
    example_dir = Path(BOP_DS_DIR) / "bop23/baseline" / args.ds_name
    dataset_dir = Path(BOP_DS_DIR) / args.ds_name

    if args.run_inference:
        run_inference(example_dir, dataset_dir, args.ds_name, args.model)


