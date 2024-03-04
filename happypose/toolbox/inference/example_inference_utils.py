import json
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
from bokeh.io import export_png
from bokeh.plotting import gridplot
from PIL import Image

from happypose.toolbox.datasets.object_dataset import RigidObject, RigidObjectDataset
from happypose.toolbox.datasets.scene_dataset import CameraData, ObjectData
from happypose.toolbox.inference.types import DetectionsType, PoseEstimatesType
from happypose.toolbox.inference.utils import make_detections_from_object_data
from happypose.toolbox.lib3d.transform import Transform
from happypose.toolbox.renderer import Panda3dLightData
from happypose.toolbox.renderer.panda3d_scene_renderer import Panda3dSceneRenderer
from happypose.toolbox.utils.conversion import convert_scene_observation_to_panda3d
from happypose.toolbox.utils.logging import get_logger
from happypose.toolbox.visualization.bokeh_plotter import BokehPlotter
from happypose.toolbox.visualization.utils import make_contour_overlay

logger = get_logger(__name__)


def make_example_object_dataset(
    example_dir: Path, mesh_units="mm"
) -> RigidObjectDataset:
    """
    TODO
    """

    rigid_objects = []
    mesh_units = "mm"
    mesh_dir = example_dir / "meshes"
    assert mesh_dir.exists(), f"Missing mesh directory {mesh_dir}"

    for mesh_path in mesh_dir.iterdir():
        if mesh_path.suffix in {".obj", ".ply"}:
            obj_name = mesh_path.with_suffix("").name
            rigid_objects.append(
                RigidObject(label=obj_name, mesh_path=mesh_path, mesh_units=mesh_units),
            )
    rigid_object_dataset = RigidObjectDataset(rigid_objects)
    return rigid_object_dataset


def load_observation_example(
    example_dir: Path,
    load_depth: bool = False,
    camera_data_name: str = "camera_data.json",
    rgb_name: str = "image_rgb.png",
    depth_name: str = "image_depth.png",
) -> Tuple[np.ndarray, Union[None, np.ndarray], CameraData]:
    camera_data = CameraData.from_json((example_dir / camera_data_name).read_text())

    rgb = np.array(Image.open(example_dir / rgb_name), dtype=np.uint8)
    assert rgb.shape[:2] == camera_data.resolution

    depth = None
    if load_depth:
        depth = np.array(Image.open(example_dir / depth_name), dtype=np.float32) / 1000
        assert depth.shape[:2] == camera_data.resolution

    return rgb, depth, camera_data


def load_detections(
    example_dir: Path,
) -> DetectionsType:
    input_object_data = load_object_data(example_dir / "object_data.json")
    detections = make_detections_from_object_data(input_object_data)
    return detections


def load_object_data(data_path: Path) -> List[ObjectData]:
    """"""
    object_data = json.loads(data_path.read_text())
    object_data = [ObjectData.from_json(d) for d in object_data]
    return object_data


def make_detections_visualization(
    rgb: np.ndarray,
    detections: DetectionsType,
    example_dir: Path,
) -> None:
    plotter = BokehPlotter()

    # TODO: put in BokehPlotter.plot_detections
    if hasattr(detections, "masks"):
        for mask in detections.masks:
            mask = mask.unsqueeze(2).tile((1, 1, 3)).numpy()
            rgb[mask] = 122

    fig_rgb = plotter.plot_image(rgb)
    fig_det = plotter.plot_detections(fig_rgb, detections=detections)
    output_fn = example_dir / "visualizations" / "detections.png"
    output_fn.parent.mkdir(exist_ok=True)
    export_png(fig_det, filename=output_fn)

    logger.info(f"Wrote detections visualization: {output_fn}")
    return


def save_predictions(
    pose_estimates: PoseEstimatesType,
    example_dir: Path,
) -> None:
    labels = pose_estimates.infos["label"]
    poses = pose_estimates.poses.cpu().numpy()
    object_data = [
        ObjectData(label=label, TWO=Transform(pose))
        for label, pose in zip(labels, poses)
    ]
    object_data_json = json.dumps([x.to_json() for x in object_data])
    output_fn = example_dir / "outputs" / "object_data_inf.json"
    output_fn.parent.mkdir(exist_ok=True)
    output_fn.write_text(object_data_json)
    logger.info(f"Wrote predictions: {output_fn}")


def make_poses_visualization(
    rgb: np.ndarray,
    object_dataset: RigidObjectDataset,
    object_datas: List[ObjectData],
    camera_data: CameraData,
    example_dir: Path,
) -> None:
    camera_data.TWC = Transform(np.eye(4))

    renderer = Panda3dSceneRenderer(object_dataset)

    camera_data, object_datas = convert_scene_observation_to_panda3d(
        camera_data,
        object_datas,
    )
    light_datas = [
        Panda3dLightData(
            light_type="ambient",
            color=((1.0, 1.0, 1.0, 1)),
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

    plotter = BokehPlotter()

    fig_rgb = plotter.plot_image(rgb)
    fig_mesh_overlay = plotter.plot_overlay(rgb, renderings.rgb)
    contour_overlay = make_contour_overlay(
        rgb,
        renderings.rgb,
        dilate_iterations=1,
        color=(0, 255, 0),
    )["img"]
    fig_contour_overlay = plotter.plot_image(contour_overlay)
    fig_all = gridplot(
        [[fig_rgb, fig_contour_overlay, fig_mesh_overlay]],
        toolbar_location=None,
    )
    vis_dir = example_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    export_png(fig_mesh_overlay, filename=vis_dir / "mesh_overlay.png")
    export_png(fig_contour_overlay, filename=vis_dir / "contour_overlay.png")
    export_png(fig_all, filename=vis_dir / "all_results.png")
    logger.info(f"Wrote visualizations to {vis_dir}.")
    return
