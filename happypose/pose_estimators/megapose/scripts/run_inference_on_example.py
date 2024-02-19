# Standard Library
import argparse
import os
from pathlib import Path

# Third Party
import torch

# HappyPose
from happypose.toolbox.datasets.object_dataset import RigidObjectDataset
from happypose.toolbox.inference.example_inference_utils import (
    load_detections,
    load_object_data,
    load_observation_example,
    make_detections_visualization,
    make_example_object_dataset,
    make_output_visualization,
    save_predictions,
)
from happypose.toolbox.inference.types import DetectionsType, ObservationTensor
from happypose.toolbox.utils.load_model import NAMED_MODELS, load_named_model
from happypose.toolbox.utils.logging import get_logger, set_logging_level

logger = get_logger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_inference(
    observation: ObservationTensor,
    detections: DetectionsType,
    object_dataset: RigidObjectDataset,
    model_name: str,
) -> None:
    model_info = NAMED_MODELS[model_name]

    if torch.cuda.is_available():
        observation.cuda()

    logger.info(f"Loading model {model_name}.")
    pose_estimator = load_named_model(model_name, object_dataset).to(device)
    logger.info("Running inference.")
    output, _ = pose_estimator.run_inference_pipeline(
        observation,
        detections=detections,
        **model_info["inference_parameters"],
    )

    return output


if __name__ == "__main__":
    set_logging_level("info")
    parser = argparse.ArgumentParser()
    parser.add_argument("example_name")
    parser.add_argument(
        "--model",
        type=str,
        default="megapose-1.0-RGB-multi-hypothesis",
    )
    parser.add_argument("--vis-detections", action="store_true")
    parser.add_argument("--run-inference", action="store_true")
    parser.add_argument("--vis-outputs", action="store_true")
    args = parser.parse_args()

    data_dir = os.getenv("HAPPYPOSE_DATA_DIR")
    assert data_dir
    example_dir = Path(data_dir) / "examples" / args.example_name

    # Load data
    detections = load_detections(example_dir).to(device)
    object_dataset = make_example_object_dataset(example_dir)
    rgb, depth, camera_data = load_observation_example(example_dir, load_depth=True)
    observation = ObservationTensor.from_numpy(rgb, depth, camera_data.K)
    object_datas = load_object_data(example_dir / "outputs" / "object_data.json")

    if args.vis_detections:
        make_detections_visualization(rgb, detections, example_dir)

    if args.run_inference:
        output = run_inference(observation, detections, object_dataset, args.model)
        save_predictions(output, example_dir)

    if args.vis_outputs:
        make_output_visualization(
            rgb, object_dataset, object_datas, camera_data, example_dir
        )
