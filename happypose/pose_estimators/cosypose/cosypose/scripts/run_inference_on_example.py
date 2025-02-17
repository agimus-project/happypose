# Standard Library
import argparse
import os
import time
from pathlib import Path

# Third Party
import torch

# CosyPose
from happypose.pose_estimators.cosypose.cosypose.utils.cosypose_wrapper import (
    CosyPoseWrapper,
)

# HappyPose
from happypose.toolbox.inference.example_inference_utils import (
    load_detections,
    load_object_data,
    load_observation_example,
    make_detections_visualization,
    make_example_object_dataset,
    make_poses_visualization,
    save_predictions,
)
from happypose.toolbox.inference.types import ObservationTensor
from happypose.toolbox.inference.utils import filter_detections
from happypose.toolbox.utils.logging import get_logger, set_logging_level

logger = get_logger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    set_logging_level("info")
    parser = argparse.ArgumentParser()
    parser.add_argument("example_name")
    parser.add_argument("--dataset", type=str, default="hope")
    parser.add_argument("--run-detections", action="store_true")
    parser.add_argument("--run-inference", action="store_true")
    parser.add_argument("--run-depth-refiner", action="store_true")
    parser.add_argument("--depth-refiner-type", type=str, default="icp")
    parser.add_argument("--vis-detections", action="store_true")
    parser.add_argument("--vis-poses", action="store_true")
    args = parser.parse_args()

    data_dir = os.getenv("HAPPYPOSE_DATA_DIR")
    assert data_dir, "Set HAPPYPOSE_DATA_DIR env variable"
    example_dir = Path(data_dir) / "examples" / args.example_name
    assert example_dir.exists(), (
        "Example {args.example_name} not available, follow download instructions"
    )
    # dataset_to_use = args.dataset  # hope/tless/ycbv

    # Load data
    detections = load_detections(example_dir).to(device)
    object_dataset = make_example_object_dataset(example_dir)
    rgb, depth, camera_data = load_observation_example(example_dir, load_depth=True)
    observation = ObservationTensor.from_numpy(rgb, depth, camera_data.K).to(device)

    # Load models
    cosy = CosyPoseWrapper(
        dataset_name=args.dataset,
        object_dataset=object_dataset,
        depth_refiner_type=args.depth_refiner_type,
        n_workers=1,
    )

    if args.run_detections:
        # Masks are not used for pose prediction, but are computed by Mask-RCNN anyway
        detections = cosy.detector.get_detections(observation, output_masks=True)
    else:
        detections = load_detections(example_dir).to(device)
    available_labels = [obj.label for obj in object_dataset.list_objects]
    detections = filter_detections(detections, available_labels)

    if args.run_inference:
        data_TCO, extra_data = cosy.pose_predictor.run_inference_pipeline(
            observation=observation,
            detections=detections,
            run_detector=False,
            n_refiner_iterations=3,
        )
        print("run_inference_pipeline timings:")
        print(extra_data["timing_str"])
        if args.run_depth_refiner:
            t1 = time.perf_counter()
            data_TCO, _ = cosy.depth_refiner.refine_poses(
                predictions=data_TCO, depth=observation.depth, K=observation.K
            )
            print(f"Depth refiner took: {time.perf_counter() - t1}")

        save_predictions(data_TCO.cpu(), example_dir)

    if args.vis_detections:
        make_detections_visualization(rgb, detections, example_dir)

    if args.vis_poses:
        if args.run_inference:
            out_filename = "object_data_inf.json"
        else:
            out_filename = "object_data.json"
        object_datas = load_object_data(example_dir / "outputs" / out_filename)
        make_poses_visualization(
            rgb, object_dataset, object_datas, camera_data, example_dir
        )


if __name__ == "__main__":
    main()
