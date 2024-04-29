"""Set of unit tests for testing inference example for MegaPose."""

import numpy as np
import pinocchio as pin
import pytest

from happypose.pose_estimators.cosypose.cosypose.config import LOCAL_DATA_DIR
from happypose.toolbox.datasets.bop_object_datasets import (
    RigidObject,
    RigidObjectDataset,
)
from happypose.toolbox.inference.example_inference_utils import load_observation_example
from happypose.toolbox.inference.types import ObservationTensor
from happypose.toolbox.inference.utils import load_detector
from happypose.toolbox.utils.load_model import NAMED_MODELS, load_named_model

from .config.test_config import DEVICE


class TestMegaPoseInference:
    """Unit tests for MegaPose inference example."""

    @pytest.mark.order(1)
    @pytest.mark.parametrize("device", DEVICE)
    def test_megapose_pipeline(self, device):
        """Run detector from with coarse and refiner from MegaPose."""

        expected_object_label = "hope-obj_000002"
        mesh_file_name = "hope-obj_000002.ply"
        data_dir = LOCAL_DATA_DIR / "examples" / "barbecue-sauce"
        mesh_dir = data_dir / "meshes"
        mesh_path = mesh_dir / mesh_file_name

        rgb, depth, camera_data = load_observation_example(data_dir, load_depth=True)
        # TODO: cosypose forward does not work if depth is loaded detection contrary to megapose
        if device == "cpu":
            observation = ObservationTensor.from_numpy(
                rgb, depth=None, K=camera_data.K
            ).cpu()
        else:
            observation = ObservationTensor.from_numpy(
                rgb, depth=None, K=camera_data.K
            ).cuda()

        detector = load_detector(run_id="detector-bop-hope-pbr--15246", device=device)
        object_dataset = RigidObjectDataset(
            objects=[
                RigidObject(
                    label=expected_object_label, mesh_path=mesh_path, mesh_units="mm"
                )
            ]
        )

        model_name = "megapose-1.0-RGB"
        model_info = NAMED_MODELS[model_name]
        pose_estimator = load_named_model(model_name, object_dataset)
        # Uniformely sumbsample the grid to increase speed
        pose_estimator._SO3_grid = pose_estimator._SO3_grid[::8]
        pose_estimator.detector_model = detector

        # Run detector and pose estimator filtering object labels
        preds, data = pose_estimator.run_inference_pipeline(
            observation,
            run_detector=True,
            **model_info["inference_parameters"],
            labels_to_keep=[expected_object_label],
        )

        scores = data["coarse"]["data"]["logits"]
        assert scores[0, 0] > scores[0, 1]

        assert len(preds) == 1
        assert preds.infos.label[0] == expected_object_label

        if device == "cpu":
            pose = pin.SE3(preds.poses[0].numpy())
        else:
            pose = pin.SE3(preds.poses[0].cpu().numpy())

        exp_pose = pin.SE3(
            pin.exp3(np.array([1.4, 1.6, -1.11])),
            np.array([0.1, 0.07, 0.45]),
        )
        diff = pose.inverse() * exp_pose
        assert np.linalg.norm(pin.log6(diff).vector) < 0.3
