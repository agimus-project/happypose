"""Set of unit tests for testing inference example for CosyPose."""
import unittest

import numpy as np
import pinocchio as pin

from happypose.pose_estimators.cosypose.cosypose.config import EXP_DIR, LOCAL_DATA_DIR
from happypose.pose_estimators.cosypose.cosypose.integrated.pose_estimator import (
    PoseEstimator,
)
from happypose.pose_estimators.cosypose.cosypose.training.pose_models_cfg import (
    load_model_cosypose,
)
from happypose.toolbox.datasets.bop_object_datasets import (
    RigidObject,
    RigidObjectDataset,
)
from happypose.toolbox.inference.example_inference_utils import load_observation_example
from happypose.toolbox.inference.types import ObservationTensor
from happypose.toolbox.inference.utils import load_detector
from happypose.toolbox.lib3d.rigid_mesh_database import MeshDataBase
from happypose.toolbox.renderer.panda3d_batch_renderer import Panda3dBatchRenderer


class TestCosyPoseInference(unittest.TestCase):
    """Unit tests for CosyPose inference example."""

    def test_cosypose_pipeline(self):
        """Run detector with coarse and refiner from CosyPose."""
        expected_object_label = "hope-obj_000002"
        mesh_file_name = "hope-obj_000002.ply"
        data_dir = LOCAL_DATA_DIR / "examples" / "barbecue-sauce"
        mesh_dir = data_dir / "meshes"
        mesh_path = mesh_dir / mesh_file_name
        device = "cpu"
        n_workers = 1

        rgb, depth, camera_data = load_observation_example(data_dir, load_depth=True)
        # TODO: cosypose forward does not work if depth is loaded detection contrary to megapose
        observation = ObservationTensor.from_numpy(rgb, depth=None, K=camera_data.K)

        detector = load_detector(run_id="detector-bop-hope-pbr--15246", device=device)
        # detections = detector.get_detections(observation=observation)

        object_dataset = RigidObjectDataset(
            objects=[
                RigidObject(
                    label=expected_object_label, mesh_path=mesh_path, mesh_units="mm"
                )
            ]
        )

        renderer = Panda3dBatchRenderer(
            object_dataset,
            n_workers=n_workers,
            preload_cache=False,
        )

        coarse_run_id = "coarse-bop-hope-pbr--225203"
        refiner_run_id = "refiner-bop-hope-pbr--955392"

        mesh_db = MeshDataBase.from_object_ds(object_dataset)
        mesh_db_batched = mesh_db.batched().to("cpu")
        coarse_model = load_model_cosypose(
            EXP_DIR / coarse_run_id, renderer, mesh_db_batched, device
        )
        refiner_model = load_model_cosypose(
            EXP_DIR / refiner_run_id, renderer, mesh_db_batched, device
        )

        pose_estimator = PoseEstimator(
            refiner_model=refiner_model,
            coarse_model=coarse_model,
            detector_model=detector,
        )

        # Run detector and pose estimator filtering object labels
        preds, _ = pose_estimator.run_inference_pipeline(
            observation=observation,
            detection_th=0.8,
            run_detector=True,
            n_refiner_iterations=3,
            labels_to_keep=[expected_object_label],
        )

        self.assertEqual(len(preds), 1)
        self.assertEqual(preds.infos.label[0], expected_object_label)

        pose = pin.SE3(preds.poses[0].numpy())
        exp_pose = pin.SE3(
            pin.exp3(np.array([1.4, 1.6, -1.11])),
            np.array([0.1, 0.07, 0.45]),
        )
        diff = pose.inverse() * exp_pose
        self.assertLess(np.linalg.norm(pin.log6(diff).vector), 0.3)


if __name__ == "__main__":
    unittest.main()
