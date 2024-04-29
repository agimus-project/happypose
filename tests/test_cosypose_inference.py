"""Set of unit tests for testing inference example for CosyPose."""

import numpy as np
import pinocchio as pin
import pytest

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

from .config.test_config import DEVICE


class TestCosyPoseInference:
    """Unit tests for CosyPose inference example."""

    @pytest.fixture(autouse=True)
    def setUp(self) -> None:
        """Run detector with coarse and refiner from CosyPose."""
        self.expected_object_label = "hope-obj_000002"
        mesh_file_name = "hope-obj_000002.ply"
        data_dir = LOCAL_DATA_DIR / "examples" / "barbecue-sauce"
        mesh_dir = data_dir / "meshes"
        mesh_path = mesh_dir / mesh_file_name

        self.coarse_run_id = "coarse-bop-hope-pbr--225203"
        self.refiner_run_id = "refiner-bop-hope-pbr--955392"
        # TODO : this should be corrected and probably use pytest.parametrize as other tests
        # however, stacking decorators seems to not work as intended.
        self.device = "cpu"

        rgb, depth, camera_data = load_observation_example(data_dir, load_depth=True)
        # TODO: cosypose forward does not work if depth is loaded detection contrary to megapose
        self.observation = ObservationTensor.from_numpy(
            rgb, depth=None, K=camera_data.K
        ).cpu()

        self.detector = load_detector(
            run_id="detector-bop-hope-pbr--15246", device=self.device
        )
        # detections = detector.get_detections(observation=observation)

        self.object_dataset = RigidObjectDataset(
            objects=[
                RigidObject(
                    label=self.expected_object_label,
                    mesh_path=mesh_path,
                    mesh_units="mm",
                )
            ]
        )
        mesh_db = MeshDataBase.from_object_ds(self.object_dataset)
        self.mesh_db_batched = mesh_db.batched().to("cpu")

    @pytest.mark.parametrize("device", DEVICE)
    @pytest.mark.order(3)
    def test_cosypose_pipeline_panda3d(self, device):
        from happypose.toolbox.renderer.panda3d_batch_renderer import (
            Panda3dBatchRenderer,
        )

        # This is a trick that should be replaced, see comment line 38
        if device == "cpu":
            self.device = device
            self.observation = self.observation.cpu()
            self.detector.to(device)
            self.mesh_db_batched.to(device)

        else:
            self.device = device
            self.observation = self.observation.cuda()
            self.detector.to(device)
            self.mesh_db_batched.to(device)

        renderer = Panda3dBatchRenderer(
            self.object_dataset,
            n_workers=1,
            preload_cache=False,
        )

        coarse_model = load_model_cosypose(
            EXP_DIR / self.coarse_run_id, renderer, self.mesh_db_batched, self.device
        )
        refiner_model = load_model_cosypose(
            EXP_DIR / self.refiner_run_id, renderer, self.mesh_db_batched, self.device
        )

        pose_estimator = PoseEstimator(
            refiner_model=refiner_model,
            coarse_model=coarse_model,
            detector_model=self.detector,
        )

        # Run detector and pose estimator filtering object labels
        preds, _ = pose_estimator.run_inference_pipeline(
            observation=self.observation,
            detection_th=0.8,
            run_detector=True,
            n_refiner_iterations=3,
            labels_to_keep=[self.expected_object_label],
        )

        assert len(preds) == 1
        assert preds.infos.label[0] == self.expected_object_label

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

    @pytest.mark.parametrize("device", DEVICE)
    def test_cosypose_pipeline_bullet(self, device):
        from happypose.toolbox.renderer.bullet_batch_renderer import BulletBatchRenderer
        # This is a trick that should be replaced, see comment line 38

        if device == "cpu":
            self.device = device
            self.observation = self.observation.cpu()
            self.detector.to(device)
            self.mesh_db_batched.to(device)
            renderer = BulletBatchRenderer(
                self.object_dataset,
                n_workers=0,
                gpu_renderer=False,
            )

        else:
            self.device = device
            self.observation = self.observation.cuda()
            self.detector.to(device)
            self.mesh_db_batched.to(device)
            renderer = BulletBatchRenderer(
                self.object_dataset,
                n_workers=0,
                gpu_renderer=True,
            )

        coarse_model = load_model_cosypose(
            EXP_DIR / self.coarse_run_id, renderer, self.mesh_db_batched, self.device
        )
        refiner_model = load_model_cosypose(
            EXP_DIR / self.refiner_run_id, renderer, self.mesh_db_batched, self.device
        )

        pose_estimator = PoseEstimator(
            refiner_model=refiner_model,
            coarse_model=coarse_model,
            detector_model=self.detector,
        )

        # Run detector and pose estimator filtering object labels
        preds, _ = pose_estimator.run_inference_pipeline(
            observation=self.observation,
            detection_th=0.8,
            run_detector=True,
            n_refiner_iterations=3,
            labels_to_keep=[self.expected_object_label],
        )

        assert len(preds) == 1
        assert preds.infos.label[0] == self.expected_object_label

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
