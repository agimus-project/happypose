"""Set of unit tests for testing inference example for CosyPose."""
import unittest
from pathlib import Path

import numpy as np
import pinocchio as pin
import torch
import yaml
from PIL import Image

from happypose.pose_estimators.cosypose.cosypose.config import EXP_DIR
from happypose.pose_estimators.cosypose.cosypose.integrated.detector import Detector
from happypose.pose_estimators.cosypose.cosypose.integrated.pose_estimator import (
    PoseEstimator,
)
from happypose.pose_estimators.cosypose.cosypose.training.detector_models_cfg import (
    check_update_config as check_update_config_detector,
)
from happypose.pose_estimators.cosypose.cosypose.training.detector_models_cfg import (
    create_model_detector,
)
from happypose.pose_estimators.cosypose.cosypose.training.pose_models_cfg import (
    check_update_config as check_update_config_pose,
)
from happypose.pose_estimators.cosypose.cosypose.training.pose_models_cfg import (
    create_model_coarse,
    create_model_refiner,
)
from happypose.toolbox.datasets.bop_object_datasets import BOPObjectDataset
from happypose.toolbox.datasets.scene_dataset import CameraData
from happypose.toolbox.inference.types import ObservationTensor
from happypose.toolbox.lib3d.rigid_mesh_database import MeshDataBase
from happypose.toolbox.renderer.panda3d_batch_renderer import Panda3dBatchRenderer


class TestCosyPoseInference(unittest.TestCase):
    """Unit tests for CosyPose inference example."""

    @staticmethod
    def _load_detector(
        device="cpu",
        ds_name="ycbv",
        run_id="detector-bop-ycbv-pbr--970850",
    ):
        """Load CosyPose detector."""
        run_dir = EXP_DIR / run_id
        assert run_dir.exists(), "The run_id is invalid, or you forget to download data"
        cfg = check_update_config_detector(
            yaml.load((run_dir / "config.yaml").read_text(), Loader=yaml.UnsafeLoader),
        )
        label_to_category_id = cfg.label_to_category_id
        ckpt = torch.load(run_dir / "checkpoint.pth.tar", map_location=device)[
            "state_dict"
        ]
        model = create_model_detector(cfg, len(label_to_category_id))
        model.load_state_dict(ckpt)
        model = model.to(device).eval()
        model.cfg = cfg
        model.config = cfg
        return Detector(model, ds_name)

    @staticmethod
    def _load_pose_model(run_id, renderer, mesh_db, device):
        """Load either coarse or refiner model (decided based on run_id/config)."""
        run_dir = EXP_DIR / run_id
        cfg = yaml.load((run_dir / "config.yaml").read_text(), Loader=yaml.UnsafeLoader)
        cfg = check_update_config_pose(cfg)

        f_mdl = create_model_refiner if cfg.train_refiner else create_model_coarse
        ckpt = torch.load(run_dir / "checkpoint.pth.tar", map_location=device)[
            "state_dict"
        ]
        model = f_mdl(cfg, renderer=renderer, mesh_db=mesh_db)
        model.load_state_dict(ckpt)
        model = model.to(device).eval()
        model.cfg = cfg
        model.config = cfg
        return model

    @staticmethod
    def _load_pose_models(
        coarse_run_id="coarse-bop-ycbv-pbr--724183",
        refiner_run_id="refiner-bop-ycbv-pbr--604090",
        n_workers=1,
        device="cpu",
    ):
        """Load coarse and refiner for the crackers example renderer."""
        object_dataset = BOPObjectDataset(
            Path(__file__).parent / "data" / "crackers_example" / "models",
            label_format="ycbv-{label}",
        )
        renderer = Panda3dBatchRenderer(
            object_dataset,
            n_workers=n_workers,
            preload_cache=False,
        )

        mesh_db = MeshDataBase.from_object_ds(object_dataset)
        mesh_db_batched = mesh_db.batched().to(device)
        kwargs = {"renderer": renderer, "mesh_db": mesh_db_batched, "device": device}
        coarse_model = TestCosyPoseInference._load_pose_model(coarse_run_id, **kwargs)
        refiner_model = TestCosyPoseInference._load_pose_model(refiner_run_id, **kwargs)
        return coarse_model, refiner_model

    @staticmethod
    def _load_crackers_example_observation():
        """Load cracker example observation tensor."""
        data_dir = Path(__file__).parent.joinpath("data").joinpath("crackers_example")
        camera_data = CameraData.from_json((data_dir / "camera_data.json").read_text())
        rgb = np.array(Image.open(data_dir / "image_rgb.png"), dtype=np.uint8)
        assert rgb.shape[:2] == camera_data.resolution
        return ObservationTensor.from_numpy(rgb=rgb, K=camera_data.K)

    def test_detector(self):
        """Run detector on known image to see if cracker box is detected."""
        observation = self._load_crackers_example_observation()
        detector = self._load_detector()
        detections = detector.get_detections(observation=observation)
        for s1, s2 in zip(detections.infos.score, detections.infos.score[1:]):
            self.assertGreater(s1, s2)  # checks that observations are ordered

        self.assertGreater(len(detections), 0)
        self.assertEqual(detections.infos.label[0], "ycbv-obj_000002")
        self.assertGreater(detections.infos.score[0], 0.8)

        xmin, ymin, xmax, ymax = detections.bboxes[0]
        # assert expected obj center inside BB
        self.assertTrue(xmin < 320 < xmax and ymin < 250 < ymax)
        # assert a few outside points are outside BB
        self.assertFalse(xmin < 100 < xmax and ymin < 50 < ymax)
        self.assertFalse(xmin < 300 < xmax and ymin < 50 < ymax)
        self.assertFalse(xmin < 500 < xmax and ymin < 50 < ymax)
        self.assertFalse(xmin < 100 < xmax and ymin < 250 < ymax)
        self.assertTrue(xmin < 300 < xmax and ymin < 250 < ymax)
        self.assertFalse(xmin < 500 < xmax and ymin < 250 < ymax)
        self.assertFalse(xmin < 100 < xmax and ymin < 450 < ymax)
        self.assertFalse(xmin < 300 < xmax and ymin < 450 < ymax)
        self.assertFalse(xmin < 500 < xmax and ymin < 450 < ymax)

    def test_cosypose_pipeline(self):
        """Run detector with coarse and refiner."""
        observation = self._load_crackers_example_observation()
        detector = self._load_detector()
        coarse_model, refiner_model = self._load_pose_models()
        pose_estimator = PoseEstimator(
            refiner_model=refiner_model,
            coarse_model=coarse_model,
            detector_model=detector,
        )
        preds, _ = pose_estimator.run_inference_pipeline(
            observation=observation,
            detection_th=0.8,
            run_detector=True,
        )

        self.assertEqual(len(preds), 1)
        self.assertEqual(preds.infos.label[0], "ycbv-obj_000002")

        pose = pin.SE3(preds.poses[0].numpy())
        exp_pose = pin.SE3(
            pin.exp3(np.array([1.44, 1.19, -0.91])),
            np.array([0, 0, 0.52]),
        )
        diff = pose.inverse() * exp_pose
        self.assertLess(np.linalg.norm(pin.log6(diff).vector), 0.1)


if __name__ == "__main__":
    unittest.main()
