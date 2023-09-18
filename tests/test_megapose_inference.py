"""Set of unit tests for testing inference example for MegaPose."""
import unittest

import numpy as np
import pinocchio as pin
from pathlib import Path

from .test_cosypose_inference import TestCosyPoseInference
from happypose.toolbox.datasets.bop_object_datasets import BOPObjectDataset
from happypose.toolbox.utils.load_model import NAMED_MODELS, load_named_model


class TestMegaPoseInference(unittest.TestCase):
    """Unit tests for MegaPose inference example."""

    def test_meagpose_pipeline(self):
        """Run detector from CosyPose with coarse and refiner from MegaPose"""
        observation = TestCosyPoseInference._load_crackers_example_observation()

        detector = TestCosyPoseInference._load_detector()
        detections = detector.get_detections(observation=observation)

        self.assertGreater(len(detections), 0)
        detections = detections[:1]  # let's keep the most promising one only.

        object_dataset = BOPObjectDataset(
            Path(__file__).parent / "data" / "crackers_example" / "models",
            label_format="ycbv-{label}",
        )

        model_info = NAMED_MODELS["megapose-1.0-RGB"]
        pose_estimator = load_named_model("megapose-1.0-RGB", object_dataset).to("cpu")
        preds, _ = pose_estimator.run_inference_pipeline(
            observation, detections=detections, **model_info["inference_parameters"]
        )

        self.assertEqual(len(preds), 1)
        self.assertEqual(preds.infos.label[0], "ycbv-obj_000002")

        pose = pin.SE3(preds.poses[0].numpy())
        exp_pose = pin.SE3(
            pin.exp3(np.array([1.44, 1.19, -0.91])), np.array([0, 0, 0.52])
        )
        diff = pose.inverse() * exp_pose
        self.assertLess(np.linalg.norm(pin.log6(diff).vector), 0.2)


if __name__ == "__main__":
    unittest.main()
