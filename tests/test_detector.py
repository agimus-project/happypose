"""Set of unit tests for testing inference example for CosyPose."""

import unittest

from happypose.pose_estimators.cosypose.cosypose.config import LOCAL_DATA_DIR
from happypose.toolbox.inference.example_inference_utils import load_observation_example
from happypose.toolbox.inference.types import ObservationTensor
from happypose.toolbox.inference.utils import filter_detections, load_detector


class TestDetector(unittest.TestCase):
    """Unit tests for CosyPose inference example."""

    def test_detector(self):
        """Run detector on known image to see if object is detected."""

        expected_object_label = "hope-obj_000002"
        data_dir = LOCAL_DATA_DIR / "examples" / "barbecue-sauce"

        rgb, depth, camera_data = load_observation_example(data_dir, load_depth=True)
        # TODO: cosypose forward does not work if depth is loaded detection contrary to megapose
        observation = ObservationTensor.from_numpy(rgb, depth=None, K=camera_data.K)

        detector = load_detector(run_id="detector-bop-hope-pbr--15246", device="cpu")
        detections = detector.get_detections(observation=observation)
        detections = filter_detections(detections, labels=[expected_object_label])

        for s1, s2 in zip(detections.infos.score, detections.infos.score[1:]):
            self.assertGreater(s1, s2)  # checks that observations are ordered

        self.assertGreater(len(detections), 0)
        self.assertEqual(detections.infos.label[0], expected_object_label)
        self.assertGreater(detections.infos.score[0], 0.8)

        bbox = detections.bboxes[0]  # xmin, ymin, xmax, ymax

        # assert if different sample points lie inside/outside the bounding box
        self.assertTrue(is_point_in([430, 300], bbox))
        self.assertTrue(is_point_in([430, 400], bbox))
        self.assertTrue(is_point_in([490, 255], bbox))

        self.assertFalse(is_point_in([0, 0], bbox))
        self.assertFalse(is_point_in([200, 400], bbox))
        self.assertFalse(is_point_in([490, 50], bbox))
        self.assertFalse(is_point_in([550, 500], bbox))


def is_point_in(p, bb):
    """
    p: pixel sequence (x,y)
    bb: bounding box sequence (xmin, ymin, xmax, ymax)
    """
    return bb[0] < p[0] < bb[2] and bb[1] < p[1] < bb[3]


if __name__ == "__main__":
    unittest.main()
