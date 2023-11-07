import unittest

from omegaconf import OmegaConf

from happypose.pose_estimators.megapose.evaluation.eval_config import (
    BOPEvalConfig,
    EvalConfig,
    FullEvalConfig,
    HardwareConfig,
)
from happypose.pose_estimators.megapose.inference.types import InferenceConfig


class TestOmegaConf(unittest.TestCase):
    """
    Check if megapose config dataclasses are valid.
    """

    def test_valid_dataclasses(self):
        OmegaConf.structured(BOPEvalConfig)
        OmegaConf.structured(HardwareConfig)
        OmegaConf.structured(InferenceConfig)
        OmegaConf.structured(EvalConfig)
        OmegaConf.structured(FullEvalConfig)


if __name__ == "__main__":
    unittest.main()
