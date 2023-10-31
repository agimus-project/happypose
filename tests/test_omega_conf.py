from happypose.pose_estimators.megapose.inference.types import InferenceConfig


from happypose.pose_estimators.megapose.evaluation.eval_config import (
    BOPEvalConfig,
    EvalConfig,
    FullEvalConfig,
    HardwareConfig,
)
from omegaconf import OmegaConf


import unittest

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


if __name__ == '__main__':
    unittest.main()