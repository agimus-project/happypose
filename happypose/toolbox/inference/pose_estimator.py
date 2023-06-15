# Standard Library
from abc import ABCMeta, abstractmethod
from typing import Tuple

# Third Party
import torch

# MegaPose
from happypose.toolbox.inference.types import PoseEstimatesType


class PoseEstimationModule(torch.nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def forward_coarse_model(self) -> Tuple[PoseEstimatesType, dict]:
        pass

    @abstractmethod
    def forward_refiner(self) -> Tuple[dict, dict]:
        pass

    @abstractmethod
    def run_inference_pipeline(self) -> Tuple[PoseEstimatesType, dict]:
        pass
