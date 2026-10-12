# Standard Library
from abc import ABCMeta, abstractmethod

# Third Party
import torch

# MegaPose
from happypose.toolbox.inference.types import PoseEstimatesType


class PoseEstimationModule(torch.nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def forward_coarse_model(
        self,
    ) -> tuple[PoseEstimatesType, dict]:
        pass

    @abstractmethod
    def forward_refiner(
        self,
    ) -> tuple[dict, dict]:
        pass

    @abstractmethod
    def run_inference_pipeline(
        self,
    ) -> tuple[PoseEstimatesType, dict]:
        pass
