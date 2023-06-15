# Standard Library
from abc import ABCMeta, abstractmethod

# Third Party
import torch

# MegaPose
from happypose.toolbox.inference.types import DetectionsType


class DetectorModule(torch.nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def get_detections(self) -> DetectionsType:
        pass
