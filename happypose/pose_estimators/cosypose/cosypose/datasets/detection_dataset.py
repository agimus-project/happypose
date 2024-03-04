import random
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

from happypose.pose_estimators.cosypose.cosypose.config import LOCAL_DATA_DIR

"""
 from .augmentations import (
    CropResizeToAspectAugmentation,
    PillowBlur,
    PillowBrightness,
    PillowColor,
    PillowContrast,
    PillowSharpness,
    VOCBackgroundAugmentation,
    to_torch_uint8,
)
"""

from happypose.toolbox.datasets.augmentations import (
    CropResizeToAspectTransform,
    PillowBlur,
    PillowBrightness,
    PillowColor,
    PillowContrast,
    PillowSharpness,
    VOCBackgroundAugmentation,
)

# HappyPose
from happypose.toolbox.datasets.scene_dataset import (
    IterableSceneDataset,
    ObjectData,
    SceneDataset,
    SceneObservation,
)

from happypose.toolbox.datasets.augmentations import (
    SceneObservationAugmentation as SceneObsAug,
)

from happypose.toolbox.datasets.scene_dataset_wrappers import remove_invisible_objects
# from happypose.toolbox.datasets.scene_dataset_wrappers import VisibilityWrapper
"""
from .augmentations import (
    CropResizeToAspectAugmentation,
    PillowBlur,
    PillowBrightness,
    PillowColor,
    PillowContrast,
    PillowSharpness,
    VOCBackgroundAugmentation,
    to_torch_uint8,
)
from .wrappers.visibility_wrapper import VisibilityWrapper
"""

def collate_fn(batch):
    print("inside collate fn")
    rgbs, targets = zip(*batch)
    print("rgbs = ", rgbs)
    print("targets =", targets)
    # Stack the rgbs and convert to a tensor
    rgbs = torch.stack(rgbs, dim=0)

    # Initialize the target dictionary
    target = {
        "boxes": [],
        "labels": [],
        "masks": [],
        "image_id": [],
        "area": [],
        "iscrowd": [],
    }

    # Concatenate the target data for each image in the batch
    for t in targets:
        target["boxes"].append(t["boxes"])
        target["labels"].append(t["labels"])
        target["masks"].append(t["masks"])
        target["image_id"].append(t["image_id"])
        target["area"].append(t["area"])
        target["iscrowd"].append(t["iscrowd"])

    # Stack the target data and convert to tensors
    for key in target.keys():
        target[key] = torch.cat(target[key], dim=0)

    # Return the batch data as a dictionary
    return {"rgbs": rgbs, "targets": target}

#TODO : Double check on types and add format documentation
@dataclass
class DetectionData:
    """rgb: (h, w, 3) uint8
    depth: (bsz, h, w) float32
    bbox: (4, ) int
    K: (3, 3) float32
    TCO: (4, 4) float32.
    """

    rgb: np.array
    bboxes: np.array
    labels: np.array
    masks: np.array
    area: np.array
    iscrowd: np.array


@dataclass
class BatchDetectionData:
    """rgbs: (bsz, 3, h, w) uint8
    depths: (bsz, h, w) float32
    bboxes: (bsz, 4) int
    TCO: (bsz, 4, 4) float32
    K: (bsz, 3, 3) float32.
    """

    rgbs: torch.Tensor
    bboxes: torch.Tensor
    labels: torch.Tensor
    masks: torch.Tensor
    area: torch.Tensor
    iscrowd: torch.Tensor

    def pin_memory(self) -> "BatchDetectionData":
        self.rgbs = self.rgbs.pin_memory()
        self.bboxes = self.bboxes.pin_memory()
        self.TCO = self.TCO.pin_memory()
        self.K = self.K.pin_memory()
        if self.depths is not None:
            self.depths = self.depths.pin_memory()
        return self

class DetectionDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        scene_ds,
        label_to_category_id,
        min_area=50,
        resize=(640, 480),
        gray_augmentation=False,
        rgb_augmentation=False,
        background_augmentation=False,
    ):
        self.scene_ds = scene_ds

        self.resize_augmentation = CropResizeToAspectTransform()

        self.background_augmentations = []
        self.background_augmentations += [
            (
                SceneObsAug(
                    VOCBackgroundAugmentation(LOCAL_DATA_DIR),
                    p=0.3,
                )
            ),
        ]

        self.rgb_augmentations = []
        self.rgb_augmentations += [
            SceneObsAug(
                [
                    SceneObsAug(PillowBlur(factor_interval=(1, 3)), p=0.4),
                    SceneObsAug(PillowSharpness(factor_interval=(0.0, 50.0)), p=0.3),
                    SceneObsAug(PillowContrast(factor_interval=(0.2, 50.0)), p=0.3),
                    SceneObsAug(PillowBrightness(factor_interval=(0.1, 6.0)), p=0.5),
                    SceneObsAug(PillowColor(factor_interval=(0.0, 20.0)), p=0.3),]
            )

        ]

        self.label_to_category_id = label_to_category_id
        self.min_area = min_area
    """
    def collate_fn(self, list_data: List[DetectionData]) -> BatchDetectionData:
        batch_data = BatchDetectionData(
            rgbs=torch.from_numpy(np.stack([d.rgb for d in list_data])).permute(
                0,
                3,
                1,
                2,
            ),
            bboxes=torch.from_numpy(np.stack([d.bboxes for d in list_data])),
            labels=torch.from_numpy(np.stack([d.labels for d in list_data])),
            masks=torch.from_numpy(np.stack([d.masks for d in list_data])),
            #image_id=torch.from_numpy(np.stack([d.image_id for d in list_data])),
            area=torch.from_numpy(np.stack([d.area for d in list_data])),
            iscrowd=torch.from_numpy(np.stack([d.iscrowd for d in list_data]))
            )

        return batch_data
    """
    def make_data_from_obs(self, obs: SceneObservation, idx):
        
        obs = remove_invisible_objects(obs)

        obs = self.resize_augmentation(obs)

        for aug in self.background_augmentations:
            obs = aug(obs)
            
        if self.rgb_augmentations and random.random() < 0.8:
            for aug in self.rgb_augmentations:
                obs = aug(obs)
        
        assert obs.object_datas is not None
        assert obs.rgb is not None
        assert obs.camera_data is not None
        categories = torch.tensor(
            [self.label_to_category_id[obj.label] for obj in obs.object_datas],
        )
        boxes = np.array(
            [torch.as_tensor(obj.bbox_modal).tolist() for obj in obs.object_datas],
        )
        area = torch.as_tensor(
            (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
        )
        obj_ids = np.array([obj.unique_id for obj in obs.object_datas])

        masks = []
        for _n, obj_data in enumerate(obs.object_datas):
            if obs.binary_masks is not None:
                binary_mask = torch.tensor(obs.binary_masks[obj_data.unique_id]).float()
                masks.append(binary_mask)
                
            if obs.segmentation is not None:
                binary_mask = np.zeros_like(obs.segmentation, dtype=np.bool_)
                binary_mask[obs.segmentation == obj_data.unique_id] = 1
                binary_mask = torch.as_tensor(binary_mask).float()
                masks.append(binary_mask)
                
        masks = np.array(masks)
        masks = masks == obj_ids[:, None, None]

        keep = area > self.min_area
        boxes = boxes[keep]
        area = area[keep]
        categories = categories[keep]
        masks = masks[keep, :, :]
        num_objs = len(keep)

        num_objs = len(obj_ids)
        area = torch.as_tensor(area)
        boxes = torch.as_tensor(boxes)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        iscrowd = torch.zeros((num_objs), dtype=torch.int64)

        rgb = torch.as_tensor(obs.rgb)
        target = {}
        target["boxes"] = boxes
        target["labels"] = categories
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        
        return rgb, target
    

    
    def __getitem__(self, index: int):
        assert isinstance(self.scene_ds, SceneDataset)
        obs = self.scene_ds[index]
        return self.make_data_from_obs(obs, index)
    
      # def find_valid_data(self, iterator: Iterator[SceneObservation]) -> PoseData:
    def find_valid_data(self, iterator):
        n_attempts = 0
        for idx, obs in enumerate(iterator):
            data = self.make_data_from_obs(obs, idx)
            if data is not None:
                return data
            n_attempts += 1
            if n_attempts > 200:
                msg = "Cannot find valid image in the dataset"
                raise ValueError(msg)
        
    def __iter__(self):
        assert isinstance(self.scene_ds, IterableSceneDataset)
        iterator = iter(self.scene_ds)
        while True:
            yield self.find_valid_data(iterator)
            
            
    
    
    
    def __init___old(
        self,
        scene_ds,
        label_to_category_id,
        min_area=50,
        resize=(640, 480),
        gray_augmentation=False,
        rgb_augmentation=False,
        background_augmentation=False,
    ):
        self.scene_ds = scene_ds

        self.resize_augmentation = CropResizeToAspectAugmentation(resize=resize)

        self.background_augmentation = background_augmentation
        self.background_augmentations = VOCBackgroundAugmentation(
            voc_root=LOCAL_DATA_DIR,
            p=0.3,
        )

        self.rgb_augmentation = rgb_augmentation
        self.rgb_augmentations = [
            PillowBlur(p=0.4, factor_interval=(1, 3)),
            PillowSharpness(p=0.3, factor_interval=(0.0, 50.0)),
            PillowContrast(p=0.3, factor_interval=(0.2, 50.0)),
            PillowBrightness(p=0.5, factor_interval=(0.1, 6.0)),
            PillowColor(p=0.3, factor_interval=(0.0, 20.0)),
        ]

        self.label_to_category_id = label_to_category_id
        self.min_area = min_area

    

    def get_data_old(self, idx):
        
        print("I am in get_data")
        rgb, mask, state = self.scene_ds[idx]

        rgb, mask, state = self.resize_augmentation(rgb, mask, state)

        if self.background_augmentation:
            rgb, mask, state = self.background_augmentations(rgb, mask, state)

        if self.rgb_augmentation and random.random() < 0.8:
            for augmentation in self.rgb_augmentations:
                rgb, mask, state = augmentation(rgb, mask, state)

        rgb, mask = to_torch_uint8(rgb), to_torch_uint8(mask)

        categories = torch.tensor(
            [self.label_to_category_id[obj["name"]] for obj in state["objects"]],
        )
        obj_ids = np.array([obj["id_in_segm"] for obj in state["objects"]])
        boxes = np.array(
            [torch.as_tensor(obj["bbox"]).tolist() for obj in state["objects"]],
        )
        boxes = torch.as_tensor(boxes, dtype=torch.float32).view(-1, 4)
        area = torch.as_tensor(
            (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
        )
        mask = np.array(mask)
        masks = mask == obj_ids[:, None, None]
        masks = torch.as_tensor(masks)

        keep = area > self.min_area
        boxes = boxes[keep]
        area = area[keep]
        categories = categories[keep]
        masks = masks[keep, :, :]
        num_objs = len(keep)

        num_objs = len(obj_ids)
        area = torch.as_tensor(area)
        boxes = torch.as_tensor(boxes)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        iscrowd = torch.zeros((num_objs), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = categories
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        return rgb, target

    def __getitem__old(self, index):
        try_index = index
        valid = False
        n_attempts = 0
        while not valid:
            print("valid =", valid)
            if n_attempts > 10:
                msg = "Cannot find valid image in the dataset"
                raise ValueError(msg)
            im, target = self.get_data(try_index)
            valid = len(target["boxes"]) > 0
            if not valid:
                try_index = random.randint(0, len(self.scene_ds) - 1)
                n_attempts += 1
        return im, target
    
