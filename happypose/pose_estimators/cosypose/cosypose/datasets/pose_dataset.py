import random
from dataclasses import dataclass

import numpy as np
import torch

from happypose.pose_estimators.cosypose.cosypose.config import LOCAL_DATA_DIR
from happypose.pose_estimators.cosypose.cosypose.lib3d import invert_T

from .augmentations import (
    CropResizeToAspectAugmentation,
    GrayScale,
    PillowBlur,
    PillowBrightness,
    PillowColor,
    PillowContrast,
    PillowSharpness,
    VOCBackgroundAugmentation,
    to_torch_uint8,
)
from .wrappers.visibility_wrapper import VisibilityWrapper


@dataclass
class PoseData:
    images: None
    bboxes: None
    TCO: None
    K: None
    objects: None

    def pin_memory(self):
        self.images = self.images.pin_memory()
        self.bboxes = self.bboxes.pin_memory()
        self.TCO = self.TCO.pin_memory()
        self.K = self.K.pin_memory()
        return self


class NoObjectError(Exception):
    pass


class PoseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        scene_ds,
        resize=(640, 480),
        min_area=None,
        rgb_augmentation=False,
        gray_augmentation=False,
        background_augmentation=False,
    ):
        self.scene_ds = VisibilityWrapper(scene_ds)

        self.resize_augmentation = CropResizeToAspectAugmentation(resize=resize)
        self.min_area = min_area

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
        if gray_augmentation:
            self.rgb_augmentations.append(GrayScale(p=0.5))

    def __len__(self):
        return len(self.scene_ds)

    def collate_fn(self, batch):
        data = {}
        for k in batch[0].__annotations__:
            v = [getattr(x, k) for x in batch]
            if k in ("images", "bboxes", "TCO", "K"):
                v = torch.as_tensor(np.stack(v))
            data[k] = v
        data = PoseData(**data)
        return data

    def get_data(self, idx):
        rgb, mask, state = self.scene_ds[idx]

        rgb, mask, state = self.resize_augmentation(rgb, mask, state)

        if self.background_augmentation:
            rgb, mask, state = self.background_augmentations(rgb, mask, state)

        if self.rgb_augmentation and random.random() < 0.8:
            for augmentation in self.rgb_augmentations:
                rgb, mask, state = augmentation(rgb, mask, state)

        rgb, mask = to_torch_uint8(rgb), to_torch_uint8(mask)
        mask_uniqs = set(np.unique(mask))
        objects_visible = []
        for obj in state["objects"]:
            add = False
            if obj["id_in_segm"] in mask_uniqs and np.all(np.array(obj["bbox"]) >= 0):
                add = True

            if add and self.min_area is not None:
                bbox = np.array(obj["bbox"])
                area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
                if area >= self.min_area:
                    add = True
                else:
                    add = False
            if add:
                objects_visible.append(obj)
        if len(objects_visible) == 0:
            raise NoObjectError
        # assert len(objects_visible) > 0, idx

        rgb = torch.as_tensor(rgb).permute(2, 0, 1).to(torch.uint8)
        assert rgb.shape[0] == 3

        obj = random.sample(objects_visible, k=1)[0]
        TWO = torch.as_tensor(obj["TWO"])
        TWC = torch.as_tensor(state["camera"]["TWC"])
        TCO = invert_T(TWC) @ TWO

        data = PoseData(
            images=np.asarray(rgb),
            bboxes=np.asarray(obj["bbox"]),
            TCO=np.asarray(TCO),
            K=np.asarray(state["camera"]["K"]),
            objects=obj,
        )
        return data

    def __getitem__(self, index):
        try_index = index
        valid = False
        n_attempts = 0
        while not valid:
            if n_attempts > 10:
                msg = "Cannot find valid image in the dataset"
                raise ValueError(msg)
            try:
                data = self.get_data(try_index)
                valid = True
            except NoObjectError:
                try_index = random.randint(0, len(self.scene_ds) - 1)
                n_attempts += 1
        return data
