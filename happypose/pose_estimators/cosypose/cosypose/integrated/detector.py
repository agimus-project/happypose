from typing import Any, Optional

import cosypose.utils.tensor_collection as tc
import numpy as np
import pandas as pd
import torch

# MegaPose
import happypose.pose_estimators.megapose
import happypose.toolbox.utils.tensor_collection as tc
from happypose.toolbox.inference.detector import DetectorModule
from happypose.toolbox.inference.types import DetectionsType, ObservationTensor
from happypose.toolbox.inference.utils import filter_detections, add_instance_id


class Detector(DetectorModule):
    def __init__(self, model, ds_name):
        super().__init__()
        self.model = model
        self.model.eval()
        self.config = model.config
        self.category_id_to_label = {v: k for k, v in self.config.label_to_category_id.items()}
        if ds_name == "ycbv.bop19":
            ds_name="ycbv"
        for k, v in self.category_id_to_label.items():
            if k ==0:
                continue
            self.category_id_to_label[k] = '{}-'.format(ds_name) + v 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @torch.no_grad()
    def get_detections(
        self,
        observation: ObservationTensor,
        detection_th: Optional[float] = None,
        output_masks: bool = False,
        mask_th: float = 0.8,
        one_instance_per_class: bool = False,
    ) -> DetectionsType:
        """Runs the detector on the given images.

        Args:
            detection_th: If specified only keep detections above this
                threshold.
            mask_th: Threshold to use when computing masks
            one_instance_per_class: If True, keep only the highest scoring
                detection within each class.


        """

        # [B,3,H,W]
        RGB_DIMS = [0, 1, 2]
        images = observation.images[:, RGB_DIMS]

        # TODO (lmanuelli): Why are we splitting this up into a list of tensors?
        outputs_ = self.model([image_n for image_n in images])

        infos = []
        bboxes = []
        masks = []
        for n, outputs_n in enumerate(outputs_):
            outputs_n["labels"] = [
                self.category_id_to_label[category_id.item()] for category_id in outputs_n["labels"]
            ]
            for obj_id in range(len(outputs_n["boxes"])):
                bbox = outputs_n["boxes"][obj_id]
                info = dict(
                    batch_im_id=n,
                    label=outputs_n["labels"][obj_id],
                    score=outputs_n["scores"][obj_id].item(),
                )
                mask = outputs_n["masks"][obj_id, 0] > mask_th
                bboxes.append(torch.as_tensor(bbox))
                masks.append(torch.as_tensor(mask))
                infos.append(info)

        if len(bboxes) > 0:
            if torch.cuda.is_available():
                bboxes = torch.stack(bboxes).cuda().float()
                masks = torch.stack(masks).cuda()
            else:
                bboxes = torch.stack(bboxes).float()
                masks = torch.stack(masks)
        else:
            infos = dict(score=[], label=[], batch_im_id=[])
            if torch.cuda.is_available():
                bboxes = torch.empty(0, 4).cuda().float()
                masks = torch.empty(0, images.shape[1], images.shape[2], dtype=torch.bool).cuda()
            else:
                bboxes = torch.empty(0, 4).float()
                masks = torch.empty(0, images.shape[1], images.shape[2], dtype=torch.bool)

        outputs = tc.PandasTensorCollection(
            infos=pd.DataFrame(infos),
            bboxes=bboxes,
        )
        if output_masks:
            outputs.register_tensor("masks", masks)
        if detection_th is not None:
            keep = np.where(outputs.infos["score"] > detection_th)[0]
            outputs = outputs[keep]

        # Keep only the top-detection for each class label
        if one_instance_per_class:
            outputs = filter_detections(
                outputs, one_instance_per_class=True
            )

        # Add instance_id column to dataframe
        # Each detection is now associated with an `instance_id` that
        # identifies multiple instances of the same object
        outputs = add_instance_id(outputs)
        return outputs
    
    def __call__(self, *args, **kwargs):
        return self.get_detections(*args, **kwargs)
