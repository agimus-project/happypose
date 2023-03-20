import torch
import numpy as np
import pandas as pd

import cosypose.utils.tensor_collection as tc


class Detector:
    def __init__(self, model, ds_name):
        model.eval()
        self.model = model
        self.config = model.config
        self.category_id_to_label = {v: k for k, v in self.config.label_to_category_id.items()}
        for k, v in self.category_id_to_label.items():
            if k ==0:
                continue
            self.category_id_to_label[k] = '{}-'.format(ds_name) + v 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @torch.no_grad()
    def get_detections(self, images, detection_th=None,
                       output_masks=False, mask_th=0.8,
                       one_instance_per_class=False):
        images = images.to(self.device).float()
        if images.shape[-1] == 3:
            images = images.permute(0, 3, 1, 2)
        if images.max() > 1:
            images = images / 255.
        outputs_ = self.model([image_n for image_n in images])

        infos = []
        bboxes = []
        masks = []
        for n, outputs_n in enumerate(outputs_):
            outputs_n['labels'] = [self.category_id_to_label[category_id.item()] \
                                   for category_id in outputs_n['labels']]
            for obj_id in range(len(outputs_n['boxes'])):
                bbox = outputs_n['boxes'][obj_id]
                info = dict(
                    batch_im_id=n,
                    label=outputs_n['labels'][obj_id],
                    score=outputs_n['scores'][obj_id].item(),
                )
                mask = outputs_n['masks'][obj_id, 0] > mask_th
                bboxes.append(torch.as_tensor(bbox))
                masks.append(torch.as_tensor(mask))
                infos.append(info)

        if len(bboxes) > 0:
            bboxes = torch.stack(bboxes).to(self.device).float()
            masks = torch.stack(masks).to(self.device)
        else:
            infos = dict(score=[], label=[], batch_im_id=[])
            bboxes = torch.empty(0, 4).to(self.device).float()
            masks = torch.empty(0, images.shape[1], images.shape[2], dtype=torch.bool).to(self.device)

        outputs = tc.PandasTensorCollection(
            infos=pd.DataFrame(infos),
            bboxes=bboxes,
        )
        if output_masks:
            outputs.register_tensor('masks', masks)
        if detection_th is not None:
            keep = np.where(outputs.infos['score'] > detection_th)[0]
            outputs = outputs[keep]

        if one_instance_per_class:
            infos = outputs.infos
            infos['det_idx'] = np.arange(len(infos))
            keep_ids = infos.sort_values('score', ascending=False).drop_duplicates('label')['det_idx'].values
            outputs = outputs[keep_ids]
            outputs.infos = outputs.infos.drop('det_idx', axis=1)
        return outputs

    def __call__(self, *args, **kwargs):
        return self.get_detections(*args, **kwargs)
