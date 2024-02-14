from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import happypose.pose_estimators.cosypose.cosypose.utils.tensor_collection as tc
from happypose.pose_estimators.cosypose.cosypose.datasets.samplers import (
    DistributedSceneSampler,
)
from happypose.pose_estimators.cosypose.cosypose.evaluation.data_utils import (
    parse_obs_data,
)
from happypose.pose_estimators.cosypose.cosypose.utils.distributed import (
    get_rank,
    get_tmp_dir,
    get_world_size,
)


class PoseEvaluation:
    def __init__(
        self,
        scene_ds,
        meters,
        batch_size=64,
        cache_data=True,
        n_workers=4,
        sampler=None,
    ):
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.tmp_dir = get_tmp_dir()

        self.scene_ds = scene_ds
        if sampler is None:
            sampler = DistributedSceneSampler(
                scene_ds,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
            )
        dataloader = DataLoader(
            scene_ds,
            batch_size=batch_size,
            num_workers=n_workers,
            sampler=sampler,
            collate_fn=self.collate_fn,
        )

        if cache_data:
            self.dataloader = list(tqdm(dataloader))
        else:
            self.dataloader = dataloader

        self.meters = meters
        self.meters = OrderedDict(
            dict(sorted(self.meters.items(), key=lambda item: item[0])),
        )

    @staticmethod
    def make_empty_predictions():
        infos = {
            "view_id": np.empty(0, dtype=int),
            "scene_id": np.empty(0, dtype=int),
            "label": np.empty(0, dtype=np.object),
            "score": np.empty(0, dtype=np.float),
        }
        poses = torch.empty(0, 4, 4, dtype=torch.float)
        return tc.PandasTensorCollection(infos=pd.DataFrame(infos), poses=poses)

    def collate_fn(self, batch):
        obj_data = []
        for data_n in batch:
            _, _, obs = data_n
            obj_data_ = parse_obs_data(obs)
            obj_data.append(obj_data_)
        obj_data = tc.concatenate(obj_data)
        return obj_data

    def evaluate(self, obj_predictions, device="cuda"):
        for meter in self.meters.values():
            meter.reset()
        obj_predictions = obj_predictions.to(device)
        for obj_data_gt in tqdm(self.dataloader):
            for _k, meter in self.meters.items():
                meter.add(obj_predictions, obj_data_gt.to(device))
        return self.summary()

    def summary(self):
        summary, dfs = {}, {}
        for meter_k, meter in sorted(self.meters.items()):
            meter.gather_distributed(tmp_dir=self.tmp_dir)
            if get_rank() == 0 and len(meter.datas) > 0:
                summary_, df_ = meter.summary()
                dfs[meter_k] = df_
                for k, v in summary_.items():
                    summary[meter_k + "/" + k] = v
        return summary, dfs
