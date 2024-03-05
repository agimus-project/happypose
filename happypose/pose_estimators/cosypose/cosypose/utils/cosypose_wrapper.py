#!/usr/bin/env python

# Copyright (c) CTU -- All Rights Reserved
# Created on: 2021-04-24
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#


"""TODO:
----
- Deprecate this class when possible
"""


from typing import Union

import torch

from happypose.pose_estimators.cosypose.cosypose.config import EXP_DIR
from happypose.pose_estimators.cosypose.cosypose.integrated.pose_estimator import (
    PoseEstimator,
)

# Detection
from happypose.pose_estimators.cosypose.cosypose.training.pose_models_cfg import (
    load_model_cosypose,
)
from happypose.toolbox.datasets.datasets_cfg import make_object_dataset
from happypose.toolbox.datasets.object_dataset import RigidObjectDataset
from happypose.toolbox.inference.utils import load_detector
from happypose.toolbox.lib3d.rigid_mesh_database import MeshDataBase

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CosyPoseWrapper:
    def __init__(
        self,
        dataset_name: str,
        object_dataset: Union[None, RigidObjectDataset] = None,
        n_workers=8,
        gpu_renderer=False,
        renderer_type: str = "panda3d",
    ) -> None:
        """
        inputs:
        - dataset_name: hope|tless|ycbv
        - object_dataset: None or already existing rigid object dataset. If None, will use dataset_name
        to build one
        - n_workers: how many processes will be spun in the batch renderer
        - renderer_type: 'panda3d'|'bullet'
        """

        self.dataset_name = dataset_name
        self.object_dataset = object_dataset
        self.detector, self.pose_predictor = self.get_model(
            dataset_name, n_workers, renderer_type
        )

    def get_model(self, dataset_name, n_workers, renderer_type):
        # load models
        if dataset_name == "hope":
            # HOPE setup
            # python -m happypose.toolbox.utils.download --cosypose_models=detector-bop-hope-pbr--15246
            # python -m happypose.toolbox.utils.download --cosypose_models=coarse-bop-hope-pbr--225203
            # python -m happypose.toolbox.utils.download --cosypose_models=refiner-bop-hope-pbr--955392
            detector_run_id = "detector-bop-hope-pbr--15246"
            coarse_run_id = "coarse-bop-hope-pbr--225203"
            refiner_run_id = "refiner-bop-hope-pbr--955392"
        elif dataset_name == "tless":
            # TLESS setup
            # python -m happypose.toolbox.utils.download --cosypose_models=detector-bop-tless-pbr--873074
            # python -m happypose.toolbox.utils.download --cosypose_models=coarse-bop-tless-pbr--506801
            # python -m happypose.toolbox.utils.download --cosypose_models=refiner-bop-tless-pbr--233420
            detector_run_id = "detector-bop-tless-pbr--873074"
            coarse_run_id = "coarse-bop-tless-pbr--506801"
            refiner_run_id = "refiner-bop-tless-pbr--233420"
        elif dataset_name == "ycbv":
            # YCBV setup
            # python -m happypose.toolbox.utils.download --cosypose_models=detector-bop-ycbv-pbr--970850
            # python -m happypose.toolbox.utils.download --cosypose_models=coarse-bop-ycbv-pbr--724183
            # python -m happypose.toolbox.utils.download --cosypose_models=refiner-bop-ycbv-pbr--604090
            detector_run_id = "detector-bop-ycbv-pbr--970850"
            coarse_run_id = "coarse-bop-ycbv-pbr--724183"
            refiner_run_id = "refiner-bop-ycbv-pbr--604090"
        else:
            msg = f"Not prepared for {dataset_name} dataset"
            raise ValueError(msg)
        detector = load_detector(detector_run_id, device)
        coarse_model, refiner_model = self.load_pose_models(
            coarse_run_id, refiner_run_id, n_workers, renderer_type
        )

        pose_estimator = PoseEstimator(
            refiner_model=refiner_model,
            coarse_model=coarse_model,
            detector_model=detector,
        )
        return detector, pose_estimator

    def load_pose_models(
        self,
        coarse_run_id: str,
        refiner_run_id: str,
        n_workers: int,
        renderer_type: str = "panda3d",
    ):
        if self.object_dataset is None:
            self.object_dataset = make_object_dataset(self.dataset_name)
        if renderer_type == "panda3d":
            from happypose.toolbox.renderer.panda3d_batch_renderer import (
                Panda3dBatchRenderer,
            )

            renderer = Panda3dBatchRenderer(
                self.object_dataset,
                n_workers=n_workers,
            )
        elif renderer_type == "bullet":
            from happypose.toolbox.renderer.bullet_batch_renderer import (
                BulletBatchRenderer,
            )

            renderer = BulletBatchRenderer(
                self.object_dataset,
                n_workers=n_workers,
                gpu_renderer=torch.cuda.is_available(),
            )
        else:
            raise ValueError(f"Renderer type {renderer_type} not supported")

        mesh_db = MeshDataBase.from_object_ds(self.object_dataset)
        mesh_db_batched = mesh_db.batched().to(device)

        coarse_model = load_model_cosypose(
            EXP_DIR / coarse_run_id, renderer, mesh_db_batched, device
        )
        refiner_model = load_model_cosypose(
            EXP_DIR / refiner_run_id, renderer, mesh_db_batched, device
        )
        return coarse_model, refiner_model

    def inference(self, observation, coarse_guess=None):
        detections = None
        run_detector = True
        if coarse_guess is None:
            final_preds, all_preds = self.pose_predictor.run_inference_pipeline(
                observation,
                detections=detections,
                run_detector=run_detector,
                data_TCO_init=None,
                n_coarse_iterations=1,
                n_refiner_iterations=4,
            )
        else:
            final_preds, all_preds = self.pose_predictor.run_inference_pipeline(
                observation,
                detections=detections,
                run_detector=run_detector,
                data_TCO_init=None,
                n_coarse_iterations=0,
                n_refiner_iterations=4,
            )
        print("inference successfull")
        return final_preds.cpu()
