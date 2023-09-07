#!/usr/bin/env python

# Copyright (c) CTU -- All Rights Reserved
# Created on: 2021-04-24
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#


"""
TODO:
- remove commented useless code
- check if all imports necessary
- refactor hardcoded model weight checkpoints
"""

from PIL import Image
import numpy as np
from copy import deepcopy
from pathlib import Path
import yaml
import torch
import argparse
import pandas as pd

# from happypose.pose_estimators.cosypose.cosypose.datasets.datasets_cfg import make_scene_dataset, make_object_dataset
from happypose.toolbox.datasets.datasets_cfg import make_scene_dataset, make_object_dataset

# Pose estimator
from happypose.toolbox.lib3d.rigid_mesh_database import MeshDataBase
from happypose.pose_estimators.cosypose.cosypose.training.pose_models_cfg import create_model_refiner, create_model_coarse
from happypose.pose_estimators.cosypose.cosypose.training.pose_models_cfg import check_update_config as check_update_config_pose
# Detection
from happypose.pose_estimators.cosypose.cosypose.training.detector_models_cfg import create_model_detector
from happypose.pose_estimators.cosypose.cosypose.training.detector_models_cfg import check_update_config as check_update_config_detector
from happypose.pose_estimators.cosypose.cosypose.integrated.detector import Detector

from happypose.pose_estimators.cosypose.cosypose.evaluation.pred_runner.bop_predictions import BopPredictionRunner
from happypose.pose_estimators.cosypose.cosypose.integrated.pose_estimator import (
    PoseEstimator,
)
from happypose.pose_estimators.cosypose.cosypose.utils.distributed import get_tmp_dir, get_rank
from happypose.pose_estimators.cosypose.cosypose.utils.distributed import init_distributed_mode

from happypose.pose_estimators.cosypose.cosypose.config import EXP_DIR, RESULTS_DIR

from happypose.toolbox.renderer.panda3d_batch_renderer import Panda3dBatchRenderer


"""
def make_object_dataset(example_dir: Path) -> RigidObjectDataset:
    print(example_dir)
    rigid_objects = []
    mesh_units = "mm"
    object_dirs = (example_dir / "meshes").iterdir()
    print(object_dirs)
    for object_dir in object_dirs:
        print(object_dir)
        label = object_dir.name
        print(label)
        mesh_path = None
        for fn in object_dir.glob("*"):
            print("fn = ", fn)
            if fn.suffix in {".obj", ".ply"}:
                assert not mesh_path, f"there multiple meshes in the {label} directory"
                mesh_path = fn
        assert mesh_path, f"couldnt find a obj or ply mesh for {label}"
        rigid_objects.append(RigidObject(label=label, mesh_path=mesh_path, mesh_units=mesh_units))
        # TODO: fix mesh units
    rigid_object_dataset = RigidObjectDataset(rigid_objects)
    return rigid_object_dataset
example_dir = Path("/home/emaitre/cosypose/local_data/bop_datasets/ycbv/examples/")
"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CosyPoseWrapper:
    def __init__(self, dataset_name, n_workers=8, gpu_renderer=False) -> None:
        self.dataset_name = dataset_name
        super().__init__()
        self.detector, self.pose_predictor = self.get_model(dataset_name, n_workers)

    @staticmethod
    def load_detector(run_id, ds_name):
        run_dir = EXP_DIR / run_id
        # cfg = yaml.load((run_dir / 'config.yaml').read_text(), Loader=yaml.FullLoader)
        cfg = yaml.load((run_dir / 'config.yaml').read_text(), Loader=yaml.UnsafeLoader)
        cfg = check_update_config_detector(cfg)
        label_to_category_id = cfg.label_to_category_id
        model = create_model_detector(cfg, len(label_to_category_id))
        ckpt = torch.load(run_dir / 'checkpoint.pth.tar', map_location=device)
        ckpt = ckpt['state_dict']
        model.load_state_dict(ckpt)
        model = model.to(device).eval()
        model.cfg = cfg
        model.config = cfg
        model = Detector(model, ds_name)
        return model

    @staticmethod
    def load_pose_models(coarse_run_id, refiner_run_id, n_workers):
        run_dir = EXP_DIR / coarse_run_id

        # cfg = yaml.load((run_dir / 'config.yaml').read_text(), Loader=yaml.FullLoader)
        cfg = yaml.load((run_dir / 'config.yaml').read_text(), Loader=yaml.UnsafeLoader)
        cfg = check_update_config_pose(cfg)
        # object_ds = BOPObjectDataset(BOP_DS_DIR / 'tless/models_cad')
        #object_ds = make_object_dataset(cfg.object_ds_name)
        #mesh_db = MeshDataBase.from_object_ds(object_ds)
        #renderer = BulletBatchRenderer(object_set=cfg.urdf_ds_name, n_workers=n_workers, gpu_renderer=gpu_renderer)
        #
        
        object_dataset = make_object_dataset("tless.cad")
        mesh_db = MeshDataBase.from_object_ds(object_dataset)
        renderer = Panda3dBatchRenderer(object_dataset, n_workers=n_workers, preload_cache=False)
        mesh_db_batched = mesh_db.batched().to(device)

        def load_model(run_id):
            run_dir = EXP_DIR / run_id

            # cfg = yaml.load((run_dir / 'config.yaml').read_text(), Loader=yaml.FullLoader)
            cfg = yaml.load((run_dir / 'config.yaml').read_text(), Loader=yaml.UnsafeLoader)
            cfg = check_update_config_pose(cfg)
            if cfg.train_refiner:
                model = create_model_refiner(cfg, renderer=renderer, mesh_db=mesh_db_batched)
            else:
                model = create_model_coarse(cfg, renderer=renderer, mesh_db=mesh_db_batched)
            ckpt = torch.load(run_dir / 'checkpoint.pth.tar', map_location=device)
            ckpt = ckpt['state_dict']
            model.load_state_dict(ckpt)
            model = model.to(device).eval()
            model.cfg = cfg
            model.config = cfg
            return model
        
        coarse_model = load_model(coarse_run_id)
        refiner_model = load_model(refiner_run_id)
        return coarse_model, refiner_model, mesh_db

    @staticmethod
    def get_model(dataset_name, n_workers):
        # load models
        if dataset_name == 'tless':
            # TLESS setup
            # python -m cosypose.scripts.download --model=detector-bop-tless-pbr--873074
            # python -m cosypose.scripts.download --model=coarse-bop-tless-pbr--506801
            # python -m cosypose.scripts.download --model=refiner-bop-tless-pbr--233420
            detector_run_id = 'detector-bop-tless-pbr--873074'
            coarse_run_id = 'coarse-bop-tless-pbr--506801'
            refiner_run_id = 'refiner-bop-tless-pbr--233420'
        elif dataset_name == 'ycbv':
            # YCBV setup
            # python -m cosypose.scripts.download --model=detector-bop-ycbv-pbr--970850
            # python -m cosypose.scripts.download --model=coarse-bop-ycbv-pbr--724183
            # python -m cosypose.scripts.download --model=refiner-bop-ycbv-pbr--604090
            detector_run_id = 'detector-bop-ycbv-pbr--970850'
            coarse_run_id = 'coarse-bop-ycbv-pbr--724183'
            refiner_run_id = 'refiner-bop-ycbv-pbr--604090'
        else:
            raise ValueError(f"Not prepared for {dataset_name} dataset")
        detector = CosyPoseWrapper.load_detector(detector_run_id, dataset_name)
        coarse_model, refiner_model , mesh_db = CosyPoseWrapper.load_pose_models(
            coarse_run_id=coarse_run_id, refiner_run_id=refiner_run_id, n_workers=n_workers
        )

        pose_estimator = PoseEstimator(
            refiner_model=refiner_model,
            coarse_model=coarse_model,
            detector_model=detector,
        )
        return detector, pose_estimator

    def inference(self, observation, coarse_guess=None):
        detections = None
        run_detector = True
        if coarse_guess is None:
            final_preds, all_preds = self.pose_predictor.run_inference_pipeline(
                observation,
                detections=detections,
                run_detector=run_detector,
                data_TCO_init=None,
                n_coarse_iterations=1, n_refiner_iterations=4)
        else:
            final_preds, all_preds = self.pose_predictor.run_inference_pipeline(
                observation,
                detections=detections,
                run_detector=run_detector,
                data_TCO_init=None,
                n_coarse_iterations=0, n_refiner_iterations=4)
        print("inference successfully.")
        # result: this_batch_detections, final_preds
        return final_preds.cpu()
