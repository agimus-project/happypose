"""
Copyright (c) 2022 Inria & NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


# Standard Library
import time
from collections import defaultdict
from typing import Dict, Optional
from pathlib import Path


# Third Party
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# MegaPose
import happypose.pose_estimators.megapose.src.megapose
import happypose.toolbox.utils.tensor_collection as tc
from happypose.pose_estimators.megapose.src.megapose.inference.pose_estimator import (
    PoseEstimator,
)
from happypose.pose_estimators.megapose.src.megapose.inference.types import (
    DetectionsType,
    InferenceConfig,
    ObservationTensor,
    PoseEstimatesType,
)
from happypose.pose_estimators.megapose.src.megapose.config import (
    BOP_DS_DIR
)

from happypose.pose_estimators.megapose.src.megapose.training.utils import CudaTimer
from happypose.toolbox.datasets.samplers import DistributedSceneSampler
from happypose.toolbox.datasets.scene_dataset import SceneDataset, SceneObservation, ObjectData
from happypose.toolbox.utils.distributed import get_rank, get_tmp_dir, get_world_size
from happypose.toolbox.utils.logging import get_logger


# Temporary
from happypose.toolbox.inference.utils import make_detections_from_object_data
import pandas as pd
import json

logger = get_logger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



##################################
##################################
import os
# ycbv: ok but missing 3 images
# lmo: ok but detection labels are actually "lm-obj_*"
# tless: pbe in make_object_dataset -> input is tless.panda3d but case not handled
# tudl: Ok, to complete
# icbin: Ok, to complete
# itodd: NOK
# hb: NOK

# OFFICIAL BOP initial CNOS submission, missing some detections
# CNOS_SUBMISSION_FILES = {
#     "ycbv": 'baseline-sam-dinov2-blenderproc4bop_ycbv-test_a491e9fe-1137-4585-9c80-0a2056a3eb9c.json',
#     "lmo": 'baseline-sam-dinov2-blenderproc4bop_lmo-test_2f321533-59ae-4541-b65e-6b4e4fb9d391.json',
#     "tless": 'baseline-sam-dinov2-blenderproc4bop_tless-test_3305b238-3d93-4954-81ba-3ff3786265d9.json',
#     "tudl": 'baseline-sam-dinov2-blenderproc4bop_tudl-test_c6cd05c1-89a1-4fe5-88b9-c1b57ef15694.json',
#     "icbin": 'baseline-sam-dinov2-blenderproc4bop_icbin-test_f58b6868-7e70-4ab2-9332-65220849f8c1.json',
#     "itodd": 'baseline-sam-dinov2-blenderproc4bop_itodd-test_82442e08-1e79-4f54-8e88-7ad6b986dd96.json',
#     "hb": 'baseline-sam-dinov2-blenderproc4bop_hb-test_f32286f9-05f5-4123-862f-18f00e67e685.json',
# }


# New CNOS detection from Nguyen drive
CNOS_SUBMISSION_FILES = {
    "ycbv": 'sam_pbr_ycbv.json', 
    "lmo": 'sam_pbr_lmo.json', 
    "tless": 'sam_pbr_tless.json', 
    "tudl": 'sam_pbr_tudl.json', 
    "icbin": 'sam_pbr_icbin.json', 
    "itodd": 'sam_pbr_itodd.json', 
    "hb": 'sam_pbr_hb.json', 
}


CNOS_SUBMISSION_DIR = os.environ.get('CNOS_SUBMISSION_DIR')
assert(CNOS_SUBMISSION_DIR is not None)
CNOS_SUBMISSION_DIR = Path(CNOS_SUBMISSION_DIR)
##################################
##################################



class PredictionRunner:
    def __init__(
        self,
        scene_ds: SceneDataset,
        inference_cfg: InferenceConfig,
        batch_size: int = 1,
        n_workers: int = 4,
    ) -> None:

        self.inference_cfg = inference_cfg
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.tmp_dir = get_tmp_dir()

        sampler = DistributedSceneSampler(scene_ds, num_replicas=self.world_size, rank=self.rank)
        self.sampler = sampler
        self.scene_ds = scene_ds
        dataloader = DataLoader(
            scene_ds,
            batch_size=batch_size,
            num_workers=n_workers,
            sampler=sampler,
            collate_fn=SceneObservation.collate_fn,
        )

        self.batch_size = batch_size
        self.load_depth = scene_ds.load_depth
        self.dataloader = dataloader

    def run_inference_pipeline(
        self,
        pose_estimator: PoseEstimator,
        obs_tensor: ObservationTensor,
        gt_detections: DetectionsType,
        sam_detections: DetectionsType,
        initial_estimates: Optional[PoseEstimatesType] = None,
    ) -> Dict[str, PoseEstimatesType]:
        """Runs inference pipeline, extracts the results.

        Returns: A dict with keys
            - 'final': final preds
            - 'refiner/final': preds at final refiner iteration (before depth refinement)
            - 'depth_refinement': preds after depth refinement.


        """
        print("gt detections =", gt_detections)
        print("sam detections =", sam_detections)



        if self.inference_cfg.detection_type == "gt":
            detections = gt_detections
            print("gt detections =", gt_detections.bboxes)
            run_detector = False
        elif self.inference_cfg.detection_type == "detector":
            detections = None
            run_detector = True
        elif self.inference_cfg.detection_type == "sam":
            print("sam_detections =", sam_detections.bboxes)
            detections = sam_detections
            run_detector = False
        else:
            raise ValueError(f"Unknown detection type {self.inference_cfg.detection_type}")


        coarse_estimates = None
        if self.inference_cfg.coarse_estimation_type == "external":
            # TODO (ylabbe): This is hacky, clean this for modelnet eval.
            coarse_estimates = initial_estimates
            coarse_estimates = happypose.toolbox.inference.utils.add_instance_id(coarse_estimates)
            coarse_estimates.infos["instance_id"] = 0
            run_detector = False

        t = time.time()
        preds, extra_data = pose_estimator.run_inference_pipeline(
            obs_tensor,
            detections=detections,
            run_detector=run_detector,
            coarse_estimates=coarse_estimates,
            n_refiner_iterations=self.inference_cfg.n_refiner_iterations,
            n_pose_hypotheses=self.inference_cfg.n_pose_hypotheses,
            run_depth_refiner=self.inference_cfg.run_depth_refiner,
            bsz_images=self.inference_cfg.bsz_images,
            bsz_objects=self.inference_cfg.bsz_objects,
        )
        elapsed = time.time() - t

        # TODO (lmanuelli): Process this into a dict with keys like
        # - 'refiner/iteration=1`
        # - 'refiner/iteration=5`
        # - `depth_refiner`
        # Note: Since we support multi-hypotheses we need to potentially
        # go back and extract out the 'refiner/iteration=1`, `refiner/iteration=5` things for the ones that were actually the highest scoring at the end.

        all_preds = dict()
        data_TCO_refiner = extra_data["refiner"]["preds"]

        all_preds = {
            "final": preds,
            f"refiner/iteration={self.inference_cfg.n_refiner_iterations}": data_TCO_refiner,
            "refiner/final": data_TCO_refiner,
            "coarse": extra_data["coarse"]["preds"],
        }

        if self.inference_cfg.run_depth_refiner:
            all_preds[f"depth_refiner"] = extra_data["depth_refiner"]["preds"]

        # Remove any mask tensors
        for k, v in all_preds.items():
            v.infos["scene_id"] = np.unique(gt_detections.infos["scene_id"]).item()
            v.infos["view_id"] = np.unique(gt_detections.infos["view_id"]).item()
            if "mask" in v.tensors:
                v.delete_tensor("mask")

        return all_preds

    def get_predictions(self, pose_estimator: PoseEstimator) -> Dict[str, PoseEstimatesType]:
        """Runs predictions

        Returns: A dict with keys
            - 'refiner/iteration=1`
            - 'refiner/iteration=5`
            - 'depth_refiner'

            With the predictions at the various settings/iterations.


        """

        predictions_list = defaultdict(list)

        ######
        # This section opens the detections stored in "baseline.json"
        # format it and store it in a dataframe that will be accessed later
        ######
        # Temporary solution
        if self.inference_cfg.detection_type == "sam":
            ds_name = self.scene_ds.ds_dir.name
            detections_path = CNOS_SUBMISSION_DIR / CNOS_SUBMISSION_FILES[ds_name]  

            """
            # dets_lst: list of dictionary, each element = detection of one object in an image
            $ df_all_dets[0].keys()
              > ['scene_id', 'image_id', 'category_id', 'bbox', 'score', 'time', 'segmentation']
            - For the evaluation of Megapose, we only need the 'scene_id', 'image_id', 'category_id', 'score' and 'bbox'
            - We also need need to change the format of bounding boxes as explained below 
            """
            dets_lst = []
            for det_cnos in json.loads(detections_path.read_text()):
                # Note: score is used at evaluation time only
                det = {k: det_cnos[k] for k in ['scene_id', 'image_id', 'category_id', 'score']}
                # Bounding box formats:
                # - CNOS/SAM baseline.json: [xmin, ymin, width, height]
                # - Megapose expects: [xmin, ymin, xmax, ymax]
                x, y, w, h = det_cnos['bbox']
                det['bbox'] = [float(v) for v in [x, y, x+w, y+h]]
                det['bbox_modal'] = det['bbox']

                if ds_name == 'lmo':
                    ds_name = 'lm'

                det['label'] = '{}-obj_{}'.format(ds_name, str(det["category_id"]).zfill(6))
                
                dets_lst.append(det)

            df_all_dets = pd.DataFrame.from_records(dets_lst)


            targets = pd.read_json(self.scene_ds.ds_dir / "test_targets_bop19.json")


        for n, data in enumerate(tqdm(self.dataloader)):
            print('\n\n\n\n################')
            print('DATA FROM DATALOADER', f'{n}/{len(self.dataloader)}')
            print('data["im_infos"]:', data['im_infos'])
            print('################')
            # data is a dict
            rgb = data["rgb"]
            depth = data["depth"]
            K = data["cameras"].K


            ############# RUN ONLY BEGINNING OF DATASET
            # if n > 0:
            # if n < 150:
            #     print('################')
            #     print('Prediction runner SKIP')
            #     print('################')
            #     continue
            ############# RUN ONLY BEGINNING OF DATASET
            
            ######
            # Filter the dataframe according to scene id and view id
            # Transform the data in ObjectData and then Detections
            ######
            # Temporary solution
            if self.inference_cfg.detection_type == "sam":
                # We assume a unique image ("view") associated with a unique scene_id is 
                im_info = data['im_infos'][0]
                scene_id, view_id = im_info['scene_id'], im_info['view_id']

                df_dets_scene_img = df_all_dets.loc[(df_all_dets['scene_id'] == scene_id) & (df_all_dets['image_id'] == view_id)]

                #################
                # Retain detections with best cnos scores 
                # based on expected number of objects in the scene
                # 
                # This should not change the resulting evaluation of Megapose as it
                # is based on the evaluation of the K "best" predictions but should
                # reduce dramatically the computational burden (some images have ~ 100 detection propositions)  
                # df_all_dets
                # __import__("IPython").embed()
                nb_gt_dets = len(targets[(targets['scene_id'] == scene_id) & (targets['im_id'] == view_id)])
                margin = 0  # consider a few other detections to prevent false negatives
                df_dets_scene_img = df_dets_scene_img.sort_values('score', ascending=False).head(nb_gt_dets+margin)
                #################

                lst_dets_scene_img = df_dets_scene_img.to_dict('records')

                if (len(lst_dets_scene_img)) == 0:
                    raise(ValueError('lst_dets_scene_img empty!: ', f'scene_id: {scene_id}, image_id/view_id: {view_id}'))                

                # Do not forget the scores that are not present in object data
                scores = []
                list_object_data = []
                for det in lst_dets_scene_img:
                    list_object_data.append(ObjectData.from_json(det))
                    scores.append(det['score'])
                sam_detections = make_detections_from_object_data(list_object_data).to(device)
                sam_detections.infos['score'] = scores

                # __import__("IPython").embed()
                print("sam_detections =\n", sam_detections)
            else:
                sam_detections = None
            gt_detections = data["gt_detections"].cuda()
            initial_data = None
            if data["initial_data"]:
                initial_data = data["initial_data"].cuda()

            obs_tensor = ObservationTensor.from_torch_batched(rgb, depth, K)
            obs_tensor = obs_tensor.cuda()

            # GPU warmup for timing
            if n == 0:
                with torch.no_grad():
                    self.run_inference_pipeline(
                        pose_estimator, obs_tensor, gt_detections, sam_detections, initial_estimates=initial_data
                    )

            cuda_timer = CudaTimer()
            cuda_timer.start()
            with torch.no_grad():
                all_preds = self.run_inference_pipeline(
                    pose_estimator, obs_tensor, gt_detections, sam_detections, initial_estimates=initial_data
                )
            cuda_timer.end()
            duration = cuda_timer.elapsed()

            for k, v in all_preds.items():
                predictions_list[k].append(v)

        # Concatenate the lists of PandasTensorCollections
        predictions = dict()
        for k, v in predictions_list.items():
            predictions[k] = tc.concatenate(v)

        return predictions
