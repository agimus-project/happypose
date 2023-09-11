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

# New CNOS detection from Nguyen drive
# CNOS_SUBMISSION_FILES = {
#     "ycbv": 'sam_pbr_ycbv.json', 
#     "lmo": 'sam_pbr_lmo.json', 
#     "tless": 'sam_pbr_tless.json', 
#     "tudl": 'sam_pbr_tudl.json', 
#     "icbin": 'sam_pbr_icbin.json', 
#     "itodd": 'sam_pbr_itodd.json', 
#     "hb": 'sam_pbr_hb.json', 
# }

# CNOS_SUBMISSION_FILES = {
#     "ycbv": 'fastSAM_pbr_ycbv.json', 
#     "lmo": 'fastSAM_pbr_lmo.json', 
#     "tless": 'fastSAM_pbr_tless.json', 
#     "tudl": 'fastSAM_pbr_tudl.json', 
#     "icbin": 'fastSAM_pbr_icbin.json', 
#     "itodd": 'fastSAM_pbr_itodd.json', 
#     "hb": 'fastSAM_pbr_hb.json', 
# }

# New official default detections -> fastSAM method, same as nguyen's drive
CNOS_SUBMISSION_FILES = {
    "ycbv": 'cnos-fastsam_ycbv-test_f4f2127c-6f59-447c-95b3-28e1e591f1a1.json', 
    "lmo": 'cnos-fastsam_lmo-test_3cb298ea-e2eb-4713-ae9e-5a7134c5da0f.json', 
    "tless": 'cnos-fastsam_tless-test_8ca61cb0-4472-4f11-bce7-1362a12d396f.json', 
    "tudl": 'cnos-fastsam_tudl-test_c48a2a95-1b41-4a51-9920-a667cb3d7149.json', 
    "icbin": 'cnos-fastsam_icbin-test_f21a9faf-7ef2-4325-885f-f4b6460f4432.json', 
    "itodd": 'cnos-fastsam_itodd-test_df32d45b-301c-4fc9-8769-797904dd9325.json', 
    "hb": 'cnos-fastsam_hb-test_db836947-020a-45bd-8ec5-c95560b68011.json', 
}



CNOS_SUBMISSION_DIR = os.environ.get('CNOS_SUBMISSION_DIR')
assert(CNOS_SUBMISSION_DIR is not None)
CNOS_SUBMISSION_DIR = Path(CNOS_SUBMISSION_DIR)

CNOS_SUBMISSION_PATHS = {ds_name: CNOS_SUBMISSION_DIR / fname for ds_name, fname in CNOS_SUBMISSION_FILES.items()}
# Check if all paths exist
assert( sum(p.exists() for p in CNOS_SUBMISSION_PATHS.values()) == len(CNOS_SUBMISSION_FILES))
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
        print("gt detections =\n", gt_detections)
        print("sam detections =\n", sam_detections)

        # TODO: this check could be done outside of run_inference_pipeline
        # and then only check if detections are None
        if self.inference_cfg.detection_type == "gt":
            detections = gt_detections
            run_detector = False
        elif self.inference_cfg.detection_type == "sam":
            # print("sam_detections =", sam_detections.bboxes)
            detections = sam_detections
            run_detector = False
        elif self.inference_cfg.detection_type == "detector":
            detections = None
            run_detector = True

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

        for k, v in all_preds.items():
            if "mask" in v.tensors:
                breakpoint()
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
            detections_path = CNOS_SUBMISSION_PATHS[ds_name]  

            """
            # dets_lst: list of dictionary, each element = detection of one object in an image
            $ df_all_dets[0].keys()
              > ['scene_id', 'image_id', 'category_id', 'bbox', 'score', 'time', 'segmentation']
            - For the evaluation of Megapose, we only need the 'scene_id', 'image_id', 'category_id', 'score', 'time' and 'bbox'
            - We also need need to change the format of bounding boxes as explained below 
            """
            dets_lst = []
            for det in json.loads(detections_path.read_text()):
                # We don't need the segmentation mask
                del det['segmentation']
                # Bounding box formats:
                # - CNOS/SAM baseline.json: [xmin, ymin, width, height]
                # - Megapose expects: [xmin, ymin, xmax, ymax]
                x, y, w, h = det['bbox']
                det['bbox'] = [float(v) for v in [x, y, x+w, y+h]]
                det['bbox_modal'] = det['bbox']

                # HACK: object models are same in lm and lmo -> obj labels start with 'lm'
                if ds_name == 'lmo':
                    ds_name = 'lm'

                det['label'] = '{}-obj_{}'.format(ds_name, str(det["category_id"]).zfill(6))
                
                dets_lst.append(det)

            df_all_dets = pd.DataFrame.from_records(dets_lst)

            df_targets = pd.read_json(self.scene_ds.ds_dir / "test_targets_bop19.json")

        for n, data in enumerate(tqdm(self.dataloader)):
            print('\n\n\n\n################')
            print(f'DATA FROM DATALOADER {self.scene_ds.ds_dir.name}', f'{n}/{len(self.dataloader)}')
            print('data["im_infos"]:', data['im_infos'])
            print('################')
            # data is a dict
            rgb = data["rgb"]
            depth = data["depth"]
            K = data["cameras"].K

            # ############ RUN ONLY BEGINNING OF DATASET
            # # if n > 0:
            if n > 10:
            # # if n != 582:
                 print('################')
                 print('Prediction runner SKIP')
                 print('################')
                 continue
            # ############ RUN ONLY BEGINNING OF DATASET

            # Dirty but avoids creating error when running with real detector
            dt_det = 0

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
                df_targets_scene_img = df_targets[(df_targets['scene_id'] == scene_id) & (df_targets['im_id'] == view_id)]

                dt_det += df_dets_scene_img.time.iloc[0]

                #################
                # Filter detections based on 2 criteria
                # - 1) Localization 6D task: we can assume that we know which object category and how many instances 
                # are present in the image
                obj_ids = df_targets_scene_img.obj_id.to_list()
                df_dets_scene_img_obj_filt = df_dets_scene_img[df_dets_scene_img['category_id'].isin(obj_ids)]
                # In case none of the detections category ids match the ones present in the scene,
                # keep only one detection to avoid downstream error
                if len(df_dets_scene_img_obj_filt) > 0:
                    df_dets_scene_img = df_dets_scene_img_obj_filt
                else:
                    df_dets_scene_img = df_dets_scene_img[:1]

                # TODO: retain only corresponding inst_count number for each detection category_id  

                # - 2) Retain detections with best cnos scores (kind of redundant with finalized 1) )
                # based on expected number of objects in the scene (from groundtruth)
                nb_gt_dets = df_targets_scene_img.inst_count.sum()
                
                # TODO: put that as a parameter somewhere?
                MARGIN = 1  # if 0, some images will have no detections
                K_MULT = 1
                nb_det = K_MULT*nb_gt_dets + MARGIN
                df_dets_scene_img = df_dets_scene_img.sort_values('score', ascending=False).head(nb_det)
                #################

                lst_dets_scene_img = df_dets_scene_img.to_dict('records')

                if len(lst_dets_scene_img) == 0:
                    raise(ValueError('lst_dets_scene_img empty!: ', f'scene_id: {scene_id}, image_id/view_id: {view_id}'))                

                # Do not forget the scores that are not present in object data
                scores = []
                list_object_data = []
                for det in lst_dets_scene_img:
                    list_object_data.append(ObjectData.from_json(det))
                    scores.append(det['score'])
                sam_detections = make_detections_from_object_data(list_object_data).to(device)
                sam_detections.infos['score'] = scores

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

            total_duration = duration + dt_det

            # Add metadata to the predictions for later evaluation
            for k, v in all_preds.items():
                v.infos['time'] = total_duration
                v.infos['scene_id'] = scene_id
                v.infos['view_id'] = view_id
                predictions_list[k].append(v)

        # Concatenate the lists of PandasTensorCollections
        predictions = dict()
        for k, v in predictions_list.items():
            predictions[k] = tc.concatenate(v)

        return predictions
