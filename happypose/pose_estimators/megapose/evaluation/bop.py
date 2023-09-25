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
import argparse
import importlib
import json
import os
import subprocess
import sys
import pandas as pd
from pathlib import Path

# Third Party
import numpy as np
import torch
from tqdm import tqdm

# MegaPose
from happypose.pose_estimators.megapose.config import BOP_TOOLKIT_DIR, LOCAL_DATA_DIR, PROJECT_DIR
from happypose.pose_estimators.megapose.evaluation.eval_config import BOPEvalConfig
from happypose.toolbox.datasets.scene_dataset import ObjectData
from happypose.toolbox.inference.utils import make_detections_from_object_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Note we are actually using the bop_toolkit_lib that is directly conda installed
# inside the docker image. This is just to access the scripts.
POSE_EVAL_SCRIPT_PATH = BOP_TOOLKIT_DIR / "scripts/eval_bop19_pose.py"
DETECTION_EVAL_SCRIPT_PATH = BOP_TOOLKIT_DIR / "scripts/eval_bop22_coco.py"
DUMMY_EVAL_SCRIPT_PATH = BOP_TOOLKIT_DIR / "scripts/eval_bop19_dummy.py"


##################################
##################################
import os

# Official Task 4 detections (CNOS fastSAM)
EXTERNAL_DETECTIONS_FILES = {
    "ycbv": 'cnos-fastsam_ycbv-test_f4f2127c-6f59-447c-95b3-28e1e591f1a1.json', 
    "lmo": 'cnos-fastsam_lmo-test_3cb298ea-e2eb-4713-ae9e-5a7134c5da0f.json', 
    "tless": 'cnos-fastsam_tless-test_8ca61cb0-4472-4f11-bce7-1362a12d396f.json', 
    "tudl": 'cnos-fastsam_tudl-test_c48a2a95-1b41-4a51-9920-a667cb3d7149.json', 
    "icbin": 'cnos-fastsam_icbin-test_f21a9faf-7ef2-4325-885f-f4b6460f4432.json', 
    "itodd": 'cnos-fastsam_itodd-test_df32d45b-301c-4fc9-8769-797904dd9325.json', 
    "hb": 'cnos-fastsam_hb-test_db836947-020a-45bd-8ec5-c95560b68011.json', 
}


# # Official Task 1 detections (gdrnppdet-pbrreal)
# EXTERNAL_DETECTIONS_FILES = {
#     "ycbv": 'gdrnppdet-pbrreal_ycbv-test_abe6c5f1-cb26-4bbd-addc-bb76dd722a96.json', 
#     "lmo": 'gdrnppdet-pbrreal_lmo-test_202a2f15-cbd0-49df-90de-650428c6d157.json', 
#     "tless": 'gdrnppdet-pbrreal_tless-test_e112ecb4-7f56-4107-8a21-945bc7661267.json', 
#     "tudl": 'gdrnppdet-pbrreal_tudl-test_66fd26f1-bebf-493b-a42a-d71e8d10c479.json', 
#     "icbin": 'gdrnppdet-pbrreal_icbin-test_a46668ed-f76b-40ca-9954-708b198c2ab0.json', 
#     "itodd": 'gdrnppdet-pbrreal_itodd-test_9559c160-9507-4d09-94a5-ef0d6e8f22ce.json', 
#     "hb": 'gdrnppdet-pbrreal_hb-test_94485f5a-98ea-48f1-9472-06f4ceecad41.json', 
# }


EXTERNAL_DETECTIONS_DIR = os.environ.get('EXTERNAL_DETECTIONS_DIR')
assert(EXTERNAL_DETECTIONS_DIR is not None)
EXTERNAL_DETECTIONS_DIR = Path(EXTERNAL_DETECTIONS_DIR)

CNOS_SUBMISSION_PATHS = {ds_name: EXTERNAL_DETECTIONS_DIR / fname for ds_name, fname in EXTERNAL_DETECTIONS_FILES.items()}
# Check if all paths exist
assert( sum(p.exists() for p in CNOS_SUBMISSION_PATHS.values()) == len(EXTERNAL_DETECTIONS_FILES))
##################################
##################################


# Third Party
import bop_toolkit_lib
from bop_toolkit_lib import inout  # noqa


def main():
    parser = argparse.ArgumentParser("Bop evaluation")
    parser.add_argument("--results_path", default="", type=str)
    parser.add_argument("--eval_dir", default="", type=str)
    parser.add_argument("--dataset", default="", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--method", default="", type=str)
    parser.add_argument("--detection-method", default="", type=str)
    parser.add_argument("--csv_path", default="", type=str)
    parser.add_argument("--convert-only", action="store_true")
    parser.add_argument("--dummy", action="store_true")
    args = parser.parse_args()
    run_evaluation(args)


def convert_results_to_coco(results_path, out_json_path, detection_method):
    sys.path = [p for p in sys.path if "bop_toolkit" not in str(p)]
    TOOLKIT_MASTER_DIR = Path(PROJECT_DIR).parent / "bop_toolkit_master"
    sys.path.append(TOOLKIT_MASTER_DIR.as_posix())
    importlib.reload(sys.modules["bop_toolkit_lib"])
    # Third Party
    from bop_toolkit_lib.pycoco_utils import binary_mask_to_polygon

    results = torch.load(results_path)
    predictions = results["predictions"][detection_method]
    print("Detections from:", results_path)
    print("Detection method:", detection_method)
    print("Number of detections: ", len(predictions))

    infos = []
    for n in tqdm(range(len(predictions))):
        row = predictions.infos.iloc[n]
        x1, y1, x2, y2 = predictions.bboxes[n].tolist()
        x, y, w, h = x1, y1, (x2 - x1), (y2 - y1)
        score = row.score
        category_id = int(row.label.split("_")[-1])
        mask = predictions.masks[n].numpy().astype(np.uint8)
        rle = binary_mask_to_polygon(mask)
        info = dict(
            scene_id=int(row.scene_id),
            view_id=int(row.view_id),
            category_id=category_id,
            bbox=[x, y, w, h],
            score=score,
            segmentation=rle,
        )
        infos.append(info)
    Path(out_json_path).write_text(json.dumps(infos))
    return


def convert_results_to_bop(
    results_path: Path, out_csv_path: Path, method: str,
    use_pose_score: bool = True
):
    predictions = torch.load(results_path)["predictions"]
    predictions = predictions[method]
    print("Predictions from:", results_path)
    print("Method:", method)
    print("Number of predictions: ", len(predictions))

    preds = []
    for n in tqdm(range(len(predictions))):
        TCO_n = predictions.poses[n]
        t = TCO_n[:3, -1] * 1e3  # m -> mm conversion
        R = TCO_n[:3, :3]
        row = predictions.infos.iloc[n]
        obj_id = int(row.label.split("_")[-1])
        if use_pose_score:
            score = row.pose_score
        else:
            score = row.score
        if "time" in row:
            time = row.time
        else:
            time = -1
        pred = dict(
            scene_id=row.scene_id,
            im_id=row.view_id,
            obj_id=obj_id,
            score=score,
            t=t,
            R=R,
            time=time,
        )
        preds.append(pred)
    print("Wrote:", out_csv_path)
    Path(out_csv_path).parent.mkdir(exist_ok=True)
    inout.save_bop_results(out_csv_path, preds)
    return out_csv_path

def _run_bop_evaluation(filename, eval_dir, eval_detection=False, dummy=False):
    myenv = os.environ.copy()
    myenv["PYTHONPATH"] = BOP_TOOLKIT_DIR.as_posix()
    ld_library_path = os.environ['LD_LIBRARY_PATH']
    conda_prefix = os.environ['CONDA_PREFIX']
    myenv["LD_LIBRARY_PATH"] = f'{conda_prefix}/lib:{ld_library_path}'
    myenv["BOP_DATASETS_PATH"] = str(LOCAL_DATA_DIR / "bop_datasets")
    myenv["BOP_RESULTS_PATH"] = str(eval_dir)
    myenv["BOP_EVAL_PATH"] = str(eval_dir)
    renderer_type = 'vispy'  # other options: 'cpp', 'python'
    if dummy:
        cmd = [
            "python",
            str(DUMMY_EVAL_SCRIPT_PATH),
            "--renderer_type",
            renderer_type,
            "--result_filenames",
            filename,
        ]
    else:
        if eval_detection:
            cmd = [
                "python",
                str(DETECTION_EVAL_SCRIPT_PATH),
                "--result_filenames",
                filename,
            ]
        else:
            cmd = [
                "python",
                str(POSE_EVAL_SCRIPT_PATH),
                "--result_filenames",
                filename,
                "--renderer_type",
                renderer_type,
            ]
    subprocess.call(cmd, env=myenv, cwd=BOP_TOOLKIT_DIR.as_posix())


def run_evaluation(cfg: BOPEvalConfig) -> None:
    """Runs the bop evaluation for the given setting."""
    print(cfg)
    results_path = Path(cfg.results_path)
    eval_dir = Path(cfg.eval_dir)

    if cfg.dataset == "hb.bop19" and cfg.split == "test":
        cfg.convert_only = True
    if cfg.dataset == "itodd.bop19" and cfg.split == "test":
        cfg.convert_only = True

    scores_pose_path = None
    if cfg.method is not None:
        method = cfg.method.replace("/", "-")
        method = method.replace("_", "-")

        # The csv file needs naming like <anything>_ycbv-test.csv since
        # this is what is expected by bop_toolkit_lib
        csv_path = eval_dir / f"{method}_{cfg.dataset.split('.')[0]}-{cfg.split}.csv"

        # pose scores give better AR scores in general
        convert_results_to_bop(results_path, csv_path, cfg.method, use_pose_score=cfg.use_post_score)

        if not cfg.convert_only:
            _run_bop_evaluation(csv_path, cfg.eval_dir, eval_detection=False)
        scores_pose_path = eval_dir / csv_path.with_suffix("").name / "scores_bop19.json"

    scores_detection_path = None
    if cfg.detection_method is not None:
        raise NotImplementedError
        method = cfg.detection_method.replace("/", "-")
        method = method.replace("_", "-")
        json_path = eval_dir / f"{method}_{cfg.dataset}-{cfg.split}.json"
        convert_results_to_coco(results_path, json_path, cfg.detection_method)
        if not cfg.convert_only:
            _run_bop_evaluation(json_path, cfg.eval_dir, eval_detection=True)
        scores_detection_path = (
            eval_dir / csv_path.with_suffix("").name / "scores_bop22_coco_bbox.json"
        )

    return scores_pose_path, scores_detection_path

def load_sam_predictions(ds_dir_name, scene_ds_dir):
    ds_name = ds_dir_name
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
        # We don't need the segmentation mask (not always present in the submissions)
        if 'segmentation' in det:
            del det['segmentation']
        # Bounding box formats:
        # - BOP format: [xmin, ymin, width, height]
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

    df_targets = pd.read_json(scene_ds_dir / "test_targets_bop19.json")

    return df_all_dets, df_targets

def get_sam_detections(data, df_all_dets, df_targets, dt_det):
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
    return sam_detections


if __name__ == "__main__":
    main()
