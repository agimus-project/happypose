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
from pathlib import Path

# Third Party
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# MegaPose
from happypose.pose_estimators.megapose.config import (
    BOP_DETECTION_EVAL_SCRIPT_NAME,
    BOP_POSE_EVAL_SCRIPT_NAME,
    LOCAL_DATA_DIR,
    PROJECT_DIR,
)
from happypose.pose_estimators.megapose.evaluation.eval_config import BOPEvalConfig
from happypose.toolbox.datasets.scene_dataset import ObjectData
from happypose.toolbox.inference.utils import make_detections_from_object_data
from happypose.toolbox.utils.tensor_collection import (
    PandasTensorCollection,
    filter_top_pose_estimates,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Third Party
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
    results_path: Path, out_csv_path: Path, method: str, use_pose_score: bool = True
):
    predictions = torch.load(results_path)["predictions"]
    predictions = predictions[method]
    if method == "coarse":
        predictions = get_best_coarse_predictions(predictions)

    print("Predictions from:", results_path)
    print("Method:", method)
    print("Number of predictions: ", len(predictions))

    preds = []
    for n in tqdm(range(len(predictions))):
        TCO_n = predictions.poses[n]
        t = TCO_n[:3, -1] * 1e3  # m -> mm conversion
        R = TCO_n[:3, :3]
        row = predictions.infos.iloc[n]
        print("row =", row)
        obj_id = int(row.label.split("_")[-1])
        if use_pose_score:
            score = row["pose_score"]
        else:
            score = row["score"]
        if "time" in row:
            time = row["time"]
        else:
            time = -1
        pred = dict(
            scene_id=row["scene_id"],
            im_id=row["view_id"],
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


def get_best_coarse_predictions(coarse_preds: PandasTensorCollection):
    group_cols = ["scene_id", "view_id", "label", "instance_id"]
    coarse_preds = filter_top_pose_estimates(
        coarse_preds,
        top_K=1,
        group_cols=group_cols,
        filter_field="coarse_score",
        ascending=False,
    )
    coarse_preds.infos = coarse_preds.infos.rename(
        columns={"coarse_score": "pose_score"}
    )
    return coarse_preds


def _run_bop_evaluation(filename, eval_dir, eval_detection=False, dummy=False):
    myenv = os.environ.copy()
    myenv["BOP_DATASETS_PATH"] = str(LOCAL_DATA_DIR / "bop_datasets")
    myenv["BOP_RESULTS_PATH"] = str(eval_dir)
    myenv["BOP_EVAL_PATH"] = str(eval_dir)
    renderer_type = "vispy"  # other options: 'cpp', 'python'
    if eval_detection:
        cmd = [
            BOP_DETECTION_EVAL_SCRIPT_NAME,
            "--result_filenames",
            filename,
        ]
    else:
        cmd = [
            BOP_POSE_EVAL_SCRIPT_NAME,
            "--result_filenames",
            filename,
            "--renderer_type",
            renderer_type,
        ]
    subprocess.call(cmd, env=myenv)


def run_evaluation(cfg: BOPEvalConfig) -> None:
    """Runs the bop evaluation for the given setting."""
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
        convert_results_to_bop(results_path, csv_path, cfg.method, use_pose_score=False)

        if not cfg.convert_only:
            _run_bop_evaluation(csv_path, cfg.eval_dir, eval_detection=False)
        scores_pose_path = (
            eval_dir / csv_path.with_suffix("").name / "scores_bop19.json"
        )

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


def load_external_detections(scene_ds_dir: Path):
    """
    Loads external detections
    """
    ds_name = scene_ds_dir.name

    bop_detections_paths = get_external_detections_paths()
    detections_path = bop_detections_paths[ds_name]

    dets_lst = []
    for det in json.loads(detections_path.read_text()):
        det = format_det_bop2megapose(det, ds_name)
        dets_lst.append(det)

    df_all_dets = pd.DataFrame.from_records(dets_lst)
    df_targets = pd.read_json(scene_ds_dir / "test_targets_bop19.json")
    return df_all_dets, df_targets


def get_external_detections_paths():
    EXTERNAL_DETECTIONS_DIR = os.environ.get("EXTERNAL_DETECTIONS_DIR")
    assert EXTERNAL_DETECTIONS_DIR is not None
    EXTERNAL_DETECTIONS_DIR = Path(EXTERNAL_DETECTIONS_DIR)

    files_name_path = EXTERNAL_DETECTIONS_DIR / "bop_detections_filenames.json"
    try:
        bop_detections_filenames = json.loads(files_name_path.read_text())
    except json.decoder.JSONDecodeError as e:
        print("Check json formatting {files_name_path.as_posix()}")
        raise e
    bop_detections_paths = {
        ds_name: EXTERNAL_DETECTIONS_DIR / fname
        for ds_name, fname in bop_detections_filenames.items()
    }

    return bop_detections_paths


def format_det_bop2megapose(det, ds_name):
    # Segmentation mask not needed
    if "segmentation" in det:
        del det["segmentation"]
    # Bounding box formats:
    # - BOP format: [xmin, ymin, width, height]
    # - Megapose expects: [xmin, ymin, xmax, ymax]
    x, y, w, h = det["bbox"]
    x1, y1, x2, y2 = x, y, x + w, y + h
    det["bbox"] = [float(v) for v in [x1, y1, x2, y2]]
    det["bbox_modal"] = det["bbox"]

    # HACK: object models are same in lm and lmo
    # -> lmo obj labels actually start with 'lm'
    if ds_name == "lmo":
        ds_name = "lm"

    det["label"] = "{}-obj_{}".format(ds_name, str(det["category_id"]).zfill(6))

    return det


def filter_detections_scene_view(scene_id, view_id, df_all_dets, df_targets):
    """
    Retrieve detections of scene/view id pair and filter using bop targets.
    """
    df_dets_scene_img = df_all_dets.loc[
        (df_all_dets["scene_id"] == scene_id) & (df_all_dets["image_id"] == view_id)
    ]
    df_targets_scene_img = df_targets[
        (df_targets["scene_id"] == scene_id) & (df_targets["im_id"] == view_id)
    ]

    df_dets_scene_img = keep_best_detections(df_dets_scene_img, df_targets_scene_img)

    # Keep only best detections for objects ("targets") given in bop target file
    lst_dets_scene_img = df_dets_scene_img.to_dict("records")

    # Do not forget the scores that are not present in object img_data
    scores, list_object_data = [], []
    for det in lst_dets_scene_img:
        list_object_data.append(ObjectData.from_json(det))
        scores.append(det["score"])
    detections = make_detections_from_object_data(list_object_data).to(device)
    detections.infos["score"] = scores
    detections.infos["time"] = df_dets_scene_img.time.iloc[0]
    return detections


def keep_best_detections(df_dets_scene_img, df_targets_scene_img):
    lst_df_target = []
    nb_targets = len(df_targets_scene_img)
    for it in range(nb_targets):
        target = df_targets_scene_img.iloc[it]
        n_best = target.inst_count
        df_filt_target = df_dets_scene_img[
            df_dets_scene_img["category_id"] == target.obj_id
        ].sort_values("score", ascending=False)[:n_best]
        if len(df_filt_target) > 0:
            lst_df_target.append(df_filt_target)

    # if missing dets, keep only one detection to avoid downstream error
    df_dets_scene_img = (
        pd.concat(lst_df_target) if len(lst_df_target) > 0 else df_dets_scene_img[:1]
    )

    return df_dets_scene_img


if __name__ == "__main__":
    main()
