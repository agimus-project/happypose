import argparse
import shutil
import subprocess

import torch
from bop_toolkit_lib import inout
from tqdm import tqdm

from happypose.pose_estimators.cosypose.cosypose.config import (
    BOP_POSE_EVAL_SCRIPT_NAME,
    RESULTS_DIR,
)


def main():
    parser = argparse.ArgumentParser("Bop evaluation")
    parser.add_argument("--result_id", default="", type=str)
    parser.add_argument("--method", default="", type=str)
    parser.add_argument("--dataset", default="", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--csv_path", default="", type=str)
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--convert_only", action="store_true")
    args = parser.parse_args()
    run_evaluation(args)


def run_evaluation(args):
    results_path = (
        RESULTS_DIR / args.result_id / f"dataset={args.dataset}" / "results.pth.tar"
    )
    csv_path = args.csv_path
    convert_results(results_path, csv_path, method=args.method)

    if not args.dummy:
        shutil.copy(
            csv_path,
            RESULTS_DIR / args.result_id / f"dataset={args.dataset}" / csv_path.name,
        )

    if not args.convert_only:
        run_bop_evaluation(csv_path)
    return csv_path


def convert_results(results_path, out_csv_path, method):
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
        score = row.score
        time = row.time
        pred = {
            "scene_id": row.scene_id,
            "im_id": row.view_id,
            "obj_id": obj_id,
            "score": score,
            "t": t,
            "R": R,
            "time": time,
        }
        preds.append(pred)
    print("Wrote:", out_csv_path)
    inout.save_bop_results(out_csv_path, preds)
    return out_csv_path


def run_bop_evaluation(filename):
    subprocess.call(
        [
            BOP_POSE_EVAL_SCRIPT_NAME,
            "--renderer_type",
            "python",
            "--result_filenames",
            filename,
        ],
    )


if __name__ == "__main__":
    main()
