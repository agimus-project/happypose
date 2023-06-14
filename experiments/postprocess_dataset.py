import os
import hydra
from dataclasses import dataclass
import submitit
import numpy as np
import json
import pathlib as p
from bop_toolkit_lib.dataset.bop_imagewise import io_load_gt
from bop_toolkit_lib import inout

from job_runner.utils import make_submitit_executor
from job_runner.configs import RunnerConfig
from hydra.core.config_store import ConfigStore


def process_key(key, ds_dir, stoi_obj, out_dir):
    is_valid = True
    extensions = [
        "gt.json",
        "gt_info.json",
        "camera.json",
        "rgb.jpg",
        "depth.png",
        "mask.json",
        "mask_visib.json",
    ]
    for ext in extensions:
        if not (ds_dir / f"{key}.{ext}").exists():
            is_valid = False

    if not is_valid:
        for ext in extensions:
            path = ds_dir / f"{key}.{ext}"
            if path.exists():
                print("unlink", path)
                path.unlink()

    if is_valid:
        out_dir.mkdir(exist_ok=True)
        with open(ds_dir / f"{key}.gt.json", "r") as f:
            gt = io_load_gt(f)
        for gt_n in gt:
            gt_n["obj_id"] = stoi_obj[gt_n["obj_id"]]
        gt = [inout._gt_as_json(d) for d in gt]
        inout.save_json(out_dir / f"{key}.gt.json", gt)
    return


def process_keys(keys, *args):
    for key in keys:
        process_key(key, *args)


def load_stoi(ds_dir):
    p = ds_dir / "gso_models.json"
    if not p.exists():
        p = ds_dir / "shapenet_models.json"
    assert p.exists()
    infos = json.load(open(p, "r"))
    stoi = dict()
    for info in infos:
        if "gso_id" in info:
            stoi[f"gso_{info['gso_id']}"] = info["obj_id"]
        else:
            k = f"shapenet_{info['shapenet_synset_id']}_{info['shapenet_source_id']}"
            stoi[k] = info["obj_id"]
    return stoi


@dataclass
class Config:
    n_jobs: int
    ds_dir: str
    runner: RunnerConfig


cs = ConfigStore.instance()
cs.store(
    group="run_ds_postproc",
    name="base_run_ds_postproc",
    node=Config,
)


@hydra.main(
    version_base=None, config_path="../configs", config_name="run_ds_postproc/default"
)
def main(cfg: Config):
    executor = make_submitit_executor(cfg.runner)

    ds_dir = p.Path(cfg.ds_dir)
    stoi = load_stoi(ds_dir)

    paths = (ds_dir / "train_pbr_v2format").glob("*")
    keys = list(set([str(p.name).split(".")[0] for p in paths]))
    keys_splits = np.array_split(keys, cfg.n_jobs)

    jobs = []
    with executor.batch():
        for keys in keys_splits:
            job = executor.submit(
                process_keys,
                keys,
                ds_dir / "train_pbr_v2format",
                stoi,
                ds_dir / "train_pbr_v2format_newann",
            )
            jobs.append(job)

    submitit.helpers.monitor_jobs(jobs)

    for job in jobs:
        job.result()


if __name__ == "__main__":
    main()
