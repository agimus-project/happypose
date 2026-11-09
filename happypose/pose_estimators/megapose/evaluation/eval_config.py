"""Copyright (c) 2022 Inria & NVIDIA CORPORATION & AFFILIATES. All rights reserved.

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
from dataclasses import dataclass, field

# MegaPose
from happypose.pose_estimators.megapose.inference.types import InferenceConfig

BOP_TEST_DATASETS = [
    "lmo.bop19",
    "tless.bop19",
    "tudl.bop19",
    "icbin.bop19",
    "itodd.bop19",
    "hb.bop19",
    "ycbv.bop19",
]


@dataclass
class HardwareConfig:
    n_cpus: int = 8
    n_gpus: int = 1


@dataclass
class EvalConfig:
    """Eval Config.

    Two options for creating an eval configuration:
    1. Create it manually, and set `run_id`.
    2. If `run_id` is None, then use `config_id`, `run_comment`and
    `run_postfix` to create a `run_id`

    In 2., the parameters of the config are set-up using the function
    `update_cfg_with_config_id`.
    """

    # Network
    detector_run_id: str = "bop_pbr"
    coarse_run_id: str | None = None
    refiner_run_id: str | None = None

    # Dataset
    ds_name: str = "ycbv.bop19"

    # Inference
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    # Run management
    result_id: str | None = None
    n_dataloader_workers: int = 8
    n_rendering_workers: int = 8
    n_frames: int | None = None
    batch_size: int = 1
    save_dir: str | None = None
    bsz_images: int = 256
    bsz_objects: int = 16
    skip_inference: bool = False
    skip_evaluation: bool = True

    # Infos
    global_batch_size: int | None = None
    hardware: HardwareConfig = field(default_factory=HardwareConfig)

    # Debug
    debug: bool = False


@dataclass
class FullEvalConfig(EvalConfig):
    # Full eval
    detection_coarse_types: list | None = None
    ds_names: list[str] | None = None
    run_bop_eval: bool = True
    eval_coarse_also: bool = False
    convert_only: bool = False


@dataclass
class BOPEvalConfig:
    results_path: str
    dataset: str
    split: str
    eval_dir: str
    method: str  # ['refiner/final', 'depth_refiner', etc.]
    detection_method: str | None = None
    convert_only: bool = False
    use_post_score: bool | None = True
