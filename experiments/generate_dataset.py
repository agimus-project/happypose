import typing as tp
import hydra
import os
import omegaconf
import tqdm
import time
import copy
import numpy as np
import submitit
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore


from job_runner.configs import (
    RunnerConfig,
)


@dataclass
class DatasetGenerationConfig:
    dataset_id: str
    chunk_ids: tp.Optional[tp.List[int]]
    debug: bool = False
    verbose: bool = True
    overwrite: bool = False
    few: bool = False


@dataclass
class Config:
    n_jobs: int
    start_chunk: int
    n_chunks: int
    dry_run: bool
    ds: DatasetGenerationConfig
    runner: RunnerConfig


cs = ConfigStore.instance()
cs.store(group="dsgen", name="base_dsgen", node=DatasetGenerationConfig)
cs.store(group="run_dsgen", name="base_run_dsgen", node=Config)


def generate_chunks(ds_cfg: DatasetGenerationConfig):
    submitit.helpers.TorchDistributedEnvironment().export()
    from happypose.pose_estimators.megapose.src.megapose.scripts.generate_shapenet_pbr import main as main_
    ds_cfg = omegaconf.DictConfig(dict(ds_cfg))
    return main_(ds_cfg)


@hydra.main(
    version_base=None,
    config_path='../configs',
    config_name='run_dsgen/default')
def main(cfg: Config):
    print(omegaconf.OmegaConf.to_yaml(cfg))
    executor = submitit.AutoExecutor(folder='logs')

    executor.update_parameters(
        tasks_per_node=cfg.runner.job.tasks_per_node,
        nodes=cfg.runner.job.nodes,
        gpus_per_node=cfg.runner.job.gpus_per_node,
    )

    chunk_ids = np.arange(cfg.start_chunk, cfg.start_chunk + cfg.n_chunks)
    chunk_splits = np.array_split(chunk_ids, cfg.n_jobs)

    jobs = []
    for n, chunk_split_ in enumerate(chunk_splits):
        ds_cfg = copy.deepcopy(cfg.ds)
        ds_cfg.chunk_ids = chunk_split_.tolist()
        if cfg.dry_run:
            job = executor.submit(time.sleep, 1)
        else:
            job = executor.submit(generate_chunks, ds_cfg)
        jobs.append(job)

    submitit.helpers.monitor_jobs(jobs, poll_frequency=5, test_mode=True)
    for job in tqdm.tqdm(submitit.helpers.as_completed(jobs), total=len(jobs)):
        print(job.result())


if __name__ == '__main__':
    main()

#         args = parse_args(use_cli=False)
#         args.command = record_cmd
#         args.ngpus = n_gpus
#         args.return_runner = True
#         args.queue = "v100"
#         args.job_name = f"record_{dataset_id}_{n}" + "_{job_id}"
#         args.code_id = str(code_id)
#         args.dry_run = True
#         runner = runjob(args)
#         runners.append(runner)
#         job_ids.append(runner.job_id)







# if __name__ == "__main__":
#     main()

#     dataset_id = "gso_1M"
#     n_gpus = 4
#     verbose = True
#     overwrite = False
#     is_debug = True
#     start_chunk = 0

#     if is_debug:
#         n_chunks = 2 * n_gpus
#         n_jobs = 2
#         few = True
#     elif dataset_id == "gso_1M":
#         n_chunks = 25000
#         # n_chunks = 5000
#         n_jobs = 64
#         few = False
#     elif dataset_id == "shapenet_1M":
#         n_chunks = 10000
#         n_jobs = 64
#         few = False

#     chunk_ids = np.arange(start_chunk, start_chunk + n_chunks)
#     chunk_splits = np.array_split(chunk_ids, n_jobs)

#     code_id = get_random_id()
#     args = parse_args_code()
#     args.tar_id = code_id
#     upload_code(args)

#     runners = []
#     job_ids = []
#     for n, chunk_split_ in enumerate(chunk_splits):
#         chunk_split_ = str(chunk_split_.tolist()).replace(" ", "")
#         record_cmd = [
#             "python",
#             "-m",
#             "megapose.scripts.generate_shapenet_pbr",
#             f"dataset_id={dataset_id}",
#             f"chunk_ids={chunk_split_}",
#         ]
#         if overwrite:
#             record_cmd.append("overwrite=True")
#         if verbose:
#             record_cmd.append("verbose=True")
#         if few:
#             record_cmd.append("few=True")

#         args = parse_args(use_cli=False)
#         args.command = record_cmd
#         args.ngpus = n_gpus
#         args.return_runner = True
#         args.queue = "v100"
#         args.job_name = f"record_{dataset_id}_{n}" + "_{job_id}"
#         args.code_id = str(code_id)
#         args.dry_run = True
#         runner = runjob(args)
#         runners.append(runner)
#         job_ids.append(runner.job_id)
