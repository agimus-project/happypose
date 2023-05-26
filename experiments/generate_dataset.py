import typing as tp
import hydra
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
from job_runner.utils import make_setup


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
    from happypose.pose_estimators.megapose.src.megapose.scripts.generate_shapenet_pbr import (
        main as main_,
    )

    ds_cfg = omegaconf.DictConfig(dict(ds_cfg))
    return main_(ds_cfg)


@hydra.main(
    version_base=None, config_path="../configs", config_name="run_dsgen/default"
)
def main(cfg: Config):

    if cfg.runner.use_slurm:
        executor = submitit.AutoExecutor(folder=cfg.runner.log_dir)
        executor.update_parameters(
            slurm_partition=cfg.runner.slurm_queue.partition,
            constraint=cfg.runner.slurm_queue.constraint,
            cpus_per_task=cfg.runner.slurm_queue.cpus_per_gpu,
            slurm_account=cfg.runner.slurm_job.account,
            slurm_qos=cfg.runner.slurm_job.qos,
            slurm_time=cfg.runner.slurm_job.time,
            slurm_setup=make_setup(cfg.runner.job_env),
            slurm_additional_parameters=cfg.runner.slurm_job.additional_parameters,
            tasks_per_node=cfg.runner.slurm_job.tasks_per_node,
            gpus_per_node=cfg.runner.slurm_job.gpus_per_node,
        )
    else:
        executor = submitit.AutoExecutor(folder=cfg.runner.log_dir, cluster="local")
        executor.update_parameters(
            nodes=cfg.runner.local_job.nodes,
            tasks_per_node=cfg.runner.local_job.tasks_per_node,
            gpus_per_node=cfg.runner.local_job.gpus_per_node,
            cpus_per_task=cfg.runner.local_job.cpus_per_task,
        )

    chunk_ids = np.arange(cfg.start_chunk, cfg.start_chunk + cfg.n_chunks)
    chunk_splits = np.array_split(chunk_ids, cfg.n_jobs)

    jobs = []
    with executor.batch():
        for n, chunk_split_ in enumerate(chunk_splits):
            ds_cfg = copy.deepcopy(cfg.ds)
            ds_cfg.chunk_ids = chunk_split_.tolist()
            if cfg.dry_run:
                job = executor.submit(time.sleep, 1)
            else:
                job = executor.submit(generate_chunks, ds_cfg)
            jobs.append(job)

    submitit.helpers.monitor_jobs(jobs, poll_frequency=5, test_mode=True)


if __name__ == "__main__":
    main()
