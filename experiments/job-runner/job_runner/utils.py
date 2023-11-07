import pathlib
import typing as tp

import submitit
from job_runner.configs import JobEnvironmentConfig, RunnerConfig


def make_setup(cfg: JobEnvironmentConfig) -> list[str]:
    setup = []
    if cfg.env:
        for k, v in cfg.env.items():
            setup.append(f"export {k}={v}")
    return setup


def make_snapshots(
    code_directories: list[pathlib.Path],
    output_dir: pathlib.Path,
    exclude: tp.Sequence[str] = (),
):
    for code_dir in code_directories:
        snapshot = submitit.helpers.RsyncSnapshot(
            snapshot_dir=output_dir / code_dir.name,
            root_dir=code_dir,
            exclude=exclude,
        )
        with snapshot:
            pass
    return


def make_submitit_executor(
    cfg: RunnerConfig,
):
    if cfg.use_slurm:
        assert cfg.slurm_queue
        assert cfg.slurm_job
        executor = submitit.AutoExecutor(folder=cfg.log_dir)
        executor.update_parameters(
            slurm_partition=cfg.slurm_queue.partition,
            constraint=cfg.slurm_queue.constraint,
            cpus_per_task=cfg.slurm_queue.cpus_per_gpu,
            slurm_account=cfg.slurm_job.account,
            slurm_qos=cfg.slurm_job.qos,
            slurm_time=cfg.slurm_job.time,
            slurm_setup=make_setup(cfg.job_env),
            slurm_additional_parameters=cfg.slurm_job.additional_parameters,
            tasks_per_node=cfg.slurm_job.tasks_per_node,
            gpus_per_node=cfg.slurm_job.gpus_per_node,
        )
    else:
        assert cfg.local_job
        executor = submitit.AutoExecutor(folder=cfg.log_dir, cluster="local")
        executor.update_parameters(
            nodes=cfg.local_job.nodes,
            tasks_per_node=cfg.local_job.tasks_per_node,
            gpus_per_node=cfg.local_job.gpus_per_node,
            cpus_per_task=cfg.local_job.cpus_per_task,
        )

    return executor
