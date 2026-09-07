import typing as tp
from dataclasses import dataclass

from hydra.core.config_store import ConfigStore


@dataclass
class NodeConfig:
    gpus_per_node: int
    mem_per_gpu: str
    cpus_per_gpu: int
    mem_per_cpu: str


@dataclass
class SlurmQueueConfig(NodeConfig):
    partition: str
    constraint: str | None = None


@dataclass
class JobConfig:
    nodes: int
    gpus_per_node: int
    cpus_per_task: int
    tasks_per_node: int


@dataclass
class SlurmJobConfig(JobConfig):
    account: str
    qos: str
    time: str
    additional_parameters: dict[str, tp.Any] | None


@dataclass
class CodeSnapshotConfig:
    snapshot_dir: str | None
    exclude_path: str | None
    python_packages_dir: list[str] | None = None


@dataclass
class JobEnvironmentConfig:
    conda_env: str
    code_snapshot: CodeSnapshotConfig | None = None
    env: dict[str, str] | None = None


@dataclass
class RunnerConfig:
    log_dir: str
    job_env: JobEnvironmentConfig
    local_node: NodeConfig | None
    local_job: JobConfig | None
    slurm_queue: SlurmQueueConfig | None
    slurm_job: SlurmJobConfig | None
    use_slurm: bool = False


cs = ConfigStore.instance()
cs.store(group="code_snapshot", name="base_code_snapshot", node=CodeSnapshotConfig)
cs.store(group="job_env", name="base_job_env", node=JobEnvironmentConfig)
cs.store(group="local_job", name="base_local_job", node=JobConfig)
cs.store(group="local_node", name="base_local_node", node=NodeConfig)
cs.store(group="slurm_job", name="base_slurm_job", node=SlurmJobConfig)
cs.store(group="slurm_queue", name="base_slurm_queue", node=SlurmQueueConfig)
cs.store(group="runner", name="base_runner", node=RunnerConfig)
