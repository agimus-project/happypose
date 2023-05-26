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
    constraint: tp.Optional[str] = None


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
    additional_parameters: tp.Optional[tp.Dict[str, tp.Any]]


@dataclass
class JobEnvironmentConfig:
    conda_env: str
    python_packages_dir: tp.Optional[tp.List[str]] = None
    env: tp.Optional[tp.Dict[str, str]] = None


@dataclass
class RunnerConfig:
    log_dir: str
    job_env: JobEnvironmentConfig
    local_node: tp.Optional[NodeConfig]
    local_job: tp.Optional[JobConfig]
    slurm_queue: tp.Optional[SlurmQueueConfig]
    slurm_job: tp.Optional[SlurmJobConfig]
    use_slurm: bool = False


cs = ConfigStore.instance()
cs.store(group="job_env", name="base_job_env", node=JobEnvironmentConfig)
cs.store(group="local_job", name="base_local_job", node=JobConfig)
cs.store(group="local_node", name="base_local_node", node=NodeConfig)
cs.store(group="slurm_job", name="base_slurm_job", node=SlurmJobConfig)
cs.store(group="slurm_queue", name="base_slurm_queue", node=SlurmQueueConfig)
cs.store(group="runner", name="base_runner", node=RunnerConfig)