import typing as tp

from job_runner.configs import JobEnvironmentConfig


def make_setup(cfg: JobEnvironmentConfig) -> tp.List[str]:
    setup = []
    if cfg.env:
        for k, v in cfg.env.items():
            setup.append("f{k}={v}")
    return setup