import typing as tp
import pathlib
import submitit

from job_runner.configs import JobEnvironmentConfig


def make_setup(cfg: JobEnvironmentConfig) -> tp.List[str]:
    setup = []
    if cfg.env:
        for k, v in cfg.env.items():
            setup.append(f"export {k}={v}")
    return setup


def make_snapshots(
    code_directories: tp.List[pathlib.Path],
    output_dir: pathlib.Path,
    exclude: tp.Sequence[str] = (),
):
    for code_dir in code_directories:
        snapshot = submitit.helpers.RsyncSnapshot(
            snapshot_dir=output_dir / code_dir.name, root_dir=code_dir, exclude=exclude
        )
        with snapshot:
            pass
    return
