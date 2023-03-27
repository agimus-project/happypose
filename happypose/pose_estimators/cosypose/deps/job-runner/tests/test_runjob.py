import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

CONFIG_PATH = ROOT_DIR / 'myconfigs/config.yaml'


def run_command(command):
    job_command = [
        'runjob',
        '--ngpus=1',
        '--queue=gpu_p1',
        '--project=job-runner',
        command,
    ]
    return subprocess.run(job_command)


def test_stdout_stderr():
    run_command('python examples/test_prints.py')
