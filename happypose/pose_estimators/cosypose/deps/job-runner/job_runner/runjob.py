import yaml
import multiprocessing
import datetime
import os
import getpass
import sys
import time
from collections import OrderedDict
import subprocess
import numpy as np
from pathlib import Path
import argparse
from .utils import SlurmRunner, LocalRunner

ROOT_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT_DIR / '.cache'
CACHE_DIR.mkdir(exist_ok=True)
CACHE_YAML_PATH = CACHE_DIR / 'cache.yaml'
SLURM_POLLING_INTERVAL = 240
LOCAL_POLLING_INTERVAL = 1
N_COLS = 80


def file_iterator(f, delay=0.1):
    f = Path(f)
    if not f.exists():
        f.write_text('')
    f = f.open('r', newline='', buffering=1)
    while True:
        line = f.readline()
        if not line:
            time.sleep(delay)    # Sleep briefly
            yield None
        yield line


def resolve_path(s):
    return Path(os.path.expandvars(s)).resolve()


def set_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', metavar='config_path')
    args = parser.parse_args()
    config_path = resolve_path(args.config_path)
    CACHE_YAML_PATH.write_text(yaml.dump(dict(config_path=config_path)))
    print("Configuration file set to:", config_path)


def load_config():
    try:
        cache = yaml.load(CACHE_YAML_PATH.read_text(), Loader=yaml.FullLoader)
    except FileNotFoundError:
        raise ValueError('Please set your configuration file with runjob-config.')
    config_path = cache['config_path']
    print("Using config: ", config_path)
    cfg = yaml.load(config_path.read_text(), Loader=yaml.FullLoader)
    return cfg, config_path


def print_logs():
    parser = argparse.ArgumentParser()
    parser.add_argument("jobid", type=str)
    args = parser.parse_args()
    cfg, _ = load_config()
    storage_dir = resolve_path(cfg['storage']['root'])
    job_dir = storage_dir / args.jobid
    for logfile in job_dir.glob('*.out'):
        print(logfile.open('r', newline='').read())


def runjob():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default='', type=str)
    parser.add_argument("--queue", default='', type=str)
    parser.add_argument("--ngpus", default=1, type=int)
    parser.add_argument("--time", default='', type=str)
    parser.add_argument("--jobid", default=str(np.random.randint(1e9)), type=str)
    parser.add_argument("--no-assign-gpu", dest='assign_gpu', action='store_false')
    parser.add_argument("command", nargs=argparse.REMAINDER, help='Command to be executed in each process')
    args = parser.parse_args()

    if not args.command:
        raise ValueError('Please provide a command to run in your job.')
    args.command = ' '.join(args.command)

    cfg, config_path = load_config()

    projects = cfg['projects']
    if not args.project:
        args.project = cfg['default_project']
    project = projects[args.project]

    queues = cfg['gpu_queues']
    if not args.queue:
        if 'default_queue' in project:
            args.queue = project['default_queue']
        else:
            args.queue = cfg['default_queue']

    queue = queues[args.queue]

    is_local = False
    if args.queue == 'local':
        is_local = True
        if queue['n_cpus_per_node'] == 'auto':
            queue['n_cpus_per_node'] = multiprocessing.cpu_count()

        if queue['n_gpus_per_node'] == 'auto':
            queue['n_gpus_per_node'] = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
            assert queue['n_gpus_per_node'] > 0

    job_name = args.jobid
    storage_dir = resolve_path(cfg['storage']['root'])
    job_dir = storage_dir / job_name
    job_dir.mkdir(exist_ok=True)

    flags = queue.get('flags', dict())
    if args.time:
        flags['time'] = args.time
    flags['ntasks'] = args.ngpus
    n_proc_per_node = min(args.ngpus, queue['n_gpus_per_node'])
    n_cpus_per_gpu = int(queue['n_cpus_per_node'] / queue['n_gpus_per_node'])
    flags['ntasks-per-node'] = n_proc_per_node
    flags['cpus-per-task'] = n_cpus_per_gpu
    flags['job-name'] = job_name
    flags['gres'] = f'gpu:{n_proc_per_node}'
    flags['output'] = job_dir / 'proc=0.out'
    flags['error'] = job_dir / 'proc=0.out'

    env = OrderedDict()
    env.update(
        JOB_DIR=job_dir,
        PROJECT_DIR=project['dir'],
        CONDA_ROOT=cfg['conda']['root'],
        CONDA_ENV=project['conda_env'],
        N_CPUS=flags['cpus-per-task'],
        N_PROCS=args.ngpus,
        PROC_ID='${SLURM_PROCID}',
        OUT_FILE='${JOB_DIR}/proc=${PROC_ID}.out',
        JOB_LOG_FILE='${JOB_DIR}/proc=0.out'
    )

    bash_script = (config_path.parent / project['preamble']).read_text()

    bash_env_def = ''
    for k, v in env.items():
        bash_env_def += f'export {k}={v}\n'

    if args.assign_gpu:
        conda_python = resolve_path(cfg['conda']['root']) / 'envs' / project['conda_env'] / 'bin/python'
        assign_gpu_path = Path(__file__).resolve().parent / 'assign_gpu.py'
        bash_env_def += f'eval $({conda_python} {assign_gpu_path})\n'

    bash_script = '# This is automatically added by job-runner\n' + bash_env_def + '\n' + bash_script
    bash_script += args.command + ' &> $OUT_FILE'
    bash_script += '\nexit 0'

    bash_script_path = job_dir / 'script.sh'
    bash_script_path.write_text(bash_script)

    if is_local:
        Runner = LocalRunner
    else:
        Runner = SlurmRunner
    runner = Runner(bash_script_path, flags, env)

    string_flags = ''
    for k, v in flags.items():
        string_flags += f'{k}={v}\n'

    print(f"""Job directory: {job_dir}

Content of script.sh:
{'-'*N_COLS}
{bash_script}
{'-'*N_COLS}

Flags:
{'-'*N_COLS}
{string_flags}
{'-'*N_COLS}

runner ({Runner}):
{runner.get_string_infos()}
""")

    if is_local:
        polling_interval = LOCAL_POLLING_INTERVAL
    else:
        polling_interval = SLURM_POLLING_INTERVAL

    def print_output():
        runner.start()
        follow_file = Path(env['JOB_LOG_FILE'].replace('${JOB_DIR}', str(job_dir)))
        start = datetime.datetime.now()
        print(f"Job started: {start}")
        print(f"Job output {follow_file}\n{'-'*N_COLS}")

        is_done = False
        time_prev_check = time.time()
        for text in file_iterator(follow_file):
            if text is not None:
                print(text, end="")
            if (time.time() - time_prev_check) >= polling_interval:
                is_done = runner.is_done()
                time_prev_check = time.time()
            if is_done:
                break
        end = datetime.datetime.now()
        print(f"{'-'*N_COLS}")
        print(f"Job finished: {start} ({end - start})")

    try:
        print_output()
    except KeyboardInterrupt:
        runner.stop()
        print("\nJob cancelled.")


if __name__ == '__main__':
    runjob()
