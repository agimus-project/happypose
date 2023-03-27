from pathlib import Path
import os
import subprocess
import getpass
import shutil

N_COLS = 80
SQUEUE_PATH = shutil.which('squeue')

class SlurmRunner:
    def __init__(self, bash_script_path, flags, env):
        script = '#!/bin/bash\n'
        for k, v in flags.items():
            script += f'#SBATCH --{k}={v}\n'
        script += '\n'
        script += f'srun bash {bash_script_path}'
        slurm_script_path = Path(env['JOB_DIR']) / 'script.slurm'
        slurm_script_path.write_text(script)
        self.slurm_script_path = slurm_script_path
        self.job_name = flags['job-name']

    def get_string_infos(self):
        slurm_script = self.slurm_script_path.read_text()
        return f"""Content of script.slurm:
{'-'*N_COLS}
{slurm_script}
{'-'*N_COLS}"""

    def start(self):
        proc_output = subprocess.run(['sbatch', self.slurm_script_path])

    def is_done(self):
        username = getpass.getuser()
        jobinfo = subprocess.check_output([SQUEUE_PATH, '-u', username, '--name', self.job_name, '--noheader'])
        return len(jobinfo) == 0

    def stop(self):
        subprocess.run(['scancel', '--name', self.job_name])


class LocalRunner:
    def __init__(self, bash_script_path, flags, env):
        self.bash_script_path = bash_script_path
        self.flags = flags
        self.cuda_ids = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        assert len(self.cuda_ids) == self.flags['ntasks-per-node']

    def get_string_infos(self):
        return f"Cuda ids: {self.cuda_ids}"

    def start(self):
        processes = []
        for n in range(self.flags['ntasks']):
            env = os.environ.copy()
            # Simulate SLURM environment
            env.update(SLURM_LOCALID=str(n),
                       SLURM_PROCID=str(n),
                       SLURM_NTASKS=str(self.flags['ntasks']),
                       CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'])
            process = subprocess.Popen(['bash', self.bash_script_path], env=env)
            processes.append(process)
        self.processes = processes

    def is_done(self):
        is_done = [process.poll() is not None for process in self.processes]
        return all(is_done)

    def stop(self):
        for process in self.processes:
            process.kill()
