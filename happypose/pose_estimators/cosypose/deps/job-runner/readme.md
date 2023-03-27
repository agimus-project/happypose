# Main features
- Support for multi-node multi-gpu jobs. The assignment of GPUs to each process (via $CUDA_VISIBLE_DEVICES) is automatically done. *Each process of the job is assumed to be single-GPU*.
- Run a command without writing submission scripts
```
runjob --ngpus=8 --project=myproject --queue=myqueue python distributed_program.py
```
- Create automatically one directory for each job and redirect the output of each process to a separate log file.
- Automatically print the outputs of your job to your monitor. No need to run `tail -f` manually.
- Cancel your job using a `CTRL+C` as if you were running you program interactively.
- Exports useful environment variables related to your project and available resources.

# Exported environment variables:
- JOB_DIR
- PROJECT_DIR
- N_CPUS
- CONDA_ROOT
- CONDA_ENV
- N_PROCS
- PROC_ID
- OUT_FILE
- JOB_LOG_FILE

# Config file
See `examples/config.yaml` for config of projects and queues.

# Usage
```
runjob-config examples/config.yaml
runjob --ngpus=8 --project=myproject --queue=myqueue python distributed_program.py
```
This will start a multi-gpu (possibly multi-node according to your queue config) job with 1 process per GPU and print the output (stdout and stderr) of one of the processes  (the one with `SLURM_LOCALID=0`).
Use keyboard interupt to cancel your job.

# Running tests
Make sure you are on a SLURM cluster `which sinfo` should output something.
```
runjob-config examples/config.yaml
pytest -vs
```

# Features that will be added in the future:
- A simple interface to resume long-running jobs automatically.
- Copy your python project to an temporary directory to ensure modifications you make to your code do not affect your job while it's running.
- A utility to `runjob` for interactive jobs.
