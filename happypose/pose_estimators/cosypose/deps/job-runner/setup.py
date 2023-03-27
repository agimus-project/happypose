from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name='job-runner',
    version='0.0.1',
    description='A simple SLURM utility for running jobs.',
    entry_points={
        'console_scripts': [
            'runjob=job_runner.runjob:runjob',
            'runjob-config=job_runner.runjob:set_config',
            'printlogs=job_runner.runjob:print_logs'
        ],
    },
    packages=find_packages()
)
