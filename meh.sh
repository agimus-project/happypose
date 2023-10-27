#!/bin/bash -eux

export HOME=/home/gsaurel
export PATH=$HOME/.local/bin:$PATH
export POETRY_VIRTUALENVS_IN_PROJECT=true

rm -rf .venv local_data
poetry env use /usr/bin/python
poetry install --with dev -E cpu -E render -E evaluation

mkdir local_data

poetry run python -m happypose.toolbox.utils.download \
    --megapose_models \
    --examples \
        crackers_example \
    --cosypose_models  \
        detector-bop-ycbv-pbr--970850 \
        coarse-bop-ycbv-pbr--724183 \
        refiner-bop-ycbv-pbr--604090

poetry run python -m unittest
