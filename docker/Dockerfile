FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

RUN mkdir dataset
ENV HAPPYPOSE_DATA_DIR=./dataset

RUN pip install -r requirements/base.txt
RUN pip install -r requirements/pypi.txt
RUN pip install -r requirements/cu124.txt