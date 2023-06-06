# syntax=docker/dockerfile:1
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES},display

RUN apt-get update && apt-get install -y --no-install-recommends \
        mesa-utils && \
        rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-c"]

COPY requirements.txt /

RUN ["/bin/bash", "-c", "pip install -r /requirements.txt"]

WORKDIR /workspace/
CMD ["/bin/bash"]