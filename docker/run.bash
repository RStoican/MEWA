#!/bin/bash

nvidia-docker run \
  --rm \
  --privileged \
  --name mewa_experiment \
  --gpus "device=0" \
  -v $PWD/mewa:/workspace/mewa \
  mewa /bin/bash