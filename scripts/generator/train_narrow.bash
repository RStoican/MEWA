#!/bin/bash

nvidia-docker run \
  --rm \
  --privileged \
  --name task_trainer_narrow \
  --gpus "device=0" \
  -v $PWD:/workspace/MEWA \
  --env PYTHONPATH=/workspace/MEWA/ \
  --workdir /workspace/MEWA \
  mewa:latest \
  /bin/bash -c "python /workspace/MEWA/scripts/task_generation/optimise_task.py /workspace/MEWA/tasks/narrow/example/example.yaml --epochs 10000 --search 20"