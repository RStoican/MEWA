#!/bin/bash

# Default values
task=/workspace/MEWA/tasks/narrow/delete
config=/workspace/MEWA/algorithms/random/configs/example.yaml

while getopts 't:,c:,h' opt; do
  case "$opt" in
    t)
      task="$OPTARG"
      echo "Processing option 't' with '${task}' argument"
      ;;
    c)
      config="$OPTARG"
      echo "Processing option 'c' with '${config}' argument"
      ;;
    h)
      echo "Usage: $(basename "$0") [-t task] [-c config]"
      exit 1
      ;;
    *)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done
shift "$((OPTIND -1))"

nvidia-docker run \
  --rm \
  --privileged \
  --name random_test \
  --gpus "device=0" \
  -v "$PWD":/workspace/MEWA \
  -w /workspace/MEWA \
  --env PYTHONPATH=/workspace/MEWA/ \
  mewa:latest \
  /bin/bash -c "python /workspace/MEWA/algorithms/random/random/run_random_agent.py '${task}' '${config}'"