#!/usr/bin/env bash

set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_DIR=${SCRIPT_DIR}

docker run \
    -e DISPLAY=host.docker.internal:0 \
    -e LD_PRELOAD=/usr/local/lib/python3.8/dist-packages/torch/lib/../../torch.libs/libgomp-6e1a1d1b.so.1.0.0 \
    --mount type=bind,src=${PROJECT_DIR},dst=/opt/nc_perception/ \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --network host \
    -it assistive-gym
