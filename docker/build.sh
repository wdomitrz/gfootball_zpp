#!/usr/bin/env bash

set -e
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR/..
GRPC_IMAGE=gcr.io/seedimages/seed:grpc

echo docker build --build-arg grpc_image=${GRPC_IMAGE} --build-arg seed_path=${SEED_PATH} -t seed_rl:${CONFIG} -f $DIR/Dockerfile.${CONFIG}

docker build --build-arg grpc_image=${GRPC_IMAGE} --build-arg seed_path=${SEED_PATH} -t seed_rl:${CONFIG} -f $DIR/Dockerfile.${CONFIG} ..