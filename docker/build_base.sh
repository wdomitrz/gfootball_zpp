#!/usr/bin/env bash

set -e
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR/..
GRPC_IMAGE=gcr.io/seedimages/seed:grpc

LABEL="${CONFIG}_${JOB_NAME_PREFIX}"

set -x
docker build -t football_base -f $DIR/Dockerfile.football_base ..
