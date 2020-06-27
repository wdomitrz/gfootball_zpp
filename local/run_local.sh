#!/bin/bash
# Copyright 2019 The SEED Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# this is a modified version of the original file

set -e
die () {
    echo >&2 "$@"
    exit 1
}
#ENVIRONMENTS="atari|dmlab|football"
#AGENTS="r2d2|vtrace"
if [ "$#" -ne 3 ]
then
    die "Usage: run_local.sh AIP_CONFIG_FILE ENV_CONFIG_FILE NUM_ACTORS"
fi
echo $3 | grep -E -q "^[0-9]+$" || die "Number of actors should be a non-negative integer"
export AIP_CONFIG_FILE="$1"
export ENV_CONFIG_FILE="$2"
export NUM_ACTORS=$3

export ENVIRONMENT="football"
export AGENT="vtrace"
export CONFIG="football"
export JOB_NAME_PREFIX="${USER}_local_${ENVIRONMENT}_${AGENT}_seed_rl"
DOCKER_IMG_TAG="${CONFIG}_${JOB_NAME_PREFIX}"

CONFIG_OUT_FILE=$(mktemp /tmp/XXXXXXXXX.local_seed.yaml)
ENV_CONFIG=$(cat "$ENV_CONFIG_FILE" | tr '\n' ' ')
sed "s|ENV_CONFIG_HERE|${ENV_CONFIG}|g" "$AIP_CONFIG_FILE" > "$CONFIG_OUT_FILE"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR

RUN_LOCAL_PARAMS=$(python3 extract_params_from_aip_config.py --filepath="$CONFIG_OUT_FILE")

../docker/build.sh
docker_version=$(docker version --format '{{.Server.Version}}')
if [[ "19.03" > $docker_version ]]; then
  docker run --entrypoint ./docker/run.sh -ti -it --name seed --rm seed_rl:$DOCKER_IMG_TAG $ENVIRONMENT $AGENT $NUM_ACTORS "$RUN_LOCAL_PARAMS"
else
  docker run --gpus all --entrypoint ./docker/run.sh -ti -it -e HOST_PERMS="$(id -u):$(id -g)" --name seed --rm seed_rl:$DOCKER_IMG_TAG $ENVIRONMENT $AGENT $NUM_ACTORS "$RUN_LOCAL_PARAMS"
fi
