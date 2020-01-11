#!/bin/bash

set -e

if [ "$#" -ne "5" ]
then
    echo "JOB_NAME_PREFIX BUCKET_NAME AI_PLATFORM_CONFIG_FILE ENV_CONFIG_FILE EXPERIMENT_PATH"
    exit 1
fi

JOB_NAME_PREFIX="$1"
BUCKET_NAME="$2"
AI_PLATFORM_CONFIG_FILE="$3"
ENV_CONFIG_FILE="$4"
EXPERIMENT_PATH="$5"

gsutil cp -r "${EXPERIMENT_PATH}/1/ckpt" "gs://${BUCKET_NAME}/${JOB_NAME_PREFIX}/1/ckpt"


run_script="$( dirname "${BASH_SOURCE[0]}" )"
run_script="${run_script}/run.sh"

eval "$run_script \"$JOB_NAME_PREFIX\" \"$BUCKET_NAME\" \"$AI_PLATFORM_CONFIG_FILE\" \"$ENV_CONFIG_FILE\""
