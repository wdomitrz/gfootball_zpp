#!/usr/bin/env bash

set -e
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR/..

set -x

"$DIR/build_base.sh"

docker build -t football_eval -f $DIR/Dockerfile.football_eval ..
