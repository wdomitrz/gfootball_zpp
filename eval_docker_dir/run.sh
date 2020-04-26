#!/usr/bin/env bash

python gfootball_zpp/eval/main.py --logdir /eval_logs "$@"

gsutil -m cp -r /eval_logs gs://marl-leaderboard
