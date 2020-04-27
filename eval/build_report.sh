#!/usr/bin/env bash
# One parameter --output ...

python3 ../gfootball_zpp/eval/build_report.py --jsons_dir eval_logs "$@"
