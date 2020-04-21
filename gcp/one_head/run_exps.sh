#!/bin/bash


current_date=$(date | tr ',' '_' | tr ' ' '_' | tr ':' '_' | tr 'ś' '_')


if [ "$1" = 'c4' ] ; then
    gfootball_zpp/gcp/run.sh one_head_4_nets_${current_date} one_head gfootball_zpp/gcp/one_head/checkpoints_bots_4_nets_entropy.yaml gfootball_zpp/gcp/one_head/checkpoints_less_randomized_bots_4_obs.json
elif [ "$1" = 'c4x1' ] ; then
    gfootball_zpp/gcp/run.sh one_head_4x1_net_${current_date} one_head gfootball_zpp/gcp/one_head/checkpoints_bots_4x1_net_entropy.yaml gfootball_zpp/gcp/one_head/checkpoints_less_randomized_bots_4_obs.json
else
    echo "Available"
    echo "    1. c4 (checkpoints, 4 one head nets)"
    echo "    2. c4x1 (checkpoints, one head net used 4 times)"
fi
