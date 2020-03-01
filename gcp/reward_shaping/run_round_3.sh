set -x

../run.sh less_then_partially_then_more_randomized_checkpoints_bots_4_heads_64                          less_randomized    checkpoints_bots_4_heads.yaml                                               checkpoints_more_randomized_bots_4_heads.json
../run.sh less_then_partially_then_more_randomized_entropy_checkpoints_bots_4_heads_65                  less_randomized     checkpoints_bots_4_heads_entropy.yaml                                       checkpoints_more_randomized_bots_4_heads.json
../run.sh less_then_partially_then_partially_randomized_checkpoints_bots_4_heads_66                     less_randomized     checkpoints_bots_4_heads.yaml                                               checkpoints_partially_randomized_bots_4_heads.json
../run.sh less_then_partially_then_partially_randomized_entropy_checkpoints_bots_4_heads_67             less_randomized     checkpoints_bots_4_heads_entropy.yaml                                       checkpoints_partially_randomized_bots_4_heads.json
../run.sh less_then_partially_bots_then_partially_self_play_randomized_checkpoints_4_heads_68           less_randomized_then_self_play  checkpoints_self_play_4_heads.yaml          checkpoints_partially_randomized_self_play_4_heads.json
../run.sh less_then_partially_bots_then_partially_self_play_randomized_entropy_checkpoints_4_heads_69   less_randomized_then_self_play  checkpoints_self_play_4_heads_entropy.yaml  checkpoints_partially_randomized_self_play_4_heads.json
