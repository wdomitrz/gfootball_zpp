set -x

gsutil -m cp -r gs://less_randomized/less_then_partially_randomized_checkpoints_bots_4_heads_38/1/ckpt			gs://less_randomized/less_then_partially_then_more_randomized_checkpoints_bots_4_heads_64/1/ckpt
gsutil -m cp -r gs://less_randomized/less_then_partially_randomized_entropy_checkpoints_bots_4_heads_39/1/ckpt  gs://less_randomized/less_then_partially_then_more_randomized_entropy_checkpoints_bots_4_heads_65/1/ckpt
gsutil -m cp -r gs://less_randomized/less_then_partially_randomized_checkpoints_bots_4_heads_38/1/ckpt			gs://less_randomized/less_then_partially_then_partially_randomized_checkpoints_bots_4_heads_66/1/ckpt
gsutil -m cp -r gs://less_randomized/less_then_partially_randomized_entropy_checkpoints_bots_4_heads_39/1/ckpt  gs://less_randomized/less_then_partially_then_partially_randomized_entropy_checkpoints_bots_4_heads_67/1/ckpt
gsutil -m cp -r gs://less_randomized/less_then_partially_randomized_checkpoints_bots_4_heads_38/1/ckpt 		    gs://less_randomized_then_self_play/less_then_partially_bots_then_partially_self_play_randomized_checkpoints_4_heads_68/1/ckpt
gsutil -m cp -r gs://less_randomized/less_then_partially_randomized_entropy_checkpoints_bots_4_heads_39/1/ckpt  gs://less_randomized_then_self_play/less_then_partially_bots_then_partially_self_play_randomized_entropy_checkpoints_4_heads_69/1/ckpt
