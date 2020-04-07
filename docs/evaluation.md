# How to add new opponent to evaluation

1. Make sure that it can be created as zpp player (ordinary multi head teams can).

2. In gfootball_zpp/eval/main.py add your player definition to `ZPP_OPPONENTS`
as: `'<name>': ZppEvalPlayerData('<name>', <arg1>=<val1>, <arg2>=<val2>)` where
additional named args are arguments passed as player config to zpp player.

3. Run `python3 gfootball_zpp/eval/main.py --name <name> --logdir <logdir>`.
    You can do this on machine `instance-2` on user `opala_zuzanna`.

4. Copy data from local logdir to `gs://marl-leaderboard` bucket using gsutil:
`gsutil cp -r <logdir>/* gs://marl-leaderboard` (does not work from the gcp machine :/).

5. Copy json file from the main directory in logdir to `eval_results` in our
repository and push it to master (this will trigger building the docs).

6. [optional] Remove the local logdir.
