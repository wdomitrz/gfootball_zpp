
Now it's time to actually run an experiment and upload it to AI Platform:
mkdir gfootball_seed && cd gfootball_seed
git clone https://github.com/CStanKonrad/seed_rl - clone our SEED repo and switch to current_setup branch
git clone https://github.com/wdomitrz/gfootball_zpp - clone our GFootball repo
cd gfootball_zpp
./gcp/run.sh <job_name_prefix> <bucket_name> <ai_platform_config_file> <env_config_file> <region> - use the run.sh script to run the experiment. For example ./gcp/run.sh test_job test_bucket gcp/f5v5/academy/scoring_bots_4_heads.yaml gcp/f5v5/academy/scoring_bots_academy_5_vs_5_4v0_with_keeper.json - This will run a 5 vs 5 players experiment. The left team is controlled by a neural network whereas the right team is controlled by a build-in ai. In this scenario our players are positioned near the goal of the opponent so they can easily score. The reward in this experiment is scoring instead of checkpoints.
job_name_prefix - a job identifier (each job must have unique name)
bucket_name - the experiment results is stored in the given bucket. If the bucket does not exist it will be created
ai_platform_config_file - path to an AI Platform configuration file. It states e.g. how many machines and of which type must be used. There are a few example configuration files in gfootball_zpp/gcp directory (the *.yaml files)
env_config_file - path to GFootball environment configuration file. It describes the parameters of the environment. Example environment file is gfootball_zpp/gcp/sample_env_config.json
region - the region parameter is optional and the default value is us-central1
The first time you run the experiment docker image must be built which can take some time. Next runs use already created docker layers. You can check if your job was successfully uploaded by logging into AI Platform console. Setting up machines might take a few minutes. After that time in your bucket there should be the data coming from the experiment.
KZ6tka/edit
