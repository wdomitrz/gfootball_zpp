# this example contains comments and data from
# a) ray.readthedocs.io
# b) https://github.com/google-research/football/

import ray

from ray import tune
from ray.tune.registry import register_env

from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.agents.dqn.distributional_q_model import DistributionalQModel
from ray.rllib.utils import try_import_tf
from ray.rllib.env.multi_agent_env import MultiAgentEnv

import gfootball.env as football_env
import gym
import numpy as np

tf = try_import_tf()



# sample config wrapper - nothing interesting
def football_config_wrapper(config):
    return football_env.create_environment(
        env_name=config['env_name'],
        stacked=config['stacked'],
        representation=config['representation'],
        rewards=config['rewards'],
        enable_goal_videos=config['enable_goal_videos'],
        enable_full_episode_videos=config['enable_full_episode_videos'],
        render=config['render'],
        write_video=config['write_video'],
        dump_frequency=config['dump_frequency'],
        logdir=config['logdir'],
        number_of_left_players_agent_controls=config['number_of_left_players_agent_controls'],
        number_of_right_players_agent_controls=config['number_of_right_players_agent_controls'])


# for example ray.init('cluster-skonrad-1') will join cluster
# whose main node is cluster-skonrad-1.
# running `cluster_make.sh yourclustername x run_ray` will set machine cluster-yourclustername-1 as main node and other machines as workers
ray.init()


register_env('gfootball', football_config_wrapper)

def example_single_agent():

    ray.tune.run(
        'PPO',

        checkpoint_freq=1,

        checkpoint_at_end=True,

        # runs experiment num_samples times
        # when config contains tune.sample_from, each time new sample is taken
        # when config contains tune.grid_search, each time full search is made
        num_samples=1,

        # restore from checkpoint
        # restore='~/ray_results/Original/PG_<xxx>/checkpoint_5/checkpoint-5',

        # stop rules
        # stops if at least one condition is met
        # can be a function
        # for more see
        # https://ray.readthedocs.io/en/latest/tune-usage.html#custom-stopping-criteria
        stop={
            'timesteps_total': 10,
            'mean_accuracy': 0.98,
        },

        config={
            ## global config (can be the same for other algorithms ex. impala)

            'env': 'gfootball', # environment was registered before

            # example of custom metric can be found at
            # https://ray.readthedocs.io/en/latest/rllib-training.html#callbacks-and-custom-metrics
            #'callbacks': ,

            # see https://github.com/ray-project/ray/blob/master/rllib/examples/custom_keras_model.py
            #'model': ,

            # number of steps after episode is forced to terminate
            'horizon': None,

            # gamma set with usage of tune.sample_from((spec->value))
            # https://ray.readthedocs.io/en/latest/tune-usage.html#tune-search-space-default
            'gamma': tune.sample_from(lambda spec: 0.99 if spec.config.horizon == None else 0.98),

            'lr': 2.5e-4,

            'num_workers': 1,

            # Number of environments to evaluate vectorwise per worker.
            'num_envs_per_worker': 1,

            # Default sample batch size (unroll length). Batches of this size are
            # collected from workers until train_batch_size is met. When using
            # multiple envs per worker, this is multiplied by num_envs_per_worker.
            'sample_batch_size': 100,

            # Training batch size, if applicable. Should be >= sample_batch_size.
            # Samples batches will be concatenated together to this size for training.
            'train_batch_size': 2000,

            # This argument, in conjunction with worker_index, sets the random seed of
            # each worker, so that identically configured trials will have identical
            # results. This makes experiments reproducible.
            # tune.grid_search runs separate trials for seed 42 and 14
            # if we have n x tune.grid_search(S_i) then number of trials is S_1 * ... * S_n
            # if we also have num_samples > 1 then nimber of trials is num_samples * S_1 * ... * S_n
            'seed': tune.grid_search([42, 14]),

            # Whether to rollout "complete_episodes" or "truncate_episodes"
            'batch_mode': 'truncate_episodes',

            'num_gpus': 0,

            'clip_rewards': False,

            ## for PPO

            # GAE parameter
            'lambda': 0.95,

            # initial for KL divergence
            'kl_coeff': 0.2,

            # Clip param for the value function. Note that this is sensitive to the
            # scale of the rewards. If your expected V is large, increase this.
            'vf_clip_param': 10.0,

            # Coefficient of the entropy regularizer
            'entropy_coeff': 0.01,

            # Total SGD batch size across all devices for SGD
            'sgd_minibatch_size': 500,

            # Number of SGD iterations in each outer loop
            'num_sgd_iter': 10,

            # Share layers for value function. If you set this to True, it's important
            # to tune vf_loss_coeff.
            'vf_share_layers': True,

            # Coefficient of the value function loss. It's important to tune this if
            # you set vf_share_layers: True
            'vf_loss_coeff': 1.0,

        'env_config': {
            'env_name': 'academy_empty_goal_close',
            'stacked': True,
            'representation': 'extracted',
            'rewards': 'scoring',
            'enable_goal_videos': False,
            'enable_full_episode_videos': False,
            'render': False,
            'write_video': False,
            'dump_frequency': 0,
            'logdir': '',
            'number_of_left_players_agent_controls': 1,
            'number_of_right_players_agent_controls': 0,
        },
        }
    )



# fixed version of
# https://github.com/google-research/football/blob/master/gfootball/examples/run_multiagent_rllib.py
# for more see https://github.com/ray-project/ray/blob/master/rllib/env/multi_agent_env.py
class RllibGFootball(MultiAgentEnv):
    """An example of a wrapper for GFootball to make it compatible with rllib."""

    def __init__(self, config):
        self.env = football_config_wrapper(config)
        self.action_space = gym.spaces.Discrete(self.env.action_space.nvec[1])
        self.observation_space = gym.spaces.Box(
            low=self.env.observation_space.low[0],
            high=self.env.observation_space.high[0],
            dtype=self.env.observation_space.dtype)
        self.num_agents = config['number_of_left_players_agent_controls']

    def reset(self):
        original_obs = self.env.reset()
        obs = {}
        for x in range(self.num_agents):
            if self.num_agents > 1:
                obs['agent_%d' % x] = original_obs[x]
            else:
                obs['agent_%d' % x] = original_obs
        return obs

    def step(self, action_dict):
        actions = []
        for key, value in sorted(action_dict.items()):
            actions.append(value)
        o, r, d, i = self.env.step(actions)
        rewards = {}
        obs = {}
        for pos, key in enumerate(sorted(action_dict.keys())):
            rewards[key] = np.mean(r) # needs to be a scalar
            if self.num_agents > 1:
                obs[key] = o[pos]
            else:
                obs[key] = o

        dones = {'{}'.format(key): d for key, _ in action_dict.items()}
        dones['__all__'] = d

        # pass same info for all agents
        # fixes ValueError: Key set for infos must be a subset of obs: dict_keys(['__all__']) vs dict_keys(['agent_0', 'agent_1'])
        infos = {'{}'.format(key): i for key, _ in action_dict.items() }
        return obs, rewards, dones, infos

def example_multi_agent():
    env_config = { 'env_name': 'academy_3_vs_1_with_keeper',
                   'stacked': True,
                   'representation': 'extracted',
                   'rewards': 'scoring',
                   'enable_goal_videos': False,
                   'enable_full_episode_videos': False,
                   'render': False,
                   'write_video': False,
                   'dump_frequency': 0,
                   'logdir': '',
                   'number_of_left_players_agent_controls': 2,
                   'number_of_right_players_agent_controls': 0}
    single_env = RllibGFootball(env_config)
    obs_space = single_env.observation_space
    act_space = single_env.action_space

    # the first tuple value is None -> uses default policy
    # (policy, observation_space (gym), action_space (gym), policy_config(for example {gamma="0.23"}))
    def gen_policy():
        return (None, obs_space, act_space, {})

    tune.run(
      'PPO',
      stop={'training_iteration': 10000},
      checkpoint_freq=50,
      config={
          'env': RllibGFootball,
          'horizon': 100,
          'lambda': 0.95,
          'kl_coeff': 0.2,
          'clip_rewards': False,
          'vf_clip_param': 10.0,
          'entropy_coeff': 0.01,
          'train_batch_size': 2000,
          'sample_batch_size': 100,
          'sgd_minibatch_size': 500,
          'num_sgd_iter': 10,
          'num_workers': 20,
          'num_envs_per_worker': 1,
          'batch_mode': 'truncate_episodes',
          'observation_filter': 'NoFilter',
          'vf_share_layers': 'true',
          'num_gpus': 1,
          'lr': 2.5e-4,
          'log_level': 'DEBUG',
          # in this case one policy is applied to all agents (one by one)
          'multiagent': {
              'policies': {'P1': gen_policy()},
              
              # specifies policy for each agent
              'policy_mapping_fn': tune.function(
                  lambda agent_id: 'P1'),
          },
          'env_config': env_config,
      },
  )

#example_single_agent()
example_multi_agent()
