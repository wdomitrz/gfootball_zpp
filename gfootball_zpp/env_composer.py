from gfootball.env import config
from gfootball.env import football_env
from gfootball.env import wrappers

from .wrappers.action_wrappers import ActionOrder
from .wrappers.player_stack_wrapper import PlayerStackWrapper
from .wrappers.old_2_multihead_nets import MultiHeadNets2
from .wrappers.old_1_multihead_net import MultiHeadNet
from .wrappers.ball_ownership import BallOwnershipRewardWrapper
from .wrappers.recreatable_env import create_recreatable_football
from .wrappers.rewards import DecayingCheckpointRewardWrapper
from .logging.api import enable_log_api_for_config, get_loggers_dict

import collections
import gym
import numpy as np
import subprocess
import os
import threading
import time
import tensorflow as tf


# This wrapper was adopted from https://github.com/google-research/football
# See link for license.
class FrameStack(gym.Wrapper):
  """Stack k last observations."""

  def __init__(self, env, k):
    gym.Wrapper.__init__(self, env)
    self.obs = collections.deque([], maxlen=k)
    low = env.observation_space.low
    high = env.observation_space.high
    low = np.concatenate([low] * k, axis=-1)
    high = np.concatenate([high] * k, axis=-1)
    self.observation_space = gym.spaces.Box(
        low=low, high=high, dtype=env.observation_space.dtype)

  def reset(self):
    observation = self.env.reset()
    self.obs.extend([observation] * self.obs.maxlen)
    return self._get_observation()

  def step(self, action):
    observation, reward, done, info = self.env.step(action)
    self.obs.append(observation)
    return self._get_observation(), reward, done, info

  def _get_observation(self):
    return np.concatenate(list(self.obs), axis=-1)

# This wrapper was adopted from https://github.com/google-research/football
# and modified. See link for license.


class PeriodicDumpWriter(gym.Wrapper):
  """A wrapper that only dumps traces/videos periodically."""

  def __init__(self, env, config):
    gym.Wrapper.__init__(self, env)
    self._dump_frequency = config['dump_frequency']
    self._original_dump_config = {
        'write_video': config['write_video'],
        'dump_full_episodes': config['enable_full_episode_videos'],
        'dump_scores': config['enable_goal_videos'],
    }
    self._current_episode_number = 0

  def __getattr__(self, attr):
    return getattr(self.env, attr)

  def step(self, action):
    return self.env.step(action)

  def reset(self):
    if (self._dump_frequency > 0 and
            (self._current_episode_number % self._dump_frequency == 0)):
      self.env._config.update(self._original_dump_config)
      # self.env.render()
    else:
      self.env._config.update({
          'write_video': False,
          'dump_full_episodes': False,
          'dump_scores': False
      })
      self.env.disable_render()
    self._current_episode_number += 1
    return self.env.reset()


def dump_wrapper(env, config):
  if config['dump_frequency'] > 1:
    return PeriodicDumpWriter(env, config)
  else:
    return env


def checkpoint_wrapper(env, config):
  assert 'scoring' in config['rewards'].split(',')
  if 'checkpoints' in config['rewards'].split(','):
    return wrappers.CheckpointRewardWrapper(env)
  else:
    return env


def decaying_checkpoint_wrapper(env, config):
  print("decaying_checkpoint_wrapper")
  assert 'scoring' in config['rewards'].split(',')
  return DecayingCheckpointRewardWrapper(env)


def ball_ownership_reward_wrapper(env, config):
  return BallOwnershipRewardWrapper(env)


def single_agent_wrapper(env, config):
  if (config['number_of_left_players_agent_controls'] +
          config['number_of_right_players_agent_controls'] == 1):
    env = wrappers.SingleAgentObservationWrapper(env)
    env = wrappers.SingleAgentRewardWrapper(env)
    return env
  else:
    return env


KNOWN_WRAPPERS = {
    'periodic_dump':
    dump_wrapper,
    'checkpoint_score':
    checkpoint_wrapper,
    'decaying_checkpoint_wrapper':
    decaying_checkpoint_wrapper,
    'ball_ownership_reward':
    ball_ownership_reward_wrapper,
    'single_agent':
    single_agent_wrapper,
    'obs_extract':
    lambda env, config: wrappers.SMMWrapper(env, config['channel_dimensions']),
    'obs_stack':
    lambda env, config: FrameStack(env, config['stacked_frames']),
    'action_order':
    ActionOrder,
    'psw':
    PlayerStackWrapper,
    'old_w':
    MultiHeadNets2,
    'old_single_map':
    MultiHeadNet
}
KNOWN_WRAPPERS.update(get_loggers_dict())


def compose_environment(env_config, wrappers):
  enable_log_api_for_config(env_config)  # we enable log api

  def extract_from_dict(dictionary, keys):
    return {new_k: dictionary[k] for (new_k, k) in keys}

  players = [('agent:left_players=%d,right_players=%d' %
              (env_config['number_of_left_players_agent_controls'],
               env_config['number_of_right_players_agent_controls']))]
  if env_config['extra_players'] is not None:
    players.extend(env_config['extra_players'])
  env_config['players'] = players
  football_config = extract_from_dict(
      env_config, [('enable_sides_swap', 'enable_sides_swap'),
                   ('dump_full_episodes', 'enable_full_episode_videos'),
                   ('dump_scores', 'enable_goal_videos'),
                   ('level', 'env_name'), ('players', 'players'),
                   ('render', 'render'), ('tracesdir', 'logdir'),
                   ('write_video', 'write_video')])
  if 'env_change_rate' not in env_config:
    env = football_env.FootballEnv(config.Config(football_config))
  else:
    env = create_recreatable_football(env_config, football_config)

  for w in wrappers:
    env = w(env, env_config)

  print(env_config)
  return env


def config_compose_environment(config):
  wrappers = []
  for w in config['wrappers'].split(','):
    assert(w in KNOWN_WRAPPERS)
    wrappers.append(KNOWN_WRAPPERS[w])

  return compose_environment(config, wrappers)


def kwargs_compose_environment(**config):
  return config_compose_environment(config)


# def sample_composed_environment():
#   return compose_environment(DEFAULT_EXTENDED_CONFIG, [
#     dump_wrapper,
#     checkpoint_wrapper,
#     ActionOrder,
#     lambda env, config: wrappers.SMMWrapper(env, config['channel_dimensions']),
#     single_agent_wrapper,
#     lambda env, config: FrameStack(env, config['stacked_frames']),
#     PlayerStackWrapper],
#   )
