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
from .wrappers.state_preserver import StatePreserver
from .wrappers.env_usage_stats import EnvUsageStatsTracker
from .wrappers.players_name import UpdateTeamNamesWrapper
from .wrappers.obs_pick import Simple115PickFirst
from .wrappers.env_utils import EnvUtilsWrapper
from .wrappers import evaluation_env
from .wrappers.discrete_as_to_multi import DiscreteToMulti

from .players.utils import PackedBitsObservation

from absl import logging

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

# added from gfootball

class Simple115StateWrapper(gym.ObservationWrapper):
  """A wrapper that converts an observation to 115-features state."""

  def __init__(self, env, fixed_positions=False):
    """Initializes the wrapper.
    Args:
      env: an envorinment to wrap
      fixed_positions: whether to fix observation indexes corresponding to teams
    Note: simple115v2 enables fixed_positions option.
    """
    gym.ObservationWrapper.__init__(self, env)
    shape = (self.env.unwrapped._config.number_of_players_agent_controls(), 115)
    self.observation_space = gym.spaces.Box(
        low=-1, high=1, shape=shape, dtype=np.float32)
    self._fixed_positions = fixed_positions

  def observation(self, observation):
    """Converts an observation into simple115 (or simple115v2) format.
    Args:
      observation: observation that the environment returns
    Returns:
      (N, 115) shaped representation, where N stands for the number of players
      being controlled.
    """
    final_obs = []
    for obs in observation:
      o = []
      if self._fixed_positions:
        for i, name in enumerate(['left_team', 'left_team_direction',
                                  'right_team', 'right_team_direction']):
          o.extend(obs[name].flatten())
          # If there were less than 11vs11 players we backfill missing values
          # with -1.
          if len(o) < (i + 1) * 22:
            o.extend([-1] * ((i + 1) * 22 - len(o)))
      else:
        o.extend(obs['left_team'].flatten())
        o.extend(obs['left_team_direction'].flatten())
        o.extend(obs['right_team'].flatten())
        o.extend(obs['right_team_direction'].flatten())

      # If there were less than 11vs11 players we backfill missing values with
      # -1.
      # 88 = 11 (players) * 2 (teams) * 2 (positions & directions) * 2 (x & y)
      if len(o) < 88:
        o.extend([-1] * (88 - len(o)))

      # ball position
      o.extend(obs['ball'])
      # ball direction
      o.extend(obs['ball_direction'])
      # one hot encoding of which team owns the ball
      if obs['ball_owned_team'] == -1:
        o.extend([1, 0, 0])
      if obs['ball_owned_team'] == 0:
        o.extend([0, 1, 0])
      if obs['ball_owned_team'] == 1:
        o.extend([0, 0, 1])

      active = [0] * 11
      if obs['active'] != -1:
        active[obs['active']] = 1
      o.extend(active)

      game_mode = [0] * 7
      game_mode[obs['game_mode']] = 1
      o.extend(game_mode)
      final_obs.append(o)
    return np.array(final_obs, dtype=np.float32)


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

  def step(self, action):
    return self.env.step(action)

  def reset(self):
    if (self._dump_frequency > 0 and
            (self._current_episode_number % self._dump_frequency == 0)):
      self.env.unwrapped._config.update(self._original_dump_config)
      # self.env.render()
    else:
      self.env.unwrapped._config.update({
          'write_video': False,
          'dump_full_episodes': False,
          'dump_scores': False
      })
      self.env.unwrapped.disable_render()
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
  assert 'scoring' in config['rewards'].split(',')
  DECAYING_CHECKPOINT_WRAPPER_PARAMS = ["checkpoint_base_reward", "decreasing_reward_treshold", "steps_to_get_from_checkpoints_to_scoring", "number_of_prev_episodes"]
  params = {}
  for param_name in DECAYING_CHECKPOINT_WRAPPER_PARAMS:
    if param_name in config:
      params[param_name] = config[param_name]
  return DecayingCheckpointRewardWrapper(env, **params)


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

def get_known_wrappers():
  result = {
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
    MultiHeadNet,
    'pack_bits':
    lambda env, config: PackedBitsObservation(env),
    'simple115_pick_first':
    Simple115PickFirst,
    'simple115v2':
    lambda env, config: Simple115StateWrapper(env, True),
    'eval': evaluation_env.EvalWrapper
  }
  result.update(get_loggers_dict())
  return result


KNOWN_WRAPPERS = get_known_wrappers()


def should_preserve_state(env_config):
  return env_config['base_logdir'] is not None and env_config['logs_enabled'] == True

def are_any_loggers(wrapper_names):
  for w in wrapper_names:
    if w.startswith('log_'):
      return True
  return False

def compose_environment(env_config):
  enable_log_api_for_config(env_config)  # we enable log api

  # we enable state preserving and env usage tracker by default
  wrappers = []
  wrappers.append(DiscreteToMulti)
  if should_preserve_state(env_config):
    wrappers.append(StatePreserver)
    wrappers.append(EnvUtilsWrapper)
    wrappers.append(EnvUsageStatsTracker)

  wrapper_names = env_config['wrappers'].split(',')
  if not are_any_loggers(wrapper_names):
    wrapper_names = ['log_all'] + wrapper_names
    logging.info('!!!!No loggers detected so added log_all: %s!!!!',
                 str(wrapper_names))

  #wrappers.append(UpdateTeamNamesWrapper)
  if env_config['actor_id'] == env_config.get('evaluation_actor', -1):
    logging.info('Creating evaluation actor!')
    env_config = evaluation_env.preprocess_config(env_config)
    wrappers.append(evaluation_env.EvalWrapper)
    #wrappCreatinger_names = evaluation_env.prepare_wrappers(wrapper_names)
  else:
    logging.warning('Not creating evaluation since current agent id %d is not %d.',
                    env_config['actor_id'], env_config.get('evaluation_actor', -1))

  for w in wrapper_names:
    assert(w in KNOWN_WRAPPERS)
    # do not apply log wrappers when logs not enabled
    # (only for speed improvements)
    if (not env_config['logs_enabled']) and w.startswith('log_'):
      continue

    wrappers.append(KNOWN_WRAPPERS[w])

  def extract_from_dict(dictionary, keys):
    return {new_k: dictionary[k] for (new_k, k) in keys if k in dictionary}

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
                   ('write_video', 'write_video'),
                   ('left_team_name', 'left_team_name'),
                   ('right_team_name', 'right_team_name')])
  if 'env_change_rate' not in env_config:
    env = football_env.FootballEnv(config.Config(football_config))
  else:
    env = create_recreatable_football(env_config, football_config)

  for w in wrappers:
    env = w(env, env_config)

  print(env_config)
  return env


def kwargs_compose_environment(**config):
  return compose_environment(config)


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
