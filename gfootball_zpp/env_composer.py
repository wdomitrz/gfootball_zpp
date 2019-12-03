from gfootball.env import config
from gfootball.env import football_env
from gfootball.env import wrappers

from .wrappers.action_wrappers import ActionOrder
from .wrappers.player_stack_wrapper import PlayerStackWrapper
import collections
import gym
import numpy as np
import subprocess
import os
import threading
import time
import tensorflow as tf

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

def dump_wrapper(env, config):
  if config['dump_frequency'] > 1:
    return wrappers.PeriodicDumpWriter(env, config['dump_frequency'])
  else:
    return env

def checkpoint_wrapper(env, config):
  assert 'scoring' in config['rewards'].split(',')
  if 'checkpoints' in config['rewards'].split(','):
    return wrappers.CheckpointRewardWrapper(env)
  else:
    return env

def single_agent_wrapper(env, config):
  if (config['number_of_left_players_agent_controls'] +
      config['number_of_right_players_agent_controls'] == 1):
     env = wrappers.SingleAgentObservationWrapper(env)
     env = wrappers.SingleAgentRewardWrapper(env)
     return env
  else:
    return env

KNOWN_WRAPPERS = {
  'periodic_dump': dump_wrapper,
  'checkpoint_score': checkpoint_wrapper,
  'single_agent': single_agent_wrapper,
  'obs_extract': lambda env, config: wrappers.SMMWrapper(env, config['channel_dimensions']),
  'obs_stack': lambda env, config: FrameStack(env, config['stacked_frames']),
  'action_order': ActionOrder,
  'psw': PlayerStackWrapper,
}

def compose_environment(env_config, wrappers):
  def extract_from_dict(dictionary, keys):
    return {new_k: dictionary[k] for (new_k, k) in keys}

  players = [('agent:left_players=%d,right_players=%d' % (
    env_config['number_of_left_players_agent_controls'],
    env_config['number_of_right_players_agent_controls']))]
  if env_config['extra_players'] is not None:
    players.extend(env_config['extra_players'])
  env_config['players'] = players
  c = config.Config(extract_from_dict(env_config,
                             [('enable_sides_swap', 'enable_sides_swap'),
                              ('dump_full_episodes', 'enable_full_episode_videos'),
                              ('dump_scores', 'enable_goal_videos'),
                              ('level', 'env_name'),
                              ('players', 'players'),
                              ('render', 'render'),
                              ('tracesdir', 'logdir'),
                              ('write_video', 'write_video')]))
  env = football_env.FootballEnv(c)

  for w in wrappers:
    env = w(env, env_config)

  return env

def upload_logs(local_logdir, remote_logdir):
  tf.io.gfile.makedirs(remote_logdir)
  while True:
    local_files = tf.io.gfile.listdir(local_logdir)
    remote_files = tf.io.gfile.listdir(remote_logdir)
    diff = list(set(local_files) - set(remote_files))
    for f in diff:
      tf.io.gfile.copy(os.path.join(local_logdir, f),
                       os.path.join(remote_logdir, f))
    time.sleep(1)

def remote_logs(config):
  if config['logdir'] != '' and config['logdir'].startswith('gs://'):
    pruned_logdir = config['logdir'].replace('/', '_').replace(':', '')
    local_logdir = '/tmp/env_log/' + pruned_logdir
    os.makedirs(local_logdir)
    remote_logdir = config['logdir']
    t = threading.Thread(target=upload_logs, args=(local_logdir, remote_logdir))
    t.start()
    # subprocess.Popen(['gsutil', 'rsync', local_logdir, remote_logdir])
    config['logdir'] = local_logdir

def log_videos_when_logging(config):
  if config['logdir'] != '':
    config['enable_goal_videos'] = True
    config['enable_full_episode_videos'] = True
    config['write_video'] = True


def config_compose_environment(config):
  remote_logs(config)
  log_videos_when_logging(config)
  wrappers = []
  for w in config['wrappers'].split(','):
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
