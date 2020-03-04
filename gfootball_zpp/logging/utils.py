from ..utils import extract_obj_attributes, extract_from_dict

import gym
import tensorflow as tf
import os
import threading
import time

def make_logdir_dirs(logdir):
    tf.io.gfile.makedirs(logdir)


def upload_logs(local_logdir, remote_logdir):
    while True:
        local_files = tf.io.gfile.listdir(local_logdir)
        remote_files = tf.io.gfile.listdir(remote_logdir)
        diff = list(set(local_files) - set(remote_files))
        for f in diff:
            tf.io.gfile.copy(os.path.join(local_logdir, f),
                             os.path.join(remote_logdir, f))
        time.sleep(1)


def fix_remote_logdir(logdir):
    """gs://... directory does not work with gfootball
    this function fixes that by creating local logdir
    and syncing it with remote one"""

    if logdir.startswith('gs://'):
        pruned_logdir = logdir.replace('/', '_').replace(':', '')
        local_logdir = '/tmp/env_log/' + pruned_logdir
        remote_logdir = logdir
        make_logdir_dirs(local_logdir)
        make_logdir_dirs(remote_logdir)
        t = threading.Thread(target=upload_logs, args=(
            local_logdir, remote_logdir))
        t.start()
        return local_logdir
    else:
        return logdir


def enable_video_logs(config):
    config['enable_goal_videos'] = True
    config['enable_full_episode_videos'] = True
    config['write_video'] = True


class LogAPI(gym.Wrapper):
    """Transparent wrapper that enables logging

    Functionalities:
    * decides whenever to enable logs ('decide_to_log_fn')
    * adds specific logdirs to config (based of 'base_logdir')
    etc
    """

    def __init__(self, env, config):
        gym.Wrapper.__init__(self, env)
        self._actor_id = config['actor_id']
        self._base_logdir = config['base_logdir']

        if 'decide_to_log_fn' in config:
            self._decide_if_log_fn = eval(config['decide_to_log_fn'])
        else:
            self._decide_if_log_fn = lambda actor_id: actor_id == 0

        if (self._actor_id is not None) and \
           (self._base_logdir is not None) and \
           (self._decide_if_log_fn(self._actor_id)):
            dumps_logdir = os.path.join(self._base_logdir, 'env_dumps')
            config['logdir'] = fix_remote_logdir(dumps_logdir)

            config['tb_logdir'] = os.path.join(self._base_logdir, 'env_tb')
            make_logdir_dirs(config['tb_logdir'])

            if 'step_log_freq' not in config:
                config['step_log_freq'] = 10 * config['dump_frequency']

            enable_video_logs(config)

            config['summary_writer'] = tf.summary.create_file_writer(
                config['tb_logdir'],
                flush_millis=20000,
                max_queue=1000)
            config['logs_enabled'] = True
        else:
            config['summary_writer'] = tf.summary.create_noop_writer()
            config['logs_enabled'] = False
            config['logdir'] = ''

    def __getattr__(self, attr):
        return getattr(self.env, attr)


def get_summary_writer(config):
    return config['summary_writer']

class LogBasicTracker(gym.Wrapper):
    """Base class for log wrappers
    Keeps track of basic statistics used by most wrappers
    """

    def __init__(self, env, config):
        gym.Wrapper.__init__(self, env)
        self.env_resets = 0
        self.env_episode_steps = 0
        self.env_total_steps = 0
        self.summary_writer = get_summary_writer(config)

    def reset(self):
        self.env_resets += 1
        self.env_episode_steps = 0
        return self.env.reset()

    def step(self, action):
        self.env_episode_steps += 1
        self.env_total_steps += 1
        return self.env.step(action)

def extract_data_from_low_level_env_cfg(env_config):
    data = []
    data.extend(extract_from_dict(env_config._values,
                                  ['action_set',
                                   'players',
                                   'level']))
    data.extend(extract_obj_attributes(env_config.ScenarioConfig(),
                                       [# 'ball_position',
                                        'deterministic',
                                        'end_episode_on_out_of_play',
                                        'end_episode_on_possession_change',
                                        'end_episode_on_score',
                                        'game_duration',
                                        # 'game_engine_random_seed',
                                        # 'left_agents',
                                        # 'left_team',
                                        'left_team_difficulty',
                                        'offsides',
                                        # 'right_agents',
                                        # 'right_team',
                                        'right_team_difficulty']))

    def second_to_string(p):
        f, s = p
        return f, str(s)

    return list(map(second_to_string, data))
