from gfootball_zpp.utils.misc import extract_obj_attributes, extract_from_dict, make_dirs

import gym
import tensorflow as tf
import os
import threading
import time

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
        make_dirs(local_logdir)
        make_dirs(remote_logdir)
        t = threading.Thread(target=upload_logs,
                             args=(local_logdir, remote_logdir))
        t.start()
        return local_logdir
    else:
        return logdir


def enable_video_logs(config):
    config['enable_goal_videos'] = True
    config['enable_full_episode_videos'] = True
    config['write_video'] = True


def log_api(config):
    """
    Functionalities:
    * decides whenever to enable logs ('decide_to_log_fn')
    * adds specific logdirs to config (based of 'base_logdir')
    etc"""
    actor_id = config['actor_id']
    base_logdir = config['base_logdir']

    if 'decide_to_log_fn' in config:
        decide_if_log_fn = eval(config['decide_to_log_fn'])
    else:
        decide_if_log_fn = lambda actor_id: actor_id == 0

    if 'step_log_freq' not in config:
        config['step_log_freq'] = 10 * config['dump_frequency']

    if 'reset_log_freq' not in config:
        config['reset_log_freq'] = 1

    if (actor_id is not None) and \
       (base_logdir is not None) and \
       (decide_if_log_fn(actor_id)):
        dumps_logdir = os.path.join(base_logdir, 'env_dumps')
        config['logdir'] = fix_remote_logdir(dumps_logdir)

        config['tb_logdir'] = os.path.join(base_logdir, 'env_tb')
        make_dirs(config['tb_logdir'])

        enable_video_logs(config)

        config['tf_summary_writer'] = tf.summary.create_file_writer(
            config['tb_logdir'], flush_millis=20000, max_queue=1000)
        config['logs_enabled'] = True
    else:
        config['tf_summary_writer'] = tf.summary.create_noop_writer()
        config['logs_enabled'] = False
        config['logdir'] = ''


import enum


class SummaryWriterBase():
    def __init__(self):
        pass

    def is_log_time(self):
        return NotImplementedError

    def set_stepping(self, step):
        return NotImplementedError

    def write_scalar(self, name, scalar):
        return NotImplementedError

    def write_text(self, name, text):
        return NotImplementedError

    def write_histogram(self, name, raw_data):
        return NotImplementedError

    def write_bars(self, name, data):
        return NotImplementedError


class EnvLogSteppingModes(enum.Enum):
    provided = 1
    env_resets = 2
    env_total_steps = 3


class EnvSummaryWriterBase(SummaryWriterBase):
    def __init__(self, log_tracker, config):
        SummaryWriterBase.__init__(self)
        self._step_log_freq = config['step_log_freq']
        self._reset_log_freq = config['reset_log_freq']
        self._logs_enabled = config['logs_enabled']

        self._log_tracker = log_tracker
        self._current_stepping = EnvLogSteppingModes.env_resets

        self._stepping_modes = {
            EnvLogSteppingModes.provided:
            None,
            EnvLogSteppingModes.env_resets:
            lambda: self._log_tracker.env_resets,
            EnvLogSteppingModes.env_total_steps:
            lambda: self._log_tracker.env_total_steps
        }

    def is_log_time(self):
        result = self._logs_enabled
        if self._current_stepping != EnvLogSteppingModes.provided:
            result = result and (self._log_tracker.env_resets %
                                 self._reset_log_freq == 0)
            if self._current_stepping == EnvLogSteppingModes.env_total_steps:
                result = result and (self._log_tracker.env_episode_steps %
                                     self._step_log_freq == 0)
            return result
        else:
            return result

    def set_stepping(self, stepping, step=None):
        if stepping == EnvLogSteppingModes.provided:
            self._stepping_modes[stepping] = lambda: step
        elif stepping == EnvLogSteppingModes.env_resets or \
                stepping == EnvLogSteppingModes.env_total_steps:
            assert step is None
        else:
            raise Exception('Stepping: ' + str(stepping) + ' not supported')

        self._current_stepping = stepping

    def get_current_step(self):
        return self._stepping_modes[self._current_stepping]()


import tensorboard


class EnvTFSummaryWriter(EnvSummaryWriterBase):
    def __init__(self, log_tracker, config):
        EnvSummaryWriterBase.__init__(self, log_tracker, config)
        self._tf_summary_writer = config['tf_summary_writer']

    def write_scalar(self, name, scalar):
        if not self.is_log_time():
            return

        with self._tf_summary_writer.as_default():
            tf.summary.scalar(name, scalar, self.get_current_step())

    def write_text(self, name, text):
        if not self.is_log_time():
            return

        with self._tf_summary_writer.as_default():
            tf.summary.text(name, text, self.get_current_step())

    def write_histogram(self, name, raw_data, buckets=None):
        """ Warning strongly inaccurate https://github.com/tensorflow/tensorflow/issues/36128 """
        if not self.is_log_time():
            return

        with self._tf_summary_writer.as_default():
            tf.summary.histogram('warning_strongly_inaccurate_' + name,
                                 raw_data,
                                 self.get_current_step(),
                                 buckets=buckets)

    def write_bars(self, name, data, span_scale_factor=1.0, offset=0.0):
        """ Warning strongly inaccurate https://github.com/tensorflow/tensorboard/issues/1803 """
        if not self.is_log_time():
            return

        with self._tf_summary_writer.as_default():
            data = tf.Variable(data, dtype=tf.float64)
            assert len(data.shape) == 1
            num_buckets = data.shape[0]
            # data = tf.expand_dims(tf.expand_dims(data, axis=-1), axis=-1)
            counts = data
            left = (tf.range(num_buckets, dtype=tf.float64) +
                    offset) * span_scale_factor
            left -= 0.5 * span_scale_factor
            right = tf.concat([
                left[1:],
                tf.Variable([left[-1] + (1.0 * span_scale_factor)],
                            dtype=tf.float64)
            ],
                              axis=0)
            data = tf.stack([left, right, counts], axis=1)
            data = tf.reshape(data, shape=(num_buckets, 3))

            summary_metadata = tensorboard.plugins.histogram.metadata.create_summary_metadata(
                display_name=None, description=None)
            summary_scope = (getattr(tf.summary.experimental, 'summary_scope',
                                     None) or tf.summary.summary_scope)
            with summary_scope(
                    'warning_strongly_inaccurate_' + name,
                    'histogram_summary',
                    values=[data, num_buckets,
                            self.get_current_step()]) as (tag, _):
                tf.summary.write(tag=tag,
                                 tensor=data,
                                 step=self.get_current_step(),
                                 metadata=summary_metadata)


class LogBasicTracker(gym.Wrapper):
    """Base class for log wrappers
    Keeps track of basic statistics used by most wrappers
    """
    def __init__(self, env, config):
        gym.Wrapper.__init__(self, env)
        self._tracker = config['env_usage_stats']
        self.summary_writer = EnvTFSummaryWriter(self, config)

    @property
    def env_resets(self):
        return self._tracker.env_resets

    @property
    def env_episode_steps(self):
        return self._tracker.env_episode_steps

    @property
    def env_total_steps(self):
        return self._tracker.env_total_steps


def extract_data_from_low_level_env_cfg(env_config):
    data = []
    data.extend(
        extract_from_dict(env_config._values,
                          ['action_set', 'players', 'level']))
    data.extend(
        extract_obj_attributes(
            env_config.ScenarioConfig(),
            [  # 'ball_position',
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
                'right_team_difficulty'
            ]))

    def second_to_string(p):
        f, s = p
        return f, str(s)

    return list(map(second_to_string, data))
