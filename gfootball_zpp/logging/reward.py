from .utils import LogBasicTracker
from ..utils import scalar_to_list

import tensorflow as tf

class LogPerPlayerReward(LogBasicTracker):

    def _trace_vars_reset(self):
        if self._num_rewards is not None:
            self._rewards = tf.zeros(self._num_rewards, dtype=tf.float32)
        else:
            self._rewards = None

    def _update_step(self, reward):
        if self._num_rewards is None:
            self._num_rewards = len(scalar_to_list(reward))
            self._trace_vars_reset()

        reward = tf.Variable(reward, dtype=tf.float32)
        reward.shape.assert_has_rank(self._rewards.shape.ndims)
        self._rewards = self._rewards + reward

        if self.env_episode_steps % self._step_log_freq == 0:
             with self.summary_writer.as_default():
                for rid in range(self._num_rewards):
                    tf.summary.scalar('reward/step/reward_{}'.format(rid),
                                      self._rewards[rid],
                                      self.env_total_steps)

    def __init__(self, env, config):
        LogBasicTracker.__init__(self, env, config)

        self._num_rewards = None
        self._step_log_freq = int(config['step_log_freq'])

        self._trace_vars_reset()

    def reset(self):
        observation = super(LogPerPlayerReward, self).reset()
        if self._rewards != None:
            with self.summary_writer.as_default():
                for rid in range(self._num_rewards):
                    tf.summary.scalar('reward/game/reward_{}'.format(rid),
                                      self._rewards[rid], self.env_resets)

        self._trace_vars_reset()
        return observation

    def step(self, action):
        observation, reward, done, info = super(
            LogPerPlayerReward, self).step(action)
        self._update_step(reward)
        return observation, reward, done, info
