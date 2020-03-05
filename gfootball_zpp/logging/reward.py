from .utils import LogBasicTracker, EnvLogSteppingModes
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

        self.summary_writer.set_stepping(EnvLogSteppingModes.env_total_steps)
        for rid in range(self._num_rewards):
            self.summary_writer.write_scalar('reward/step/reward_{}'.format(rid),
                              self._rewards[rid])

    def __init__(self, env, config):
        LogBasicTracker.__init__(self, env, config)

        self._num_rewards = None

        self._trace_vars_reset()

    def reset(self):

        if self._rewards != None:
            self.summary_writer.set_stepping(EnvLogSteppingModes.env_resets)
            for rid in range(self._num_rewards):
                self.summary_writer.write_scalar('reward/game/reward_{}'.format(rid),
                                    self._rewards[rid])

        observation = super(LogPerPlayerReward, self).reset()

        self._trace_vars_reset()
        return observation

    def step(self, action):
        observation, reward, done, info = super(
            LogPerPlayerReward, self).step(action)
        self._update_step(reward)
        return observation, reward, done, info
