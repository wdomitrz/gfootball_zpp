from .utils import LogBasicTracker
from ..utils import scalar_to_list

import tensorflow as tf


class LogActionStats(LogBasicTracker):
    def __init__(self, env, config):
        LogBasicTracker.__init__(self, env, config)
        self._action_list = []

        self._num_players = len(scalar_to_list(
            self.env.action_space.sample()))

    def reset(self):
        observation = super(LogActionStats, self).reset()
        if self._action_list != []:
            actions = tf.Variable(self._action_list, dtype=tf.int64)
            with self.summary_writer.as_default():
                for pid in range(self._num_players):
                    tf.summary.histogram('actions/player_{}'.format(pid),
                                         actions[:, pid], self.env_resets)

        self._action_list = []
        return observation

    def step(self, action):
        result = super(LogActionStats, self).step(action)
        self._action_list.append(scalar_to_list(action))
        return result
