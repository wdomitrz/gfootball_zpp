from .utils import LogBasicTracker, EnvLogSteppingModes
from ..utils import scalar_to_list

import tensorflow as tf


class LogActionStats(LogBasicTracker):
    def __init__(self, env, config):
        LogBasicTracker.__init__(self, env, config)
        self._action_list = []

        self._num_players = len(scalar_to_list(
            self.env.action_space.sample()))

        self.summary_writer.set_stepping(EnvLogSteppingModes.env_resets)

    def reset(self):
        if self._action_list != []:
            actions = tf.Variable(self._action_list, dtype=tf.int64)
            for pid in range(self._num_players):
                self.summary_writer.write_histogram('actions/player_{}'.format(pid),
                                                    actions[:, pid])

        observation = super(LogActionStats, self).reset()

        self._action_list = []
        return observation

    def step(self, action):
        result = super(LogActionStats, self).step(action)
        self._action_list.append(scalar_to_list(action))
        return result
