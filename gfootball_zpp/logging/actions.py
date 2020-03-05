from .utils import LogBasicTracker, EnvLogSteppingModes
from ..utils import scalar_to_list, get_max_discrete_action, pretty_list_of_pairs_to_string

import numpy as np


class LogActionStats(LogBasicTracker):

    def _update_action_counts(self, action):
        for pid, a in enumerate(action):
            self._action_counter[pid][a] += 1

    def __init__(self, env, config):
        LogBasicTracker.__init__(self, env, config)

        self._num_players = len(scalar_to_list(
            self.env.action_space.sample()))

        self._discrete_actions = get_max_discrete_action(self.env)
        self._action_counter = np.zeros(shape=(self._num_players, self._discrete_actions), dtype=np.int64)

        self.summary_writer.set_stepping(EnvLogSteppingModes.env_resets)

    def reset(self):
        text_log = ''
        if self.env_episode_steps != 0:
            actions = self._action_counter
            for pid in range(self._num_players):
                self.summary_writer.write_bars('actions/player_{}'.format(pid),
                                                    actions[pid])

                text_actions = [('action_{}'.format(aid), actions[pid][aid]) for aid in range(self._discrete_actions)]
                text_log += '## For player_{}  \n'.format(pid) + pretty_list_of_pairs_to_string(text_actions)

            self.summary_writer.write_text('actions/players', text_log)


        observation = super(LogActionStats, self).reset()

        self._action_counter *= 0
        return observation

    def step(self, action):
        result = super(LogActionStats, self).step(action)
        self._update_action_counts(action)
        return result
