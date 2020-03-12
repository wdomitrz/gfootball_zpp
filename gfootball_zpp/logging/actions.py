from .utils import LogBasicTracker, EnvLogSteppingModes
from ..utils.misc import scalar_to_list, get_max_discrete_action, pretty_list_of_pairs_to_string
from gfootball.env.football_action_set import named_action_from_action_set, get_action_set

import numpy as np
import tensorflow as tf


class LogActionStats(LogBasicTracker):
    """ This is a low level wrapper """

    def _update_action_counts(self, action):
        for pid, a in enumerate(action):
            self._action_counter[pid][a] += 1

    def _get_action_set(self):
        return get_action_set(self.env._config._values)

    def _get_action_name(self, action):
        return named_action_from_action_set(self._get_action_set(), action)

    def __init__(self, env, config):
        LogBasicTracker.__init__(self, env, config)

        self._num_players = len(scalar_to_list(
            self.env.action_space.sample()))

        self._discrete_actions = get_max_discrete_action(self.env)
        self._action_counter = np.zeros(shape=(self._num_players,
                                               self._discrete_actions),
                                        dtype=np.int64)

        self.summary_writer.set_stepping(EnvLogSteppingModes.env_resets)

    def __getattr__(self, attr):
        return getattr(self.env, attr)

    def reset(self):
        text_log = '# Players action stats  \n'
        if self.env_episode_steps != 0:
            actions = self._action_counter
            for pid in range(self._num_players):
                self.summary_writer.write_bars('actions/proportions_player_{}'.format(pid),
                                               actions[pid])

                text_actions = [('action:**{}** aka:**{}**'.format(
                    aid, self._get_action_name(aid)),
                                 actions[pid][aid])
                                for aid in range(self._discrete_actions)]
                text_log += '## For player_{}  \n'.format(pid) + \
                            pretty_list_of_pairs_to_string(text_actions)

            self.summary_writer.write_text('actions/players', text_log)

        observation = super(LogActionStats, self).reset()

        self._action_counter *= 0
        return observation

    def step(self, action):
        result = super(LogActionStats, self).step(action)
        self._update_action_counts(scalar_to_list(action))
        return result
