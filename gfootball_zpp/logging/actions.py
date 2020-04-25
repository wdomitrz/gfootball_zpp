from .utils import LogBasicTracker, EnvLogSteppingModes, player_with_ball_action, get_opponent_name
from ..utils.misc import scalar_to_list, get_max_discrete_action, pretty_list_of_pairs_to_string
from gfootball.env.football_action_set import named_action_from_action_set, get_action_set

import numpy as np
import tensorflow as tf


class LogActionStats(LogBasicTracker):
    """ This is a low level wrapper """
    def _update_action_counts(self, observation, action):
        for pid, a in enumerate(action):
            self._action_counter[pid][a] += 1

        ball_action = player_with_ball_action(observation, action)
        if ball_action is not None:
            self._ball_action_counter[ball_action] += 1

    def _get_action_set(self):
        return get_action_set(self.env.unwrapped._config._values)

    def _get_action_name(self, action):
        return named_action_from_action_set(self._get_action_set(), action)

    def __init__(self, env, config):
        LogBasicTracker.__init__(self, env, config)

        self._num_players = len(scalar_to_list(self.env.action_space.sample()))

        self._discrete_actions = get_max_discrete_action(self.env)
        self._action_counter = np.zeros(shape=(self._num_players,
                                               self._discrete_actions),
                                        dtype=np.int64)

        self._ball_action_counter = np.zeros(shape=(self._discrete_actions, ),
                                             dtype=np.int64)

        self.summary_writer.set_stepping(EnvLogSteppingModes.env_resets)

    def _write_logs(self, category):
        def make_text_log_data(actions, name):
            text_actions = [
                ('action:**{}** aka:**{}**'.format(aid,
                                                   self._get_action_name(aid)),
                 actions[aid]) for aid in range(self._discrete_actions)
            ]
            return '## {}  \n'.format(name) + \
                pretty_list_of_pairs_to_string(text_actions)

        text_log = '# Players action stats  \n'
        actions = self._action_counter
        for pid in range(self._num_players):
            self.summary_writer.write_bars(
                '{}/proportions_player_{}'.format(category, pid), actions[pid])
            text_log += make_text_log_data(actions[pid],
                                           'player_{}'.format(pid))

        self.summary_writer.write_text('{}/players'.format(category), text_log)

        ball_actions = self._ball_action_counter
        self.summary_writer.write_bars(
            '{}/proportions_ball_owned_controlled_player'.format(category),
            ball_actions)
        ball_text = text_log = '# Ball owning player action stats  \n'
        ball_text += make_text_log_data(ball_actions,
                                        'ball_owned_controlled_player')
        self.summary_writer.write_text('{}/ball_text'.format(category),
                                       ball_text)

    def reset(self):

        if self.env_episode_steps != 0:
            self._write_logs('actions')
            self._write_logs('per_opponent_actions/' +
                             get_opponent_name(self.env))

        observation = super(LogActionStats, self).reset()

        self._action_counter *= 0
        self._ball_action_counter *= 0
        return observation

    def step(self, action):
        observation, reward, done, info = super(LogActionStats,
                                                self).step(action)
        self._update_action_counts(observation, scalar_to_list(action))
        return observation, reward, done, info
