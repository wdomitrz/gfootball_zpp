from .utils import LogBasicTracker, EnvLogSteppingModes, get_opponent_name
import numpy as np


class LogCardsStatsTeam(LogBasicTracker):
    def _trace_vars_reset(self):
        self._first_team_yellow_cards = None
        self._first_team_red_cards = None
        self._second_team_yellow_cards = None
        self._second_team_red_cards = None

    def _update_stats(self, observation):
        def convert_red_cards(team_active):
            return list(map(lambda x: int(not x), team_active))

        self._first_team_yellow_cards = observation[0]['left_team_yellow_card']
        self._first_team_red_cards = convert_red_cards(
            observation[0]['left_team_active'])
        self._second_team_yellow_cards = observation[0][
            'right_team_yellow_card']
        self._second_team_red_cards = convert_red_cards(
            observation[0]['right_team_active'])

    def _write_logs(self, category):
        first_team_yellow = np.sum(self._first_team_yellow_cards)
        first_team_red = np.sum(self._first_team_red_cards)
        second_team_yellow = np.sum(self._second_team_yellow_cards)
        second_team_red = np.sum(self._second_team_red_cards)
        self.summary_writer.write_scalar(
            '{}/first_team/yellow_cards'.format(category), first_team_yellow)
        self.summary_writer.write_scalar(
            '{}/first_team/red_cards'.format(category), first_team_red)
        self.summary_writer.write_scalar(
            '{}/second_team/yellow_cards'.format(category), second_team_yellow)
        self.summary_writer.write_scalar(
            '{}/second_team/red_cards'.format(category), second_team_red)

    def __init__(self, env, config):
        LogBasicTracker.__init__(self, env, config)

        self._trace_vars_reset()

        self.summary_writer.set_stepping(EnvLogSteppingModes.env_resets)

    def reset(self):
        env_episode_steps = self.env_episode_steps
        if env_episode_steps != 0:
            self._write_logs('cards')
            self._write_logs('per_opponent_cards/' +
                             get_opponent_name(self.env))

        observation = super(LogCardsStatsTeam, self).reset()

        self._trace_vars_reset()
        self._update_stats(observation)
        return observation

    def step(self, action):
        observation, reward, done, info = super(LogCardsStatsTeam,
                                                self).step(action)
        self._update_stats(observation)
        return observation, reward, done, info
