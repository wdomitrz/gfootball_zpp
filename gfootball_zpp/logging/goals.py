from .utils import LogBasicTracker, EnvLogSteppingModes, get_opponent_name

import tensorflow as tf


class LogGoalStats(LogBasicTracker):
    """ WARNING this wrapper returns approximate results
    due to the fact that observation is generated after x
    game engine steps """
    def _trace_vars_reset(self):
        self._goals = [0, 0]
        self._own_goals = [0, 0]
        self._last_score = [0, 0]
        self._last_team_with_ball = 0

    def _update_stats(self, observation):
        current_score = observation[0]['score']
        team_scored = None
        for tid in [0, 1]:
            assert abs(current_score[tid] - self._last_score[tid]) <= 1
            if current_score[tid] > self._last_score[tid]:
                team_scored = tid

        self._last_score = current_score

        if team_scored is not None:
            if team_scored == self._last_team_with_ball:
                self._goals[team_scored] += 1
                print("GOL", team_scored)
            else:
                self._own_goals[self._last_team_with_ball] += 1
                print("Samoboj", self._last_team_with_ball)

        current_ball_owned_team = observation[0]['ball_owned_team']
        if current_ball_owned_team != -1:
            self._last_team_with_ball = current_ball_owned_team
            assert self._last_team_with_ball == 0 or self._last_team_with_ball == 1

    def _write_goal_log(self, category, team_id):
        team_name = self._team_names[team_id]
        self.summary_writer.write_scalar('{}/{}/shot'.format(category, team_name),
                                         self._goals[team_id])
        self.summary_writer.write_scalar('{}/{}/own'.format(category, team_name),
                                         self._own_goals[team_id])

    def __init__(self, env, config):
        LogBasicTracker.__init__(self, env, config)

        self._trace_vars_reset()
        self._team_names = ['first_team', 'second_team']

        self.summary_writer.set_stepping(EnvLogSteppingModes.env_resets)

    def reset(self):
        env_episode_steps = self.env_episode_steps
        if env_episode_steps != 0:
            for (tid, _) in enumerate(self._team_names):
                self._write_goal_log('goals', tid)
                self._write_goal_log('per_opponent_goals/' + get_opponent_name(self.env), tid)

        observation = super(LogGoalStats, self).reset()

        self._trace_vars_reset()
        self._update_stats(observation)
        return observation

    def step(self, action):
        observation, reward, done, info = super(LogGoalStats,
                                                self).step(action)
        self._update_stats(observation)
        return observation, reward, done, info
