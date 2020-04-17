from .utils import LogBasicTracker, EnvLogSteppingModes, get_opponent_name
from gfootball_zpp.utils.misc import get_with_prec
import numpy as np
import math

class LogLocalAdvantageStats(LogBasicTracker):
    def _trace_vars_reset(self):
        self._first_team_advantages_sum = np.zeros(self._num_circles, dtype=np.float64)

    def _update_stats(self, observation):
        ball_pos = observation[0]['ball'][0:2]
        player_pos = [[],[]]
        player_pos[0] = observation[0]['left_team']
        player_pos[1] = observation[0]['right_team']
        ball_distances = []
        for team_pos in player_pos:
            ball_team_dists = []
            for pl_pos in team_pos:
                diff_x = pl_pos[0] - ball_pos[0]
                diff_y = pl_pos[1] - ball_pos[1]
                dist = math.sqrt(pow(diff_x, 2) + pow(diff_y, 2))
                ball_team_dists.append(dist)
            ball_distances.append(ball_team_dists)

        #print(ball_distances)
        ball_distances = np.asarray(ball_distances, dtype=np.float64)
        for (rId, r) in enumerate(self._circles):
            first_team_in = np.where(ball_distances[0] <= r)[0].size
            second_team_in = np.where(ball_distances[1] <= r)[0].size
            #print(np.where(ball_distances[0] <= r))
            #print(np.where(ball_distances[1] <= r))
            advantage = first_team_in - second_team_in
            self._first_team_advantages_sum[rId] += advantage

    def _write_logs(self, category):
        episode_steps = self.env_episode_steps

        avg_first_team_adv = self._first_team_advantages_sum / episode_steps

        for (rId, r) in enumerate(self._circles):
            precR = get_with_prec(r)
            self.summary_writer.write_scalar(
                '{}/first_team/avg_advantage_r_{}'.format(category, precR), avg_first_team_adv[rId])
        

    def __init__(self, env, config):
        LogBasicTracker.__init__(self, env, config)

        self._circles = [0.05, 0.1, 0.2, 0.3]
        self._num_circles = len(self._circles)
        self._trace_vars_reset()

        self.summary_writer.set_stepping(EnvLogSteppingModes.env_resets)

    def reset(self):
        env_episode_steps = self.env_episode_steps
        if env_episode_steps != 0:
            self._write_logs('local_advantages')
            self._write_logs('per_opponent_local_advantages/' +
                             get_opponent_name(self.env))

        observation = super(LogLocalAdvantageStats, self).reset()

        self._trace_vars_reset()
        # self._update_stats(observation) # we do not count first observation
        return observation

    def step(self, action):
        observation, reward, done, info = super(LogLocalAdvantageStats,
                                                self).step(action)
        self._update_stats(observation)
        return observation, reward, done, info
