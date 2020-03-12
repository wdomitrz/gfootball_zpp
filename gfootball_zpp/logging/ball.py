from .utils import LogBasicTracker, EnvLogSteppingModes

import tensorflow as tf


class LogBallOwningTeam(LogBasicTracker):
    """ WARNING this wrapper returns approximate results
    due to the fact that observation is generated after x
    game engine steps """
    def _trace_vars_reset(self):
        self._first_team_time = 0  # in environment steps
        self._second_team_time = 0  # in environment steps

    def _update_stats(self, observation):
        if observation[0]['ball_owned_team'] == 0:
            self._first_team_time += 1
        elif observation[0]['ball_owned_team'] == 1:
            self._second_team_time += 1

    def __init__(self, env, config):
        LogBasicTracker.__init__(self, env, config)

        self._trace_vars_reset()

        self.summary_writer.set_stepping(EnvLogSteppingModes.env_resets)

    def reset(self):
        env_episode_steps = self.env_episode_steps
        if env_episode_steps != 0:
            self.summary_writer.write_scalar('ball/owning_first_team',
                                             self._first_team_time / \
                                             env_episode_steps)
            self.summary_writer.write_scalar('ball/owning_second_team',
                                             self._second_team_time / \
                                             env_episode_steps)
            ball_free_time = env_episode_steps - \
                             (self._first_team_time +
                              self._second_team_time)
            self.summary_writer.write_scalar(
                'ball/not_owning', ball_free_time / env_episode_steps)

        observation = super(LogBallOwningTeam, self).reset()

        self._trace_vars_reset()
        # self._update_stats(observation) - we do not count initial observation
        return observation

    def step(self, action):
        observation, reward, done, info = super(LogBallOwningTeam,
                                                self).step(action)
        self._update_stats(observation)
        return observation, reward, done, info
