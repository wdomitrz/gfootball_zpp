from .utils import LogBasicTracker, EnvLogSteppingModes, get_opponent_name
import tensorflow as tf


NUM_GAME_MODES = 7
GAME_MODE_DICT = dict([
    (0, 'Normal'),
    (1, 'KickOff'),
    (2, 'GoalKick'),
    (3, 'FreeKick'),
    (4, 'Corner'),
    (5, 'ThrowIn'),
    (6, 'Penalty')
])

class LogGameModeStats(LogBasicTracker):
    def _trace_vars_reset(self):
        self._game_mode_counter = [0] * NUM_GAME_MODES
        self._prev_game_mode = None

    def _update_stats(self, observation):
        current_game_mode = observation[0]['game_mode']
        if self._prev_game_mode != current_game_mode:
            self._game_mode_counter[current_game_mode] += 1
            self._prev_game_mode = current_game_mode
            print("gamemode changed to:", current_game_mode, " aka ", GAME_MODE_DICT[current_game_mode])
        
    def _write_logs(self, category):
        for gmid in range(NUM_GAME_MODES):
            game_mode_name = GAME_MODE_DICT[gmid]
            game_mode_count = self._game_mode_counter[gmid]
            self.summary_writer.write_scalar('{}/{}'.format(category, game_mode_name),
                                             game_mode_count)

    def __init__(self, env, config):
        LogBasicTracker.__init__(self, env, config)

        self._trace_vars_reset()

        self.summary_writer.set_stepping(EnvLogSteppingModes.env_resets)

    def reset(self):
        env_episode_steps = self.env_episode_steps
        if env_episode_steps != 0:
            self._write_logs('game_modes')
            self._write_logs('per_opponent_game_modes/' + get_opponent_name(self.env))

        observation = super(LogGameModeStats, self).reset()

        self._trace_vars_reset()
        self._update_stats(observation)
        return observation

    def step(self, action):
        observation, reward, done, info = super(LogGameModeStats,
                                                self).step(action)
        self._update_stats(observation)
        return observation, reward, done, info
