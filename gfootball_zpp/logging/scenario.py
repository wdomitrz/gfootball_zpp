from .utils import LogBasicTracker, extract_data_from_low_level_env_cfg, EnvLogSteppingModes
from ..utils.misc import pretty_list_of_pairs_to_string

import tensorflow as tf


class LogLowLevelScenarioData(LogBasicTracker):
    def __init__(self, env, config):
        LogBasicTracker.__init__(self, env, config)

    def reset(self):
        observation = super(LogLowLevelScenarioData, self).reset()
        print(dir(self.env.unwrapped._config.ScenarioConfig()))
        print(dir(self.env.unwrapped._config))
        return observation


class LogScenarioDifficulty(LogBasicTracker):
    """ This is a low level wrapper
    Logs scenario difficulty after each reset call"""
    def __init__(self, env, config):
        LogBasicTracker.__init__(self, env, config)
        self.summary_writer.set_stepping(EnvLogSteppingModes.env_resets)

    def reset(self):
        scenario_config = self.env.unwrapped._config.ScenarioConfig()
        left_team_difficulty = scenario_config.left_team_difficulty
        right_team_difficulty = scenario_config.right_team_difficulty
        self.summary_writer.write_scalar('scenario/left_team_difficulty',
                                         left_team_difficulty)
        self.summary_writer.write_scalar('scenario/right_team_difficulty',
                                         right_team_difficulty)

        observation = super(LogScenarioDifficulty, self).reset()
        return observation


class LogScenarioDataOnChange(LogBasicTracker):
    """ This is a low level wrapper.
    Logs scenario data after detecting a change.
    In order to see what is being logged look for
    extract_data_from_low_level_env_cfg in utils.py"""
    def __init__(self, env, config):
        LogBasicTracker.__init__(self, env, config)
        self._current_data = None
        self.summary_writer.set_stepping(EnvLogSteppingModes.env_resets)

    def reset(self):
        low_level_cfg = self.env.unwrapped._config
        new_data = extract_data_from_low_level_env_cfg(low_level_cfg)
        if new_data != self._current_data:  # todo czy if is_log_time
            self._current_data = new_data
            self.summary_writer.write_text(
                'scenario/low_level_cfg', '# Scenario changed to:  \n' +
                pretty_list_of_pairs_to_string(new_data))

        observation = super(LogScenarioDataOnChange, self).reset()
        return observation


class LogScenarioReset(LogBasicTracker):
    def __init__(self, env, config):
        LogBasicTracker.__init__(self, env, config)
        self._current_data = None
        self.summary_writer.set_stepping(EnvLogSteppingModes.env_total_steps)

    def reset(self):
        self.summary_writer.write_scalar('scenario/reset_occurred_after',
                                         self.env_episode_steps)

        observation = super(LogScenarioReset, self).reset()
        return observation
