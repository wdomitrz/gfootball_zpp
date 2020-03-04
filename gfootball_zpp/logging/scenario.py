from .utils import LogBasicTracker, extract_data_from_low_level_env_cfg
from ..utils import pretty_list_of_pairs_to_string

import tensorflow as tf

class LogLowLevelScenarioData(LogBasicTracker):
    def __init__(self, env, config):
        LogBasicTracker.__init__(self, env, config)

    def __getattr__(self, attr):
        return getattr(self.env, attr)

    def reset(self):
        observation = super(LogLowLevelScenarioData, self).reset()
        print(dir(self.env._config.ScenarioConfig()))
        print(dir(self.env._config))
        return observation


class LogScenarioDifficulty(LogBasicTracker):
    """ This is a low level wrapper
    Logs scenario difficulty after each reset call"""

    def __init__(self, env, config):
        LogBasicTracker.__init__(self, env, config)

    def __getattr__(self, attr):
        return getattr(self.env, attr)

    def reset(self):
        observation = super(LogScenarioDifficulty, self).reset()
        scenario_config = self.env._config.ScenarioConfig()
        left_team_difficulty = scenario_config.left_team_difficulty
        right_team_difficulty = scenario_config.right_team_difficulty
        with self.summary_writer.as_default():
            tf.summary.scalar('scenario/left_team_difficulty',
                              left_team_difficulty, self.env_resets)
            tf.summary.scalar('scenario/right_team_difficulty',
                              right_team_difficulty, self.env_resets)
        return observation


class LogScenarioDataOnChange(LogBasicTracker):
    """ This is a low level wrapper.
    Logs scenario data after detecting a change.
    In order to see what is being logged look for
    extract_data_from_low_level_env_cfg in utils.py"""

    def __init__(self, env, config):
        LogBasicTracker.__init__(self, env, config)
        self._current_data = None

    def __getattr__(self, attr):
        return getattr(self.env, attr)

    def reset(self):
        observation = super(LogScenarioDataOnChange, self).reset()
        low_level_cfg = self.env._config
        new_data = extract_data_from_low_level_env_cfg(low_level_cfg)
        if new_data != self._current_data:
            self._current_data = new_data
            with self.summary_writer.as_default():
                tf.summary.text('scenario/low_level_cfg',
                                '# Scenario changed to:  \n' +
                                pretty_list_of_pairs_to_string(new_data),
                                self.env_resets)
        return observation
