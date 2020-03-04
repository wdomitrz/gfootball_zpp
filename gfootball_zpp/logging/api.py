from .utils import LogAPI
from .actions import LogActionStats
from .scenario import LogLowLevelScenarioData, LogScenarioDifficulty, LogScenarioDataOnChange
from .ball import LogBallOwningTeam
from .reward import LogPerPlayerReward

import gym

def get_default_loggers():
    """ Returns the list of pairs (logger_name, logger_wrapper)
    Order of wrappers is relevant in particular
    low level wrappers should be used before other wrappers.
    Please note that LogAll wrapper is not included here."""

    result = []
    result.append(('log_api', LogAPI))
    # result.append(('log_low_level_scenario_data', LogLowLevelScenarioData))
    result.append(('log_scenario_difficulty', LogScenarioDifficulty))
    result.append(('log_scenario_data_on_change', LogScenarioDataOnChange))
    result.append(('log_action_stats', LogActionStats))
    result.append(('log_ball_owning_team', LogBallOwningTeam))
    result.append(('log_per_player_reward', LogPerPlayerReward))
    return result

class LogAll(gym.Wrapper):
    """ Applies all wrappers returned by get_default_loggers."""

    def __init__(self, env, config):
        for _, w in get_default_loggers():
            env = w(env, config)
        gym.Wrapper.__init__(self, env)


def get_loggers_dict():
    """ Returns the dict of wrappers returned by
    get_default_loggers() + log_all wrapper."""

    result = get_default_loggers()
    result.append(('log_all', LogAll))
    return dict(result)
