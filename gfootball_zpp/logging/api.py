from .utils import LogAPI
from .actions import LogActionStats
from .scenario import LogLowLevelScenarioData, LogScenarioDifficulty, LogScenarioDataOnChange, LogScenarioReset
from .ball import LogBallOwningTeam
from .reward import LogPerPlayerReward, LogAveragePerPlayerRewardByDifficulty

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
    result.append(('log_average_per_player_reward_by_difficulty', LogAveragePerPlayerRewardByDifficulty))

    med_level_start_id = len(result)
    result.append(('log_scenario_reset', LogScenarioReset))
    result.append(('log_action_stats', LogActionStats))
    result.append(('log_ball_owning_team', LogBallOwningTeam))

    high_level_start_id = len(result)
    result.append(('log_per_player_reward', LogPerPlayerReward))
    return result, med_level_start_id, high_level_start_id


class LogAll(gym.Wrapper):
    """ Applies all wrappers returned by get_default_loggers."""

    def __init__(self, env, config):
        wrappers, _, _ = get_default_loggers()
        for _, w in wrappers:
            env = w(env, config)
        gym.Wrapper.__init__(self, env)


class LogLowLevel(gym.Wrapper):
    def __init__(self, env, config):
        wrappers, med_level_start_id, _ = get_default_loggers()
        for _, w in wrappers[:med_level_start_id]:
            env = w(env, config)
        gym.Wrapper.__init__(self, env)


class LogMedLevel(gym.Wrapper):
    def __init__(self, env, config):
        wrappers, med_level_start_id, high_level_start_id = get_default_loggers()
        for _, w in wrappers[med_level_start_id:high_level_start_id]:
            env = w(env, config)
        gym.Wrapper.__init__(self, env)


class LogHighLevel(gym.Wrapper):
    def __init__(self, env, config):
        wrappers, _, high_level_start_id = get_default_loggers()
        for _, w in wrappers[high_level_start_id:]:
            env = w(env, config)
        gym.Wrapper.__init__(self, env)


def get_loggers_dict():
    """ Returns the dict of wrappers returned by
    get_default_loggers() + log_all, log_low, log_med, log_high wrapper."""

    result, _, _ = get_default_loggers()
    result.append(('log_all', LogAll))
    result.append(('log_low', LogLowLevel))
    result.append(('log_med', LogMedLevel))
    result.append(('log_high', LogHighLevel))
    return dict(result)
