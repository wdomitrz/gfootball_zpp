from numpy import random
from gfootball.env import config
from gfootball.env import football_env
import gym


class BaseRecreatableEnv(gym.Wrapper):
    """It recreates env every change_rate episodes."""

    def __init__(self, change_rate):
        self.change_rate = change_rate
        self.episodes_completed = 0
        self.episodes_since_change = 0
        self.env = None
        env = self.create_new_env()
        super().__init__(env)

    def create_new_env(self):
        raise NotImplementedError

    def reset(self):
        self.episodes_since_change += 1
        if self.episodes_since_change >= self.change_rate:
            self.env = self.create_new_env()
            self.episodes_since_change += 1
        return super().reset()


class RecreatableFootballEnv(BaseRecreatableEnv):
    """Recreates football env with new parameters."""
    def get_new_config(self):
        raise NotImplementedError

    def create_new_env(self):
        if self.env:
            self.env.close()
        return football_env.FootballEnv(config.Config(self.get_new_config()))


class RandomParametersEnv(RecreatableFootballEnv):
    """Creates new football env with randomly chosen parameters.

    The choice is based on given probabilities distribution (uniform if not
    specified). If the default_config is given the parameters override or extend
    existing in it fields.
    """

    def __init__(self, change_rate, parameters, default_config=None, probabilities=None):
        if default_config is not None:
            self.parameters = [dict(default_config, **p) for p in parameters]
        else:
            self.parameters = parameters
        print(self.parameters)
        self.probabilities = probabilities
        super().__init__(change_rate)

    def get_new_config(self):
        new_config = random.choice(self.parameters, p=self.probabilities)
        print(new_config)
        return new_config


def create_recreatable_football(env_config, football_config=None):
    assert 'env_change_rate' in env_config, \
           "'env_change_rate' is required parameter for RecreatableEnv"
    assert 'env_change_params' in env_config, \
           "'env_change_params' is required parameter for RecreatableEnv"

    params = env_config['env_change_params']
    rate = env_config['env_change_rate']
    probabilities = env_config.get('env_change_probabilities', None)

    return RandomParametersEnv(rate, params, default_config=football_config,
                               probabilities=probabilities)
