import gym
from absl import logging

class DiscreteToMulti(gym.Wrapper):
    def __init__(self, env, config):
        gym.Wrapper.__init__(self, env)
        if isinstance(self.action_space, gym.spaces.Discrete):
            logging.info('Converting action space to multidiscrete')
            self._convert = True
            self.action_space = gym.spaces.MultiDiscrete([env.action_space.n])
        else:
            self._convert = False

    def step(self, action):
        if self._convert:
            return self.env.step(action[0])
        else:
            return self.env.step(action)
