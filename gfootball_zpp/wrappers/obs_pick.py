import gym
import numpy as np


class Simple115PickFirst(gym.Wrapper):
    def __init__(self, env, env_config):
        gym.Wrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(
        low=-1, high=1, shape=(1, 115), dtype=np.float32)
        self.action_space = env.action_space

    def _convert_obs(self, observation):
        return np.expand_dims(observation[0], axis=0)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        observation = self._convert_obs(observation)
        reward = np.array([np.max(reward)])

        return observation, reward, done, info

    def reset(self):
        observation = self.env.reset()
        observation = self._convert_obs(observation)
        return observation

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        return self.env.close()
