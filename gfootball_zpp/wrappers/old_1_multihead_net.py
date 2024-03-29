import gym
import numpy as np


class MultiHeadNet(gym.Wrapper):
    """Supports only extracted observations (stacked)"""

    def __init__(self, env, env_config):
        gym.Wrapper.__init__(self, env)

        # observation shape will be (_, _, 4*frame_size)
        self.players = env.observation_space.shape[0]
        self.frame_size = 2 + (self.players) + 1
        obs_shape = np.array((1,) + env.observation_space.shape[1:])
        obs_shape[len(obs_shape) - 1] = self.frame_size * 4

        print('!!Squashed observations to!!', obs_shape)

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8)
        self.action_space = env.action_space

    def _convert_obs(self, observation):
        conv_obs = np.zeros(self.observation_space.shape, dtype=np.uint8)

        for i in range(conv_obs.shape[3]):
            j = i // self.frame_size
            layer_id = i % self.frame_size

            if layer_id == 0:
                layer = observation[0, ..., j * 4]
                conv_obs[0, ..., i] = layer
            elif layer_id == 1:
                layer = observation[0, ..., j * 4 + 1]
                conv_obs[0, ..., i] = layer
            elif layer_id >= 2 and layer_id < self.frame_size - 1:
                player_id = layer_id - 2
                layer1 = observation[player_id, ..., j * 4 + 3]
                conv_obs[0, ..., i] = layer1
            else:
                layer = observation[0, ..., j * 4 + 2]
                conv_obs[0, ..., i] = layer

            #print(np.where(conv_obs > 0))

        return conv_obs

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
