import gym
import numpy as np

from gfootball.env.wrappers import SMMWrapper, FrameStack
from gfootball_zpp.players.utils import ManualEnv

def get_role_id(raw_observation):
    player_number = raw_observation['active']
    return raw_observation['left_team_roles'][player_number]

# note that this is used only to maintain information format for compression,
# it should be converted to one_hot by network on master replica
def encode_in_minimap_layer(layer_shape, value):
    layer = np.zeros(layer_shape, dtype=np.uint8)
    layer[0, value] = 255
    return layer

class ObservationExtractStackWithRoles(gym.Wrapper):
    def __init__(self, env, env_config):
        gym.Wrapper.__init__(self, env)

        self._manual_env = ManualEnv(self.env.unwrapped._config)
        #self._manual_env.set_action_space(self.action_space)

        self._obs_wrapped = FrameStack(SMMWrapper(self._manual_env, env_config['channel_dimensions']), env_config['stacked_frames'])
        obs_shape = self._obs_wrapped.observation_space.shape
        obs_shape = (obs_shape[0], obs_shape[1], obs_shape[2], obs_shape[3] + 1)

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def _convert_obs(self, raw_observations, observations):
        num_observations = len(raw_observations)

        layer_shape = (self.observation_space.shape[-3], self.observation_space.shape[-2])

        result = []
        for obs_id in range(num_observations):
            obs = observations[obs_id]
            role_id = get_role_id(raw_observations[obs_id])

            new_layer = encode_in_minimap_layer(layer_shape, role_id)


            new_layer = np.expand_dims(new_layer, axis=-1)

            obs = np.concatenate((obs, new_layer), axis=-1)
            result.append(obs)
        return np.array(result, dtype=np.uint8)

    def step(self, action):
        raw_observation, reward, done, info = self.env.step(action)

        self._manual_env.set_observation(raw_observation)
        wrapped_observation, _, _, _ = self._obs_wrapped.step(action)
        observation = self._convert_obs(raw_observation, wrapped_observation)

        return observation, reward, done, info

    def reset(self):
        raw_observation = self.env.reset()

        self._manual_env.set_observation(raw_observation)
        wrapped_observation = self._obs_wrapped.reset()
        observation = self._convert_obs(raw_observation, wrapped_observation)

        return observation

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        return self.env.close()
