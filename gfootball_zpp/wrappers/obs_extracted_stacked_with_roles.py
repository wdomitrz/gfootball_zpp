import gym
import numpy as np

from gfootball.env.wrappers import SMMWrapper, FrameStack


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
        self._obs_wrapped = FrameStack(SMMWrapper(env), env_config['stacked_frames'])
        obs_shape = self._obs_wrapped.observation_space.shape
        obs_shape[-1] += 1
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8)
        self.action_space = env.action_space

    def _convert_obs(self, raw_observations):
        num_observations = len(raw_observations)
        
        layer_shape = (self.observation_space.shape[-3], self.observation_space.shape[-2])
        
        observations = self._obs_wrapped.observation(raw_observations)
        
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
        observation, reward, done, info = self.env.step(action)

        observation = self._convert_obs(observation)

        return observation, reward, done, info

    def reset(self):
        observation = self.env.reset()
        observation = self._convert_obs(observation)
        return observation

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        return self.env.close()
