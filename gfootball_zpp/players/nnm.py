import os
import tensorflow as tf

from absl import logging
from gfootball_zpp.env_composer import get_known_wrappers
from gfootball_zpp.utils.config_encoder import decode_config
from gfootball.env import player_base
from gfootball_zpp.players.utils import add_external_player_data, create_converter, download_model, SimulatedConfig, EnvOutput


def get_latest_model_path(path):
    while not tf.io.gfile.exists(path):
        pass
    model_list = list(map(lambda x: x[:-1], tf.io.gfile.listdir(path)))
    latest_model = str(max(map(int, model_list)))
    return os.path.join(path, latest_model)


def pack_nnm_input(num_actions, num_rewards, observation, core_state):
    prev_actions = tf.zeros(shape=(1, num_actions), dtype=tf.int64)
    reward = tf.zeros(shape=(1, num_rewards), dtype= tf.float32)
    done = tf.constant([False], dtype=tf.bool)
    observation = tf.constant(observation)
    observation = tf.expand_dims(observation, 0)

    return (prev_actions, EnvOutput(reward=reward,
                                   done=done,
                                   observation=observation),
            core_state)


class Player(player_base.PlayerBase):
    """An agent handled by NNManager
    example:
    nnm:models_dir={models_dir},model={model},right_players={n},encoded_env_config={see_config_encoder_in_utils}"""

    def __init__(self, player_config, env_config):
        player_base.PlayerBase.__init__(self, player_config)
        self._action_set = 'default'
        self._player_prefix = 'player_{}'.format(player_config['index'])

        model = player_config['model']
        models_dir = player_config['models_dir']

        if models_dir[0:2] == "GS":
            models_dir = "gs:" + models_dir[2:]

        if model == '!latest':
            model_path = get_latest_model_path(models_dir)
        else:
            model_path = os.path.join(models_dir, model)

        logging.info('NNM player loading: %s', model_path)
        self._nn_manager = download_model(model_path)


        player_data = {
            'name': 'nnm',
            'description': model_path
        }

        add_external_player_data(env_config, player_data)

        config = decode_config(player_config['encoded_env_config'])

        self._right_players = int(player_config['right_players'])

        wrapper_names = config['wrappers'].split(',')
        known_wrappers = get_known_wrappers()
        self._wrappers = [known_wrappers[name] for name in wrapper_names]
        self._wrappers = list(map(lambda w: lambda env: w(env, config), self._wrappers))
        self._converter = create_converter(self._wrappers, SimulatedConfig(
            number_of_players_agent_controls=self._right_players))
        self._converter.unwrapped.set_reward([0] * self._right_players)

        self._num_rewards = None


        self._last_action = None
        self._core_state = self._nn_manager.initial_state(1)


    def take_action(self, observation):
        self._converter.unwrapped.set_observation(observation)
        if self._num_rewards is None:
            observation = self._converter.reset()
            _, rewards, _, _ = self._converter.step([0] * self._right_players)
            self._num_rewards = len(rewards)

        if self._last_action is None: # we are after reset
            observation = self._converter.reset()
        else:
            observation, _, _, _ = self._converter.step(self._last_action)

        prev_actions, env_output, core_state = pack_nnm_input(self._right_players, self._num_rewards, observation, self._core_state)

        (action, _, _), self._core_state = self._nn_manager.get_action(prev_actions,
                                                                  env_output,
                                                                  core_state)
        self._last_action = action.numpy().flatten()
        assert self._last_action.shape[0] == self._right_players
        return self._last_action

    def reset(self):
        self._last_action = None
        self._core_state = self._nn_manager.initial_state(1)
