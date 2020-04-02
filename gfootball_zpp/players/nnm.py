import os
import tensorflow as tf

from absl import logging
from gfootball_zpp.env_composer import get_known_wrappers
from gfootball_zpp.utils.config_encoder import decode_config
from gfootball.env import player_base
from gfootball_zpp.players.utils import add_external_player_data, create_converter, download_model

def expand_input(input_):
    # batch dim
    input_ = tf.nest.map_structure(lambda b: tf.expand_dims(b, 0), input_)
    # time dim
    return tf.nest.map_structure(lambda t: tf.expand_dims(t, 0), input_)


def get_latest_model_path(path):
    while not tf.io.gfile.exists(path):
        pass
    model_list = list(map(lambda x: x[:-1], tf.io.gfile.listdir(path)))
    latest_model = str(max(map(int, model_list)))
    return os.path.join(path, latest_model)


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

        print(config)
        wrapper_names = config['wrappers'].split(',')
        known_wrappers = get_known_wrappers()
        self._wrappers = [lambda env: known_wrappers[name](env, config) for name in wrapper_names]
        self._converter = create_converter(self._wrappers)

        self._last_action = None
        self._prev_actions = []
        self._right_players = player_config['right_players']
        self._core_state = self._nn_manager.initial_state(1)


    def take_action(self, observation):
        self._converter.unwrapped.set_observation(observation)
        if self._last_action is None: # we are after reset
            observation = self._converter.reset()
        else:
            observation, _, _, _ = self._converter.step(self._last_action)

        prev_actions, observation = expand_input((self._prev_actions,
                                                  observation))

        action, _, self._core_state = self._nn_manager.get_action(prev_actions,
                                                                  ((), (), observation),
                                                                  self._core_state)
        self._last_action = action.numpy().flatten()
        self.prev_actions.append(self._last_action)
        assert self._last_action.shape[0] == self._right_players
        return self._last_action

    def reset(self):
        self._last_action = None
        self._prev_actions = []
        self._core_state = self._nn_manager.initial_state(1)
