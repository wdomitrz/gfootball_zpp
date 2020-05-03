import os
import tensorflow as tf

from numpy import random

from absl import logging
from gfootball_zpp.env_composer import get_known_wrappers
from gfootball_zpp.utils.config_encoder import decode_config
from gfootball_zpp.utils.misc import extract_number_from_txt
from gfootball.env import player_base
from gfootball_zpp.players.utils import add_external_player_data, create_converter, download_model, SimulatedConfig, EnvOutput, change_external_player_data
from gfootball_zpp.players.checkpoints import select_mostly_latest


def get_models_path(path):
    if not tf.io.gfile.exists(path):
        return None

    model_list = list(map(extract_number_from_txt, tf.io.gfile.listdir(path)))

    model_list.sort()

    if len(model_list) == 0:
        return None

    models_paths = [os.path.join(path, str(m)) for m in model_list]
    return models_paths


def get_latest_model_path(path):
    if not tf.io.gfile.exists(path):
        return None

    model_list = list(map(extract_number_from_txt, tf.io.gfile.listdir(path)))

    if len(model_list) == 0:
        return None

    latest_model = str(max(map(int, model_list)))
    return os.path.join(path, latest_model)


def get_random_model_path(path):
    models_paths = get_models_path(path)

    if models_paths is None:
        return None

    return random.choice(models_paths)


def get_mostly_latest_model_path(path):
    models_paths = get_models_path(path)

    if models_paths is None:
        return None
    return select_mostly_latest(model_paths)


def pack_nnm_input(num_actions, num_rewards, observation, core_state):
    prev_actions = tf.zeros(shape=(1, num_actions), dtype=tf.int64)
    reward = tf.zeros(shape=(1, num_rewards), dtype=tf.float32)
    done = tf.constant([False], dtype=tf.bool)
    observation = tf.constant(observation)
    observation = tf.expand_dims(observation, 0)

    return (prev_actions,
            EnvOutput(reward=reward, done=done,
                      observation=observation), core_state)


def handle_gs(path):
    if len(path) >= 2 and path[:2] == 'GS':
        return 'gs:' + path[2:]
    else:
        return path


class Player(player_base.PlayerBase):
    """An agent handled by NNManager"""
    def __init__(self, player_config, env_config):
        player_base.PlayerBase.__init__(self, player_config)

        self._env_config = env_config
        self._resets = 0
        self._hidden = False
        self._action_set = 'default'
        self._player_prefix = 'player_{}'.format(player_config['index'])
        self._left_players = int(player_config.get('left_players', 0))
        self._right_players = int(player_config.get('right_players', 0))

        self._model_reload_rate = int(player_config.get(
            'model_reload_rate', 0))

        if 'models_dirs' not in player_config:
            raise Exception('wrong config for nnm player')

        models_dirs_spec = list(
            map(lambda x: x.split(';'),
                player_config['models_dirs'].split('*')))
        if len(models_dirs_spec[0]) != 3:
            raise Exception('wrong config for nnm player')

        self._models_dirs = [mds[0] for mds in models_dirs_spec]
        self._models_dirs_p = [mds[1] for mds in models_dirs_spec]
        self._models_configs = [mds[2] for mds in models_dirs_spec]

        self._current_model_name = None
        self._nn_manager = None
        player_data = {
            'name': 'nnm',
            'description': str(self._current_model_name),
        }

        self._id_in_p_data = add_external_player_data(self._env_config,
                                                      player_data)

        logging.info('NNM player: Model configs %s', str(self._models_configs))

        self._update_model()

    def _update_model(self):
        model_dir_id = random.choice(len(self._models_dirs),
                                     p=self._models_dirs_p)
        model_dir = self._models_dirs[model_dir_id]

        logging.info('NNM player: Chosed model_dir %s', model_dir)

        if model_dir[:8] == '!latest-':
            dir_path = handle_gs(model_dir[8:])
            model_path = get_latest_model_path(dir_path)
        elif model_dir[:8] == '!random-':
            dir_path = handle_gs(model_dir[8:])
            model_path = get_random_model_path(dir_path)
        elif model_dir[:15] == '!mostly_latest-':
            dir_path = handle_gs(model_dir[15:])
            model_path = get_mostly_latest_model_path(dir_path)
        else:
            model_path = dir_path = handle_gs(model_dir)

        self._nn_manager = None
        self._current_model_name = None
        self._wrappers = None
        self._converter = None
        self._num_rewards = None
        self._last_action = None
        self._core_state = None

        if model_path is not None:
            logging.info('NNM player: chosed dir with id %s',
                         str(model_dir_id))
            config = decode_config(self._models_configs[model_dir_id])

            logging.info('NNM player loading: %s', model_path)
            self._nn_manager = download_model(model_path)
            logging.info('NNM player loading done: %s', model_path)
            self._current_model_name = model_path

            player_data = {
                'name': 'nnm',
                'description': str(self._current_model_name),
            }

            wrapper_names = config['wrappers'].split(',')
            known_wrappers = get_known_wrappers()
            self._wrappers = [known_wrappers[name] for name in wrapper_names]
            self._wrappers = list(
                map(lambda w: lambda env: w(env, config), self._wrappers))
            self._converter = create_converter(
                self._wrappers,
                SimulatedConfig(
                    number_of_players_agent_controls=self._right_players +
                    self._left_players))
            self._converter.unwrapped.set_reward(
                [0] * (self._left_players + self._right_players))
            self._core_state = self._nn_manager.initial_state(1)
        else:
            player_data = {
                'name': 'nnm',
                'description': str(self._current_model_name),
            }

        change_external_player_data(self._env_config, self._id_in_p_data,
                                    player_data)

    def hide(self):
        self._hidden = True
        self._num_left_controlled_players = 0
        self._num_right_controlled_players = 0

    def show(self):
        self._hidden = False
        self._num_left_controlled_players = self._left_players
        self._num_right_controlled_players = self._right_players

    def take_action(self, observation):
        if self._hidden:
            return []
        elif self._nn_manager is None:
            return [0] * (self._right_players + self._left_players)

        self._converter.unwrapped.set_observation(observation)
        if self._num_rewards is None:
            observation = self._converter.reset()
            _, rewards, _, _ = self._converter.step(
                [0] * (self._left_players + self._right_players))
            self._num_rewards = len(rewards)

        if self._last_action is None:  # we are after reset
            observation = self._converter.reset()
        else:
            observation, _, _, _ = self._converter.step(self._last_action)

        prev_actions, env_output, core_state = pack_nnm_input(
            self._left_players + self._right_players, self._num_rewards,
            observation, self._core_state)

        (action, _, _), self._core_state = self._nn_manager.get_action(
            prev_actions, env_output, core_state)

        self._last_action = action.numpy().flatten()
        assert self._last_action.shape[
            0] == self._left_players + self._right_players
        return self._last_action

    def reset(self):
        self._resets += 1
        if self._model_reload_rate != 0 and self._resets % self._model_reload_rate == 0 or self._nn_manager is None:
            self._resets = 0
            self._update_model()

        if self._nn_manager is not None:
            self._last_action = None
            self._core_state = self._nn_manager.initial_state(1)
