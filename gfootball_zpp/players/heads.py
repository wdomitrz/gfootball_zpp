from .utils import PackedBitsObservation, DummyEnv, packbits, unpackbits
from .network import GFootball
from .zpp_player import BaseZppPlayer
from gfootball.env.observation_preprocessing import generate_smm
from gfootball_zpp.wrappers.old_1_multihead_net import MultiHeadNet
from gfootball_zpp.utils import gsutil
import numpy as np
import tensorflow as tf

CHANNELS_STEP = 16

class HeadsPlayer(BaseZppPlayer):
    def __init__(self, controlled_players, player_config):
        sample = player_config.get('sample', False)
        self.multihead = MultiHeadNet(DummyEnv('default', controlled_players, controlled_agents=controlled_players), ())
        env = PackedBitsObservation(self.multihead)
        print('*' * 20)
        print(env.action_space.nvec)
        print('*' * 20)
        self.net = GFootball(env.action_space.nvec)
        self.net.change_config({'sample_actions': sample})

        # TODO: make it work with differen number of heads
        sample_input = tf.convert_to_tensor(np.zeros((1, 1, 72, 96, self._get_channels(controlled_players))))
        self.net(((), ((), (), sample_input)), ())

    @staticmethod
    def _get_channels(controlled_players):
        return CHANNELS_STEP * (1 + (controlled_players + 2) // 4)

    def _expand_dims(self, obs):
        obs = packbits(obs)
        obs = tf.convert_to_tensor(obs)
        obs = tf.expand_dims(obs, axis=0)
        obs = unpackbits(obs)
        # print('Converting for network:', obs.shape)
        return obs

    def pre_stacking_convert_obs(self, obs):
        return generate_smm(obs)

    def take_action(self, obs):
        obs = self.multihead._convert_obs(obs)
        obs = self._expand_dims(obs)
        obs = (), ((), (), obs)
        action, _ = self.net(obs, ())
        action = action.action.numpy().flatten()
        return action

    def load_checkpoint(self, checkpoint):
        ckpt = tf.train.Checkpoint(agent=self.net)
        print('restoring:', checkpoint)
        status = ckpt.restore(checkpoint)
        print(status)
