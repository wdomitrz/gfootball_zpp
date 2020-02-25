from .utils import PackedBitsObservation, DummyEnv, packbits, unpackbits
from .network import GFootball
from gfootball.env.observation_preprocessing import generate_smm
from gfootball_zpp.wrappers.old_1_multihead_net import MultiHeadNet
import numpy as np
import tensorflow as tf

def expand_dims(obs):
    obs = packbits(obs)
    obs = tf.convert_to_tensor(obs)
    obs = tf.expand_dims(obs, axis=0)
    obs = unpackbits(obs)
    print('Converting for network:', obs.shape)
    return obs


def preprocess_obs(obs):
    return (), ((), (), obs)


def create_net(controlled_players, checkpoint_path):
    multihead = MultiHeadNet(DummyEnv('default', 4, controlled_agents=controlled_players), ())
    env = PackedBitsObservation(multihead)
    print('*' * 20)
    print(env.action_space.nvec)
    print('*' * 20)
    net = GFootball(env.action_space.nvec)
    net.change_config({'sample_actions': False})

    # TODO: make it work with differen number of heads
    sample_input = tf.convert_to_tensor(np.zeros((1, 1, 72, 96, 32)))
    net(((), ((), (), sample_input)), ())
    if checkpoint_path is not None:
        checkpoint = tf.train.Checkpoint(agent=net)
        print('restoring:', checkpoint_path)
        status = checkpoint.restore(checkpoint_path)
        print(status)

    return lambda obs: net(preprocess_obs(expand_dims(multihead._convert_obs(obs))), ())[0].action.numpy().flatten()


def convert_observation(o):
    o = generate_smm(o)
    return o
