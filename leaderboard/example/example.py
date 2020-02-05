import tensorflow as tf
from absl import flags
from absl import app
from gfootball.env import create_environment

from network import GFootball
from old_1_multihead_net import MultiHeadNet
import observation



flags.DEFINE_string('ckpt', './f5v5_bots_one_net_then_self_play_3_2_1_ckpt_0_ckpt-456', 'Path to checkpoint')

FLAGS = flags.FLAGS


def convert_observations(obs):
    obs = tf.convert_to_tensor(obs)
    obs = tf.expand_dims(obs, axis=0)
    obs = observation.unpackbits(obs)
    # observation should have the following form  (TIME, BATCH, MINIMAP_DATA...)
    print('Converting for network:', obs.shape)
    # Network takes input_ of form (PREV_ACTIONS, EnvOutput(reward, done, observation))
    return (), ((), (), obs)


def example_play_with_bots():
    env = create_environment('5_vs_5', stacked=True, representation='extracted', rewards='scoring',
                             render=True, number_of_left_players_agent_controls=4)
    # Wrapper, second argument is unused
    env = MultiHeadNet(env, ())

    # Used in order to maintain compatibility with network
    # Packing makes number of minimap layers divisible by 16
    env = observation.PackedBitsObservation(env)

    obs = env.reset()
    print(obs.shape)
    
    net = GFootball(env.action_space.nvec)
    net.change_config({'sample_actions': False})
    # create variables
    print(net(convert_observations(obs), ()))

    #load checkpoint
    optimizer = tf.keras.optimizers.Adam(0.01) # only to supress some warnings
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, agent=net)
    print('restoring:', FLAGS.ckpt)
    status = checkpoint.restore(FLAGS.ckpt)
    print(status)


    done = False
    while not done:
        obs = convert_observations(obs)
        agent_output, _ = net(obs, ())
        action = agent_output.action.numpy().flatten()
        print(action)
        obs, rew, done, _ = env.step(action)


if __name__ == '__main__':
    app.run(lambda _: example_play_with_bots())
