import tensorflow as tf
from absl import flags
from absl import app
from absl import logging
from gfootball.env import create_environment, create_remote_environment
import grpc

from network import GFootball
from old_1_multihead_net import MultiHeadNet
import observation


flags.DEFINE_string('ckpt', './f5v5_bots_one_net_then_self_play_3_2_1_ckpt_0_ckpt-456', 'Path to checkpoint')
flags.DEFINE_string('username', 'multiPandasUW', 'Username to use')
flags.DEFINE_string('token', None, 'Token to use.')
flags.mark_flag_as_required('token')
flags.DEFINE_string('track', 'multiagent', 'Name of the competition track.')
flags.DEFINE_string('model_name', None,
                    'A model identifier to be displayed on the leaderboard.')
flags.mark_flag_as_required('model_name')
flags.DEFINE_integer('how_many', 1000, 'How many games to play')

flags.DEFINE_bool('render', False, 'Whether to render a game.')
FLAGS = flags.FLAGS


def convert_observations(obs):
    obs = tf.convert_to_tensor(obs)
    obs = tf.expand_dims(obs, axis=0)
    obs = observation.unpackbits(obs)
    # observation should have the following form  (TIME, BATCH, MINIMAP_DATA...)
    print('Converting for network:', obs.shape)
    # Network takes input_ of form (PREV_ACTIONS, EnvOutput(reward, done, observation))
    return (), ((), (), obs)


def wrap_env(env):
    env = MultiHeadNet(env, ())

    # Used in order to maintain compatibility with network
    # Packing makes number of minimap layers divisible by 16
    env = observation.PackedBitsObservation(env)
    return env


def restore_checkpoint(env, sample_obs):
    net = GFootball(env.action_space.nvec)
    net.change_config({'sample_actions': False})

    # create variables
    net(convert_observations(sample_obs), ())

    # load checkpoint
    optimizer = tf.keras.optimizers.Adam(0.01)  # only to supress some warnings
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, agent=net)
    print('restoring:', FLAGS.ckpt)
    status = checkpoint.restore(FLAGS.ckpt)
    print(status)
    return net


def example_play_with_bots():
    env = create_environment('5_vs_5', stacked=True, representation='extracted', rewards='scoring',
                             render=True, number_of_left_players_agent_controls=4)
    # Wrapper, second argument is unused
    env = wrap_env(env)

    net = restore_checkpoint(env)

    obs = env.reset()
    done = False
    while not done:
        obs = convert_observations(obs)
        agent_output, _ = net(obs, ())
        action = agent_output.action.numpy().flatten()
        print(action)
        obs, rew, done, _ = env.step(action)


def leaderboard(unused_argv):
  env = create_remote_environment(
      FLAGS.username, FLAGS.token, FLAGS.model_name, track=FLAGS.track,
      representation='raw', stacked=True, include_rendering=FLAGS.render)
  env = wrap_env(env)
  obs = env.reset()
  model = restore_checkpoint(env, obs)
  for _ in range(FLAGS.how_many):
      ob = obs
      cnt = 1
      done = False
      while not done:
          try:
              ob = convert_observations(ob)
              agent_output, _ = model(ob, ())
              action = agent_output.action.numpy().flatten()
              ob, rew, done, _ = env.step(action)
              logging.info('Playing the game, step %d, action %s, rew %s, done %d',
                           cnt, action, rew, done)
              cnt += 1
          except grpc.RpcError as e:
              print(e)
              break
      print('=' * 50)

def leaderboard_test(unused_argv):
  env = create_environment('5_vs_5', stacked=True, representation='extracted',
                           rewards='scoring', render=True,
                           number_of_left_players_agent_controls=4)
  env = wrap_env(env)
  obs = env.reset()
  model = restore_checkpoint(env, obs)
  for _ in [0]:#range(FLAGS.how_many):
    ob = obs
    cnt = 1
    done = False
    while not done:
      try:
        ob = convert_observations(ob)
        agent_output, _ = model(ob, ())
        action = agent_output.action.numpy().flatten()
        ob, rew, done, _ = env.step(action)
        logging.info('Playing the game, step %d, action %s, rew %s, done %d',
                     cnt, action, rew, done)
        cnt += 1
      except grpc.RpcError as e:
        print(e)
        break
    print('=' * 50)



if __name__ == '__main__':
    app.run(leaderboard)
