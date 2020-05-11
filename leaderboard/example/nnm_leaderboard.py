from absl import flags
from absl import app
from absl import logging
from gfootball.env import create_environment, create_remote_environment
from gfootball.env.config import parse_player_definition
import grpc
import time

from gfootball_zpp.players.nnm import Player as NNMPlayer


flags.DEFINE_string('player_config_file', None, 'Path to nnm player config file')
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


def make_nnm_player(config_file):
    with open(config_file, 'r') as f:
        player_config = f.read()
    player_name, player_config = parse_player_definition(player_config)
    assert player_name == 'nnm'
    p = NNMPlayer(player_config, {})
    return p

def leaderboard(unused_argv):
  game_number = 0
  player = make_nnm_player(FLAGS.player_config_file)
  while game_number < FLAGS.how_many:
      print('Creating environment...')
      env = create_remote_environment(
          FLAGS.username, FLAGS.token, FLAGS.model_name, track=FLAGS.track,
          representation='raw', stacked=False, include_rendering=FLAGS.render)

      obs = env.reset()
      player.reset()
      print('Player created. Starting game.')
      while game_number < FLAGS.how_many:
          ob = obs
          cnt = 1
          done = False
          while not done:
              try:
                  action = player.take_action(ob)
                  ob, rew, done, _ = env.step(action)
                  logging.info('Playing the game, step %d, action %s, rew %s, done %d',
                               cnt, action, rew, done)
                  cnt += 1
              except grpc.RpcError as e:
                  print(e)
                  print('Waiting 1 minute before retrying...')
                  time.sleep(60)
                  break
          game_number += 1
          if game_number < FLAGS.how_many:
              obs = env.reset()
          print('=' * 50)




def leaderboard_test(unused_argv):
  env = create_environment('5_vs_5', stacked=False, representation='raw',
                           rewards='scoring', render=True,
                           number_of_left_players_agent_controls=4)
  obs = env.reset()
  player = make_nnm_player(FLAGS.player_config_file)
  player.reset()
  print('Player created. Starting game.')
  for _ in [0]:#range(FLAGS.how_many):
    ob = obs
    cnt = 1
    done = False
    while not done:
      try:
        action = player.take_action(ob)
        ob, rew, done, _ = env.step(action)
        logging.info('Playing the game, step %d, action %s, rew %s, done %d',
                     cnt, action, rew, done)
        cnt += 1
      except grpc.RpcError as e:
        print(e)
        break
    print('=' * 50)


if __name__ == '__main__':
    app.run(leaderboard_test)
