from collections import namedtuple

from absl import logging
from gfootball.env import create_environment
from gfootball_zpp.players.zpp import Player
from gfootball_zpp.players import nnm
from gfootball_zpp.logging.api import LogAll
from gfootball_zpp.wrappers.state_preserver import StatePreserver
from gfootball_zpp.wrappers.env_usage_stats import EnvUsageStatsTracker
from gfootball_zpp.wrappers.env_utils import EnvUtilsWrapper
import tensorflow as tf

class EvalPlayerData:
    def __init__(self, type, name, extra_player_args=None):
        self.type = type
        self.name = name
        self.extra_player_args = extra_player_args

    def write_summary(self):
        return {
            'name': self.name,
            'type': self.type
        }

    def __str__(self):
        return 'EvalPlayer(' + self.type + ':' + self.name + ')'


class ZppEvalPlayerData(EvalPlayerData):
    def __init__(self, name, controlled_players=4, **kwargs):
        EvalPlayerData.__init__(self, 'zpp', name,
                                extra_player_args="zpp:left_players=0,right_players=" + str(controlled_players) +
                                ',' + ','.join([k + '=' + str(kwargs[k]) for k in kwargs]))
        print(self.extra_player_args)
        self.args = kwargs
        self._player = None

    @property
    def player(self):
        if not self._player:
            self._player = Player(
                dict(self.args, left_players=4, right_players=0, index=0), None)
        return self._player

    def write_summary(self):
        summary = EvalPlayerData.write_summary(self)
        summary['args'] = self.args
        return summary

class NNMEvalPlayerData(EvalPlayerData):
    def __init__(self, name, controlled_players=4, **kwargs):
        EvalPlayerData.__init__(self, 'nnm', name,
                                extra_player_args="nnm:left_players=0,right_players=" + str(controlled_players) +
                                ',' + ','.join([k + '=' + str(kwargs[k]) for k in kwargs]))
        print(self.extra_player_args)
        self.args = kwargs
        self._player = None

    @property
    def player(self):
        if not self._player:
            self._player = nnm.Player(
                dict(self.args,
                     left_players=4,
                     right_players=0,
                     index=0,
                     model_reload_rate=1000500100900), {})
        return self._player

    def write_summary(self):
        summary = EvalPlayerData.write_summary(self)
        summary['args'] = self.args
        return summary

class BotEvalPlayerData(EvalPlayerData):
    def __init__(self, name, dificulty):
        EvalPlayerData.__init__(self, 'bots', name)
        self.difficulty = dificulty

    def write_summary(self):
        summary = EvalPlayerData.write_summary(self)
        summary['difficulty'] = self.difficulty
        return summary


def stage_to_logdir(base_logdir, stage, player):
    if base_logdir == '':
        return ''
    return base_logdir + '/' + stage.scenario + '/' + player.name + '/' + stage.opponent.name


def evaluate(player, stage, env_args, base_logdir):
    args = env_args.copy()
    args['extra_players'] = stage.opponent.extra_player_args
    if args['extra_players']:
        args['extra_players'] = [args['extra_players']]
    else:
        args['extra_players'] = []
    args['env_name'] = stage.scenario
    args['logdir'] = stage_to_logdir(base_logdir, stage, player)
    json_config = {
        'dump_frequency': 1,
        'base_logdir': base_logdir,
        'extra_players': args['extra_players'],
        'step_log_freq': 1,
        'reset_log_freq': 1,
        'logs_enabled': True,
        'tf_summary_writer': tf.summary.create_file_writer(
            base_logdir + '/tf/', flush_millis=20000, max_queue=1000)
    }
    env = create_environment(**args)
    env = StatePreserver(env, json_config)
    env = EnvUtilsWrapper(env, json_config)
    env = EnvUsageStatsTracker(env, json_config)
    env = LogAll(env, json_config)

    env.set_right_player_name(stage.opponent.name)
    env.set_left_player_name(player.name)
    env.unwrapped._config['external_players_data'] = [{
        'name': stage.opponent.type,
        'description': stage.opponent.name
    }]

    scores = []
    for i in range(stage.games):
        done = False
        obs = env.reset()
        score = {'left': 0, 'right': 0}
        while not done:
            action = player.player.take_action(obs)
            obs, rew, done, info = env.step(action)
            if info['score_reward'] < 0:
                score['right'] -= info['score_reward']
            else:
                score['left'] += info['score_reward']
        scores.append(score)
        logging.info('Finished game (%d/%d) %s: %s - %d : %d - %s', i + 1, stage.games,
                     stage.scenario, player, score['left'], score['right'], stage.opponent)
    env.reset()
    env.close()
    return EvaluationResult(scores=scores, stage=stage, logdir=args['logdir'])


EvaluationStage = namedtuple(
    'EvaluationStage', ['scenario', 'opponent', 'games'])
EvaluationResult = namedtuple('EvaluationResult', ['stage', 'scores', 'logdir'])


def evaluate_all(player, stages, env_args, base_logdir):
    return [evaluate(player, stage, env_args, base_logdir) for stage in stages]
