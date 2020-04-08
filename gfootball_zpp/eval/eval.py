from collections import namedtuple

from absl import logging
from gfootball.env import create_environment
from gfootball_zpp.players.zpp import Player


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
    args['env_name'] = stage.scenario
    args['logdir'] = stage_to_logdir(base_logdir, stage, player)
    env = create_environment(**args)

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

    return EvaluationResult(scores=scores, stage=stage, logdir=args['logdir'])


EvaluationStage = namedtuple(
    'EvaluationStage', ['scenario', 'opponent', 'games'])
EvaluationResult = namedtuple('EvaluationResult', ['stage', 'scores', 'logdir'])


def evaluate_all(player, stages, env_args, base_logdir):
    return [evaluate(player, stage, env_args, base_logdir) for stage in stages]
