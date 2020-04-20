import os

from numpy import random

from gfootball.env import player_base
from gfootball_zpp.players.players import build_policy
from gfootball_zpp.players.utils import ObservationStacker, add_external_player_data
from gfootball_zpp.utils import gsutil
from gfootball_zpp.players import checkpoints


class Player(player_base.PlayerBase):
    """An agent loaded from one of ZPP checkpoints.

    This file should be copied into gfootball/env/players directory.
    To use it in the game you need to pass extra_players:
    'zpp:policy={policy_name},right_players={n},checkpoint={path}'."""

    def __init__(self, player_config, env_config):
        player_base.PlayerBase.__init__(self, player_config)

        self._action_set = 'default'
        self._player_prefix = 'player_{}'.format(player_config['index'])
        stacking = 4 #if player_config.get('stacked', True) else 1
        self._stacker = ObservationStacker(stacking)
        policy = player_config.get('policy', '')
        self.checkpoints_info = []
        self.current_checkpoint = None

        self._policy = build_policy(policy, self.num_controlled_players(), player_config)
        self._checkpoints_p = None

        if 'checkpoints' in player_config:
            checkpoints = list(map(lambda x: x.split(';'), player_config['checkpoints'].split('*')))
            self._checkpoints = [x[0] for x in checkpoints]
            if len(checkpoints[0]) == 2:
                self._checkpoints_p = [x[1] for x in checkpoints]
        else:
            self._checkpoints = [player_config.get('checkpoint', None)]

        self.update_checkpoint()

        self.resets = 0
        self.checkpoint_reload_rate = int(player_config.get('checkpoint_reload_rate', 0))

        self.args = {
            'policy': policy,
            'checkpoint': player_config.get('checkpoint', None),
            'config': player_config
        }

        player_data = {
            'name': policy,
            'description': self.current_checkpoint['path']
                           if self.current_checkpoint else None,
            'checkpoints': self.checkpoints_info
        }

        add_external_player_data(env_config, player_data)

    def take_action(self, observation):
        observation = self._policy.pre_stacking_convert_obs(observation)
        observation = self._stacker.get(observation)
        return self._policy.take_action(observation)

    def update_checkpoint(self):
        checkpoint = random.choice(self._checkpoints, p=self._checkpoints_p)
        if checkpoint is None:
            return
        checkpoint_info = {
            'raw': checkpoint,
            'type': 'local',
            'path': checkpoint
        }
        if checkpoint[:10] == '!latest-GS':
            checkpoint_info['type'] = 'latest-GS'
            checkpoint = checkpoints.get_checkpoint('gs:' + checkpoint[10:], checkpoints.select_latest)
            if checkpoint:
                checkpoint_info['path'] = checkpoint
                checkpoint = gsutil.cp_ckpt(checkpoint)
        elif checkpoint[:10] == '!random-GS':
            checkpoint_info['type'] = 'random-GS'
            checkpoint = checkpoints.get_checkpoint('gs:' + checkpoint[10:], checkpoints.select_random)
            if checkpoint:
                checkpoint_info['path'] = checkpoint
                checkpoint = gsutil.cp_ckpt(checkpoint)
        elif checkpoint[:17] == '!mostly_latest-GS':
            checkpoint_info['type'] = 'mostly_latest-GS'
            checkpoint = checkpoints.get_checkpoint('gs:' + checkpoint[17:], checkpoints.select_mostly_latest)
            if checkpoint:
                checkpoint_info['path'] = checkpoint
                checkpoint = gsutil.cp_ckpt(checkpoint)
        elif checkpoint[:2] == 'GS':
            checkpoint_info['type'] = 'gs'
            checkpoint = 'gs:' + checkpoint[2:]
            checkpoint_info['path'] = checkpoint
            checkpoint = gsutil.cp_ckpt(checkpoint)
        if checkpoint:
            self._policy.load_checkpoint(checkpoint)
            self.checkpoints_info.append(checkpoint_info)
            self.current_checkpoint = checkpoint_info

    def __getattr__(self, item):
        return getattr(self._policy, item)

    def reset(self):
        self._stacker.reset()
        self.resets += 1
        if self.checkpoint_reload_rate and self.resets % self.checkpoint_reload_rate == 0:
            self.resets = 0
            self.update_checkpoint()
        self._policy.reset()
