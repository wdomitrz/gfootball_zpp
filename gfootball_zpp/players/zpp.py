from gfootball.env import player_base
from gfootball_zpp.players.players import build_policy
from gfootball_zpp.players.utils import ObservationStacker


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

        self._policy, self._convert_observation = build_policy(
            policy, self.num_controlled_players(), player_config.get('checkpoint', None))

    def take_action(self, observation):
        print(observation)
        observation = self._convert_observation(observation)
        observation = self._stacker.get(observation)
        return self._policy(observation)

    def reset(self):
        self._stacker.reset()