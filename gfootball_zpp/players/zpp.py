from gfootball.env import player_base
from gfootball_zpp.players import build_policy, restore_checkpoint


class Player(player_base.PlayerBase):
    """An agent loaded from one of ZPP checkpoints.

    This file should be copied into gfootball/env/players directory.
    To use it in the game you need to pass extra_players:
    'zpp:policy={policy_name},right_players={n},checkpoint={path}'."""

    def __init__(self, player_config, env_config):
        player_base.PlayerBase.__init__(self, player_config)

        self._action_set = 'default'
        self._player_prefix = 'player_{}'.format(player_config['index'])
        # todo: stack observations if needed
        # stacking = 4 if player_config.get('stacked', True) else 1
        policy = player_config.get('policy', '')

        self._policy, self._convert_observation = build_policy(
            policy, self.num_controlled_players())
        restore_checkpoint(self._policy, player_config.get('checkpoint', None))

    def take_action(self, observation):
        observation = self._convert_observation(observation)
        agent_output, _ = self._policy(observation)
        action = agent_output.action
        return action

    def reset(self):
        # todo: reset stacking
        pass
