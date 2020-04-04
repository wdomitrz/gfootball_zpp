from numpy import random
from .zpp_player import BaseZppPlayer


class RandomPolicy(BaseZppPlayer):
    """Outputs random actions."""
    def __init__(self, controlled_players=1, player_config=None):
        self.actions = controlled_players

    def pre_stacking_convert_obs(self, obs):
        return obs

    def take_action(self, obs):
        return random.randint(0, 19, self.actions)
