from gfootball_zpp.players.random import RandomPolicy
from gfootball_zpp.players.heads import HeadsPlayer

policies = {'random': RandomPolicy, 'multihead': HeadsPlayer}


def build_policy(name, controlled_players, player_config=None):
    """Returns network and function converting observation to expected format."""
    policy = policies[name]
    return policy(controlled_players, player_config)
