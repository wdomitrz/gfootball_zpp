import gfootball_zpp.players.random as random_policy
import gfootball_zpp.players.heads as multiheads_policy

policies = {'random': random_policy, 'multihead': multiheads_policy}


def build_policy(name, controlled_players, checkpoint=None):
    """Returns network and function converting observation to expected format."""
    policy = policies[name]
    return policy.create_net(controlled_players, checkpoint), policy.convert_observation
