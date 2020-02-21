import gfootball_zpp.players.random as random_policy

policies = {'random': random_policy}


def build_policy(name, controlled_players):
    """Returns network and function converting observation to expected format."""
    policy = policies[name]
    return policy.create_net(controlled_players), policy.convert_observation


def restore_checkpoint(network, checkpoint_path=None):
    """Restores given checkpoint to the network if path is given."""
    if checkpoint_path is None:
        return network
    raise NotImplementedError
