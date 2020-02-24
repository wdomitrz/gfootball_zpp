from numpy import random


class RandomPolicy:
    """Outputs random actions."""
    def __init__(self, output_actions=1):
        self.actions = output_actions

    def __call__(self, *args, **kwargs):
        return random.randint(0, 19, self.actions)


def create_net(controlled_players, checkpoint):
    return RandomPolicy(controlled_players)


def convert_observation(o):
    return o
