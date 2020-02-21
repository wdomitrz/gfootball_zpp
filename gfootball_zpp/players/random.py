from .utils import AgentOutput
from numpy import random


class RandomPolicy:
    """Outputs random actions."""
    def __init__(self, output_actions=1):
        self.actions = output_actions

    def __call__(self, *args, **kwargs):
        return AgentOutput(random.randint(0, 19, self.actions),
                           None, None), None


def create_net(controlled_players):
    return RandomPolicy(controlled_players)


def convert_observation(o):
    return o
