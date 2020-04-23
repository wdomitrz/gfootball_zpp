import gym
import re

r_pl_re = re.compile(r'right_players=/d+')
l_pl_re = re.compile(r'left_players=/d+')


class HidePlayerWrapper(gym.Wrapper):
    """Exposes function to disable or enable zpp players.

    Should be used directly before reset!
    """
    def __init__(self, env, config):
        super().__init__(env)
        self._extra_players = config.extra_players
        self._hidden_players = list(
            map(lambda p: re.sub(r_pl_re, 'right_playes=0', re.sub(l_pl_re, 'left_players=0', p)),
                self._extra_players))

    def hide_player(self, id=0):
        self.unwrapped._players[id].hide()
        self.unwrapped._config.values['players'][id + 1] = self._hidden_players[id]

    def show_player(self, id=0):
        self.unwrapped._players[id].show()
        self.unwrapped._config.values['players'][id + 1] = self._extra_players[id]
