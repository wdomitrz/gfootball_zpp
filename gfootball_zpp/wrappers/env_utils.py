import gym
import re
from gfootball_zpp.utils.env import change_scenario

r_pl_re = re.compile(r'right_players=/d+')
l_pl_re = re.compile(r'left_players=/d+')


class EnvUtilsWrapper(gym.Wrapper):
    """Exposes useful functions that change low level parameters of football env.
    """
    def __init__(self, env, config):
        super().__init__(env)
        self._extra_players = config['extra_players']
        self._hidden_players = list(
            map(lambda p: re.sub(r_pl_re, 'right_playes=0', re.sub(l_pl_re, 'left_players=0', p)),
                self._extra_players))
        self._ids_to_hide = set()
        self._ids_to_show = set()

    def hide_player(self, id=0):
        self._ids_to_show.remove(id)
        self._ids_to_hide.add(id)
        self.unwrapped._config._values['players'][id + 1] = self._hidden_players[id]

    def show_player(self, id=0):
        self._ids_to_hide.remove(id)
        self._ids_to_show.add(id)
        self.unwrapped._config._values['players'][id + 1] = self._extra_players[id]

    def reset(self, **kwargs):
        for id in self._ids_to_hide:
            self.unwrapped._players[id].hide()
        for id in self._ids_to_show:
            self.unwrapped._players[id].hide()
        self._ids_to_hide.clear()
        self._ids_to_show.clear()
        return super().reset(**kwargs)

    def set_right_player_name(self, name):
        self.unwrapped._config._values['right_team_name'] = name

    def set_left_player_name(self, name):
        self.unwrapped._config._values['left_team_name'] = name

    def change_scenario(self, level):
        change_scenario(self, level)

    def load_player_checkpoint(self, checkpoint, id=0):
        self.unwrapped._config._values['players'][id + 1].load_checkpoint(checkpoint)

