import gym
from gfootball_zpp.utils.env import change_scenario
from absl import logging


def preprocess_config(config):
    config['extra_players'] = ['zpp:policy=multihead,right_players=4,sample=True']
    config['enable_full_episode_videos'] = True
    config['dump_frequency'] = 1


class EvalWrapper(gym.Wrapper):
    def __init__(self, env, config):
        logging.info("Creating eval wrapper")
        super().__init__(env)
        self._scenarios = config['eval_scenarios']
        self._current_scenario = 0
        self._prepare_scenario()

    def _prepare_scenario(self):
        level, checkpoint = self._scenarios[self._current_scenario]
        logging.info("Preparing scenario %s %s", level, checkpoint if checkpoint else "bots.")
        change_scenario(self, level)
        if not checkpoint:
            self.hide_player()
            self.set_right_player_name(level + '_bots')
        else:
            self.show_player()
            self.unwrapped._players[1].load_checkpoint(checkpoint)
            self.set_right_player_name('_'.join(filter(lambda x: x != '' and x != 'GS',checkpoint.split('/'))))

    def reset(self):
        self._prepare_scenario()
        self._current_scenario = (self._current_scenario + 1) % len(self._scenarios)
        return super().reset()
