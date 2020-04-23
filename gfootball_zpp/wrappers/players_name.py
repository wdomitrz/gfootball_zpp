from gym import Wrapper
from gfootball_zpp.players.utils import retrieve_external_players_data


class UpdateTeamNamesWrapper(Wrapper):
    def __init__(self, env, config):
        super().__init__(env)
        self.players_data = retrieve_external_players_data(config)

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        if len(self.players_data) == 1:
            name = self.players_data[0]['name']
            if len(self.players_data[0]['checkpoints']):
                checkpoint = self.players_data[0]['checkpoints'][-1]
                path = checkpoint['path'].split('/')
                path = '/'.join(path[2:-4] + [path[-1]])
                checkpoint = '(' + checkpoint['type'] + ': ' + path + ')'
            else:
                checkpoint = ''
            self.unwrapped._config['right_team_name'] = name + checkpoint
        return obs
