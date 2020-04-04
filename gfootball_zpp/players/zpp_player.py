
class BaseZppPlayer:
    def take_action(self, obs):
        raise NotImplementedError

    def pre_stacking_convert_obs(self, obs):
        raise NotImplementedError

    def reset(self):
        """Optional action performed at env reset"""
        pass

    def load_checkpoint(self, checkpoint):
        raise NotImplementedError
