import gym


class BallOwnershipRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward."""
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self._ball_ownership_reward = 0.2

    def reset(self):
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            if 'ball_owned_team' not in o:
                if o['ball_owned_team'] == 0:
                    reward[rew_index] += self._ball_ownership_reward
                elif o['ball_owned_team'] == 1:
                    reward[rew_index] -= self._ball_ownership_reward

        return reward
