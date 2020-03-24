import gym
import numpy as np
import collections
from absl import logging


class DecayingCheckpointRewardWrapper(gym.RewardWrapper):
  """A wrapper that adds a dense checkpoint reward."""

  def __init__(self, env, checkpoint_base_reward=0.1, decreasing_reward_treshold=0.5, steps_to_get_from_checkpoints_to_scoring=300, number_of_prev_episodes=None):
    gym.RewardWrapper.__init__(self, env)
    self._collected_checkpoints = {}
    self._num_checkpoints = 10
    self._checkpoint_reward = checkpoint_base_reward

    self._checkpoint_reward_change = self._checkpoint_reward / \
        steps_to_get_from_checkpoints_to_scoring
    self._decreasing_reward_treshold = decreasing_reward_treshold
    self._episode_rewards = []
    self._prev_episodes_rewards = collections.deque(
        maxlen=number_of_prev_episodes)

  def reset(self):
    self._collected_checkpoints = {}

    self._prev_episodes_rewards.append(np.sum(self._episode_rewards))
    self._episode_rewards = []

    def safemean(xs):
      return 0 if len(xs) == 0 else np.mean(xs)

    if safemean(self._prev_episodes_rewards) > self._decreasing_reward_treshold:
      self._checkpoint_reward = max(
          self._checkpoint_reward - self._checkpoint_reward_change, 0)

    return self.env.reset()

  def get_state(self, to_pickle):
    to_pickle['CheckpointRewardWrapper'] = (
        self._collected_checkpoints, self._checkpoint_reward, self._prev_episodes_rewards)
    return self.env.get_state(to_pickle)

  def set_state(self, state):
    from_pickle = self.env.set_state(state)
    self._collected_checkpoints, self._checkpoint_reward, self._prev_episodes_rewards = from_pickle[
        'CheckpointRewardWrapper']
    return from_pickle

  def reward(self, reward):
    observation = self.env.unwrapped.observation()
    if observation is None:
      return reward

    assert len(reward) == len(observation)

    for rew_index in range(len(reward)):
      o = observation[rew_index]
      if reward[rew_index] == 1:
        reward[rew_index] += self._checkpoint_reward * (
            self._num_checkpoints -
            self._collected_checkpoints.get(rew_index, 0))
        self._collected_checkpoints[rew_index] = self._num_checkpoints
        continue

      # Check if the active player has the ball.
      if ('ball_owned_team' not in o or
          o['ball_owned_team'] != 0 or
          'ball_owned_player' not in o or
              o['ball_owned_player'] != o['active']):
        continue

      d = ((o['ball'][0] - 1) ** 2 + o['ball'][1] ** 2) ** 0.5

      # Collect the checkpoints.
      # We give reward for distance 1 to 0.2.
      while (self._collected_checkpoints.get(rew_index, 0) <
             self._num_checkpoints):
        if self._num_checkpoints == 1:
          threshold = 0.99 - 0.8
        else:
          threshold = (0.99 - 0.8 / (self._num_checkpoints - 1) *
                       self._collected_checkpoints.get(rew_index, 0))
        if d > threshold:
          break
        reward[rew_index] += self._checkpoint_reward
        self._collected_checkpoints[rew_index] = (
            self._collected_checkpoints.get(rew_index, 0) + 1)
    return reward

  def step(self, action):
    observation, reward, done, info = self.env.step(action)

    if "score_reward" in info:
      self._episode_rewards.append(info["score_reward"])

    return observation, self.reward(reward), done, info
