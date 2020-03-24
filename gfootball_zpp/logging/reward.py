from .utils import LogBasicTracker, EnvLogSteppingModes
from ..utils.misc import scalar_to_list, pretty_list_of_pairs_to_string, get_with_prec

import tensorflow as tf
import numpy as np
import math


class LogPerPlayerReward(LogBasicTracker):
    def _trace_vars_reset(self):
        if self._num_rewards is not None:
            self._rewards = np.zeros(self._num_rewards, dtype=np.float64)
        else:
            self._rewards = None

    def _update_step(self, reward):
        if self._num_rewards is None:
            self._num_rewards = len(reward)
            self._trace_vars_reset()

        reward = np.array(reward, dtype=np.float64)
        assert reward.ndim == self._rewards.ndim
        self._rewards = self._rewards + reward

        self.summary_writer.set_stepping(EnvLogSteppingModes.env_total_steps)
        for rid in range(self._num_rewards):
            self.summary_writer.write_scalar(
                'reward/step/reward_{}'.format(rid), self._rewards[rid])

    def __init__(self, env, config):
        LogBasicTracker.__init__(self, env, config)

        self._num_rewards = None

        self._trace_vars_reset()

    def reset(self):

        if self._rewards is not None:
            self.summary_writer.set_stepping(EnvLogSteppingModes.env_resets)
            for rid in range(self._num_rewards):
                self.summary_writer.write_scalar(
                    'reward/game/reward_{}'.format(rid), self._rewards[rid])

        observation = super(LogPerPlayerReward, self).reset()

        self._trace_vars_reset()
        return observation

    def step(self, action):
        observation, reward, done, info = super(LogPerPlayerReward,
                                                self).step(action)
        self._update_step(scalar_to_list(reward))
        return observation, reward, done, info


class LogAveragePerPlayerRewardByDifficulty(LogBasicTracker):
    """ This is a low level wrapper. """
    def _trace_vars_reset(self):
        if self._num_rewards is not None:
            self._episode_rewards = np.zeros(shape=(self._num_rewards, ))
        else:
            self._episode_rewards = None

    def _trace_vars_set(self):
        self._trace_vars_reset()
        if self._num_rewards is not None:
            self._rewards = np.zeros(shape=(2, self._num_difficulties,
                                            self._average_last,
                                            self._num_rewards),
                                     dtype=np.float64)
            self._rewards_step = np.zeros(shape=(2, self._num_difficulties),
                                          dtype=np.int64)
        else:
            self._rewards = None
            self._rewards_step = None

    def _get_reward_bucket(self, difficulty):
        return min(math.floor(difficulty * self._num_difficulties),
                   self._num_difficulties - 1)

    def _get_difficulties(self):
        scenario_config = self.env.unwrapped._config.ScenarioConfig()
        return [
            scenario_config.left_team_difficulty,
            scenario_config.right_team_difficulty
        ]

    def _update_reset(self):
        difficulties = self._get_difficulties()

        reward = np.array(self._episode_rewards, dtype=np.float64)
        for tid, diff in enumerate(difficulties):
            rew_bucket = self._get_reward_bucket(diff)
            put_id = self._rewards_step[tid][rew_bucket] % self._average_last
            self._rewards[tid][rew_bucket][put_id] = reward
            self._rewards_step[tid][
                rew_bucket] = self._rewards_step[tid][rew_bucket] + 1

    def _update_step(self, reward):
        if self._num_rewards is None:
            self._num_rewards = len(reward)
            self._trace_vars_set()

        reward = np.array(reward, dtype=np.float64)
        assert reward.ndim == self._episode_rewards.ndim
        self._episode_rewards = self._episode_rewards + reward

    def _log_rewards(self):
        team_names = ['left_team', 'right_team']
        text_log = ''
        for tid, _ in enumerate(self._get_difficulties()):
            text_log += '# Per difficulty reward for {}  \n'.format(
                team_names[tid])
            for rid in range(self._num_rewards):
                scale_factor = 1.0 / self._num_difficulties
                rewards = np.sum(self._rewards[tid, :, :, rid], axis=1)
                counts = np.minimum(self._rewards_step[tid],
                                    self._average_last)
                norm_counts = np.maximum(counts, 1)
                rewards /= norm_counts

                self.summary_writer.write_bars(
                    'reward/game/{}_difficulty_diff_props_{}'.format(
                        team_names[tid], rid), np.floor(rewards * 10 + 1000.0),
                    scale_factor, 0.5)

                text_reward = [('difficulty interval:{}_{} \
                avg:{}'.format(get_with_prec(scale_factor * did),
                               get_with_prec(scale_factor * (did + 1)),
                               counts[did]), get_with_prec(rewards[did]))
                               for did in range(self._num_difficulties)]
                text_log += '## Reward {}  \n'.format(rid) + \
                            pretty_list_of_pairs_to_string(text_reward)
        self.summary_writer.write_text('reward/game/difficulty_reward',
                                       text_log)

    def __init__(self, env, config):
        LogBasicTracker.__init__(self, env, config)

        self._num_rewards = None

        self._num_difficulties = 5
        self._average_last = 10

        self._trace_vars_set()

        self.summary_writer.set_stepping(EnvLogSteppingModes.env_resets)

    def reset(self):

        if self._rewards is not None:
            self._update_reset()
            self._log_rewards()

        self._trace_vars_reset()
        observation = super(LogAveragePerPlayerRewardByDifficulty,
                            self).reset()

        return observation

    def step(self, action):
        observation, reward, done, info = super(
            LogAveragePerPlayerRewardByDifficulty, self).step(action)
        self._update_step(scalar_to_list(reward))
        return observation, reward, done, info


class LogMeanPerOpponentReward(LogBasicTracker):
    """ This is a low level wrapper. """
    def _trace_vars_reset(self):
        self._mean_reward = None

    def _update_step(self, reward):
        if self._mean_reward is None:
            self._mean_reward = np.mean(reward)
        else:
            self.mean_reward += np.mean(reward)

    def _get_opponent_name(self):
        return NotImplementedError

    def __init__(self, env, config):
        LogBasicTracker.__init__(self, env, config)

        self._trace_vars_reset()

        self.summary_writer.set_stepping(EnvLogSteppingModes.env_resets)

    def __getattr__(self, attr):
        return getattr(self.env, attr)

    def reset(self):
        # prevents from logging empty episodes
        if self._mean_reward is not None:
            current_opponent_name = self._get_opponent_name()
            self.summary_writer.write_scalar(
                'reward_per_opponent/{}'.format(current_opponent_name),
                self._mean_reward,
                self.env_resets)

        self._trace_vars_reset()
        return super(LogMeanPerOpponentReward, self).reset()

    def step(self, action):
        observation, reward, done, info = super(LogMeanPerOpponentReward,
                                                self).step(action)
        self._update_step(reward)
        return observation, reward, done, info
