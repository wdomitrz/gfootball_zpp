from .utils import LogBasicTracker, EnvLogSteppingModes, get_opponent_name, player_with_ball_action
import tensorflow as tf
import numpy as np
import math
from absl import logging


class LogBallOwningTeam(LogBasicTracker):
    """ WARNING this wrapper returns approximate results
    due to the fact that observation is generated after x
    game engine steps """
    def _trace_vars_reset(self):
        self._first_team_time = 0  # in environment steps
        self._second_team_time = 0  # in environment steps

    def _update_stats(self, observation):
        if observation[0]['ball_owned_team'] == 0:
            self._first_team_time += 1
        elif observation[0]['ball_owned_team'] == 1:
            self._second_team_time += 1

    def _write_logs(self, category, first_team_owning, second_team_owning,
                    not_owning):
        self.summary_writer.write_scalar(
            '{}/owning_first_team'.format(category), first_team_owning)
        self.summary_writer.write_scalar(
            '{}/owning_second_team'.format(category), second_team_owning)
        self.summary_writer.write_scalar('{}/not_owning'.format(category),
                                         not_owning)

    def __init__(self, env, config):
        LogBasicTracker.__init__(self, env, config)

        self._trace_vars_reset()

        self.summary_writer.set_stepping(EnvLogSteppingModes.env_resets)

    def reset(self):
        env_episode_steps = self.env_episode_steps
        if env_episode_steps != 0:
            first_team_owning = self._first_team_time / env_episode_steps
            second_team_owning = self._second_team_time / env_episode_steps
            ball_free_time = env_episode_steps - \
                             (self._first_team_time +
                              self._second_team_time)
            not_owning = ball_free_time / env_episode_steps

            self._write_logs('ball', first_team_owning, second_team_owning,
                             not_owning)
            self._write_logs(
                'per_opponent_ball/' + get_opponent_name(self.env),
                first_team_owning, second_team_owning, not_owning)

        observation = super(LogBallOwningTeam, self).reset()

        self._trace_vars_reset()
        # self._update_stats(observation) - we do not count initial observation
        return observation

    def step(self, action):
        observation, reward, done, info = super(LogBallOwningTeam,
                                                self).step(action)
        self._update_stats(observation)
        return observation, reward, done, info


FRAME_THRESHOLD = 5

PASS_ACTIONS = [9, 10, 11]
SHOT_ACTIONS = [12]

def get_player_obs(playerNum, observation):
    for obs in observation:
        if obs['active'] == playerNum:
            return obs
    raise Exception('Cannot find observation for player with number {}'.format(playerNum))

def get_player_action(playerNum, observation, action):
    def player_id(playerNum, observation):
        for i, obs in enumerate(observation):
            if obs['active'] == playerNum:
                return i
        return None

    pId = player_id(playerNum, observation)
    if pId is not None:
        return action[pId]
    else:
        return None

def remove_closest_goal(players, goal_id):
    s = sorted(players, key=lambda p: p[0] if goal_id == 0 else -p[0])
    it = iter(s)
    _, tail = next(it), list(it)
    return tail
    

# def remove_gk(players, roles):
#     result = []
#     for (p, r) in zip(players, roles):
#         if r != 0: #not goal keeper
#             result.append(p)
#     return result


class BallOwnInfo():
    def __init__(self, last_team=None, last_player=None, last_own_pos=None, last_observation=None):
        self.last_team = last_team
        self.last_player = last_player
        self.last_own_pos = last_own_pos
        self.last_observation = last_observation
        self.intentionall_pass = False
        self.intentionall_shot = False
        self.delay_counter = 0
        self.time_since_update = 0

    def update(self, observation, action):
        current_player = observation[0]['ball_owned_player']
        if current_player != -1:
            current_team = observation[0]['ball_owned_team']
            current_pos = observation[0]['ball'][0:2]  # third is alt
            current_obs = observation
        else:
            current_pos = None
            current_team = -1
            current_obs = None

        self.time_since_update += 1
        if self.last_team == 0 \
           and self.time_since_update <= 1:  # todo
            ball_owner_action = get_player_action(self.last_player, self.last_observation, action)
            if ball_owner_action in PASS_ACTIONS:
                self.intentionall_pass = True
                self.delay_counter = FRAME_THRESHOLD  # Expected time when ball should change state to not owned or owned by other player

            if ball_owner_action in SHOT_ACTIONS:
                self.intentionall_shot = True
                self.delay_counter = FRAME_THRESHOLD

        if (current_player == -1) or (self.delay_counter > 0 and (current_player == self.last_player)):
            self.delay_counter -= 1
            self.delay_counter = max(0, self.delay_counter)
            if current_player == -1:
                self.delay_counter = 0
            return self
        else:
            return BallOwnInfo(current_team,
                               current_player,
                               current_pos,
                               current_obs)

    def inited(self):
        if self.last_team is not None:
            assert ((self.last_player is not None) \
                    and (self.last_player >= 0) \
                    and (self.last_own_pos is not None) \
                    and (self.last_observation is not None))
            return True
        else:
            return False

    def ball_passed(self, ball_own_info):
        return self.last_team == ball_own_info.last_team and self.last_player != ball_own_info.last_player and self.inited(
        ) and ball_own_info.inited()

    def ball_passed_intentionally(self, ball_own_info):
        return self.intentionall_pass and self.ball_passed(ball_own_info)

    def offside_position(self, ball_pos, goal_id): # todo
        player_pos = self.last_own_pos
        obs = self.last_observation[0]
        player_team = self.last_team

        #print (obs)

        if goal_id == 1:
            op = lambda x, y: x >= y
        elif goal_id == 0:
            op = lambda x, y: x <= y
        else:
            raise Exception('Wrong goal parameter')

        opponents = [remove_closest_goal(obs['left_team'], 0),
                     remove_closest_goal(obs['right_team'], 1)]

        #print('player: {} opponenets {}'.format(player_team, opponents[1 - player_team]))

        if op(0.0, player_pos[0]): # half check
            return False
        
        for p_pos in opponents[1 - player_team]:
            if op(p_pos[0], player_pos[0]):
                return False
        
        if op(ball_pos[0], player_pos[0]):
            return False

        return True
        
        

    def ball_offside(self, ball_own_info):
        return self.inited() and ball_own_info.inited() and self.ball_passed(ball_own_info) and ball_own_info.offside_position(self.last_own_pos, 1 - self.last_team)

    def ball_lost(self, ball_own_info):
        return self.last_team != ball_own_info.last_team and self.inited(
        ) and ball_own_info.inited()

    def dist(self, ball_own_info):
        #assert(self.last_team == ball_own_info.last_team)
        diff_x = self.last_own_pos[0] - ball_own_info.last_own_pos[0]
        diff_y = self.last_own_pos[1] - ball_own_info.last_own_pos[1]
        return math.sqrt(pow(diff_x, 2) + pow(diff_y, 2))

    def dist_from_goal(self, goal_id):
        if goal_id == 1:
            goal_x = 1.0
        elif goal_id == 0:
            goal_x = -1.0
        else:
            raise Exception('Wrong goal parameter')
        goal_y = 0.0
        diff_x = self.last_own_pos[0] - goal_x
        diff_y = self.last_own_pos[1] - goal_y
        return math.sqrt(pow(diff_x, 2) + pow(diff_y, 2))


class LogPassStatsTeam(LogBasicTracker):
    """ Warning can produce inaccurate results """
    def _trace_vars_reset(self):
        self._ball_own_info = BallOwnInfo()
        self._all_passes = [0, 0]
        self._all_passes_dist_sum = [0.0, 0.0]

        # currently only for controlled players/team
        self._intentional_passes = [0, 0]
        self._intentional_passes_dist_sum = [0, 0]

    def _update_stats(self, observation, action):
        if observation[0][
                'game_mode'] != 0:  # reset tracking when gamemode is not normal
            self._ball_own_info = BallOwnInfo()
            return

        current_ball_own_info = self._ball_own_info.update(observation, action)

        if self._ball_own_info.ball_passed(current_ball_own_info):
            self._all_passes[self._ball_own_info.last_team] += 1
            self._all_passes_dist_sum[
                self._ball_own_info.last_team] += self._ball_own_info.dist(
                    current_ball_own_info)
            #print("$$$$$$$$$$PASS DETECTED", self._all_passes,
                  #self._all_passes_dist_sum)
            if self._ball_own_info.ball_passed_intentionally(
                    current_ball_own_info):
                self._intentional_passes[self._ball_own_info.last_team] += 1
                self._intentional_passes_dist_sum[
                    self._ball_own_info.last_team] += self._ball_own_info.dist(
                        current_ball_own_info)
                #print("$$$$$$$$$$###INT PASS DETECTED",
                      #self._intentional_passes,
                      #self._intentional_passes_dist_sum)

        self._ball_own_info = current_ball_own_info

    def _write_logs(self, category):
        for tId, teamName in enumerate(['first_team', 'second_team']):
            all_passes = self._all_passes[tId]
            if all_passes == 0:
                all_passes_avg_dist = 0.0
            else:
                all_passes_avg_dist = self._all_passes_dist_sum[
                    tId] / all_passes

            self.summary_writer.write_scalar(
                '{}/{}/all_passes'.format(category, teamName), all_passes)
            self.summary_writer.write_scalar(
                '{}/{}/all_passes_avg_dist'.format(category, teamName),
                all_passes_avg_dist)

            intentional_passes = self._intentional_passes[tId]
            if intentional_passes == 0:
                intentional_passes_avg_dist = 0.0
            else:
                intentional_passes_avg_dist = self._intentional_passes_dist_sum[
                    tId] / intentional_passes
                
            self.summary_writer.write_scalar(
                '{}/{}/intentional_passes'.format(category, teamName),
                intentional_passes)
            self.summary_writer.write_scalar(
                '{}/{}/intentional_passes_avg_dist'.format(
                    category, teamName), intentional_passes_avg_dist)

    def __init__(self, env, config):
        LogBasicTracker.__init__(self, env, config)

        self._trace_vars_reset()
        self._idle_action = np.array(self.env.action_space.sample(),
                                     dtype=np.int64) * 0

        self.summary_writer.set_stepping(EnvLogSteppingModes.env_resets)

    def reset(self):
        env_episode_steps = self.env_episode_steps
        if env_episode_steps != 0:
            self._write_logs('passes')
            self._write_logs('per_opponent_passes/' +
                             get_opponent_name(self.env))

        observation = super(LogPassStatsTeam, self).reset()

        self._trace_vars_reset()
        self._update_stats(observation, self._idle_action)
        return observation

    def step(self, action):
        observation, reward, done, info = super(LogPassStatsTeam,
                                                self).step(action)
        self._update_stats(observation, action)
        return observation, reward, done, info


class LogShotStatsTeam(LogBasicTracker):
    """ Warning can produce inaccurate results """
    def _trace_vars_reset(self):
        self._ball_own_info = BallOwnInfo()
        self._last_score = [0, 0]
        self._all_shots = [0, 0]
        self._all_shots_dist_sum = [0.0, 0.0]

    def _update_stats(self, observation, action):
        current_ball_own_info = self._ball_own_info.update(observation, action)

        if self._ball_own_info.intentionall_shot \
           and observation[0]['ball_owned_team'] == -1:
            self._all_shots[self._ball_own_info.last_team] += 1
            self._all_shots_dist_sum[
                self._ball_own_info.
                last_team] += self._ball_own_info.dist_from_goal(1 - self._ball_own_info.
                last_team)
            #print("$$$ shot detected", self._all_shots, ",", self._all_shots_dist_sum)
            self._ball_own_info = BallOwnInfo()
        else:
            self._ball_own_info = current_ball_own_info

    def _write_logs(self, category):
        for tId, teamName in enumerate(['first_team', 'second_team']):
            all_shots = self._all_shots[tId]
            if all_shots == 0:
                all_shots_avg_dist = 0.0
            else:
                all_shots_avg_dist = self._all_shots_dist_sum[tId] / all_shots

            self.summary_writer.write_scalar(
                '{}/{}/all_shots'.format(category, teamName), all_shots)
            self.summary_writer.write_scalar(
                '{}/{}/all_shots_avg_dist_from_goal'.format(
                    category, teamName), all_shots_avg_dist)

    def __init__(self, env, config):
        LogBasicTracker.__init__(self, env, config)

        self._trace_vars_reset()
        self._idle_action = np.array(self.env.action_space.sample(),
                                     dtype=np.int64) * 0

        self.summary_writer.set_stepping(EnvLogSteppingModes.env_resets)

    def reset(self):
        env_episode_steps = self.env_episode_steps
        if env_episode_steps != 0:
            self._write_logs('shots')
            self._write_logs('per_opponent_shots/' +
                             get_opponent_name(self.env))

        observation = super(LogShotStatsTeam, self).reset()

        self._trace_vars_reset()
        self._update_stats(observation, self._idle_action)
        return observation

    def step(self, action):
        observation, reward, done, info = super(LogShotStatsTeam,
                                                self).step(action)
        self._update_stats(observation, action)
        return observation, reward, done, info


class LogOffsidesTeam(LogBasicTracker):
    """ Warning can produce inaccurate results """
    def _trace_vars_reset(self):
        self._ball_own_info = BallOwnInfo()
        self._offsides = [0, 0]

    def _update_stats(self, observation, action):
        current_ball_own_info = self._ball_own_info.update(observation, action)

        #print(observation)
        #print(self._ball_own_info.last_player)
        if self._ball_own_info.inited() and self._ball_own_info.ball_offside(current_ball_own_info):
            self._offsides[self._ball_own_info.last_team] += 1
            #print("&&&OFFSIDE ", str(self._offsides))

        current_ball_own_info.delay_counter = 0
        if observation[0]['game_mode'] != 0:  # reset tracking when gamemode is not normal
            self._ball_own_info = BallOwnInfo()
        else:
            self._ball_own_info = current_ball_own_info

    def _write_logs(self, category):
        for tId, teamName in enumerate(['first_team', 'second_team']):
            team_offsides = self._offsides[tId]
            self.summary_writer.write_scalar(
                '{}/{}'.format(category, teamName), team_offsides)

    def __init__(self, env, config):
        LogBasicTracker.__init__(self, env, config)

        self._trace_vars_reset()
        self._idle_action = np.array(self.env.action_space.sample(),
                                     dtype=np.int64) * 0

        self.summary_writer.set_stepping(EnvLogSteppingModes.env_resets)

    def reset(self):
        env_episode_steps = self.env_episode_steps
        if env_episode_steps != 0:
            self._write_logs('offsides')
            self._write_logs('per_opponent_offsides/' +
                             get_opponent_name(self.env))

        observation = super(LogOffsidesTeam, self).reset()

        self._trace_vars_reset()
        self._update_stats(observation, self._idle_action)
        return observation

    def step(self, action):
        observation, reward, done, info = super(LogOffsidesTeam,
                                                self).step(action)
        self._update_stats(observation, action)
        return observation, reward, done, info
