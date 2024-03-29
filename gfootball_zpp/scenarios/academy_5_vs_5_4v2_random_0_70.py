# coding=utf-8
# Copyright 2019 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import random
from gfootball.scenarios import *
from .utils import get_player_position_from_uniform

def build_scenario(builder):
    builder.config().game_duration = 400
    builder.config().deterministic = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = True
    builder.config().left_team_difficulty = random.uniform(0, 1)
    builder.config().right_team_difficulty = random.uniform(0, 1)

    left_players = []
    left_players.append(get_player_position_from_uniform(0.6, 0.1, 0.7, -0.1, e_PlayerRole_RM))
    left_players.append(get_player_position_from_uniform(0.6, 0.1, 0.7, -0.1, e_PlayerRole_CF))
    left_players.append(get_player_position_from_uniform(0.65, -0.1, 0.75, -0.15, e_PlayerRole_LB))
    left_players.append(get_player_position_from_uniform(0.65, 0.1, 0.75, 0.15, e_PlayerRole_CB))

    right_players = []
    # Two players actually defend
    right_players.append(get_player_position_from_uniform(-0.8, -0.1, -0.9, 0.1, e_PlayerRole_RM))
    right_players.append(get_player_position_from_uniform(-0.8, -0.1, -0.9, 0.1, e_PlayerRole_CF))
    # The other two are at the opposite side of the pitch
    right_players.append(get_player_position_from_uniform(0.8, -0.2, 0.9, 0.2, e_PlayerRole_LB))
    right_players.append(get_player_position_from_uniform(0.8, -0.2, 0.9, 0.2, e_PlayerRole_CB))

    (ball_x, ball_y, _) = random.choice(left_players)
    builder.SetBallPosition(ball_x, ball_y)

    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.000000, 0.000000, e_PlayerRole_GK, controllable=False)
    for (x, y, player_role) in left_players:
        builder.AddPlayer(x, y, player_role)


    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.000000, 0.000000, e_PlayerRole_GK, controllable=False)
    for (x, y, player_role) in right_players:
        builder.AddPlayer(x, y, player_role)
