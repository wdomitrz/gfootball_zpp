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


from gfootball.scenarios import *


def build_scenario(builder):
    builder.config().game_duration = 400
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = True
    builder.SetBallPosition(-0.62, 0.050000)

    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.000000, 0.000000, e_PlayerRole_GK, controllable=False)
    builder.AddPlayer(-0.750000, -0.050000, e_PlayerRole_RM)
    builder.AddPlayer(-0.750000, 0.050000, e_PlayerRole_LM)
    builder.AddPlayer(-0.800000, -0.100000, e_PlayerRole_CF)
    builder.AddPlayer(-0.800000, 0.100000, e_PlayerRole_CF)
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.000000, 0.000000, e_PlayerRole_GK, controllable=False)
    builder.AddPlayer(-0.650000, 0.050000, e_PlayerRole_RM)
    builder.AddPlayer(-0.650000, -0.050000, e_PlayerRole_CF)
    builder.AddPlayer(0.600000, -0.050000, e_PlayerRole_LB)
    builder.AddPlayer(0.600000,  0.050000, e_PlayerRole_CB)
