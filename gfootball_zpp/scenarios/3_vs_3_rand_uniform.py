from gfootball.scenarios import *
from .utils import get_player_position_from_uniform


def build_scenario(builder):
    builder.config().game_duration = 3000
    builder.config().right_team_difficulty = 0.05
    builder.config().left_team_difficulty = 0.05
    builder.config().deterministic = False
    if builder.EpisodeNumber() % 2 == 0:
        first_team = Team.e_Left
        second_team = Team.e_Right
    else:
        first_team = Team.e_Right
        second_team = Team.e_Left

    
    #ball_x, ball_y, _ = get_player_position_from_uniform(-0.4, -0.2, 0.4, 0.2, e_PlayerRole_CF)
    #builder.SetBallPosition(ball_x, ball_y)

    builder.SetTeam(first_team)
    builder.AddPlayer(-100.000000, 0.000000, e_PlayerRole_GK, controllable=False)
    builder.AddPlayer(*get_player_position_from_uniform(-0.8, -0.45, 0.8, 0.45, e_PlayerRole_CF))
    builder.AddPlayer(*get_player_position_from_uniform(-0.8, -0.45, 0.8, 0.45, e_PlayerRole_CF))
    builder.SetTeam(second_team)
    builder.AddPlayer(-100.000000, 0.000000, e_PlayerRole_GK, controllable=False)
    builder.AddPlayer(*get_player_position_from_uniform(-0.8, -0.45, 0.8, 0.45, e_PlayerRole_CF))
    builder.AddPlayer(*get_player_position_from_uniform(-0.8, -0.45, 0.8, 0.45, e_PlayerRole_CF))
