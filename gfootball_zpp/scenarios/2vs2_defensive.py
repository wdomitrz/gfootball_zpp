from gfootball.scenarios import *

def build_scenario(builder):
  builder.config().game_duration = 3000
  builder.config().right_team_difficulty = 0.95
  builder.config().left_team_difficulty = 0.95
  builder.config().deterministic = False
  if builder.EpisodeNumber() % 2 == 0:
    first_team = Team.e_Left
    second_team = Team.e_Right
  else:
    first_team = Team.e_Right
    second_team = Team.e_Left
  builder.SetTeam(first_team)
  builder.AddPlayer(-0.1, -0.1, e_PlayerRole_GK)
  builder.AddPlayer(0.000000, 0.020000, e_PlayerRole_CF)
  builder.SetTeam(second_team)
  builder.AddPlayer(-0.5, -0.1, e_PlayerRole_GK)
  builder.AddPlayer(-0.2, 0.05, e_PlayerRole_CF)