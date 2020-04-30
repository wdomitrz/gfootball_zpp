
import pandas as pd 

relevant_names = [
    #('actions-ball_text', None, 'actions-ball_text'),
    #('actions-players',None,'actions-players'),
    ('ball-not_owning',None,'ball-not_owning'),
    ('ball-owning_first_team', 'ball-owning_second_team', 'ball_possesion'),
    ('cards-first_team-red_cards','cards-second_team-red_cards','red_cards'),
    ('cards-first_team-yellow_cards','cards-second_team-yellow_cards', 'yellow_cards'),
    ('game_modes-Corner','game_modes-Corner','Corner'),
    ('game_modes-FreeKick','game_modes-FreeKick','FreeKick'),
    ('game_modes-GoalKick','game_modes-GoalKick','GoalKick'),
    ('game_modes-KickOff','game_modes-KickOff','KickOff'),
    ('game_modes-Normal','game_modes-Normal','Normal'),
    ('game_modes-Penalty','game_modes-Penalty','Penalty'),
    ('game_modes-ThrowIn','game_modes-ThrowIn','ThrowIn'),
    ('goals-first_team-own', 'goals-second_team-own', 'goals-own'),
    ('goals-first_team-shot', 'goals-second_team-shot', 'goals-shot'),
    ('local_advantages-first_team-avg_advantage_r_0.05', None, 'local_advantage_r_0.05'),
    ('local_advantages-first_team-avg_advantage_r_0.1', None, 'local_advantage_r_0.1'),
    ('local_advantages-first_team-avg_advantage_r_0.2', None, 'local_advantage_r_0.2'),
    ('local_advantages-first_team-avg_advantage_r_0.3', None, 'local_advantage_r_0.3'),
    ('passes-first_team-all_passes','passes-second_team-all_passes', 'all_passes'),
    ('passes-first_team-all_passes_avg_dist','passes-second_team-all_passes_avg_dist', 'all_passes_avg_dist'),
    ('passes-first_team-intentional_passes','passes-second_team-intentional_passes', 'intentional_passes'),
    ('passes-first_team-intentional_passes_avg_dist','passes-second_team-intentional_passes_avg_dist', 'intentional_passes_avg_dist'),
    ('player_entropy-first_team-ball_avg_dist_avg', 'player_entropy-second_team-ball_avg_dist_avg', 'player_entropy_ball_avg_dist_avg'),
    ('shots-first_team-all_shots','shots-second_team-all_shots','all_shots'),
    ('shots-first_team-all_shots_avg_dist_from_goal','shots-second_team-all_shots_avg_dist_from_goal','all_shots_avg_dist_from_goal'),
    #('warning_strongly_inaccurate_actions-proportions_ball_owned_controlled_player', None, 'actions-proportions_ball_owned_controlled_player'),
    #('warning_strongly_inaccurate_actions-proportions_player_0', None, 'actions-proportions_player_0'),
    #('warning_strongly_inaccurate_actions-proportions_player_1', None, 'actions-proportions_player_1'),
    #('warning_strongly_inaccurate_actions-proportions_player_2', None, 'actions-proportions_player_2'),
    #('warning_strongly_inaccurate_actions-proportions_player_3', None, 'actions-proportions_player_3')
]

def join_left_right_team_statistics(left_name, right_name, statistic_name='values'):
    dfl = pd.read_csv('raw/' + left_name, sep=';')
    df_joined = pd.DataFrame(data={
            'team': dfl['1'],
            'opponent': dfl['2'],
            'scenario': dfl['0'],
            'was_first': [True] * len(dfl),
            statistic_name: dfl['values']
        })
    if right_name is None:
        return df_joined
    dfr = pd.read_csv('raw/' + right_name, sep=';')
    return df_joined.append(pd.DataFrame(data={
            'team': dfr['2'],
            'opponent': dfr['1'],
            'scenario': dfr['0'],
            'was_first': [False] * len(dfl),
            statistic_name: dfr['values']
        }))

df = pd.DataFrame()
for left_stat, right_stat, name in relevant_names:
    df = df.append(join_left_right_team_statistics(left_stat, right_stat, name))

df.to_csv('aggregated.csv')
