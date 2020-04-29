from absl import flags, app
import pandas as pd
import os
from extract_all import extract
flags.DEFINE_string('in_path', None, 'Path to dirs file')
flags.DEFINE_string('out', None, 'Path to output dir')
flags.DEFINE_integer('level', 3, 'dir depth')
flags.DEFINE_string(
    'metric_files',
    'ball-not_owning,ball-owning_first_team,ball-owning_second_team,cards-first_team-red_cards,cards-first_team-yellow_cards,cards-second_team-red_cards,cards-second_team-yellow_cards,game_modes-Corner,game_modes-FreeKick,game_modes-GoalKick,game_modes-KickOff,game_modes-Normal,game_modes-Penalty,game_modes-ThrowIn,goals-first_team-own,goals-first_team-shot,goals-second_team-own,goals-second_team-shot,local_advantages-first_team-avg_advantage_r_0.05,local_advantages-first_team-avg_advantage_r_0.1,local_advantages-first_team-avg_advantage_r_0.2,local_advantages-first_team-avg_advantage_r_0.3,passes-first_team-all_passes,passes-first_team-all_passes_avg_dist,passes-first_team-intentional_passes,passes-first_team-intentional_passes_avg_dist,passes-second_team-all_passes,passes-second_team-all_passes_avg_dist,passes-second_team-intentional_passes,passes-second_team-intentional_passes_avg_dist,player_entropy-first_team-ball_avg_dist_avg,player_entropy-second_team-ball_avg_dist_avg,reward-game-reward_0,reward-game-reward_1,reward-game-reward_2,reward-game-reward_3,reward-step-reward_0,reward-step-reward_1,reward-step-reward_2,reward-step-reward_3,scenario-left_team_difficulty,scenario-reset_occurred_after,scenario-right_team_difficulty,shots-first_team-all_shots,shots-first_team-all_shots_avg_dist_from_goal,shots-second_team-all_shots,shots-second_team-all_shots_avg_dist_from_goal',
    'files to consider')
flags.DEFINE_boolean('extract', False, 'extract tb data to csv (slow)')
flags.DEFINE_string('mode', 'raw', 'extract tb data to csv (slow)')

FLAGS = flags.FLAGS


def only_dirs(path):
    dirs = []
    for name in os.listdir(path):
        if os.path.isdir(os.path.join(path, name)):
            dirs.append(name)
    return dirs


def make_summary(in_path, out_path, depth, metric_files, tf_extract, mode):
    extracted_data_folder = 'env_tb_extracted'
    summary_dict = {}

    def add_metrics(specs, metrics_dir):
        for mf in metric_files:
            mfp = os.path.join(metrics_dir, mf)
            if not os.path.isfile(mfp):
                print("Warning metric {} not present".format(mfp))
                continue
            raw_df = pd.read_csv(mfp, sep=';', encoding='utf-8')
            #print(raw_df)
            result = raw_df['values']
            if mode == 'avg':
                col_names = list(map(str, range(len(specs)))) + ['avg_result']
                result = result.mean()  # so not all values are supported
                data = [specs + [result]]
                #print(col_names, ' fft ', data)
                df_with_specs = pd.DataFrame(data, columns=col_names)
            elif mode == 'raw':
                col_names = list(map(str, range(len(specs)))) + ['values']
                data = {}
                for sId, s in enumerate(specs):
                    data[str(sId)] = [s] * result.shape[0]

                data['values'] = result
                #print(data)
                df_with_specs = pd.DataFrame(data, columns=col_names)
                #print(df_with_specs)
            else:
                raise Exception('unsupported mode {}'.format(mode))
                    

            if mf in summary_dict:
                summary_dict[mf] = summary_dict[mf].append(df_with_specs)
            else:
                summary_dict[mf] = df_with_specs

    def dfs(cur_path, prev_dirs, cur_depth):
        if cur_depth == depth:
            metrics_dir = os.path.join(cur_path, extracted_data_folder)
            print(metrics_dir)
            os.makedirs(metrics_dir, exist_ok=True)
            if tf_extract:
                extract(os.path.join(cur_path, 'tf'), metrics_dir)
            if os.path.isdir(metrics_dir):
                add_metrics(prev_dirs, metrics_dir)
            else:
                print("WARNING dir {} does not exists (no tf metrics?)".format(
                    metrics_dir))

        else:
            dirs = only_dirs(cur_path)
            dirs.sort()
            for d in dirs:
                dfs(os.path.join(cur_path, d), prev_dirs + [d], cur_depth + 1)

    def write_summary():
        os.makedirs(out_path, exist_ok=True)
        for name, df in summary_dict.items():
            df.to_csv(
                os.path.join(out_path, name),
                sep=';',
                encoding='utf-8',
                index=False)

    dfs(in_path, [], 0)
    print(summary_dict)
    write_summary()


def main(argv):
    make_summary(FLAGS.in_path, FLAGS.out, FLAGS.level,
                 FLAGS.metric_files.split(','), FLAGS.extract, FLAGS.mode)


if __name__ == '__main__':
    app.run(main)
