from absl import app, flags, logging
from gfootball_zpp.eval.eval import evaluate_all, EvaluationStage, \
    ZppEvalPlayerData, BotEvalPlayerData, NNMEvalPlayerData
from time import time
import json

flags.DEFINE_list('filter_opponents', [],
                  help='list of opponents to skip during evaluation')

flags.DEFINE_string('name', None, 'Name of the player to evaluate')
flags.DEFINE_string('logdir', '', 'Place to save results')

FLAGS = flags.FLAGS

GAMES = 2

SELF_PLAY_ZPP_OPPONENTS = {
    'checkpoints_sp_c_e2': ZppEvalPlayerData('checkpoints_sp_c_e2', sample=True, policy='multihead', checkpoint='GS//zuzanna-seed/checkpoints_sp/checkpoints_sp_c_e2/1/ckpt/0/ckpt-617'),
    'm_fsp_scon_e82_24': ZppEvalPlayerData('m_fsp_scon_e82_24', sample=True, policy='multihead', checkpoint='GS//m_fsp/m_fsp_scon_e82_24/1/ckpt/0/ckpt-1138'),
    'checkpoints_sp_c_nb_e2': ZppEvalPlayerData('checkpoints_sp_c_nb_e2', sample=True, policy='multihead', checkpoint='GS//zuzanna-seed/checkpoints_sp/checkpoints_sp_c_nb_e2/1/ckpt/0/ckpt-632'),
    'f0to1to5tosp_e5': ZppEvalPlayerData('f0to1to5tosp_e5', sample=True, policy='multihead', checkpoint='GS//zuzanna-seed/checkpoints_sp/f0to1to5tosp_e5/1/ckpt/0/ckpt-586'),
    'm_fsp_e71_64': ZppEvalPlayerData('m_fsp_e71_64', sample=True, policy='multihead', checkpoint='GS//m_fsp/m_fsp_e71_64/1/ckpt/0/ckpt-440'),
    'm_sp_in_ob_32_e61_p1': ZppEvalPlayerData('m_sp_in_ob_32_e61_p1', sample=True, policy='multihead', checkpoint='GS//m_sp/m_sp_in_ob_32_e61_p1/1/ckpt/0/ckpt-439'),
    'm_sp_bt_32_e26_p1': ZppEvalPlayerData('m_sp_bt_32_e26_p1', sample=True, policy='multihead', checkpoint='GS//m_sp/m_sp_bt_32_e26_p1/1/ckpt/0/ckpt-334'),
    'checkpoints_selfplay_e9': ZppEvalPlayerData('checkpoints_selfplay_e9', sample=True, policy='multihead', checkpoint='GS//zuzanna-seed/checkpoints_sp/checkpoints_selfplay_e9/1/ckpt/0/ckpt-335'),
}


NNM_OPPONENTS = {
    'm_sp_arch_longer_heads_bt_64_r100': NNMEvalPlayerData('m_sp_arch_longer_heads_bt_64_r100', models_dirs='GS//m_sp_arch/m_sp_arch_longer_heads_bt_64_r100/1/model/nn_manager/183;1.0;{᭤env_name᭤¦ ᭤5_vs_5_partially_randomized᭤᎖ ᭤enable_goal_videos᭤¦ false᎖ ᭤enable_full_episode_videos᭤¦ false᎖ ᭤logdir᭤¦ ᭤᭤᎖ ᭤enable_sides_swap᭤¦ false᎖ ᭤number_of_left_players_agent_controls᭤¦ 4᎖ ᭤number_of_right_players_agent_controls᭤¦ 0᎖ ᭤render᭤¦ false᎖ ᭤write_video᭤¦ false᎖ ᭤stacked_frames᭤¦ 4᎖ ᭤channel_dimensions᭤¦ [96᎖ 72]᎖ ᭤rewards᭤¦ ᭤scoring᎖checkpoints᭤᎖ ᭤dump_frequency᭤¦ 10᎖ ᭤wrappers᭤¦ ᭤checkpoint_score᎖obs_extract᎖single_agent᎖obs_stack᎖old_single_map᎖pack_bits᭤}'),

    'm_sp_arch_longer_heads_bt_32_r201_m335': NNMEvalPlayerData('m_sp_arch_longer_heads_bt_32_r201_m335', models_dirs='GS//m_sp_arch/m_sp_arch_longer_heads_bt_32_r201/1/model/nn_manager/335;1.0;{᭤env_name᭤¦ ᭤5_vs_5_partially_randomized᭤᎖ ᭤enable_goal_videos᭤¦ false᎖ ᭤enable_full_episode_videos᭤¦ false᎖ ᭤logdir᭤¦ ᭤᭤᎖ ᭤enable_sides_swap᭤¦ false᎖ ᭤number_of_left_players_agent_controls᭤¦ 4᎖ ᭤number_of_right_players_agent_controls᭤¦ 0᎖ ᭤render᭤¦ false᎖ ᭤write_video᭤¦ false᎖ ᭤stacked_frames᭤¦ 4᎖ ᭤channel_dimensions᭤¦ [96᎖ 72]᎖ ᭤rewards᭤¦ ᭤scoring᎖checkpoints᭤᎖ ᭤dump_frequency᭤¦ 10᎖ ᭤wrappers᭤¦ ᭤checkpoint_score᎖obs_extract᎖single_agent᎖obs_stack᎖old_single_map᎖pack_bits᭤}'),

    'm_sp_arch_medium_heads_bt_32_r201_m335': NNMEvalPlayerData('m_sp_arch_medium_heads_bt_32_r201_m335', models_dirs='GS//m_sp_arch/m_sp_arch_medium_heads_bt_32_r201/1/model/nn_manager/335;1.0;{᭤env_name᭤¦ ᭤5_vs_5_partially_randomized᭤᎖ ᭤enable_goal_videos᭤¦ false᎖ ᭤enable_full_episode_videos᭤¦ false᎖ ᭤logdir᭤¦ ᭤᭤᎖ ᭤enable_sides_swap᭤¦ false᎖ ᭤number_of_left_players_agent_controls᭤¦ 4᎖ ᭤number_of_right_players_agent_controls᭤¦ 0᎖ ᭤render᭤¦ false᎖ ᭤write_video᭤¦ false᎖ ᭤stacked_frames᭤¦ 4᎖ ᭤channel_dimensions᭤¦ [96᎖ 72]᎖ ᭤rewards᭤¦ ᭤scoring᎖checkpoints᭤᎖ ᭤dump_frequency᭤¦ 10᎖ ᭤wrappers᭤¦ ᭤checkpoint_score᎖obs_extract᎖single_agent᎖obs_stack᎖old_single_map᎖pack_bits᭤}'),
    
    'm_sp_arch_smiple115_r100': NNMEvalPlayerData('m_sp_arch_smiple115_r100', models_dirs='GS//m_sp_arch/m_sp_arch_smiple115_r100/1/model/nn_manager/183;1.0;{᭤env_name᭤¦ ᭤5_vs_5_partially_randomized᭤᎖ ᭤enable_goal_videos᭤¦ false᎖ ᭤enable_full_episode_videos᭤¦ false᎖ ᭤logdir᭤¦ ᭤᭤᎖ ᭤enable_sides_swap᭤¦ false᎖ ᭤number_of_left_players_agent_controls᭤¦ 4᎖ ᭤number_of_right_players_agent_controls᭤¦ 0᎖ ᭤render᭤¦ false᎖ ᭤write_video᭤¦ false᎖ ᭤stacked_frames᭤¦ 4᎖ ᭤channel_dimensions᭤¦ [96᎖ 72]᎖ ᭤rewards᭤¦ ᭤scoring᎖checkpoints᭤᎖ ᭤dump_frequency᭤¦ 10᎖ ᭤wrappers᭤¦ ᭤checkpoint_score᎖simple115v2᎖simple115_pick_first᭤}'),

    'm_sp_arch_smiple115_bt32_r201_m204': NNMEvalPlayerData('m_sp_arch_smiple115_bt32_r201_m204', models_dirs='GS//m_sp_arch/m_sp_arch_smiple115_bt32_r201/1/model/nn_manager/204;1.0;{᭤env_name᭤¦ ᭤5_vs_5_partially_randomized᭤᎖ ᭤enable_goal_videos᭤¦ false᎖ ᭤enable_full_episode_videos᭤¦ false᎖ ᭤logdir᭤¦ ᭤᭤᎖ ᭤enable_sides_swap᭤¦ false᎖ ᭤number_of_left_players_agent_controls᭤¦ 4᎖ ᭤number_of_right_players_agent_controls᭤¦ 0᎖ ᭤render᭤¦ false᎖ ᭤write_video᭤¦ false᎖ ᭤stacked_frames᭤¦ 4᎖ ᭤channel_dimensions᭤¦ [96᎖ 72]᎖ ᭤rewards᭤¦ ᭤scoring᎖checkpoints᭤᎖ ᭤dump_frequency᭤¦ 10᎖ ᭤wrappers᭤¦ ᭤checkpoint_score᎖simple115v2᎖simple115_pick_first᭤}'),
    
    'm_sp_arch_4x1_bt_32_r201_m335':  NNMEvalPlayerData('m_sp_arch_4x1_bt_32_r201_m335', models_dirs='GS//m_sp_arch/m_sp_arch_4x1_bt_32_r201/1/model/nn_manager/335;1.0;{᭤env_name᭤¦ ᭤5_vs_5_partially_randomized᭤᎖ ᭤enable_goal_videos᭤¦ false᎖ ᭤enable_full_episode_videos᭤¦ false᎖ ᭤logdir᭤¦ ᭤᭤᎖ ᭤enable_sides_swap᭤¦ false᎖ ᭤number_of_left_players_agent_controls᭤¦ 4᎖ ᭤number_of_right_players_agent_controls᭤¦ 0᎖ ᭤render᭤¦ false᎖ ᭤write_video᭤¦ false᎖ ᭤stacked_frames᭤¦ 4᎖ ᭤channel_dimensions᭤¦ [96᎖ 72]᎖ ᭤rewards᭤¦ ᭤scoring᎖checkpoints᭤᎖ ᭤dump_frequency᭤¦ 10᎖ ᭤wrappers᭤¦ ᭤checkpoint_score᎖obs_extract᎖single_agent᎖obs_stack᎖pack_bits᭤}')
}


SELF_PLAY_BOTS_STAGES = [
    EvaluationStage('5_vs_5', BotEvalPlayerData('easy_bots', 0.05), GAMES * 2),
    EvaluationStage('5_vs_5_medium', BotEvalPlayerData(
        'medium_bots', 0.6), GAMES * 2),
    EvaluationStage('5_vs_5_hard', BotEvalPlayerData(
        'hard_bots', 0.95), GAMES * 2),
]

ZPP_OPPONENTS = {**SELF_PLAY_ZPP_OPPONENTS, **NNM_OPPONENTS}


ZPP_SCENARIOS = [
    '5_vs_5'
]

BOTS_STAGES = SELF_PLAY_BOTS_STAGES

ENV_ARGS = {
    'write_goal_dumps': False,
    'write_full_episode_dumps': True,
    'write_video': True,
    'render': False,
    'dump_frequency': 1,
    'number_of_left_players_agent_controls': 4,
    # the representation should stay as raw since this is how
    # the env handle players internally
    'representation': 'raw'
}


def build_stages(filter=None):
    if not filter:
        filter = []
    stages = BOTS_STAGES + []
    for opponent in ZPP_OPPONENTS:
        if opponent in filter:
            continue
        for scenario in ZPP_SCENARIOS:
            stages.append(EvaluationStage(
                scenario, ZPP_OPPONENTS[opponent], GAMES))
    return stages


def build_summary(player, results):
    return [{
        'scenario': r.stage.scenario,
        'left_team': player.write_summary(),
        'right_team': r.stage.opponent.write_summary(),
        'scores': r.scores,
        'dump_files': r.logdir

    } for r in results]


def main(args):
    logging.info("Starting evaluation of %s.", FLAGS.name)
    assert FLAGS.name in ZPP_OPPONENTS, "There is no such player definition!"

    player = ZPP_OPPONENTS[FLAGS.name]
    opponents_to_filer = FLAGS.filter_opponents
    # I guess we want also selfplay results
    # opponents_to_filer.append(FLAGS.name)
    stages = build_stages(opponents_to_filer)

    logging.info("Prepared %d stages.", len(stages))
    logging.debug(stages)
    logging.info("Starting the evaluations.")
    logging.info("All logs should be available in `%s`", FLAGS.logdir)

    results = evaluate_all(player, stages, ENV_ARGS, FLAGS.logdir)

    logging.info("Evaluation finished. Collected %d results.", len(results))
    logging.debug(results)

    summary = build_summary(player, results)

    logging.info("Prepared summary.")
    logging.debug(summary)

    save_path = FLAGS.logdir + '/eval_results_' + \
        player.name + '_' + str(time()) + '.json'
    logging.info("Saving summary to `%s`", save_path)
    try:
        with open(save_path, 'w') as f:
            f.write(json.dumps(summary))
    except:
        logging.error("Could not save the summary!")
        logging.error(summary)
        raise
    logging.info("Evaluation finished successfully.")


if __name__ == '__main__':
    app.run(main)
