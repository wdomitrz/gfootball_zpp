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

GAMES = 4

ZPP_OPPONENTS = {
    'random': ZppEvalPlayerData('random', policy='random'),
    'transfered0_sp': ZppEvalPlayerData('transfered0_sp', policy='multihead', sample='True', checkpoint='GS//scon/scon_e3_p1/1/ckpt/0/ckpt-505'),
    'transfered0': ZppEvalPlayerData('transfered0', policy='multihead', checkpoint='GS//scon/scon_e3_p1/1/ckpt/0/ckpt-505'),
    'from0to1to5': ZppEvalPlayerData('from0to1to5', policy='multihead', sample='True', checkpoint='GS//f5v01/f5v0to1to5_e2/1/ckpt/0/ckpt-252'),
    'scon_e3_p2_hard_ns': ZppEvalPlayerData('scon_e3_p2_hard_ns', sample=False, policy='multihead', checkpoint='GS//scon/scon_e3_p2_hard/1/ckpt/0/ckpt-817'),
    'scon_e3_p2_hard_sp': ZppEvalPlayerData('scon_e3_p2_hard_sp', sample=True, policy='multihead', checkpoint='GS//scon/scon_e3_p2_hard/1/ckpt/0/ckpt-817'),
    'scon_e3_p2_nhm_hard_ns': ZppEvalPlayerData('scon_e3_p2_nhm_hard_ns', sample=False, policy='multihead', checkpoint='GS//scon/scon_e3_p2_nhm_hard/1/ckpt/0/ckpt-814'),
    'scon_e3_p2_nhm_hard_sp': ZppEvalPlayerData('scon_e3_p2_nhm_hard_sp', sample=True, policy='multihead', checkpoint='GS//scon/scon_e3_p2_nhm_hard/1/ckpt/0/ckpt-814'),
    'scon_e3_p3_hard_ns': ZppEvalPlayerData('scon_e3_p3_hard_ns', sample=False, policy='multihead', checkpoint='GS//scon/scon_e3_p3_hard/1/ckpt/0/ckpt-812'),
    'scon_e3_p3_hard_sp': ZppEvalPlayerData('scon_e3_p3_hard_sp', sample=True, policy='multihead', checkpoint='GS//scon/scon_e3_p3_hard/1/ckpt/0/ckpt-812'),
    'checkpoints_selfplay_e9_ns': ZppEvalPlayerData('checkpoints_selfplay_e9_ns', sample=False, policy='multihead', checkpoint='GS//zuzanna-seed/checkpoints_sp/checkpoints_selfplay_e9/1/ckpt/0/ckpt-274'),
    'checkpoints_selfplay_e9_sp': ZppEvalPlayerData('checkpoints_selfplay_e9_sp', sample=True, policy='multihead', checkpoint='GS//zuzanna-seed/checkpoints_sp/checkpoints_selfplay_e9/1/ckpt/0/ckpt-274'),
    'm_sp_bt_16_e2_p1': ZppEvalPlayerData('m_sp_bt_16_e2_p', policy='multihead', sample='True', checkpoint='!latest-GS//m_sp/m_sp_bt_16_e2_p1/1/ckpt/0/'),
    'm_sp_bt_64_e2_p1': ZppEvalPlayerData('m_sp_bt_64_e2_p1', policy='multihead', sample='True', checkpoint='!latest-GS//m_sp/m_sp_bt_64_e2_p1/1/ckpt/0/'),
}

NNM_OPPONENTS = {
    'm_sp_arch_longer_heads_bt_64_r100': NNMEvalPlayerData('m_sp_arch_longer_heads_bt_64_r100', models_dirs='GS//m_sp_arch/m_sp_arch_longer_heads_bt_64_r100/1/model/nn_manager/183;1.0;{᭤env_name᭤¦ ᭤5_vs_5_partially_randomized᭤᎖ ᭤enable_goal_videos᭤¦ false᎖ ᭤enable_full_episode_videos᭤¦ false᎖ ᭤logdir᭤¦ ᭤᭤᎖ ᭤enable_sides_swap᭤¦ false᎖ ᭤number_of_left_players_agent_controls᭤¦ 4᎖ ᭤number_of_right_players_agent_controls᭤¦ 0᎖ ᭤render᭤¦ false᎖ ᭤write_video᭤¦ false᎖ ᭤stacked_frames᭤¦ 4᎖ ᭤channel_dimensions᭤¦ [96᎖ 72]᎖ ᭤rewards᭤¦ ᭤scoring᎖checkpoints᭤᎖ ᭤dump_frequency᭤¦ 10᎖ ᭤wrappers᭤¦ ᭤checkpoint_score᎖obs_extract᎖single_agent᎖obs_stack᎖old_single_map᎖pack_bits᭤}'),
    
    'm_sp_arch_separate_baseline_r101': NNMEvalPlayerData('m_sp_arch_separate_baseline_r101', models_dirs='GS//m_sp_arch/m_sp_arch_separate_baseline_r101/1/model/nn_manager/181;1.0;{᭤env_name᭤¦ ᭤5_vs_5_partially_randomized᭤᎖ ᭤enable_goal_videos᭤¦ false᎖ ᭤enable_full_episode_videos᭤¦ false᎖ ᭤logdir᭤¦ ᭤᭤᎖ ᭤enable_sides_swap᭤¦ false᎖ ᭤number_of_left_players_agent_controls᭤¦ 4᎖ ᭤number_of_right_players_agent_controls᭤¦ 0᎖ ᭤render᭤¦ false᎖ ᭤write_video᭤¦ false᎖ ᭤stacked_frames᭤¦ 4᎖ ᭤channel_dimensions᭤¦ [96᎖ 72]᎖ ᭤rewards᭤¦ ᭤scoring᎖checkpoints᭤᎖ ᭤dump_frequency᭤¦ 10᎖ ᭤wrappers᭤¦ ᭤checkpoint_score᎖obs_extract᎖single_agent᎖obs_stack᎖old_single_map᎖pack_bits᭤}'),
    
    'm_sp_arch_smiple115_r100': NNMEvalPlayerData('m_sp_arch_smiple115_r100', models_dirs='GS//m_sp_arch/m_sp_arch_smiple115_r100/1/model/nn_manager/183;1.0;{᭤env_name᭤¦ ᭤5_vs_5_partially_randomized᭤᎖ ᭤enable_goal_videos᭤¦ false᎖ ᭤enable_full_episode_videos᭤¦ false᎖ ᭤logdir᭤¦ ᭤᭤᎖ ᭤enable_sides_swap᭤¦ false᎖ ᭤number_of_left_players_agent_controls᭤¦ 4᎖ ᭤number_of_right_players_agent_controls᭤¦ 0᎖ ᭤render᭤¦ false᎖ ᭤write_video᭤¦ false᎖ ᭤stacked_frames᭤¦ 4᎖ ᭤channel_dimensions᭤¦ [96᎖ 72]᎖ ᭤rewards᭤¦ ᭤scoring᎖checkpoints᭤᎖ ᭤dump_frequency᭤¦ 10᎖ ᭤wrappers᭤¦ ᭤checkpoint_score᎖simple115v2᎖simple115_pick_first᭤}'),
    
    
}


ABSTRACT_OPPONENTS = {
    'random': ZppEvalPlayerData('random', policy='random'),
    #'random_net': ZppEvalPlayerData('random_net', policy='multihead', sample='True'),
    'academy_curriculum_eb': ZppEvalPlayerData('academy_curriculum_eb', policy='multihead', sample='True', checkpoint='GS//academy_5vs5_transfer/academy_5vs5_transfer_eval_3/1/ckpt/0/ckpt-809'),
    #'opponents_curriculum_015': ZppEvalPlayerData('opponents_curriculum_015', policy='multihead', sample='True', checkpoint='GS//f5v01/f5v0to1to5_e2/1/ckpt/0/ckpt-252'),
    'opponents_curriculum_0125': ZppEvalPlayerData('opponents_curriculum_0125', policy='multihead', sample='True', checkpoint='GS//f5v01/f5v0to1to2t05_e2/1/ckpt/0/ckpt-291'),
    'checkpoints_selfplay': ZppEvalPlayerData('checkpoints_selfplay_e9_sp', sample=True, policy='multihead', checkpoint='GS//zuzanna-seed/checkpoints_sp/checkpoints_selfplay_e9/1/ckpt/0/ckpt-274'),
    'm_sp_in_bt_4_e2_p1': ZppEvalPlayerData('m_sp_in_bt_4_e2_p1', policy='multihead', sample='True', checkpoint='GS//m_sp/m_sp_in_bt_4_e2_p1/1/ckpt/0/ckpt-294')
}

ZPP_SCENARIOS = [
    '5_vs_5'
]

BOTS_STAGES = [
    EvaluationStage('5_vs_5', BotEvalPlayerData('easy_bots', 0.05), GAMES),
    EvaluationStage('5_vs_5_medium', BotEvalPlayerData('medium_bots', 0.6), GAMES),
    EvaluationStage('5_vs_5_hard', BotEvalPlayerData('hard_bots', 0.95), GAMES)
]

ABSTRACT_BOT_STAGES = [
    EvaluationStage('5_vs_5', BotEvalPlayerData('easy_bots', 0.05), GAMES),
    EvaluationStage('5_vs_5_medium', BotEvalPlayerData('medium_bots', 0.6), GAMES),
    EvaluationStage('5_vs_5_hard', BotEvalPlayerData('hard_bots', 0.95), GAMES),
    EvaluationStage('5_vs_0', BotEvalPlayerData('0_easy', 0.05), GAMES),
    EvaluationStage('5_vs_1', BotEvalPlayerData('1_easy', 0.05), GAMES),
    EvaluationStage('5_vs_2', BotEvalPlayerData('2_easy', 0.05), GAMES),
    EvaluationStage('5_vs_3', BotEvalPlayerData('3_easy', 0.05), GAMES),
    EvaluationStage('academy_5_vs_5_3v3_pass_and_shoot', BotEvalPlayerData('academy_5_vs_5_3v3_pass_and_shoot', 0.05), GAMES),
    EvaluationStage('academy_5_vs_5_4v0_with_keeper', BotEvalPlayerData('academy_5_vs_5_4v0_with_keeper', 0.05), GAMES),
    EvaluationStage('academy_5_vs_5_4v1', BotEvalPlayerData('academy_5_vs_5_4v1', 0.05), GAMES),
    EvaluationStage('academy_5_vs_5_4v2_random_0_70', BotEvalPlayerData('academy_5_vs_5_4v2_random_0_70', 0.05), GAMES),
    EvaluationStage('academy_5_vs_5_4v3_long_pass_and_shoot', BotEvalPlayerData('academy_5_vs_5_4v3_long_pass_and_shoot', 0.05), GAMES),
    EvaluationStage('academy_5_vs_5_4v3_random_0_65', BotEvalPlayerData('academy_5_vs_5_4v3_random_0_65', 0.05), GAMES),
    EvaluationStage('academy_5_vs_5_corner', BotEvalPlayerData('academy_5_vs_5_corner', 0.05), GAMES)
]

AGENT_OPPONENTS = {**ZPP_OPPONENTS, **NNM_OPPONENTS}

OPPONENTS = {**NNM_OPPONENTS, **ABSTRACT_OPPONENTS}

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
    for opponent in OPPONENTS:
        if opponent in filter:
            continue
        for scenario in ZPP_SCENARIOS:
            stages.append(EvaluationStage(
                scenario, OPPONENTS[opponent], GAMES))
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
    assert FLAGS.name in AGENT_OPPONENTS, "There is no such player definition!"

    player = AGENT_OPPONENTS[FLAGS.name]
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
