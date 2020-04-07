from absl import app, flags, logging
from gfootball_zpp.eval.eval import evaluate_all, EvaluationStage, \
    ZppEvalPlayerData, BotEvalPlayerData
from time import time
import json

flags.DEFINE_list('filter_opponents', [],
                  help='list of opponents to skip during evaluation')

flags.DEFINE_string('name', None, 'Name of the player to evaluate')
flags.DEFINE_string('logdir', '', 'Place to save results')

FLAGS = flags.FLAGS

GAMES = 5

ZPP_OPPONENTS = {
    'random': ZppEvalPlayerData('random', policy='random'),
    'transfered0_sp': ZppEvalPlayerData('transfered0_sp', policy='multihead', sample='True', checkpoint='GS//scon/scon_e3_p1/1/ckpt/0/ckpt-505'),
    'transfered0': ZppEvalPlayerData('transfered0', policy='multihead', checkpoint='GS//scon/scon_e3_p1/1/ckpt/0/ckpt-505'),
    'from0to1to5': ZppEvalPlayerData('from0to1to5', policy='multihead', sample='True', checkpoint='GS//f5v01/f5v0to1to5_e2/1/ckpt/0/ckpt-252')
    'scon_e3_p2_hard_ns': ZppEvalPlayerData('scon_e3_p2_hard_ns', sample=False, policy='multihead', checkpoint='GS//scon/scon_e3_p2_hard/1/ckpt/0/ckpt-817'),
    'scon_e3_p2_hard_sp': ZppEvalPlayerData('scon_e3_p2_hard_sp', sample=True, policy='multihead', checkpoint='GS//scon/scon_e3_p2_hard/1/ckpt/0/ckpt-817'),
    'scon_e3_p2_nhm_hard_ns': ZppEvalPlayerData('scon_e3_p2_nhm_hard_ns', sample=False, policy='multihead', checkpoint='GS//scon/scon_e3_p2_nhm_hard/1/ckpt/0/ckpt-814'),
    'scon_e3_p2_nhm_hard_sp': ZppEvalPlayerData('scon_e3_p2_nhm_hard_sp', sample=True, policy='multihead', checkpoint='GS//scon/scon_e3_p2_nhm_hard/1/ckpt/0/ckpt-814'),
    'scon_e3_p3_hard_ns': ZppEvalPlayerData('scon_e3_p3_hard_ns', sample=False, policy='multihead', checkpoint='GS//scon/scon_e3_p3_hard/1/ckpt/0/ckpt-812'),
    'scon_e3_p3_hard_sp': ZppEvalPlayerData('scon_e3_p3_hard_sp', sample=True, policy='multihead', checkpoint='GS//scon/scon_e3_p3_hard/1/ckpt/0/ckpt-812')
}

ZPP_SCENARIOS = [
    '5_vs_5'
]

BOTS_STAGES = [
    EvaluationStage('5_vs_5', BotEvalPlayerData('easy_bots', 0.05), GAMES),
    EvaluationStage('5_vs_5_medium', BotEvalPlayerData(
        'medium_bots', 0.6), GAMES),
    EvaluationStage('5_vs_5_hard', BotEvalPlayerData('hard_bots', 0.95), GAMES)
]


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
