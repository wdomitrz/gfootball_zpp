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

GAMES = 10

ZPP_OPPONENTS = {
    'random': ZppEvalPlayerData('random', policy='random')
}

ZPP_SCENARIOS = [
    '5_vs_5'
]

BOTS_STAGES = [
    EvaluationStage('5_vs_5', BotEvalPlayerData('easy_bots', 0.05), GAMES)
]


ENV_ARGS = {
    'write_goal_dumps': False,
    'write_full_episode_dumps': True,
    'write_video': True,
    'render': True,
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
            stages.append(EvaluationStage(scenario, opponent, GAMES))
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
    opponents_to_filer.append(FLAGS.name)
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

    save_path = FLAGS.logdir + '/eval_results_' + player.name + '_' + time() + '.json'
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
