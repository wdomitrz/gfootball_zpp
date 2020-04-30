from absl import app, flags
import absl.logging as log
from os import listdir, path
import pandas as pd
import json

flags.DEFINE_string('jsons_dir', None, 'Path to local dir containing all evaluation results')
flags.DEFINE_string('output', None, 'output csv file')
FLAGS = flags.FLAGS

def load_dir(jsons_dir):
    data = []
    for filename in listdir(jsons_dir):
        if filename[-5:] != '.json':
            continue
        log.info("Loading:\t" + filename)
        with open(path.join(jsons_dir, filename)) as f:
            data += json.load(f)
    return data


def get_dataframe_from_json(data):
    matches = [{'left_team': entry['left_team']['name'],
                'right_team': entry['right_team']['name'],
                'left_scores': result['left'],
                'right_scores': result['right'],
                'left_minus_right_scores': result['left'] - result['right'],
                'scenario': entry['scenario']}
               for entry in data for result in entry['scores']]
    return pd.DataFrame(matches)

def jsons_dir_to_csv_file(jsons_dir, csv_file):
    data = load_dir(jsons_dir)
    data = get_dataframe_from_json(data)
    data.to_csv(csv_file)

def main(args):
    jsons_dir_to_csv_file(FLAGS.jsons_dir, FLAGS.csv_file)

if __name__ == '__main__':
    app.run(main)
