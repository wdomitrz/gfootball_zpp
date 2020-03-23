from absl import app, flags
from os import listdir, path
import json

flags.DEFINE_string('jsons_dir', None, 'Path to local dir containing all evaluation results')
flags.DEFINE_string('output', None, 'output file')
FLAGS = flags.FLAGS


def build_per_team_statistics(data):
    players = {}
    for item in data:
        for (player, opponent) in [('left', 'right'), ('right', 'left')]:
            if not item[player + '_team']['name'] in players:
                players[item[player + '_team']['name']] = {
                    'games': 0,
                    'wins': 0,
                    'loses': 0,
                    'scores': 0,
                    'lost_scores': 0,
                    'opponents': {}
                }
            p = players[item[player + '_team']['name']]
            op_name = item['scenario'] + '_' + item[opponent + '_team']['name']
            if not op_name in p['opponents']:
                p['opponents'][op_name] = {
                    'games': 0,
                    'wins': 0,
                    'loses': 0,
                    'scores': 0,
                    'lost_scores': 0,
                    'results': [],
                    'logdirs': []
                }

            games = len(item['scores'])
            scores = sum([s[player] for s in item['scores']])
            lost_scores = sum([s[opponent] for s in item['scores']])
            wins = len([1 for s in item['scores'] if s[player] > s[opponent]])
            loses = len([1 for s in item['scores'] if s[player] < s[opponent]])

            for stats in [p, p['opponents'][op_name]]:
                stats['games'] += games
                stats['scores'] += scores
                stats['lost_scores'] += lost_scores
                stats['wins'] += wins
                stats['loses'] += loses
            p['opponents'][op_name]['logdirs'].append(item['dump_files'])
            p['opponents'][op_name]['results'].append([(s[player], s[opponent]) for s in item['scores']])
    return players


def build_scores_table(results):
    res = list(results.items())
    res.sort(key=lambda x: (x[1]['wins'] / x[1]['games'],
                            -x[1]['loses'] / x[1]['games'],
                            x[1]['scores'] - x[1]['lost_scores']))
    table = [
        [
            name,
            data['wins'] / data['games'],
            data['loses'] / data['games'],
            data['scores'] / data['lost_scores'] if data['lost_scores'] > 0 else 100
        ]
        for name, data in res
    ]
    return html_table(table, ['name', 'won', 'lost', 'score ratio'])


def build_team_scores_table(results, player):
    res = sorted(results[player]['opponents'].items())

    table = [
        [
            name,
            data['wins'] / data['games'],
            data['loses'] / data['games'],
            data['scores'] / data['lost_scores'] if data['lost_scores'] > 0 else 100,
            ';'.join(data['logdirs'])
        ]
        for name, data in res
    ]
    return html_table(table, ['name', 'won', 'lost', 'score ratio', 'logdirs'])


def html_table(table, header):
    t = [['<a href="#' + row[0] + '">' + row[0] + '</a>'] + row[1:] for row in table]
    t = '\n'.join(
        ['\t<tr>\n' +
         '\n'.join(map(lambda content: '\t\t<td> ' + str(content) + '</td>', row)) +
         '\n\t</tr>'
         for row in t])

    h = '\t<tr>\n' + '\n'.join(['\t\t<th>' + c + '</th>' for c in header]) + '\n\t</tr>'

    return '<table class="table sortable table-striped table-bordered table-sm">\n<thead>\n' + h + '\n</thead>\n<tbody>\n' + t + '\n</tbody>\n</table>'


def render_body(results):
    elems = ['<h1>Evaluation results</h1>',
             '<h2>Overview</h2>',
             build_scores_table(results),
             '<h2>Per player results</h2>']
    for name in sorted(results.keys()):
        elems.append('<h3 id="' + name + '">' + name + '</h3>')
        elems.append(build_team_scores_table(results, name))

    return '\n\n'.join(elems)


def render_site(results, save_path):
    with open(path.join(path.dirname(path.realpath(__file__)), 'results_template.html'), 'r') as f:
        lines = f.readlines()
    i = lines.index('!!content_here!!\n')
    lines[i] = render_body(results)
    with open(save_path, 'w') as f:
        f.write(''.join(lines))


def main(args):
    data = []
    for filename in listdir(FLAGS.jsons_dir):
        if filename[-5:] != '.json':
            continue
        with open(path.join(FLAGS.jsons_dir, filename)) as f:
            data += json.load(f)
    parsed = build_per_team_statistics(data)
    render_site(parsed, FLAGS.output)


if __name__ == '__main__':
    app.run(main)
