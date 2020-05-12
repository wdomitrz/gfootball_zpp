from absl import app
from absl import flags
from absl import logging
import json

flags.DEFINE_string('f', None, 'File to encode')

FLAGS = flags.FLAGS


REPLACE_TABLE = [
    (':', '¦'),
    (',', '᎖'),
    ('\n', ' '),
    ('"', '᭤'),
    ('\'', '᭥')
]


def encode_config(cfg):
    str_config = json.dumps(cfg)
    for (f, t) in REPLACE_TABLE:
        str_config = str_config.replace(f, t)
    return str_config


def decode_config(str_config):
    for (t, f) in REPLACE_TABLE:
        str_config = str_config.replace(f, t)
    logging.info('JSON: %s', str_config)
    cfg = json.loads(str_config)
    return cfg


def maybe_str(s):
    if isinstance(s, str):
        return s
    return str(s)

def encode_models_dirs(model_dirs_cfg):
    """
    The config for model_dirs accepts a list of dicts. Each dict should contain
    "path", "proba" and "config" keys. The "config" key will be parsed by the
    `encode_config` function.
    """
    each_model_parsed = [';'.join([maybe_str(model_dir_cfg["path"]),
                                   maybe_str(model_dir_cfg["proba"]),
                                   encode_config(model_dir_cfg["config"])])
                         for model_dir_cfg in model_dirs_cfg]
    return '*'.join(each_model_parsed)


def encode_config_extra_player(player_name, player_cfg):
    if "models_dirs" in player_cfg:
        player_cfg["model_dirs"] = encode_models_dirs(player_cfg["models_dirs"])
    return player_name + ':' + ','.join(maybe_str(name) + '=' + maybe_str(value) for name, value in player_cfg.items())


def encode_config_base(cfg):
    if "extra_players" in cfg:
        cfg["extra_players"] = [encode_config_extra_player(
            *player_name_and_cfg) for player_name_and_cfg in cfg["extra_players"].items()]
    return cfg


def main(argv):
    with open(FLAGS.f) as f:
        str_config = f.read()
        cfg = json.loads(str_config)
        print(encode_config_base(cfg))


if __name__ == '__main__':
    app.run(main)
