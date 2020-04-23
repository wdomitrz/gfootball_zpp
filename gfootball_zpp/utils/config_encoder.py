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


def main(argv):
    with open(FLAGS.f) as f:
        str_config = f.read()
        cfg = json.loads(str_config)
        print(encode_config(cfg))


if __name__ == '__main__':
  app.run(main)
