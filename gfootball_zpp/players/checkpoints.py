from gfootball_zpp.utils import gsutil
import tensorflow as tf
import re
from numpy import random

from absl import logging

checkpoint_number_re = re.compile('^.*-(?P<num>\d+).index$')


def get_num(filename):
    return int(re.sub(checkpoint_number_re, r'\g<num>', filename))


def get_checkpoint(checkpoint_directory, selection_fn):
    if checkpoint_directory[-1] != '/':
        checkpoint_directory += '/'
    try:
        checkpoints = gsutil.ls(checkpoint_directory)
    except tf.errors.NotFoundError:
        logging.warning("No such checkpoints directory: %s", checkpoint_directory)
        logging.warning('Note that at the beginning of training it might be normal.')
        return None
    checkpoints = sorted([(get_num(f), f[:-6])
                          for f in checkpoints if f[-6:] == '.index'])
    if 0 == len(checkpoints):
        return None
    return checkpoint_directory + selection_fn(checkpoints)[1]


def select_random(checkpoints):
    return checkpoints[random.choice(len(checkpoints))]


def select_latest(checkpoints):
    return checkpoints[-1]


def select_mostly_latest(checkpoints):
    sample = max(0, round((1 - abs(random.normal())) * (len(checkpoints) - 1)))
    return checkpoints[sample]
