from gfootball_zpp.utils import gsutil
import tensorflow as tf
import re

checkpoint_number_re = re.compile('^.*-(?P<num>\d+).index$')


def get_num(filename):
    return int(re.sub(checkpoint_number_re, r'\g<num>', filename))


def get_latest_checkpoint(checkpoint_directory):
    try:
        checkpoints = gsutil.ls(checkpoint_directory)
    except tf.errors.NotFoundError:
        print("No such directory:", checkpoint_directory)
        print('Note that at the beginning of training it is normal.')
        return None
    checkpoints = sorted([(get_num(f), f[:-6])
                          for f in checkpoints if f[-6:] == '.index'])
    if len(checkpoints):
        return checkpoint_directory + checkpoints[-1][1]
    else:
        return None
