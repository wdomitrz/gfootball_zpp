import tensorflow as tf


def cp(source, destination, overwrite=False):
    filename = destination + '_'.join(source.split('/')[2:])
    tf.io.gfile.copy(source, filename, overwrite=overwrite)
    return filename


def ls(path):
    return tf.io.gfile.listdir(path)


def cp_ckpt(path, out_dir='/tmp/', overwrite=True):
    # todo: parametrize number of checkpoint data parts
    for ext in ['.index', '.data-00000-of-00002', '.data-00001-of-00002']:
        filename = cp(path + ext, out_dir, overwrite)
    return '.'.join(filename.split('.')[:-1])

