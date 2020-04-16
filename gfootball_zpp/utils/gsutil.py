import os
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


def cp_dir(remote_path, local_path):
    remote_path = os.path.join(remote_path, "") # add / if not present
    remote_prefix_len = len(remote_path)
    def get_local_path(remote_file_path):
        rel_file_path = remote_file_path[remote_prefix_len:]
        return os.path.join(local_path, rel_file_path)

    walk = tf.io.gfile.walk(remote_path)

    for (path, dirs, files) in walk:
        for f in files:
            remote_file_path = os.path.join(path, f)
            local_file_path = get_local_path(remote_file_path)
            tf.io.gfile.copy(remote_file_path, local_file_path)
        for d in dirs:
            remote_dir_path = os.path.join(path, d)
            local_dir_path = get_local_path(remote_dir_path)
            tf.io.gfile.makedirs(local_dir_path)
