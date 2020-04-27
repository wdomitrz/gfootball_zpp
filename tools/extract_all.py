from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorflow.python.summary.summary_iterator import summary_iterator
from absl import flags, app
import pandas as pd
import tensorflow as tf
import os
flags.DEFINE_string('tb_path', None, 'Path to tensorboard file')
flags.DEFINE_string('out_path', None, 'Path to output dir')

FLAGS = flags.FLAGS


def main(argv):
    os.makedirs(FLAGS.out_path, exist_ok=True)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    tf_size_guidance = {
        'compressedHistograms': 10,
        'images': 10,
        'scalars': 100,
        'histograms': 10,
        'tensors': 100000000
    }

    event_acc = EventAccumulator(FLAGS.tb_path, tf_size_guidance)
    event_acc.Reload()

    
    print('Value names:', event_acc.Tags())
    data_names = event_acc.Tags()['tensors']

    for data_n in data_names:
        print(data_n)
        tensor_event_list = event_acc.Tensors(data_n)

        data = []
        for tensor_event in tensor_event_list:
            wall_time = tensor_event.wall_time
            step = tensor_event.step
            value = tf.make_ndarray(tensor_event.tensor_proto)
            data.append([wall_time, step, value])
        names = ['wall_time', 'step', 'values']
        df = pd.DataFrame(data, columns=names)
        print("$$$$")
        print(df)
        print("$$$$")
        file_name = data_n.replace('/', '-')
        df.to_csv(os.path.join(FLAGS.out_path, file_name), sep=';', encoding='utf-8')



if __name__ == '__main__':
    app.run(main)

