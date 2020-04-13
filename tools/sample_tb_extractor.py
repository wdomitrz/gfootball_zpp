from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from absl import flags, app
import tensorflow as tf
flags.DEFINE_string('tb_path', None, 'Path to tensorboard file')

FLAGS = flags.FLAGS


def main(argv):
    tf_size_guidance = {
        'compressedHistograms': 10,
        'images': 10,
        'scalars': 100,
        'histograms': 10
    }

    event_acc = EventAccumulator(FLAGS.tb_path, tf_size_guidance)
    event_acc.Reload()

    print('Value names:', event_acc.Tags())
    tensor_event = event_acc.Tensors('reward/game/reward_0')[0]
    print(tf.make_ndarray(tensor_event.tensor_proto))
    tensor_event = event_acc.Tensors('warning_strongly_inaccurate_actions/proportions_player_0')[0]
    print(tf.make_ndarray(tensor_event.tensor_proto))
    print(event_acc.Tensors('reward/game/difficulty_reward'))
    #print(event_acc.Tensors('reward/game/reward_0'))
    #print(event_acc.Tensors('warning_strongly_inaccurate_actions/proportions_player_0'))



if __name__ == '__main__':
    app.run(main)

