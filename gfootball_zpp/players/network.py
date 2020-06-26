# coding=utf-8
# Copyright 2019 The SEED Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This is a modified version of the original files

"""SEED agent using Keras."""

import collections
import tensorflow as tf

def batch_apply(fn, inputs):
  """Folds time into the batch dimension, runs fn() and unfolds the result.
  Args:
    fn: Function that takes as input the n tensors of the tf.nest structure,
      with shape [time*batch, <remaining shape>], and returns a tf.nest
      structure of batched tensors.
    inputs: tf.nest structure of n [time, batch, <remaining shape>] tensors.
  Returns:
    tf.nest structure of [time, batch, <fn output shape>]. Structure is
    determined by the output of fn.
  """
  time_to_batch_fn = lambda t: tf.reshape(t, [-1] + t.shape[2:].as_list())
  batched = tf.nest.map_structure(time_to_batch_fn, inputs)
  output = fn(*batched)
  prefix = [int(tf.nest.flatten(inputs)[0].shape[0]), -1]
  batch_to_time_fn = lambda t: tf.reshape(t, prefix + t.shape[1:].as_list())
  return tf.nest.map_structure(batch_to_time_fn, output)




AgentOutput = collections.namedtuple('AgentOutput',
                                     'action policy_logits baseline')


class _Stack(tf.Module):
  """Stack of pooling and convolutional blocks with residual connections."""

  def __init__(self, num_ch, num_blocks):

    super(_Stack, self).__init__(name='stack')
    self._conv = tf.keras.layers.Conv2D(num_ch, 3, strides=1, padding='same',
                                        kernel_initializer='lecun_normal')
    self._max_pool = tf.keras.layers.MaxPool2D(
        pool_size=3, padding='same', strides=2)

    self._res_convs0 = [
        tf.keras.layers.Conv2D(
            num_ch, 3, strides=1, padding='same', name='res_%d/conv2d_0' % i,
            kernel_initializer='lecun_normal')
        for i in range(num_blocks)
    ]
    self._res_convs1 = [
        tf.keras.layers.Conv2D(
            num_ch, 3, strides=1, padding='same', name='res_%d/conv2d_1' % i,
            kernel_initializer='lecun_normal')
        for i in range(num_blocks)
    ]

  def __call__(self, conv_out):
    # Downscale.
    conv_out = self._conv(conv_out)
    conv_out = self._max_pool(conv_out)

    # Residual block(s).
    for (res_conv0, res_conv1) in zip(self._res_convs0, self._res_convs1):
      block_input = conv_out
      conv_out = tf.nn.relu(conv_out)
      conv_out = res_conv0(conv_out)
      conv_out = tf.nn.relu(conv_out)
      conv_out = res_conv1(conv_out)
      conv_out += block_input

    return conv_out


def make_logits(layer_fn, action_specs):
  return [layer_fn(n, 'policy_logits') for n in action_specs]


def apply_net(action_specs, policy_logits, core_output):
  n_actions = len(action_specs)
  arr = [policy_logits[i](core_output) for i in range(n_actions)]
  arr = tf.stack(arr)
  arr = tf.transpose(arr, perm=[1, 0, 2])
  return arr


def choose_action(action_specs, policy_logits, sample=True):
  n_actions = len(action_specs)
  policy_logits = tf.transpose(policy_logits, perm=[1, 0, 2])

  if not sample:
    new_action = tf.stack([
      tf.math.argmax(
        policy_logits[i], -1, output_type=tf.int32) for i in range(n_actions)])
  else:
    new_action = tf.stack([tf.squeeze(
      tf.random.categorical(
        policy_logits[i], 1, dtype=tf.int32), 1) for i in range(n_actions)])

  new_action = tf.transpose(new_action, perm=[1, 0])
  return new_action


class GFootball(tf.Module):
  """Agent with ResNet, but without LSTM and additional inputs.

  Four blocks instead of three in ImpalaAtariDeep.
  """

  def __init__(self, action_specs):
    super(GFootball, self).__init__(name='gfootball')

    self._config = {'sample_actions': True}

    # Parameters and layers for unroll.


    self._action_specs = action_specs

    # Parameters and layers for _torso.
    self._stacks = [
        _Stack(num_ch, num_blocks)
        for num_ch, num_blocks in [(16, 2), (32, 2), (32, 2), (32, 2)]
    ]
    self._conv_to_linear = tf.keras.layers.Dense(
        256, kernel_initializer='lecun_normal')

    # Layers for _head.
    self._policy_logits = make_logits(
          lambda num_units, name: tf.keras.layers.Dense(
            num_units,
            name=name,
            kernel_initializer='lecun_normal'),
          self._action_specs)

    self._baseline = tf.keras.layers.Dense(
        1, name='baseline', kernel_initializer='lecun_normal')

  def initial_state(self, batch_size):
    return ()

  def change_config(self, new_config):
    self._config = new_config

  def _torso(self, unused_prev_action, env_output):
    _, _, frame = env_output

    #frame = observation.unpackbits(frame)
    frame /= 255

    conv_out = frame
    for stack in self._stacks:
      conv_out = stack(conv_out)

    conv_out = tf.nn.relu(conv_out)
    conv_out = tf.keras.layers.Flatten()(conv_out)

    conv_out = self._conv_to_linear(conv_out)
    return tf.nn.relu(conv_out)

  def _head(self, core_output):

    policy_logits = apply_net(
          self._action_specs,
          self._policy_logits,
          core_output)
    baseline = tf.squeeze(self._baseline(core_output), axis=-1)

    # Sample an action from the policy.
    new_action = choose_action(self._action_specs, policy_logits, self._config['sample_actions'])

    return AgentOutput(new_action, policy_logits, baseline)

  def __call__(self, input_, core_state):
    prev_actions, env_outputs = input_
    outputs, core_state = self._unroll(prev_actions, env_outputs, core_state)
    return outputs, core_state

  def _unroll(self, prev_actions, env_outputs, core_state):
    torso_outputs = batch_apply(self._torso, (prev_actions, env_outputs))
    return batch_apply(self._head, (torso_outputs,)), core_state
