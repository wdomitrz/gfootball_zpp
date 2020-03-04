import collections
from gfootball.env import football_action_set
import gym
import tensorflow as tf
import numpy as np

AgentOutput = collections.namedtuple('AgentOutput',
                                     'action policy_logits baseline')


class ObservationStacker(object):
    """Utility class that produces stacked observations."""

    def __init__(self, stacking):
        self._stacking = stacking
        self._data = []

    def get(self, observation):
        if self._data:
            self._data.append(observation)
            self._data = self._data[-self._stacking:]
        else:
            self._data = [observation] * self._stacking
        return np.concatenate(self._data, axis=-1)

    def reset(self):
        self._data = []


class DummyEnv(object):
    # We need env object to pass to build_policy, however real environment
    # is not there yet.
    reward_range = None
    metadata = None
    spec = None

    def __init__(self, action_set, stacking, controlled_agents):
        self.action_space = gym.spaces.MultiDiscrete(
            [len(football_action_set.action_set_dict[action_set])] * controlled_agents)
        self.observation_space = gym.spaces.Box(
            0, 255, shape=[controlled_agents, 72, 96, 4 * stacking], dtype=np.uint8)
        self.players = controlled_agents


class PackedBitsObservation(gym.ObservationWrapper):
    """Wrapper that encodes a frame as packed bits instead of booleans.

    8x less to be transferred across the wire (16 booleans stored as uint16
    instead of 16 uint8) and 8x less to be transferred from CPU to TPU (16
    booleans stored as uint32 instead of 16 bfloat16).

    """

    def __init__(self, env):
        super(PackedBitsObservation, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=np.iinfo(np.uint16).max,
            shape=env.observation_space.shape[:-1] + \
                  ((env.observation_space.shape[-1] + 15) // 16,),
            dtype=np.uint16)

    def observation(self, observation):
        return packbits(observation)


def packbits(frame):
    data = np.packbits(frame, axis=-1)  # This packs to uint8
    # Now we want to pack pairs of uint8 into uint16's.
    # We first need to ensure that the last dimention has even size.
    if data.shape[-1] % 2 == 1:
        data = np.pad(data, [(0, 0)] * (data.ndim - 1) + [(0, 1)], 'constant')
    return data.view(np.uint16)


def unpackbits(frame):
    def _(frame):
        # Unpack each uint16 into 16 bits
        bit_patterns = [
            2 ** 7, 2 ** 6, 2 ** 5, 2 ** 4, 2 ** 3, 2 ** 2, 2 ** 1, 2 ** 0, 2 ** 15, 2 ** 14, 2 ** 13,
            2 ** 12, 2 ** 11, 2 ** 10, 2 ** 9, 2 ** 8
        ]
        frame = tf.bitwise.bitwise_and(frame[..., tf.newaxis], bit_patterns)
        frame = tf.cast(tf.cast(frame, tf.bool), tf.float32) * 255
        # Reshape to the right size.
        frame = tf.reshape(frame, frame.shape[:-2] + \
                           (frame.shape[-2] * frame.shape[-1],))
        return frame

    # if tf.test.is_gpu_available():
    # return tf.xla.experimental.compile(_, [frame])[0]
    return _(frame)