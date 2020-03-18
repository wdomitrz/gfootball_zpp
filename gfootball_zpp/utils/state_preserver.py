import gym
import os
import json
import tensorflow as tf


class StatePreserver(gym.Wrapper):
    class StatePreserverAPI():
        def __init__(self, preserved_state):
            self._preserved_state = preserved_state

        def create_variable(self, name, default_value):
            if name not in self._preserved_state:
                self._preserved_state[name] = default_value
            return self._preserved_state[name]

        def update_variable(self, name, value):
            assert name in self._preserved_state
            self._preserved_state[name] = value
            return self._preserved_state[name]

    def __init__(self, env, config):
        gym.Wrapper.__init__(self, env)
        self._state_file_path = os.path.join(config['base_logdir'],
                                             'env_state_file.json')
        self._state_dump_freq = int(config['dump_frequency'])
        self._resets_after_last_dump = -1

        self._preserved_state = {}
        if tf.io.gfile.exists(self._state_file_path):
            with tf.io.gfile.GFile(self._state_file_path, mode='r') as f:
                json_data = f.read(n=-1)
            self._preserved_state = json.loads(json_data)

        config['state_preserver'] = StatePreserver.StatePreserverAPI(
            self._preserved_state)

    def __getattr__(self, attr):
        return getattr(self.env, attr)

    def reset(self):
        observation = super(StatePreserver, self).reset()
        self._resets_after_last_dump += 1
        if self._resets_after_last_dump % self._state_dump_freq == 0:
            json_data = json.dumps(self._preserved_state)
            if tf.io.gfile.exists(self._state_file_path):
                tf.io.gfile.remove(self._state_file_path)
            with tf.io.gfile.GFile(self._state_file_path, mode='w') as f:
                f.write(json_data)
            self._resets_after_last_dump = 0
        return observation
