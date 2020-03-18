import gym


class EnvUsageStatsTracker(gym.Wrapper):
    class EnvUsageStatsAPI():
        def __init__(self, tracker):
            self._tracker = tracker

        @property
        def env_resets(self):
            return self._tracker._env_resets

        @property
        def env_episode_steps(self):
            return self._tracker._env_episode_steps

        @property
        def env_total_steps(self):
            return self._tracker._env_total_steps

    def _get_store_name(self, name):
        return 'env_usage_stats_' + name

    def _get_preserved_variable_names(self):
        return ['_env_resets', '_env_total_steps']

    def _preserve_variables(self, var_name_list, preserve_fn):
        for v in var_name_list:
            setattr(self, v,
                    preserve_fn(self._get_store_name(v), getattr(self, v)))

    def __init__(self, env, config):
        gym.Wrapper.__init__(self, env)
        self._state_preserver = config['state_preserver']
        self._env_resets = 0
        self._env_episode_steps = 0
        self._env_total_steps = 0

        self._preserve_variables(self._get_preserved_variable_names(),
                                 self._state_preserver.create_variable)

        config['env_usage_stats'] = EnvUsageStatsTracker.EnvUsageStatsAPI(self)

    def __getattr__(self, attr):
        return getattr(self.env, attr)

    def reset(self):
        self._env_resets += 1
        self._env_episode_steps = 0

        self._preserve_variables(self._get_preserved_variable_names(),
                                 self._state_preserver.update_variable)

        return self.env.reset()

    def step(self, action):
        self._env_episode_steps += 1
        self._env_total_steps += 1
        return self.env.step(action)
