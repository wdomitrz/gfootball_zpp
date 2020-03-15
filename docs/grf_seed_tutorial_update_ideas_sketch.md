
# Warning only ideas for using seed with gfootball tutorial (this is a sketch)
1. Environment is created here:
    https://github.com/google-research/seed_rl/blob/master/football/env.py
2. you can change network here:
    https://github.com/google-research/seed_rl/blob/091e540541eaaedec31d5855059018b50ab11293/football/vtrace_main.py#L39
3. Beware of packetbit observation wrapper (its padding, and format of data that it expects):
    https://github.com/google-research/seed_rl/blob/master/football/observation.py
    
4. Default football network expects single agent observation (single minimap
   of shape [X,Y,L]), and single scalar reward, but can handle multi-discrete
   action spaces,
   so if you want to train on multiagent scenario you can create 
   wrappers like these:

```
class SampleMultiAgentRewardWrapper(gym.RewardWrapper):
  def __init__(self, env):
    super(SampleMultiAgentRewardWrapper, self).__init__(env)

  def reward(self, reward):
    return numpy.max(reward)

# Beware that this wrapper is probably not the best (information
# about teams and ball is replicated)
class SampleMultiAgentObservationWrapper(gym.ObservationWrapper):
  def __init__(self, env):
     super(SampleMultiAgentObservationWrapper, self).__init__(env)
     print(self.observation_space)
     observation_shape = env.observation_space.shape[1:-1] +\
                         (env.observation_space.shape[0] *\
                          env.observation_space.shape[-1], )
     self.observation_space = gym.spaces.Box(
       low=0,
       high=255,
       shape=observation_shape,
       dtype=numpy.uint8)
     print(self.observation_space)

  def observation(self, observation):
    return numpy.concatenate(observation, axis=-1)
```

and apply them  

```
def create_environment(_):
  """Returns a gym Football environment."""
  logging.info('Creating environment: %s', FLAGS.game)
  assert FLAGS.num_action_repeats == 1, 'Only action repeat of 1 is supported.'
  channel_dimensions = {
      'default': (96, 72),
      'medium': (120, 90),
      'large': (144, 108),
  }[FLAGS.smm_size]
  env = gym.make(
      'gfootball:GFootball-%s-SMM-v0' % FLAGS.game,
      stacked=True,
      rewards=FLAGS.reward_experiment,
      channel_dimensions=channel_dimensions,
      number_of_left_players_agent_controls=4)
  # Beware that football network expects one scalar reward
  # and observation of shape [X,Y,L]
  env = SampleMultiAgentRewardWrapper(env)
  env = SampleMultiAgentObservationWrapper(env)
  
  # Beware that PackedBitsObservation expects that observation
  # consist of 255 and 0
  return observation.PackedBitsObservation(env)
```
