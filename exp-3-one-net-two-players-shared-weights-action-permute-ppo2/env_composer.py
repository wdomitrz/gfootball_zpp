from gfootball.env import config
from gfootball.env import football_env
from gfootball.env import observation_preprocessing
from gfootball.env import wrappers

from gfootball.examples.action_wrappers import ActionOrder
from gfootball.examples.player_stack_wrapper import PlayerStackWrapper

def dump_wrapper(env, env_config):
  if env_config['dump_frequency'] > 1:
    return wrappers.PeriodicDumpWriter(env, env_config['dump_frequency'])
  else:
    return env

def checkpoint_wrapper(env, env_config):
  assert 'scoring' in env_config['rewards'].split(',')
  if 'checkpoints' in env_config['rewards'].split(','):
    return wrappers.CheckpointRewardWrapper(env)
  else:
    return env
    
def single_agent_wrapper(env, env_config):
  if (env_config['number_of_left_players_agent_controls'] +
      env_config['number_of_right_players_agent_controls'] == 1):
     env = wrappers.SingleAgentObservationWrapper(env)
     env = wrappers.SingleAgentRewardWrapper(env)
     return env
  else:
    return env

KNOWN_WRAPPERS = {
  'periodic_dump': dump_wrapper,
  'checkpoint_score': checkpoint_wrapper,
  'single_agent': single_agent_wrapper,
  'obs_extract': lambda env, env_config: wrappers.SMMWrapper(env, env_config['channel_dimensions']),
  'obs_stack': lambda env, env_config: wrappers.FrameStack(env, env_config['stacked_frames']),
  'action_order': ActionOrder
}
  
DEFAULT_BASIC_CONFIG = {
  'env_name': 'academy_3_vs_1_with_keeper',                
  'enable_goal_videos': False,
  'enable_full_episode_videos': False,
  'logdir': '',
  'enable_sides_swap': False,
  'number_of_left_players_agent_controls': 2,
  'number_of_right_players_agent_controls': 0,
  'extra_players': None,
  'render': False,
  'write_video': False
}

DEFAULT_EXTENDED_CONFIG = {
  'stacked_frames': 4,
  'channel_dimensions': (observation_preprocessing.SMM_WIDTH,
                         observation_preprocessing.SMM_HEIGHT),
  'rewards': 'scoring',
  'dump_frequency': 1,
  'wrappers': 'periodic_dump,checkpoint_score,single_agent,obs_extract,obs_stack'
}

DEFAULT_EXTENDED_CONFIG = {**DEFAULT_BASIC_CONFIG, **DEFAULT_EXTENDED_CONFIG}
  
def compose_environment(env_config, wrappers):
  def extract_from_dict(dictionary, keys):
    return {new_k: dictionary[k] for (new_k, k) in keys}

  players = [('agent:left_players=%d,right_players=%d' % (
    env_config['number_of_left_players_agent_controls'],
    env_config['number_of_right_players_agent_controls']))]
  if env_config['extra_players'] is not None:
    players.extend(env_config['extra_players'])
  env_config['players'] = players
  c = config.Config(extract_from_dict(env_config,
                             [('enable_sides_swap', 'enable_sides_swap'),
                              ('dump_full_episodes', 'enable_full_episode_videos'),
                              ('dump_scores', 'enable_goal_videos'),
                              ('level', 'env_name'),
                              ('players', 'players'),
                              ('render', 'render'),
                              ('tracesdir', 'logdir'),
                              ('write_video', 'write_video')]))
  env = football_env.FootballEnv(c)

  for w in wrappers:
    env = w(env, env_config)

  return env

def cmd_compose_environment(flags):
  """todo"""
  env_config = {**DEFAULT_EXTENDED_CONFIG, **flags}
  wrappers = []
  for w in env_config['wrappers'].split(','):
    wrappers.append(KNOWN_WRAPPERS[w])

  return compose_environment(env_config, wrappers)
  
def sample_composed_environment():
  return compose_environment(DEFAULT_EXTENDED_CONFIG, [
    dump_wrapper,
    checkpoint_wrapper,
    ActionOrder,
    KNOWN_WRAPPERS['obs_extract'],
    single_agent_wrapper,
    KNOWN_WRAPPERS['obs_stack'],
    PlayerStackWrapper],
  )
