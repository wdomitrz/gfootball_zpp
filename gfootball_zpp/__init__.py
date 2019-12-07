from .env_composer import kwargs_compose_environment as create_custom_enviroment
import gym

gym.register(
    id='gfootball-custom-v1',
    entry_point='gfootball_zpp:create_custom_enviroment',
    kwargs={
        'number_of_right_players_agent_controls': 0,
        'enable_goal_videos': False,
        'enable_full_episode_videos': False,
        'render': False,
        'write_video': False,
        'stacked': False,
        'enable_sides_swap': False,
        'extra_players': None
    }
)
