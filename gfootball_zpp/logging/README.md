# Log API
- Functionalities
	+ api for creating loggers as environment wrappers
	+ some loggers
		* ball owning time logger
		* reward stats logger
		* action stats logger
		* scenario change detector
- remarks
	+ in our setup log api is enabled by default (see `env_composer.py/compose_environment`)
	+ loggers are divided in three parts
		- `log_low` - low level loggers
			+ should be placed first
			+ they use low level football environment data
		- `log_med` - medium level loggers
			+ should be placed before observation changing wrappers
		- `log_high`- can be placed almost anywhere (but not before `low_level` loggers)


## Sample json config
```
{
    "env_name": "5_vs_5_less_randomized",
    "enable_goal_videos": false,
    "enable_full_episode_videos": false,
    "logdir": "",
    "enable_sides_swap": false,
    "number_of_left_players_agent_controls": 4,
    "number_of_right_players_agent_controls": 0,
    "extra_players": null,
    "render": false,
    "write_video": false,
    "stacked_frames": 4,
    "channel_dimensions": [
        96,
        72
    ],
    "rewards": "scoring,checkpoints",
    "dump_frequency": 2, /* frequency of video dumps */
    "reset_log_freq": 3, /* see utils.py/is_log_time */
    "step_log_freq": 100, /* see utils.py/is_log_time */
    "decide_to_log_fn": "lambda x: x % 20 == 0", /* this function takes actor id and decides whenever actor should provide logs */
    "wrappers": "log_low,periodic_dump,log_med,checkpoint_score,log_high,obs_extract,single_agent,obs_stack"
}
```
