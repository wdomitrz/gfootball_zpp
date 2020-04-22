# Football env modifications

## Scenarios

New scenarios can be added to the `gfootball_zpp/scenraios` directory.
Then, during docker creation, they will be automatically copied to
proper directory in local gfootball instance.

### Manual setup

To use scenarios you need to manually add them to
`gfootball/scenraios` directory and **remove `__pycache__` directory**.
I you forget about the second one gfootball might not detect new
scenrios and you'll get an error.

## Players

The gfootball environment support adding new players types which
then can be automatically created by the environment.

In this repo we create zpp player which then creates actual policy
from ones available in this repository. The `gfootball_zpp/players/zpp.py`
file must be copied to `gfootball/env/players` directory (and then __pycache__
have to be removed) and, as in scenarios, this is done during docker
creation.

Unlike scenarios, only the zpp player is actually copied to gfootball.
All it does is importing actual players placed in gfootball_zpp.

###  Using created players

Players are added in following format:

```
<player_name>:arg1=val1,arg2=val2,...
```

In our case player name will always be `zpp` with one required
argument specifying the policy to be used, so it is looks like this:
`zpp:policy=<policy_name>,...`.

It's also important to specify which players should be controlled which is
controlled by two integer args: `left_players` and `right_players`.
If not added they default to 0. 

To add such player to game it's description should be added to 
json file to `extra_players` table.

**Example**
Multi-heads opponent that sample actions controlling four right players that is initialised with checkpoint form `gs://bucket/job/1/ckpt/0/ckpt-123`:
```
"extra_players": ["zpp:policy=multihead,sample=True,right_players=4,checkpoints=GS//bucket/job/1/ckpt/0/ckpt-123"]
```

#### Specifing checkpoints

There are three options:
- `!latest-GS//bucket/path/to/checkpoints/dir` - takes newest chceckpoint from given dir
- `!random-GS//bucket/path/to/checkpoints/dir` - takes random checkpoint form given dir
- `!mostly_latest-GS//bucket/path/to/checkpoints/dir` - takes random checkpoint but newer have higher propbability
- `GS//bucket/path/to/checkpoint/without/extenstion` - takes exactly given checkpoint
- `/path/to/local/checkpoint/without/extenstion` - take exactly given checkpoint

Definitions like these can be provided as list separated `*` sign in `checkpoints` parameters:
```
zpp:policy=<policy_name>,checkpoints=!mostly_latest-GS//zuzanna-seed/mixed_sp_e3/1/ckpt/0/*!random-GS//scon/scon_e3_p2_hard_sp/1/ckpt/0
```

The parameter `checkpoint_reload_rate` specifies nuber of episodes after which th checkpoint will be
reloaded.

Then one of the definition will be choosen and checkpoint will be updated according to it.

By default probability is uniform but it is possible to define specific probabilities:

```
<definition1>;<probability1>*<definition2>;<probability2>,checkpoint_reload_rate=100
```

## Playing with different scenarios or both checkpoints and opponents

To do so we need to use so called recreatable env wrapper. 
In json we add array that specifies modification to envirnoment config that we
want to make in different workers. The config parameters used to do this are:
- `env_change_params` - array of dictionaries with properties to change (eg. to change
scenario use `level`, to change opponents use `extra_player`).
- `env_change_rate` - after how many episodes the env should reload; you can use really high
value to not reload env at all.
- `env_change_probabilities` - the probability with which each params set will be selected;
not specifying it leaves uniform distribution. 

**Example**
playing simultaneously on different scenarios

```json
{
    "env_change_rate": 200000000,
    "env_change_params": [
        {"level": "5_vs_5_3v3_pass_and_shoot"},
        {"level": "5_vs_5_4v0_empty_goal"},
        {"level": "5_vs_5_4v1"},
        {"level": "5_vs_5_4v0_with_keeper"},
        {"level": "5_vs_5_corner"},
        {"level": "5_vs_5"}
    ],
    "env_change_probabilities": [
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.5
    ],
    ...
}
```

**Example** playing on 1/2 workers with random opponent and on 1/2 with bots:
```json
{
    "extra_players": ["zpp:policy=random,right_players=4"],
    "env_change_rate": 200000000,
    "env_change_params": [
        {},
        {"extra_players": []}
    ],
    ...
}
```