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

### Specifing checkpoints

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
