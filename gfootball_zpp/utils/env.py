

def change_scenario(env, level):
    """Changes scenario used by given env.

    The actual change will take place during reset
    """
    env.unwrapped._config._values['level'] = level
