import random

def get_player_position_from_uniform(a_x, a_y, b_x, b_y, player_role):
    """
    Args:
        a_x, a_y: Coordinates of the left bottom point of the rectangle
        b_x, b_y: Coordinates of the right up point of the rectangle
        player_role: e.g. e_PlayerRole_CB

    Returns: (x, y, player_role) - randomly chosen starting coordinates of the player and his role
    """
    x = random.uniform(a_x, b_x)
    y = random.uniform(a_y, b_y)
    return (x, y, player_role)

def get_player_position_from_gaussian(x, y, sigma_x, sigma_y, player_role):
    """
    Args:
        x, y: Coordinates of the center of the gaussian distribution
        sigma_x, sigma_y: sigma in x and y direction
        player_role: e.g. e_PlayerRole_CB

    Returns: (x, y, player_role) - randomly chosen starting coordinates of the player and his role
    """
    x = random.gauss(x, sigma_x)
    y = random.gauss(y, sigma_y)
    return (x, y, player_role)
