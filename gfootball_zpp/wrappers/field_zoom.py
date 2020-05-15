from gfootball.env.observation_preprocessing import _MARKER_VALUE, \
    MINIMAP_NORM_Y_MAX, MINIMAP_NORM_Y_MIN, MINIMAP_NORM_X_MAX,MINIMAP_NORM_X_MIN
import numpy as np


def zoom(observation, zoom_center='active', zoom_ratio=4, dimensions=(48,48), layers=None):
    if layers is None:
        layers = ['left_team', 'right_team', 'ball', 'active']
    if zoom_center == 'active':
        zoom_center = [o['left_team'][o['active']] for o in observation]
    elif zoom_center == 'ball':
        zoom_center = [(observation[0]['ball'][0], observation[0]['ball'][1])] * len(observation)
    assert len(zoom_center) == len(observation), 'Expected zoom center for each observation'
    return generate_smm(observation, zoom_center, zoom_ratio, layers, dimensions)


def mark_points(frame, center_point, points, minimap_norm_X, minimap_norm_Y):
    """Draw dots corresponding to 'points'.

    Args:
    frame: 2-d matrix representing one SMM channel ([y, x])
    points: a list of (x, y) coordinates to be marked
    """
    for p in range(len(points) // 2):
        px = points[p * 2] - center_point[0]
        py = points[p * 2 + 1] - center_point[1]
        if px < minimap_norm_X[0] or px > minimap_norm_X[1] or\
           py < minimap_norm_Y[0] or py > minimap_norm_Y[1]:
            continue

        x = int((px - minimap_norm_X[0]) /
                (minimap_norm_X[1] - minimap_norm_X[0]) * frame.shape[1])
        y = int((py - minimap_norm_Y[0]) /
                (minimap_norm_Y[1] - minimap_norm_Y[0]) * frame.shape[0])
        x = max(0, min(frame.shape[1] - 1, x))
        y = max(0, min(frame.shape[0] - 1, y))
        frame[y, x] = _MARKER_VALUE


def generate_smm(observation, zoom_center, zoom_ratio, layers,
                 channel_dimensions):
    """Returns a list of minimap observations given the raw features for each
    active player.

    Args:
    observation: raw features from the environment
    channel_dimensions: resolution of SMM to generate
    config: environment config

    Returns:
    (N, H, W, C) - shaped np array representing SMM. N stands for the number of
    players we are controlling.
    """
    frame = np.zeros((len(observation), channel_dimensions[1],
                      channel_dimensions[0], len(layers)), dtype=np.uint8)
    zoom = 1 / zoom_ratio
    x_norm = (MINIMAP_NORM_X_MIN * zoom, MINIMAP_NORM_X_MAX * zoom)
    y_norm = (MINIMAP_NORM_Y_MIN * zoom, MINIMAP_NORM_Y_MAX * zoom)
    for o_i, o in enumerate(observation):
        for index, layer in enumerate(layers):
            if layer not in o:
                continue
            if layer == 'active':
                if o[layer] == -1:
                    continue
                mark_points(frame[o_i, :, :, index], zoom_center[o_i],
                            np.array(o['left_team'][o[layer]]).reshape(-1),
                            x_norm, y_norm)
            else:
                mark_points(frame[o_i, :, :, index], zoom_center[o_i],
                            np.array(o[layer]).reshape(-1), x_norm, y_norm)
    return frame
