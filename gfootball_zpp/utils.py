import gym
import numpy as np

def scalar_to_list(scalar):
    if isinstance(scalar, np.ndarray):
        return scalar.tolist()
    elif not isinstance(scalar, list):
        return [scalar]
    else:
        return scalar
