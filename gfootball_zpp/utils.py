import gym
import numpy as np

def scalar_to_list(scalar):
    if isinstance(scalar, np.ndarray):
        return scalar.tolist()
    elif not isinstance(scalar, list):
        return [scalar]
    else:
        return scalar

def extract_from_dict(dictionary, keys):
    result = []
    for k in keys:
        result.append((k, dictionary[k]))
    return result

def extract_obj_attributes(obj, attributes):
    result = []
    for attr in attributes:
        # print('for ', attr, ' dir ', dir(getattr(obj, attr)))
        result.append((attr, getattr(obj, attr)))
    return result

def pretty_list_of_pairs_to_string(list_of_pairs):
    result = ''
    for k, v in list_of_pairs:
        result += '* ' + k + ': `' + v + '`  \n'
    return result
