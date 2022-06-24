import numpy as np


def trunc(a, decimals=0):
    return np.trunc(a * 10 ** decimals) / (10 ** decimals)


def ceil(a, decimals=0):
    return np.round(np.ceil(trunc(a * 10 ** decimals, 1)) / 10 ** decimals, decimals)


def floor(a, decimals=0):
    return np.round(np.floor(trunc(a * 10 ** decimals, 1)) / 10 ** decimals, decimals)


def symceil(array, decimals):
    assert decimals >= 0 and decimals % 1 == 0
    array = np.array(array, dtype=float)
    positive = array > 0
    array[positive] = ceil(array[positive], decimals)
    array[~positive] = floor(array[~positive], decimals)
    return array