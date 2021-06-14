import numpy as np
import torch


def slice_arrays(arrays, start=None, stop=None):
    if arrays is None:
        return [None]

    if isinstance(arrays, np.ndarray):
        arrays = [arrays]

    if isinstance(start, list) and stop is not None:
        raise ValueError('The stop argument has to be None if the value of start '
                         'is a list.')
    elif isinstance(arrays, list):
        if hasattr(start, '__len__'):
            # hdf5 datasets only support list objects as indices
            if hasattr(start, 'shape'):
                start = start.tolist()
            return [None if x is None else x[start] for x in arrays]
        else:
            if len(arrays) == 1:
                return arrays[0][start:stop]
            return [None if x is None else x[start:stop] for x in arrays]
    else:
        if hasattr(start, '__len__'):
            if hasattr(start, 'shape'):
                start = start.tolist()
            return arrays[start]
        elif hasattr(start, '__getitem__'):
            return arrays[start:stop]
        else:
            return [None]


def div(x, y):
    try:
        return torch.div(x, y)
    except AttributeError:
        return torch.divide(x, y)


def clip_by_value(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    t = t.float()
    t_min = t_min
    t_max = t_max
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result


def Cosine_Similarity(query, candidate, gamma=1, dim=-1):
    query_norm = torch.norm(query, dim=dim)
    candidate_norm = torch.norm(candidate, dim=dim)
    cosine_score = torch.sum(torch.multiply(query, candidate), dim=-1)
    cosine_score = div(cosine_score, query_norm*candidate_norm+1e-8)
    cosine_score = clip_by_value(cosine_score, -1, 1.0)*gamma
    return cosine_score