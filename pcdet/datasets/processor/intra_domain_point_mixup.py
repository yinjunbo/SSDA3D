import numpy as np
import copy
import torch

def shuffle_points(data_dict):
    points = data_dict['points']
    shuffle_idx = np.random.permutation(points.shape[0])
    points = points[shuffle_idx]
    data_dict['points'] = points
    return data_dict

def intra_domain_point_mixup(data_dict_1, data_dict_2, alpha=None):
    new_data_dict = copy.deepcopy(data_dict_1)
    
    new_data_dict['points'] = []
    new_data_dict['gt_boxes'] = []
    
    lam = np.random.beta(alpha, alpha)
    
    data_dict_1 = shuffle_points(data_dict_1)
    data_dict_2 = shuffle_points(data_dict_2)

    new_data_dict['points'] = np.concatenate((data_dict_1['points'][:int(data_dict_1['points'].shape[0] * lam)], 
                                              data_dict_2['points'][:int(data_dict_2['points'].shape[0] * (1 - lam))]), axis=0)
    new_data_dict['gt_boxes'] = np.concatenate((data_dict_1['gt_boxes'], data_dict_2['gt_boxes']), axis=0)
    
    return new_data_dict