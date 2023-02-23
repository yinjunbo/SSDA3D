import numpy as np
import copy
import torch
from ...ops.iou3d_nms import iou3d_nms_utils
from ...utils import common_utils
from ..augmentor.augmentor_utils import get_points_in_box

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

# collision detection
def intra_domain_point_mixup_cd(data_dict_1, data_dict_2, alpha=None):
    new_data_dict = copy.deepcopy(data_dict_1)

    new_data_dict['points'] = []
    new_data_dict['gt_boxes'] = []

    lam = np.random.beta(alpha, alpha)

    valid_boxes = data_dict_2['gt_boxes']
    try:
        # collision detection
        iou = iou3d_nms_utils.boxes_bev_iou_cpu(data_dict_1['gt_boxes'][:, 0:7], data_dict_2['gt_boxes'][:, 0:7])
        valid_mask = (iou.max(axis=0) == 0).nonzero()[0]
        invalid_mask = (iou.max(axis=0) > 0).nonzero()[0]
        valid_boxes = data_dict_2['gt_boxes'][valid_mask]
        invalid_boxes = data_dict_2['gt_boxes'][invalid_mask]
        assert len(valid_boxes) + len(invalid_boxes) == len(data_dict_2['gt_boxes'])

        cur_mask = None
        for box in invalid_boxes:
            points_in_box, mask = get_points_in_box(data_dict_2['points'], box)
            if cur_mask is not None:
                cur_mask = cur_mask & ~mask
            else:
                cur_mask = ~mask

        if cur_mask is not None:
            data_dict_2['points'] = data_dict_2['points'][cur_mask]
        # end collision detection
    except:
        pass

    data_dict_1 = shuffle_points(data_dict_1)
    data_dict_2 = shuffle_points(data_dict_2)

    new_data_dict['points'] = np.concatenate((data_dict_1['points'][:int(data_dict_1['points'].shape[0] * lam)], 
                                              data_dict_2['points'][:int(data_dict_2['points'].shape[0] * (1 - lam))]), axis=0)
    new_data_dict['gt_boxes'] = np.concatenate((data_dict_1['gt_boxes'], valid_boxes), axis=0)

    return new_data_dict
    