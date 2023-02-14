import numpy as np
import copy
from ...utils.box_utils import mask_boxes_outside_range_numpy
import torch

def check_aspect2D(crop_range, aspect_min):
    xy_aspect = np.min(crop_range[:2]) / np.max(crop_range[:2])
    return (xy_aspect >= aspect_min)

def inter_domain_point_cutmix(data_source, data_target, pc_range):
    """
    Random crop a range in data_source, replace the points in this area with that in
    data_target, and boxes
    Args:
        data_source (_type_): an input sample point cloud, assert Waymo
        data_target (_type_): another input sample point cloud, assert NuScenes
        pc_range (_type_): point cloud range
    """

    cutmixed_data = copy.deepcopy(data_target)
    cutmixed_data['points'] = []
    assert len(cutmixed_data['points']) == 0, 'new generated cutmixed_data should contain 0 point before inter_domain point_cutmix!'
    cutmixed_data['gt_boxes'] = []
    assert len(cutmixed_data['gt_boxes']) == 0, 'new generated cutmixed_data should contain 0 gt_box before inter_domain point_cutmix!'
    
    range_xy = pc_range[3:5] - pc_range[0:2]

    crop_range = 0.5 + (np.random.rand(2) * 0.5)

    loop_count = 0
    while not check_aspect2D(crop_range, 0.75):
        loop_count += 1
        crop_range = 0.5 + (np.random.rand(2) * 0.5)
        if loop_count > 100:
            break

    while True:
        new_range = range_xy * crop_range / 2.0
        sample_center = data_source['points'][np.random.choice(len(data_source['points'])), 0:3]
        # print(sample_center)
        max_xy = sample_center[:2] + new_range
        min_xy = sample_center[:2] - new_range
        
        upper_idx_source = np.sum((data_source['points'][:, :2] < max_xy).astype(np.int32), 1) == 2
        lower_idx_source = np.sum((data_source['points'][:, :2] > min_xy).astype(np.int32), 1) == 2
        
        upper_idx_target = np.sum((data_target['points'][:, :2] < max_xy).astype(np.int32), 1) == 2
        lower_idx_target = np.sum((data_target['points'][:, :2] > min_xy).astype(np.int32), 1) == 2
        
        inside_region_point_idx_source = ((upper_idx_source) & (lower_idx_source))
        outside_region_point_idx_source = ~inside_region_point_idx_source
        
        inside_region_point_idx_target = ((upper_idx_target) & (lower_idx_target))
        outside_region_point_idx_target = ~inside_region_point_idx_target
        
        # avoid nus having too few points
        if (np.sum(inside_region_point_idx_target) > 10000):
            break
        
    # Old version: loop to add the points from source and target domain to the final data dict.
    # for i in range(len(inside_region_point_idx_target)):
    #     if inside_region_point_idx_target[i]:
    #         cutmixed_data['points'].append(data_target['points'][i, :])

    # for i in range(len(outside_region_point_idx_source)):
    #     if outside_region_point_idx_source[i]:
    #         cutmixed_data['points'].append(data_source['points'][i, :])

    # New version: broadcast to add the points from source and target domain to the final data dict.
    cutmixed_data['points'].extend(data_target['points'][inside_region_point_idx_target, :])
    cutmixed_data['points'].extend(data_source['points'][outside_region_point_idx_source, :])

    assert len(cutmixed_data['points']) != 0, 'new generated cutmixed_data should contain more than 0 point after inter_domain point_cutmix!'

    region_range = [min_xy[0], min_xy[1], pc_range[2], max_xy[0], max_xy[1], pc_range[5]]
    inside_region_gt_boxes_mask_source = mask_boxes_outside_range_numpy(data_source['gt_boxes'], region_range, min_num_corners=1)
    inside_region_gt_boxes_mask_target = mask_boxes_outside_range_numpy(data_target['gt_boxes'], region_range, min_num_corners=1)

    inside_region_gt_boxes_source = data_source['gt_boxes'][inside_region_gt_boxes_mask_source]
    inside_region_gt_boxes_target = data_target['gt_boxes'][inside_region_gt_boxes_mask_target]
    outside_region_gt_boxes_source = data_source['gt_boxes'][~inside_region_gt_boxes_mask_source]
    outside_region_gt_boxes_target = data_target['gt_boxes'][~inside_region_gt_boxes_mask_target]

    cutmixed_data['gt_boxes'].extend(outside_region_gt_boxes_source)
    cutmixed_data['gt_boxes'].extend(inside_region_gt_boxes_target)

    cutmixed_data['points'] = np.array(cutmixed_data['points'])
    cutmixed_data['gt_boxes'] = np.array(cutmixed_data['gt_boxes'])

    return cutmixed_data



