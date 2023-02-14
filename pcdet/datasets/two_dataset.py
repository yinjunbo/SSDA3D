from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.utils.data as torch_data

from ..utils import common_utils
from .augmentor.data_augmentor import DataAugmentor
from .processor.data_processor import DataProcessor
from .processor.point_feature_encoder import PointFeatureEncoder
from .processor.inter_domain_point_cutmix import inter_domain_point_cutmix
# from .processor.random_patch_replacement import random_patch_replacement
# from .processor.point_mixup import pc_mixup

class CutMixDatasetTemplate(torch_data.Dataset):
    def __init__(self, dataset_cfg=None, training=True, dataset_names=None, logger=None):
        super(CutMixDatasetTemplate, self).__init__()
        self.dataset_cfg = dataset_cfg
        self.training = training
        self.dataset_names = dataset_names
        self.class_names = self.dataset_cfg.CLASS_NAMES
        self.class_names_source = self.dataset_cfg[self.dataset_names['Source']].CLASS_NAMES
        self.class_names_target = self.dataset_cfg[self.dataset_names['Target']].CLASS_NAMES

        self.logger = logger

        if self.dataset_names['Source'] == 'NuScenesDataset':
            self.root_path_source = Path(self.dataset_cfg[self.dataset_names['Source']].DATA_PATH) / self.dataset_cfg[self.dataset_names['Source']].VERSION
        else:
            self.root_path_source = Path(self.dataset_cfg[self.dataset_names['Source']].DATA_PATH)

        if self.dataset_names['Target'] == 'NuScenesDataset':
            self.root_path_target = Path(self.dataset_cfg[self.dataset_names['Target']].DATA_PATH) / self.dataset_cfg[self.dataset_names['Target']].VERSION
        else:
            self.root_path_target = Path(self.dataset_cfg[self.dataset_names['Target']].DATA_PATH)

        if self.dataset_cfg is None or dataset_names is None:
            return

        self.point_cloud_range = np.array(self.dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32)
        self.point_feature_encoder = PointFeatureEncoder(
            self.dataset_cfg.POINT_FEATURE_ENCODING,
            point_cloud_range=self.point_cloud_range
        )

        self.data_augmentor_source = DataAugmentor(
            self.root_path_source, self.dataset_cfg[self.dataset_names['Source']].DATA_AUGMENTOR, self.class_names_source, logger=logger
        ) if self.training else None

        self.data_augmentor_target = DataAugmentor(
            self.root_path_target, self.dataset_cfg[self.dataset_names['Target']].DATA_AUGMENTOR, self.class_names_target, logger=logger
        ) if self.training else None

        self.data_processor = DataProcessor(
            self.dataset_cfg.DATA_PROCESSOR, point_cloud_range=self.point_cloud_range,
            training=self.training,
            num_point_features=self.point_feature_encoder.num_point_features + 1 if self.dataset_cfg.get('USE_DOMAIN_LABEL', False) else self.point_feature_encoder.num_point_features
        )

        self.grid_size = self.data_processor.grid_size
        self.voxel_size = self.data_processor.voxel_size

        self.total_epochs = 0
        self._merge_all_iters_to_one_epoch = False

        if hasattr(self.data_processor, "depth_downsample_factor"):
            self.depth_downsample_factor = self.data_processor.depth_downsample_factor
        else:
            self.depth_downsample_factor = None


    @property
    def mode(self):
        return 'train' if self.training else 'test'

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, state):
        self.__dict__.update(state)

    def merge_all_iters_to_one_epoch(self, merge=True, epochs=None):
        if merge:
            self._merge_all_iters_to_one_epoch = True
            self.total_epochs = epochs
        else:
            self._merge_all_iters_to_one_epoch = False

    def __len__(self):
        raise NotImplementedError

    def prepare_ori_data(self, data_dict, source=True):
        if self.training:
            assert 'gt_boxes' in data_dict, 'gt_boxes should be provided for training'
            if source:
                gt_boxes_mask = np.array([n in self.class_names_source for n in data_dict['gt_names']], dtype=np.bool_)
                data_dict = self.data_augmentor_source.forward(
                    data_dict={
                        **data_dict,
                        'gt_boxes_mask': gt_boxes_mask
                    }
                )
            else:
                gt_boxes_mask = np.array([n in self.class_names_target for n in data_dict['gt_names']], dtype=np.bool_)
                data_dict = self.data_augmentor_target.forward(
                    data_dict={
                        **data_dict,
                        'gt_boxes_mask': gt_boxes_mask
                    }
                )


        if data_dict.get('gt_boxes', None) is not None:
            if source:
                selected = common_utils.keep_arrays_by_name(data_dict['gt_names'], self.class_names_source)
            else:
                selected = common_utils.keep_arrays_by_name(data_dict['gt_names'], self.class_names_target)

            data_dict['gt_boxes'] = data_dict['gt_boxes'][selected]
            data_dict['gt_names'] = data_dict['gt_names'][selected]
            if source:
                gt_classes = np.array([self.class_names_source.index(n) + 1 for n in data_dict['gt_names']], dtype=np.int32)
            else:
                gt_classes = np.array([self.class_names_target.index(n) + 1 for n in data_dict['gt_names']], dtype=np.int32)

            for idx, name in enumerate(data_dict['gt_names']):
                if source:
                    if name == self.class_names_source[0]:
                        data_dict['gt_names'][idx] = self.class_names[0]
                else:
                    if name == self.class_names_target[0]:
                        data_dict['gt_names'][idx] = self.class_names[0]

            gt_boxes = np.concatenate((data_dict['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)

            data_dict['gt_boxes'] = gt_boxes

            if data_dict.get('gt_boxes2d', None) is not None:
                data_dict['gt_boxes2d'] = data_dict['gt_boxes2d'][selected]

        if data_dict.get('points', None) is not None:
            data_dict = self.point_feature_encoder.forward(data_dict)

        data_dict = self.data_processor.forward(
            data_dict=data_dict
        )

        if self.training and len(data_dict['gt_boxes']) == 0:
            new_index = np.random.randint(self.__len__())
            return self.__getitem__(new_index)

        data_dict.pop('gt_names', None)

        return data_dict

    def prepare_data(self, data_dict_source, data_dict_target):
        if self.training:
            assert 'gt_boxes' in data_dict_source, 'gt_boxes should be provided for training in data_dict_source!'
            assert 'gt_boxes' in data_dict_target, 'gt_boxes should be proviced for training in data_dict_target!'

            gt_boxes_mask_source = np.array([n in self.class_names_source for n in data_dict_source['gt_names']], dtype=np.bool_)
            gt_boxes_mask_target = np.array([n in self.class_names_target for n in data_dict_target['gt_names']], dtype=np.bool_)

            data_dict_source = self.data_augmentor_source.forward(
                data_dict={
                    **data_dict_source,
                    'gt_boxes_mask': gt_boxes_mask_source
                }
            )

            data_dict_target = self.data_augmentor_target.forward(
                data_dict={
                    **data_dict_target,
                    'gt_boxes_mask': gt_boxes_mask_target
                }
            )

        if data_dict_source.get('gt_boxes', None) is not None:
            selected = common_utils.keep_arrays_by_name(data_dict_source['gt_names'], self.class_names_source)
            data_dict_source['gt_boxes'] = data_dict_source['gt_boxes'][selected]
            data_dict_source['gt_names'] = data_dict_source['gt_names'][selected]
            gt_classes = np.array([self.class_names_source.index(n) + 1 for n in data_dict_source['gt_names']], dtype=np.int32)

            for idx, name in enumerate(data_dict_source['gt_names']):
                if name == self.class_names_source[0]:
                    data_dict_source['gt_names'][idx] = self.class_names[0]

            gt_boxes = np.concatenate((data_dict_source['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
            data_dict_source['gt_boxes'] = gt_boxes

            if data_dict_source.get('gt_boxes2d', None) is not None:
                data_dict_source['gt_boxes2d'] = data_dict_source['gt_boxes2d'][selected]

        if data_dict_target.get('gt_boxes', None) is not None:
            selected = common_utils.keep_arrays_by_name(data_dict_target['gt_names'], self.class_names_target)
            data_dict_target['gt_boxes'] = data_dict_target['gt_boxes'][selected]
            data_dict_target['gt_names'] = data_dict_target['gt_names'][selected]
            gt_classes = np.array([self.class_names_target.index(n) + 1 for n in data_dict_target['gt_names']], dtype=np.int32)

            for idx, name in enumerate(data_dict_target['gt_names']):
                if name == self.class_names_target[0]:
                    data_dict_target['gt_names'][idx] = self.class_names[0]

            gt_boxes = np.concatenate((data_dict_target['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
            data_dict_target['gt_boxes'] = gt_boxes

            if data_dict_target.get('gt_boxes2d', None) is not None:
                data_dict_target['gt_boxes2d'] = data_dict_target['gt_boxes2d'][selected]


        if data_dict_source.get('points', None) is not None:
            data_dict_source = self.point_feature_encoder.forward(data_dict_source)

        if data_dict_target.get('points', None) is not None:
            data_dict_target = self.point_feature_encoder.forward(data_dict_target)
            
        assert data_dict_source is not None and data_dict_target is not None

        if self.dataset_cfg.MIX_TYPE == 'cutmix':
            # new_data_dict = random_patch_replacement(data_dict_source, data_dict_target, self.point_cloud_range, self.dataset_cfg.CROP_RANGE_PERCENT)
            # new_data_dict = random_patch_replacement(data_dict_source, data_dict_target, self.point_cloud_range)
            cutmixed_data_dict = inter_domain_point_cutmix(data_dict_source, data_dict_target, self.point_cloud_range)
            # print("to here")
            # new_data_dict_1, new_data_dict_2 = new_data_dict[0], new_data_dict[1]
        else:
            raise NotImplementedError
        
        # print(cutmixed_data_dict.keys())
        if len(cutmixed_data_dict['gt_boxes'].shape) != 2:
            new_index = np.random.randint(self.__len__())
            return self.__getitem__(new_index)
        # if len(new_data_dict_1['gt_boxes'].shape) != 2 or len(new_data_dict_2['gt_boxes'].shape) != 2:
        #     new_index = np.random.randint(self.__len__())
        #     return self.__getitem__(new_index)

        cutmixed_data_dict = self.data_processor.forward(cutmixed_data_dict)

        # new_data_dict_1 = self.data_processor.forward(data_dict=new_data_dict_1)
        # new_data_dict_2 = self.data_processor.forward(data_dict=new_data_dict_2)

        if self.training and len(cutmixed_data_dict['gt_boxes']) == 0:
            new_index = np.random.randint(self.__len__())
            return self.__getitem__(new_index)

        # if self.training and (len(new_data_dict_1['gt_boxes']) == 0 or len(new_data_dict_2['gt_boxes']) == 0):
        #     new_index = np.random.randint(self.__len__())
        #     return self.__getitem__(new_index)

        cutmixed_data_dict.pop('gt_names', None)
        # new_data_dict_1.pop('gt_names', None)
        # new_data_dict_2.pop('gt_names', None)

        return cutmixed_data_dict
        # return [new_data_dict_1, new_data_dict_2]

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)

        batch_size = len(batch_list)
        ret = {}

        for key, val in data_dict.items():
            try:
                if key in ['voxels', 'voxel_num_points']:
                    ret[key] = np.concatenate(val, axis=0)
                elif key in ['points', 'voxel_coords']:
                    coors = []
                    for i, coor in enumerate(val):
                        coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                        coors.append(coor_pad)
                    ret[key] = np.concatenate(coors, axis=0)
                elif key in ['gt_boxes']:
                    max_gt = max([len(x) for x in val])
                    batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_gt_boxes3d
                elif key in ['gt_boxes2d']:
                    max_boxes = 0
                    max_boxes = max([len(x) for x in val])
                    batch_boxes2d = np.zeros((batch_size, max_boxes, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        if val[k].size > 0:
                            batch_boxes2d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_boxes2d
                elif key in ['images', 'depth_maps']:
                    # Get largest image size (H, w)
                    max_h = 0
                    max_w = 0
                    for image in val:
                        max_h = max(max_h, image.shape[0])
                        max_w = max(max_w, image.shape[1])

                    # Change size of images
                    images = []
                    for image in val:
                        pad_h = common_utils.get_pad_params(desired_size=max_h, cur_size=image.shape[0])
                        pad_w = common_utils.get_pad_params(desired_size=max_w, cur_size=image.shape[1])
                        pad_width = (pad_h, pad_w)
                        # Pad with nan, to be replaced later in the pipeline.
                        pad_value = np.nan

                        if key == 'images':
                            pad_width = (pad_h, pad_w, (0, 0))
                        elif key == 'depth_maps':
                            pad_width = (pad_h, pad_w)

                        image_pad = np.pad(image, pad_width=pad_width, mode='constant', constant_values=pad_value)

                        images.append(image_pad)

                    ret[key] = np.stack(images, axis=0)

                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size
        return ret
