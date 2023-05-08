import os.path
import numpy as np
import pickle
import copy
from pathlib import Path

from pcdet.datasets import CutMixDatasetTemplate
from pcdet.utils import box_utils

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import common_utils

class WaymoNusCutMixDataset(CutMixDatasetTemplate):
    def __init__(self, dataset_cfg=None, training=True, dataset_names=None, logger=None):
        super().__init__(dataset_cfg, training, dataset_names, logger)

        self.nus_infos = []
        self.waymo_infos = []

        # for NuScenes
        self.include_nuscenes_data(self.mode)

        # for Waymo
        self.waymo_data_path = self.root_path_source / self.dataset_cfg['WaymoDataset'].PROCESSED_DATA_TAG
        self.waymo_split = self.dataset_cfg['WaymoDataset'].DATA_SPLIT[self.mode]
        split_dir = self.root_path_source / 'ImageSets' / (self.waymo_split + '.txt')
        self.waymo_sample_sequence_list = [x.strip() for x in open(split_dir).readlines()]

        self.include_waymo_data(self.mode)
        self.logger.info('Total samples for Waymo: %d' % (len(self.waymo_infos)))
        self.logger.info('Total samples for NuScenes: %d' % (len(self.nus_infos)))

    # for nus
    def include_nuscenes_data(self, mode):
        self.logger.info('Loading NuScenes Dataset')
        nuscenes_infos = []

        for info_path in self.dataset_cfg['NuScenesDataset'].INFO_PATH[mode]:
            info_path = self.root_path_target / info_path
            if not info_path.exists():
                self.logger.info(f'NuScenesDataset info path: {info_path} doesnt exist!')
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                nuscenes_infos.extend(infos)

        self.nus_infos.extend(nuscenes_infos)

    def get_lidar_with_sweeps(self, index, max_sweeps=1):
        info = self.nus_infos[index]
        lidar_path = self.root_path_target / info['lidar_path']
        points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]

        sweep_points_list = [points]
        sweep_times_list = [np.zeros((points.shape[0], 1))]

        for k in np.random.choice(len(info['sweeps']), max_sweeps-1, replace=False):
            points_sweep, times_sweep = self.get_sweep(info['sweeps'][k])
            sweep_points_list.append(points_sweep)
            sweep_times_list.append(times_sweep)

        points = np.concatenate(sweep_points_list, axis=0)
        times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

        points = np.concatenate((points, times), axis=1)
        return points

    def get_sweep(self, sweep_info):
        def remove_ego_points(points, center_radius=1.0):
            mask = ~((np.abs(points[:, 0]) < center_radius) & (np.abs(points[:, 1]) < center_radius))
            return points[mask]

        lidar_path = self.root_path_target / sweep_info['lidar_path']
        points_sweep = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]
        points_sweep = remove_ego_points(points_sweep).T
        if sweep_info['transform_matrix'] is not None:
            num_points = points_sweep.shape[1]
            points_sweep[:3, :] = sweep_info['transform_matrix'].dot(
                np.vstack((points_sweep[:3, :], np.ones(num_points)))
            )[:3, :]

        cur_times = sweep_info['time_lag'] * np.ones((1, points_sweep.shape[1]))
        return points_sweep.T, cur_times.T

    # for waymo
    def include_waymo_data(self, mode):
        self.logger.info('Loading Waymo Dataset')
        waymo_infos = []

        num_skipped_infos = 0
        for k in range(len(self.waymo_sample_sequence_list)):
            sequence_name = os.path.splitext(self.waymo_sample_sequence_list[k])[0]
            info_path = self.waymo_data_path / sequence_name / ('%s.pkl' % sequence_name)
            info_path = self.check_sequence_name_with_all_version(info_path)
            if not info_path.exists():
                num_skipped_infos += 1
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                waymo_infos.extend(infos)

        # self.logger.inf
        if self.dataset_cfg['WaymoDataset'].SAMPLED_INTERVAL[mode] > 1:
            sampled_waymo_infos = []
            for k in range(0, len(waymo_infos), self.dataset_cfg['WaymoDataset'].SAMPLED_INTERVAL[mode]):
                sampled_waymo_infos.append(waymo_infos[k])
            self.waymo_infos.extend(sampled_waymo_infos)
        else:
            self.waymo_infos.extend(waymo_infos)

    def get_lidar(self, sequence_name, sample_idx):
        lidar_file = self.waymo_data_path / sequence_name / ('%04d.npy' % sample_idx)
        point_features = np.load(lidar_file)

        points_all, NLZ_flag = point_features[:, 0:5], point_features[:, 5]
        if not self.dataset_cfg['WaymoDataset'].get('DISABLE_NLZ_FLAG_ON_POINTS', False):
            points_all = points_all[NLZ_flag == -1]
        points_all[:, 3] = np.tanh(points_all[:, 3])
        return points_all

    @staticmethod
    def check_sequence_name_with_all_version(sequence_file):
        if not sequence_file.exists():
            found_sequence_file = sequence_file
            for pre_text in ['training', 'validation', 'testing']:
                if not sequence_file.exists():
                    temp_sequence_file = Path(str(sequence_file).replace('segment', pre_text + '_segment'))
                    if temp_sequence_file.exists():
                        found_sequence_file = temp_sequence_file
                        break
            if not found_sequence_file.exists():
                found_sequence_file = Path(str(sequence_file).replace('_with_camera_labels', ''))
            if found_sequence_file.exists():
                sequence_file = found_sequence_file

        return sequence_file

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return (len(self.waymo_infos) + len(self.nus_infos)) * self.total_epochs
        return len(self.waymo_infos) + len(self.nus_infos)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index * (len(self.waymo_infos) + len(self.nus_infos))

        prob = np.random.random(1)
        if prob < self.dataset_cfg.CUTMIX_PROB:
            waymo_info = copy.deepcopy(self.waymo_infos[index % len(self.waymo_infos)])
            nus_info = copy.deepcopy(self.nus_infos[index % len(self.nus_infos)])

            # for nus
            nus_points = self.get_lidar_with_sweeps(index % len(self.nus_infos), max_sweeps=self.dataset_cfg['NuScenesDataset'].MAX_SWEEPS)
            if self.dataset_cfg['NuScenesDataset'].get('SHIFT_COOR', None):
                nus_points[:, 0:3] += np.array(self.dataset_cfg['NuScenesDataset'].SHIFT_COOR, dtype=np.float32)
            nus_input_dict = {
                'points': nus_points,
                'frame_id': Path(nus_info['lidar_path']).stem,
                'metadata': {'token': nus_info['token']}
            }

            if 'gt_boxes' in nus_info:
                if self.dataset_cfg['NuScenesDataset'].get('FILTER_MIN_POINTS_IN_GT', False):
                    mask = (nus_info['num_lidar_pts'] > self.dataset_cfg['NuScenesDataset'].FILTER_MIN_POINTS_IN_GT - 1)
                else:
                    mask = None

                nus_input_dict.update({
                    'gt_names': nus_info['gt_names'] if mask is None else nus_info['gt_names'][mask],
                    'gt_boxes': nus_info['gt_boxes'] if mask is None else nus_info['gt_boxes'][mask]
                })

                if self.dataset_cfg['NuScenesDataset'].get('SHIFT_COOR', None):
                    nus_input_dict['gt_boxes'][:, 0:3] += self.dataset_cfg['NuScenesDataset'].SHIFT_COOR

                if self.dataset_cfg['NuScenesDataset'].get('SET_NAN_VELOCITY_TO_ZEROS', False):
                    gt_boxes = nus_input_dict['gt_boxes']
                    gt_boxes[np.isnan(gt_boxes)] = 0
                    nus_input_dict['gt_boxes'] = gt_boxes

                if not self.dataset_cfg['NuScenesDataset'].PRED_VELOCITY and 'gt_boxes' in nus_input_dict:
                    nus_input_dict['gt_boxes'] = nus_input_dict['gt_boxes'][:, [0, 1, 2, 3, 4, 5, 6]]


            # for waymo
            pc_info = waymo_info['point_cloud']
            sequence_name = pc_info['lidar_sequence']
            sample_idx = pc_info['sample_idx']

            waymo_points = self.get_lidar(sequence_name, sample_idx)

            waymo_input_dict = {
                'points': waymo_points,
                'frame_id': waymo_info['frame_id'],
            }

            if 'annos' in waymo_info:
                annos = waymo_info['annos']
                annos = common_utils.drop_info_with_name(annos, name='unknown')

                if self.dataset_cfg['WaymoDataset'].get('INFO_WITH_FAKELIDAR', False):
                    gt_boxes_lidar = box_utils.boxes3d_kitti_fakelidar_to_lidar(annos['gt_boxes_lidar'])
                else:
                    gt_boxes_lidar = annos['gt_boxes_lidar']

                if self.training and self.dataset_cfg['WaymoDataset'].get('FILTER_EMPTY_BOXES_FOR_TRAIN', False):
                    mask = (annos['num_points_in_gt'] > 0)
                    annos['name'] = annos['name'][mask]
                    gt_boxes_lidar = gt_boxes_lidar[mask]
                    annos['num_points_in_gt'] = annos['num_points_in_gt'][mask]

                waymo_input_dict.update({
                    'gt_names': annos['name'],
                    'gt_boxes': gt_boxes_lidar,
                    'num_points_in_gt': annos.get('num_points_in_gt', None)
                })

                waymo_input_dict['metadata'] = waymo_info.get('metadata', waymo_info['frame_id'])
                waymo_input_dict.pop('num_points_in_gt', None)

            data_dict = self.prepare_data(waymo_input_dict, nus_input_dict)

            # if len(data_dict_list) != 2:
            #     new_index = np.random.randint(self.__len__())
            #     return self.__getitem__(new_index)

            # data_dict = data_dict_list[1]

        else:
            if index < len(self.waymo_infos):
                waymo_info = copy.deepcopy(self.waymo_infos[index])

                pc_info = waymo_info['point_cloud']
                sequence_name = pc_info['lidar_sequence']
                sample_idx = pc_info['sample_idx']
                waymo_points = self.get_lidar(sequence_name, sample_idx)

                waymo_input_dict = {
                    'points': waymo_points,
                    'frame_id': waymo_info['frame_id'],
                }

                if 'annos' in waymo_info:
                    annos = waymo_info['annos']
                    annos = common_utils.drop_info_with_name(annos, name='unknown')

                    if self.dataset_cfg['WaymoDataset'].get('INFO_WITH_FAKELIDAR', False):
                        gt_boxes_lidar = box_utils.boxes3d_kitti_fakelidar_to_lidar(annos['gt_boxes_lidar'])
                    else:
                        gt_boxes_lidar = annos['gt_boxes_lidar']

                    if self.training and self.dataset_cfg['WaymoDataset'].get('FILTER_EMPTY_BOXES_FOR_TRAIN', False):
                        mask = (annos['num_points_in_gt'] > 0)
                        annos['name'] = annos['name'][mask]
                        gt_boxes_lidar = gt_boxes_lidar[mask]
                        annos['num_points_in_gt'] = annos['num_points_in_gt'][mask]

                    waymo_input_dict.update({
                        'gt_names': annos['name'],
                        'gt_boxes': gt_boxes_lidar,
                        'num_points_in_gt': annos.get('num_points_in_gt', None)
                    })

                    waymo_input_dict['metadata'] = waymo_info.get('metadata', waymo_info['frame_id'])
                    waymo_input_dict.pop('num_points_in_gt', None)

                data_dict = self.prepare_ori_data(waymo_input_dict, source=True)

            else:
                index = index - len(self.waymo_infos)
                nus_info = copy.deepcopy(self.nus_infos[index])
                nus_points = self.get_lidar_with_sweeps(index, max_sweeps=self.dataset_cfg['NuScenesDataset'].MAX_SWEEPS)
                if self.dataset_cfg['NuScenesDataset'].get('SHIFT_COOR', None):
                    nus_points[:, 0:3] += np.array(self.dataset_cfg['NuScenesDataset'].SHIFT_COOR, dtype=np.float32)

                nus_input_dict = {
                    'points': nus_points,
                    'frame_id': Path(nus_info['lidar_path']).stem,
                    'metadata': {'token': nus_info['token']}
                }

                if 'gt_boxes' in nus_info:
                    if self.dataset_cfg['NuScenesDataset'].get('FILTER_MIN_POINTS_IN_GT', False):
                        mask = (nus_info['num_lidar_pts'] > self.dataset_cfg['NuScenesDataset'].FILTER_MIN_POINTS_IN_GT - 1)
                    else:
                        mask = None

                    nus_input_dict.update({
                        'gt_names': nus_info['gt_names'] if mask is None else nus_info['gt_names'][mask],
                        'gt_boxes': nus_info['gt_boxes'] if mask is None else nus_info['gt_boxes'][mask]
                    })

                    if self.dataset_cfg['NuScenesDataset'].get('SHIFT_COOR', None):
                        nus_input_dict['gt_boxes'][:, 0:3] += self.dataset_cfg['NuScenesDataset'].SHIFT_COOR

                    if self.dataset_cfg['NuScenesDataset'].get('SET_NAN_VELOCITY_TO_ZEROS', False):
                        gt_boxes = nus_input_dict['gt_boxes']
                        gt_boxes[np.isnan(gt_boxes)] = 0
                        nus_input_dict['gt_boxes'] = gt_boxes

                    if not self.dataset_cfg['NuScenesDataset'].PRED_VELOCITY and 'gt_boxes' in nus_input_dict:
                        nus_input_dict['gt_boxes'] = nus_input_dict['gt_boxes'][:, [0, 1, 2, 3, 4, 5, 6]]

                data_dict = self.prepare_ori_data(nus_input_dict, source=False)

        return data_dict

