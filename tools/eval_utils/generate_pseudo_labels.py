from pathlib import Path
import pickle
import time

import numpy as np
import torch
import tqdm

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils

def generate_pseudo_label_samples(unlabel_infos_path, predict_dict, output_infos_path, score_thresh={'car': 0}):
    def check_gt_boxes(info):
        return 'gt_boxes' in info
    
    def check_gt_names(info):
        return 'gt_names' in info
    
    with open(unlabel_infos_path, 'rb') as f:
        unlabel_infos = pickle.load(f)
        
    print(f"total unlabel samples: {len(unlabel_infos)}")
    
    pseudo_results_map = {}
    num_gt_box = 0
    for idx, val_result in enumerate(predict_dict):
        pseudo_results_map[val_result["frame_id"]] = val_result
    
    for idx, raw_info in tqdm.tqdm(enumerate(unlabel_infos)):
        if check_gt_boxes(raw_info):
            unlabel_infos[idx].pop("gt_boxes")
        if check_gt_names(raw_info):
            unlabel_infos[idx].pop("gt_names")
        
        unlabel_infos[idx]["gt_boxes"] = np.array(0)
        unlabel_infos[idx]["gt_names"] = np.array(0)

        if 'lidar_path' in raw_info:
            pseudo_result = pseudo_results_map[Path(raw_info['lidar_path']).stem]
        elif 'point_cloud' in raw_info:
            pseudo_result = pseudo_results_map[raw_info['point_cloud']['lidar_idx']]

        if score_thresh is not None:
            sample_info = {'name': [], 'boxes_3d': []}
            for _, class_name in enumerate(score_thresh.keys()):
                this_class_mask = pseudo_result['name'] == class_name
                mask = pseudo_result['score'][this_class_mask] > score_thresh[class_name]
                name = pseudo_result['name'][this_class_mask][mask]
                score = pseudo_result['score'][this_class_mask][mask]
                boxes_3d = pseudo_result['boxes_lidar'][this_class_mask][mask]
                sample_info['name'].append(name)
                sample_info['boxes_3d'].append(boxes_3d)

            name = np.concatenate(sample_info['name'])
            boxes_3d = np.concatenate(sample_info['boxes_3d'])
        
        else:
            name = pseudo_result['name']
            boxes_3d = pseudo_result['boxes_lidar']

        num_gt_box += len(name)
        unlabel_infos[idx]['gt_names'] = name
        unlabel_infos[idx]['gt_boxes'] = boxes_3d

    print("Total box num: %d" % num_gt_box)
    print("Total infos num: %d" % len(unlabel_infos))
    
    with open(output_infos_path, 'wb') as f:
        pickle.dump(unlabel_infos, f)
        
    print("NuScenes pseudo infos file is saved to %s" % (output_infos_path))


def inference_and_generate_pseudo_labes(cfg, args, model, dataloader, logger, dist_test=False, save_to_file=False, result_dir=None, unlabel_infos_path=None):
    # result_dir.mkdir()
    
    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    logger.info('*************** INFERENCING UNLABELD INFOS *****************')
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='test', dynamic_ncols=True)
    start_time = time.time()
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)
        disp_dict = {}
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names, 
            output_path=result_dir if save_to_file else None
        )
        
        det_annos += annos
        # print(len(det_annos))

        if cfg.LOCAL_RANK == 0:
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        # print(len(det_annos))
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        # print(len(det_annos))

    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    
    logger.info('Average predicted number of objects(%d sample): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))
    logger.info('*************** Start to generate pseudo labels *****************')

    car_score = args.pseudo_thresh
    thresh_score = {'car': car_score}

    pseudo_file_name = 'score_' + str(thresh_score['car']) + '_' + str(unlabel_infos_path).split('/')[-1]
    pseudo_label_output_path = result_dir / Path(pseudo_file_name)

    generate_pseudo_label_samples(unlabel_infos_path=unlabel_infos_path, predict_dict=det_annos, output_infos_path=pseudo_label_output_path, score_thresh=thresh_score)

    # cfg.DATA_CONFIG.INFO_PATH['pseudo'].append(pseudo_file_name)
    # print(cfg.DATA_CONFIG.INFO_PATH['pseudo'].append(pseudo_file_name))
    # pseudo_label_output_path_str = str(result_dir) + '/' + pseudo_file_name
    # print(pseudo_label_output_path_str)
    # return pseudo_label_output_path_str

