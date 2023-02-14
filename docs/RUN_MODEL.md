# **Training and Evaluating SSDA3D**
## **Training Scripts**
SSDA3D is a two-stage framework, which are Inter-domain Adaptation stage and Intra-domain Generalization stage, between which is generating pseudo labels for unlabeled target data. So there are three steps to train SSDA3D. Here is an example for training with 1% labeled nuScenes data (and 99% unlabeled nuScenes data). For different amount of labeled nuScenes settings, you can specify different config files, such as 5% labeled (and 95% unlabeled), 10% labeled(and 90% unlabeled) and 20% labeled (and 80% unlabeled).

* **Inter-domain Adaptation.**. You can run the following command to conduct STAGE_I training. 
```shell script
bash scripts/stage1_cutmix_dist_train.sh ${NUM_GPUS} --cfg_file cfgs/stage1_cutmix/centerpoint_20_waymo_1_nus_frames_cutmix.yaml --fix_random_seed
```

* **Generating Pseudo Labeles.** After STAGE_I training, you can run the following command to generate pseudo labels for the left unlabeled target data with the model from STAGE_I. Note that, you need to specify a thresholds score to filter the predicted boxes. Specifically, we select 0.2 for 99% unlabeled target data.
```shell script
bash scripts/generate_pseudo_labels_dish.sh ${NUM_GPUS} --cfg_file cfgs/pseudo_labels/centerpoint_generate_99_pseudo_nus_frames.yaml --ckpt ../output/stage1_cutmix/centerpoint_20_waymo_1_nus_frames_cutmix/default/ckpt/checkpoint_epoch_20.pth --pseudo_thresh 0.2
```

* **Intra-domain Generalization.** After generating pseudo labels, you can run the following command to conduct STAGE_II training.
```shell script
bash scripts/stage2_mixup_dist_train.sh ${NUM_GPUS} --cfg_file cfgs/stage2_mixup/centerpoint_1_lab_99_unlab_nus_frames_mixup.yaml --pretrained_model ../output/stage1_cutmix centerpoint_20_waymo_1_nus_frames_cutmix/default/ckpt/checkpoint_epoch_20.pth --pseudo_info_path ../output/pseudo_labels/centerpoint_generate_99_pseudo_nus_frames/default/score_0.2_last_80_percent_frames_nuscenes_infos_10sweeps_train.pkl --fix_random_seed
```

In our experiments, we set different thresholds scores for different setting, which can be found in the following table.

| Labeled Target | 1% | 5% | 10% | 20% |
| -------------- | -- | -- | --- | --- | 
| Thresh score | 0.2 | 0.2 | 0.2 | 0.3 | 
||

## **Evaluation Scripts**

You can run this command to evaluate a model trained by **SSDA3D** on nuScenes dataset.
```shell script
bash scripts/dist_test.sh ${NUM_GPUS} --cfg_file cfgs/nuscenes_models/ssda3d_centerpoint.yaml --ckpt ../../OwnOpenPCDet/output/xxx/xxx/xxx/ckpt/checkpoint_epoch.pth
```