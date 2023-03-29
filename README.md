# **SSDA3D: Semi-supervised Domain Adaptation for 3D Object Detection from Point Cloud**
Thie repo provides the official implementation of our AAAI-2023 paper [*SSDA3D: Semi-supervised Domain Adaptation for 3D Object Detection from Point Cloud*](https://arxiv.org/pdf/2212.02845.pdf). This work presents a new task SSDA for 3D object detection and makes the first effort for handling this task by proposing SSDA3D, a novel framework that jointly addresses inter-domain adaptation and intradomain generalization.

## **Abstract**

LiDAR-based 3D object detection is an indispensable task in advanced autonomous driving systems. Though impressive detection results have been achieved by superior 3D detectors, they suffer from significant performance degeneration when facing unseen domains, such as different Li-DAR configurations, different cities, and weather conditions. The mainstream approaches tend to solve these challenges by leveraging unsupervised domain adaptation (UDA) techniques. However, these UDA solutions just yield unsatisfactory 3D detection results when there is a severe domain shift, e.g., from Waymo (64-beam) to nuScenes (32-beam). To address this, we present a novel Semi-Supervised Domain Adaptation method for 3D object detection (SSDA3D), where
only a few labeled target data is available, yet can significantly improve the adaptation performance. In particular, our SSDA3D includes an Inter-domain Adaptation stage and an Intra-domain Generalization stage. In the first stage, an Inter-domain Point-CutMix module is presented to efficiently align the point cloud distribution across domains. The Point-CutMix generates mixed samples of an intermediate domain, thus encouraging to learn domain-invariant knowledge. Then, in the second stage, we further enhance the model for better generalization on the unlabeled target set. This is achieved by exploring Intra-domain Point-MixUp in semi-supervised learning, which essentially regularizes the pseudo label distribution. Experiments from Waymo to nuScenes show that, with only 10% labeled target data, our SSDA3D can surpass the fully-supervised oracle model with 100%target label. 


## **Main Results**

Our experiments are conducted on two widely used datasets: Waymo with 64-beam LiDAR and nuScenes with 32-beam LiDAR. We adapt from Waymo to nuScenes, i.e., 100% Waymo annotations together with partial nuScenes annotations are used. In particular, we uniformly downsample the nuScenes training samples into 1%, 5%, 10% and 20% (resulting in 282, 1407, 2813 and 5626 frames), and the rest of the samples remain unlabeled.

| Methods | 1% | 5% | 10% | 100% | 
| ------- | -- | -- | --- | --- |
| Labeled Target | 37.2 / 38.1 | 61.0 / 53.2 | 65.6 / 58.2 | 78.4 / 69.9 | 
| Ours    | 73.4 / 67.1 | 76.2 / 68.8 | 78.8 / 70.9 | 79.8 / 71.8 |
| Oracle  | 78.4 / 69.9 | 78.4 / 69.9 | 78.4 / 69.9 | 78.4 / 69.9 |

## **Use SSDA3D**

### **Installation**
First, please refer to [INSTALL](docs/INSTALL.md) for the installation of `OpenPCDet`.

### **Data Preparation**

* First, please refer to [GETTING_STARTED](docs/GETTING_STARTED.md) to prepare Waymo and nuScenes dataset.
* Then please go to [here](https://drive.google.com/drive/folders/1NBU-PUwJ5seuAy83gLyRzCSBEVU1NQPf?usp=share_link) to download the partial info data (1%, 5%, 10% and 20%) and the rest of them (99%, 95%, 90% and 80%) of nuScenes. 
* After that, put these files to `/data/nuscenes/v1.0-trainval`.
* Finally, generate downsampled nuScenes gt database by running the following command. For example, for generating 1% gt database, you can run:
```python
python -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_sub_nuscenes_gt_database \
    --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset.yaml \
    --version v1.0-trainval --percent 1
```

### **Training and Evaluation**

Our training contains two stages, which are Inter-domain Adaptation stage and Intra-domain Generalization stage. For evaluation, there is no difference compared original evaluation of the fully supervised model. The scripts for training and evaluation can be found in [RUN_MODEL](docs/RUN_MODEL.md).

## **License**

This project is released under MIT license, as seen in [LICENSE](LICENSE).

## **Acknowlegement**
Our project is based on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet). We would like to thank for their contributions.


## **Citation**

If you find our project is helpful for you, please cite:


    @inproceedings{wang2023ssda3d,
      title={SSDA3D: Semi-supervised Domain Adaptation for 3D Object Detection from Point Cloud},
      author={Wang, Yan and Yin, Junbo and Li, Wei and Pascal Frossard and Yang, Ruigang and Shen, Jianbing},
      booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
      year={2023}
    }