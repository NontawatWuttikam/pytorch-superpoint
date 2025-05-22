#!/bin/bash
EXPER=PRETRAINED_train_v16.1_sunlit2024_lr0.0005_descLossOnly_gradac8
# EXPER=test1
CONFIG=configs/superpoint_coco_train_heatmap_proxyopt.yaml

GPU_DEVICE="1"

# mkdir logs/$EXPER
# cp $CONFIG logs/$EXPER
# mv logs/$EXPER/$(basename "${CONFIG}") logs/$EXPER/train_config.yaml

source ~/miniconda3/etc/profile.d/conda.sh
conda activate py36-sp

CUDA_VISIBLE_DEVICES=$GPU_DEVICE python train4.py train_joint $CONFIG $EXPER --eval --debug