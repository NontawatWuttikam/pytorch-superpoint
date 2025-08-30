#!/bin/bash
# EXPER=PRETRAINED_train_v16.1_RawEdgeContrastIndexTop100_lr0.0005_descLossOnly_gradac8
EXPER=PRETRAINED_train_v16.1_sunlit_lr0.0005_bothLoss_gradac1
# EXPER=PRETRAINED_train_v16.1_allset_lr0.0005_descLossOnly_gradac8
# EXPER=stress_test3
CONFIG=configs/superpoint_coco_train_heatmap_proxyopt.yaml

GPU_DEVICE="0"

# mkdir logs/$EXPER
# cp $CONFIG logs/$EXPER
# mv logs/$EXPER/$(basename "${CONFIG}") logs/$EXPER/train_config.yaml

source ~/miniconda3/etc/profile.d/conda.sh
conda activate py36-sp

CUDA_VISIBLE_DEVICES=$GPU_DEVICE python train4.py train_joint $CONFIG $EXPER --eval --debug