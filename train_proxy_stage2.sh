#!/bin/bash
EXPER=train_v16.1_sunlit_lr0.0005
CONFIG=configs/superpoint_coco_train_heatmap_proxyopt.yaml

# mkdir logs/$EXPER
# cp $CONFIG logs/$EXPER
# mv logs/$EXPER/$(basename "${CONFIG}") logs/$EXPER/train_config.yaml

source ~/miniconda3/etc/profile.d/conda.sh
conda activate py36-sp

python train4.py train_joint $CONFIG $EXPER --eval --debug