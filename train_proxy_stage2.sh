#!/bin/bash
# EXPER=PRETRAINED_train_v16.1_RawEdgeContrastIndexTop100_lr0.0005_descLossOnly_gradac8
# EXPER=DESCLAMBDA5.0_PRETRAINED_SAMEMODELHOMOADAPT_AGGRESSIVEHOMOADAPT_CFANORMALIZE_XHOMOWARP_DETERMHOMOADAPT_train_v16.2-chroma-HumanTunedInitialHype_sunlithpatchesv4.1_lr0.01_bothLoss_initialHypeHomoAdaptOnly_gradac32
EXPER=DESCLAMBDA5.0_PRETRAINED_SAMEMODELHOMOADAPT_AGGRESSIVEHOMOADAPT_CFANORMALIZE_XHOMOWARP_DETERMHOMOADAPT_train_v16.2-chroma-HumanTunedInitialHype_sunlit_lr0.005_bothLoss_initialHypeHomoAdaptOnly_gradac187
# EXPER=PRETRAINED_train_v16.1_allset_lr0.0005_descLossOnly_gradac8
# EXPER=test_run1
CONFIG=configs/superpoint_coco_train_heatmap_proxyopt.yaml

GPU_DEVICE="0"

# mkdir logs/$EXPER
# cp $CONFIG logs/$EXPER
# mv logs/$EXPER/$(basename "${CONFIG}") logs/$EXPER/train_config.yaml

source ~/miniconda3/etc/profile.d/conda.sh
conda activate py36-sp

CUDA_VISIBLE_DEVICES=$GPU_DEVICE python train4.py train_joint $CONFIG $EXPER --eval --debug