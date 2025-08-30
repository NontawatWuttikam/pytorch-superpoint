#!/bin/bash

# proxyopt config path
proxyoptConfig="../ProxyOpt/train_configs/v16.1.yaml"

# optimized hype path, specify "original" if wanted original hype rather than optimized hype according to proxyopt config file.
stage2Checkpoint="/home/boat/proxyISP/pytorch-superpoint/logs/PRETRAINED_train_v16.1_sunlit_lr0.0005_bothLoss_gradac1/proxyopt_checkpoints/checkpoint_5000.pkl"
extraSuffix="_HPatchesV3"
gpu_devices="0"
# stage2Checkpoint="original"

PreHPatchesPath="/mnt/ssd2tb/boat/thesis/s21fe_hpatches_v3"

# hpatches sequence prefix
hpatchesSeqPrefix="sl" # ll, wl, sl
if [ "$hpatchesSeqPrefix" == "" ]; then
    hpatchesSeqPrefix="all"
fi

# Determine dataName based on stage2Checkpoint
if [ "$stage2Checkpoint" == "original" ]; then
    dataName="eval_${hpatchesSeqPrefix}_$(basename "$proxyoptConfig" .yaml)_original"
else
    parent_dir=$(basename "$(dirname "$(dirname "$stage2Checkpoint")")")
    checkpoint_name=$(basename "$stage2Checkpoint")
    step_number="${checkpoint_name%.pkl}"
    step_number="${step_number#checkpoint_}"
    dataName="eval_${hpatchesSeqPrefix}_${parent_dir}_${step_number}"
fi

dataName="${dataName}${extraSuffix}"
echo "Using dataName: $dataName"
# Clean up old dataset
rm -rf datasets/HPatches

# Activate environment and run hpatches generation
source ~/miniconda3/etc/profile.d/conda.sh
conda activate proxyopt
python make_hpatches.py "$proxyoptConfig" "$stage2Checkpoint" "$hpatchesSeqPrefix" "$PreHPatchesPath"
conda deactivate

# Run export and evaluation
conda activate py36-sp
CUDA_VISIBLE_DEVICES=$gpu_devices python export.py export_descriptor configs/magicpoint_repeatability_heatmap_proxyopt.yaml "$dataName"
CUDA_VISIBLE_DEVICES=$gpu_devices python evaluation.py "logs/$dataName/predictions" --repeatibility --outputImg --homography --plotMatching
