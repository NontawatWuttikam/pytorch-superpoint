#!/bin/bash

# proxyopt config path
proxyoptConfig="../ProxyOpt/train_configs/v16.1.yaml"

# optimized hype path, specify "original" if wanted original hype rather than optimized hype according to proxyopt config file.
stage2Checkpoint="/home/boat/proxyISP/pytorch-superpoint/logs/train_v16.1_lowlight_lr0.0005/proxyopt_checkpoints/checkpoint_5000.pkl"
extraSuffix="_eval3"
# stage2Checkpoint="original"

# hpatches sequence prefix
hpatchesSeqPrefix="ll" # ll, wl, sl

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
python make_hpatches.py "$proxyoptConfig" "$stage2Checkpoint" "$hpatchesSeqPrefix"
conda deactivate

# Run export and evaluation
conda activate py36-sp
python export.py export_descriptor configs/magicpoint_repeatability_heatmap_proxyopt.yaml "$dataName"
python evaluation.py "logs/$dataName/predictions" --repeatibility --outputImg --homography --plotMatching
