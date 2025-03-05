#!/bin/bash
# set the directory name of this evaluation.
dataName="v13_chkpt20300_allim"

# proxyopt config path
proxyoptConfig="../ProxyOpt/train_configs/v13.yaml"

# optimized hype path, specify "original" if wanted original hype rather than optimized hype. according to proxyopt config file.
stage2Checkpoint="logs/superpoint_coco_proxyoptv13_allim/proxyopt_checkpoints/hype_checkpoint_20300.npy"

rm -rf datasets/HPatches

source ~/miniconda3/etc/profile.d/conda.sh
conda activate proxyopt
python make_hpatches.py $proxyoptConfig $stage2Checkpoint
conda deactivate

conda activate py36-sp
python export.py export_descriptor  configs/magicpoint_repeatability_heatmap.yaml $dataName
python evaluation.py logs/$dataName/predictions --repeatibility --outputImg --homography --plotMatching