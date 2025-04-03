#!/bin/bash
# set the directory name of this evaluation.
# dataName="eval_proxy_superpoint_v15_lowlight_lr0.005_noLRsched_gradac16_4500"
dataName="eval_v15_welllit_externalImageHomoadapt"

# proxyopt config path
proxyoptConfig="../ProxyOpt/train_configs/v15.yaml"

# optimized hype path, specify "original" if wanted original hype rather than optimized hype. according to proxyopt config file.
stage2Checkpoint="/home/boat/proxyISP/pytorch-superpoint/logs/train_v15_welllit_expoLR_externalImageHomoadapt/proxyopt_checkpoints/checkpoint_5000.pkl"
# stage2Checkpoint="original"

hpatchesSeqPrefix="wl" # ll, wl, sl

rm -rf datasets/HPatches

source ~/miniconda3/etc/profile.d/conda.sh
conda activate proxyopt
python make_hpatches.py $proxyoptConfig $stage2Checkpoint $hpatchesSeqPrefix
conda deactivate

conda activate py36-sp
python export.py export_descriptor  configs/magicpoint_repeatability_heatmap_proxyopt.yaml $dataName
python evaluation.py logs/$dataName/predictions --repeatibility --outputImg --homography --plotMatching